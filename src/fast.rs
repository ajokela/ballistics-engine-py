// Keystone for migrating the ballistics Flask app off its bespoke `ballistics_rust`
// wrapper onto this binding. Provides:
//   * `ballistic_inputs_from_dict` — full-fidelity dict -> engine BallisticInputs,
//     mirroring ballistics_rust::extract_ballistic_inputs + geometry_mass_to_si.
//   * `fast_integrate` pyfunction — same signature/return contract as
//     ballistics_rust.fast_integrate_rust: returns the scipy-like
//     {t, y(6xN), t_events[3], success} object the app's integrator consumes,
//     over ballistics-engine's fast_trajectory::fast_integrate_with_segments.

use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use ::ballistics_engine::wind::WindSegment;

// MBA-1295: the rich dict -> BallisticInputs parser (+ its unit-conversion helper) now
// lives in `inputs.rs`, shared with `PyBallisticInputs::from_dict` and the new
// solver-class pyfunctions. Re-exported here (rather than updating every call site in this
// file plus montecarlo.rs/effects.rs, which reference these via `crate::fast::...`) so the
// move is a pure relocation with no behavior change for this file's own entry points.
pub(crate) use crate::inputs::{ballistic_inputs_from_dict, geometry_mass_to_si};

const GRAINS_TO_KG: f64 = 0.00006479891;
const INCHES_TO_METERS: f64 = 0.0254;
const KMH_TO_MPS: f64 = 1000.0 / 3600.0;

fn wind_segments_kmh_to_mps(segments: Vec<(f64, f64, f64)>) -> Vec<(f64, f64, f64)> {
    segments
        .into_iter()
        .map(|(speed_kmh, angle_deg, until_distance_m)| {
            (speed_kmh * KMH_TO_MPS, angle_deg, until_distance_m)
        })
        .collect()
}

/// Convert the Python-facing (speed_kmh, angle_deg, until_distance_m) tuples into the
/// engine's `wind::WindSegment` struct (engine 0.24.0 boundary change: `WindSock`,
/// `TrajectorySolver::set_wind_segments`, and `fast_integrate_with_segments` now take
/// `Vec<WindSegment>` instead of tuples). `WindShearWindSock` still takes raw
/// `(speed_mps, angle_deg, until_m)` tuples, so `wind_segments_kmh_to_mps` above is
/// unaffected and keeps returning tuples for that path.
fn to_wind_segments(segments: Vec<(f64, f64, f64)>) -> Vec<WindSegment> {
    segments
        .into_iter()
        .map(|(speed_kmh, angle_deg, until_m)| WindSegment::new(speed_kmh, angle_deg, until_m))
        .collect()
}

/// Fast fixed-step trajectory integration over ballistics-engine's RK45 kernel.
/// Mirrors `ballistics_rust.fast_integrate_rust` exactly: same positional args and
/// the same `{t, y, t_events, success}` return contract the app's integrator reads.
///
/// `wind_segments` is a list of (speed_kmh, angle_deg, until_distance_m) tuples
/// (engine `wind::WindSegment`); empty = no wind. `atmo_params` = 4-vector
/// (base_altitude_m, base_temp_c, base_pressure_hpa, base_density_ratio).
#[pyfunction]
#[pyo3(signature = (inputs, wind_segments, horiz, vert, initial_state, t_span, atmo_params, atmo_segments=Vec::new()))]
pub fn fast_integrate<'py>(
    py: Python<'py>,
    inputs: &Bound<'py, PyDict>,
    wind_segments: Vec<(f64, f64, f64)>,
    horiz: f64,
    vert: f64,
    initial_state: PyReadonlyArray1<'py, f64>,
    t_span: (f64, f64),
    atmo_params: PyReadonlyArray1<'py, f64>,
    // MBA-1137: per-zone downrange atmosphere (temp_c, pressure_hpa, humidity_%, until_distance_m),
    // station-referenced. Empty = single-station (unchanged). Builds an engine AtmoSock so drag
    // density varies by downrange distance, composing with the altitude lapse.
    atmo_segments: Vec<(f64, f64, f64, f64)>,
) -> PyResult<Bound<'py, PyDict>> {
    use ::ballistics_engine::atmosphere::AtmoSock;
    use ::ballistics_engine::fast_trajectory::{fast_integrate_with_segments, FastIntegrationParams};

    let mut bi = ballistic_inputs_from_dict(inputs)?;
    // ballistics-engine >= 0.16.0: fast_integrate reads bullet_mass as kg (SI).
    geometry_mass_to_si(&mut bi);

    let is = initial_state.as_array();
    let ap = atmo_params.as_array();
    if is.len() != 6 {
        return Err(PyValueError::new_err("Initial state must have 6 elements"));
    }
    if ap.len() != 4 {
        return Err(PyValueError::new_err("Atmospheric parameters must have 4 elements"));
    }
    let mut initial_state_arr = [0.0f64; 6];
    for i in 0..6 {
        initial_state_arr[i] = is[i];
    }

    // Both the Python solver layer and ballistics-engine use the McCoy frame, so
    // the initial state passes straight through with no axis swap.
    let atmo_sock = if atmo_segments.is_empty() {
        None
    } else {
        Some(AtmoSock::new(atmo_segments))
    };
    let params = FastIntegrationParams {
        horiz,
        vert,
        initial_state: initial_state_arr,
        t_span,
        atmo_params: (ap[0], ap[1], ap[2], ap[3]),
        atmo_sock,
    };

    let solution = fast_integrate_with_segments(&bi, to_wind_segments(wind_segments), params);

    let dict = PyDict::new(py);
    dict.set_item("t", PyArray1::from_vec(py, solution.t))?;
    let y = PyArray2::from_vec2(py, &solution.y)
        .map_err(|e| PyValueError::new_err(format!("y matrix shape error: {e}")))?;
    dict.set_item("y", y)?;
    let t_events = PyList::empty(py);
    for events in &solution.t_events {
        t_events.append(PyArray1::from_vec(py, events.clone()))?;
    }
    dict.set_item("t_events", t_events)?;
    dict.set_item("success", solution.success)?;
    Ok(dict)
}

/// Single RK-stage derivatives, mirroring `ballistics_rust.derivatives_rust`:
/// returns the 6-vector d(state)/dt = [vx, vy, vz, ax, ay, az] for the McCoy-frame
/// state, over ballistics-engine's `derivatives::compute_derivatives`.
#[pyfunction]
#[pyo3(signature = (_t, state, inputs, wind_segments, atmos_params, bc_used, _target_horizontal_dist_m, _target_vertical_height_m, omega_vector=None))]
#[allow(clippy::too_many_arguments)]
pub fn derivatives<'py>(
    py: Python<'py>,
    _t: f64,
    state: PyReadonlyArray1<'py, f64>,
    inputs: &Bound<'py, PyDict>,
    wind_segments: Vec<(f64, f64, f64)>,
    atmos_params: PyReadonlyArray1<'py, f64>,
    bc_used: f64,
    _target_horizontal_dist_m: f64,
    _target_vertical_height_m: f64,
    omega_vector: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    use ::ballistics_engine::derivatives::compute_derivatives;
    use ::ballistics_engine::wind::WindSock;
    use ::ballistics_engine::wind_shear::{
        WindLayer, WindShearModel, WindShearProfile, WindShearWindSock,
    };
    use nalgebra::Vector3;

    let state_array = state.as_array();
    let atmos_array = atmos_params.as_array();
    if state_array.len() != 6 {
        return Err(PyValueError::new_err("State array must have 6 elements"));
    }
    if atmos_array.len() != 4 {
        return Err(PyValueError::new_err("Atmospheric parameters must have 4 elements"));
    }

    // Python solver layer and ballistics-engine both use the McCoy frame.
    let pos = Vector3::new(state_array[0], state_array[1], state_array[2]);
    let vel = Vector3::new(state_array[3], state_array[4], state_array[5]);
    let atmos_tuple = (atmos_array[0], atmos_array[1], atmos_array[2], atmos_array[3]);

    let mut bi = ballistic_inputs_from_dict(inputs)?;
    geometry_mass_to_si(&mut bi);

    let wind_vector = if bi.enable_wind_shear && bi.wind_shear_model != "none" {
        let mut profile = WindShearProfile {
            model: match bi.wind_shear_model.as_str() {
                "logarithmic" => WindShearModel::Logarithmic,
                "power_law" => WindShearModel::PowerLaw,
                "ekman_spiral" => WindShearModel::EkmanSpiral,
                _ => WindShearModel::None,
            },
            ..Default::default()
        };
        if !wind_segments.is_empty() {
            profile.surface_wind = WindLayer {
                altitude_m: 0.0,
                speed_mps: wind_segments[0].0 * KMH_TO_MPS,
                direction_deg: wind_segments[0].1,
            };
        }
        let sock = WindShearWindSock::with_shooter_altitude(
            wind_segments_kmh_to_mps(wind_segments),
            Some(profile),
            bi.altitude,
        );
        sock.vector_for_position(pos)
    } else {
        // MBA-1338: checked construction — malformed segments (non-finite / negative
        // fields) now surface as a structured ValueError naming the segment index and
        // field, instead of silently feeding a poisoned wind vector into the step.
        let sock = WindSock::try_new(to_wind_segments(wind_segments))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        sock.vector_for_range_stateless(pos[0])
    };

    let omega_vec = match omega_vector {
        Some(o) => {
            let s = o.as_array();
            if s.len() != 3 {
                return Err(PyValueError::new_err("Omega vector must have 3 elements"));
            }
            Some(Vector3::new(s[0], s[1], s[2]))
        }
        None => None,
    };

    // MBA-1137: this per-step derivatives entry backs the deprecated scipy/legacy solver path,
    // not the live fast_integrate path. Per-zone atmosphere is not threaded here (no downrange
    // context per RK stage); pass None so it behaves as single-station, same as today.
    let r = compute_derivatives(
        pos, vel, &bi, wind_vector, atmos_tuple, bc_used, omega_vec, _t, None,
    );
    Ok(PyArray1::from_vec(py, vec![r[0], r[1], r[2], r[3], r[4], r[5]]))
}

/// Solve the barrel elevation (radians) that zeroes at `target_distance_yards`,
/// mirroring `ballistics_rust.calculate_zero_angle_rust`. NOTE the inputs dict here
/// is FULLY imperial (fps/grains/inches/yards) — distinct from fast_integrate's SI
/// dict — and is converted to SI in place. Raises on non-convergence (engine >=0.17.0
/// returns Err where it previously returned a best-effort angle).
#[pyfunction]
#[pyo3(signature = (inputs, target_distance_yards, target_height_inches=0.0, wind_speed_mph=0.0, wind_direction_deg=0.0, temperature_f=59.0, pressure_inhg=29.92, humidity_pct=50.0, altitude_ft=0.0))]
#[allow(clippy::too_many_arguments)]
pub fn calculate_zero_angle(
    inputs: &Bound<'_, PyDict>,
    target_distance_yards: f64,
    target_height_inches: f64,
    wind_speed_mph: f64,
    wind_direction_deg: f64,
    temperature_f: f64,
    pressure_inhg: f64,
    humidity_pct: f64,
    altitude_ft: f64,
) -> PyResult<f64> {
    use ::ballistics_engine::{
        calculate_zero_angle_with_conditions, AtmosphericConditions, WindConditions,
    };

    let mut bi = ballistic_inputs_from_dict(inputs)?;
    // Fully-imperial dict -> SI in place (matches ballistics_rust; NOT geometry_mass_to_si).
    bi.muzzle_velocity *= 0.3048; // fps -> m/s
    bi.bullet_mass *= GRAINS_TO_KG; // grains -> kg
    bi.bullet_diameter *= INCHES_TO_METERS; // inches -> m
    bi.bullet_length *= INCHES_TO_METERS; // inches -> m
    bi.twist_rate *= INCHES_TO_METERS; // inches -> m
    bi.sight_height *= INCHES_TO_METERS; // inches -> m

    let wind = WindConditions {
        speed: wind_speed_mph * 0.44704, // mph -> m/s
        direction: wind_direction_deg.to_radians(),
        // ballistics-engine 0.24.0 added vertical_speed (MBA-728); this entry point has no
        // vertical-wind input, so leave it at the engine default (0.0).
        ..Default::default()
    };
    let atmosphere = AtmosphericConditions {
        temperature: (temperature_f - 32.0) * 5.0 / 9.0, // F -> C
        pressure: pressure_inhg * 33.8639,               // inHg -> hPa
        humidity: humidity_pct,
        altitude: altitude_ft * 0.3048, // ft -> m
    };

    calculate_zero_angle_with_conditions(
        bi,
        target_distance_yards * 0.9144, // yards -> m
        target_height_inches * INCHES_TO_METERS,
        wind,
        atmosphere,
    )
    .map_err(|e| {
        PyRuntimeError::new_err(format!(
            "Unable to find zero angle for target distance {target_distance_yards} yards: {e}"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ::ballistics_engine::wind::WindSock;
    use ::ballistics_engine::wind_shear::WindShearWindSock;
    use nalgebra::Vector3;

    #[test]
    fn shear_and_uniform_wind_segments_share_the_kmh_contract() {
        let segments = vec![(36.0, 90.0, 1_000.0)];

        let uniform =
            WindSock::new(to_wind_segments(segments.clone())).vector_for_range_stateless(100.0);
        let shear = WindShearWindSock::new(wind_segments_kmh_to_mps(segments), None)
            .vector_for_position(Vector3::new(100.0, 0.0, 0.0));

        assert!((uniform.z - shear.z).abs() < 1e-12);
        assert!((shear.norm() - 36.0 * KMH_TO_MPS).abs() < 1e-12);
    }
}
