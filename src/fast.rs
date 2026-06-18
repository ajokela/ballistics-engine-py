// Keystone for migrating the ballistics Flask app off its bespoke `ballistics_rust`
// wrapper onto this binding. Provides:
//   * `ballistic_inputs_from_dict` — full-fidelity dict -> engine BallisticInputs,
//     mirroring ballistics_rust::extract_ballistic_inputs + geometry_mass_to_si.
//   * `fast_integrate` pyfunction — same signature/return contract as
//     ballistics_rust.fast_integrate_rust: returns the scipy-like
//     {t, y(6xN), t_events[3], success} object the app's integrator consumes,
//     over ballistics-engine's fast_trajectory::fast_integrate_with_segments.

use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use ::ballistics_engine::{BallisticInputs as RustBallisticInputs, DragModel};

const GRAINS_TO_KG: f64 = 0.00006479891;
const INCHES_TO_METERS: f64 = 0.0254;

fn parse_drag_model(s: &str) -> PyResult<DragModel> {
    Ok(match s {
        x if x.contains("G1") => DragModel::G1,
        x if x.contains("G6") => DragModel::G6,
        x if x.contains("G7") => DragModel::G7,
        x if x.contains("G8") => DragModel::G8,
        _ => {
            return Err(PyValueError::new_err(format!(
                "Invalid BC type: '{s}'. Expected G1, G6, G7, or G8."
            )))
        }
    })
}

/// Mirror of ballistics_rust::geometry_mass_to_si: convert the imperial
/// geometry/mass the Python layer supplies (grains/inches) to the SI the engine
/// reads (kg/meters) and populate the imperial mirror fields. Call exactly once
/// on a freshly-extracted inputs object (NOT idempotent).
fn geometry_mass_to_si(inputs: &mut RustBallisticInputs) {
    if inputs.caliber_inches == 0.0 {
        inputs.caliber_inches = inputs.bullet_diameter;
    }
    if inputs.weight_grains == 0.0 {
        inputs.weight_grains = inputs.bullet_mass;
    }
    inputs.bullet_mass *= GRAINS_TO_KG;
    inputs.bullet_diameter *= INCHES_TO_METERS;
    inputs.bullet_length *= INCHES_TO_METERS;
}

/// Build a full engine `BallisticInputs` from the app's inputs dict, starting
/// from engine defaults and overriding any key that is present (and non-None).
/// Required: bc_value, bc_type. bc_segments / custom_drag_table are not yet
/// wired (PoC scope) — they default to None.
pub(crate) fn ballistic_inputs_from_dict(d: &Bound<'_, PyDict>) -> PyResult<RustBallisticInputs> {
    let mut inp = RustBallisticInputs::default();

    inp.bc_value = d
        .get_item("bc_value")?
        .ok_or_else(|| PyKeyError::new_err("Missing required field: bc_value"))?
        .extract()?;
    let bc_type: String = d
        .get_item("bc_type")?
        .ok_or_else(|| PyKeyError::new_err("Missing required field: bc_type"))?
        .extract()?;
    inp.bc_type = parse_drag_model(&bc_type)?;

    // Override-if-present: keeps the engine default when a key is absent/None.
    macro_rules! set {
        ($key:literal, $field:ident) => {
            if let Some(v) = d.get_item($key)? {
                if !v.is_none() {
                    inp.$field = v.extract()?;
                }
            }
        };
    }
    set!("bullet_mass", bullet_mass);
    set!("muzzle_velocity", muzzle_velocity);
    set!("bullet_diameter", bullet_diameter);
    set!("bullet_length", bullet_length);
    set!("altitude", altitude);
    set!("twist_rate", twist_rate);
    set!("is_twist_right", is_twist_right);
    set!("target_distance", target_distance);
    set!("muzzle_angle", muzzle_angle);
    set!("wind_speed", wind_speed);
    set!("wind_angle", wind_angle);
    set!("temperature", temperature);
    set!("pressure", pressure);
    set!("humidity", humidity);
    set!("latitude", latitude);
    set!("shooting_angle", shooting_angle);
    set!("sight_height", sight_height);
    set!("ground_threshold", ground_threshold);
    set!("caliber_inches", caliber_inches);
    set!("weight_grains", weight_grains);
    set!("enable_advanced_effects", enable_advanced_effects);
    set!("enable_magnus", enable_magnus);
    set!("enable_coriolis", enable_coriolis);
    set!("use_powder_sensitivity", use_powder_sensitivity);
    set!("powder_temp_sensitivity", powder_temp_sensitivity);
    set!("powder_temp", powder_temp);
    set!("tipoff_yaw", tipoff_yaw);
    set!("tipoff_decay_distance", tipoff_decay_distance);
    set!("use_bc_segments", use_bc_segments);
    set!("use_enhanced_spin_drift", use_enhanced_spin_drift);
    set!("use_form_factor", use_form_factor);
    set!("enable_wind_shear", enable_wind_shear);
    set!("wind_shear_model", wind_shear_model);
    set!("enable_trajectory_sampling", enable_trajectory_sampling);
    set!("sample_interval", sample_interval);
    set!("enable_pitch_damping", enable_pitch_damping);
    set!("enable_precession_nutation", enable_precession_nutation);
    set!("use_cluster_bc", use_cluster_bc);
    set!("use_rk4", use_rk4);
    set!("use_adaptive_rk45", use_adaptive_rk45);

    Ok(inp)
}

/// Fast fixed-step trajectory integration over ballistics-engine's RK45 kernel.
/// Mirrors `ballistics_rust.fast_integrate_rust` exactly: same positional args and
/// the same `{t, y, t_events, success}` return contract the app's integrator reads.
///
/// `wind_segments` is a list of (speed_mps, angle_rad, up_to_range_m) tuples
/// (engine `wind::WindSegment`); empty = no wind. `atmo_params` = 4-vector
/// (base_altitude_m, base_temp_c, base_pressure_hpa, base_density_ratio).
#[pyfunction]
#[pyo3(signature = (inputs, wind_segments, horiz, vert, initial_state, t_span, atmo_params))]
pub fn fast_integrate<'py>(
    py: Python<'py>,
    inputs: &Bound<'py, PyDict>,
    wind_segments: Vec<(f64, f64, f64)>,
    horiz: f64,
    vert: f64,
    initial_state: PyReadonlyArray1<'py, f64>,
    t_span: (f64, f64),
    atmo_params: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyDict>> {
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
    let params = FastIntegrationParams {
        horiz,
        vert,
        initial_state: initial_state_arr,
        t_span,
        atmo_params: (ap[0], ap[1], ap[2], ap[3]),
    };

    let solution = fast_integrate_with_segments(&bi, wind_segments, params);

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
                speed_mps: wind_segments[0].0 * 0.2777778, // km/h -> m/s at reference height
                direction_deg: wind_segments[0].1,
            };
        }
        let sock =
            WindShearWindSock::with_shooter_altitude(wind_segments, Some(profile), bi.altitude);
        sock.vector_for_position(pos)
    } else {
        let sock = WindSock::new(wind_segments);
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

    let r = compute_derivatives(pos, vel, &bi, wind_vector, atmos_tuple, bc_used, omega_vec, _t);
    Ok(PyArray1::from_vec(py, vec![r[0], r[1], r[2], r[3], r[4], r[5]]))
}
