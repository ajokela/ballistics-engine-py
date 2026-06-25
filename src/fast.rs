// Keystone for migrating the ballistics Flask app off its bespoke `ballistics_rust`
// wrapper onto this binding. Provides:
//   * `ballistic_inputs_from_dict` — full-fidelity dict -> engine BallisticInputs,
//     mirroring ballistics_rust::extract_ballistic_inputs + geometry_mass_to_si.
//   * `fast_integrate` pyfunction — same signature/return contract as
//     ballistics_rust.fast_integrate_rust: returns the scipy-like
//     {t, y(6xN), t_events[3], success} object the app's integrator consumes,
//     over ballistics-engine's fast_trajectory::fast_integrate_with_segments.

use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use ::ballistics_engine::drag::DragTable;
use ::ballistics_engine::{BCSegmentData, BallisticInputs as RustBallisticInputs, DragModel};

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
pub(crate) fn geometry_mass_to_si(inputs: &mut RustBallisticInputs) {
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

/// Velocity-segmented BC pairs (mach, bc) — mirrors ballistics_rust::extract_bc_segments.
fn extract_bc_segments(segments: &Bound<'_, PyAny>) -> PyResult<Vec<(f64, f64)>> {
    if segments.is_none() {
        return Ok(Vec::new());
    }
    segments.extract::<Vec<(f64, f64)>>().map_err(|_| {
        PyValueError::new_err("Could not extract BC segments - expected list of (f64, f64)")
    })
}

/// Velocity-based BC segment data — mirrors ballistics_rust::extract_bc_segments_data.
fn extract_bc_segments_data(data: &Bound<'_, PyAny>) -> PyResult<Vec<BCSegmentData>> {
    let mut result = Vec::new();
    if let Ok(list) = data.downcast::<PyList>() {
        for item in list.iter() {
            if let Ok(dict) = item.downcast::<PyDict>() {
                let g = |k: &str| -> PyResult<f64> {
                    dict.get_item(k)?
                        .ok_or_else(|| PyKeyError::new_err(format!("BC segment missing {k}")))?
                        .extract::<f64>()
                };
                result.push(BCSegmentData {
                    velocity_min: g("velocity_min")?,
                    velocity_max: g("velocity_max")?,
                    bc_value: g("bc_value")?,
                });
            }
        }
    }
    Ok(result)
}

/// Optional String field (None when absent or Python None).
fn opt_string(d: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<String>> {
    match d.get_item(key)? {
        Some(v) if !v.is_none() => Ok(Some(v.extract::<String>()?)),
        _ => Ok(None),
    }
}

/// Build a full engine `BallisticInputs` from the app's inputs dict — a FAITHFUL
/// port of ballistics_rust::extract_ballistic_inputs: same field defaults, the
/// derived `enable_magnus`/`enable_coriolis`, parsed bc_segments / bc_segments_data
/// / custom_drag_function, and the hardcoded integrator + datum defaults
/// (use_rk4=true, use_adaptive_rk45=false, sight_height=0.0, ...). NOT a
/// default()+override (that silently diverged on absent keys). Required keys:
/// bc_value, bc_type, bullet_mass, altitude.
pub(crate) fn ballistic_inputs_from_dict(d: &Bound<'_, PyDict>) -> PyResult<RustBallisticInputs> {
    macro_rules! req {
        ($key:literal) => {
            d.get_item($key)?
                .ok_or_else(|| PyKeyError::new_err(concat!("Missing required field: ", $key)))?
                .extract()?
        };
    }
    macro_rules! opt {
        ($key:literal, $default:expr) => {
            match d.get_item($key)? {
                Some(v) if !v.is_none() => v.extract()?,
                _ => $default,
            }
        };
    }

    let bc_value: f64 = req!("bc_value");
    let bc_type: String = req!("bc_type");
    let bc_type_enum = parse_drag_model(&bc_type)?;
    let bullet_mass: f64 = req!("bullet_mass");
    let altitude: f64 = req!("altitude");

    let latitude: Option<f64> = match d.get_item("latitude")? {
        Some(v) if !v.is_none() => Some(v.extract()?),
        _ => None,
    };
    let bullet_cluster: Option<usize> = match d.get_item("bullet_cluster")? {
        Some(v) if !v.is_none() => Some(v.extract()?),
        _ => None,
    };
    let enable_advanced_effects: bool = opt!("enable_advanced_effects", false);

    let bc_segments = match d.get_item("bc_segments")? {
        Some(s) if !s.is_none() => {
            let v = extract_bc_segments(&s)?;
            if v.is_empty() { None } else { Some(v) }
        }
        _ => None,
    };
    let bc_segments_data = match d.get_item("bc_segments_data")? {
        Some(s) if !s.is_none() => {
            let v = extract_bc_segments_data(&s)?;
            if v.is_empty() { None } else { Some(v) }
        }
        _ => None,
    };
    let custom_drag_table = match d.get_item("custom_drag_function")? {
        Some(cdm) if !cdm.is_none() => match cdm.downcast::<PyDict>() {
            Ok(dict) => {
                let mach = dict
                    .get_item("mach_numbers")?
                    .and_then(|v| v.extract::<Vec<f64>>().ok());
                let cd = dict
                    .get_item("drag_coefficients")?
                    .and_then(|v| v.extract::<Vec<f64>>().ok());
                match (mach, cd) {
                    (Some(m), Some(c)) if m.len() == c.len() && !m.is_empty() => {
                        Some(DragTable::new(m, c))
                    }
                    _ => None,
                }
            }
            Err(_) => None,
        },
        _ => None,
    };

    Ok(RustBallisticInputs {
        bc_value,
        bc_type: bc_type_enum,
        bullet_mass,
        muzzle_velocity: opt!("muzzle_velocity", 0.0),
        altitude,
        twist_rate: opt!("twist_rate", 0.0),
        bullet_length: opt!("bullet_length", 0.0),
        bullet_diameter: opt!("bullet_diameter", 0.0),
        target_distance: opt!("target_distance", 0.0),
        muzzle_angle: opt!("muzzle_angle", 0.0),
        wind_speed: opt!("wind_speed", 0.0),
        wind_angle: opt!("wind_angle", 0.0),
        temperature: opt!("temperature", 15.0),
        pressure: opt!("pressure", 1013.25),
        humidity: opt!("humidity", 0.0),
        latitude,
        enable_advanced_effects,
        enable_magnus: enable_advanced_effects,
        enable_coriolis: enable_advanced_effects && latitude.is_some(),
        is_twist_right: opt!("is_twist_right", true),
        shooting_angle: opt!("shooting_angle", 0.0),
        azimuth_angle: 0.0,
        // Coriolis firing bearing (engine 0.21.0+): degrees, 0=N, 90=E. Drives the
        // Eotvos vertical term + lateral azimuth on the fast/MC path.
        shot_azimuth: opt!("shot_direction", 0.0_f64).to_radians(),
        use_powder_sensitivity: opt!("use_powder_sensitivity", false),
        powder_temp_sensitivity: opt!("powder_temp_sensitivity", 0.0),
        powder_temp: opt!("powder_temp", 70.0),
        tipoff_yaw: opt!("tipoff_yaw", 0.0),
        tipoff_decay_distance: opt!("tipoff_decay_distance", 20.0),
        ground_threshold: opt!("ground_threshold", -100.0),
        bc_segments,
        caliber_inches: opt!("caliber_inches", 0.0),
        weight_grains: opt!("weight_grains", 0.0),
        use_bc_segments: opt!("use_bc_segments", false),
        bullet_id: opt_string(d, "bullet_id")?,
        bc_segments_data,
        use_enhanced_spin_drift: false,
        use_form_factor: opt!("use_form_factor", true),
        manufacturer: opt_string(d, "manufacturer")?,
        bullet_model: opt_string(d, "bullet_model")?,
        enable_wind_shear: opt!("enable_wind_shear", false),
        wind_shear_model: opt!("wind_shear_model", "none".to_string()),
        use_cluster_bc: opt!("use_cluster_bc", false),
        bullet_cluster,
        custom_drag_table,
        bc_type_str: Some(bc_type),
        enable_pitch_damping: false,
        enable_precession_nutation: false,
        enable_aerodynamic_jump: opt!("enable_aerodynamic_jump", false),
        use_rk4: true,
        use_adaptive_rk45: false,
        enable_trajectory_sampling: false,
        sample_interval: 10.0,
        sight_height: 0.0,
        muzzle_height: 0.0,
        target_height: 0.0,
    })
}

/// Fast fixed-step trajectory integration over ballistics-engine's RK45 kernel.
/// Mirrors `ballistics_rust.fast_integrate_rust` exactly: same positional args and
/// the same `{t, y, t_events, success}` return contract the app's integrator reads.
///
/// `wind_segments` is a list of (speed_kmh, angle_deg, until_distance_m) tuples
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
