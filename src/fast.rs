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
