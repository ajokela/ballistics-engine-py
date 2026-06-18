// Scalar query helpers (drag / atmosphere), mirroring the ballistics_rust
// pyfunctions of the same purpose. Thin pass-throughs over public engine APIs.

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use ::ballistics_engine::DragModel;

fn parse_drag_model(s: &str) -> PyResult<DragModel> {
    Ok(match s.to_uppercase().as_str() {
        "G1" => DragModel::G1,
        "G6" => DragModel::G6,
        "G7" => DragModel::G7,
        "G8" => DragModel::G8,
        _ => {
            return Err(PyValueError::new_err(format!(
                "Invalid drag model: '{s}'. Expected G1, G6, G7, or G8."
            )))
        }
    })
}

/// Drag coefficient for a Mach number and standard drag model.
#[pyfunction]
pub fn get_drag_coefficient(mach: f64, drag_model: &str) -> PyResult<f64> {
    Ok(::ballistics_engine::drag::get_drag_coefficient(
        mach,
        &parse_drag_model(drag_model)?,
    ))
}

/// Drag coefficient with the transonic drag-rise correction optionally applied.
#[pyfunction]
#[pyo3(signature = (mach, drag_model, apply_transonic_correction=true, caliber=None, weight_grains=None))]
pub fn get_drag_coefficient_transonic(
    mach: f64,
    drag_model: &str,
    apply_transonic_correction: bool,
    caliber: Option<f64>,
    weight_grains: Option<f64>,
) -> PyResult<f64> {
    Ok(::ballistics_engine::drag::get_drag_coefficient_with_transonic(
        mach,
        &parse_drag_model(drag_model)?,
        apply_transonic_correction,
        None, // let the engine infer projectile shape
        caliber,
        weight_grains,
    ))
}

/// Velocity-segmented BC interpolation: segments are (mach, bc) pairs.
#[pyfunction]
pub fn interpolated_bc(mach: f64, segments: Vec<(f64, f64)>) -> PyResult<f64> {
    Ok(::ballistics_engine::drag::interpolated_bc(mach, &segments))
}

/// Standard/overridden atmosphere -> (air_density_kg_m3, speed_of_sound_mps).
/// `temp_override_c` / `press_override_hpa` = None means derive from altitude.
#[pyfunction]
#[pyo3(signature = (altitude_m, temp_override_c=None, press_override_hpa=None, humidity_percent=0.0))]
pub fn calculate_atmosphere(
    altitude_m: f64,
    temp_override_c: Option<f64>,
    press_override_hpa: Option<f64>,
    humidity_percent: f64,
) -> PyResult<(f64, f64)> {
    Ok(::ballistics_engine::atmosphere::calculate_atmosphere(
        altitude_m,
        temp_override_c,
        press_override_hpa,
        humidity_percent,
    ))
}

/// CIPM air density (kg/m^3) from temperature (C), pressure (hPa), humidity (%).
#[pyfunction]
pub fn calculate_air_density_cipm(
    temp_c: f64,
    pressure_hpa: f64,
    humidity_percent: f64,
) -> PyResult<f64> {
    Ok(::ballistics_engine::atmosphere::calculate_air_density_cipm(
        temp_c,
        pressure_hpa,
        humidity_percent,
    ))
}

/// Local atmosphere at `altitude_m`. `atmos_params` is either a 2-vector of
/// already-resolved (density, sound) or a 4-vector
/// (base_altitude_m, base_temp_c, base_pressure_hpa, base_density_ratio).
#[pyfunction]
pub fn get_local_atmosphere<'py>(
    _py: Python<'py>,
    altitude_m: f64,
    atmos_params: PyReadonlyArray1<'py, f64>,
) -> PyResult<(f64, f64)> {
    let p = atmos_params.as_array();
    match p.len() {
        2 => Ok((p[0], p[1])),
        4 => Ok(::ballistics_engine::atmosphere::get_local_atmosphere(
            altitude_m, p[0], p[1], p[2], p[3],
        )),
        n => Err(PyValueError::new_err(format!(
            "Atmosphere params must contain 2 or 4 values, got {n}"
        ))),
    }
}
