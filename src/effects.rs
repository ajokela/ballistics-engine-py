// Stability / spin-drift / transonic scalar helpers, mirroring the
// ballistics_rust pyfunctions. Thin pass-throughs over public engine APIs.

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use ::ballistics_engine::atmosphere::calculate_air_density_cipm;
use ::ballistics_engine::spin_drift_advanced::calculate_advanced_spin_drift;
use ::ballistics_engine::stability_advanced::calculate_advanced_stability;
use ::ballistics_engine::transonic_drag::{
    get_projectile_shape as engine_get_projectile_shape,
    transonic_correction as engine_transonic_correction, ProjectileShape,
};

/// Transonic drag-rise correction factor for a base Cd at a Mach number.
/// `shape_str` in {spitzer, round_nose, flat_base, boat_tail}; default spitzer.
#[pyfunction]
#[pyo3(signature = (mach, base_cd, shape_str=None, include_wave_drag=None))]
pub fn transonic_correction(
    mach: f64,
    base_cd: f64,
    shape_str: Option<&str>,
    include_wave_drag: Option<bool>,
) -> PyResult<f64> {
    let shape = shape_str
        .map(ProjectileShape::from_str)
        .unwrap_or(ProjectileShape::Spitzer);
    Ok(engine_transonic_correction(
        mach,
        base_cd,
        shape,
        include_wave_drag.unwrap_or(true),
    ))
}

/// Infer projectile shape from caliber (in), weight (grains), drag model;
/// returns one of spitzer / round_nose / flat_base / boat_tail.
#[pyfunction]
pub fn get_projectile_shape(caliber: f64, weight_grains: f64, g_model: &str) -> PyResult<String> {
    let shape = engine_get_projectile_shape(caliber, weight_grains, g_model);
    Ok(match shape {
        ProjectileShape::Spitzer => "spitzer",
        ProjectileShape::RoundNose => "round_nose",
        ProjectileShape::FlatBase => "flat_base",
        ProjectileShape::BoatTail => "boat_tail",
    }
    .to_string())
}

/// Advanced gyroscopic stability factor (Sg) with full explicit parameters.
#[pyfunction]
#[pyo3(signature = (mass_grains, velocity_fps, twist_rate_inches, caliber_inches, length_inches, air_density_kg_m3, temperature_k, bullet_type="match", has_boat_tail=true, has_plastic_tip=false))]
#[allow(clippy::too_many_arguments)]
pub fn compute_stability_advanced(
    mass_grains: f64,
    velocity_fps: f64,
    twist_rate_inches: f64,
    caliber_inches: f64,
    length_inches: f64,
    air_density_kg_m3: f64,
    temperature_k: f64,
    bullet_type: &str,
    has_boat_tail: bool,
    has_plastic_tip: bool,
) -> PyResult<f64> {
    Ok(calculate_advanced_stability(
        mass_grains,
        velocity_fps,
        twist_rate_inches,
        caliber_inches,
        length_inches,
        air_density_kg_m3,
        temperature_k,
        bullet_type,
        has_boat_tail,
        has_plastic_tip,
    ))
}

/// Advanced spin drift (meters) with full explicit parameters.
#[pyfunction]
#[pyo3(signature = (stability_factor, time_of_flight_s, velocity_mps, muzzle_velocity_mps, spin_rate_rad_s, caliber_m, mass_kg, air_density_kg_m3, is_right_twist, bullet_type="match"))]
#[allow(clippy::too_many_arguments)]
pub fn compute_spin_drift_advanced(
    stability_factor: f64,
    time_of_flight_s: f64,
    velocity_mps: f64,
    muzzle_velocity_mps: f64,
    spin_rate_rad_s: f64,
    caliber_m: f64,
    mass_kg: f64,
    air_density_kg_m3: f64,
    is_right_twist: bool,
    bullet_type: &str,
) -> PyResult<f64> {
    Ok(calculate_advanced_spin_drift(
        stability_factor,
        time_of_flight_s,
        velocity_mps,
        muzzle_velocity_mps,
        spin_rate_rad_s,
        caliber_m,
        mass_kg,
        air_density_kg_m3,
        is_right_twist,
        bullet_type,
    ))
}

/// Simple spin drift interface (uses the advanced model with typical .308
/// defaults for the unspecified parameters) — mirrors ballistics_rust.
#[pyfunction]
pub fn compute_spin_drift(
    time_s: f64,
    stability: f64,
    twist_rate: f64,
    is_twist_right: bool,
) -> PyResult<f64> {
    if twist_rate == 0.0 || time_s <= 0.0 || stability <= 0.0 {
        return Ok(0.0);
    }
    let velocity_mps = 600.0;
    let muzzle_velocity_mps = 850.0;
    let spin_rate_rad_s = (muzzle_velocity_mps * 39.37 / twist_rate) * 2.0 * std::f64::consts::PI;
    let caliber_m = 0.00308;
    let mass_kg = 0.0108;
    let air_density = 1.225;
    Ok(calculate_advanced_spin_drift(
        stability,
        time_s,
        velocity_mps,
        muzzle_velocity_mps,
        spin_rate_rad_s,
        caliber_m,
        mass_kg,
        air_density,
        is_twist_right,
        "match",
    ))
}

/// Bullet-type tag from the dict's bullet_model (mirrors ballistics_rust).
fn bullet_type_from_dict(inputs: &Bound<'_, PyDict>) -> PyResult<String> {
    let s = match inputs.get_item("bullet_model")? {
        Some(v) if !v.is_none() => v.extract::<String>().unwrap_or_default(),
        _ => String::new(),
    };
    Ok(if s.contains("Match") || s.contains("SMK") {
        "match"
    } else if s.contains("VLD") {
        "vld"
    } else if s.contains("Hybrid") {
        "hybrid"
    } else {
        "match"
    }
    .to_string())
}

/// Gyroscopic stability factor (Sg) from an imperial inputs dict + a 4-vector
/// (base_alt_m, temp_c, pressure_hpa, density_ratio). Uses the ADVANCED model
/// with 50% humidity, boat-tail assumed — mirrors ballistics_rust.
#[pyfunction]
pub fn compute_stability_coefficient<'py>(
    inputs: &Bound<'py, PyDict>,
    atmos_params: PyReadonlyArray1<'py, f64>,
) -> PyResult<f64> {
    // Imperial dict (grains/fps/inches) used directly — NO geometry_mass_to_si.
    let bi = crate::fast::ballistic_inputs_from_dict(inputs)?;
    let p = atmos_params.as_array();
    if p.len() != 4 {
        return Err(PyValueError::new_err(format!(
            "Atmosphere params must contain 4 values, got {}",
            p.len()
        )));
    }
    let (_altitude, temp_c, pressure_hpa, _density_ratio) = (p[0], p[1], p[2], p[3]);
    let temp_k = temp_c + 273.15;
    let air_density = calculate_air_density_cipm(temp_c, pressure_hpa, 50.0);
    let bullet_type = bullet_type_from_dict(inputs)?;
    Ok(calculate_advanced_stability(
        bi.bullet_mass,     // grains (unconverted)
        bi.muzzle_velocity, // fps (unconverted)
        bi.twist_rate,      // inches
        bi.bullet_diameter, // inches
        bi.bullet_length,   // inches
        air_density,
        temp_k,
        &bullet_type,
        true,  // boat tail assumed for modern bullets
        false, // plastic tip detection not wired
    ))
}
