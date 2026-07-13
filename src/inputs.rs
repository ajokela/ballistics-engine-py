// Copyright 2025 Alex Jokela
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// or the MIT license:
//
//     http://opensource.org/licenses/MIT
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// MBA-1295 Phase 1: the ONE dict -> engine BallisticInputs pipeline, shared by
// `PyBallisticInputs::from_dict`, the fast path (fast_integrate / derivatives /
// calculate_zero_angle / monte_carlo_parallel, via fast.rs's re-export), and the new
// solver-class pyfunctions `auto_zero_inputs` / `run_monte_carlo`.
//
// Two dict key vocabularies resolve onto the same BallisticInputs fields:
//   * the richer ballistics_rust-derived keys (bc_value, bullet_mass, altitude,
//     enable_coriolis, bc_segments, custom_drag_function, wind shear, cluster BC, ...) —
//     historically only fast.rs's `ballistic_inputs_from_dict` understood these;
//   * the legacy PyBallisticInputs imperial keys (bc, bullet_weight_grains,
//     muzzle_velocity_fps, bullet_diameter_inches, ...) — historically only
//     `PyBallisticInputs::from_dict`'s 11-key allowlist understood these, silently
//     dropping everything else.
// Where a field has both a rich key and a legacy alias, the rich key wins when both are
// present; the legacy alias is honored when the rich key is absent; the engine/binding
// default applies when neither is present. This relaxes the old fast-path contract of
// hard-requiring bc_value/bc_type/bullet_mass/altitude (PyKeyError on absence) to
// alias-then-default, so a legacy-keyed dict (which never had those names) parses cleanly.
//
// The returned BallisticInputs is in the same "mixed imperial, pre-SI" representation the
// existing fast-path callers already expect at this boundary: mass/diameter/length in
// grains/inches, muzzle_velocity in fps, target_distance in yards, altitude in feet, wind
// speed in km/h, muzzle_angle/wind_angle/cant_angle in degrees, temperature/pressure/
// humidity in their app-native units (Celsius / hPa / percent) — EXCEPT `shooting_angle`,
// which (matching the pre-existing convention baked into fast_integrate/derivatives/
// monte_carlo_parallel) is already RADIANS when supplied via its rich key name; the legacy
// `shooting_angle_degrees` alias has no rich-key equivalent to defer to, so it is converted
// to radians immediately at parse time. Call `full_to_si` to convert everything else to the
// engine's native SI/radians in one step (generalizes `geometry_mass_to_si` +
// monte_carlo_parallel's retired `mc_inputs_to_si` + calculate_zero_angle's manual block).

use ::ballistics_engine::drag::DragTable;
use ::ballistics_engine::{BCSegmentData, BallisticInputs as RustBallisticInputs, DragModel};
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

const GRAINS_TO_KG: f64 = 0.00006479891;
const INCHES_TO_METERS: f64 = 0.0254;
const FPS_TO_MPS: f64 = 0.3048;
const YARDS_TO_METERS: f64 = 0.9144;
const KMH_TO_MPS: f64 = 1000.0 / 3600.0;

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

/// Convert every remaining pre-SI field `ballistic_inputs_from_dict` leaves in its
/// mixed-imperial representation into the engine's native SI/radians (MBA-1295).
/// Generalizes `geometry_mass_to_si` (called internally) plus the ad hoc conversion
/// blocks previously duplicated across `calculate_zero_angle` and
/// `monte_carlo_parallel::mc_inputs_to_si`. `shooting_angle` is intentionally left
/// untouched — `ballistic_inputs_from_dict` already resolves it to radians (see the
/// module doc comment).
pub(crate) fn full_to_si(inputs: &mut RustBallisticInputs) {
    geometry_mass_to_si(inputs);
    inputs.muzzle_velocity *= FPS_TO_MPS;
    inputs.target_distance *= YARDS_TO_METERS;
    inputs.altitude *= 0.3048; // feet -> m
    inputs.sight_height *= INCHES_TO_METERS;
    inputs.muzzle_height *= INCHES_TO_METERS;
    inputs.target_height *= INCHES_TO_METERS;
    inputs.wind_speed *= KMH_TO_MPS;
    inputs.wind_angle = inputs.wind_angle.to_radians();
    inputs.muzzle_angle = inputs.muzzle_angle.to_radians();
    inputs.cant_angle = inputs.cant_angle.to_radians();
    inputs.humidity /= 100.0; // percent -> 0-1 fraction (engine >=0.17.0 multiplies by 100)
}

/// Velocity-segmented BC pairs (mach, bc) — mirrors ballistics_rust::extract_bc_segments.
/// Accepts both `[mach, bc]` lists and `(mach, bc)` tuples per row (MBA-1295: pyo3's
/// blanket tuple `FromPyObject` only accepts actual Python tuples, which would reject the
/// list-of-lists shape a JSON-decoded Flask request body actually sends).
fn extract_bc_segments(segments: &Bound<'_, PyAny>) -> PyResult<Vec<(f64, f64)>> {
    if segments.is_none() {
        return Ok(Vec::new());
    }
    let err = || {
        PyValueError::new_err("Could not extract BC segments - expected list of [mach, bc] pairs")
    };
    let rows: Vec<Vec<f64>> = segments.extract().map_err(|_| err())?;
    rows.into_iter()
        .map(|row| match row.as_slice() {
            [mach, bc] => Ok((*mach, *bc)),
            _ => Err(err()),
        })
        .collect()
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

/// f64 field with a canonical (rich) key and an optional legacy alias key; canonical wins
/// when both are present.
fn f64_field(
    d: &Bound<'_, PyDict>,
    primary: &str,
    legacy: Option<&str>,
    default: f64,
) -> PyResult<f64> {
    if let Some(v) = d.get_item(primary)? {
        if !v.is_none() {
            return v.extract();
        }
    }
    if let Some(legacy_key) = legacy {
        if let Some(v) = d.get_item(legacy_key)? {
            if !v.is_none() {
                return v.extract();
            }
        }
    }
    Ok(default)
}

/// bool field with a canonical (rich) key and an optional legacy alias key.
fn bool_field(
    d: &Bound<'_, PyDict>,
    primary: &str,
    legacy: Option<&str>,
    default: bool,
) -> PyResult<bool> {
    if let Some(v) = d.get_item(primary)? {
        if !v.is_none() {
            return v.extract();
        }
    }
    if let Some(legacy_key) = legacy {
        if let Some(v) = d.get_item(legacy_key)? {
            if !v.is_none() {
                return v.extract();
            }
        }
    }
    Ok(default)
}

/// Resolve BC drag model: canonical `bc_type` (string, strictly validated) wins; falls
/// back to the legacy `drag_model` key (a `DragModel` object OR a loosely-matched string,
/// mirroring the historical `PyBallisticInputs.from_dict` behavior exactly); defaults to
/// G7 when neither is present.
fn resolve_bc_type(d: &Bound<'_, PyDict>) -> PyResult<(DragModel, String)> {
    if let Some(v) = d.get_item("bc_type")? {
        if !v.is_none() {
            let s: String = v.extract()?;
            let dm = parse_drag_model(&s)?;
            return Ok((dm, s));
        }
    }
    if let Some(v) = d.get_item("drag_model")? {
        if !v.is_none() {
            if let Ok(pdm) = v.extract::<crate::PyDragModel>() {
                let dm = pdm.inner;
                return Ok((dm, format!("{dm:?}")));
            }
            let s: String = v.extract()?;
            let dm = if s.contains("G1") {
                DragModel::G1
            } else if s.contains("G8") {
                DragModel::G8
            } else if s.contains("G6") {
                DragModel::G6
            } else {
                DragModel::G7
            };
            return Ok((dm, s));
        }
    }
    Ok((DragModel::G7, "G7".to_string()))
}

/// Resolve a custom drag table. Two dict shapes are accepted, checked in order:
///   1. `custom_drag_table`: a list of `[mach, cd]` pairs, VALIDATED via
///      `DragTable::try_new` (equal-length axes, >=2 points, ascending finite Mach,
///      finite positive Cd) — parse errors surface as `ValueError`. (MBA-1295)
///   2. `custom_drag_function`: `{"mach_numbers": [...], "drag_coefficients": [...]}`,
///      the pre-existing UNVALIDATED path (silently ignored on malformed input, matching
///      historical behavior).
fn resolve_custom_drag_table(d: &Bound<'_, PyDict>) -> PyResult<Option<DragTable>> {
    if let Some(v) = d.get_item("custom_drag_table")? {
        if !v.is_none() {
            // Extract each row as `Vec<f64>` (accepts BOTH `[mach, cd]` lists and
            // `(mach, cd)` tuples — pyo3's blanket tuple `FromPyObject` only accepts
            // actual Python tuples, which would reject the more natural JSON-shaped
            // list-of-lists a Flask/JSON caller sends).
            let err = || {
                PyValueError::new_err(
                    "custom_drag_table must be a list of [mach, cd] pairs (>=2 points)",
                )
            };
            let rows: Vec<Vec<f64>> = v.extract().map_err(|_| err())?;
            let mut mach = Vec::with_capacity(rows.len());
            let mut cd = Vec::with_capacity(rows.len());
            for row in rows {
                match row.as_slice() {
                    [m, c] => {
                        mach.push(*m);
                        cd.push(*c);
                    }
                    _ => return Err(err()),
                }
            }
            let table = DragTable::try_new(mach, cd).map_err(PyValueError::new_err)?;
            return Ok(Some(table));
        }
    }
    if let Some(cdm) = d.get_item("custom_drag_function")? {
        if !cdm.is_none() {
            if let Ok(dict) = cdm.downcast::<PyDict>() {
                let mach = dict
                    .get_item("mach_numbers")?
                    .and_then(|v| v.extract::<Vec<f64>>().ok());
                let cd = dict
                    .get_item("drag_coefficients")?
                    .and_then(|v| v.extract::<Vec<f64>>().ok());
                if let (Some(m), Some(c)) = (mach, cd) {
                    if m.len() == c.len() && !m.is_empty() {
                        return Ok(Some(DragTable::new(m, c)));
                    }
                }
            }
        }
    }
    Ok(None)
}

/// Resolve `shooting_angle` (radians): the canonical rich key is already radians
/// (pre-existing convention); the legacy `shooting_angle_degrees` alias is degrees and is
/// converted immediately since there is no later pass that would do it.
fn resolve_shooting_angle(d: &Bound<'_, PyDict>) -> PyResult<f64> {
    if let Some(v) = d.get_item("shooting_angle")? {
        if !v.is_none() {
            return v.extract();
        }
    }
    if let Some(v) = d.get_item("shooting_angle_degrees")? {
        if !v.is_none() {
            let deg: f64 = v.extract()?;
            return Ok(deg.to_radians());
        }
    }
    Ok(0.0)
}

/// Build a full engine `BallisticInputs` from an inputs dict, accepting BOTH the rich
/// ballistics_rust-derived keys and the legacy `PyBallisticInputs` imperial keys (see the
/// module doc comment for the full alias table and unit conventions). No fields are
/// hard-required; every field falls back to a sensible default (relaxed from the historical
/// fast-path contract, which raised `PyKeyError` on a handful of missing keys).
pub(crate) fn ballistic_inputs_from_dict(d: &Bound<'_, PyDict>) -> PyResult<RustBallisticInputs> {
    macro_rules! opt {
        ($key:literal, $default:expr) => {
            match d.get_item($key)? {
                Some(v) if !v.is_none() => v.extract()?,
                _ => $default,
            }
        };
    }

    let bc_value = f64_field(d, "bc_value", Some("bc"), 0.5)?;
    let (bc_type_enum, bc_type_str) = resolve_bc_type(d)?;
    let bullet_mass = f64_field(d, "bullet_mass", Some("bullet_weight_grains"), 168.0)?;
    let altitude = f64_field(d, "altitude", None, 0.0)?;

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
            if v.is_empty() {
                None
            } else {
                Some(v)
            }
        }
        _ => None,
    };
    let bc_segments_data = match d.get_item("bc_segments_data")? {
        Some(s) if !s.is_none() => {
            let v = extract_bc_segments_data(&s)?;
            if v.is_empty() {
                None
            } else {
                Some(v)
            }
        }
        _ => None,
    };
    let custom_drag_table = resolve_custom_drag_table(d)?;
    let is_twist_right = bool_field(d, "is_twist_right", Some("is_right_twist"), true)?;
    let shooting_angle = resolve_shooting_angle(d)?;

    Ok(RustBallisticInputs {
        bc_value,
        bc_type: bc_type_enum,
        bullet_mass,
        muzzle_velocity: f64_field(d, "muzzle_velocity", Some("muzzle_velocity_fps"), 2650.0)?,
        altitude,
        twist_rate: f64_field(d, "twist_rate", Some("twist_rate_inches"), 11.25)?,
        bullet_length: f64_field(d, "bullet_length", Some("bullet_length_inches"), 1.2)?,
        bullet_diameter: f64_field(d, "bullet_diameter", Some("bullet_diameter_inches"), 0.308)?,
        target_distance: f64_field(d, "target_distance", Some("zero_distance_yards"), 100.0)?,
        muzzle_angle: opt!("muzzle_angle", 0.0),
        wind_speed: opt!("wind_speed", 0.0),
        wind_angle: opt!("wind_angle", 0.0),
        temperature: opt!("temperature", 15.0),
        pressure: opt!("pressure", 1013.25),
        humidity: opt!("humidity", 0.0),
        latitude,
        enable_advanced_effects,
        enable_magnus: enable_advanced_effects,
        // Coriolis is independent of advanced effects (engine 0.21.2+): default to the
        // historical derivation, but let callers request Coriolis-only via the dict key.
        enable_coriolis: opt!("enable_coriolis", enable_advanced_effects && latitude.is_some()),
        is_twist_right,
        shooting_angle,
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
        // MBA-1134: enable the engine's canonical Litz spin drift on the fast path when the
        // caller requests advanced effects (mirrors enable_magnus above and the CLI). Honor
        // an explicit request if the caller sets it, otherwise derive from
        // enable_advanced_effects.
        use_enhanced_spin_drift: opt!("use_enhanced_spin_drift", enable_advanced_effects),
        use_form_factor: opt!("use_form_factor", true),
        manufacturer: opt_string(d, "manufacturer")?,
        bullet_model: opt_string(d, "bullet_model")?,
        enable_wind_shear: opt!("enable_wind_shear", false),
        wind_shear_model: opt!("wind_shear_model", "none".to_string()),
        use_cluster_bc: opt!("use_cluster_bc", false),
        bullet_cluster,
        custom_drag_table,
        bc_type_str: Some(bc_type_str),
        enable_pitch_damping: false,
        enable_precession_nutation: false,
        enable_aerodynamic_jump: opt!("enable_aerodynamic_jump", false),
        // MBA-1295: previously hardcoded true/false regardless of dict content; now
        // dict-overridable (same defaults, so absent-key callers are unaffected).
        use_rk4: opt!("use_rk4", true),
        use_adaptive_rk45: opt!("use_adaptive_rk45", false),
        // MBA-1295: previously hardcoded false/10.0 regardless of dict content.
        enable_trajectory_sampling: opt!("enable_trajectory_sampling", false),
        sample_interval: f64_field(d, "sample_interval", Some("sample_interval_m"), 10.0)?,
        // MBA-1295: previously hardcoded 0.0 regardless of dict content, so sight/muzzle/
        // target heights were silently unsettable on the fast path. sight_height keeps the
        // legacy PyBallisticInputs missing-key default of 1.5 INCHES (review fix — the
        // Phase 1 first cut regressed it to 0.0); monte_carlo_parallel re-zeroes all three
        // datum fields at its own boundary to stay bore-relative (see mc_inputs_to_si).
        sight_height: f64_field(d, "sight_height", Some("sight_height_inches"), 1.5)?,
        muzzle_height: f64_field(d, "muzzle_height", Some("muzzle_height_inches"), 0.0)?,
        target_height: f64_field(d, "target_height", Some("target_height_inches"), 0.0)?,
        // MBA-1295: rifle cant, degrees pre-SI (converted to radians by `full_to_si`); no
        // prior rich-key equivalent existed.
        cant_angle: f64_field(d, "cant_angle_degrees", None, 0.0)?,
        ..Default::default()
    })
}
