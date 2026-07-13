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

// Common, intentional patterns in PyO3 wrappers: many-arg pyfunctions mirroring
// the Python API, and `default() + override-if-present` dict extraction.
#![allow(clippy::too_many_arguments)]
#![allow(clippy::field_reassign_with_default)]

use pyo3::prelude::*;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::types::PyDict;
use ::ballistics_engine::{
    DragModel, BallisticInputs as RustBallisticInputs,
    WindConditions as RustWindConditions,
    AtmosphericConditions as RustAtmosphericConditions,
    TrajectorySolver as RustTrajectorySolver,
    TrajectoryResult as RustTrajectoryResult,
    TrajectoryPoint as RustTrajectoryPoint,
};

mod effects;
mod fast;
mod helpers;
mod inputs;
mod montecarlo;

// Unit conversion constants
const GRAINS_TO_KG: f64 = 0.00006479891;
const FPS_TO_MPS: f64 = 0.3048;
const YARDS_TO_METERS: f64 = 0.9144;
const INCHES_TO_METERS: f64 = 0.0254;
const MPH_TO_MPS: f64 = 0.44704;
const DEGREES_TO_RADIANS: f64 = std::f64::consts::PI / 180.0;

/// Python wrapper for DragModel enum
#[pyclass(name = "DragModel")]
#[derive(Clone)]
pub struct PyDragModel {
    pub(crate) inner: DragModel,
}

#[pymethods]
impl PyDragModel {
    #[staticmethod]
    fn g1() -> Self {
        PyDragModel { inner: DragModel::G1 }
    }

    #[staticmethod]
    fn g7() -> Self {
        PyDragModel { inner: DragModel::G7 }
    }

    #[staticmethod]
    fn g8() -> Self {
        PyDragModel { inner: DragModel::G8 }
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

/// Wind conditions
#[pyclass(name = "WindConditions")]
#[derive(Clone)]
pub struct PyWindConditions {
    #[pyo3(get, set)]
    pub speed_mph: f64,
    #[pyo3(get, set)]
    pub direction_degrees: f64,
    /// Vertical wind component, m/s (engine-native unit — no imperial conversion, matching
    /// its documented convention). Positive = updraft (raises POI downrange). Default 0.0.
    /// (MBA-1295, wrapping the ballistics-engine 0.24.0 / MBA-728 field.)
    #[pyo3(get, set)]
    pub vertical_speed_mps: f64,
}

#[pymethods]
impl PyWindConditions {
    #[new]
    #[pyo3(signature = (speed_mph=0.0, direction_degrees=0.0, vertical_speed_mps=0.0))]
    fn new(speed_mph: f64, direction_degrees: f64, vertical_speed_mps: f64) -> Self {
        PyWindConditions {
            speed_mph,
            direction_degrees,
            vertical_speed_mps,
        }
    }
}

impl PyWindConditions {
    fn to_rust(&self) -> RustWindConditions {
        RustWindConditions {
            speed: self.speed_mph * MPH_TO_MPS,
            direction: self.direction_degrees * DEGREES_TO_RADIANS,
            vertical_speed: self.vertical_speed_mps,
        }
    }
}

/// Atmospheric conditions
#[pyclass(name = "AtmosphericConditions")]
#[derive(Clone)]
pub struct PyAtmosphericConditions {
    #[pyo3(get, set)]
    pub temperature_f: f64,
    #[pyo3(get, set)]
    pub pressure_inhg: f64,
    #[pyo3(get, set)]
    pub humidity_percent: f64,
    #[pyo3(get, set)]
    pub altitude_feet: f64,
}

#[pymethods]
impl PyAtmosphericConditions {
    #[new]
    #[pyo3(signature = (temperature_f=59.0, pressure_inhg=29.92, humidity_percent=50.0, altitude_feet=0.0))]
    fn new(temperature_f: f64, pressure_inhg: f64, humidity_percent: f64, altitude_feet: f64) -> Self {
        PyAtmosphericConditions {
            temperature_f,
            pressure_inhg,
            humidity_percent,
            altitude_feet,
        }
    }
}

impl PyAtmosphericConditions {
    fn to_rust(&self) -> RustAtmosphericConditions {
        RustAtmosphericConditions {
            temperature: (self.temperature_f - 32.0) * 5.0 / 9.0,  // F to C
            pressure: self.pressure_inhg * 33.8639,  // inHg to hPa
            humidity: self.humidity_percent,
            altitude: self.altitude_feet * 0.3048,  // feet to meters
        }
    }
}

/// Trajectory point
#[pyclass(name = "TrajectoryPoint")]
pub struct PyTrajectoryPoint {
    #[pyo3(get)]
    pub time: f64,
    #[pyo3(get)]
    pub x: f64,  // yards
    #[pyo3(get)]
    pub y: f64,  // yards
    #[pyo3(get)]
    pub z: f64,  // yards
    #[pyo3(get)]
    pub velocity_fps: f64,
    #[pyo3(get)]
    pub energy_ftlbs: f64,
}

impl PyTrajectoryPoint {
    fn from_rust(point: &RustTrajectoryPoint, bullet_mass_kg: f64) -> Self {
        let vel_fps = point.velocity_magnitude / FPS_TO_MPS;
        let energy_ftlbs = 0.5 * bullet_mass_kg * point.velocity_magnitude * point.velocity_magnitude / 1.35582;  // J to ft-lbs

        PyTrajectoryPoint {
            time: point.time,
            x: point.position.x / YARDS_TO_METERS,
            y: point.position.y / YARDS_TO_METERS,
            z: point.position.z / YARDS_TO_METERS,
            velocity_fps: vel_fps,
            energy_ftlbs,
        }
    }
}

/// Trajectory result
#[pyclass(name = "TrajectoryResult")]
pub struct PyTrajectoryResult {
    #[pyo3(get)]
    pub max_range_yards: f64,
    #[pyo3(get)]
    pub max_height_yards: f64,
    #[pyo3(get)]
    pub time_of_flight: f64,
    #[pyo3(get)]
    pub impact_velocity_fps: f64,
    #[pyo3(get)]
    pub impact_energy_ftlbs: f64,
    #[pyo3(get)]
    pub points: Vec<Py<PyTrajectoryPoint>>,
    /// Regular-interval trajectory samples (MBA-1295), populated only when
    /// `enable_trajectory_sampling` is set on the inputs; `None` otherwise. Each row is a
    /// dict: `{distance_m, drop_m, wind_drift_m, velocity_mps, energy_j, time_s, flags}`
    /// (SI units, matching the engine's `TrajectorySample`; `flags` is a list of strings
    /// such as `"apex"` / `"mach_transition"` / `"zero_crossing"`).
    #[pyo3(get)]
    pub sampled_points: Option<Vec<Py<PyDict>>>,
    /// Mach number when the projectile enters the transonic regime, if it does (MBA-1295
    /// passthrough of `TrajectoryResult::transonic_mach`).
    #[pyo3(get)]
    pub transonic_mach: Option<f64>,
}

impl PyTrajectoryResult {
    fn from_rust(result: RustTrajectoryResult, bullet_mass_kg: f64, py: Python) -> PyResult<Self> {
        let points: PyResult<Vec<Py<PyTrajectoryPoint>>> = result.points.iter()
            .map(|pt| {
                let py_point = PyTrajectoryPoint::from_rust(pt, bullet_mass_kg);
                Py::new(py, py_point)
            })
            .collect();

        let sampled_points = match &result.sampled_points {
            Some(samples) => {
                let mut rows = Vec::with_capacity(samples.len());
                for s in samples {
                    let row = PyDict::new(py);
                    row.set_item("distance_m", s.distance_m)?;
                    row.set_item("drop_m", s.drop_m)?;
                    row.set_item("wind_drift_m", s.wind_drift_m)?;
                    row.set_item("velocity_mps", s.velocity_mps)?;
                    row.set_item("energy_j", s.energy_j)?;
                    row.set_item("time_s", s.time_s)?;
                    let flags: Vec<String> = s.flags.iter().map(|f| f.to_string()).collect();
                    row.set_item("flags", flags)?;
                    rows.push(row.unbind());
                }
                Some(rows)
            }
            None => None,
        };

        Ok(PyTrajectoryResult {
            max_range_yards: result.max_range / YARDS_TO_METERS,
            max_height_yards: result.max_height / YARDS_TO_METERS,
            time_of_flight: result.time_of_flight,
            impact_velocity_fps: result.impact_velocity / FPS_TO_MPS,
            impact_energy_ftlbs: result.impact_energy / 1.35582,
            points: points?,
            sampled_points,
            transonic_mach: result.transonic_mach,
        })
    }
}

/// Ballistic calculation inputs.
///
/// MBA-1295: internally wraps the full engine `BallisticInputs` (in SI/radians) as the
/// single source of truth, so every field the shared rich-dict parser (`inputs.rs`) can
/// populate — cant angle, trajectory sampling, wind shear, BC segments, custom drag
/// tables, Coriolis, ... — survives round-trips through `TrajectorySolver`, not just the
/// original 10-field imperial subset. The legacy imperial getters/setters below
/// (`bc`, `bullet_weight_grains`, `muzzle_velocity_fps`, ...) are computed properties over
/// `inner`, so `inputs.bc = 0.6` and `BallisticInputs.from_dict({"bc": 0.6, ...})` write
/// the same field and can never desync.
#[pyclass(name = "BallisticInputs")]
#[derive(Clone)]
pub struct PyBallisticInputs {
    pub(crate) inner: RustBallisticInputs,
}

#[pymethods]
impl PyBallisticInputs {
    #[new]
    #[pyo3(signature = (
        bc=0.5,
        bullet_weight_grains=168.0,
        muzzle_velocity_fps=2650.0,
        bullet_diameter_inches=0.308,
        bullet_length_inches=1.2,
        sight_height_inches=1.5,
        zero_distance_yards=100.0,
        shooting_angle_degrees=0.0,
        twist_rate_inches=11.25,
        is_right_twist=true
    ))]
    fn new(
        bc: f64,
        bullet_weight_grains: f64,
        muzzle_velocity_fps: f64,
        bullet_diameter_inches: f64,
        bullet_length_inches: f64,
        sight_height_inches: f64,
        zero_distance_yards: f64,
        shooting_angle_degrees: f64,
        twist_rate_inches: f64,
        is_right_twist: bool,
    ) -> Self {
        let mut inner = RustBallisticInputs::default();
        inner.bc_value = bc;
        inner.bc_type = DragModel::G7; // Default to G7
        inner.bc_type_str = Some("G7".to_string());
        inner.bullet_mass = bullet_weight_grains * GRAINS_TO_KG;
        inner.muzzle_velocity = muzzle_velocity_fps * FPS_TO_MPS;
        inner.bullet_diameter = bullet_diameter_inches * INCHES_TO_METERS;
        inner.bullet_length = bullet_length_inches * INCHES_TO_METERS;
        inner.sight_height = sight_height_inches * INCHES_TO_METERS;
        inner.target_distance = zero_distance_yards * YARDS_TO_METERS;
        inner.shooting_angle = shooting_angle_degrees * DEGREES_TO_RADIANS;
        inner.twist_rate = twist_rate_inches;
        inner.is_twist_right = is_right_twist;
        inner.caliber_inches = bullet_diameter_inches;
        inner.weight_grains = bullet_weight_grains;
        PyBallisticInputs { inner }
    }

    /// Build a BallisticInputs from a dict. Accepts EVERYTHING the shared rich parser
    /// understands (see `inputs.rs`): the original 11 legacy imperial keys (`bc`,
    /// `bullet_weight_grains`, `muzzle_velocity_fps`, `bullet_diameter_inches`,
    /// `bullet_length_inches`, `sight_height_inches`, `zero_distance_yards`,
    /// `shooting_angle_degrees`, `twist_rate_inches`, `is_right_twist`, `drag_model`)
    /// PLUS the richer ballistics_rust-derived keys (`bc_value`, `bullet_mass`, `altitude`,
    /// `enable_coriolis`, `bc_segments`, `custom_drag_function`, wind shear, cluster BC,
    /// ...) PLUS new (MBA-1295) keys: `cant_angle_degrees`, `enable_trajectory_sampling`,
    /// `sample_interval_m`, `use_rk4`, `use_adaptive_rk45`, `muzzle_height_inches`,
    /// `target_height_inches`, `custom_drag_table` (list of `[mach, cd]` pairs, validated —
    /// raises `ValueError` on a malformed table). Missing keys fall back to sensible
    /// defaults (the original 11-key defaults for the legacy names). `drag_model` accepts a
    /// `DragModel` or a string.
    #[staticmethod]
    fn from_dict(d: &Bound<'_, PyDict>) -> PyResult<Self> {
        let mut inner = crate::inputs::ballistic_inputs_from_dict(d)?;
        crate::inputs::full_to_si(&mut inner);
        Ok(PyBallisticInputs { inner })
    }

    #[getter]
    fn bc(&self) -> f64 {
        self.inner.bc_value
    }
    #[setter]
    fn set_bc(&mut self, v: f64) {
        self.inner.bc_value = v;
    }

    #[getter]
    fn drag_model(&self) -> PyDragModel {
        PyDragModel { inner: self.inner.bc_type }
    }
    #[setter]
    fn set_drag_model(&mut self, v: PyDragModel) {
        self.inner.bc_type = v.inner;
        self.inner.bc_type_str = Some(format!("{:?}", v.inner));
    }

    #[getter]
    fn bullet_weight_grains(&self) -> f64 {
        self.inner.bullet_mass / GRAINS_TO_KG
    }
    #[setter]
    fn set_bullet_weight_grains(&mut self, v: f64) {
        self.inner.bullet_mass = v * GRAINS_TO_KG;
        self.inner.weight_grains = v;
    }

    #[getter]
    fn muzzle_velocity_fps(&self) -> f64 {
        self.inner.muzzle_velocity / FPS_TO_MPS
    }
    #[setter]
    fn set_muzzle_velocity_fps(&mut self, v: f64) {
        self.inner.muzzle_velocity = v * FPS_TO_MPS;
    }

    #[getter]
    fn bullet_diameter_inches(&self) -> f64 {
        self.inner.bullet_diameter / INCHES_TO_METERS
    }
    #[setter]
    fn set_bullet_diameter_inches(&mut self, v: f64) {
        self.inner.bullet_diameter = v * INCHES_TO_METERS;
        self.inner.caliber_inches = v;
    }

    #[getter]
    fn bullet_length_inches(&self) -> f64 {
        self.inner.bullet_length / INCHES_TO_METERS
    }
    #[setter]
    fn set_bullet_length_inches(&mut self, v: f64) {
        self.inner.bullet_length = v * INCHES_TO_METERS;
    }

    #[getter]
    fn sight_height_inches(&self) -> f64 {
        self.inner.sight_height / INCHES_TO_METERS
    }
    #[setter]
    fn set_sight_height_inches(&mut self, v: f64) {
        self.inner.sight_height = v * INCHES_TO_METERS;
    }

    #[getter]
    fn zero_distance_yards(&self) -> f64 {
        self.inner.target_distance / YARDS_TO_METERS
    }
    #[setter]
    fn set_zero_distance_yards(&mut self, v: f64) {
        self.inner.target_distance = v * YARDS_TO_METERS;
    }

    #[getter]
    fn shooting_angle_degrees(&self) -> f64 {
        self.inner.shooting_angle / DEGREES_TO_RADIANS
    }
    #[setter]
    fn set_shooting_angle_degrees(&mut self, v: f64) {
        self.inner.shooting_angle = v * DEGREES_TO_RADIANS;
    }

    #[getter]
    fn twist_rate_inches(&self) -> f64 {
        self.inner.twist_rate
    }
    #[setter]
    fn set_twist_rate_inches(&mut self, v: f64) {
        self.inner.twist_rate = v;
    }

    #[getter]
    fn is_right_twist(&self) -> bool {
        self.inner.is_twist_right
    }
    #[setter]
    fn set_is_right_twist(&mut self, v: bool) {
        self.inner.is_twist_right = v;
    }

    /// Rifle cant angle about the line of sight, degrees; positive = clockwise from the
    /// shooter's view (MBA-1295, wraps `BallisticInputs::cant_angle`; see its doc for the
    /// physical convention). 0.0 = level rifle (default).
    #[getter]
    fn cant_angle_degrees(&self) -> f64 {
        self.inner.cant_angle / DEGREES_TO_RADIANS
    }
    #[setter]
    fn set_cant_angle_degrees(&mut self, v: f64) {
        self.inner.cant_angle = v * DEGREES_TO_RADIANS;
    }

    /// Whether `TrajectoryResult.sampled_points` is populated (MBA-1295).
    #[getter]
    fn enable_trajectory_sampling(&self) -> bool {
        self.inner.enable_trajectory_sampling
    }
    #[setter]
    fn set_enable_trajectory_sampling(&mut self, v: bool) {
        self.inner.enable_trajectory_sampling = v;
    }

    /// Downrange spacing (meters) between `TrajectoryResult.sampled_points` rows (MBA-1295).
    #[getter]
    fn sample_interval_m(&self) -> f64 {
        self.inner.sample_interval
    }
    #[setter]
    fn set_sample_interval_m(&mut self, v: f64) {
        self.inner.sample_interval = v;
    }

    /// Use RK4 (vs Euler) integration; default true (MBA-1295).
    #[getter]
    fn use_rk4(&self) -> bool {
        self.inner.use_rk4
    }
    #[setter]
    fn set_use_rk4(&mut self, v: bool) {
        self.inner.use_rk4 = v;
    }

    /// Use adaptive RK45 (requires `use_rk4=true`); default false on this binding's
    /// constructor path (MBA-1295; matches the fast path's historical default — NOTE this
    /// differs from the bare engine `BallisticInputs::default()`, which defaults both to
    /// true).
    #[getter]
    fn use_adaptive_rk45(&self) -> bool {
        self.inner.use_adaptive_rk45
    }
    #[setter]
    fn set_use_adaptive_rk45(&mut self, v: bool) {
        self.inner.use_adaptive_rk45 = v;
    }

    /// Muzzle height above ground, inches (MBA-1295, wraps `BallisticInputs::muzzle_height`).
    #[getter]
    fn muzzle_height_inches(&self) -> f64 {
        self.inner.muzzle_height / INCHES_TO_METERS
    }
    #[setter]
    fn set_muzzle_height_inches(&mut self, v: f64) {
        self.inner.muzzle_height = v * INCHES_TO_METERS;
    }

    /// Target height above ground for zeroing, inches (MBA-1295, wraps
    /// `BallisticInputs::target_height`).
    #[getter]
    fn target_height_inches(&self) -> f64 {
        self.inner.target_height / INCHES_TO_METERS
    }
    #[setter]
    fn set_target_height_inches(&mut self, v: f64) {
        self.inner.target_height = v * INCHES_TO_METERS;
    }

    fn __repr__(&self) -> String {
        format!(
            "BallisticInputs(bc={}, weight={}gr, mv={}fps, diameter={}\", zero={}yd)",
            self.bc(),
            self.bullet_weight_grains(),
            self.muzzle_velocity_fps(),
            self.bullet_diameter_inches(),
            self.zero_distance_yards()
        )
    }
}

impl PyBallisticInputs {
    fn to_rust(&self) -> RustBallisticInputs {
        self.inner.clone()
    }
}

/// Trajectory solver
#[pyclass(name = "TrajectorySolver")]
pub struct PyTrajectorySolver {
    solver: RustTrajectorySolver,
    bullet_mass_kg: f64,
}

#[pymethods]
impl PyTrajectorySolver {
    #[new]
    #[pyo3(signature = (inputs, wind=None, atmosphere=None))]
    fn new(
        inputs: PyBallisticInputs,
        wind: Option<PyWindConditions>,
        atmosphere: Option<PyAtmosphericConditions>,
    ) -> Self {
        let rust_inputs = inputs.to_rust();
        let bullet_mass_kg = rust_inputs.bullet_mass;

        let rust_wind = wind.unwrap_or_else(|| PyWindConditions::new(0.0, 0.0, 0.0)).to_rust();
        let rust_atmosphere = atmosphere.unwrap_or_else(|| PyAtmosphericConditions::new(59.0, 29.92, 50.0, 0.0)).to_rust();

        let solver = RustTrajectorySolver::new(rust_inputs, rust_wind, rust_atmosphere);

        PyTrajectorySolver {
            solver,
            bullet_mass_kg,
        }
    }

    fn solve(&self, py: Python) -> PyResult<PyTrajectoryResult> {
        let result = self.solver.solve()
            .map_err(|e| PyValueError::new_err(format!("Trajectory calculation failed: {}", e)))?;

        PyTrajectoryResult::from_rust(result, self.bullet_mass_kg, py)
    }

    /// Maximum solve distance, meters (MBA-1295; engine default 1000.0).
    fn set_max_range(&mut self, range_m: f64) {
        self.solver.set_max_range(range_m);
    }

    /// Fixed integration time step, seconds (MBA-1295; engine default 0.001; only used by
    /// Euler/fixed-step RK4, not adaptive RK45).
    fn set_time_step(&mut self, step_s: f64) {
        self.solver.set_time_step(step_s);
    }

    /// Downrange-segmented wind (MBA-1295). Each element is a 3-tuple
    /// `(speed_kmh, angle_deg, until_m)` or a 4-tuple adding `vertical_mps`
    /// (m/s, positive = updraft); a bare 3-tuple defaults `vertical_mps` to 0.0. The wind
    /// for a given downrange distance is the first segment whose `until_m` exceeds it (a
    /// step function); wind is zero beyond the last segment. An empty list clears segmented
    /// wind (reverts to the solver's scalar `WindConditions`). Angle convention matches
    /// `WindConditions` (0 = headwind, 90 = from the right).
    fn set_wind_segments(&mut self, segments: &Bound<'_, pyo3::types::PyList>) -> PyResult<()> {
        use ::ballistics_engine::wind::WindSegment;
        let mut out = Vec::with_capacity(segments.len());
        for item in segments.iter() {
            let tup: Vec<f64> = item.extract()?;
            let seg = match tup.as_slice() {
                [speed_kmh, angle_deg, until_m] => WindSegment {
                    speed_kmh: *speed_kmh,
                    angle_deg: *angle_deg,
                    until_m: *until_m,
                    vertical_mps: 0.0,
                },
                [speed_kmh, angle_deg, until_m, vertical_mps] => WindSegment {
                    speed_kmh: *speed_kmh,
                    angle_deg: *angle_deg,
                    until_m: *until_m,
                    vertical_mps: *vertical_mps,
                },
                _ => {
                    return Err(PyValueError::new_err(
                        "wind segment tuples must have 3 or 4 elements: \
                         (speed_kmh, angle_deg, until_m[, vertical_mps])",
                    ))
                }
            };
            out.push(seg);
        }
        self.solver.set_wind_segments(out);
        Ok(())
    }

    /// Downrange-segmented atmosphere (MBA-1295). Each element is a 4-tuple
    /// `(temp_c, pressure_hpa, humidity_percent, until_distance_m)`, station-referenced at
    /// the shooter's base altitude; the zone for a given downrange distance is the first
    /// whose `until_distance_m` exceeds it (the last zone holds beyond its threshold). An
    /// empty list clears segmented atmosphere (reverts to the solver's single-station
    /// `AtmosphericConditions`). Cheap to expose: the engine's `AtmoSegment` is already a
    /// plain `(f64, f64, f64, f64)` tuple, so no conversion is needed.
    fn set_atmo_segments(&mut self, segments: Vec<(f64, f64, f64, f64)>) {
        self.solver.set_atmo_segments(segments);
    }
}

/// Solve the barrel elevation (radians) that zeroes at `zero_distance_yards`, wrapping
/// `calculate_zero_angle_with_conditions` via the SAME shared rich dict parser as
/// `PyBallisticInputs::from_dict` (MBA-1295). Lets a caller zero and then solve from ONE
/// consistent inputs dict — e.g. `BallisticInputs.from_dict(d)` for the solver plus
/// `auto_zero_inputs(d, ...)` for the muzzle angle to set on it — rather than juggling the
/// older `calculate_zero_angle` pyfunction's separate flat imperial parameters.
///
/// `wind`/`atmo` mirror `PyTrajectorySolver`'s constructor: when supplied they are used
/// as-is; when omitted, wind/atmosphere are derived from the inputs dict's own
/// `wind_speed`/`wind_angle` and `temperature`/`pressure`/`humidity`/`altitude` keys (all
/// defaulting to none/standard atmosphere if absent, matching the dict parser's own
/// defaults). Raises `RuntimeError` on non-convergence.
#[pyfunction]
#[pyo3(signature = (inputs, zero_distance_yards, target_height_inches=0.0, wind=None, atmo=None))]
pub fn auto_zero_inputs(
    inputs: &Bound<'_, PyDict>,
    zero_distance_yards: f64,
    target_height_inches: f64,
    wind: Option<PyWindConditions>,
    atmo: Option<PyAtmosphericConditions>,
) -> PyResult<f64> {
    let mut bi = crate::inputs::ballistic_inputs_from_dict(inputs)?;
    crate::inputs::full_to_si(&mut bi);

    let rust_wind = wind.map(|w| w.to_rust()).unwrap_or(RustWindConditions {
        speed: bi.wind_speed,
        direction: bi.wind_angle,
        vertical_speed: 0.0,
    });
    let rust_atmo = atmo.map(|a| a.to_rust()).unwrap_or(RustAtmosphericConditions {
        temperature: bi.temperature,
        pressure: bi.pressure,
        humidity: bi.humidity_percent(),
        altitude: bi.altitude,
    });

    ::ballistics_engine::calculate_zero_angle_with_conditions(
        bi,
        zero_distance_yards * YARDS_TO_METERS,
        target_height_inches * INCHES_TO_METERS,
        rust_wind,
        rust_atmo,
    )
    .map_err(|e| {
        PyRuntimeError::new_err(format!(
            "Unable to find zero angle for target distance {zero_distance_yards} yards: {e}"
        ))
    })
}

/// Unit conversion utilities
#[pymodule]
fn ballistics_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDragModel>()?;
    m.add_class::<PyBallisticInputs>()?;
    m.add_class::<PyWindConditions>()?;
    m.add_class::<PyAtmosphericConditions>()?;
    m.add_class::<PyTrajectoryPoint>()?;
    m.add_class::<PyTrajectoryResult>()?;
    m.add_class::<PyTrajectorySolver>()?;

    // Raw fixed-step integration kernel (scipy-like {t,y,t_events,success} contract)
    m.add_function(pyo3::wrap_pyfunction!(fast::fast_integrate, m)?)?;
    // Single RK-stage derivatives ([vx,vy,vz,ax,ay,az])
    m.add_function(pyo3::wrap_pyfunction!(fast::derivatives, m)?)?;
    // Zero-angle solve (radians) from a fully-imperial inputs dict
    m.add_function(pyo3::wrap_pyfunction!(fast::calculate_zero_angle, m)?)?;
    // Zero-angle solve (radians) via the shared rich dict parser + TrajectorySolver path
    m.add_function(pyo3::wrap_pyfunction!(auto_zero_inputs, m)?)?;

    // Scalar query helpers (drag / atmosphere)
    m.add_function(pyo3::wrap_pyfunction!(helpers::get_drag_coefficient, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(helpers::get_drag_coefficient_transonic, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(helpers::interpolated_bc, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(helpers::calculate_atmosphere, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(helpers::calculate_air_density_cipm, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(helpers::get_local_atmosphere, m)?)?;

    // Parallel Monte Carlo (nested statistics/dispersion/metadata dict)
    m.add_function(pyo3::wrap_pyfunction!(montecarlo::monte_carlo_parallel, m)?)?;
    // Engine's own Monte Carlo driver (wind + wind-direction std dev)
    m.add_function(pyo3::wrap_pyfunction!(montecarlo::run_monte_carlo, m)?)?;

    // Stability / spin-drift / transonic scalar helpers
    m.add_function(pyo3::wrap_pyfunction!(effects::transonic_correction, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(effects::get_projectile_shape, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(effects::compute_stability_advanced, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(effects::compute_spin_drift_advanced, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(effects::compute_spin_drift, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(effects::compute_stability_coefficient, m)?)?;

    // Version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
