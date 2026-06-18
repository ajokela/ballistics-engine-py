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

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
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
    inner: DragModel,
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
}

#[pymethods]
impl PyWindConditions {
    #[new]
    #[pyo3(signature = (speed_mph=0.0, direction_degrees=0.0))]
    fn new(speed_mph: f64, direction_degrees: f64) -> Self {
        PyWindConditions {
            speed_mph,
            direction_degrees,
        }
    }
}

impl PyWindConditions {
    fn to_rust(&self) -> RustWindConditions {
        RustWindConditions {
            speed: self.speed_mph * MPH_TO_MPS,
            direction: self.direction_degrees * DEGREES_TO_RADIANS,
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
}

impl PyTrajectoryResult {
    fn from_rust(result: RustTrajectoryResult, bullet_mass_kg: f64, py: Python) -> PyResult<Self> {
        let points: PyResult<Vec<Py<PyTrajectoryPoint>>> = result.points.iter()
            .map(|pt| {
                let py_point = PyTrajectoryPoint::from_rust(pt, bullet_mass_kg);
                Py::new(py, py_point)
            })
            .collect();

        Ok(PyTrajectoryResult {
            max_range_yards: result.max_range / YARDS_TO_METERS,
            max_height_yards: result.max_height / YARDS_TO_METERS,
            time_of_flight: result.time_of_flight,
            impact_velocity_fps: result.impact_velocity / FPS_TO_MPS,
            impact_energy_ftlbs: result.impact_energy / 1.35582,
            points: points?,
        })
    }
}

/// Ballistic calculation inputs
#[pyclass(name = "BallisticInputs")]
#[derive(Clone)]
pub struct PyBallisticInputs {
    // Core ballistics (imperial units for Python API)
    #[pyo3(get, set)]
    pub bc: f64,
    #[pyo3(get, set)]
    pub drag_model: PyDragModel,
    #[pyo3(get, set)]
    pub bullet_weight_grains: f64,
    #[pyo3(get, set)]
    pub muzzle_velocity_fps: f64,
    #[pyo3(get, set)]
    pub bullet_diameter_inches: f64,
    #[pyo3(get, set)]
    pub bullet_length_inches: f64,

    // Targeting
    #[pyo3(get, set)]
    pub sight_height_inches: f64,
    #[pyo3(get, set)]
    pub zero_distance_yards: f64,
    #[pyo3(get, set)]
    pub shooting_angle_degrees: f64,

    // Barrel
    #[pyo3(get, set)]
    pub twist_rate_inches: f64,
    #[pyo3(get, set)]
    pub is_right_twist: bool,
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
        PyBallisticInputs {
            bc,
            drag_model: PyDragModel::g7(),  // Default to G7
            bullet_weight_grains,
            muzzle_velocity_fps,
            bullet_diameter_inches,
            bullet_length_inches,
            sight_height_inches,
            zero_distance_yards,
            shooting_angle_degrees,
            twist_rate_inches,
            is_right_twist,
        }
    }

    /// Build a BallisticInputs from a dict of the binding's imperial field names
    /// (bc, bullet_weight_grains, muzzle_velocity_fps, ...); missing keys fall back
    /// to the constructor defaults. `drag_model` accepts a DragModel or a string.
    #[staticmethod]
    fn from_dict(d: &Bound<'_, pyo3::types::PyDict>) -> PyResult<Self> {
        let f = |k: &str, default: f64| -> PyResult<f64> {
            Ok(match d.get_item(k)? {
                Some(v) if !v.is_none() => v.extract()?,
                _ => default,
            })
        };
        let drag_model = match d.get_item("drag_model")? {
            Some(v) if !v.is_none() => {
                if let Ok(dm) = v.extract::<PyDragModel>() {
                    dm
                } else {
                    let s: String = v.extract()?;
                    if s.contains("G1") {
                        PyDragModel::g1()
                    } else if s.contains("G8") {
                        PyDragModel::g8()
                    } else {
                        PyDragModel::g7()
                    }
                }
            }
            _ => PyDragModel::g7(),
        };
        let is_right_twist = match d.get_item("is_right_twist")? {
            Some(v) if !v.is_none() => v.extract()?,
            _ => true,
        };
        Ok(PyBallisticInputs {
            bc: f("bc", 0.5)?,
            drag_model,
            bullet_weight_grains: f("bullet_weight_grains", 168.0)?,
            muzzle_velocity_fps: f("muzzle_velocity_fps", 2650.0)?,
            bullet_diameter_inches: f("bullet_diameter_inches", 0.308)?,
            bullet_length_inches: f("bullet_length_inches", 1.2)?,
            sight_height_inches: f("sight_height_inches", 1.5)?,
            zero_distance_yards: f("zero_distance_yards", 100.0)?,
            shooting_angle_degrees: f("shooting_angle_degrees", 0.0)?,
            twist_rate_inches: f("twist_rate_inches", 11.25)?,
            is_right_twist,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "BallisticInputs(bc={}, weight={}gr, mv={}fps, diameter={}\", zero={}yd)",
            self.bc,
            self.bullet_weight_grains,
            self.muzzle_velocity_fps,
            self.bullet_diameter_inches,
            self.zero_distance_yards
        )
    }
}

impl PyBallisticInputs {
    fn to_rust(&self) -> RustBallisticInputs {
        let mut inputs = RustBallisticInputs::default();

        // Convert imperial to metric
        inputs.bc_value = self.bc;
        inputs.bc_type = self.drag_model.inner.clone();
        inputs.bullet_mass = self.bullet_weight_grains * GRAINS_TO_KG;
        inputs.muzzle_velocity = self.muzzle_velocity_fps * FPS_TO_MPS;
        inputs.bullet_diameter = self.bullet_diameter_inches * INCHES_TO_METERS;
        inputs.bullet_length = self.bullet_length_inches * INCHES_TO_METERS;
        inputs.sight_height = self.sight_height_inches * INCHES_TO_METERS;
        inputs.target_distance = self.zero_distance_yards * YARDS_TO_METERS;
        inputs.shooting_angle = self.shooting_angle_degrees * DEGREES_TO_RADIANS;
        inputs.twist_rate = self.twist_rate_inches;
        inputs.is_twist_right = self.is_right_twist;
        inputs.caliber_inches = self.bullet_diameter_inches;
        inputs.weight_grains = self.bullet_weight_grains;

        inputs
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

        let rust_wind = wind.unwrap_or_else(|| PyWindConditions::new(0.0, 0.0)).to_rust();
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

    // Scalar query helpers (drag / atmosphere)
    m.add_function(pyo3::wrap_pyfunction!(helpers::get_drag_coefficient, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(helpers::get_drag_coefficient_transonic, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(helpers::interpolated_bc, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(helpers::calculate_atmosphere, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(helpers::calculate_air_density_cipm, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(helpers::get_local_atmosphere, m)?)?;

    // Parallel Monte Carlo (nested statistics/dispersion/metadata dict)
    m.add_function(pyo3::wrap_pyfunction!(montecarlo::monte_carlo_parallel, m)?)?;

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
