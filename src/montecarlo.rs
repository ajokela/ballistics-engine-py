// Parallel Monte Carlo, mirroring ballistics_rust.monte_carlo_parallel. Takes a
// base inputs dict + a [n_samples x n_params] array of pre-generated ABSOLUTE
// parameter values (Python owns the sampling), evaluates trajectories in parallel
// over the engine kernel, and assembles the app's nested
// {statistics, dispersion, metadata} result dict.

use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use ::ballistics_engine::monte_carlo::{
    calculate_cep, calculate_confidence_ellipse, percentile, sample_points_for_visualization,
    solve_trajectory_for_monte_carlo, TrajectoryOutput,
};
use ::ballistics_engine::BallisticInputs as RustBallisticInputs;

const M_TO_INCHES: f64 = 39.3701;
const FIELDS: [&str; 6] = [
    "drop_m",
    "wind_drift_m",
    "time_of_flight_s",
    "final_vel_fps",
    "energy_ft_lbs",
    "mach",
];

fn field_value(r: &TrajectoryOutput, f: &str) -> f64 {
    match f {
        "drop_m" => r.drop,
        "wind_drift_m" => r.wind_drift,
        "time_of_flight_s" => r.time,
        "final_vel_fps" => r.velocity * 3.28084,
        "energy_ft_lbs" => r.energy * 0.737562,
        "mach" => r.mach,
        _ => 0.0,
    }
}

/// Set an absolute parameter value on the inputs (matches ballistics_rust).
fn apply_parameter(inp: &mut RustBallisticInputs, name: &str, value: f64) {
    match name {
        "bc_value" => inp.bc_value = value,
        "bullet_mass" => inp.bullet_mass = value,
        "muzzle_velocity" => inp.muzzle_velocity = value,
        "wind_speed" => inp.wind_speed = value,
        "wind_angle" => inp.wind_angle = value,
        "target_distance" => inp.target_distance = value,
        "muzzle_angle" => inp.muzzle_angle = value,
        "altitude" => inp.altitude = value,
        "temperature" => inp.temperature = value,
        "pressure" => inp.pressure = value,
        "humidity" => inp.humidity = value,
        "latitude" => inp.latitude = Some(value),
        "twist_rate" => inp.twist_rate = value,
        "bullet_length" => inp.bullet_length = value,
        "bullet_diameter" => inp.bullet_diameter = value,
        _ => {} // ignore unknown
    }
}

/// Convert the imperial MC dict to SI for the engine kernel (mirrors
/// ballistics_rust::monte_carlo_inputs_to_si) PLUS the engine>=0.17.0 humidity
/// fix: percent -> 0-1 fraction (the kernel multiplies humidity by 100).
fn mc_inputs_to_si(i: &mut RustBallisticInputs) {
    i.target_distance *= 0.9144; // yards -> m
    i.muzzle_velocity *= 0.3048; // fps -> m/s
    i.altitude *= 0.3048; // feet -> m
    i.sight_height *= 0.0254; // inches -> m
    i.wind_speed *= 0.2777778; // km/h -> m/s
    i.wind_angle = i.wind_angle.to_radians(); // degrees -> radians
    i.muzzle_angle = i.muzzle_angle.to_radians(); // degrees -> radians
    i.humidity /= 100.0; // percent -> 0-1 fraction (engine >=0.17.0 multiplies by 100)
    crate::fast::geometry_mass_to_si(i); // mass/diameter/length + caliber/weight mirrors
}

#[pyfunction]
#[pyo3(signature = (base_inputs, param_samples, param_names, num_threads=None, include_dispersion=true, max_viz_points=500))]
#[allow(clippy::too_many_arguments)]
pub fn monte_carlo_parallel<'py>(
    py: Python<'py>,
    base_inputs: &Bound<'py, PyDict>,
    param_samples: PyReadonlyArray2<'py, f64>,
    param_names: Vec<String>,
    num_threads: Option<usize>,
    include_dispersion: bool,
    max_viz_points: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let _ = num_threads; // default rayon pool; thread count does not affect results

    let base = crate::fast::ballistic_inputs_from_dict(base_inputs)?;
    let samples = param_samples.as_array();
    let n_samples = samples.shape()[0];
    let n_params = if samples.ndim() >= 2 { samples.shape()[1] } else { 0 };

    // Materialize sample rows (owned, Send) so the parallel closure is GIL-free.
    let rows: Vec<Vec<f64>> = (0..n_samples)
        .map(|i| (0..n_params).map(|j| samples[[i, j]]).collect())
        .collect();

    let results: Vec<Option<TrajectoryOutput>> = rows
        .par_iter()
        .map(|row| {
            let mut ri = base.clone();
            for (j, name) in param_names.iter().enumerate() {
                if j < row.len() {
                    apply_parameter(&mut ri, name, row[j]);
                }
            }
            mc_inputs_to_si(&mut ri);
            solve_trajectory_for_monte_carlo(&ri).ok()
        })
        .collect();

    let valid: Vec<&TrajectoryOutput> = results.iter().filter_map(|r| r.as_ref()).collect();
    let valid_runs = valid.len();
    let failed_runs = results.len() - valid_runs;

    let out = PyDict::new(py);

    // ---- statistics ----
    let stats = PyDict::new(py);
    if valid_runs == 1 {
        for f in FIELDS {
            let v = field_value(valid[0], f);
            let fd = PyDict::new(py);
            fd.set_item("mean", v)?;
            fd.set_item("std", 0.0)?;
            fd.set_item("min", v)?;
            fd.set_item("max", v)?;
            let pd = PyDict::new(py);
            pd.set_item("p50", v)?;
            fd.set_item("percentiles", pd)?;
            stats.set_item(f, fd)?;
        }
    } else if valid_runs >= 2 {
        for f in FIELDS {
            let mut vals: Vec<f64> = valid.iter().map(|r| field_value(r, f)).collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = vals.len() as f64;
            let mean = vals.iter().sum::<f64>() / n;
            let variance = vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
            let std = variance.sqrt();
            let fd = PyDict::new(py);
            fd.set_item("mean", mean)?;
            fd.set_item("std", std)?;
            fd.set_item("min", vals[0])?;
            fd.set_item("max", vals[vals.len() - 1])?;
            let pd = PyDict::new(py);
            for p in [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95] {
                pd.set_item(format!("p{}", (p * 100.0) as i32), percentile(&vals, p))?;
            }
            fd.set_item("percentiles", pd)?;
            stats.set_item(f, fd)?;
        }
    }
    out.set_item("statistics", stats)?;

    // ---- dispersion ----
    if include_dispersion && valid_runs >= 2 {
        let wd: Vec<f64> = valid.iter().map(|r| r.wind_drift).collect();
        let dr: Vec<f64> = valid.iter().map(|r| r.drop).collect();
        let cep = calculate_cep(&wd, &dr);
        let (cx, cy, w, h, rot) = calculate_confidence_ellipse(&wd, &dr);
        let pts = sample_points_for_visualization(&wd, &dr, max_viz_points);

        let disp = PyDict::new(py);
        disp.set_item("cep_m", cep)?;
        disp.set_item("cep_inches", cep * M_TO_INCHES)?;
        let el = PyDict::new(py);
        el.set_item("center_x_m", cx)?;
        el.set_item("center_y_m", cy)?;
        el.set_item("center_x_inches", cx * M_TO_INCHES)?;
        el.set_item("center_y_inches", cy * M_TO_INCHES)?;
        el.set_item("semi_major_m", w)?;
        el.set_item("semi_minor_m", h)?;
        el.set_item("semi_major_inches", w * M_TO_INCHES)?;
        el.set_item("semi_minor_inches", h * M_TO_INCHES)?;
        el.set_item("rotation_deg", rot)?;
        disp.set_item("ellipse", el)?;
        let pts2: Vec<(f64, f64)> = pts.iter().map(|(x, y)| (*x, *y)).collect();
        disp.set_item("sample_points", pts2)?;
        out.set_item("dispersion", disp)?;
    }

    // ---- metadata ----
    let meta = PyDict::new(py);
    meta.set_item("valid_runs", valid_runs)?;
    meta.set_item("failed_runs", failed_runs)?;
    meta.set_item("total_runs", n_samples)?;
    meta.set_item(
        "success_rate",
        if n_samples > 0 {
            valid_runs as f64 / n_samples as f64
        } else {
            0.0
        },
    )?;
    out.set_item("metadata", meta)?;

    Ok(out)
}
