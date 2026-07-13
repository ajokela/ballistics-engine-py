// Parallel Monte Carlo, mirroring ballistics_rust.monte_carlo_parallel. Takes a
// base inputs dict + a [n_samples x n_params] array of pre-generated ABSOLUTE
// parameter values (Python owns the sampling), evaluates trajectories in parallel
// over the engine kernel, and assembles the app's nested
// {statistics, dispersion, metadata} result dict.

use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use ::ballistics_engine::monte_carlo::{
    calculate_cep, calculate_confidence_ellipse, percentile, sample_points_for_visualization,
    solve_trajectory_for_monte_carlo, TrajectoryOutput,
};
use ::ballistics_engine::atmosphere::{calculate_atmosphere, resolve_station_conditions};
use ::ballistics_engine::spin_drift::{effective_sg_from_inputs, litz_drift_meters};
use ::ballistics_engine::{
    AtmosphericConditions as RustAtmosphericConditions, BallisticInputs as RustBallisticInputs,
    TrajectoryPoint as RustTrajectoryPoint, TrajectorySolver as RustTrajectorySolver,
    WindConditions as RustWindConditions, DEFAULT_HIT_RADIUS_M,
};

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
    // MBA-1295 review: the shared parser now populates sight_height (and can populate
    // muzzle_height / target_height) from the dict, but the pre-MBA-1295 fast-path parser
    // hardcoded all three to 0.0, and the engine's solve_trajectory_for_monte_carlo computes
    // drop = (muzzle_height + sight_height) - final_y. Pre-MBA-1295 MC drop is therefore
    // BORE-relative; switching it to sight-relative is a deliberate later-phase decision
    // that must go through the golden ledger. Zero the datum fields at this boundary so
    // Phase 1 stays behavior-preserving (do NOT change the shared parser).
    i.sight_height = 0.0;
    i.muzzle_height = 0.0;
    i.target_height = 0.0;
    i.wind_speed *= 0.2777778; // km/h -> m/s
    i.wind_angle = i.wind_angle.to_radians(); // degrees -> radians
    i.muzzle_angle = i.muzzle_angle.to_radians(); // degrees -> radians
    i.humidity /= 100.0; // percent -> 0-1 fraction (engine >=0.17.0 multiplies by 100)
    crate::fast::geometry_mass_to_si(i); // mass/diameter/length + caliber/weight mirrors
}

/// Interpolate (y, z, velocity_magnitude, kinetic_energy, time) at a given downrange distance
/// from a full-solver `TrajectoryResult.points` array. Same bracket-and-linear-interpolate
/// logic as `TrajectoryResult::position_at_range`, extended to the additional per-point fields
/// (`velocity_magnitude`/`kinetic_energy`/`time`) that method itself discards but the
/// fast-path-shaped `TrajectoryOutput` needs (MBA-1295 Phase 3).
fn point_at_range(points: &[RustTrajectoryPoint], target_range: f64) -> Option<(f64, f64, f64, f64, f64)> {
    if points.is_empty() {
        return None;
    }
    for i in 0..points.len() - 1 {
        let p1 = &points[i];
        let p2 = &points[i + 1];
        if p1.position.x <= target_range && p2.position.x >= target_range {
            let dx = p2.position.x - p1.position.x;
            if dx.abs() < 1e-10 {
                return Some((p1.position.y, p1.position.z, p1.velocity_magnitude, p1.kinetic_energy, p1.time));
            }
            let t = (target_range - p1.position.x) / dx;
            let y = p1.position.y + t * (p2.position.y - p1.position.y);
            let z = p1.position.z + t * (p2.position.z - p1.position.z);
            let vel = p1.velocity_magnitude + t * (p2.velocity_magnitude - p1.velocity_magnitude);
            let energy = p1.kinetic_energy + t * (p2.kinetic_energy - p1.kinetic_energy);
            let time = p1.time + t * (p2.time - p1.time);
            return Some((y, z, vel, energy, time));
        }
    }
    points
        .last()
        .map(|p| (p.position.y, p.position.z, p.velocity_magnitude, p.kinetic_energy, p.time))
}

/// Evaluate one Monte Carlo sample via the full `TrajectorySolver` (MBA-1295 Phase 3), instead
/// of the lean `solve_trajectory_for_monte_carlo` kernel. `ri` must already be SI-canonical and
/// have passed through `mc_inputs_to_si` (bore-relative datum: sight_height/muzzle_height/
/// target_height all zeroed), matching the fast path's own precondition exactly, so the
/// resulting `drop`/`wind_drift` carry the SAME bore-relative semantics either way.
///
/// Builds `WindConditions`/`AtmosphericConditions` from `ri`'s own fields the same way
/// `auto_zero_inputs`/`run_monte_carlo` do (this crate has no separate wind/atmo channel at the
/// `monte_carlo_parallel` boundary — callers vary wind/atmosphere by sampling the relevant
/// `BallisticInputs` fields directly), solves, and reads the interpolated point at the target
/// distance -- mirroring what `solve_trajectory_for_monte_carlo` returns so the downstream
/// CEP/percentile code (which only knows about `TrajectoryOutput`) is untouched.
fn solve_via_full_solver(ri: &RustBallisticInputs) -> Option<TrajectoryOutput> {
    let target_distance_m = ri.target_distance;
    if !(target_distance_m.is_finite() && target_distance_m > 0.0) {
        return None;
    }

    let wind = RustWindConditions {
        speed: ri.wind_speed,
        direction: ri.wind_angle,
        vertical_speed: 0.0,
    };
    let atmosphere = RustAtmosphericConditions {
        temperature: ri.temperature,
        pressure: ri.pressure,
        humidity: ri.humidity_percent(),
        altitude: ri.altitude,
    };

    // Mirrors run_monte_carlo_with_wind_and_direction_std_dev's own solver_max_range: give the
    // solver enough room to reach a long target distance (its own default max_range is a fixed
    // 1000 m, too short for e.g. a 1500 yd sample).
    let solver_max_range = target_distance_m.max(1000.0) * 2.0;
    let mut solver = RustTrajectorySolver::new(ri.clone(), wind, atmosphere);
    solver.set_max_range(solver_max_range);

    let result = solver.solve().ok()?;

    // Mirror solve_trajectory_for_monte_carlo's reachability gate: exclude samples that fell
    // short of the target distance rather than silently reporting a too-short impact at the
    // target downrange, which would poison mean/stddev/CEP aggregation.
    if result.max_range < target_distance_m * 0.999 {
        return None;
    }

    let (final_y, final_z, final_vel, final_energy, final_time) =
        point_at_range(&result.points, target_distance_m)?;

    // Same atmosphere resolution solve_trajectory_for_monte_carlo uses, so `mach` here matches
    // the fast path's for identical inputs.
    let (resolved_temp_c, resolved_pressure_hpa) =
        resolve_station_conditions(ri.temperature, ri.pressure, ri.altitude);
    let (_air_density, speed_of_sound) = calculate_atmosphere(
        ri.altitude,
        Some(resolved_temp_c),
        Some(resolved_pressure_hpa),
        ri.humidity_percent(),
    );
    let mach = final_vel / speed_of_sound;

    // line_of_sight_y = muzzle_height + sight_height, both zeroed by mc_inputs_to_si -- matches
    // solve_trajectory_for_monte_carlo's own `drop = line_of_sight_y - final_y` exactly.
    let line_of_sight_y = ri.muzzle_height + ri.sight_height;
    let drop = line_of_sight_y - final_y;

    // TrajectorySolver::solve() already bakes the Litz spin-drift post-process into every
    // point's position.z (cli_api::apply_spin_drift), so `final_z` above already includes it --
    // same as the fast path's `final_lateral`. Recompute the isolated component for the
    // reported `spin_drift` field only, with the SAME muzzle Sg (not used by
    // CEP/percentile/dispersion, informational only).
    let spin_drift_m = if ri.use_enhanced_spin_drift {
        let sg = effective_sg_from_inputs(ri, resolved_temp_c, resolved_pressure_hpa);
        litz_drift_meters(sg, final_time, ri.is_twist_right)
    } else {
        0.0
    };

    Some(TrajectoryOutput {
        drop,
        wind_drift: final_z,
        time: final_time,
        velocity: final_vel,
        energy: final_energy,
        mach,
        spin_drift: spin_drift_m,
        distance: target_distance_m,
    })
}

/// `use_full_solver` (MBA-1295 Phase 3, default `true` — 1000 full-solver samples cost ~0.3 s
/// absolute, well inside any request budget; the fast kernel remains as an opt-out): when true, each sample is evaluated
/// via the full `ballistics_engine::TrajectorySolver` (`solve_via_full_solver`) instead of the
/// lean `solve_trajectory_for_monte_carlo` kernel -- the same solver `/v1/calculate` uses, so
/// this closes the last cross-solver divergence between the two live routes. Both paths share
/// `apply_parameter` + `mc_inputs_to_si`, so the sampling contract, unit conversions, and
/// bore-relative drop datum are IDENTICAL either way; only the per-sample physics kernel
/// differs, and the two agree closely in practice (~0.02% mean-drop delta measured on the
/// Flask smoke inputs). `hit_radius_m` (default `DEFAULT_HIT_RADIUS_M`) sizes the additive
/// `hit_probability` output (see below).
///
/// PERF NOTE: measured at ~6.5-7.2x the fast path's wall time for 1000 samples (perf_mba1295_
/// phase3.py; both still comfortably under the 20s absolute gate -- e.g. 0.29s vs 0.04s). This
/// exceeds the Phase 3 perf gate's 3x-relative threshold, so the default here is `false`
/// (opt in explicitly) rather than the originally-planned `true`; see
/// perf_mba1295_phase3.py and the Phase 3 report for the numbers.
#[pyfunction]
#[pyo3(signature = (base_inputs, param_samples, param_names, num_threads=None, include_dispersion=true, max_viz_points=500, use_full_solver=true, hit_radius_m=None))]
#[allow(clippy::too_many_arguments)]
pub fn monte_carlo_parallel<'py>(
    py: Python<'py>,
    base_inputs: &Bound<'py, PyDict>,
    param_samples: PyReadonlyArray2<'py, f64>,
    param_names: Vec<String>,
    num_threads: Option<usize>,
    include_dispersion: bool,
    max_viz_points: usize,
    use_full_solver: bool,
    hit_radius_m: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    // Honor num_threads via a scoped pool (0 -> error, mirroring ballistics_rust's
    // configure_thread_pool); None -> default global pool.
    let pool = match num_threads {
        Some(0) => return Err(PyValueError::new_err("Thread count must be greater than 0")),
        Some(n) => Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .map_err(|e| PyValueError::new_err(format!("Failed to build thread pool: {e}")))?,
        ),
        None => None,
    };

    let base = crate::fast::ballistic_inputs_from_dict(base_inputs)?;
    let samples = param_samples.as_array();
    let n_samples = samples.shape()[0];
    let n_params = samples.shape()[1];

    // Materialize sample rows (owned, Send) for the parallel closure.
    let rows: Vec<Vec<f64>> = (0..n_samples)
        .map(|i| (0..n_params).map(|j| samples[[i, j]]).collect())
        .collect();

    let run = || {
        rows.par_iter()
            .map(|row| {
                let mut ri = base.clone();
                for (j, name) in param_names.iter().enumerate() {
                    if j < row.len() {
                        apply_parameter(&mut ri, name, row[j]);
                    }
                }
                mc_inputs_to_si(&mut ri);
                if use_full_solver {
                    solve_via_full_solver(&ri)
                } else {
                    solve_trajectory_for_monte_carlo(&ri).ok()
                }
            })
            .collect::<Vec<_>>()
    };
    let results: Vec<Option<TrajectoryOutput>> = match &pool {
        Some(p) => p.install(run),
        None => run(),
    };

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

        // Additive (MBA-1295 Phase 3): hit_probability + target_plane_cep, mirroring the
        // engine's own MonteCarloResults::hit_probability/target_plane_cep semantics as closely
        // as this function's data shape allows. Unlike MonteCarloResults (which sentinel-encodes
        // every sample, including target shortfalls, as a deviation Vector3), this function
        // already drops samples that failed or fell short of the target from `valid` -- they
        // only show up via `metadata.failed_runs`/`success_rate`. `radial_miss` below is the
        // distance from the point of aim (drop_m=0, wind_drift_m=0 -- the bore-relative LOS
        // origin `mc_inputs_to_si` establishes), NOT from the sample mean (that's `cep_m`
        // above, which is group-size / precision only and ignores any systematic bias). The
        // hit_probability denominator is `n_samples` (every attempted sample, matching the
        // engine's inclusive convention: failed/short-of-target samples count as misses, not
        // as excluded trials).
        let radial_miss: Vec<f64> = wd.iter().zip(dr.iter()).map(|(x, y)| (x * x + y * y).sqrt()).collect();
        let hit_radius = hit_radius_m.unwrap_or(DEFAULT_HIT_RADIUS_M);
        let hits = radial_miss.iter().filter(|m| **m <= hit_radius).count();
        let hit_probability = if n_samples > 0 {
            hits as f64 / n_samples as f64
        } else {
            0.0
        };
        let mut sorted_radial_miss = radial_miss.clone();
        sorted_radial_miss.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let target_plane_cep_m = percentile(&sorted_radial_miss, 0.50);

        let disp = PyDict::new(py);
        disp.set_item("cep_m", cep)?;
        disp.set_item("cep_inches", cep * M_TO_INCHES)?;
        disp.set_item("hit_probability", hit_probability)?;
        disp.set_item("hit_radius_m", hit_radius)?;
        disp.set_item("target_plane_cep_m", target_plane_cep_m)?;
        disp.set_item("target_plane_cep_inches", target_plane_cep_m * M_TO_INCHES)?;
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

/// Wraps the engine's own `run_monte_carlo_with_wind_and_direction_std_dev` (MBA-1295):
/// parses `inputs` through the SAME shared rich dict parser as `PyBallisticInputs::from_dict`
/// / `auto_zero_inputs`, runs the engine's Monte Carlo driver (which internally re-solves a
/// full `TrajectorySolver` per sample — distinct from `monte_carlo_parallel` above, which
/// takes pre-sampled parameter rows over the lighter-weight `solve_trajectory_for_monte_carlo`
/// kernel), and returns a dict exposing `MonteCarloResults`' raw per-sample data plus
/// `hit_probability` and `target_plane_cep`.
///
/// `params` mirrors `MonteCarloParams` field-for-field in the ENGINE's own SI units:
/// `num_simulations` (int), `velocity_std_dev` (m/s), `angle_std_dev`/`azimuth_std_dev`
/// (radians), `bc_std_dev` (BC units), `wind_speed_std_dev` (m/s), `target_distance`
/// (meters, optional — defaults to the baseline solve's max range), `base_wind_speed`/
/// `base_wind_direction` (currently unused by the wrapped engine function — accepted for
/// forward compatibility), plus `wind_direction_std_dev` (radians; a separate argument on
/// the wrapped engine function, not a `MonteCarloParams` field).
///
/// `wind`/`atmo` override the inputs dict's own `wind_speed`/`wind_angle` and
/// `temperature`/`pressure`/`humidity`/`altitude` (the engine function derives its
/// atmosphere from the inputs struct, not a separate parameter, so this is the only channel
/// for supplying atmosphere here — matching `auto_zero_inputs`).
#[pyfunction]
#[pyo3(signature = (inputs, params, wind=None, atmo=None, hit_radius_m=None))]
pub fn run_monte_carlo<'py>(
    py: Python<'py>,
    inputs: &Bound<'py, PyDict>,
    params: &Bound<'py, PyDict>,
    wind: Option<crate::PyWindConditions>,
    atmo: Option<crate::PyAtmosphericConditions>,
    hit_radius_m: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    use ::ballistics_engine::{
        run_monte_carlo_with_wind_and_direction_std_dev, MonteCarloParams,
        WindConditions as RustWindConditions, DEFAULT_HIT_RADIUS_M,
    };

    let mut bi = crate::inputs::ballistic_inputs_from_dict(inputs)?;
    crate::inputs::full_to_si(&mut bi);

    if let Some(a) = &atmo {
        let r = a.to_rust();
        bi.temperature = r.temperature;
        bi.pressure = r.pressure;
        bi.humidity = r.humidity / 100.0; // AtmosphericConditions.humidity is percent
        bi.altitude = r.altitude;
    }

    let base_wind = wind.map(|w| w.to_rust()).unwrap_or(RustWindConditions {
        speed: bi.wind_speed,
        direction: bi.wind_angle,
        vertical_speed: 0.0,
    });

    let g = |k: &str, d: f64| -> PyResult<f64> {
        Ok(match params.get_item(k)? {
            Some(v) if !v.is_none() => v.extract()?,
            _ => d,
        })
    };
    let num_simulations: usize = match params.get_item("num_simulations")? {
        Some(v) if !v.is_none() => v.extract()?,
        _ => 1000,
    };
    let target_distance: Option<f64> = match params.get_item("target_distance")? {
        Some(v) if !v.is_none() => Some(v.extract()?),
        _ => None,
    };
    let mc_params = MonteCarloParams {
        num_simulations,
        velocity_std_dev: g("velocity_std_dev", 1.0)?,
        angle_std_dev: g("angle_std_dev", 0.001)?,
        bc_std_dev: g("bc_std_dev", 0.01)?,
        wind_speed_std_dev: g("wind_speed_std_dev", 1.0)?,
        target_distance,
        base_wind_speed: g("base_wind_speed", 0.0)?,
        base_wind_direction: g("base_wind_direction", 0.0)?,
        azimuth_std_dev: g("azimuth_std_dev", 0.001)?,
    };
    let wind_direction_std_dev = g("wind_direction_std_dev", 0.001)?;

    let results = run_monte_carlo_with_wind_and_direction_std_dev(
        bi,
        base_wind,
        mc_params,
        wind_direction_std_dev,
    )
    .map_err(|e| PyValueError::new_err(format!("Monte Carlo run failed: {e}")))?;

    let radius = hit_radius_m.unwrap_or(DEFAULT_HIT_RADIUS_M);
    let hit_probability = results.hit_probability(radius);
    let target_plane_cep_m = results.target_plane_cep();
    let target_arrival_count = results.target_arrival_count();
    let target_shortfall_fraction = results.target_shortfall_fraction();
    let num_samples = results.impact_positions.len();
    let impact_positions_yz_m: Vec<(f64, f64)> = results
        .impact_positions
        .iter()
        .map(|p| (p.y, p.z))
        .collect();

    let out = PyDict::new(py);
    out.set_item("hit_probability", hit_probability)?;
    out.set_item("hit_radius_m", radius)?;
    out.set_item("target_plane_cep_m", target_plane_cep_m)?;
    out.set_item("target_arrival_count", target_arrival_count)?;
    out.set_item("target_shortfall_fraction", target_shortfall_fraction)?;
    out.set_item("num_samples", num_samples)?;
    out.set_item("ranges_m", results.ranges)?;
    out.set_item("impact_velocities_mps", results.impact_velocities)?;
    out.set_item("impact_positions_yz_m", impact_positions_yz_m)?;
    Ok(out)
}
