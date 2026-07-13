#!/usr/bin/env python3
"""MBA-1295 Phase 3 perf gate: fast (`solve_trajectory_for_monte_carlo`) vs full
(`ballistics_engine::TrajectorySolver`) per-sample evaluation in `monte_carlo_parallel`.

Hard gate (per the Phase 3 task): full-solver 1000-run wall time must not exceed 3x the
fast-path wall time, AND must not exceed 20 seconds absolute. If either is violated,
`use_full_solver=True` must NOT ship as the default.

Uses the Flask app's own "basic_imperial_smoke" golden fixture params
(tests/golden/fixtures.json in the ballistics-mba1295 worktree) as the base inputs, with
the same muzzle_velocity/bc_value jitter as the "mc_param_stddevs" fixture, at 1000 runs
(the fixture itself uses runs=500 -- this script uses 1000 per the task's explicit perf-gate
spec).

Run via:
    source /tmp/mba1295-phase3-venv/bin/activate   # or any venv with the 0.24.2 wheel installed
    python perf_mba1295_phase3.py
"""
import time

import numpy as np

from ballistics_engine import monte_carlo_parallel

N_SAMPLES = 1000
SEED = 20260713

# Flask "basic_imperial_smoke" golden fixture base params (tests/golden/fixtures.json,
# ballistics-mba1295 worktree), expressed in monte_carlo_parallel's rich-key vocabulary.
BASE_INPUTS = {
    "bc_value": 0.5,
    "bc_type": "G1",
    "muzzle_velocity": 2800.0,   # fps
    "bullet_mass": 175.0,       # grains
    "target_distance": 500.0,   # yards
    "bullet_diameter": 0.308,
    "bullet_length": 1.2,
    "twist_rate": 10.0,
    "temperature": 15.0,
    "pressure": 1013.25,
    "humidity": 50.0,
    "altitude": 0.0,
    "sight_height": 1.5,
}

# Same jitter as the "mc_param_stddevs" golden fixture: muzzle_velocity stddev 10 fps,
# bc_value stddev 0.005.
PARAM_NAMES = ["muzzle_velocity", "bc_value"]
STDDEVS = {"muzzle_velocity": 10.0, "bc_value": 0.005}


def make_samples():
    rng = np.random.default_rng(SEED)
    samples = np.zeros((N_SAMPLES, len(PARAM_NAMES)))
    for j, name in enumerate(PARAM_NAMES):
        base = BASE_INPUTS[name]
        samples[:, j] = rng.normal(base, STDDEVS[name], N_SAMPLES)
    return samples


def timed_run(use_full_solver, samples):
    t0 = time.perf_counter()
    out = monte_carlo_parallel(
        BASE_INPUTS,
        samples,
        PARAM_NAMES,
        None,      # num_threads (default global rayon pool, matching Flask's call)
        True,      # include_dispersion
        500,       # max_viz_points
        use_full_solver,
    )
    elapsed = time.perf_counter() - t0
    return elapsed, out


def main():
    samples = make_samples()

    # Warm up (JIT-free in Rust, but pays for first-call thread pool spin-up / page faults;
    # exclude from the timed comparison).
    monte_carlo_parallel(BASE_INPUTS, samples[:10], PARAM_NAMES, None, False, 500, False)
    monte_carlo_parallel(BASE_INPUTS, samples[:10], PARAM_NAMES, None, False, 500, True)

    fast_time, fast_out = timed_run(False, samples)
    full_time, full_out = timed_run(True, samples)

    fast_valid = fast_out["metadata"]["valid_runs"]
    full_valid = full_out["metadata"]["valid_runs"]

    ratio = full_time / fast_time if fast_time > 0 else float("inf")

    print(f"N_SAMPLES = {N_SAMPLES}")
    print(f"fast path   (solve_trajectory_for_monte_carlo): {fast_time:8.4f}s  valid_runs={fast_valid}")
    print(f"full solver (TrajectorySolver):                 {full_time:8.4f}s  valid_runs={full_valid}")
    print(f"ratio (full/fast): {ratio:.2f}x")
    print(f"gate: ratio <= 3.0x -> {'PASS' if ratio <= 3.0 else 'FAIL'}")
    print(f"gate: full_time <= 20s -> {'PASS' if full_time <= 20.0 else 'FAIL'}")

    gate_pass = ratio <= 3.0 and full_time <= 20.0
    print()
    print("PERF GATE:", "PASS - default use_full_solver=True is safe to ship" if gate_pass
          else "FAIL - report DONE_WITH_CONCERNS, do not ship default-true")

    # Sanity: drop_m means should be in the same ballpark (loose check here; the tight
    # per-sample agreement check lives in test_mba1295.py).
    fast_drop = fast_out["statistics"]["drop_m"]["mean"]
    full_drop = full_out["statistics"]["drop_m"]["mean"]
    rel_diff = abs(fast_drop - full_drop) / abs(fast_drop) if fast_drop else float("nan")
    print(f"\nmean drop_m: fast={fast_drop:.6f}  full={full_drop:.6f}  rel_diff={rel_diff*100:.3f}%")

    # Additive hit_probability / target_plane_cep_m sanity.
    disp = full_out.get("dispersion", {})
    print(f"\nfull-solver dispersion.hit_probability = {disp.get('hit_probability')}")
    print(f"full-solver dispersion.hit_radius_m     = {disp.get('hit_radius_m')}")
    print(f"full-solver dispersion.target_plane_cep_m = {disp.get('target_plane_cep_m')}")


if __name__ == "__main__":
    main()
