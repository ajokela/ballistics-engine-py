#!/usr/bin/env python3
"""MBA-1295 Phase 1 coverage: the extended BallisticInputs dict parser, the new
TrajectorySolver setters, WindConditions.vertical_speed_mps, TrajectoryResult sampling
passthrough, auto_zero_inputs, and run_monte_carlo.

Plain-python smoke test (no pytest dependency), matching this repo's existing
test_bindings.py convention. Run via:

    maturin develop --release   # or install the built wheel
    python test_mba1295.py
"""
import math
import sys
import traceback

from ballistics_engine import (
    AtmosphericConditions,
    BallisticInputs,
    DragModel,
    TrajectorySolver,
    WindConditions,
    auto_zero_inputs,
    run_monte_carlo,
)

FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print(f"  ok: {name}")
    else:
        msg = f"FAIL: {name} {detail}".rstrip()
        print(f"  {msg}")
        FAILURES.append(msg)


def yz_at(points, x_target):
    """Interpolate (y, z) yards at downrange distance x_target yards."""
    for i in range(1, len(points)):
        if points[i].x >= x_target:
            p1, p2 = points[i - 1], points[i]
            dx = p2.x - p1.x
            t = 0.0 if abs(dx) < 1e-12 else (x_target - p1.x) / dx
            y = p1.y + t * (p2.y - p1.y)
            z = p1.z + t * (p2.z - p1.z)
            return y, z
    raise AssertionError(f"trajectory never reached x={x_target}yd")


BASE_DICT = {
    "bc": 0.5,
    "drag_model": "G7",
    "bullet_weight_grains": 168.0,
    "muzzle_velocity_fps": 2625.0,  # ~800 m/s
    "bullet_diameter_inches": 0.308,
    "bullet_length_inches": 1.2,
    "sight_height_inches": 1.97,  # ~0.05 m
    "twist_rate_inches": 10.0,
    "zero_distance_yards": 100.0,
}


def solver_from_dict(d, wind=None, atmo=None, max_range_m=500.0):
    inputs = BallisticInputs.from_dict(d)
    s = TrajectorySolver(inputs, wind=wind, atmosphere=atmo)
    s.set_max_range(max_range_m)
    return s


# ---------------------------------------------------------------------------


def test_backward_compat_legacy_from_dict():
    print("test_backward_compat_legacy_from_dict")
    # Every key that worked before (test_bindings.py's 11-key style) must still parse.
    d = dict(BASE_DICT)
    inputs = BallisticInputs.from_dict(d)
    check("bc round-trips", inputs.bc == 0.5, f"got {inputs.bc}")
    check(
        "bullet_weight_grains round-trips",
        abs(inputs.bullet_weight_grains - 168.0) < 1e-9,
    )
    check(
        "muzzle_velocity_fps round-trips",
        abs(inputs.muzzle_velocity_fps - 2625.0) < 1e-6,
    )
    check("is_right_twist defaults true", inputs.is_right_twist is True)
    repr(inputs)  # must not raise

    # Missing keys fall back to the documented constructor defaults.
    empty = BallisticInputs.from_dict({})
    ctor_default = BallisticInputs()
    check(
        "empty-dict bc matches constructor default",
        empty.bc == ctor_default.bc,
        f"{empty.bc} vs {ctor_default.bc}",
    )
    check(
        "empty-dict muzzle_velocity_fps matches constructor default",
        abs(empty.muzzle_velocity_fps - ctor_default.muzzle_velocity_fps) < 1e-6,
    )

    s = TrajectorySolver(inputs)
    s.set_max_range(500.0)
    result = s.solve()
    check("legacy-keyed dict solves", result.time_of_flight > 0)


def test_new_properties_roundtrip():
    print("test_new_properties_roundtrip")
    i = BallisticInputs()
    i.cant_angle_degrees = 10.0
    check("cant_angle_degrees round-trips", abs(i.cant_angle_degrees - 10.0) < 1e-9)

    i.enable_trajectory_sampling = True
    check("enable_trajectory_sampling round-trips", i.enable_trajectory_sampling is True)

    i.sample_interval_m = 25.0
    check("sample_interval_m round-trips", abs(i.sample_interval_m - 25.0) < 1e-9)

    i.use_rk4 = False
    check("use_rk4 round-trips", i.use_rk4 is False)
    i.use_adaptive_rk45 = True
    check("use_adaptive_rk45 round-trips", i.use_adaptive_rk45 is True)

    i.muzzle_height_inches = 2.0
    check("muzzle_height_inches round-trips", abs(i.muzzle_height_inches - 2.0) < 1e-9)
    i.target_height_inches = 3.0
    check("target_height_inches round-trips", abs(i.target_height_inches - 3.0) < 1e-9)


def test_rich_keys_parse_on_solver_path():
    print("test_rich_keys_parse_on_solver_path")
    d = dict(BASE_DICT)
    d.update({
        "cant_angle_degrees": 0.0,
        "enable_trajectory_sampling": False,
        "use_rk4": True,
        "use_adaptive_rk45": False,
        "enable_coriolis": False,
    })
    inputs = BallisticInputs.from_dict(d)
    check("use_rk4 parsed from dict", inputs.use_rk4 is True)
    check("use_adaptive_rk45 parsed from dict", inputs.use_adaptive_rk45 is False)
    s = TrajectorySolver(inputs)
    s.set_max_range(500.0)
    result = s.solve()
    check("rich-keyed dict solves", result.time_of_flight > 0)


def test_cant_tilts_poi_right_and_low():
    print("test_cant_tilts_poi_right_and_low")
    # ~0.003 rad (~10 MOA) up, matching the engine's own
    # cant_sign_clockwise_up_offset_goes_right_and_low unit test.
    muzzle_angle_deg = math.degrees(0.003)

    level_dict = dict(BASE_DICT)
    level_dict["muzzle_angle"] = muzzle_angle_deg
    level = solver_from_dict(level_dict, max_range_m=500.0).solve()

    canted_dict = dict(level_dict)
    canted_dict["cant_angle_degrees"] = 10.0
    canted = solver_from_dict(canted_dict, max_range_m=500.0).solve()

    y0, z0 = yz_at(level.points, 300.0)
    y1, z1 = yz_at(canted.points, 300.0)
    check("clockwise cant moves POI right (+z)", z1 > z0 + 0.01, f"z0={z0} z1={z1}")
    check("clockwise cant moves POI low (-y)", y1 < y0 - 0.001, f"y0={y0} y1={y1}")


def test_vertical_wind_raises_poi():
    print("test_vertical_wind_raises_poi")
    d = dict(BASE_DICT)
    calm = solver_from_dict(d, wind=WindConditions(0.0, 0.0, 0.0), max_range_m=500.0).solve()
    updraft = solver_from_dict(
        d, wind=WindConditions(0.0, 0.0, 5.0), max_range_m=500.0
    ).solve()

    y_calm, _ = yz_at(calm.points, 400.0)
    y_updraft, _ = yz_at(updraft.points, 400.0)
    check(
        "5 m/s updraft raises POI at 400yd",
        y_updraft > y_calm,
        f"calm={y_calm} updraft={y_updraft}",
    )


def test_custom_drag_table_changes_drop():
    print("test_custom_drag_table_changes_drop")
    d = dict(BASE_DICT)
    baseline = solver_from_dict(d, max_range_m=500.0).solve()

    dragged_dict = dict(d)
    # Flat, much higher Cd than G7 across the whole Mach range -> far more drag.
    # JSON-shaped list-of-lists, matching what a Flask/JSON caller actually sends.
    dragged_dict["custom_drag_table"] = [[0.3, 1.0], [1.0, 1.0], [2.0, 1.0], [3.5, 1.0]]
    dragged = solver_from_dict(dragged_dict, max_range_m=500.0).solve()

    y_base, _ = yz_at(baseline.points, 150.0)
    y_drag, _ = yz_at(dragged.points, 150.0)
    check(
        "custom_drag_table with much higher Cd increases drop at 150yd",
        y_drag < y_base - 0.02,
        f"baseline={y_base} dragged={y_drag}",
    )

    bad_dict = dict(d)
    bad_dict["custom_drag_table"] = [[1.0, 0.3]]  # only 1 point -> invalid
    raised = False
    try:
        BallisticInputs.from_dict(bad_dict)
    except ValueError:
        raised = True
    check("malformed custom_drag_table raises ValueError", raised)


def test_trajectory_sampling_returns_rows():
    print("test_trajectory_sampling_returns_rows")
    d = dict(BASE_DICT)
    d["enable_trajectory_sampling"] = True
    d["sample_interval_m"] = 50.0
    result = solver_from_dict(d, max_range_m=500.0).solve()

    check("sampled_points is populated", result.sampled_points is not None)
    if result.sampled_points is not None:
        check("sampled_points has rows", len(result.sampled_points) >= 2)
        row = result.sampled_points[0]
        for key in ("distance_m", "drop_m", "wind_drift_m", "velocity_mps", "energy_j", "time_s", "flags"):
            check(f"sample row has {key}", key in row, f"keys={list(row.keys())}")
        check("flags is a list", isinstance(row["flags"], list))

    # Sampling off (default) -> None.
    d_off = dict(BASE_DICT)
    result_off = solver_from_dict(d_off, max_range_m=500.0).solve()
    check("sampled_points is None when sampling disabled", result_off.sampled_points is None)


def test_four_tuple_wind_segment_parses():
    print("test_four_tuple_wind_segment_parses")
    d = dict(BASE_DICT)
    inputs = BallisticInputs.from_dict(d)
    s = TrajectorySolver(inputs)
    s.set_max_range(500.0)
    # 3-tuple and 4-tuple (with vertical_mps) in the same call.
    s.set_wind_segments([(20.0, 90.0, 200.0), (15.0, 90.0, 1000.0, 3.0)])
    result = s.solve()
    check("solve succeeds with mixed 3/4-tuple wind segments", result.time_of_flight > 0)
    _, z_far = yz_at(result.points, 400.0)
    check("segmented crosswind produces nonzero windage", abs(z_far) > 0.01, f"z={z_far}")


def test_atmo_segments_accepted():
    print("test_atmo_segments_accepted")
    d = dict(BASE_DICT)
    inputs = BallisticInputs.from_dict(d)
    s = TrajectorySolver(inputs)
    s.set_max_range(500.0)
    s.set_atmo_segments([(20.0, 1000.0, 50.0, 200.0), (0.0, 1013.25, 50.0, 1000.0)])
    result = s.solve()
    check("solve succeeds with atmo segments", result.time_of_flight > 0)
    s.set_atmo_segments([])  # clears back to single-station; must not raise
    result2 = s.solve()
    check("clearing atmo segments still solves", result2.time_of_flight > 0)


def test_auto_zero_then_solve_lands_near_sight_line():
    print("test_auto_zero_then_solve_lands_near_sight_line")
    d = dict(BASE_DICT)
    d["zero_distance_yards"] = 200.0

    unzeroed = solver_from_dict(d, max_range_m=500.0).solve()
    y_unzeroed, _ = yz_at(unzeroed.points, 200.0)

    muzzle_angle_rad = auto_zero_inputs(d, 200.0)
    check("auto_zero_inputs returns a finite angle", math.isfinite(muzzle_angle_rad))
    check("auto_zero_inputs returns a positive (up) angle", muzzle_angle_rad > 0.0)

    zeroed_dict = dict(d)
    zeroed_dict["muzzle_angle"] = math.degrees(muzzle_angle_rad)
    zeroed = solver_from_dict(zeroed_dict, max_range_m=500.0).solve()
    y_zeroed, _ = yz_at(zeroed.points, 200.0)

    check(
        "zeroed POI is much closer to the sight line at zero distance than unzeroed",
        abs(y_zeroed) < abs(y_unzeroed) * 0.1,
        f"y_unzeroed={y_unzeroed} y_zeroed={y_zeroed}",
    )


def test_from_dict_missing_sight_height_defaults_to_1_5_inches():
    """MBA-1295 review lock: the legacy from_dict documented a 1.5-inch default for
    sight_height_inches; the shared parser must preserve it (0.0381 m internally)."""
    print("test_from_dict_missing_sight_height_defaults_to_1_5_inches")
    d = dict(BASE_DICT)
    d.pop("sight_height_inches", None)
    inputs = BallisticInputs.from_dict(d)
    # The getter divides the internal SI meters by 0.0254, so 1.5 here implies the
    # engine-side field carries exactly 1.5 * 0.0254 = 0.0381 m.
    check(
        "from_dict without sight_height_inches defaults to 1.5 in (0.0381 m)",
        abs(inputs.sight_height_inches - 1.5) < 1e-12,
        f"got {inputs.sight_height_inches}",
    )
    empty = BallisticInputs.from_dict({})
    check(
        "empty-dict sight_height_inches also defaults to 1.5",
        abs(empty.sight_height_inches - 1.5) < 1e-12,
        f"got {empty.sight_height_inches}",
    )


def test_monte_carlo_parallel_drop_is_bore_relative():
    """MBA-1295 review lock: pre-MBA-1295 the fast-path parser hardcoded sight_height 0.0,
    so monte_carlo_parallel drop has always been BORE-relative (engine kernel computes
    drop = (muzzle_height + sight_height) - final_y). The shared parser now reads
    sight_height from the dict, so monte_carlo_parallel must re-zero it at its own
    boundary. This test FAILS if sight_height leaks in: a Flask-shaped dict with
    sight_height 1.5 must produce mean drop identical to one with sight_height 0.0
    (a leak would shift it by exactly 1.5 in = 0.0381 m)."""
    print("test_monte_carlo_parallel_drop_is_bore_relative")
    import numpy as np

    from ballistics_engine import monte_carlo_parallel

    # Flask-shaped MC dict (the imperial MC convention: fps/grains/yards/feet/km/h/deg/%).
    flask_dict = {
        "bc_value": 0.5,
        "bc_type": "G7",
        "bullet_mass": 168.0,
        "altitude": 0.0,
        "muzzle_velocity": 2650.0,
        "target_distance": 300.0,
        "twist_rate": 10.0,
        "bullet_length": 1.2,
        "bullet_diameter": 0.308,
        "temperature": 15.0,
        "pressure": 1013.25,
        "humidity": 50.0,
        "sight_height": 1.5,
    }
    samples = np.array([[0.5], [0.5]])  # 2 identical deterministic runs
    names = ["bc_value"]

    def mean_drop(d):
        out = monte_carlo_parallel(d, samples, names)
        assert out["metadata"]["valid_runs"] == 2, out["metadata"]
        return out["statistics"]["drop_m"]["mean"]

    drop_with_sight = mean_drop(flask_dict)

    zeroed_dict = dict(flask_dict)
    zeroed_dict["sight_height"] = 0.0
    drop_zeroed = mean_drop(zeroed_dict)

    absent_dict = dict(flask_dict)
    del absent_dict["sight_height"]  # parser default (1.5) must ALSO be zeroed at MC boundary
    drop_absent = mean_drop(absent_dict)

    check(
        "MC mean drop with sight_height 1.5 equals old (bore-relative) behavior",
        abs(drop_with_sight - drop_zeroed) < 1e-12,
        f"with={drop_with_sight} zeroed={drop_zeroed} (leak would differ by 0.0381)",
    )
    check(
        "MC mean drop with sight_height absent equals old (bore-relative) behavior",
        abs(drop_absent - drop_zeroed) < 1e-12,
        f"absent={drop_absent} zeroed={drop_zeroed}",
    )
    check("MC mean drop is finite and positive at 300yd", drop_zeroed > 0.0, f"drop={drop_zeroed}")


def test_run_monte_carlo_hit_probability():
    print("test_run_monte_carlo_hit_probability")
    d = dict(BASE_DICT)
    d["zero_distance_yards"] = 100.0
    params = {
        "num_simulations": 40,
        "velocity_std_dev": 5.0,
        "angle_std_dev": 0.0005,
        "bc_std_dev": 0.005,
        "wind_speed_std_dev": 0.5,
        "wind_direction_std_dev": 0.05,
        "target_distance": 100.0 * 0.9144,
    }
    out = run_monte_carlo(d, params)
    check("hit_probability present", "hit_probability" in out)
    hp = out["hit_probability"]
    check("hit_probability in [0, 1]", 0.0 <= hp <= 1.0, f"hp={hp}")
    check("num_samples > 0", out["num_samples"] > 0, f"num_samples={out['num_samples']}")
    check(
        "ranges_m length matches num_samples",
        len(out["ranges_m"]) == out["num_samples"],
    )
    cep = out["target_plane_cep_m"]
    check("target_plane_cep_m is None or non-negative float", cep is None or cep >= 0.0)


def test_monte_carlo_parallel_full_vs_fast_agreement():
    """MBA-1295 Phase 3: `use_full_solver=True` (TrajectorySolver) must agree closely with
    the historical `use_full_solver=False` fast path (solve_trajectory_for_monte_carlo) --
    this is the whole point of the phase (kill the cross-solver divergence between
    monte_carlo_parallel and /v1/calculate). monte_carlo_parallel only returns AGGREGATE
    statistics, not raw per-sample rows, so this drives it with a single-sample (n=1)
    param_samples array per row: with valid_runs==1 the reported "mean" is exactly that one
    trajectory's output (see the valid_runs == 1 branch in montecarlo.rs), giving an exact
    per-sample comparison for a handful of distinct rows."""
    print("test_monte_carlo_parallel_full_vs_fast_agreement")
    import numpy as np

    from ballistics_engine import monte_carlo_parallel

    base = {
        "bc_value": 0.5,
        "bc_type": "G1",
        "muzzle_velocity": 2800.0,
        "bullet_mass": 175.0,
        "target_distance": 500.0,
        "bullet_diameter": 0.308,
        "bullet_length": 1.2,
        "twist_rate": 10.0,
        "temperature": 15.0,
        "pressure": 1013.25,
        "humidity": 50.0,
        "altitude": 0.0,
        "sight_height": 1.5,
    }
    # Crosswind base (wind_angle=90, matches the engine's headwind=0/from-the-right=90
    # convention) so the wind_speed row below produces a NONZERO wind_drift_m -- the plain
    # headwind base above would give a trivial 0 vs 0 "agreement" for that field.
    crosswind_base = dict(base, wind_angle=90.0)

    # A handful of distinct single-parameter rows spanning the kind of jitter a real MC
    # param_stddevs call would sample (muzzle_velocity fps, bc_value, wind_speed km/h).
    rows = [
        (base, "muzzle_velocity", 2790.0),
        (base, "muzzle_velocity", 2820.0),
        (base, "bc_value", 0.495),
        (base, "bc_value", 0.51),
        (crosswind_base, "wind_speed", 8.0),
    ]
    fields = ["drop_m", "wind_drift_m", "time_of_flight_s", "final_vel_fps", "energy_ft_lbs", "mach"]

    for base_dict, name, value in rows:
        samples = np.array([[value]])
        fast = monte_carlo_parallel(base_dict, samples, [name], None, False, 500, False)
        full = monte_carlo_parallel(base_dict, samples, [name], None, False, 500, True)
        check(
            f"{name}={value}: fast path solved (valid_runs==1)",
            fast["metadata"]["valid_runs"] == 1,
            fast["metadata"],
        )
        check(
            f"{name}={value}: full solver solved (valid_runs==1)",
            full["metadata"]["valid_runs"] == 1,
            full["metadata"],
        )
        for f in fields:
            fv = fast["statistics"][f]["mean"]
            gv = full["statistics"][f]["mean"]
            denom = max(abs(fv), 1e-9)
            rel = abs(fv - gv) / denom
            check(
                f"{name}={value}: {f} agrees within 1% (fast={fv:.6g} full={gv:.6g})",
                rel < 0.01,
                f"rel_diff={rel*100:.4f}%",
            )


def test_monte_carlo_parallel_hit_probability_sane():
    """MBA-1295 Phase 3: the additive dispersion.hit_probability / hit_radius_m /
    target_plane_cep_m fields must be present and sane for both use_full_solver values."""
    print("test_monte_carlo_parallel_hit_probability_sane")
    import numpy as np

    from ballistics_engine import monte_carlo_parallel

    base = {
        "bc_value": 0.5,
        "bc_type": "G1",
        "muzzle_velocity": 2800.0,
        "bullet_mass": 175.0,
        "target_distance": 200.0,  # short range so a tight mechanical jitter can land hits
        "bullet_diameter": 0.308,
        "bullet_length": 1.2,
        "twist_rate": 10.0,
        "temperature": 15.0,
        "pressure": 1013.25,
        "humidity": 50.0,
        "altitude": 0.0,
        "sight_height": 1.5,
        "muzzle_angle": 0.5,  # a bit of elevation so the group centers near the point of aim
    }
    rng = np.random.default_rng(1295)
    n = 200
    samples = np.column_stack([
        rng.normal(2800.0, 5.0, n),
        rng.normal(0.5, 0.003, n),
    ])
    names = ["muzzle_velocity", "bc_value"]

    for use_full_solver in (False, True):
        out = monte_carlo_parallel(base, samples, names, None, True, 500, use_full_solver, 5.0)
        disp = out.get("dispersion")
        check(f"dispersion present (use_full_solver={use_full_solver})", disp is not None)
        if disp is None:
            continue
        hp = disp.get("hit_probability")
        check(
            f"hit_probability in [0, 1] (use_full_solver={use_full_solver})",
            hp is not None and 0.0 <= hp <= 1.0,
            f"hp={hp}",
        )
        check(
            f"hit_radius_m round-trips the passed radius (use_full_solver={use_full_solver})",
            abs(disp.get("hit_radius_m", -1.0) - 5.0) < 1e-9,
        )
        cep = disp.get("target_plane_cep_m")
        check(
            f"target_plane_cep_m is a non-negative float (use_full_solver={use_full_solver})",
            cep is not None and cep >= 0.0,
            f"cep={cep}",
        )


def main():
    tests = [
        test_backward_compat_legacy_from_dict,
        test_new_properties_roundtrip,
        test_rich_keys_parse_on_solver_path,
        test_cant_tilts_poi_right_and_low,
        test_vertical_wind_raises_poi,
        test_custom_drag_table_changes_drop,
        test_trajectory_sampling_returns_rows,
        test_four_tuple_wind_segment_parses,
        test_atmo_segments_accepted,
        test_auto_zero_then_solve_lands_near_sight_line,
        test_from_dict_missing_sight_height_defaults_to_1_5_inches,
        test_monte_carlo_parallel_drop_is_bore_relative,
        test_run_monte_carlo_hit_probability,
        test_monte_carlo_parallel_full_vs_fast_agreement,
        test_monte_carlo_parallel_hit_probability_sane,
    ]
    for t in tests:
        try:
            t()
        except Exception:
            print(f"  EXCEPTION in {t.__name__}:")
            traceback.print_exc()
            FAILURES.append(f"{t.__name__} raised an exception")

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s) failed")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("All MBA-1295 checks passed.")


if __name__ == "__main__":
    main()
