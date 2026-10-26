#!/usr/bin/env python3
"""Station presence (engine 0.49.0): a temperature or pressure the caller supplied flies
exactly as given, whatever its value and whatever the altitude. One the caller left out keeps
the engine's legacy rule, which flies ICAO standard air at a real altitude.

Before this, every typed entry point handed the engine plain numbers with 59 °F / 29.92 inHg
defaults. Above 1 m of altitude the engine reads a temperature within 0.1 °C of 15 °C, or a
pressure within 0.5 hPa of 1013.25 hPa, as "not given" and flies the ICAO standard value
instead. So a real 29.92 inHg reading at 5000 ft was silently replaced by about 24.9 inHg, and
a real 59 °F by about 41 °F.

Plain-python smoke test (no pytest dependency), matching this repo's existing
test_bindings.py convention. Run via:

    maturin develop --release   # or install the built wheel
    python test_station_presence.py
"""
import sys
import traceback

import numpy as np

from ballistics_engine import (
    AtmosphericConditions,
    BallisticInputs,
    TrajectorySolver,
    auto_zero_inputs,
    calculate_zero_angle,
    monte_carlo_parallel,
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


ALT_FT = 5000.0
ALT_M = ALT_FT * 0.3048

# Legacy imperial keys, as BallisticInputs(...) takes them.
BASE_DICT = {
    "bc": 0.5,
    "drag_model": "G7",
    "bullet_weight_grains": 168.0,
    "muzzle_velocity_fps": 2650.0,
    "bullet_diameter_inches": 0.308,
    "bullet_length_inches": 1.2,
    "sight_height_inches": 1.5,
    "twist_rate_inches": 10.0,
    "zero_distance_yards": 100.0,
}

# The Monte Carlo dict convention (fps / grains / yards / feet / °C / hPa / %).
MC_BASE = {
    "bc_value": 0.5,
    "bc_type": "G7",
    "bullet_mass": 168.0,
    "muzzle_velocity": 2650.0,
    "target_distance": 600.0,
    "twist_rate": 10.0,
    "bullet_length": 1.2,
    "bullet_diameter": 0.308,
    "humidity": 50.0,
    "altitude": ALT_FT,
}


def icao_station_air(altitude_m):
    """ICAO standard (temperature °C, pressure hPa) at a geometric altitude, the same
    expression the engine flies for an omitted reading (troposphere layer)."""
    radius = 6_356_766.0
    h = radius * altitude_m / (radius + altitude_m)
    t_k = 288.15 - 0.0065 * h
    p_pa = 101325.0 * (t_k / 288.15) ** (-9.80665 / (-0.0065 * 287.0531))
    return t_k - 273.15, p_pa / 100.0


def solve(atmo, max_range_m=800.0):
    solver = TrajectorySolver(BallisticInputs.from_dict(BASE_DICT), atmosphere=atmo)
    solver.set_max_range(max_range_m)
    result = solver.solve()
    return result.impact_velocity_fps, result.points[-1].y


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-12)


# ---------------------------------------------------------------------------


def test_atmospheric_conditions_records_presence():
    print("test_atmospheric_conditions_records_presence")
    a = AtmosphericConditions()
    check("omitted getters still read 59 °F / 29.92 inHg",
          a.temperature_f == 59.0 and a.pressure_inhg == 29.92,
          f"{a.temperature_f} {a.pressure_inhg}")
    check("nothing typed: neither flag set",
          not a.temperature_supplied and not a.pressure_supplied)

    a = AtmosphericConditions(altitude_feet=ALT_FT)
    check("altitude alone sets neither flag",
          not a.temperature_supplied and not a.pressure_supplied)

    a = AtmosphericConditions(temperature_f=59.0, pressure_inhg=29.92)
    check("typed 59 °F / 29.92 inHg are supplied", a.temperature_supplied and a.pressure_supplied)

    a = AtmosphericConditions(59, 29.92, 50, 0)
    check("positional values are supplied (ints accepted)",
          a.temperature_supplied and a.pressure_supplied and a.temperature_f == 59.0)

    a = AtmosphericConditions(temperature_f=None, pressure_inhg=30.1)
    check("None is omitted, a number is supplied",
          not a.temperature_supplied and a.pressure_supplied and a.temperature_f == 59.0)

    a = AtmosphericConditions()
    a.pressure_inhg = 29.92
    check("assigning pressure_inhg marks it supplied",
          a.pressure_supplied and not a.temperature_supplied)
    a.temperature_f = 59.0
    check("assigning temperature_f marks it supplied", a.temperature_supplied)

    a = AtmosphericConditions(temperature_f=59.0, pressure_inhg=29.92)
    a.temperature_supplied = False
    a.pressure_supplied = False
    check("the flags can be cleared without touching the values",
          not a.temperature_supplied and not a.pressure_supplied
          and a.temperature_f == 59.0 and a.pressure_inhg == 29.92)


def test_typed_sea_level_pressure_at_altitude_flies_as_typed():
    print("test_typed_sea_level_pressure_at_altitude_flies_as_typed")
    omitted = solve(AtmosphericConditions(altitude_feet=ALT_FT))
    typed = solve(AtmosphericConditions(pressure_inhg=29.92, altitude_feet=ALT_FT))
    lower = solve(AtmosphericConditions(pressure_inhg=29.80, altitude_feet=ALT_FT))
    higher = solve(AtmosphericConditions(pressure_inhg=30.04, altitude_feet=ALT_FT))
    check("typed 29.92 inHg at 5000 ft differs from omitted",
          rel(typed[0], omitted[0]) > 1e-3, f"typed={typed} omitted={omitted}")
    # Denser air slows the bullet more: v(29.80) > v(29.92) > v(30.04).
    check("typed 29.92 inHg lies between its out-of-band neighbours",
          lower[0] > typed[0] > higher[0], f"29.80={lower[0]} 29.92={typed[0]} 30.04={higher[0]}")


def test_typed_standard_temperature_at_altitude_flies_as_typed():
    print("test_typed_standard_temperature_at_altitude_flies_as_typed")
    omitted = solve(AtmosphericConditions(altitude_feet=ALT_FT))
    typed = solve(AtmosphericConditions(temperature_f=59.0, altitude_feet=ALT_FT))
    colder = solve(AtmosphericConditions(temperature_f=58.6, altitude_feet=ALT_FT))
    warmer = solve(AtmosphericConditions(temperature_f=59.4, altitude_feet=ALT_FT))
    check("typed 59 °F at 5000 ft differs from omitted",
          rel(typed[0], omitted[0]) > 1e-4, f"typed={typed} omitted={omitted}")
    # Colder air is denser: v(58.6) < v(59) < v(59.4).
    check("typed 59 °F lies between its out-of-band neighbours",
          colder[0] < typed[0] < warmer[0], f"58.6={colder[0]} 59={typed[0]} 59.4={warmer[0]}")


def test_omitted_values_still_fly_icao_air():
    print("test_omitted_values_still_fly_icao_air")
    omitted = solve(AtmosphericConditions(altitude_feet=ALT_FT))
    t_c, p_hpa = icao_station_air(ALT_M)
    icao = solve(AtmosphericConditions(temperature_f=t_c * 9.0 / 5.0 + 32.0,
                                       pressure_inhg=p_hpa / 33.8639,
                                       altitude_feet=ALT_FT))
    check("omitted at 5000 ft flies the ICAO standard air",
          rel(omitted[0], icao[0]) < 1e-9 and abs(omitted[1] - icao[1]) < 1e-9,
          f"omitted={omitted} icao={icao}")

    # Flask's pattern: pass its own defaults, then clear the flags.
    legacy = AtmosphericConditions(temperature_f=59.0, pressure_inhg=29.92, altitude_feet=ALT_FT)
    legacy.temperature_supplied = False
    legacy.pressure_supplied = False
    check("defaults passed with the flags cleared equal omitted, bit for bit",
          solve(legacy) == omitted, f"{solve(legacy)} vs {omitted}")


def test_sea_level_is_unchanged():
    print("test_sea_level_is_unchanged")
    inputs = BallisticInputs.from_dict(BASE_DICT)
    s = TrajectorySolver(inputs)
    s.set_max_range(800.0)
    r = s.solve()
    no_atmo = (r.impact_velocity_fps, r.points[-1].y)
    check("no atmosphere equals AtmosphericConditions()",
          no_atmo == solve(AtmosphericConditions()))
    check("at sea level typed 59 °F / 29.92 inHg equal omitted",
          solve(AtmosphericConditions(59.0, 29.92, 50.0, 0.0)) == no_atmo)


def test_auto_zero_inputs_honours_presence():
    print("test_auto_zero_inputs_honours_presence")
    omitted = auto_zero_inputs(BASE_DICT, 600.0, atmo=AtmosphericConditions(altitude_feet=ALT_FT))
    typed = auto_zero_inputs(BASE_DICT, 600.0,
                             atmo=AtmosphericConditions(pressure_inhg=29.92, altitude_feet=ALT_FT))
    check("atmo: typed 29.92 inHg at 5000 ft zeroes differently from omitted",
          typed != omitted, f"typed={typed} omitted={omitted}")

    # Without atmo the dict's own keys are the atmosphere (°C / hPa / feet).
    d = dict(BASE_DICT, altitude=ALT_FT)
    d_omitted = auto_zero_inputs(d, 600.0)
    d_typed = auto_zero_inputs(dict(d, pressure=1013.25), 600.0)
    d_none = auto_zero_inputs(dict(d, pressure=None), 600.0)
    d_marked_off = auto_zero_inputs(dict(d, pressure=1013.25, pressure_supplied=False), 600.0)
    check("dict: a present pressure key is supplied", d_typed != d_omitted,
          f"typed={d_typed} omitted={d_omitted}")
    check("dict: a None pressure is omitted", d_none == d_omitted)
    check("dict: pressure_supplied=False restores the omitted rule",
          d_marked_off == d_omitted, f"{d_marked_off} vs {d_omitted}")

    t_typed = auto_zero_inputs(dict(d, temperature=15.0), 600.0)
    t_marked_off = auto_zero_inputs(dict(d, temperature=15.0, temperature_supplied=False), 600.0)
    check("dict: a typed 15 °C is supplied", t_typed != d_omitted,
          f"typed={t_typed} omitted={d_omitted}")
    check("dict: temperature_supplied=False restores the omitted rule", t_marked_off == d_omitted)

    try:
        auto_zero_inputs(dict(d, pressure=1013.25, pressure_supplied="yes"), 600.0)
        check("dict: a non-bool marker is rejected", False, "no exception raised")
    except TypeError:
        check("dict: a non-bool marker is rejected", True)


def test_calculate_zero_angle_kwargs_honour_presence():
    print("test_calculate_zero_angle_kwargs_honour_presence")
    omitted = calculate_zero_angle(BASE_DICT, 600.0, altitude_ft=ALT_FT)
    typed_p = calculate_zero_angle(BASE_DICT, 600.0, pressure_inhg=29.92, altitude_ft=ALT_FT)
    typed_t = calculate_zero_angle(BASE_DICT, 600.0, temperature_f=59.0, altitude_ft=ALT_FT)
    check("typed 29.92 inHg at 5000 ft zeroes differently from omitted", typed_p != omitted,
          f"typed={typed_p} omitted={omitted}")
    check("typed 59 °F at 5000 ft zeroes differently from omitted", typed_t != omitted,
          f"typed={typed_t} omitted={omitted}")
    check("at sea level typed 59 °F / 29.92 inHg equal omitted",
          calculate_zero_angle(BASE_DICT, 600.0, temperature_f=59.0, pressure_inhg=29.92)
          == calculate_zero_angle(BASE_DICT, 600.0))


def mc_velocity(base, use_full_solver, names=("bc_value",), values=(0.5,)):
    samples = np.array([list(values)])
    out = monte_carlo_parallel(base, samples, list(names), None, False, 500, use_full_solver)
    assert out["metadata"]["valid_runs"] == 1, out["metadata"]
    stats = out["statistics"]
    return stats["final_vel_fps"]["mean"], stats["drop_m"]["mean"], stats["mach"]["mean"]


def test_monte_carlo_parallel_honours_presence():
    print("test_monte_carlo_parallel_honours_presence")
    for full in (True, False):
        path = "full solver" if full else "fast kernel"
        omitted = mc_velocity(MC_BASE, full)
        typed = mc_velocity(dict(MC_BASE, pressure=1013.25), full)
        marked_off = mc_velocity(dict(MC_BASE, pressure=1013.25, pressure_supplied=False), full)
        typed_t = mc_velocity(dict(MC_BASE, temperature=15.0), full)
        check(f"{path}: a typed 1013.25 hPa at 5000 ft differs from omitted",
              rel(typed[0], omitted[0]) > 1e-3, f"typed={typed} omitted={omitted}")
        check(f"{path}: pressure_supplied=False restores the omitted rule",
              marked_off == omitted, f"{marked_off} vs {omitted}")
        check(f"{path}: a typed 15 °C at 5000 ft differs from omitted",
              rel(typed_t[0], omitted[0]) > 1e-4, f"typed={typed_t} omitted={omitted}")

        # A sampled pressure column is a value the caller supplied, like the key.
        sampled = mc_velocity(MC_BASE, full, names=("pressure",), values=(1013.25,))
        sampled_off = mc_velocity(dict(MC_BASE, pressure_supplied=False), full,
                                  names=("pressure",), values=(1013.25,))
        check(f"{path}: a sampled 1013.25 hPa flies as sampled", sampled == typed,
              f"sampled={sampled} typed={typed}")
        check(f"{path}: a sampled column with pressure_supplied=False keeps the omitted rule",
              sampled_off == omitted, f"{sampled_off} vs {omitted}")

    # The full solver reports Mach against the air it flew, not the swapped air: the two
    # solvers agree to about 0.01%, while Mach against the ICAO air at 5000 ft (5 °C instead
    # of the typed 15 °C) would be about 1.7% higher.
    typed_full = mc_velocity(dict(MC_BASE, pressure=1013.25, temperature=15.0), True)
    typed_fast = mc_velocity(dict(MC_BASE, pressure=1013.25, temperature=15.0), False)
    check("full solver Mach agrees with the fast kernel's for typed air",
          rel(typed_full[2], typed_fast[2]) < 0.002, f"full={typed_full[2]} fast={typed_fast[2]}")


def test_run_monte_carlo_honours_presence():
    print("test_run_monte_carlo_honours_presence")
    params = {
        "num_simulations": 3,
        "velocity_std_dev": 0.0,
        "angle_std_dev": 0.0,
        "bc_std_dev": 0.0,
        "wind_speed_std_dev": 0.0,
        "wind_direction_std_dev": 0.0,
        "azimuth_std_dev": 0.0,
        "target_distance": 600.0,
    }

    def velocities(d, atmo=None):
        return run_monte_carlo(d, params, atmo=atmo)["impact_velocities_mps"]

    omitted = velocities(BASE_DICT, AtmosphericConditions(altitude_feet=ALT_FT))
    typed = velocities(BASE_DICT, AtmosphericConditions(pressure_inhg=29.92, altitude_feet=ALT_FT))
    check("atmo: typed 29.92 inHg at 5000 ft differs from omitted",
          rel(typed[0], omitted[0]) > 1e-3, f"typed={typed[0]} omitted={omitted[0]}")

    d = dict(BASE_DICT, altitude=ALT_FT)
    d_omitted = velocities(d)
    d_typed = velocities(dict(d, pressure=1013.25))
    d_marked_off = velocities(dict(d, pressure=1013.25, pressure_supplied=False))
    check("dict: a typed 1013.25 hPa at 5000 ft differs from omitted",
          rel(d_typed[0], d_omitted[0]) > 1e-3, f"typed={d_typed[0]} omitted={d_omitted[0]}")
    check("dict: pressure_supplied=False restores the omitted rule", d_marked_off == d_omitted)


def main():
    tests = [
        test_atmospheric_conditions_records_presence,
        test_typed_sea_level_pressure_at_altitude_flies_as_typed,
        test_typed_standard_temperature_at_altitude_flies_as_typed,
        test_omitted_values_still_fly_icao_air,
        test_sea_level_is_unchanged,
        test_auto_zero_inputs_honours_presence,
        test_calculate_zero_angle_kwargs_honour_presence,
        test_monte_carlo_parallel_honours_presence,
        test_run_monte_carlo_honours_presence,
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
        print("All station-presence checks passed.")


if __name__ == "__main__":
    main()
