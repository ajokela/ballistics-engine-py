#!/usr/bin/env python3
"""Coverage for ballistics_engine.bridge_call, the engine's JSON command bridge.

The point of the wrapper is that ONE string-in/string-out function reaches the whole
v1 service surface, including solve inputs that have no typed wrapper in this module.
So the assertions here are about the transport (envelope in, envelope out, errors
in-band rather than as exceptions) and about one of those otherwise-unreachable
inputs actually reaching the physics: effects.wind_shear_model.

The shear check uses a LOFTED shot on purpose. The boundary-layer profile floors at
1.0 below the 10 m reference height, so a flat-fire trajectory is bit-identical with
and without shear and would prove nothing. The lofted shot apexes ~180 m up, well
inside the sheared region, and fixes muzzle_angle_rad directly so a re-converged zero
search cannot confound the comparison.

Plain-python smoke test (no pytest dependency), matching this repo's existing
test_bindings.py / test_mba1295.py convention. Run via:

    maturin develop --release   # or install the built wheel
    python test_bridge_call.py
"""
import json
import sys

import ballistics_engine
from ballistics_engine import bridge_call

EXPECTED_ENGINE_VERSION = "0.36.1"

FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print(f"  ok: {name}")
    else:
        msg = f"FAIL: {name} {detail}".rstrip()
        print(f"  {msg}")
        FAILURES.append(msg)


def call(command, request=None, api_version=1):
    envelope = {"api_version": api_version, "command": command}
    if request is not None:
        envelope["request"] = request
    raw = bridge_call(json.dumps(envelope))
    check(f"{command}: returns str", isinstance(raw, str), f"got {type(raw).__name__}")
    return raw


def lofted_request(effects):
    """~2500 m lofted shot; apex ~186 m above the muzzle, so shear is engaged."""
    return {
        "schema_version": 1,
        "projectile": {
            "mass_kg": 0.01134,
            "diameter_m": 0.00782,
            "length_m": 0.0338,
            "drag_model": "G7",
            "ballistic_coefficient": 0.243,
        },
        "rifle": {"muzzle_velocity_mps": 823.0, "sight_height_m": 0.0381},
        "shot": {
            "max_range_m": 2500.0,
            "muzzle_angle_rad": 0.15,
            "ground_threshold_m": -2000.0,
        },
        "atmosphere": {
            "altitude_m": 0.0,
            "temperature_k": 288.15,
            "pressure_pa": 101325.0,
            "relative_humidity": 0.5,
        },
        # A pure 90-degree crosswind, so shear shows up as windage.
        "wind": {"speed_mps": 4.4704, "direction_from_rad": 1.5707963267948966},
        "solver": {},
        "effects": effects,
        "sampling": {"interval_m": 250.0},
    }


def windage_at_max_range(result):
    """Rightmost sample's lateral offset, meters."""
    samples = result["samples"]
    return samples[-1]["windage_m"]


def main():
    print(f"ballistics_engine {ballistics_engine.__version__}")

    print("\nmeta.version")
    out = json.loads(call("meta.version"))
    check("ok is True", out.get("ok") is True, str(out))
    check("envelope api_version is 1", out.get("api_version") == 1, str(out.get("api_version")))
    check(
        f"envelope engine_version is {EXPECTED_ENGINE_VERSION}",
        out.get("engine_version") == EXPECTED_ENGINE_VERSION,
        f"got {out.get('engine_version')!r}",
    )
    check(
        f"result.engine_version is {EXPECTED_ENGINE_VERSION}",
        out["result"].get("engine_version") == EXPECTED_ENGINE_VERSION,
        f"got {out['result'].get('engine_version')!r}",
    )

    print("\nmeta.capabilities")
    caps = json.loads(call("meta.capabilities"))
    check("ok is True", caps.get("ok") is True, str(caps))
    commands = caps["result"]["commands"]
    for expected in ("solve", "card.come_ups", "true.wind", "profile.validate"):
        check(f"advertises {expected}", expected in commands, str(commands))
    check(
        "capabilities and version agree on engine_version",
        caps["result"]["engine_version"] == out["result"]["engine_version"],
    )

    print("\nsolve round trip (no shear)")
    plain = json.loads(call("solve", lofted_request({})))
    check("ok is True", plain.get("ok") is True, str(plain)[:400])
    check("command echoed", plain.get("command") == "solve", str(plain.get("command")))
    plain_result = plain["result"]
    check("result carries samples", len(plain_result["samples"]) > 1)
    check(
        "no shear model echoed when the field is omitted",
        "wind_shear_model" not in plain_result["resolved_request"]["effects"],
        str(plain_result["resolved_request"]["effects"]),
    )
    plain_windage = windage_at_max_range(plain_result)
    print(f"    windage at last sample: {plain_windage:.6f} m")

    print("\nsolve with effects.wind_shear_model (unreachable without the bridge)")
    for model in ("logarithmic", "power_law"):
        sheared = json.loads(call("solve", lofted_request({"wind_shear_model": model})))
        check(f"{model}: ok is True", sheared.get("ok") is True, str(sheared)[:400])
        result = sheared["result"]
        check(
            f"{model}: echoed at resolved_request.effects.wind_shear_model",
            result["resolved_request"]["effects"].get("wind_shear_model") == model,
            str(result["resolved_request"]["effects"]),
        )
        sheared_windage = windage_at_max_range(result)
        print(f"    windage at last sample: {sheared_windage:.6f} m")
        check(
            f"{model}: shear actually CHANGES the answer",
            sheared_windage != plain_windage,
            f"sheared {sheared_windage!r} == unsheared {plain_windage!r}; "
            "the flag was accepted but inert",
        )

    print("\nekman_spiral is accepted but warns rather than passing silently")
    ekman = json.loads(call("solve", lofted_request({"wind_shear_model": "ekman_spiral"})))
    check("ok is True", ekman.get("ok") is True, str(ekman)[:400])
    warnings = json.dumps(ekman["result"].get("warnings", []))
    check(
        "carries wind_shear_model_not_modeled",
        "wind_shear_model_not_modeled" in warnings,
        warnings[:400],
    )

    print("\nthe other two bridge-only solve inputs reach the engine")
    # atmosphere.pressure_reference: "qnh" reads the given pressure as sea-level-corrected
    # and back-converts it for the station altitude, so at 1500 m it must move the answer.
    flat = lofted_request({})
    flat["shot"] = {"max_range_m": 900.0, "zero_distance_m": 100.0}
    flat["sampling"] = {"interval_m": 300.0}
    flat["atmosphere"]["altitude_m"] = 1500.0

    absolute = json.loads(call("solve", flat))
    check("pressure_reference absolute (default): ok", absolute.get("ok") is True, str(absolute)[:400])

    qnh_request = json.loads(json.dumps(flat))
    qnh_request["atmosphere"]["pressure_reference"] = "qnh"
    qnh = json.loads(call("solve", qnh_request))
    check("pressure_reference qnh: ok", qnh.get("ok") is True, str(qnh)[:400])
    if absolute.get("ok") and qnh.get("ok"):
        check(
            "pressure_reference qnh CHANGES the answer at 1500 m",
            qnh["result"]["samples"][-1]["drop_m"]
            != absolute["result"]["samples"][-1]["drop_m"],
            "qnh was accepted but inert",
        )

    # corrections.bc5d_table_path: no table ships with this wheel, so prove reachability
    # the honest way — a missing file must fail AT THAT FIELD'S PATH, which only happens
    # if the field was parsed and acted on rather than ignored.
    bc5d_request = json.loads(json.dumps(flat))
    bc5d_request["corrections"] = {"bc5d_table_path": "/nonexistent/bc5d_308.bin"}
    bc5d = json.loads(call("solve", bc5d_request))
    bc5d_error = bc5d.get("error", {}).get("details", {}).get("error", {})
    check(
        "bc5d_table_path is acted on (error carries its own JSON path)",
        bc5d_error.get("path") == "$.corrections.bc5d_table_path",
        json.dumps(bc5d_error)[:300],
    )

    print("\nerrors come back IN the envelope, not as exceptions")
    for name, raw in (
        ("not JSON at all", bridge_call("this is not json")),
        ("empty string", bridge_call("")),
        (
            "unknown command",
            call("no.such.command"),
        ),
        (
            "unsupported api_version",
            call("meta.version", api_version=99),
        ),
        (
            "invalid solve field",
            call("solve", lofted_request({"wind_shear_model": "typo_model"})),
        ),
        (
            "missing required solve section",
            call("solve", {"schema_version": 1}),
        ),
    ):
        try:
            err = json.loads(raw)
        except json.JSONDecodeError as exc:
            check(f"{name}: response is valid JSON", False, str(exc))
            continue
        check(f"{name}: response is valid JSON", True)
        check(f"{name}: ok is False", err.get("ok") is False, str(err)[:300])
        check(
            f"{name}: carries error.code",
            isinstance(err.get("error", {}).get("code"), str),
            str(err)[:300],
        )
        print(f"    error.code = {err['error']['code']}")

    print("\nthe wrapper does not parse or validate for the caller")
    check(
        "argument must be a str",
        isinstance(bridge_call.__doc__, str) and "envelope" in bridge_call.__doc__,
        "docstring should describe the envelope",
    )
    raised = None
    try:
        bridge_call(12345)
    except TypeError as exc:
        raised = exc
    check("non-str argument raises TypeError", isinstance(raised, TypeError), repr(raised))

    print()
    if FAILURES:
        print(f"{len(FAILURES)} failure(s):")
        for msg in FAILURES:
            print(f"  {msg}")
        return 1
    print("All bridge_call checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
