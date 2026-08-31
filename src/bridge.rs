// The engine's versioned JSON command bridge, exposed as one string-in/string-out
// pyfunction. This is deliberately NOT a typed mirror of the v1 solve DTOs: the bridge
// is a single entry point onto every command the engine's service layer exposes, so
// wrapping the one `&str -> String` function keeps the whole surface reachable — and
// keeps it reachable as the engine grows commands — without this crate having to grow a
// wrapper per field.

use pyo3::prelude::*;

/// Call the engine's versioned JSON command bridge.
///
/// One request envelope in, one response envelope out, both as JSON strings. The
/// bridge is a transport onto the engine's service layer, not a second
/// implementation, so a command routed through here returns exactly what the
/// library service returns.
///
/// Request envelope::
///
///     {"api_version": 1, "command": "solve", "request": {...}}
///
/// `request` is the command's own payload (for ``solve`` that is a v1 solve request,
/// whose own ``schema_version`` is separate from the envelope's ``api_version``).
/// Commands that take no payload — ``meta.capabilities``, ``meta.version`` — may omit
/// it.
///
/// Success envelope::
///
///     {"ok": true, "api_version": 1, "engine_version": "0.36.1",
///      "command": "solve", "result": {...}}
///
/// Failure envelope::
///
///     {"ok": false, "api_version": 1, "engine_version": "0.36.1",
///      "error": {"code": "...", "message": "..."}}
///
/// **Failures come back inside the returned JSON, not as a Python exception.** A
/// malformed envelope, an unknown command, an invalid field, a request over the size
/// limit, and even an internal panic (the bridge is ``catch_unwind`` guarded) all
/// return a well-formed ``{"ok": false, ...}`` document. This function does not parse
/// the response, does not inspect ``ok``, and does not raise on an error envelope —
/// the caller owns the envelope and can ``json.loads`` it if they want objects.
///
/// Commands include ``meta.capabilities``, ``meta.version``, ``solve``,
/// ``card.come_ups``, ``card.range_table``, ``card.wind``, ``profile.validate``,
/// ``profile.normalize``, ``true.fit``, ``true.wind``, ``true.tall_target``,
/// ``true.dsf``, ``true.plan``, ``true.dial_plan`` and ``bc5d.info``. The list is
/// build-dependent — this wheel links the engine with default features off, so the
/// PDF- and profile-import-gated commands (``pdf``, ``card.pdf``,
/// ``profile.import_a7p``) are not compiled in. Ask ``meta.capabilities`` rather than
/// assuming; it reports exactly what this build can run.
///
/// ``solve`` is the only route to ``corrections.bc5d_table_path`` and
/// ``atmosphere.pressure_reference``, which the typed inputs in this module cannot
/// carry at all.
///
/// It is also the better route to ``effects.wind_shear_model``, though not the only
/// one: the inputs dict has taken ``wind_shear_model`` for some time, but it accepts
/// any string and an unrecognised one is silently resolved to the power law rather
/// than reported. The bridge validates it against a typed enum and tells you which
/// field was wrong.
///
/// Example::
///
///     import json
///     from ballistics_engine import bridge_call
///
///     out = json.loads(bridge_call(json.dumps({
///         "api_version": 1,
///         "command": "meta.version",
///     })))
///     out["result"]["engine_version"]
#[pyfunction]
#[pyo3(text_signature = "(request_json)")]
pub fn bridge_call(py: Python<'_>, request_json: &str) -> String {
    // Pure Rust with no Python interaction for the whole call, and a solve is long
    // enough to be worth not blocking other threads on.
    py.detach(|| ::ballistics_engine::bridge::bridge_call(request_json))
}
