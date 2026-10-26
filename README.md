# ballistics-engine

High-performance ballistics trajectory engine with professional physics modeling.

## Features

- Professional-grade trajectory calculations with multiple drag models (G1, G7, G8)
- Advanced physics including wind effects and atmospheric modeling
- Fast Rust implementation with Python bindings via PyO3
- Imperial units API (grains, fps, yards, inches) with automatic metric conversion

## Installation

```bash
pip install ballistics-engine
```

## Quick Start

```python
from ballistics_engine import BallisticInputs, TrajectorySolver, WindConditions, AtmosphericConditions, DragModel

# Create ballistic inputs (all imperial units)
inputs = BallisticInputs(
    bc=0.505,                        # G7 BC
    bullet_weight_grains=168,        # grains
    muzzle_velocity_fps=2650,        # feet per second
    bullet_diameter_inches=0.308,    # inches
    bullet_length_inches=1.24,       # inches
    sight_height_inches=1.5,         # inches above bore
    zero_distance_yards=100,         # yards
    twist_rate_inches=11.25,         # inches per turn
)
inputs.drag_model = DragModel.g7()  # Use G7 drag model

# Create wind conditions (optional)
wind = WindConditions(
    speed_mph=10,                    # mph
    direction_degrees=90,            # degrees (0=headwind, 90=from right)
)

# Create atmospheric conditions (optional). Pass the readings you have and leave
# the rest out: a value you pass flies exactly as given.
atmosphere = AtmosphericConditions(
    temperature_f=72,                # Fahrenheit, a reading
    pressure_inhg=29.31,             # inHg, absolute station pressure (a reading)
    humidity_percent=50,             # percent
    altitude_feet=600,               # feet
)

# Solve trajectory
solver = TrajectorySolver(inputs, wind=wind, atmosphere=atmosphere)
result = solver.solve()

# Print results
print(f"Max range: {result.max_range_yards:.1f} yards")
print(f"Time of flight: {result.time_of_flight:.2f} seconds")
print(f"Impact velocity: {result.impact_velocity_fps:.1f} fps")
print(f"Impact energy: {result.impact_energy_ftlbs:.1f} ft-lbs")

# Iterate through trajectory points
for point in result.points:
    print(f"Time: {point.time:.2f}s, X: {point.x:.1f}yd, Y: {point.y:.3f}yd, V: {point.velocity_fps:.1f}fps")
```

## Units

The Python API uses **imperial units** for convenience:

- **Mass**: grains (gr)
- **Velocity**: feet per second (fps)
- **Distance**: yards (yd) and inches (in)
- **Pressure**: inches of mercury (inHg)
- **Temperature**: Fahrenheit (°F)
- **Wind speed**: miles per hour (mph)

All conversions to metric (used internally by the Rust engine) are handled automatically.

## API Reference

### `BallisticInputs`

Main input parameters for trajectory calculation.

**Parameters:**
- `bc` (float): Ballistic coefficient
- `bullet_weight_grains` (float): Bullet mass in grains
- `muzzle_velocity_fps` (float): Muzzle velocity in fps
- `bullet_diameter_inches` (float): Bullet diameter in inches
- `bullet_length_inches` (float): Bullet length in inches
- `sight_height_inches` (float): Sight height above bore in inches
- `zero_distance_yards` (float): Zero distance in yards
- `shooting_angle_degrees` (float): Uphill/downhill angle in degrees
- `twist_rate_inches` (float): Barrel twist rate (inches per turn)
- `is_right_twist` (bool): True for right-hand twist (default: True)

### `WindConditions`

Wind parameters.

**Parameters:**
- `speed_mph` (float): Wind speed in mph (default: 0)
- `direction_degrees` (float): Wind direction in degrees (0=headwind, 90=from right, default: 0)

### `AtmosphericConditions`

Atmospheric parameters at the shooter.

**Parameters:**
- `temperature_f` (float or None): Temperature in Fahrenheit. Omit it, or pass `None`, when you have no reading.
- `pressure_inhg` (float or None): Absolute station pressure in inHg, not a sea-level-corrected altimeter setting. Omit it, or pass `None`, when you have no reading.
- `humidity_percent` (float): Relative humidity percentage (default: 50)
- `altitude_feet` (float): Altitude in feet (default: 0)

**Readings and omitted values.** A temperature or pressure you pass is a reading and
flies exactly as given, whatever its value and whatever the altitude. A real 59 °F or
29.92 inHg at 5000 ft flies as 59 °F or 29.92 inHg. A value you leave out reads back as
59 / 29.92 but is not a reading: above 1 m of altitude it flies the ICAO standard air
at `altitude_feet` (about 41 °F and 24.9 inHg at 5000 ft).

Before engine 0.49.0 the binding could not tell the two apart. At any altitude above
1 m, a temperature within 0.1 °C of 15 °C (58.82 to 59.18 °F) or a pressure within
0.5 hPa of 1013.25 hPa (about 29.906 to 29.936 inHg) was treated as omitted and replaced
with standard air, even when it was a real reading.

`temperature_supplied` and `pressure_supplied` (bool) record which values are readings.
The constructor sets each one when you pass its value, and assigning `temperature_f` or
`pressure_inhg` sets it too. You can also set them directly. For example, a caller that
always fills in a default can pass it and then clear the flag:

```python
atmo = AtmosphericConditions(temperature_f=59.0, pressure_inhg=form_pressure, altitude_feet=5000)
atmo.temperature_supplied = False   # 59 was our placeholder, not a reading
```

### Inputs dicts (`auto_zero_inputs`, `run_monte_carlo`, `monte_carlo_parallel`)

When these functions take the atmosphere from the inputs dict, `temperature` (°C) and
`pressure` (hPa, absolute) follow the same rule. A key that is present and not `None` is a
reading. For `monte_carlo_parallel`, a value sampled per run through `param_names` is also a
reading. A missing key keeps the legacy rule. The optional bool keys `temperature_supplied`
and `pressure_supplied` override this, for a caller that always sends every key:
`{"pressure": 1013.25, "pressure_supplied": False}` flies the omitted rule, not 1013.25 hPa.
A Monte Carlo sample keeps the base dict's presence.

`calculate_zero_angle(inputs, yards, ..., temperature_f=None, pressure_inhg=None, ...)`
follows the same rule for its keyword arguments.

### `TrajectorySolver`

Trajectory calculation engine.

**Methods:**
- `__init__(inputs, wind=None, atmosphere=None)`: Create solver with inputs
- `solve()`: Calculate trajectory, returns `TrajectoryResult`

### `TrajectoryResult`

Trajectory calculation results.

**Properties:**
- `max_range_yards` (float): Maximum range in yards
- `max_height_yards` (float): Maximum height in yards
- `time_of_flight` (float): Total flight time in seconds
- `impact_velocity_fps` (float): Impact velocity in fps
- `impact_energy_ftlbs` (float): Impact energy in ft-lbs
- `points` (list[TrajectoryPoint]): List of trajectory points

### `TrajectoryPoint`

Individual point along trajectory.

**Properties:**
- `time` (float): Time in seconds
- `x` (float): Downrange distance in yards
- `y` (float): Vertical position in yards (relative to line of sight)
- `z` (float): Lateral position in yards
- `velocity_fps` (float): Velocity in fps
- `energy_ftlbs` (float): Kinetic energy in ft-lbs

### `DragModel`

Drag model selection.

**Static methods:**
- `DragModel.g1()`: G1 drag model (flat base)
- `DragModel.g7()`: G7 drag model (boat tail)
- `DragModel.g8()`: G8 drag model (boat tail with meplat)

### `bridge_call(request_json: str) -> str`

The engine's versioned JSON command bridge: one request envelope in, one response
envelope out, both as strings. It is a transport onto the engine's own service layer,
so it reaches `corrections.bc5d_table_path` and `atmosphere.pressure_reference`,
which the typed classes above cannot carry at all. It is also the safer route to
`effects.wind_shear_model`: the inputs dict has accepted `wind_shear_model` for some
time, but it takes any string and silently resolves an unrecognised one (to the power
law in `TrajectorySolver`, to no shear in `fast_integrate`), where the bridge validates
it and names the offending field.

In the inputs dict, `enable_wind_shear` without a `wind_shear_model` means the power
law (`"power_law"`, the CLI's default model); with shear off the model defaults to
`"none"`. A model you name, `"none"` included, is used exactly as given.
`BallisticInputs.wind_shear_model` reads back the resolved value.

```python
import json
from ballistics_engine import bridge_call

response = json.loads(bridge_call(json.dumps({
    "api_version": 1,
    "command": "solve",
    "request": {
        "schema_version": 1,
        "projectile": {"mass_kg": 0.01134, "diameter_m": 0.00782, "length_m": 0.0338,
                       "drag_model": "G7", "ballistic_coefficient": 0.243},
        "rifle": {"muzzle_velocity_mps": 823.0, "sight_height_m": 0.0381},
        "shot": {"max_range_m": 1000.0, "zero_distance_m": 100.0},
        "atmosphere": {}, "wind": {"speed_mps": 4.47, "direction_from_rad": 1.5708},
        "solver": {}, "effects": {"wind_shear_model": "logarithmic"},
        "sampling": {"interval_m": 100.0},
    },
})))

if response["ok"]:
    print(response["result"]["samples"][-1])
else:
    print(response["error"]["code"], response["error"]["message"])
```

Unlike the rest of this API, **failures come back inside the returned JSON rather than
as a Python exception** — a bad envelope, an unknown command or an invalid field all
return a well-formed `{"ok": false, "api_version": 1, "engine_version": "...",
"error": {"code": ..., "message": ...}}` document. The wrapper returns the string
verbatim; parsing and error handling are the caller's.

Unlike the classes above, the bridge speaks **SI units** throughout (kg, m, m/s, K, Pa,
radians), because it is the engine's own wire contract.

Ask `meta.capabilities` for the command list this wheel can actually run — it is
build-dependent, and this wheel links the engine with default features off, so the
PDF- and profile-import-gated commands are not compiled in.

## License

Dual licensed under MIT or Apache-2.0.

## Links

- **Homepage**: https://ballistics.rs/
- **Repository**: https://github.com/ajokela/ballistics-engine
- **Documentation**: https://docs.rs/ballistics-engine
