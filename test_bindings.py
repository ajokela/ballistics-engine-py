#!/usr/bin/env python3
"""Test script for ballistics-engine Python bindings"""

try:
    from ballistics_engine import (
        BallisticInputs, TrajectorySolver, WindConditions,
        AtmosphericConditions, DragModel
    )

    print("✓ Successfully imported ballistics_engine")
    print(f"  Module location: {BallisticInputs.__module__}")

    # Create ballistic inputs
    inputs = BallisticInputs(
        bc=0.505,
        bullet_weight_grains=168,
        muzzle_velocity_fps=2650,
        bullet_diameter_inches=0.308,
        bullet_length_inches=1.24,
        sight_height_inches=1.5,
        zero_distance_yards=100,
        twist_rate_inches=11.25,
    )
    inputs.drag_model = DragModel.g7()

    print(f"✓ Created BallisticInputs: {inputs}")

    # Create wind conditions
    wind = WindConditions(speed_mph=10, direction_degrees=90)
    print(f"✓ Created WindConditions: {wind.speed_mph} mph at {wind.direction_degrees}°")

    # Create atmospheric conditions
    atmo = AtmosphericConditions(
        temperature_f=59,
        pressure_inhg=29.92,
        humidity_percent=50,
        altitude_feet=0
    )
    print(f"✓ Created AtmosphericConditions: {atmo.temperature_f}°F, {atmo.pressure_inhg} inHg")

    # Solve trajectory
    solver = TrajectorySolver(inputs, wind=wind, atmosphere=atmo)
    print("✓ Created TrajectorySolver")

    result = solver.solve()
    print("✓ Solved trajectory successfully!")
    print(f"  Max range: {result.max_range_yards:.1f} yards")
    print(f"  Time of flight: {result.time_of_flight:.2f} seconds")
    print(f"  Impact velocity: {result.impact_velocity_fps:.1f} fps")
    print(f"  Impact energy: {result.impact_energy_ftlbs:.1f} ft-lbs")
    print(f"  Number of trajectory points: {len(result.points)}")

    # Check a few trajectory points
    if len(result.points) > 0:
        print("\nFirst few trajectory points:")
        for i, pt in enumerate(result.points[:5]):
            print(f"  Point {i}: t={pt.time:.3f}s, x={pt.x:.1f}yd, y={pt.y:.3f}yd, v={pt.velocity_fps:.1f}fps")

    print("\n✅ All tests passed!")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Install the wheel first: pip install target/wheels/ballistics_engine-*.whl")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
