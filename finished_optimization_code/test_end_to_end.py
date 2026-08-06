#!/usr/bin/env python3
"""
test_end_to_end.py — Verify germinating vs growing behavior end-to-end.

Creates synthetic test images → classifies them → maps to MPC parameters
→ runs the MPC solver → compares pump/lights targets.

Two modes:
  GERMINATING (no plant): 3 min pump, 0h lights
  GROWING (has plant):    7 min pump, 6h lights
  HARVEST_READY:          same as growing + alert
"""

import sys
import os
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plant_classifier import classify_from_array, PlantState
from config import PLANT_STATES


def create_germinating_image() -> np.ndarray:
    """Synthesize a 'no plant' image: mostly brown soil, tiny green specks."""
    img = np.random.randint(60, 130, (480, 640, 3), dtype=np.uint8)  # brown soil
    # A few very tiny green dots (seedlings just starting)
    for _ in range(3):
        y = np.random.randint(200, 300)
        x = np.random.randint(250, 400)
        r = np.random.randint(2, 5)
        img[y:y+r, x:x+r] = [34, 139, 34]
    return img


def create_growing_image() -> np.ndarray:
    """Synthesize a 'growing' image: visible green plants, moderate coverage."""
    img = np.random.randint(60, 130, (480, 640, 3), dtype=np.uint8)
    # A few medium green regions (plant canopy)
    for _ in range(5):
        y = np.random.randint(50, 400)
        x = np.random.randint(50, 550)
        h = np.random.randint(40, 120)
        w = np.random.randint(40, 120)
        img[y:y+h, x:x+w] = [34, 139, 34]
    return img


def create_harvest_image() -> np.ndarray:
    """Synthesize a 'harvest ready' image: lots of green, dense canopy."""
    img = np.random.randint(40, 100, (480, 640, 3), dtype=np.uint8)
    # Fill most of the image with green
    img[30:420, 30:600] = [34, 139, 34]
    return img


def test_classifier():
    """Test classifier produces correct states."""
    print("=" * 60)
    print("Step 1: Test classifier on synthetic images")
    print("=" * 60)

    tests = [
        ("Germinating (no plant)", create_germinating_image(), PlantState.GERMINATING),
        ("Growing (has plant)", create_growing_image(), PlantState.GROWING),
        ("Harvest ready", create_harvest_image(), PlantState.HARVEST_READY),
    ]

    results = []
    for name, img, expected in tests:
        result = classify_from_array(img)
        state = result["state"]
        green = result["green_ratio"]
        area = result["plant_area_pct"]
        conf = result["confidence"]

        status = "PASS" if state == expected else "FAIL"
        print(f"  {status} {name}: state={state.value}  "
              f"green={green:.3f}  area={area:.3f}  confidence={conf:.2f}  "
              f"(expected={expected.value})")
        results.append((name, state))

    return results


def test_mpc_params():
    """Test MPC parameters for each plant state."""
    print()
    print("=" * 60)
    print("Step 2: Map plant states → MPC parameters")
    print("=" * 60)

    for state_name, params in PLANT_STATES.items():
        print(f"\n  {state_name}:")
        print(f"    Pump target:   {params['pump_daily_min']:.1f} min/day")
        print(f"    Lights target: {params['lights_daily_h']:.1f} h/day")
        print(f"    Temp target:   {params['temp_target_c']:.1f}°C (±{params['temp_band_c']:.1f}°C)")
        print(f"    Alert user:    {params['alert_user']}")
        print(f"    Description:   {params['description']}")

    # Show the key difference
    germ = PLANT_STATES["GERMINATING"]
    grow = PLANT_STATES["GROWING"]
    harvest = PLANT_STATES["HARVEST_READY"]

    print()
    print("  Key differences:")
    print(f"    Germinating vs Growing:")
    print(f"      Pump:   {germ['pump_daily_min']:.0f} → {grow['pump_daily_min']:.0f} min  "
          f"({grow['pump_daily_min']/germ['pump_daily_min']:.1f}x more)")
    print(f"      Lights: {germ['lights_daily_h']:.0f} → {grow['lights_daily_h']:.0f} h   "
          f"({'on' if grow['lights_daily_h'] > 0 else 'off'})")
    print(f"    Harvest ready = same as growing + alert user")


def test_mpc_solver():
    """Test MPC solver with both states to verify different behavior."""
    print()
    print("=" * 60)
    print("Step 3: Run MPC solver with both plant states")
    print("=" * 60)

    from weather_forecast import _synthetic_scenarios
    from mpc_solver import solve_mpc

    # Generate synthetic weather (48h, 5 scenarios)
    ghi_scen, temp_scen, _ = _synthetic_scenarios(48, 5, rng_seed=42)
    H = min(192, ghi_scen.shape[1])

    # Common initial conditions
    soc_init = 0.60  # 60% SOC
    temp_init_c = 25.0  # 25°C
    soil_moisture = 50.0  # moderate

    results = {}
    for state_name in ["GERMINATING", "GROWING", "HARVEST_READY"]:
        params = PLANT_STATES[state_name]
        print(f"\n  --- {state_name} ---")
        print(f"  Target: pump={params['pump_daily_min']:.0f}min  "
              f"lights={params['lights_daily_h']:.0f}h")

        result = solve_mpc(
            soc_init=soc_init,
            temp_init_c=temp_init_c,
            ghi_scenarios=ghi_scen[:, :H],
            temp_out_scenarios=temp_scen[:, :H],
            pump_remaining_min=params["pump_daily_min"],
            lights_remaining_h=params["lights_daily_h"],
            pump_daily_min=params["pump_daily_min"],
            lights_daily_h=params["lights_daily_h"],
            soil_moisture=soil_moisture,
            plant_temp_target=params["temp_target_c"],
            plant_temp_band=params["temp_band_c"],
            t0_hours=14.0,  # 2 PM (daytime)
            verbose=True,
        )
        results[state_name] = result

    # Compare results
    print()
    print("=" * 60)
    print("Step 4: Compare solver outputs")
    print("=" * 60)

    for state_name, result in results.items():
        pump_total = float(np.sum(result.pump_plan))
        lights_total = float(np.sum(result.lights_plan)) * 0.25
        print(f"  {state_name}:")
        print(f"    Pump total:   {pump_total:.1f} min")
        print(f"    Lights total: {lights_total:.1f} h")
        print(f"    Status:       {result.status}")
        print(f"    Cost:         {result.expected_cost:.2e}")


def main():
    print("End-to-End Test: Germinating vs Growing Behavior")
    print()

    # Step 1: Classifier
    classifier_results = test_classifier()

    # Step 2: MPC parameters
    test_mpc_params()

    # Step 3: MPC solver
    test_mpc_solver()

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print()
    print("  Two operational modes:")
    print("    1. GERMINATING (no plant): 3 min pump, 0h lights")
    print("    2. GROWING (has plant):    7 min pump, 6h lights")
    print("    3. HARVEST_READY:          same as growing + alert")
    print()
    print("  The classifier detects plant presence from the camera image.")
    print("  The MPC schedules WHEN to run pump/lights, not WHETHER.")
    print("  When harvest-ready, the daemon alerts the user.")


if __name__ == "__main__":
    main()
