#!/usr/bin/env python3
"""
run_cycle.py — Single optimization cycle for GitHub Actions.

No daemon loop, no SQLite, no persistent process.
Just: GET state → weather → MPC → PUT commands → exit.

State between runs is persisted via a JSON file cached by GitHub Actions.
"""

import os
import sys
import json
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DT_S, DT_H, HORIZON_STEPS,
    SUNRISE, SUNSET,
    BATTERY_Wh, PV_WPEAK,
    P_ESP32, P_LIGHTS, P_PUMP,
    TARGET_C, TEMP_BAND,
    N_SCENARIOS,
    PLANT_STATES,
    TIMEZONE,
    ALPHA_SOLAR, ALPHA_VENT, ALPHA_INFIL, ALPHA_RAD, ALPHA_GND,
    T_SKY_OFFSET, T_GROUND_C,
    FLOOR_AREA, GLAZING_TRANS,
    SOIL_MAX_PUMP_RATE,
    PUMP_COOLDOWN_STEPS,
)
from greenhouse_client import get_greenhouse_state, put_commands
from mpc_solver import solve_mpc
from weather_forecast import get_weather_scenarios

STATE_FILE = os.path.join(
    os.environ.get("GITHUB_WORKSPACE", "."),
    "optimizer_state.json",
)


def load_state() -> dict:
    """Load persistent state from last run."""
    defaults = {
        "prev_interior_c": None,
        "last_pump_time_s": None,
        "cycle_count": 0,
    }
    if not os.path.exists(STATE_FILE):
        return defaults
    try:
        with open(STATE_FILE, "r") as f:
            data = json.load(f)
        defaults.update(data)
        return defaults
    except Exception:
        return defaults


def save_state(state: dict):
    """Persist state for next run."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def estimate_interior_temp(
    outside_temp_c, ghi_wm2, vent_open,
    prev_interior_c=None, dt_s=900.0, is_day=True,
):
    if prev_interior_c is None:
        prev_interior_c = outside_temp_c

    T_in = prev_interior_c
    T_out = outside_temp_c

    solar_heat_w = ghi_wm2 * FLOOR_AREA * GLAZING_TRANS * 0.15
    vent_w_per_k = 17.6 if vent_open else 3.5
    convective_w = vent_w_per_k * (T_out - T_in)

    if not is_day:
        T_sky = T_out - 10.0
        radiative_w = -2.0 * max(T_in - T_sky, 0)
    else:
        radiative_w = 0.0

    ground_w = 1.5 * (T_GROUND_C - T_in)

    C_TOTAL = 352000.0
    dT = (solar_heat_w + convective_w + radiative_w + ground_w) / C_TOTAL * dt_s
    T_new = T_in + dT
    T_new = max(T_out - 5.0, min(T_out + 15.0, T_new))
    return T_new


def run_one_cycle():
    t_start = time.time()
    saved = load_state()

    print(f"\n{'='*60}")
    print(f"  Optimization cycle — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Cycle #{saved['cycle_count'] + 1}")
    print(f"{'='*60}")

    # 1. GET current state
    state = get_greenhouse_state()
    if state is None:
        print("[cycle] ERROR: Could not fetch greenhouse state")
        return False

    plant_state_str = state.plant_state_str
    if plant_state_str not in PLANT_STATES:
        plant_state_str = "GROWING"
    soc_now = state.soc / 100.0

    print(f"  State: SOC={soc_now*100:.1f}%  "
          f"Soil=[{state.soil1},{state.soil2}] ({state.soil_moisture_pct:.0f}%)  "
          f"Plant={plant_state_str}")

    plant_params = PLANT_STATES[plant_state_str]

    # 2. Weather forecast
    ghi_scen, temp_scen, wind_scen, forecast = get_weather_scenarios(
        horizon_hours=96, n_scenarios=N_SCENARIOS,
        rng_seed=int(time.time()) % 100000,
    )
    H = min(HORIZON_STEPS, ghi_scen.shape[1])

    rain_mm_h = np.zeros(H)
    precip_prob_h = np.zeros(H)
    if forecast is not None and len(forecast.temperature) > 0:
        outside_temp_c = float(forecast.temperature[0])
        current_ghi = float(forecast.ghi[0]) if len(forecast.ghi) > 0 else 0.0
        daily_solar_kwh = forecast.daily_solar_kwh()
        if len(forecast.precipitation) >= H:
            rain_mm_h = forecast.precipitation[:H]
        if len(forecast.precip_prob) >= H:
            precip_prob_h = forecast.precip_prob[:H]
    else:
        outside_temp_c = float(temp_scen[0, 0])
        current_ghi = float(ghi_scen[0, 0])
        daily_solar_kwh = np.zeros(4)

    # 3. Estimate interior temp
    local_tz = ZoneInfo(TIMEZONE)
    now_local = datetime.now(timezone.utc).astimezone(local_tz)
    t0_hours = now_local.hour + now_local.minute / 60.0 + now_local.second / 3600.0
    is_day = SUNRISE <= (t0_hours % 24.0) <= SUNSET

    temp_now_c = estimate_interior_temp(
        outside_temp_c, current_ghi, state.vent,
        saved["prev_interior_c"], DT_S, is_day,
    )

    print(f"  Temp: interior={temp_now_c:.1f}°C  outside={outside_temp_c:.1f}°C  "
          f"GHI={current_ghi:.0f}W/m²")

    # 4. Daily totals — approximate from pump lockout logic
    # (No SQLite in GH Actions — just use the Worker state for pump timing)
    pump_used_today = state.pump  # pump minutes already used today
    pump_remaining_min = max(0, plant_params["pump_daily_min"] - pump_used_today)
    lights_remaining_h = plant_params["lights_daily_h"]  # assume fresh day

    if soc_now < 0.15:
        lights_remaining_h = 0.0
        pump_remaining_min = min(pump_remaining_min, 2.0)
    elif soc_now < 0.25:
        lights_remaining_h *= 0.5
        pump_remaining_min *= 0.5

    soil_deficit = max(0, plant_params["soil_target"] - state.soil_moisture_pct)
    if soil_deficit > 5:
        pump_needed_min = soil_deficit / SOIL_MAX_PUMP_RATE
        pump_remaining_min = max(pump_remaining_min, pump_needed_min)

    # Cooldown: if pump ran recently, skip
    pump_lockout = False
    if saved["last_pump_time_s"] is not None:
        elapsed = t_start - saved["last_pump_time_s"]
        if elapsed < PUMP_COOLDOWN_STEPS * DT_S:
            pump_lockout = True

    print(f"  MPC: SOC={soc_now*100:.1f}%  Temp={temp_now_c:.1f}°C  "
          f"pump_rem={pump_remaining_min:.1f}min  lights_rem={lights_remaining_h:.1f}h  "
          f"t0={t0_hours:.1f}h")

    # 5. Solve MPC
    try:
        result = solve_mpc(
            soc_init=soc_now,
            temp_init_c=temp_now_c,
            ghi_scenarios=ghi_scen[:, :H],
            temp_out_scenarios=temp_scen[:, :H],
            pump_remaining_min=pump_remaining_min,
            lights_remaining_h=lights_remaining_h,
            pump_daily_min=plant_params["pump_daily_min"],
            lights_daily_h=plant_params["lights_daily_h"],
            soil_moisture=state.soil_moisture_pct,
            soil_target=plant_params["soil_target"],
            soil_band=plant_params["soil_band"],
            plant_temp_target=plant_params["temp_target_c"],
            plant_temp_band=plant_params["temp_band_c"],
            t0_hours=t0_hours,
            daily_solar_kwh=daily_solar_kwh,
            pump_lockout=pump_lockout,
            rain_mm_h=rain_mm_h,
            precip_prob=precip_prob_h,
        )
    except Exception as e:
        print(f"[cycle] MPC solver error: {e}")
        import traceback; traceback.print_exc()
        return False

    # 6. Extract commands
    lights_cmd = bool(result.lights_plan[0] > 0.5) if len(result.lights_plan) > 0 else False
    vent_cmd = bool(result.vent_plan[0] > 0.5) if len(result.vent_plan) > 0 else False
    pump_cmd = float(result.pump_plan[0]) if len(result.pump_plan) > 0 else 0.0

    if pump_lockout:
        pump_cmd = 0.0

    pump_timestamp = int(time.time() * 1000)
    outside_temp_f = outside_temp_c * 9.0 / 5.0 + 32.0
    humidity_now = 50.0
    if forecast is not None and len(forecast.humidity) > 0:
        humidity_now = float(forecast.humidity[0])

    print(f"  MPC: {result.status}  cost={result.expected_cost:.2e}  "
          f"solve={result.solve_time_ms:.0f}ms")
    print(f"  Dispatch: lights={'ON' if lights_cmd else 'OFF'}  "
          f"vent={'OPEN' if vent_cmd else 'CLOSED'}  pump={pump_cmd:.1f}min")

    # 7. PUT commands
    soc_end_val = float(result.soc_trajectory[-1, 0]) if result.soc_trajectory.size > 0 else soc_now
    temp_end_val = float(result.temp_trajectory[-1]) if len(result.temp_trajectory) > 0 else temp_now_c
    import math
    cost_val = result.expected_cost
    if isinstance(cost_val, float) and math.isnan(cost_val):
        cost_val = 0.0

    mpc_data = {
        "solve_time_ms": round(result.solve_time_ms, 1),
        "status": result.status,
        "soc_end": round(soc_end_val, 4),
        "temp_end": round(temp_end_val, 2),
        "pump_total_min": round(float(np.sum(result.pump_plan)), 2),
        "lights_total_h": round(float(np.sum(result.lights_plan)) * DT_H, 2),
        "cost": round(cost_val, 4),
        "pred_soc_1step": round(soc_end_val * 100, 1),
        "pred_temp_1step_c": round(temp_end_val, 1),
        "pv_predicted_w": round(float(result.pv_trajectory[0]), 1) if len(result.pv_trajectory) > 0 else 0.0,
        "pv_actual_w": round(state.battery_voltage_out * state.battery_amps_in, 1) if state.battery_amps_in > 0 else 0.0,
    }

    success = put_commands(
        lights=lights_cmd, vent=vent_cmd, pump=pump_cmd,
        pump_timestamp=pump_timestamp,
        temperature_f=round(outside_temp_f, 1),
        humidity=round(humidity_now, 1),
        mpc_result=mpc_data,
    )

    if not success:
        print("[cycle] WARNING: PUT commands failed")

    # 8. Save state for next run
    new_state = {
        "prev_interior_c": temp_now_c,
        "last_pump_time_s": t_start if pump_cmd > 0.1 else saved.get("last_pump_time_s"),
        "cycle_count": saved["cycle_count"] + 1,
        "last_cycle_s": round(time.time() - t_start, 1),
    }
    save_state(new_state)

    print(f"\n  Cycle time: {(time.time() - t_start)*1000:.0f}ms")
    return success


if __name__ == "__main__":
    ok = run_one_cycle()
    sys.exit(0 if ok else 1)
