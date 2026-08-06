#!/usr/bin/env python3
"""
optimizer_daemon.py — Main greenhouse optimizer loop.

Runs every 15 minutes:
  1. GET current state from Cloudflare Worker (SOC, soil, plant_state)
  2. Fetch 2-day weather forecast (outside temp, solar irradiance)
  3. Estimate greenhouse interior temperature from weather + thermal model
  4. Solve stochastic MPC
  5. PUT first-step commands
  6. Log everything (SQLite + CSV)

Plant state is set by the user from the website (not from camera).
Interior temperature is estimated from outside weather since the
temperature sensor is broken.
"""

import sys
import os
import time
import signal
import argparse
import traceback
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
from mpc_solver import solve_mpc, MPCResult
from weather_forecast import get_weather_scenarios
from data_logger import DataLogger
from model_tracker import ModelTracker
from soc_calibrator import SoCCalibrator

_running = True


def _signal_handler(sig, frame):
    global _running
    print("\n[daemon] Shutting down gracefully...")
    _running = False


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def _now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _get_plant_params(plant_state_str: str) -> dict:
    """Get MPC parameters for the current plant state."""
    return PLANT_STATES.get(plant_state_str, PLANT_STATES["GROWING"])


def estimate_interior_temp(
    outside_temp_c: float,
    ghi_wm2: float,
    vent_open: bool,
    prev_interior_c: float = None,
    dt_s: float = 900.0,
    is_day: bool = True,
) -> float:
    """Estimate greenhouse interior temperature from outside weather.

    Simple thermal model for a 6x4 ft polycarbonate hobby greenhouse:
    - Solar gain through glazing heats the interior
    - Ventilation (vents open) brings interior toward outside temp
    - Infiltration (gaps) provides baseline heat exchange
    - Radiative cooling at night

    Returns estimated interior temperature in °C.
    """
    if prev_interior_c is None:
        prev_interior_c = outside_temp_c

    T_in = prev_interior_c
    T_out = outside_temp_c

    # Solar gain: GHI * effective area * glazing transmission
    # Only a fraction of solar energy heats the air (rest absorbed by soil/structure)
    solar_heat_w = ghi_wm2 * FLOOR_AREA * GLAZING_TRANS * 0.15  # 15% heats air directly

    # Ventilation: when open, strong coupling to outside
    # When closed, only infiltration (weak coupling)
    if vent_open:
        vent_w_per_k = 17.6  # W/K with vents open (15 ACH)
    else:
        vent_w_per_k = 3.5   # W/K infiltration only (3 ACH)

    # Heat loss/gain from通风 + infiltration
    convective_w = vent_w_per_k * (T_out - T_in)

    # Radiative cooling at night (simplified)
    if not is_day:
        T_sky = T_out - 10.0
        radiative_w = -2.0 * max(T_in - T_sky, 0)
    else:
        radiative_w = 0.0

    # Ground coupling
    ground_w = 1.5 * (T_GROUND_C - T_in)

    # Total heat input (W) → temperature change
    # C_TOTAL ≈ 352,000 J/K
    C_TOTAL = 352000.0
    dT = (solar_heat_w + convective_w + radiative_w + ground_w) / C_TOTAL * dt_s

    T_new = T_in + dT

    # Clamp to reasonable range: at most 15°C above/outside, at most 5°C below outside
    T_new = max(T_out - 5.0, min(T_out + 15.0, T_new))

    return T_new


def run_cycle(logger: DataLogger, tracker: ModelTracker = None,
              soc_calibrator: SoCCalibrator = None,
              verbose: bool = True,
              cycle_count: int = 1, prev_interior_c: float = None,
              prev_update_ts: float = 0.0,
              pump_lockout: bool = False,
              prev_mpc_result=None,
              last_pump_time_s: float = None) -> tuple[bool, float, float, object, float]:
    """Run one optimization cycle.

    Returns (success, estimated_interior_temp_c, current_soc_for_next_cycle).
    """
    t_start = time.time()

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Optimization cycle — {_now_str()}")
        print(f"{'='*60}")

    # ── 1. GET current state ────────────────────────────────────────────────
    state = get_greenhouse_state()
    if state is None:
        print("[daemon] ERROR: Could not fetch greenhouse state")
        return False, prev_interior_c or TARGET_C, 0.5, None, 0.0

    # Plant state from user (website), not from camera
    plant_state_str = state.plant_state_str
    if plant_state_str not in PLANT_STATES:
        plant_state_str = "GROWING"

    soc_now = state.soc / 100.0

    # Apply SOC calibration if calibrator is available
    if soc_calibrator is not None:
        dt_between = t_start - prev_update_ts if prev_update_ts > 0 else 900.0
        soc_now = soc_calibrator.update(
            soc_raw_pct=state.soc,
            battery_amps_in=state.battery_amps_in,
            battery_amps_out=state.battery_amps_out,
            battery_voltage_out=state.battery_voltage_out,
            dt_s=dt_between,
        ) / 100.0
        cal_status = soc_calibrator.get_status()
    else:
        cal_status = None

    if verbose:
        print(f"  State: SOC={soc_now*100:.1f}%  "
              f"Soil=[{state.soil1},{state.soil2}] ({state.soil_moisture_pct:.0f}%)  "
              f"Plant={plant_state_str}  "
              f"Lights={'ON' if state.lights else 'OFF'}  "
              f"Vent={'ON' if state.vent else 'OFF'}  "
              f"Pump={state.pump:.1f}min")

    plant_params = _get_plant_params(plant_state_str)

    if verbose:
        print(f"  Target: {plant_params['description']}")
        if cal_status and not cal_status["calibrated"]:
            print(f"  SOC calibrator: tracking {cal_status['tracking_soc_pct']:.1f}% "
                  f"(raw={state.soc:.1f}%, not yet calibrated)")

    # ── 1b. Log model prediction error ──────────────────────────────────────
    if tracker is not None:
        batt_amps = (state.battery_amps_in - state.battery_amps_out)
        error_row = tracker.log_cycle_error(
            soc_observed=soc_now,
            temp_observed_c=prev_interior_c or TARGET_C,
            battery_amps_avg=batt_amps,
        )
        if error_row and verbose:
            print(f"  Model error: SOC={error_row['soc_error']:+.1f}%  "
                  f"Temp={error_row['temp_error']:+.1f}°C")

    # ── 2. Weather forecast ─────────────────────────────────────────────────
    ghi_scen, temp_scen, wind_scen, forecast = get_weather_scenarios(
        horizon_hours=96,
        n_scenarios=N_SCENARIOS,
        rng_seed=int(time.time()) % 100000,
    )
    H = min(HORIZON_STEPS, ghi_scen.shape[1])

    # Get current outside temperature and solar estimate from forecast
    rain_mm_h = np.zeros(H)
    precip_prob_h = np.zeros(H)
    if forecast is not None and len(forecast.temperature) > 0:
        outside_temp_c = float(forecast.temperature[0])
        current_ghi = float(forecast.ghi[0]) if len(forecast.ghi) > 0 else 0.0
        daily_solar_kwh = forecast.daily_solar_kwh()
        # Downsample rain forecast to solver horizon
        if len(forecast.precipitation) >= H:
            rain_mm_h = forecast.precipitation[:H]
        if len(forecast.precip_prob) >= H:
            precip_prob_h = forecast.precip_prob[:H]
    else:
        # Fallback: use first scenario
        outside_temp_c = float(temp_scen[0, 0])
        current_ghi = float(ghi_scen[0, 0])
        daily_solar_kwh = np.zeros(4)

    # ── 3. Estimate interior temperature ────────────────────────────────────
    local_tz = ZoneInfo(TIMEZONE)
    now_local = datetime.now(timezone.utc).astimezone(local_tz)
    t0_hours = now_local.hour + now_local.minute / 60.0 + now_local.second / 3600.0
    is_day = SUNRISE <= (t0_hours % 24.0) <= SUNSET

    temp_now_c = estimate_interior_temp(
        outside_temp_c=outside_temp_c,
        ghi_wm2=current_ghi,
        vent_open=state.vent,
        prev_interior_c=prev_interior_c,
        dt_s=DT_S,
        is_day=is_day,
    )

    if verbose:
        print(f"  Temp estimate: interior={temp_now_c:.1f}°C  "
              f"outside={outside_temp_c:.1f}°C  "
              f"vent={'OPEN' if state.vent else 'CLOSED'}  "
              f"GHI={current_ghi:.0f}W/m²")

    if verbose:
        peak_ghi = float(np.max(ghi_scen))
        print(f"  Weather: {H} steps, peak GHI={peak_ghi:.0f} W/m², "
              f"temp range=[{temp_scen.min():.1f}, {temp_scen.max():.1f}]°C")

    # ── 4. Solve MPC ────────────────────────────────────────────────────────
    daily_totals = logger.get_today_totals()
    pump_used_today = daily_totals["pump_min"]
    lights_used_today = daily_totals["lights_h"]

    pump_remaining_min = max(0, plant_params["pump_daily_min"] - pump_used_today)
    lights_remaining_h = max(0, plant_params["lights_daily_h"] - lights_used_today)

    # SOC safety overrides
    if soc_now < 0.15:
        lights_remaining_h = 0.0
        pump_remaining_min = min(pump_remaining_min, 2.0)
    elif soc_now < 0.25:
        lights_remaining_h *= 0.5
        pump_remaining_min *= 0.5

    plant_temp_target = plant_params["temp_target_c"]
    plant_temp_band = plant_params["temp_band_c"]
    soil_target = plant_params["soil_target"]
    soil_band = plant_params["soil_band"]

    # Soil safety override: when critically dry, boost pump budget so solver
    # isn't penalized for emergency watering
    soil_deficit = max(0, soil_target - state.soil_moisture_pct)
    if soil_deficit > 5:
        pump_needed_min = soil_deficit / SOIL_MAX_PUMP_RATE
        pump_remaining_min = max(pump_remaining_min, pump_needed_min)

    # One-time pump lockout: no pumping until next day
    if pump_lockout:
        pump_remaining_min = 0.0

    if verbose:
        print(f"  MPC input: SOC={soc_now*100:.1f}%  Temp={temp_now_c:.1f}°C  "
              f"Soil={state.soil_moisture_pct:.0f}% (target={soil_target:.0f}±{soil_band:.0f}%)  "
              f"Pump={pump_used_today:.1f}/{plant_params['pump_daily_min']:.0f}min used  "
              f"remaining={pump_remaining_min:.1f}min  "
              f"Lights={lights_used_today:.1f}/{plant_params['lights_daily_h']:.0f}h used  "
              f"remaining={lights_remaining_h:.1f}h  "
              f"t0={t0_hours:.1f}h ({now_local.strftime('%H:%M %Z')})")

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
            soil_target=soil_target,
            soil_band=soil_band,
            plant_temp_target=plant_temp_target,
            plant_temp_band=plant_temp_band,
            t0_hours=t0_hours,
            daily_solar_kwh=daily_solar_kwh,
            pump_lockout=pump_lockout,
            rain_mm_h=rain_mm_h,
            precip_prob=precip_prob_h,
            verbose=verbose,
            prev_result=prev_mpc_result,
        )
    except Exception as e:
        print(f"[daemon] MPC solver error: {e}")
        traceback.print_exc()
        return False, temp_now_c, soc_now, None, 0.0

    # ── 5. Extract first-step commands and PUT ──────────────────────────────
    lights_cmd = bool(result.lights_plan[0] > 0.5) if len(result.lights_plan) > 0 else False
    vent_cmd = bool(result.vent_plan[0] > 0.5) if len(result.vent_plan) > 0 else False
    pump_cmd = float(result.pump_plan[0]) if len(result.pump_plan) > 0 else 0.0

    # Hard override: pump lockout kills all pumping regardless of solver
    if pump_lockout:
        pump_cmd = 0.0

    # Cooldown override: if pump ran recently, zero the command.
    # This enforces percolation wait across daemon cycles.
    if last_pump_time_s is not None:
        elapsed_since_pump = t_start - last_pump_time_s
        cooldown_s = PUMP_COOLDOWN_STEPS * DT_S  # e.g. 4 steps × 900s = 1 hour
        if elapsed_since_pump < cooldown_s:
            remaining_cd = cooldown_s - elapsed_since_pump
            if pump_cmd > 0 and verbose:
                print(f"  COOLDOWN: pump suppressed, {remaining_cd/60:.1f}min remaining")
            pump_cmd = 0.0

    pump_timestamp = int(time.time() * 1000)

    # 1-step predictions (sanitize NaN from solver)
    def _safe(v, fallback):
        import math
        return fallback if (isinstance(v, float) and math.isnan(v)) else v

    if result.soc_trajectory.shape[0] > 1 and result.soc_trajectory.shape[1] > 0:
        soc_1step = _safe(float(result.soc_trajectory[1, 0]), soc_now)
    else:
        soc_1step = soc_now

    if len(result.temp_trajectory) > 1:
        temp_1step_c = _safe(float(result.temp_trajectory[1]), temp_now_c)
    else:
        temp_1step_c = temp_now_c

    if len(result.soil_trajectory) > 1:
        soil_1step = _safe(float(result.soil_trajectory[1]), state.soil_moisture_pct)
    else:
        soil_1step = state.soil_moisture_pct

    pump_total_plan = float(np.sum(result.pump_plan))
    lights_total_plan = float(np.sum(result.lights_plan)) * DT_H

    if tracker is not None:
        tracker.store_prediction(
            soc_predicted_1step=soc_1step,
            temp_predicted_1step_c=temp_1step_c,
            pump_planned_total_min=pump_total_plan,
            lights_planned_total_h=lights_total_plan,
            pump_dispatched_min=pump_cmd,
            lights_dispatched_h=DT_H if lights_cmd else 0.0,
        )

    if verbose:
        print(f"  MPC result: {result.status}  cost={result.expected_cost:.2e}  "
              f"solve={result.solve_time_ms:.0f}ms")
        print(f"  Plan: lights={lights_total_plan:.1f}h  pump={pump_total_plan:.1f}min  "
              f"SOC_1step={soc_1step*100:.1f}%  Temp_1step={temp_1step_c:.1f}°C  "
              f"Soil_1step={soil_1step:.0f}%")
        print(f"  Dispatch: lights={'ON' if lights_cmd else 'OFF'}  "
              f"vent={'OPEN' if vent_cmd else 'CLOSED'}  "
              f"pump={pump_cmd:.1f}min")

    # GET current humidity from forecast for website display
    humidity_now = 50.0  # default
    if forecast is not None and len(forecast.humidity) > 0:
        humidity_now = float(forecast.humidity[0])

    # Push API weather data to Worker so website shows real values
    outside_temp_f = outside_temp_c * 9.0 / 5.0 + 32.0

    # Build MPC result dict for D1 logging (sanitize NaN)
    soc_end_val = float(result.soc_trajectory[-1, 0]) if result.soc_trajectory.size > 0 else soc_now
    temp_end_val = float(result.temp_trajectory[-1]) if len(result.temp_trajectory) > 0 else temp_now_c
    cost_val = result.expected_cost
    import math
    if isinstance(cost_val, float) and math.isnan(cost_val):
        cost_val = 0.0
    mpc_data = {
        "solve_time_ms": round(result.solve_time_ms, 1),
        "status": result.status,
        "soc_end": round(_safe(soc_end_val, soc_now), 4),
        "temp_end": round(_safe(temp_end_val, temp_now_c), 2),
        "pump_total_min": round(pump_total_plan, 2),
        "lights_total_h": round(lights_total_plan, 2),
        "cost": round(cost_val, 4),
        "pred_soc_1step": round(soc_1step * 100, 1),
        "pred_temp_1step_c": round(temp_1step_c, 1),
        "pred_soil_1step": round(_safe(soil_1step, state.soil_moisture_pct), 1),
        "pv_predicted_w": round(_safe(float(result.pv_trajectory[0]), 0.0), 1) if len(result.pv_trajectory) > 0 else 0.0,
        "pv_actual_w": round(state.battery_voltage_out * state.battery_amps_in, 1) if state.battery_amps_in > 0 else 0.0,
        "pv_daily_kwh": round(float(daily_solar_kwh[0]), 3) if len(daily_solar_kwh) > 0 else 0.0,
        "cloud_cover": round(float(forecast.cloud_cover[0]), 0) if forecast is not None and len(forecast.cloud_cover) > 0 else None,
    }

    # PUT commands to the Worker
    success = put_commands(
        lights=lights_cmd,
        vent=vent_cmd,
        pump=pump_cmd,
        pump_timestamp=pump_timestamp,
        temperature_f=round(outside_temp_f, 1),
        humidity=round(humidity_now, 1),
        mpc_result=mpc_data,
        take_picture=False,
    )

    if not success:
        print("[daemon] WARNING: PUT commands failed")

    # ── 6. Log everything ───────────────────────────────────────────────────
    cycle_ms = (time.time() - t_start) * 1000.0

    # Plant result placeholder (no camera classification)
    plant_result = {"green_ratio": 0.0, "plant_area_pct": 0.0, "confidence": 0.0}

    logger.log_observation(state, plant_result, weather_source="open-meteo")
    logger.log_action(lights_cmd, vent_cmd, pump_cmd, pump_timestamp)
    logger.log_system(state.raw if hasattr(state, "raw") else {}, result, cycle_ms)
    logger.log_weather_scenario(ghi_scen[:, :H], temp_scen[:, :H])

    # CSV row
    now_utc = datetime.now(timezone.utc)
    csv_row = {
        "timestamp_ms": int(time.time() * 1000),
        "datetime_utc": now_utc.strftime("%Y-%m-%d %H:%M:%S"),
        "soc": round(soc_now * 100, 1),
        "soc_raw": round(state.soc, 1),
        "soc_calibrated": int(cal_status["calibrated"]) if cal_status else 1,
        "temperature_f": round(outside_temp_f, 1),
        "temperature_c": round(outside_temp_c, 2),
        "humidity": round(humidity_now, 1),
        "soil1": state.soil1,
        "soil2": state.soil2,
        "soil_moisture": round(state.soil_moisture_pct, 1),
        "soil_mpc_end": round(float(result.soil_trajectory[-1]), 1) if len(result.soil_trajectory) > 0 else state.soil_moisture_pct,
        "soil1": state.soil1,
        "soil2": state.soil2,
        "battery_voltage": round(state.battery_voltage_out, 2),
        "battery_amps_in": round(state.battery_amps_in, 3),
        "battery_amps_out": round(state.battery_amps_out, 3),
        "plant_state": plant_state_str,
        "green_ratio": 0.0,
        "lights": int(lights_cmd),
        "vent": int(vent_cmd),
        "pump_minutes": round(pump_cmd, 2),
        "mpc_soc_end": round(float(result.soc_trajectory[-1, 0]) * 100, 1) if result.soc_trajectory.size > 0 else 0.0,
        "mpc_temp_end": round(float(result.temp_trajectory[-1]), 2) if len(result.temp_trajectory) > 0 else 0.0,
        "mpc_solve_ms": round(result.solve_time_ms, 1),
        "mpc_status": result.status,
        "pv_estimate_w": round(float(result.pv_trajectory[0]), 1) if len(result.pv_trajectory) > 0 else 0.0,
        "pv_actual_w": round(state.battery_voltage_out * state.battery_amps_in, 1) if state.battery_amps_in > 0 else 0.0,
        "pv_daily_kwh": round(float(daily_solar_kwh[0]), 3) if len(daily_solar_kwh) > 0 else 0.0,
        "load_estimate_w": round(P_ESP32 + (P_LIGHTS if lights_cmd else 0) + (P_PUMP * pump_cmd / 15.0 if pump_cmd > 0 else 0), 2),
        "harvest_alert": int(plant_params.get("alert_user", False)),
    }
    logger.log_csv_row(csv_row)

    if verbose:
        print(f"  Logged. Cycle time: {cycle_ms:.0f}ms. "
              f"Total observations: {logger.get_observation_count()}")

    return True, temp_now_c, soc_now, result, pump_cmd


def main():
    parser = argparse.ArgumentParser(description="Greenhouse Optimizer Daemon")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    parser.add_argument("--interval", type=int, default=int(DT_S),
                        help=f"Seconds between cycles (default: {int(DT_S)})")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--summary", action="store_true",
                        help="Print model error summary and exit")
    parser.add_argument("--soc-init", type=float, default=50.0,
                        help="Initial SOC%% for calibrator (default: 50)")
    parser.add_argument("--reset-cal", action="store_true",
                        help="Reset SOC calibrator and start fresh")
    parser.add_argument("--pump-lockout", action="store_true",
                        help="One-time: no pumping until next day")
    args = parser.parse_args()

    verbose = args.verbose and not args.quiet

    logger = DataLogger()
    tracker = ModelTracker()

    if args.summary:
        from model_tracker import print_daily_summary
        print_daily_summary()
        return

    if args.once:
        cal = SoCCalibrator(initial_soc=args.soc_init)
        success, _, _, _, _ = run_cycle(logger, tracker=tracker, soc_calibrator=cal, verbose=verbose, pump_lockout=args.pump_lockout, last_pump_time_s=None)
        return

    soc_calibrator = SoCCalibrator(initial_soc=args.soc_init)
    if args.reset_cal:
        soc_calibrator.reset(initial_soc=args.soc_init)
        print(f"[daemon] SOC calibrator reset to {args.soc_init:.1f}%")

    print(f"Greenhouse optimizer daemon starting")
    print(f"  Interval: {args.interval}s ({args.interval / 60:.0f} min)")
    print(f"  Database: {logger.db_path}")
    print(f"  CSV dir:  {logger.csv_dir}")
    print(f"  Model tracking: {tracker.db_path}")
    print(f"  SOC calibrator: initial={args.soc_init:.1f}%")
    if args.pump_lockout:
        print(f"  Pump lockout: ACTIVE (no pumping until restart)")
    print(f"  Plant state: from website (user input)")
    print(f"  Temperature: estimated from weather + thermal model")
    print(f"  Press Ctrl+C to stop\n")

    cycle_count = 0
    prev_interior_c = None
    prev_update_ts = time.time()
    prev_mpc_result = None
    last_pump_time_s = None

    while _running:
        try:
            cycle_count += 1
            if verbose:
                print(f"\n--- Cycle {cycle_count} ---")

            success, prev_interior_c, soc_now, prev_mpc_result, actual_pump_cmd = run_cycle(
                logger, tracker=tracker, soc_calibrator=soc_calibrator,
                verbose=verbose,
                cycle_count=cycle_count, prev_interior_c=prev_interior_c,
                prev_update_ts=prev_update_ts,
                pump_lockout=args.pump_lockout,
                prev_mpc_result=prev_mpc_result,
                last_pump_time_s=last_pump_time_s,
            )
            prev_update_ts = time.time()

            # Track pump timing for cooldown across cycles
            if success and actual_pump_cmd > 0.1:
                last_pump_time_s = time.time()

            if not success:
                print("[daemon] Cycle failed, will retry next interval")

            if cycle_count % 96 == 0:
                from model_tracker import print_daily_summary
                print_daily_summary(n_days=1)

        except Exception as e:
            print(f"[daemon] Unexpected error: {e}")
            traceback.print_exc()

        if not _running:
            break

        next_time = time.time() + args.interval
        while _running and time.time() < next_time:
            time.sleep(1.0)

    print(f"\n[daemon] Stopped after {cycle_count} cycles")


if __name__ == "__main__":
    main()
