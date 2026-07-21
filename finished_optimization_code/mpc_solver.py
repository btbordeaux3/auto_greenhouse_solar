"""
mpc_solver.py - Stochastic MPC for greenhouse optimization (vectorized).

Uses CasADi + IPOPT over a 2-day horizon with hybrid resolution:
  - First 24h: 15-min steps (96 steps)
  - Next 24h: hourly steps (24 steps)
  - Total: 168 steps, 12 weather scenarios

States:
  - SOC (battery state of charge, 0-1)
  - T (interior temperature, °C)
  - Soil (soil moisture, 0-100%)

Design principles:
  - Pump and lights are REQUIRED loads (from plant state) — the MPC
    schedules WHEN to run them, not WHETHER to run them.
  - Soil moisture is tracked via a simple bucket model (pump adds water,
    evapotranspiration removes it, drainage above field capacity).
  - Lights double as SOC management — when SOC is high, running lights
    absorbs excess solar energy AND helps plants.
  - Daytime watering preferred (plants can use it, avoid night waste).
"""

import numpy as np
import casadi as ca
import time as _time
from dataclasses import dataclass
from typing import Optional

from config import (
    BATTERY_Wh, BATTERY_EFF,
    BATTERY_MAX_CHARGE_W, BATTERY_MAX_DISCHARGE_W,
    SOC_MIN, SOC_MAX, SOC_TARGET,
    P_LIGHTS, P_PUMP, P_ESP32,
    PUMP_MAX_MINUTES,
    PV_WPEAK, PV_MPPT,
    DT_S, N_FINE, N_COARSE, N_HYBRID,
    TARGET_C, TEMP_BAND,
    ALPHA_SOLAR, ALPHA_VENT, ALPHA_INFIL, ALPHA_RAD, ALPHA_INT, ALPHA_GND,
    T_SKY_OFFSET, T_GROUND_C,
    SUNRISE, SUNSET,
    W_SOC_CRIT, W_SOC_FULL, W_SOC_TRACK, W_SOC_FORECAST, W_TEMP,
    W_BIN_LIGHTS, W_BIN_VENT, W_VENT_MOVE, W_RAMP,
    W_SLACK_SOC, W_SLACK_TEMP, W_PUMP_ENERGY,
    W_REQUIRE, W_LIGHTS_ABSORB,
    SOIL_FIELD_CAPACITY, SOIL_WILTING_POINT,
    SOIL_MAX_PUMP_RATE, SOIL_ET_RATE, SOIL_ET_NIGHT_SCALE,
    SOIL_DRAIN_RATE, SOIL_INIT_DEFAULT,
    W_SOIL_TRACK, W_SOIL_DRY, W_SOIL_WET,
    W_PUMP_DAY_BONUS, W_PUMP_NIGHT_PENALTY,
    W_PUMP_CLOSE,
    W_VENT_DAY_BONUS,
    MPC_VERBOSE, MPC_MAX_ITER, MPC_TOL,
    N_SCENARIOS, PUMP_COOLDOWN_STEPS,
)


@dataclass
class MPCResult:
    lights_plan: np.ndarray
    vent_plan: np.ndarray
    pump_plan: np.ndarray
    soc_trajectory: np.ndarray
    temp_trajectory: np.ndarray
    soil_trajectory: np.ndarray
    expected_cost: float
    solve_time_ms: float
    status: str
    pv_trajectory: np.ndarray    # predicted PV watts at each step


def _build_hybrid_dt():
    dt = np.full(N_HYBRID, 3600.0)
    dt[:N_FINE] = DT_S
    return dt


def _downsample_scenarios(ghi_192, temp_192):
    S = ghi_192.shape[0]
    ghi_fine = ghi_192[:, :N_FINE]
    ghi_coarse = ghi_192[:, N_FINE:].reshape(S, N_COARSE, 4).mean(axis=2)
    ghi = np.hstack([ghi_fine, ghi_coarse])
    temp_fine = temp_192[:, :N_FINE]
    temp_coarse = temp_192[:, N_FINE:].reshape(S, N_COARSE, 4).mean(axis=2)
    temp = np.hstack([temp_fine, temp_coarse])
    return ghi, temp


def _day_night_flags(t0_hours, dt_arr):
    H = len(dt_arr)
    is_day = np.zeros(H)
    is_night = np.zeros(H)
    hour = t0_hours % 24.0
    for t in range(H):
        if SUNRISE <= hour <= SUNSET:
            is_day[t] = 1.0
        else:
            is_night[t] = 1.0
        hour = (hour + dt_arr[t] / 3600.0) % 24.0
    return is_day, is_night


def _pump_time_mask(t0_hours, dt_arr, pump_start_h=9.0, pump_end_h=21.0):
    """Return boolean mask: True when pump is allowed (9am-9pm default)."""
    H = len(dt_arr)
    mask = np.zeros(H)
    hour = t0_hours % 24.0
    for t in range(H):
        if pump_start_h <= hour < pump_end_h:
            mask[t] = 1.0
        hour = (hour + dt_arr[t] / 3600.0) % 24.0
    return mask


def solve_mpc(
    soc_init,
    temp_init_c,
    ghi_scenarios,
    temp_out_scenarios,
    pump_remaining_min,
    lights_remaining_h,
    pump_daily_min=None,
    lights_daily_h=None,
    soil_moisture=SOIL_INIT_DEFAULT,
    soil_target=70.0,
    soil_band=10.0,
    plant_temp_target=TARGET_C,
    plant_temp_band=TEMP_BAND,
    pump_max_min=PUMP_MAX_MINUTES,
    scenario_weights=None,
    t0_hours=0.0,
    daily_solar_kwh=None,
    pump_lockout=False,
    rain_mm_h=None,        # hourly rain forecast (mm/h), downsampled to horizon
    precip_prob=None,      # hourly precipitation probability (0-100)
    verbose=MPC_VERBOSE,
    prev_result=None,      # previous MPCResult for warm-starting
):
    t0 = _time.perf_counter()
    S = ghi_scenarios.shape[0]
    H = N_HYBRID

    # Pump lockout: zero the budget so solver doesn't fight itself.
    # Also disable soil cost terms — can't fix soil without pump,
    # so penalizing it causes IPOPT to stall on an impossible gradient.
    if pump_lockout:
        pump_remaining_min = 0.0
        pump_max_min = 0.0

    ghi_h, temp_h = _downsample_scenarios(ghi_scenarios, temp_out_scenarios)

    if scenario_weights is None:
        scenario_weights = np.ones(S) / S

    temp_out_avg = np.mean(temp_h, axis=0)
    ghi_avg = np.mean(ghi_h, axis=0)
    dt_arr = _build_hybrid_dt()
    dt_h_arr = dt_arr / 3600.0
    is_day, is_night = _day_night_flags(t0_hours, dt_arr)
    pump_time_mask = _pump_time_mask(t0_hours, dt_arr)  # 1 when pump allowed (9am-9pm)

    dt_h = ca.DM(dt_h_arr.reshape(-1, 1))
    dt_s = ca.DM(dt_arr.reshape(-1, 1))
    is_day_v = ca.DM(is_day.reshape(-1, 1))
    is_night_v = ca.DM(is_night.reshape(-1, 1))
    pump_time_v = ca.DM(pump_time_mask.reshape(-1, 1))  # 0/1 mask for allowed pump hours
    temp_out_v = ca.DM(temp_out_avg.reshape(-1, 1))
    w_v = ca.DM(scenario_weights.reshape(-1, 1))

    ghi_cas = ca.DM(ghi_h).T

    opti = ca.Opti()

    lights = opti.variable(H)
    vent = opti.variable(H)
    pump = opti.variable(H)
    SoC = opti.variable(H + 1, S)
    T = opti.variable(H + 1)
    Soil = opti.variable(H + 1)
    slack_soc = opti.variable(H, S)
    slack_temp = opti.variable(H)
    slack_soil = opti.variable(H)

    opti.subject_to(opti.bounded(0, lights, 1))
    opti.subject_to(opti.bounded(0, vent, 1))
    pump_max_step = min(pump_max_min, 20.0)  # hard cap: 20 min per step
    # Pump bounds: 0 to pump_max_step, but 0 outside 9am-9pm window
    pump_upper = np.minimum(pump_max_step, pump_max_step * pump_time_mask)
    opti.subject_to(opti.bounded(0, pump, pump_upper))
    opti.subject_to(ca.vec(slack_soc) >= 0)
    opti.subject_to(slack_temp >= 0)
    opti.subject_to(slack_soil >= 0)

    opti.subject_to(lights <= is_night_v)

    pump_frac = pump / (dt_s / 60.0)
    load_W = lights * P_LIGHTS + pump_frac * P_PUMP + P_ESP32

    # ── Pump target (fixed daily budget) ──────────────────────────────────────
    pump_horizon_target = pump_remaining_min

    # Lights: target the full remaining (today + tomorrow) since lights
    # affect plant growth which is cumulative over the horizon
    if lights_daily_h is not None:
        lights_horizon_target = lights_remaining_h + lights_daily_h
    else:
        lights_horizon_target = lights_remaining_h * 2.0

    # ── SOC dynamics ─────────────────────────────────────────────────────────
    pv_W = PV_WPEAK * ca.fmax(ghi_cas, 0) / 1000.0 * PV_MPPT
    load_rep = ca.repmat(load_W, 1, S)
    net_W = pv_W - load_rep

    SoC_curr = SoC[:-1, :]
    SoC_next = SoC[1:, :]

    headroom = ca.fmax(1.0 - SoC_curr, 0)
    max_charge_soc = BATTERY_MAX_CHARGE_W * ca.fmin(headroom / 0.45, 1.0)
    max_discharge_soc = ca.repmat(ca.DM(BATTERY_MAX_DISCHARGE_W), H, S)

    net_pos = ca.fmin(ca.fmax(net_W, 0), max_charge_soc)
    net_neg = ca.fmax(ca.fmin(net_W, 0), -max_discharge_soc)

    dt_h_mat = ca.repmat(dt_h, 1, S)
    d_soc = (net_pos * BATTERY_EFF + net_neg / BATTERY_EFF) * dt_h_mat / BATTERY_Wh

    opti.subject_to(ca.vec(SoC[0, :]) == soc_init)
    opti.subject_to(ca.vec(SoC_next) == ca.vec(SoC_curr + d_soc))
    opti.subject_to(ca.vec(SoC_next) >= SOC_MIN)
    opti.subject_to(ca.vec(SoC_next) <= ca.vec(SOC_MAX + slack_soc))

    # ── Temperature dynamics ─────────────────────────────────────────────────
    T_curr = T[:-1]
    T_next = T[1:]
    ghi_0 = ca.fmax(ghi_cas[:H, 0:1], 0)
    T_sky = temp_out_v - T_SKY_OFFSET

    dT = (ALPHA_SOLAR * ghi_0 * is_day_v * dt_s
          + ALPHA_VENT * vent * (temp_out_v - T_curr) * dt_s
          + ALPHA_INFIL * (temp_out_v - T_curr) * dt_s
          - ALPHA_RAD * ca.fmax(T_curr - T_sky, 0) * is_night_v * dt_s
          + ALPHA_GND * (T_GROUND_C - T_curr) * dt_s
          + ALPHA_INT * load_W * dt_s)

    opti.subject_to(T[0] == temp_init_c)
    opti.subject_to(T_next == T_curr + dT)
    opti.subject_to(T_next >= (plant_temp_target - plant_temp_band) - slack_temp)
    opti.subject_to(T_next <= (plant_temp_target + plant_temp_band) + slack_temp)

    # ── Soil moisture dynamics (bucket model) ────────────────────────────────
    Soil_curr = Soil[:-1]
    Soil_next = Soil[1:]

    # Irrigation: pump adds water proportional to pump minutes
    # SOIL_MAX_PUMP_RATE = % per minute of pumping
    irrigation = pump * SOIL_MAX_PUMP_RATE

    # Evapotranspiration: removes water, proportional to solar radiation
    et_rate = ca.DM(SOIL_ET_RATE * dt_h_arr)
    et_scale = is_day_v + SOIL_ET_NIGHT_SCALE * is_night_v
    # Scale ET by solar intensity (higher GHI = more transpiration)
    ghi_norm = ca.fmin(ghi_avg[:H] / 600.0, 1.0)  # normalize to [0,1] at 600 W/m²
    et = et_rate * et_scale * (0.3 + 0.7 * ghi_norm)  # 30% base + 70% solar-driven

    # Drainage: water above field capacity drains away
    excess = ca.fmax(Soil_curr - SOIL_FIELD_CAPACITY, 0)
    drainage = excess * ca.DM(dt_h_arr) * SOIL_DRAIN_RATE

    # Simple dynamics: next = current + irrigation - ET - drainage
    d_soil = irrigation - et - drainage
    opti.subject_to(Soil[0] == soil_moisture)
    opti.subject_to(Soil_next == Soil_curr + d_soil)
    opti.subject_to(Soil_next >= 0)
    opti.subject_to(Soil_next <= SOIL_FIELD_CAPACITY)

    # ── Cost function ────────────────────────────────────────────────────────
    cost = ca.DM(0)

    # 1. Soil moisture tracking (primary objective)
    #    Track stage-specific target with soft band
    #    Skip when pump locked out — can't fix soil, causes IPOPT stall
    if not pump_lockout:
        soil_track = (Soil_next - soil_target) ** 2
        cost += W_SOIL_TRACK * ca.sum(soil_track)

        # 1b. Catastrophic penalties for extreme soil moisture
        soil_below_wilt = ca.fmax(SOIL_WILTING_POINT - Soil_next, 0)
        soil_above_capacity = ca.fmax(Soil_next - SOIL_FIELD_CAPACITY, 0)
        cost += W_SOIL_DRY * ca.sum(soil_below_wilt ** 2)
        cost += W_SOIL_WET * ca.sum(soil_above_capacity ** 2)

        # 1c. Day/night watering preference
        #     Reward pumping during daytime, penalize at nighttime
        day_pump = ca.dot(pump, is_day_v)
        night_pump = ca.dot(pump, is_night_v)
        cost -= W_PUMP_DAY_BONUS * day_pump
        cost += W_PUMP_NIGHT_PENALTY * night_pump

    # 1d. Vent comfort: reward venting when outside temp is in comfort band
    #     Open vent whenever outside is comfortable — day and night for airflow.
    temp_hi = plant_temp_target + plant_temp_band
    temp_lo = plant_temp_target - plant_temp_band
    outside_hi = ca.fmax(temp_out_v - temp_hi, 0)
    outside_lo = ca.fmax(temp_lo - temp_out_v, 0)
    outside_penalty = (outside_hi + outside_lo) / 5.0
    outside_in_band = ca.fmax(1.0 - outside_penalty, 0)
    vent_comfort = ca.dot(vent, outside_in_band)
    cost -= W_VENT_DAY_BONUS * vent_comfort

    # 2. Load timing: no daily caps — solver decides based on soil/plant needs.
    #    Energy cost keeps loads from being wasteful; timing preference
    #    (day/night bonuses) tells the solver WHEN to run.

    # 2b. Lights as SOC management — when SOC is high, running lights
    #     absorbs excess solar energy AND helps plants. Scale by plant stage:
    #     germinating=0 (no lights), growing=1, harvest=0.5.
    soc_avg = ca.mtimes(SoC_curr, w_v)  # (H,1) weighted avg across scenarios
    soc_above_70 = ca.fmax(soc_avg - 0.70, 0)
    lights_absorption = ca.dot(lights * soc_above_70, dt_h)
    lights_need_scale = 1.0 if lights_remaining_h > 0 else 0.0
    cost -= W_LIGHTS_ABSORB * lights_absorption * lights_need_scale

    # 3. SOC safety + tracking + forecast-aware energy planning
    #    When sun is coming (high GHI ahead), relax SOC target (spend now).
    #    When no sun (cloudy/night ahead), tighten target (conserve).
    below = ca.fmax(SOC_MIN - SoC_next, 0)
    above = ca.fmax(SoC_next - SOC_MAX, 0)

    # Per-step dynamic SOC target using daily solar budget:
    #   High-yield day (0.8 kWh) → target drops to 15% (will recharge)
    #   Low-yield day (0.1 kWh)  → target stays at 55% (conserve)
    #   Smooth interpolation between the two.
    if daily_solar_kwh is not None and len(daily_solar_kwh) > 0:
        # Map daily kWh to per-step target using the hour of each step
        hour_arr = np.cumsum(dt_h_arr)  # cumulative hours from t0
        day_targets = np.full(H, SOC_TARGET)
        for d, kwh in enumerate(daily_solar_kwh):
            # Map kWh to target: 0 kWh → 0.55, 1+ kWh → 0.15
            target = SOC_TARGET - (SOC_TARGET - 0.15) * min(kwh / 1.0, 1.0)
            day_start_h = d * 24
            day_end_h = (d + 1) * 24
            mask = (hour_arr >= day_start_h) & (hour_arr < day_end_h)
            day_targets[mask] = target
        dynamic_soc_target = ca.DM(day_targets.reshape(-1, 1))
    else:
        # Fallback: use cumulative GHI in current scenario
        ghi_now = ca.fmax(ghi_cas[:H, 0:1], 0)
        solar_incoming = ca.cumsum(ghi_now) / 600.0
        solar_incoming_clipped = ca.fmin(solar_incoming, 1.0)
        dynamic_soc_target = 0.15 + (SOC_TARGET - 0.15) * (1.0 - solar_incoming_clipped)

    track = (SoC_next - dynamic_soc_target) ** 2
    soc_cost = W_SOC_CRIT * below ** 2 + W_SOC_FULL * above ** 2 + W_SOC_TRACK * track
    cost += ca.sum(ca.mtimes(soc_cost, w_v))

    # Additional: reward being above safety margin when no sun is forecast
    # (extra incentive to conserve when cloudy)
    if daily_solar_kwh is not None and len(daily_solar_kwh) > 0:
        # Low solar day = conserve energy
        tomorrow_solar = daily_solar_kwh[1] if len(daily_solar_kwh) > 1 else daily_solar_kwh[0]
        conserve_factor = max(0, 1.0 - tomorrow_solar / 0.8)  # 0=high solar, 1=no solar
        soc_above_floor = ca.fmax(SoC_next - SOC_MIN, 0)
        cost -= W_SOC_FORECAST * conserve_factor * ca.sum(soc_above_floor)

    # 4. Temperature comfort
    cost += W_TEMP * ca.sum((T_next - plant_temp_target) ** 2)

    # 5. Binary corner-pushing
    cost += W_BIN_LIGHTS * ca.sum(lights * (1 - lights))
    cost += W_BIN_VENT * ca.sum(vent * (1 - vent))

    # 6. Vent switching + pump energy + smoothness
    cost += W_VENT_MOVE * ca.sumsqr(ca.diff(vent))
    cost += W_VENT_MOVE * vent[0] ** 2
    if pump_max_min > 0:
        cost += W_PUMP_ENERGY * ca.sumsqr(pump) / (pump_max_min ** 2)
        cost += W_RAMP * ca.sumsqr(ca.diff(pump) / pump_max_min)

        # 6b. Pump cooldown: penalize pumping within PUMP_COOLDOWN_STEPS of
        #     another pump activation. Forces ~1hr spacing between bursts.
        CD = min(PUMP_COOLDOWN_STEPS, H)
        for k in range(1, CD + 1):
            cost += W_PUMP_CLOSE * ca.dot(pump[k:], pump[:-k])

    cost += W_RAMP * ca.sumsqr(ca.diff(lights))

    # 7. Slack penalties
    cost += W_SLACK_SOC * ca.sum(slack_soc)
    cost += W_SLACK_TEMP * ca.sum(slack_temp)
    cost += W_SOIL_WET * ca.sum(slack_soil)

    opti.minimize(cost)

    opts = {
        "ipopt.print_level": 5 if verbose else 1,
        "print_time": 1 if verbose else 0,
        "ipopt.max_iter": min(MPC_MAX_ITER, 5000),
        "ipopt.max_cpu_time": 120,
        "ipopt.tol": MPC_TOL,
        "ipopt.acceptable_tol": 1e-3,
        "ipopt.linear_solver": "mumps",
        "ipopt.mu_strategy": "adaptive",
        "ipopt.warm_start_init_point": "yes",
        "ipopt.print_frequency_iter": 1 if verbose else 10000,
    }
    opti.solver("ipopt", opts)

    # ── Warm starts ──────────────────────────────────────────────────────────
    # Use previous solution if available (shift by 1 step), else use heuristics
    if prev_result is not None:
        try:
            prev_lights = np.atleast_1d(prev_result.lights_plan)
            prev_vent = np.atleast_1d(prev_result.vent_plan)
            prev_pump = np.atleast_1d(prev_result.pump_plan)
            prev_soc = np.atleast_2d(prev_result.soc_trajectory)
            prev_temp = np.atleast_1d(prev_result.temp_trajectory)
            prev_soil = np.atleast_1d(prev_result.soil_trajectory)
            # Shift previous solution by 1 step (drop first, pad last)
            opti.set_initial(lights, np.append(prev_lights[1:], prev_lights[-1]))
            opti.set_initial(vent, np.append(prev_vent[1:], prev_vent[-1]))
            opti.set_initial(pump, np.append(prev_pump[1:], prev_pump[-1]))
            if prev_soc.shape == (H + 1, S):
                opti.set_initial(SoC, prev_soc)
            if len(prev_temp) >= H + 1:
                opti.set_initial(T, prev_temp[:H + 1])
            if len(prev_soil) >= H + 1:
                opti.set_initial(Soil, prev_soil[:H + 1])
            if verbose:
                print(f"  [MPC] warm-started from previous solution")
        except Exception:
            pass  # fall through to heuristics

    opti.set_initial(SoC, soc_init)
    opti.set_initial(T, temp_init_c)
    opti.set_initial(Soil, soil_moisture)

    # Vent: on whenever outside temp is comfortable
    vent_init = np.zeros(H)
    for t in range(H):
        if temp_out_avg[t] > plant_temp_target - plant_temp_band - 2.0:
            vent_init[t] = 1.0
    opti.set_initial(vent, vent_init)

    # Pump bursting: soil deficit determines burst size
    soil_deficit = max(0, soil_target - soil_moisture)

    # Pump: concentrate into burst(s) during daytime based on soil deficit
    pump_init = np.zeros(H)
    day_steps = [t for t in range(H) if is_day[t] > 0.5]
    pump_needed = soil_deficit / SOIL_MAX_PUMP_RATE  # minutes needed
    if day_steps and pump_needed > 0:
        remaining = min(pump_needed, pump_max_min * len(day_steps))
        for t in day_steps:
            if remaining <= 0:
                break
            chunk = min(remaining, pump_max_min)
            pump_init[t] = chunk
            remaining -= chunk
    opti.set_initial(pump, pump_init)

    # Lights: initialize on during night hours (solver decides optimal amount)
    lights_init = np.zeros(H)
    if lights_remaining_h > 0:
        for t in range(H):
            if is_night[t] > 0.5:
                lights_init[t] = 1.0
    opti.set_initial(lights, lights_init)

    try:
        sol = opti.solve()
        status = "optimal"
    except Exception as exc:
        if verbose:
            print(f"  [MPC] IPOPT warning: {exc}")
        sol = opti.debug
        status = "suboptimal"

    lights_opt = np.clip(np.atleast_1d(sol.value(lights)), 0, 1)
    vent_opt = np.clip(np.atleast_1d(sol.value(vent)), 0, 1)
    pump_opt = np.clip(np.atleast_1d(sol.value(pump)), 0, pump_max_min)

    soc_traj = sol.value(SoC)
    if soc_traj is None:
        soc_traj = np.full((H + 1, S), soc_init)
    elif np.asarray(soc_traj).ndim == 1:
        soc_traj = np.asarray(soc_traj).reshape(-1, 1)
    else:
        soc_traj = np.asarray(soc_traj)

    temp_traj = np.atleast_1d(sol.value(T)).flatten()
    if len(temp_traj) == 0:
        temp_traj = np.full(H + 1, temp_init_c)

    soil_traj = np.atleast_1d(sol.value(Soil)).flatten()
    if len(soil_traj) == 0:
        soil_traj = np.full(H + 1, soil_moisture)

    try:
        obj_val = float(sol.value(cost))
    except Exception:
        obj_val = float("inf")

    solve_ms = (_time.perf_counter() - t0) * 1000.0

    # Compute predicted PV watts at each step from GHI average
    ghi_avg_steps = ghi_avg[:H] if len(ghi_avg) >= H else np.zeros(H)
    temp_avg_steps = temp_out_avg[:H] if len(temp_out_avg) >= H else np.zeros(H)
    temp_derate = np.maximum(1.0 - 0.004 * (temp_avg_steps - 25.0), 0.7)
    pv_watts = ghi_avg_steps * (PV_WPEAK / 1000.0) * PV_MPPT * temp_derate

    if verbose:
        pump_total = float(np.sum(pump_opt))
        lights_total = float(np.sum(lights_opt * dt_h_arr))
        n_day_steps = int(np.sum(is_day))
        n_night_steps = int(np.sum(is_night))
        daily_kwh = float(np.sum(pv_watts * dt_h_arr)) / 1000.0
        print(f"  [MPC] status={status}  cost={obj_val:.2e}  "
              f"pump={pump_total:.1f}min (target={pump_horizon_target:.1f})  "
              f"lights={lights_total:.1f}h  "
              f"soil={soil_moisture:.0f}%→{float(soil_traj[-1]):.0f}% "
              f"(target={soil_target:.0f}±{soil_band:.0f}%)  "
              f"SOC_end={float(soc_traj[-1, 0])*100:.1f}%  "
              f"PV={daily_kwh:.2f}kWh  "
              f"day={n_day_steps} night={n_night_steps} steps  "
              f"t={solve_ms:.0f}ms")

    return MPCResult(
        lights_plan=lights_opt,
        vent_plan=vent_opt,
        pump_plan=pump_opt,
        soc_trajectory=np.asarray(soc_traj),
        temp_trajectory=temp_traj,
        soil_trajectory=soil_traj,
        expected_cost=obj_val,
        solve_time_ms=solve_ms,
        status=status,
        pv_trajectory=pv_watts,
    )
