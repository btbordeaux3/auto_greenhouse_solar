"""
run_twin.py — Solar Greenhouse Digital Twin

Priority hierarchy:
  1. Controller + sensors + comms always powered
  2. SOC stays in [20%, 90%]
  3. Temperature ≈ 80°F via vent (binary: open or closed)
  4. Pump runs 30 min in one contiguous block per day
  5. Grow-lights run 4 h total at night

ALL CONTROL SIGNALS ARE BINARY (0 or 1).
  After each MPC solve the continuous NLP solution is snapped:
    u_binary[i] = 1 if u_continuous[i] > 0.5 else 0
  for ALL signals including the vent.  Critical loads (controller,
  sensors, comms) are always forced to 1 after snapping.

MATLAB PLANT IMPROVEMENTS
──────────────────────────
  - Single shared MATLAB engine per run() call (not restarted between
    steps), so the sizing sweep no longer pays 20-30 s engine startup
    per hardware combo. The engine is passed in as an optional argument
    or created / destroyed once inside run().
  - Historical weather (get_scenarios true_realization) drives GHI and
    T_amb for the MATLAB plant, identical to the Python plant.
  - PV cell temperature derating (NOCT model) computed in Python and
    passed to Simulink as pv_derating each step.
  - Full loss/degradation readback: log_SoH, log_cumE, log_eta_inv,
    log_P_dcdc_dc from Simulink are extracted and returned in the
    dataframe, so collect_metrics() in size_optimizer.py sees real
    numbers for both plants.
  - _setup_matlab() / _teardown_matlab() helpers separate engine
    lifecycle from the per-step logic so size_optimizer.py can
    call them once across all combos.
"""

import argparse
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional, Any

sys.path.insert(0, str(Path(__file__).parent))

from weather.scenario_generator import clear_sky_ghi
from weather.historical_weather import get_scenarios
from optimizer.stochastic_mpc import (
    solve_stochastic_mpc, MPCResult,
    P_MAX, P_VENT, E_MAX_J, ETA_TOTAL, ALWAYS_ON,
    ALWAYS_ON_W,
    SOC_MIN, SOC_MAX,
    TARGET_C, TEMP_BAND,
    ALPHA_SOLAR, ALPHA_VENT, ALPHA_INFIL, ALPHA_RAD, ALPHA_INT, T_SKY_OFFSET,
    T_MAX_ABOVE_AMBIENT,
    inverter_efficiency, dc_dc_efficiency,
    solar_cell_temp, pv_power_corrected, battery_soh,
    BATTERY_NOM_V, BATTERY_MAX_CHARGE_W, BATTERY_MAX_DISCHARGE_W,
    BATTERY_CHARGE_TEMP_C, BATTERY_DISCHARGE_TEMP_C,
    BATTERY_J, BATTERY_EFF,
    NOCT, TEMP_COEFF_PMAX,
    CYCLE_LIFE_80PCT, DEGRADATION_PER_CYCLE,
)
from sensors.state_estimator import SensorConfig, build_kalman, add_sensor_noise, kalman_update

try:
    from optimizer.milp_optimizer import solve_milp
    HAS_MILP = True
except ImportError:
    HAS_MILP = False

RESULTS_DIR = Path(__file__).parent.parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

SUNRISE  = 6.0
SUNSET   = 18.0
TARGET_F = TARGET_C * 9 / 5 + 32   # ≈ 80°F

SNAP_THRESHOLD = 0.5

# Ground temperature for NC (used in MATLAB plant thermal model)
T_GROUND_C = 15.0

# Thermal parameters derived from physical constants (match stochastic_mpc.py)
# These are the W/K conductances for the Simulink plant
_C_TOTAL    = 352_000.0                       # J/K  total thermal capacitance
_SOLAR_TO_W = ALPHA_SOLAR * _C_TOTAL          # ≈ 1.90  m² effective
_VENT_W_K   = ALPHA_VENT  * _C_TOTAL          # ≈ 56.3  W/K
_INF_W_K    = ALPHA_INFIL * _C_TOTAL          # ≈ 0.59  W/K
_RAD_W_K    = ALPHA_RAD   * _C_TOTAL          # ≈ 2.0   W/K
_GND_W_K    = 1.5                             # W/K  ground coupling (~0.5 ACH equiv.)

# Battery degradation constant for Simulink (J basis)
# DEGRADATION_PER_CYCLE is per equivalent full cycle.
# Convert to per-Joule: deg_per_J = DEGRADATION_PER_CYCLE / (2 * E_bat_J)
# We pass E_bat_J per combo, so compute per combo in _write_matlab_init.


# ──────────────────────────────────────────────────────────────────────────────
# Binary snap
# ──────────────────────────────────────────────────────────────────────────────
def snap_binary(u_cont: np.ndarray, u_vent_cont: float) -> Tuple[np.ndarray, float]:
    u_bin = (u_cont >= SNAP_THRESHOLD).astype(float)
    for idx in ALWAYS_ON:
        u_bin[idx] = 1.0
    u_vent_bin = 1.0 if u_vent_cont >= SNAP_THRESHOLD else 0.0
    return u_bin, u_vent_bin


# ──────────────────────────────────────────────────────────────────────────────
# Python physics plant (unchanged from original)
# ──────────────────────────────────────────────────────────────────────────────
class PythonPlant:
    FLOOR_AREA   = 2.23
    VOLUME       = 3.5
    GLAZING_TRANS = 0.85
    ACH_INFIL    = 0.5
    ACH_VENT     = 48.0
    RHO_AIR      = 1.2
    CP_AIR       = 1005.0
    SOIL_DEPTH   = 0.1
    RHO_SOIL     = 1500.0
    CP_SOIL      = 800.0
    C_STRUCT     = 80000.0

    def __init__(self, soc_init: float = 0.50, temp_init: float = 20.0,
                 true_ghi: Optional[np.ndarray] = None,
                 true_t_amb: Optional[np.ndarray] = None,
                 dt_s: float = 300.0,
                 battery_wh: float = 2000.0, pv_wpeak: float = 342.0,
                 inverter_w: float = 500.0):
        self.soc    = float(np.clip(soc_init, SOC_MIN, SOC_MAX))
        self.temp   = float(temp_init)
        self.time_s = 0.0
        self._true_ghi = true_ghi
        self._true_t_amb = true_t_amb
        self._dt_s     = dt_s
        self._battery_wh  = battery_wh
        self._battery_j   = battery_wh * 3600.0
        self._pv_wpeak    = pv_wpeak
        self._pv_factor   = pv_wpeak / 1000.0
        self._inverter_w  = inverter_w

        C_air = self.VOLUME * self.RHO_AIR * self.CP_AIR
        C_soil = self.SOIL_DEPTH * self.FLOOR_AREA * self.RHO_SOIL * self.CP_SOIL
        self._C_total = C_air + C_soil + self.C_STRUCT
        self._solar_W_per_GHI = self.FLOOR_AREA * self.GLAZING_TRANS
        ach_to_WperK = self.VOLUME * self.RHO_AIR * self.CP_AIR / 3600.0
        self._inf_W_per_K  = ach_to_WperK * self.ACH_INFIL
        self._vent_W_per_K = ach_to_WperK * self.ACH_VENT
        self._rad_W_per_K  = 2.0
        self._gnd_W_per_K  = _GND_W_K

        self._cumulative_charge_j    = 0.0
        self._cumulative_discharge_j = 0.0
        self._battery_cycles = 0.0
        self._battery_soh    = 1.0
        self._inv_loss_j     = 0.0

        self._last_u      = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
        self._last_u_vent = 0.0
        self._last_pv_power = 0.0
        self._last_pv_dc    = 0.0
        self._last_inv_dc   = 0.0
        self._last_inv_loss = 0.0
        self._last_ghi      = 0.0

    def get_state(self) -> dict:
        step = int(round(self.time_s / self._dt_s))
        T_out = self._get_t_amb(step)
        p_load = float(P_MAX @ self._last_u)
        p_vent = P_VENT * self._last_u_vent
        p_dcdc_out = P_MAX[2] + P_MAX[3] + P_MAX[4]
        dcdc_eff_now = dc_dc_efficiency(p_dcdc_out)
        p_dcdc_loss_now = p_dcdc_out / max(dcdc_eff_now, 0.01) - p_dcdc_out
        return {
            'soc':             self.soc,
            'p_pv':            self._last_pv_power,
            'p_load':          p_load + p_vent + p_dcdc_loss_now,
            'p_lights':        self._last_u[0] * P_MAX[0],
            'p_pump':          self._last_u[1] * P_MAX[1],
            'p_controller':    P_MAX[2],
            'p_sensors':       P_MAX[3],
            'p_comms':         P_MAX[4],
            'p_vent':          p_vent,
            'p_dcdc_loss':     p_dcdc_loss_now,
            'temp':            self.temp,
            't_ambient':       T_out,
            'ghi':             self._last_ghi,
            'battery_soh':     self._battery_soh,
            'battery_cycles':  self._battery_cycles,
            'p_inv_loss':      self._last_inv_loss,
            'p_pv_dc':         self._last_pv_dc,
            'p_inv_dc':        self._last_inv_dc,
        }

    def _ambient(self, hour: float) -> float:
        return 18.0 + 8.0 * np.sin(2.0 * np.pi * (hour % 24.0 - 14.0) / 24.0)

    def _get_t_amb(self, step: int) -> float:
        if self._true_t_amb is not None and step < len(self._true_t_amb):
            return float(self._true_t_amb[step])
        return self._ambient(step * self._dt_s / 3600.0)

    def _get_ghi(self, step: int) -> float:
        if self._true_ghi is not None and step < len(self._true_ghi):
            return float(self._true_ghi[step])
        return clear_sky_ghi(self.time_s)

    def step(self, u: np.ndarray, u_vent: float, dt_s: float,
             step_idx: int = 0) -> dict:
        self._last_u      = np.clip(u, 0, 1).copy()
        self._last_u_vent = float(np.clip(u_vent, 0, 1))

        hour     = (self.time_s / 3600.0) % 24.0
        is_day   = 1.0 if (SUNRISE <= hour <= SUNSET) else 0.0
        is_night = 1.0 - is_day

        self._last_ghi = self._get_ghi(step_idx)
        T_out = self._get_t_amb(step_idx)
        T_sky = T_out - T_SKY_OFFSET

        self._last_pv_dc    = pv_power_corrected(self._pv_wpeak, self._last_ghi, T_out)
        self._last_pv_power = self._last_pv_dc

        u_clip   = self._last_u
        u_vent_c = self._last_u_vent

        p_dcdc_out  = P_MAX[2] + P_MAX[3] + P_MAX[4]
        dcdc_eff    = dc_dc_efficiency(p_dcdc_out)
        p_dcdc_loss = p_dcdc_out / max(dcdc_eff, 0.01) - p_dcdc_out

        p_load_full = float(P_MAX @ u_clip)
        p_vent_full = P_VENT * u_vent_c
        net_before  = self._last_pv_dc - p_load_full - p_vent_full - p_dcdc_loss

        if self.soc <= SOC_MIN and net_before < 0:
            u_shed    = u_clip.copy()
            u_shed[0] = 0.0
            u_shed[1] = 0.0
            p_load_shed = float(P_MAX @ u_shed)
            net_shed    = self._last_pv_dc - p_load_shed - p_vent_full - p_dcdc_loss
            if net_shed < 0:
                u_vent_c = 0.0
                u_clip   = u_shed
            else:
                u_clip = u_shed

        p_load  = float(P_MAX @ u_clip)
        p_vent  = P_VENT * u_vent_c
        p_total = p_load + p_vent + p_dcdc_loss

        # All loads are 12V DC direct from battery bus; 5V rail via DC-DC converter.
        # p_load includes P_MAX[2:] (5V loads at output power), so for battery draw we
        # count 12V loads directly and 5V loads at DC-DC input power.
        p_load_12v = u_clip[0] * P_MAX[0] + u_clip[1] * P_MAX[1] + P_VENT * u_vent_c
        p_dcdc_input = p_dcdc_out + p_dcdc_loss
        total_battery_draw_dc = p_load_12v + p_dcdc_input
        net_battery_dc = self._last_pv_dc - total_battery_draw_dc
        self._last_inv_dc   = 0.0
        self._last_inv_loss = 0.0

        temp_ok_charge    = BATTERY_CHARGE_TEMP_C[0]    <= T_out <= BATTERY_CHARGE_TEMP_C[1]
        temp_ok_discharge = BATTERY_DISCHARGE_TEMP_C[0] <= T_out <= BATTERY_DISCHARGE_TEMP_C[1]

        if net_battery_dc >= 0:
            if temp_ok_charge:
                max_charge_w = min(BATTERY_MAX_CHARGE_W, 0.5 * self._battery_j / 3600.0)
                charge_w = min(net_battery_dc, max_charge_w)
            else:
                charge_w = 0.0
            soc_change = charge_w * ETA_TOTAL * dt_s / self._battery_j
            self._cumulative_charge_j += charge_w * dt_s * ETA_TOTAL
        else:
            if temp_ok_discharge:
                max_discharge_w = min(BATTERY_MAX_DISCHARGE_W, 0.5 * self._battery_j / 3600.0)
                discharge_w = min(abs(net_battery_dc), max_discharge_w)
            else:
                discharge_w = 0.0
            soc_change = -discharge_w / ETA_TOTAL * dt_s / self._battery_j
            self._cumulative_discharge_j += discharge_w * dt_s / ETA_TOTAL

        self._inv_loss_j += self._last_inv_loss * dt_s
        self.soc = float(np.clip(self.soc + soc_change, 0.0, 1.0))

        total_throughput = self._cumulative_charge_j + self._cumulative_discharge_j
        self._battery_cycles = total_throughput / (2.0 * self._battery_j)
        self._battery_soh    = battery_soh(self._battery_cycles)

        Q_solar = self._solar_W_per_GHI * self._last_ghi * is_day
        Q_infil = self._inf_W_per_K * (T_out - self.temp)
        Q_vent  = self._vent_W_per_K * u_vent_c * (T_out - self.temp)
        Q_rad   = -self._rad_W_per_K * max(self.temp - T_sky, 0.0) * is_night
        Q_int   = p_load
        Q_gnd   = self._gnd_W_per_K * (T_GROUND_C - self.temp)

        Q_net = Q_solar + Q_infil + Q_vent + Q_rad + Q_int + Q_gnd
        dT = Q_net / self._C_total * dt_s
        self.temp += dT
        self.temp = float(np.clip(self.temp, -10.0, T_out + T_MAX_ABOVE_AMBIENT))
        self.time_s += dt_s

        return {
            'soc':            self.soc,
            'p_pv':           self._last_pv_dc,
            'p_load':         p_total,
            'p_lights':       u_clip[0] * P_MAX[0],
            'p_pump':         u_clip[1] * P_MAX[1],
            'p_controller':   P_MAX[2],
            'p_sensors':      P_MAX[3],
            'p_comms':        P_MAX[4],
            'p_vent':         p_vent,
            'p_dcdc_loss':    p_dcdc_loss,
            'temp':           self.temp,
            't_ambient':      T_out,
            'ghi':            self._last_ghi,
            'p_pv_dc':        self._last_pv_dc,
            'p_inv_dc':       self._last_inv_dc,
            'p_inv_loss':     self._last_inv_loss,
            'battery_soh':    self._battery_soh,
            'battery_cycles': self._battery_cycles,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Rule-based baseline controller
# ──────────────────────────────────────────────────────────────────────────────
def rule_based_control(
    state: dict, hour: float, is_dark: bool,
    pump_done_today: bool, lights_done_today: bool,
    u_vent_prev: float, soc: float, temp: float,
) -> tuple:
    u = np.zeros(5)
    u[2:] = 1.0
    if temp > TARGET_C + 2.0:
        u_vent = 1.0
    elif temp < TARGET_C - 2.0:
        u_vent = 0.0
    else:
        u_vent = u_vent_prev
    if not pump_done_today and 10.0 <= hour < 10.5:
        u[1] = 1.0
    if not lights_done_today and is_dark and soc > 0.15:
        u[0] = 1.0
    if soc <= 0.10:
        u[0] = 0.0
        u[1] = 0.0
    return u, float(u_vent)


# ──────────────────────────────────────────────────────────────────────────────
# Deterministic MPC wrapper
# ──────────────────────────────────────────────────────────────────────────────
def deterministic_mpc_control(
    soc_est: float, temp: float,
    ghi_scen: np.ndarray, dhi_scen: np.ndarray, dni_scen: np.ndarray,
    temp_scen: np.ndarray, wind_scen: np.ndarray,
    dt: float, actual_H: int,
    u_prev: np.ndarray, u_vent_prev: float, hour: float,
    pump_done_today: bool, lights_done_today: bool,
    pump_remaining: int, lights_remaining: int,
    battery_wh: float, pv_wpeak: float, inverter_w: float,
) -> MPCResult:
    return solve_stochastic_mpc(
        soc_init=soc_est,
        temp_inside_init=float(temp),
        ghi_scenarios=ghi_scen[:1],
        dhi_scenarios=dhi_scen[:1],
        dni_scenarios=dni_scen[:1],
        temp_out_scenarios=temp_scen[:1],
        wind_scenarios=wind_scen[:1],
        dt_s=dt,
        horizon_steps=actual_H,
        u_prev=u_prev,
        u_vent_prev=u_vent_prev,
        time_of_day_hours=hour,
        pump_done_today=pump_done_today,
        lights_done_today=lights_done_today,
        pump_remaining_steps=pump_remaining,
        lights_remaining_steps=lights_remaining,
        verbose=False,
        battery_j=battery_wh * 3600.0,
        pv_factor=pv_wpeak / 1000.0,
    )


# ──────────────────────────────────────────────────────────────────────────────
# MATLAB engine lifecycle helpers
# ──────────────────────────────────────────────────────────────────────────────

def _import_matlab():
    try:
        import matlab.engine
        return matlab.engine
    except ImportError:
        raise RuntimeError(
            "matlab.engine not found. Install with:\n"
            "  cd <matlabroot>/extern/engines/python && python setup.py install"
        )


def setup_matlab(model_path: Optional[Path] = None) -> Tuple[Any, str]:
    """
    Start a MATLAB engine, build the greenhouse model, and return
    (eng, model_name).  Call once before a sizing sweep and pass eng
    to each run() call via the `matlab_eng` argument.
    """
    matlab = _import_matlab()
    if model_path is None:
        model_path = Path(__file__).parent / 'greenhouse_twin_model.slx'

    print("  Starting MATLAB engine …")
    eng = matlab.engine.start_matlab()
    eng.addpath(str(model_path.parent), nargout=0)

    if model_path.exists():
        model_path.unlink()
    print("  Building realistic Simulink model via greenhouse.m …")
    eng.eval("run('greenhouse.m')", nargout=0)
    print("  ✓ Model built")

    model_name = model_path.stem
    eng.load_system(model_name, nargout=0)
    print(f"  ✓ Loaded {model_name}\n")
    return eng, model_name


def teardown_matlab(eng: Any) -> None:
    """Quit the shared MATLAB engine."""
    try:
        eng.quit()
    except Exception:
        pass


def _write_matlab_init(eng: Any, model_name: str,
                       battery_wh: float, pv_wpeak: float, inverter_w: float):
    """
    Write component-level constant parameters to the MATLAB workspace.
    Call once per hardware combo (battery/PV/inverter change).
    """
    battery_j = battery_wh * 3600.0
    bat_max_charge_w = min(BATTERY_NOM_V * 50.0,         # 0.5C for 100Ah-equiv
                           0.5 * battery_j / 3600.0)
    bat_max_disch_w  = min(BATTERY_NOM_V * 100.0,
                           battery_j / 3600.0)

    eng.workspace['pv_wpeak']          = float(pv_wpeak)
    eng.workspace['inv_rated_W']       = float(inverter_w)
    eng.workspace['dcdc_rated_W']      = float(360.0)        # Victron Orion-TR rated
    eng.workspace['E_bat_J']           = float(battery_j)
    eng.workspace['bat_eff']           = float(BATTERY_EFF)
    eng.workspace['bat_max_charge_W']  = float(bat_max_charge_w)
    eng.workspace['bat_max_disch_W']   = float(bat_max_disch_w)
    eng.workspace['deg_per_J']         = float(DEGRADATION_PER_CYCLE / (2.0 * battery_j))
    eng.workspace['solar_to_W']        = float(_SOLAR_TO_W)
    eng.workspace['vent_W_per_K']      = float(_VENT_W_K)
    eng.workspace['inf_W_per_K']       = float(_INF_W_K)
    eng.workspace['rad_W_per_K']       = float(_RAD_W_K)
    eng.workspace['gnd_W_per_K']       = float(_GND_W_K)
    eng.workspace['T_ground_C']        = float(T_GROUND_C)
    eng.workspace['T_sky_offset']      = float(T_SKY_OFFSET)
    eng.workspace['C_total_J_K']       = float(_C_TOTAL)
    # Named load powers
    eng.workspace['p_lights'] = float(P_MAX[0])
    eng.workspace['p_pump']   = float(P_MAX[1])
    eng.workspace['p_ctrl']   = float(P_MAX[2])
    eng.workspace['p_sens']   = float(P_MAX[3])
    eng.workspace['p_comms']  = float(P_MAX[4])
    eng.workspace['p_vent']   = float(P_VENT)


def _write_matlab_step(eng: Any, model_name: str,
                        u_now: np.ndarray, u_vent_now: float,
                        ghi_now: float, t_amb_now: float,
                        soc_ic: float, t_ic_K: float,
                        cum_e_ic: float, dt_s: float):
    """
    Write per-step workspace variables and set StopTime.
    PV cell-temperature derating is computed here in Python
    (NOCT model) and passed to Simulink as a scalar.
    """
    # NOCT derating: cap to [0.5, 1.0] to avoid negative output
    if ghi_now > 0:
        t_cell = t_amb_now + (NOCT - 20.0) * (ghi_now / 800.0)
        derating = 1.0 + TEMP_COEFF_PMAX * (t_cell - 25.0)
        derating = float(np.clip(derating, 0.5, 1.0))
    else:
        derating = 0.0

    names = ['u_Lights_now', 'u_Pump_now', 'u_Controller_now',
             'u_Sensors_now', 'u_Comms_now']
    for i, name in enumerate(names):
        eng.workspace[name] = float(u_now[i])
    eng.workspace['u_vent_now']  = float(u_vent_now)
    eng.workspace['GHI_now']     = float(ghi_now)
    eng.workspace['T_amb_now']   = float(t_amb_now)
    eng.workspace['pv_derating'] = float(derating)
    eng.workspace['SoC_ic']      = float(soc_ic)
    eng.workspace['T_ic']        = float(t_ic_K)
    eng.workspace['cumE_ic']     = float(cum_e_ic)
    eng.workspace['dt_s']        = float(dt_s)
    eng.eval(f"set_param('{model_name}', 'StopTime', num2str(dt_s))", nargout=0)


def _run_simulink(eng: Any, model_name: str):
    eng.eval(f"simOut = sim('{model_name}');", nargout=0)
    for var in ('log_SoC', 'log_P_pv', 'log_T_inside',
                'log_SoH', 'log_cumE', 'log_eta_inv',
                'log_P_ac', 'log_P_dcdc_dc', 'log_Net_DC'):
        eng.eval(f"{var} = simOut.{var};", nargout=0)


def _read_simulink_state(eng: Any, battery_wh: float) -> dict:
    """
    Read all logged signals from the last Simulink step.
    Returns a dict matching the PythonPlant.get_state() schema.
    """
    def _last(varname):
        try:
            data = np.array(eng.workspace[varname]).flatten()
            return float(data[-1]) if len(data) > 0 else 0.0
        except Exception:
            return 0.0

    soc        = _last('log_SoC')
    p_pv_dc    = _last('log_P_pv')
    t_inside_c = _last('log_T_inside')
    soh        = _last('log_SoH')
    cum_e      = _last('log_cumE')
    p_ac       = _last('log_P_ac')
    p_dcdc_dc  = _last('log_P_dcdc_dc')
    # Inverter DC draw = P_ac / eta  (already in log_Net_DC indirectly)
    # Reconstruct p_inv_dc from net: net = pv - inv_dc - dcdc_dc
    net_dc     = _last('log_Net_DC')
    p_inv_dc   = p_pv_dc - net_dc - p_dcdc_dc   # may be 0 if PV=0

    p_dcdc_demand = P_MAX[2] + P_MAX[3] + P_MAX[4]   # output side
    p_dcdc_loss   = max(p_dcdc_dc - p_dcdc_demand, 0.0)
    p_inv_loss    = max(p_inv_dc  - p_ac, 0.0)

    battery_j   = battery_wh * 3600.0
    bat_cycles  = cum_e / (2.0 * battery_j) if battery_j > 0 else 0.0
    p_load_total = p_ac + p_dcdc_dc   # total power drawn from bus for loads

    return {
        'soc':            float(soc),
        'p_pv':           float(p_pv_dc),
        'p_load':         float(p_load_total),
        'p_lights':       0.0,   # not individually logged in Simulink
        'p_pump':         0.0,
        'p_controller':   float(P_MAX[2]),
        'p_sensors':      float(P_MAX[3]),
        'p_comms':        float(P_MAX[4]),
        'p_vent':         0.0,
        'p_dcdc_loss':    float(p_dcdc_loss),
        'temp':           float(t_inside_c),
        't_ambient':      0.0,   # filled in caller
        'ghi':            0.0,   # filled in caller
        'battery_soh':    float(soh),
        'battery_cycles': float(bat_cycles),
        'p_inv_loss':     float(p_inv_loss),
        'p_pv_dc':        float(p_pv_dc),
        'p_inv_dc':       float(p_inv_dc),
        'cumE':           float(cum_e),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main simulation loop
# ──────────────────────────────────────────────────────────────────────────────
def run(
        sim_days:       int   = 1,
        n_scenarios:    int   = 10,
        horizon_steps:  int   = 12,
        use_matlab:     bool  = False,
        weather_method: str   = 'gp',
        dt_minutes:     int   = 5,
        battery_wh:    float  = 2000.0,
        pv_wpeak:      float  = 342.0,
        inverter_w:    float  = 500.0,
        controller:    str    = 'mpc',
        log_callback:  callable = None,
        results_dir:   Optional[Path] = None,
        # Shared MATLAB engine (pass from size_optimizer to avoid repeated startups)
        matlab_eng:    Any  = None,
        matlab_model:  str  = 'greenhouse_twin_model',
):
    controller_label = {
        'mpc': 'Stochastic MPC', 'deterministic': 'Deterministic MPC',
        'rule': 'Rule-Based', 'milp': 'MILP',
    }.get(controller, controller)

    print(f"\n{'='*68}")
    print(f"  Solar Greenhouse Digital Twin  (6×4 ft) — {controller_label}")
    print(f"  Target: {TARGET_F:.0f}°F / {TARGET_C:.1f}°C")
    print(f"  Battery: {battery_wh:.0f} Wh | PV: {pv_wpeak:.0f} Wp | Inverter: {inverter_w:.0f} W")
    print(f"  {sim_days} day(s) | {n_scenarios} scenarios | horizon={horizon_steps} steps")
    print(f"  Plant: {'Simulink (realistic)' if use_matlab else 'Python'}")
    print(f"  ALL SIGNALS BINARY (snap threshold = {SNAP_THRESHOLD})")
    print(f"{'='*68}\n")

    dt      = dt_minutes * 60.0
    total_s = sim_days * 86400
    n_steps = int(total_s / dt)

    rng = np.random.default_rng(42)
    cfg = SensorConfig()

    soc_init  = 0.50
    temp_init = 24.0

    # ── Pre-sample "true" historical weather for the full simulation ──────────
    true_scenarios = get_scenarios(
        method           = weather_method,
        horizon_s        = int(total_s),
        dt_s             = int(dt),
        n_scenarios      = 1,
        current_time_s   = 0.0,
        rng_seed         = 99999,
        true_realization = True,
    )
    true_ghi   = true_scenarios[0].ghi
    true_t_amb = true_scenarios[0].temperature

    u_prev      = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
    u_vent_prev = 0.0

    # ── MATLAB or Python plant setup ─────────────────────────────────────────
    _own_engine = False
    if use_matlab:
        if matlab_eng is None:
            model_path = Path(__file__).parent / 'greenhouse_twin_model.slx'
            matlab_eng, matlab_model = setup_matlab(model_path)
            _own_engine = True

        # Write hardware-combo constants (cheap — just workspace assignments)
        _write_matlab_init(matlab_eng, matlab_model, battery_wh, pv_wpeak, inverter_w)

        # Initial step to establish plant state
        ghi_0  = float(true_ghi[0])  if len(true_ghi)   > 0 else 0.0
        tamb_0 = float(true_t_amb[0]) if len(true_t_amb) > 0 else 20.0
        _write_matlab_step(matlab_eng, matlab_model,
                           u_prev, u_vent_prev,
                           ghi_0, tamb_0,
                           soc_init, temp_init + 273.15,
                           0.0, dt)
        print("  Running initial Simulink step …")
        _run_simulink(matlab_eng, matlab_model)
        print("  ✓ MATLAB plant ready\n")

        # Read initial state
        mat_state = _read_simulink_state(matlab_eng, battery_wh)
        mat_state['t_ambient'] = tamb_0
        mat_state['ghi']       = ghi_0
        cumE_current = mat_state.get('cumE', 0.0)

        matlab_eng.eval("clear simOut", nargout=0)
        plant = None
    else:
        plant = PythonPlant(soc_init, temp_init,
                            true_ghi=true_ghi, true_t_amb=true_t_amb,
                            dt_s=dt, battery_wh=battery_wh,
                            pv_wpeak=pv_wpeak, inverter_w=inverter_w)
        plant.step(u_prev, u_vent_prev, dt, step_idx=0)
        cumE_current = 0.0

    # ── Kalman filter ────────────────────────────────────────────────────────
    ks = build_kalman(np.array([soc_init, 0.0, 0.0]), cfg)

    # ── Daily step counters ──────────────────────────────────────────────────
    pump_steps_target   = max(1, int(1800 / dt))
    lights_steps_target = max(1, int(14400 / dt))
    pump_steps_done   = 0
    lights_steps_done = 0
    last_day          = -1

    keys = [
        'time_s', 'hour', 'is_dark', 'dt_s',
        'soc_true', 'soc_est',
        'p_pv', 'p_load', 'p_lights', 'p_pump',
        'p_controller', 'p_sensors', 'p_comms', 'p_vent',
        'p_dcdc_loss', 'p_pv_dc', 'p_inv_dc', 'p_inv_loss',
        'battery_soh', 'battery_cycles',
        'temp', 't_ambient', 'ghi',
        'u_lights', 'u_pump', 'u_vent',
        'mpc_cost', 'solve_time_ms',
    ]
    log = {k: [] for k in keys}

    print(f"  {'Step':>5}  {'Hour':>6}  {'SoC%':>5}  {'P_pv':>6}  "
          f"{'P_ld':>6}  {'T_in°F':>7}  {'Pump':>4}  {'Lit':>3}  "
          f"{'Vnt':>3}  {'ms':>5}")
    print(f"  {'-'*5}  {'-'*6}  {'-'*5}  {'-'*6}  "
          f"{'-'*6}  {'-'*7}  {'-'*4}  {'-'*3}  {'-'*3}  {'-'*5}")

    for step in range(n_steps):
        t_now = step * dt
        hour  = (t_now / 3600.0) % 24.0
        day   = int(t_now / 86400)

        if day != last_day:
            pump_steps_done   = 0
            lights_steps_done = 0
            last_day          = day

        is_dark = (hour < SUNRISE or hour > SUNSET)

        pump_done_today   = pump_steps_done   >= pump_steps_target
        lights_done_today = lights_steps_done >= lights_steps_target
        pump_remaining    = max(0, pump_steps_target   - pump_steps_done)
        lights_remaining  = max(0, lights_steps_target - lights_steps_done)

        # ── Read plant state ─────────────────────────────────────────────────
        step_idx   = min(step, len(true_ghi) - 1)   if len(true_ghi)   > 0 else step
        ghi_now    = float(true_ghi[step_idx])       if len(true_ghi)   > 0 else 0.0
        t_amb_now  = float(true_t_amb[step_idx])     if len(true_t_amb) > 0 else 20.0

        if use_matlab:
            state         = mat_state
            state['t_ambient'] = t_amb_now
            state['ghi']       = ghi_now
            true_state    = np.array([state['soc'], state['p_pv'], state['p_load']])
            temp          = state['temp']
            t_ambient     = t_amb_now
            p_lights      = u_prev[0] * P_MAX[0]
            p_pump        = u_prev[1] * P_MAX[1]
            p_vent_r      = P_VENT * u_vent_prev
        else:
            state      = plant.get_state()
            true_state = np.array([state['soc'], state['p_pv'], state['p_load']])
            temp       = state['temp']
            t_ambient  = state['t_ambient']
            p_lights   = state['p_lights']
            p_pump     = state['p_pump']
            p_vent_r   = state['p_vent']

        # ── Kalman update ────────────────────────────────────────────────────
        z       = add_sensor_noise(true_state, cfg, rng)
        ks      = kalman_update(ks, z, cfg, dt)
        soc_est = float(np.clip(ks.x[0], 0.0, 1.0))

        # ── MPC horizon ──────────────────────────────────────────────────────
        t_rem    = total_s - t_now
        h_s      = min(horizon_steps * dt, t_rem)
        if h_s < dt:
            break
        actual_H = max(1, int(h_s / dt))

        scenarios = get_scenarios(
            method         = weather_method,
            horizon_s      = int(actual_H * dt),
            dt_s           = int(dt),
            n_scenarios    = n_scenarios,
            current_time_s = t_now,
            rng_seed       = step,
        )

        ghi_scen  = np.stack([s.ghi[:actual_H]         for s in scenarios])
        dhi_scen  = np.stack([s.dhi[:actual_H]         for s in scenarios])
        dni_scen  = np.stack([s.dni[:actual_H]         for s in scenarios])
        temp_scen = np.stack([s.temperature[:actual_H] for s in scenarios])
        wind_scen = np.stack([s.wind_speed[:actual_H]  for s in scenarios])

        # ── Controller ───────────────────────────────────────────────────────
        cost_val = 0.0
        solve_ms = 0.0

        if controller == 'rule':
            u_now, u_vent_now = rule_based_control(
                state={'temp': temp, 'soc': soc_est, 't_ambient': t_ambient},
                hour=hour, is_dark=is_dark,
                pump_done_today=pump_done_today,
                lights_done_today=lights_done_today,
                u_vent_prev=u_vent_prev, soc=soc_est, temp=temp,
            )

        elif controller == 'deterministic':
            result = deterministic_mpc_control(
                soc_est, temp, ghi_scen, dhi_scen, dni_scen,
                temp_scen, wind_scen, dt, actual_H,
                u_prev, u_vent_prev, hour,
                pump_done_today, lights_done_today,
                pump_remaining, lights_remaining,
                battery_wh, pv_wpeak, inverter_w,
            )
            u_cont      = result.u_opt[0]      if result.u_opt.ndim > 1      else result.u_opt
            u_vent_cont = result.u_vent_opt[0] if result.u_vent_opt.ndim > 0 else float(result.u_vent_opt)
            u_now, u_vent_now = snap_binary(u_cont, u_vent_cont)
            cost_val = result.expected_cost
            solve_ms = result.solve_time_ms

        elif controller == 'milp':
            if not HAS_MILP:
                raise ImportError("milp_optimizer not available. pip install pulp")
            result = solve_milp(
                soc_init=soc_est,
                temp_inside_init=float(temp),
                ghi_forecast=ghi_scen[0],
                dhi_forecast=dhi_scen[0],
                dni_forecast=dni_scen[0],
                temp_out_forecast=temp_scen[0],
                wind_forecast=wind_scen[0],
                dt_s=dt,
                horizon_steps=actual_H,
                u_prev=u_prev,
                u_vent_prev=u_vent_prev,
                time_of_day_hours=hour,
                pump_done_today=pump_done_today,
                lights_done_today=lights_done_today,
                pump_remaining_steps=pump_remaining,
                lights_remaining_steps=lights_remaining,
            )
            u_cont      = result.u_opt[0]      if result.u_opt.ndim > 1      else result.u_opt
            u_vent_cont = result.u_vent_opt[0] if result.u_vent_opt.ndim > 0 else float(result.u_vent_opt)
            u_now, u_vent_now = snap_binary(u_cont, u_vent_cont)
            cost_val = result.expected_cost
            solve_ms = result.solve_time_ms

        else:  # 'mpc' — stochastic MPC (default)
            result = solve_stochastic_mpc(
                soc_init               = soc_est,
                temp_inside_init       = float(temp),
                ghi_scenarios          = ghi_scen,
                dhi_scenarios          = dhi_scen,
                dni_scenarios          = dni_scen,
                temp_out_scenarios     = temp_scen,
                wind_scenarios         = wind_scen,
                dt_s                   = dt,
                horizon_steps          = actual_H,
                u_prev                 = u_prev,
                u_vent_prev            = u_vent_prev,
                time_of_day_hours      = hour,
                pump_done_today        = pump_done_today,
                lights_done_today      = lights_done_today,
                pump_remaining_steps   = pump_remaining,
                lights_remaining_steps = lights_remaining,
                verbose                = False,
                battery_j              = battery_wh * 3600.0,
                pv_factor              = pv_wpeak / 1000.0,
            )
            u_cont      = result.u_opt[0]      if result.u_opt.ndim > 1      else result.u_opt
            u_vent_cont = result.u_vent_opt[0] if result.u_vent_opt.ndim > 0 else float(result.u_vent_opt)
            u_now, u_vent_now = snap_binary(u_cont, u_vent_cont)
            cost_val = result.expected_cost
            solve_ms = result.solve_time_ms

        # ── Update counters ──────────────────────────────────────────────────
        if u_now[1] > 0.5:
            pump_steps_done   += 1
        if u_now[0] > 0.5:
            lights_steps_done += 1

        # ── Log ──────────────────────────────────────────────────────────────
        log['time_s'].append(t_now)
        log['hour'].append(hour)
        log['is_dark'].append(is_dark)
        log['dt_s'].append(dt)
        log['soc_true'].append(float(true_state[0]))
        log['soc_est'].append(soc_est)
        log['p_pv'].append(float(true_state[1]))
        log['p_load'].append(float(true_state[2]))
        log['p_lights'].append(float(p_lights))
        log['p_pump'].append(float(p_pump))
        log['p_controller'].append(P_MAX[2])
        log['p_sensors'].append(P_MAX[3])
        log['p_comms'].append(P_MAX[4])
        log['p_vent'].append(float(p_vent_r))
        log['p_dcdc_loss'].append(float(state.get('p_dcdc_loss', 0.0)))
        log['p_pv_dc'].append(float(state.get('p_pv_dc', 0.0)))
        log['p_inv_dc'].append(float(state.get('p_inv_dc', 0.0)))
        log['p_inv_loss'].append(float(state.get('p_inv_loss', 0.0)))
        log['battery_soh'].append(float(state.get('battery_soh', 1.0)))
        log['battery_cycles'].append(float(state.get('battery_cycles', 0.0)))
        log['temp'].append(float(temp))
        log['t_ambient'].append(float(t_ambient))
        log['ghi'].append(float(ghi_now))
        log['u_lights'].append(float(u_now[0]))
        log['u_pump'].append(float(u_now[1]))
        log['u_vent'].append(float(u_vent_now))
        log['mpc_cost'].append(cost_val)
        log['solve_time_ms'].append(solve_ms)

        temp_f = temp * 9.0 / 5.0 + 32.0
        if step % 12 == 0:
            print(f"  {step:>5}  {hour:>5.1f}h  "
                  f"{soc_est*100:>5.1f}  {true_state[1]:>6.0f}  "
                  f"{true_state[2]:>6.0f}  {temp_f:>7.1f}  "
                  f"{'✓' if pump_done_today else '✗':>4}  "
                  f"{'✓' if lights_done_today else '✗':>3}  "
                  f"{'O' if u_vent_now > 0.5 else '-':>3}  "
                  f"{solve_ms:>5.0f}")

        if log_callback:
            log_callback(log)

        # ── Apply controls (advance plant one step) ──────────────────────────
        if use_matlab:
            cumE_current = state.get('cumE', cumE_current)
            _write_matlab_step(matlab_eng, matlab_model,
                               u_now, u_vent_now,
                               ghi_now, t_amb_now,
                               true_state[0],        # SoC IC = current SoC
                               temp + 273.15,        # T_ic in Kelvin
                               cumE_current, dt)
            _run_simulink(matlab_eng, matlab_model)
            mat_state = _read_simulink_state(matlab_eng, battery_wh)
            mat_state['t_ambient'] = t_amb_now
            mat_state['ghi']       = ghi_now
            cumE_current = mat_state.get('cumE', cumE_current)
            matlab_eng.eval("clear simOut", nargout=0)
        else:
            plant.step(u_now, u_vent_now, dt, step_idx=step)

        u_prev      = u_now
        u_vent_prev = u_vent_now

    # ── Teardown owned engine ────────────────────────────────────────────────
    if use_matlab and _own_engine:
        teardown_matlab(matlab_eng)

    save_dir = Path(results_dir) if results_dir else RESULTS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(log)
    csv_path = save_dir / 'twin_log.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n  ✓ Log saved → {csv_path}")

    _print_summary(df, sim_days)
    plot_results(df, dt, save_dir=save_dir)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Summary statistics
# ──────────────────────────────────────────────────────────────────────────────
def _print_summary(df: pd.DataFrame, sim_days: int):
    dt_s = float(df['dt_s'].iloc[0]) if 'dt_s' in df.columns else 300.0
    dt_h = dt_s / 3600.0
    print(f"\n  {'─'*80}")
    print(f"  Compliance summary ({sim_days} day(s))")
    print(f"  {'─'*80}")
    for day in range(sim_days):
        d = df[(df['time_s'] >= day * 86400) & (df['time_s'] < (day+1) * 86400)]
        if d.empty:
            continue
        lights_h = d['u_lights'].sum() * dt_h
        pump_h   = d['u_pump'].sum()   * dt_h
        temp_ok  = ((d['temp'] * 9/5 + 32).between(80 - 5.4, 80 + 5.4)).mean() * 100
        vent_on  = d['u_vent'].sum() * dt_h
        print(f"  Day {day+1}: lights={lights_h:.2f}h (target 4.0h) | "
              f"pump={pump_h:.2f}h (target 0.5h) | "
              f"temp in band={temp_ok:.1f}% | vent open={vent_on:.2f}h")

    final_soh   = df['battery_soh'].iloc[-1]   if 'battery_soh'    in df.columns else 1.0
    battery_cyc = df['battery_cycles'].iloc[-1] if 'battery_cycles' in df.columns else 0.0
    total_pv_kwh   = df['p_pv'].sum()   * dt_h / 1000.0
    total_load_kwh = df['p_load'].sum() * dt_h / 1000.0
    inv_loss_kwh   = df['p_inv_loss'].sum() * dt_h / 1000.0 if 'p_inv_loss' in df.columns else 0.0
    dcdc_loss_kwh  = df['p_dcdc_loss'].sum() * dt_h / 1000.0 if 'p_dcdc_loss' in df.columns else 0.0
    print(f"  {'─'*80}")
    print(f"  Battery: SoH={final_soh*100:.4f}% | Cycles={battery_cyc:.4f}")
    print(f"  Energy:  PV={total_pv_kwh:.2f} kWh | Load={total_load_kwh:.2f} kWh | "
          f"Inv.Loss={inv_loss_kwh:.3f} kWh | DC-DC.Loss={dcdc_loss_kwh:.3f} kWh")
    print(f"  {'─'*80}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────
def plot_results(df: pd.DataFrame, dt_s: float = 300.0,
                 save_dir: Optional[Path] = None):
    t_h    = df['time_s'] / 3600.0
    temp_f = df['temp'] * 9.0 / 5.0 + 32.0
    amb_f  = df['t_ambient'] * 9.0 / 5.0 + 32.0
    has_inv   = 'p_inv_loss'  in df.columns and df['p_inv_loss'].notna().any()
    has_soh   = 'battery_soh' in df.columns and df['battery_soh'].notna().any()
    has_solve = df['solve_time_ms'].sum() > 0

    fig, axes = plt.subplots(6, 1, figsize=(14, 22), sharex=True)
    fig.suptitle('Solar Greenhouse Digital Twin — Simulation Results',
                 fontsize=15, fontweight='bold')
    ax_i = iter(axes)

    ax = next(ax_i)
    ax.plot(t_h, df['soc_true'] * 100, 'g-',  lw=1.8, label='SoC (true)')
    ax.plot(t_h, df['soc_est']  * 100, 'b--', lw=1.0, label='SoC (est.)')
    ax.axhline(20, color='r',      ls='--', lw=1,   label='Min 20%')
    ax.axhline(90, color='orange', ls='--', lw=1,   label='Max 90%')
    ax.axhline(55, color='gray',   ls=':',  lw=0.8, label='Target 55%')
    ax.set_ylabel('Battery SoC (%)')
    ax.set_ylim(-2, 102)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = next(ax_i)
    ax.fill_between(t_h, df['p_pv'],   alpha=0.45, color='gold',  label='PV DC')
    ax.fill_between(t_h, df['p_load'], alpha=0.35, color='coral', label='Total load')
    ax.plot(t_h, df['p_vent'],   'g-', lw=1, label='Vent motor')
    always_on_W = P_MAX[2] + P_MAX[3] + P_MAX[4]
    ax.axhline(always_on_W, color='gray', ls=':', lw=1.5, alpha=0.7,
               label=f'Always-on ({always_on_W:.0f}W)')
    if has_inv:
        ax.fill_between(t_h, df['p_inv_loss'],  alpha=0.3, color='lightsalmon', label='Inverter loss')
        ax.fill_between(t_h, df['p_dcdc_loss'], alpha=0.3, color='lightblue',   label='DC-DC loss')
    ax.set_ylabel('Power (W)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = next(ax_i)
    ax.plot(t_h, temp_f, 'r-',  lw=1.8, label='Inside (°F)')
    ax.plot(t_h, amb_f,  'b:',  lw=1.0, label='Ambient (°F)')
    ax.axhline(TARGET_F,         color='g',    ls='--', lw=1,   label=f'Target {TARGET_F:.0f}°F')
    ax.axhline(TARGET_F - 5.4,   color='gray', ls=':',  lw=0.8)
    ax.axhline(TARGET_F + 5.4,   color='gray', ls=':',  lw=0.8, label='±3°C band')
    T_cap_f = df['t_ambient'] * 9/5 + 32 + 18
    ax.plot(t_h, T_cap_f, 'm--', lw=0.8, alpha=0.5, label='T_amb+10°C cap')
    ax.set_ylabel('Temperature (°F)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = next(ax_i)
    ax.step(t_h, df['u_lights'], 'r-',  lw=1.5, label='Lights',        where='post')
    ax.step(t_h, df['u_pump'],   'b-',  lw=1.5, label='Pump',          where='post')
    ax.step(t_h, df['u_vent'],   'g-',  lw=1.5, label='Vent (open=1)', where='post')
    for i, row in df[df['is_dark'] == True].iterrows():
        ax.axvspan(row['time_s']/3600, row['time_s']/3600 + dt_s/3600,
                   alpha=0.03, color='navy')
    ax.set_ylabel('Control signal')
    ax.set_ylim(-0.05, 1.15)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = next(ax_i)
    if has_solve:
        ax.plot(t_h, df['solve_time_ms'], 'purple', lw=1, label='Solve time (ms)')
        ax.set_ylabel('MPC solve time (ms)')
    else:
        ax.text(0.5, 0.5, 'Rule-based controller — no MPC solve',
                ha='center', va='center', transform=ax.transAxes)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = next(ax_i)
    if has_soh:
        ax2 = ax.twinx()
        ax2.plot(t_h, df['battery_soh'] * 100, 'g-', lw=1.5, label='SoH (%)')
        ax2.set_ylabel('State of Health (%)', color='g')
        ax2.legend(loc='lower right', fontsize=8)
    if has_inv:
        ax.fill_between(t_h, df['p_inv_loss'].cumsum() * dt_s/3600/1000,
                        alpha=0.3, color='gray', label='Cum. inv. loss (kWh)')
        ax.set_ylabel('Cumulative Inverter Loss (kWh)')
        ax.legend(loc='upper left', fontsize=8)
    ax.set_xlabel('Time (hours from simulation start)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_dir = Path(save_dir) if save_dir else RESULTS_DIR
    out = out_dir / 'twin_results.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"  ✓ Plot saved → {out}")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Controller comparison
# ──────────────────────────────────────────────────────────────────────────────
def compare_controllers(
    sim_days: int = 1,
    dt_minutes: int = 5,
    battery_wh: float = 2000.0,
    pv_wpeak: float = 342.0,
    inverter_w: float = 500.0,
    n_scenarios: int = 10,
    horizon_steps: int = 12,
    weather_method: str = 'historical',
    results_dir: Optional[Path] = None,
    use_matlab: bool = False,
    matlab_eng: Any = None,
    matlab_model: str = 'greenhouse_twin_model',
):
    print(f"\n{'='*72}")
    print(f"  CONTROLLER COMPARISON — 4 controllers × {sim_days} day(s)")
    print(f"  Battery: {battery_wh:.0f} Wh | PV: {pv_wpeak:.0f} Wp | Inverter: {inverter_w:.0f} W")
    print(f"{'='*72}\n")

    controllers = ['milp', 'mpc', 'deterministic', 'rule']
    labels      = ['MILP', 'Stochastic MPC', 'Deterministic MPC', 'Rule-Based']
    results     = {}

    for ctrl, label in zip(controllers, labels):
        if ctrl == 'milp' and not HAS_MILP:
            print("  [MILP] Skipping — pulp not available")
            continue
        print(f"\n  ── Running {label} ──")
        df = run(
            sim_days       = sim_days,
            n_scenarios    = n_scenarios,
            horizon_steps  = horizon_steps,
            dt_minutes     = dt_minutes,
            battery_wh     = battery_wh,
            pv_wpeak       = pv_wpeak,
            inverter_w     = inverter_w,
            controller     = ctrl,
            weather_method = weather_method,
            use_matlab     = use_matlab,
            matlab_eng     = matlab_eng,
            matlab_model   = matlab_model,
            results_dir    = results_dir / ctrl if results_dir else None,
        )
        dt_h = dt_minutes / 60.0
        results[ctrl] = {
            'df':             df,
            'mean_soc':       df['soc_true'].mean() * 100,
            'min_soc':        df['soc_true'].min() * 100,
            'temp_in_band':   ((df['temp'] * 9/5 + 32).between(80 - 5.4, 80 + 5.4)).mean() * 100,
            'pump_h':         df['u_pump'].sum()   * dt_h,
            'lights_h':       df['u_lights'].sum() * dt_h,
            'total_pv_kwh':   df['p_pv'].sum()    * dt_h / 1000.0,
            'total_load_kwh': df['p_load'].sum()  * dt_h / 1000.0,
            'final_soh':      df['battery_soh'].iloc[-1]   if 'battery_soh'    in df.columns else 1.0,
            'battery_cycles': df['battery_cycles'].iloc[-1] if 'battery_cycles' in df.columns else 0.0,
            'max_temp_f':     df['temp'].max() * 9/5 + 32,
            'min_temp_f':     df['temp'].min() * 9/5 + 32,
        }

    _print_comparison_table(results, labels)
    plot_comparison(results, labels, dt_minutes, save_dir=results_dir)
    return results


def _print_comparison_table(results: dict, labels: list):
    print(f"\n{'='*100}")
    print(f"  CONTROLLER COMPARISON SUMMARY")
    print(f"{'='*100}")
    metrics = [
        ('Mean SoC (%)',       'mean_soc'),
        ('Min SoC (%)',        'min_soc'),
        ('Temp in band (%)',   'temp_in_band'),
        ('Pump run (h/day)',   'pump_h'),
        ('Lights run (h/day)', 'lights_h'),
        ('Total PV (kWh)',     'total_pv_kwh'),
        ('Total Load (kWh)',   'total_load_kwh'),
        ('Max Temp (°F)',      'max_temp_f'),
        ('Min Temp (°F)',      'min_temp_f'),
        ('Battery Cycles',     'battery_cycles'),
        ('Final SoH',          'final_soh'),
    ]
    header = f"  {'Metric':<25}"
    for label in labels:
        header += f"  {label:>20}"
    print(header)
    for name, key in metrics:
        row = f"  {name:<25}"
        for ctrl in results:
            val = results[ctrl][key]
            fmt = f"{val:>19.4f}" if key == 'final_soh' else \
                  f"{val:>19.2f}" if key == 'battery_cycles' else \
                  f"{val:>19.1f}"
            row += f"  {fmt}"
        print(row)
    print(f"{'='*100}\n")


def plot_comparison(results: dict, labels: list, dt_minutes: int = 5,
                    save_dir: Optional[Path] = None):
    dt_h = dt_minutes / 60.0
    fig, axes = plt.subplots(3, 2, figsize=(18, 14))
    fig.suptitle('Controller Comparison — Greenhouse Digital Twin',
                 fontsize=15, fontweight='bold')
    colors  = {'milp': '#9C27B0', 'mpc': '#2196F3',
                'deterministic': '#FF9800', 'rule': '#4CAF50'}
    markers = {'milp': '-.', 'mpc': '-', 'deterministic': '--', 'rule': ':'}

    axes[0, 0].axhline(5,  color='r',      ls='--', lw=0.8, alpha=0.5)
    axes[0, 0].axhline(90, color='orange', ls='--', lw=0.8, alpha=0.5)
    axes[0, 1].axhline(TARGET_F, color='g', ls='--', lw=0.8, alpha=0.5)

    for ctrl in results:
        df  = results[ctrl]['df']
        t_h = df['time_s'] / 3600.0
        c   = colors.get(ctrl, '#333')
        m   = markers.get(ctrl, '-')
        idx = list(results.keys()).index(ctrl)
        lbl = labels[idx] if idx < len(labels) else ctrl

        axes[0, 0].plot(t_h, df['soc_true'] * 100, c, ls=m, lw=1.5, label=lbl)
        axes[0, 0].set_ylabel('SoC (%)')
        axes[0, 0].set_title('Battery State of Charge')

        temp_f = df['temp'] * 9/5 + 32
        axes[0, 1].plot(t_h, temp_f, c, ls=m, lw=1.5, label=lbl)
        axes[0, 1].set_ylabel('Temp (°F)')
        axes[0, 1].set_title('Inside Temperature')

        axes[1, 0].fill_between(t_h, df['p_pv'],  alpha=0.15, color=c, label=f'{lbl} PV')
        axes[1, 0].plot(t_h, df['p_load'], c, ls=m, lw=1.2, label=f'{lbl} Load')
        axes[1, 0].set_ylabel('Power (W)')
        axes[1, 0].set_title('Generation & Load')

        cum_pv   = df['p_pv'].cumsum()   * dt_h / 1000.0
        cum_load = df['p_load'].cumsum() * dt_h / 1000.0
        axes[1, 1].plot(t_h, cum_pv,   c, ls=m, lw=1.5, label=f'{lbl} PV')
        axes[1, 1].plot(t_h, cum_load, c, ls=m, lw=1.5, label=f'{lbl} Load', alpha=0.6)
        axes[1, 1].set_ylabel('Cumulative Energy (kWh)')
        axes[1, 1].set_title('Energy Balance')

        axes[2, 0].step(t_h, df['u_vent'],   c, ls=m, lw=1.2, label=f'{lbl} Vent',   where='post')
        axes[2, 0].step(t_h, df['u_lights'], c, ls=m, lw=0.8, label=f'{lbl} Lights', where='post', alpha=0.5)
        axes[2, 0].set_ylabel('Control (0/1)')
        axes[2, 0].set_title('Vent & Lights')
        axes[2, 0].set_ylim(-0.05, 1.15)

        if 'p_inv_loss' in df.columns and df['p_inv_loss'].notna().any():
            cum_inv = df['p_inv_loss'].cumsum() * dt_h / 1000.0
            axes[2, 1].plot(t_h, cum_inv, c, ls=m, lw=1.5, label=f'{lbl} Inv.loss')

    if results:
        first_df = list(results.values())[0]['df']
        t_h_f    = first_df['time_s'] / 3600.0
        amb_f    = first_df['t_ambient'] * 9/5 + 32
        axes[0, 1].plot(t_h_f, amb_f, 'k:', lw=1.0, alpha=0.5, label='Ambient')
        if 'battery_soh' in first_df.columns:
            ax_soh = axes[2, 1].twinx()
            ax_soh.plot(t_h_f, first_df['battery_soh'] * 100, 'g-', lw=1.5, alpha=0.7, label='SoH')
            ax_soh.set_ylabel('SoH (%)')
            ax_soh.legend(fontsize=8, loc='lower left')

    for ax in axes.flat:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[2, 1].set_ylabel('Cumulative Inv. Loss (kWh)')
    axes[2, 1].set_title('Inverter Losses & Battery SoH')

    plt.tight_layout()
    out_dir = Path(save_dir) if save_dir else RESULTS_DIR
    out = out_dir / 'controller_comparison.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"  ✓ Comparison plot saved → {out}")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Solar Greenhouse Digital Twin')
    parser.add_argument('--sim-days',   type=int,   default=1)
    parser.add_argument('--scenarios',  type=int,   default=10)
    parser.add_argument('--horizon',    type=int,   default=12)
    parser.add_argument('--dt_minutes', type=int,   default=5)
    parser.add_argument('--weather',    default='historical')
    parser.add_argument('--matlab',     action='store_true')
    parser.add_argument('--controller', default='mpc',
                        help="mpc | deterministic | rule | milp")
    parser.add_argument('--compare',    action='store_true')
    parser.add_argument('--battery-wh', type=float, default=2000)
    parser.add_argument('--pv-wpeak',   type=float, default=342)
    parser.add_argument('--inverter-w', type=float, default=500)
    args = parser.parse_args()

    hw_label = f'{int(args.battery_wh)}Wh_{int(args.pv_wpeak)}Wp_{int(args.inverter_w)}W'
    out_dir  = RESULTS_DIR / hw_label
    out_dir.mkdir(parents=True, exist_ok=True)

    eng, model = (None, 'greenhouse_twin_model')
    if args.matlab:
        model_path = Path(__file__).parent / 'greenhouse_twin_model.slx'
        eng, model = setup_matlab(model_path)

    try:
        if args.compare:
            compare_controllers(
                sim_days       = args.sim_days,
                dt_minutes     = args.dt_minutes,
                n_scenarios    = args.scenarios,
                horizon_steps  = args.horizon,
                battery_wh     = args.battery_wh,
                pv_wpeak       = args.pv_wpeak,
                inverter_w     = args.inverter_w,
                use_matlab     = args.matlab,
                matlab_eng     = eng,
                matlab_model   = model,
                results_dir    = out_dir,
            )
        else:
            run(
                sim_days       = args.sim_days,
                n_scenarios    = args.scenarios,
                horizon_steps  = args.horizon,
                use_matlab     = args.matlab,
                weather_method = args.weather,
                dt_minutes     = args.dt_minutes,
                battery_wh     = args.battery_wh,
                pv_wpeak       = args.pv_wpeak,
                inverter_w     = args.inverter_w,
                controller     = args.controller,
                results_dir    = out_dir,
                matlab_eng     = eng,
                matlab_model   = model,
            )
    finally:
        if args.matlab and eng is not None:
            teardown_matlab(eng)
