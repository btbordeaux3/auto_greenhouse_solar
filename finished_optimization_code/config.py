"""
config.py — System configuration for the greenhouse optimizer.

All constants derived from real hardware measurements and component datasheets.
Settings can be overridden via environment variables (for Docker/cloud deployment).
"""

import os

# ──────────────────────────────────────────────────────────────────────────────
# Cloudflare Worker endpoint
# ──────────────────────────────────────────────────────────────────────────────
ENDPOINT_URL = os.environ.get(
    "GREENHOUSE_ENDPOINT_URL",
    "https://greenhouse-api.ffnfghnhzt.workers.dev/",
)
ENDPOINT_PASSWORD = os.environ.get("GREENHOUSE_ENDPOINT_PASSWORD")
if not ENDPOINT_PASSWORD:
    raise RuntimeError(
        "GREENHOUSE_ENDPOINT_PASSWORD is not set. Set it in the environment or .env "
        "(see env.example)."
    )

# ──────────────────────────────────────────────────────────────────────────────
# Location (Havelock, NC) — for weather API
# ──────────────────────────────────────────────────────────────────────────────
LATITUDE = 34.88
LONGITUDE = -76.90
TIMEZONE = "America/New_York"

# ──────────────────────────────────────────────────────────────────────────────
# Battery: 30Ah LiFePO4 12V nominal
# ──────────────────────────────────────────────────────────────────────────────
BATTERY_AH = 30.0
BATTERY_NOM_V = 12.8
BATTERY_Wh = BATTERY_AH * BATTERY_NOM_V          # 384 Wh
BATTERY_J = BATTERY_Wh * 3600.0                   # 1,382,400 J
BATTERY_EFF = 0.95                                # LiFePO4 round-trip
BATTERY_MAX_CHARGE_A = 15                         # 0.5C for 30Ah
BATTERY_MAX_DISCHARGE_A = 30                      # 1C for 30Ah
BATTERY_MAX_CHARGE_W = BATTERY_NOM_V * BATTERY_MAX_CHARGE_A
BATTERY_MAX_DISCHARGE_W = BATTERY_NOM_V * BATTERY_MAX_DISCHARGE_A
BATTERY_CHARGE_TEMP_C = (-5, 45)
BATTERY_DISCHARGE_TEMP_C = (-20, 60)

SOC_MIN = 0.10       # hard floor — load shed above this
SOC_MAX = 0.90
SOC_TARGET = 0.55

# ──────────────────────────────────────────────────────────────────────────────
# Solar panel: 100W Renogy + Victron BlueSolar MPPT
# ──────────────────────────────────────────────────────────────────────────────
PV_WPEAK = 100.0
PV_MPPT = 0.95                                     # Victron MPPT efficiency
NOCT = 45.0                                        # Nominal Operating Cell Temp
TEMP_COEFF_PMAX = -0.004                           # /°C crystalline silicon

# ──────────────────────────────────────────────────────────────────────────────
# Power loads — measured on bench with 12V supply (ground truth from main.cpp)
# ──────────────────────────────────────────────────────────────────────────────
P_LIGHTS = 0.84 * 12.0       # 10.08 W  (light relay, 0.84A @ 12V)
P_PUMP = 0.55 * 12.0         #  6.60 W  (pump motor via H-bridge, 0.55A @ 12V)
P_VENT_MOVE = 0.12 * 12.0    #  1.44 W  (vent motor during movement, 0.12A @ 12V)
P_ESP32 = 0.08 * 12.0        #  0.96 W  (main ESP32 always-on, 0.08A @ 12V)
P_CAMERA = 0.20 * 12.0       #  2.40 W  (ESP32-CAM, only when taking picture)

# Vent motor runs for ~15.5s then stops (self-locking leadscrew).
# Energy per open/close action: 1.44W × 15.5s / 3600 = 0.006 Wh (negligible).
VENT_MOVE_TIME_S = 15.5
VENT_MOVE_ENERGY_Wh = P_VENT_MOVE * VENT_MOVE_TIME_S / 3600.0

# Pump runs as a float (0-14 minutes per 15-min cycle).
PUMP_MAX_MINUTES = 14.0
PUMP_COOLDOWN_STEPS = 4    # 4 × 15 min = 1 hour between pump bursts

# ──────────────────────────────────────────────────────────────────────────────
# Greenhouse thermal model (6×4 ft polycarbonate hobby greenhouse)
# ──────────────────────────────────────────────────────────────────────────────
FLOOR_AREA = 2.23          # m² (6×4 ft ≈ 1.83×1.22 m)
VOLUME = 3.5               # m³
GLAZING_TRANS = 0.85
RHO_AIR = 1.2              # kg/m³
CP_AIR = 1005.0            # J/(kg·K)
SOIL_DEPTH = 0.1           # m
RHO_SOIL = 1500.0          # kg/m³
CP_SOIL = 800.0            # J/(kg·K)
C_STRUCT = 80000.0         # J/K frame + glazing thermal mass

# Total thermal capacitance (air + soil + structure)
C_AIR = VOLUME * RHO_AIR * CP_AIR                     # ~4,221 J/K
C_SOIL = SOIL_DEPTH * FLOOR_AREA * RHO_SOIL * CP_SOIL # ~267,600 J/K
C_TOTAL = C_AIR + C_SOIL + C_STRUCT                    # ~352,000 J/K

# Solar gain: effective area through glazing
SOLAR_AREA = FLOOR_AREA * GLAZING_TRANS                 # ~1.90 m²

# Ventilation: natural through motorized window (NOT forced fan).
# Small greenhouse with one opening: ~15 ACH (buoyancy + wind driven).
ACH_VENT = 15.0
ACH_TO_W_K = VOLUME * RHO_AIR * CP_AIR / 3600.0      # 1.17 W/K per ACH
VENT_W_K = ACH_TO_W_K * ACH_VENT                      # ~17.6 W/K

# Natural infiltration — hobby greenhouse has gaps at panels/doors.
# ~3 ACH realistic for polycarbonate panels with imperfect seals.
ACH_INFIL = 3.0
INFIL_W_K = ACH_TO_W_K * ACH_INFIL                    # ~3.5 W/K

# Night-sky radiative cooling
RAD_W_K = 2.0                                          # W/K linearised

# Ground coupling
GND_W_K = 1.5                                          # W/K
T_GROUND_C = 15.0                                      # NC average

# Derived thermal rate constants (°C/s per W or per °C delta)
ALPHA_SOLAR = SOLAR_AREA / C_TOTAL                     # ~5.40e-6
ALPHA_VENT = VENT_W_K / C_TOTAL                        # ~5.00e-5
ALPHA_INFIL = INFIL_W_K / C_TOTAL                      # ~1.67e-6
ALPHA_RAD = RAD_W_K / C_TOTAL                          # ~5.68e-6
ALPHA_INT = 1.0 / C_TOTAL                              # ~2.84e-6
ALPHA_GND = GND_W_K / C_TOTAL                          # ~4.26e-6

T_SKY_OFFSET = 10.0        # T_sky = T_ambient - 10°C (clear night)
T_MAX_ABOVE_AMBIENT = 10.0

# ──────────────────────────────────────────────────────────────────────────────
# Temperature targets
# 80°F (26.67°C) target. Wide band because small greenhouse in full sun
# can't always maintain setpoint — solar gain overwhelms ventilation.
# ──────────────────────────────────────────────────────────────────────────────
TARGET_C = (80.0 - 32.0) * 5.0 / 9.0   # 26.67°C / 80°F
TEMP_BAND = 5.0                          # ±5°C (loose — hard to achieve)

# ──────────────────────────────────────────────────────────────────────────────
# Scheduling — hybrid resolution
# First 24h: 15-min steps (96 steps) for fine control
# Next 24h: hourly steps (24 steps) for long-range planning
# ──────────────────────────────────────────────────────────────────────────────
DT_S = float(os.environ.get("GREENHOUSE_INTERVAL", "900.0"))  # 15 minutes (base / fine step)
DT_H = DT_S / 3600.0                     # 0.25 hours
HORIZON_DAYS = 4
HORIZON_STEPS = 384                      # full 15-min resolution (for weather fetch)
N_FINE = 96                              # first 24h at 15-min
N_COARSE = 72                            # next 3 days at 1-hour
N_HYBRID = N_FINE + N_COARSE             # 168 total steps

# Daily pump target: minutes per day (adjusted by plant state)
# Two modes: germinating (no plant) and growing (has plant).
# Harvest-ready = alert only, same objectives as growing.
PUMP_DAILY_MINUTES_GERMINATING = 3.0
PUMP_DAILY_MINUTES_GROWING = 12.0

# Daily lights target: hours per day (nighttime only)
LIGHTS_DAILY_HOURS_GERMINATING = 0.0   # no lights for germinating
LIGHTS_DAILY_HOURS_GROWING = 6.0

# ──────────────────────────────────────────────────────────────────────────────
# Solar schedule (Havelock, NC — approximate)
# ──────────────────────────────────────────────────────────────────────────────
SUNRISE = 6.0
SUNSET = 18.0

# ──────────────────────────────────────────────────────────────────────────────
# Plant classifier — midday-only scheduling
# Only classify and take photos during midday hours when lighting is good.
# 3 classifications per day: ~10:00, ~11:20, ~12:40 (every 80 min in window).
# ──────────────────────────────────────────────────────────────────────────────
CLASSIFY_MIDDAY_START = 10    # earliest local hour to classify (10 AM)
CLASSIFY_MIDDAY_END = 14      # latest local hour to classify (2 PM)
CLASSIFY_INTERVAL_MIN = 80    # minutes between classifications within midday window

# ──────────────────────────────────────────────────────────────────────────────
# MPC objective weights
# Priority: SOC safety >> temperature >> pump/lights fulfillment > efficiency
# ──────────────────────────────────────────────────────────────────────────────
W_SOC_CRIT = 1_000.0             # SOC below SOC_MIN: catastrophic
W_SOC_FULL = 1_000.0             # SOC above SOC_MAX
W_SOC_TRACK = 50.0               # soft tracking toward SOC_TARGET (energy planner)
W_SOC_FORECAST = 20.0            # forecast-aware: relax target when sun coming, tighten when cloudy
W_TEMP = 500.0                   # temperature squared deviation
W_BIN_LIGHTS = 100.0             # binary corner-pushing for lights
W_BIN_VENT = 100.0               # binary corner-pushing for vent
W_VENT_MOVE = 300.0              # vent actuator wear
W_RAMP = 10.0                    # smooth control transitions
W_SLACK_SOC = 5_000.0            # SOC soft constraint escape
W_SLACK_TEMP = 1_000.0           # temp soft constraint escape
W_PUMP_ENERGY = 100.0            # penalize excessive pump energy
W_REQUIRE = 1_000.0              # penalty for not meeting pump/lights targets

# Lights as SOC management: when SOC is high, running lights absorbs
# excess solar energy AND helps plants. Negative weight = reward.
W_LIGHTS_ABSORB = 500.0          # per-step reward: lights * (SOC - 0.70)

# Soil-moisture pump scaling: dry soil → pump more
SOIL_DRY_THRESHOLD = 7.5          # below this → dry, scale up pump
SOIL_WET_THRESHOLD = 20.0         # above this → wet, scale down pump
SOIL_SCALE_MAX = 2.0             # max pump multiplier when bone dry

# ──────────────────────────────────────────────────────────────────────────────
# Soil moisture model
# Simple bucket model: pump adds water, evapotranspiration removes it.
# Soil moisture is 0-100% on sensor (0=dry, 100=saturated).
# ──────────────────────────────────────────────────────────────────────────────
SOIL_FIELD_CAPACITY = 80.0        # % — sensor reading at saturation (0-100% scale)
SOIL_WILTING_POINT = 15.0         # % — below this, plants wilt
SOIL_MAX_PUMP_RATE = 2.5          # % per minute of pumping
SOIL_ET_RATE = 0.15               # % per hour — evapotranspiration rate (daytime)
SOIL_ET_NIGHT_SCALE = 0.3         # nighttime ET is 30% of daytime
SOIL_DRAIN_RATE = 0.05            # % per hour — gravity drainage above field capacity
SOIL_RAIN_SCALE = 0.001           # % per W/m² of GHI
SOIL_INIT_DEFAULT = float(os.environ.get("GREENHOUSE_SOC_INIT", "40.0"))  # % — default initial soil moisture if unknown

# ──────────────────────────────────────────────────────────────────────────────
# MPC objective weights for soil moisture
# ──────────────────────────────────────────────────────────────────────────────
W_SOIL_TRACK = 5_000.0           # tracking soil moisture target (DOMINANT objective)
W_SOIL_DRY = 1_000.0             # penalty below wilting point (catastrophic)
W_SOIL_WET = 500.0               # penalty above field capacity (waterlogged)
W_PUMP_DAY_BONUS = 200.0         # reward pumping during daytime
W_PUMP_NIGHT_PENALTY = 100.0     # penalty pumping at nighttime
W_PUMP_CLOSE = 200.0             # penalty for pumping within cooldown window
W_VENT_DAY_BONUS = 5_000.0       # reward venting during daytime (CO2, airflow)

# ──────────────────────────────────────────────────────────────────────────────
# Plant state definitions
# Per-stage targets for pump, lights, temperature, and soil moisture.
# ──────────────────────────────────────────────────────────────────────────────
PLANT_STATES = {
    "GERMINATING": {
        "pump_daily_min": PUMP_DAILY_MINUTES_GERMINATING,
        "lights_daily_h": LIGHTS_DAILY_HOURS_GERMINATING,
        "temp_target_c": 28.0,      # warmer for germination
        "temp_band_c": 4.0,
        "soil_target": 50.0,        # 50% of FC (0-100% scale)
        "soil_band": 10.0,          # ±10% range
        "alert_user": False,
        "description": "Seeds sprouting, light water, no lights needed",
    },
    "GROWING": {
        "pump_daily_min": PUMP_DAILY_MINUTES_GROWING,
        "lights_daily_h": LIGHTS_DAILY_HOURS_GROWING,
        "temp_target_c": TARGET_C,
        "temp_band_c": TEMP_BAND,
        "soil_target": 65.0,        # 65% of FC (0-100% scale)
        "soil_band": 10.0,          # ±10% range
        "alert_user": False,
        "description": "Active growth, high water and light",
    },
    "HARVEST_READY": {
        "pump_daily_min": 3.0,      # reduce watering before harvest
        "lights_daily_h": LIGHTS_DAILY_HOURS_GROWING,
        "temp_target_c": 25.0,      # slightly cooler for harvest
        "temp_band_c": 5.0,
        "soil_target": 55.0,        # 55% of FC (0-100% scale)
        "soil_band": 10.0,          # ±10% range
        "alert_user": True,
        "description": "Harvest ready — dry out, alert user",
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# Data logging
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR = "data"
LOG_DB = "data/greenhouse.db"
LOG_CSV_DIR = "data"

# ──────────────────────────────────────────────────────────────────────────────
# MPC solver settings
# ──────────────────────────────────────────────────────────────────────────────
MPC_VERBOSE = False
MPC_MAX_ITER = 10000
MPC_TOL = 1e-4
N_SCENARIOS = 12      # number of weather scenarios for stochastic MPC
