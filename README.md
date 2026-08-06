# Solar Greenhouse Digital Twin — 6×4 ft

A high-fidelity physics-based digital twin of a small 6×4 ft hobby greenhouse, designed to size and compare off-grid solar + battery power systems across five controllers: **Stochastic MPC**, **Deterministic MPC**, **MILP**, **Rule-based**, and **RL (PPO)**.

The model includes realistic thermal dynamics (air, structure, soil), an hourly weather engine with forecast uncertainty, battery degradation (cycle-based SoH), DC-DC converter losses, and native 12V DC loads. It was used to find the **cheapest viable off-grid build** for a small greenhouse with grow lights, water pump, automated venting, and always-on monitoring/comms.

---

## Cheapest Viable Build (All-DC, No Inverter)

**Total cost: ~$480** (energy system marginal cost: **$222**)

### Key Insight: Pi Zero 2W is the Linchpin

The single biggest cost-saving change was swapping the **Raspberry Pi 4 (7W) → Pi Zero 2W (2W)**. This cuts the always-on base load from **8.2W → 2.8W**, meaning a **30Ah battery works where a 50Ah was needed before**. The Pi Zero is $35 (vs $65 for RPi 4), and the battery is $100 (vs $180), saving **$110** total while maintaining viability.

### Parts Breakdown

| Part | Model | Power | Price |
|---|---|---|---|
| **Battery** | LiTime 12V 30Ah LiFePO4 | 384 Wh | **$100** |
| **Solar Panel** | HQST 100W monocrystalline | 100 Wp | **$75** |
| **DC-DC Converter** | Recom R-78B5.0-2.0 (12V→5V, 10W) | 90% eff at 2.4W | **$12** |
| **Grow Light** | Barrina LED T5 12V (full spectrum) | 30 W | **$25** |
| **Water Pump** | Active Aqua AAPW15 | 15 W | **$18** |
| **Vent Actuator** | Firgelli L12-P + ESP32-C3 controller | 1.5 W avg | **$45** |
| **Controller** | Raspberry Pi Zero 2W + DS3231 RTC | 2.0 W | **$35** |
| **Sensors** | BME280 + VEML7700 + INA219 + INA226 + DS18B20 | 16 mW | **$50** |
| **Comms** | LoRaWAN RFM95W (duty-cycled) | 0.3 W avg | **$20** |
| **MPPT Controller** | EPEver 10A or similar | 95% eff | **$60** |
| **Wiring/fuses** | Misc hardware | — | **$25** |
| | | **Total** | **~$480** |

### Energy System Marginal Cost
If you already have sensors/controller/comms, the marginal cost to add solar power is just the energy system:

    Battery (LiTime 30Ah)       $100
    Solar panel (HQST 100W)      $75
    DC-DC converter (Recom)      $12
    MPPT charge controller       $60
    Wiring/misc                  $25
    ─────────────────────────
    Marginal energy cost        $272

Or just the bare minimum for the sizing optimizer (`battery + PV + DC-DC`): **$187**.

### Alternative: Use RPi 4 (need 50Ah battery)

If you already own a Raspberry Pi 4 or need the extra compute:

| Component | Cost |
|---|---|
| Dakota Lithium DL+ 12V 50Ah | $180 |
| Renogy Eclipse 100W | $90 |
| DC-DC + MPPT + wiring | $97 |
| **Total energy marginal** | **$367** |

The Pi Zero 2W saves **$110** ($85 cheaper battery + $30 cheaper controller + $15 cheaper panel) while providing identical control capability for this application.

---

## Design Evolution

### Original Design (before optimization)
- 100 Wp grow light (Spider Farmer SF-300, AC) requiring a 300 W inverter ($120)
- 80 W water pump (Active Aqua AAPWC25, AC) requiring inverter
- Massive 360 W Victron Orion-TR DC-DC converter ($160) running at 44% efficiency on the 7.4 W always-on rail, wasting 12 W continuously
- Always-on base load of **22 W** (RPi 7 W + sensors 0.1 W + comms 2.5 W + DC-DC loss 12 W)
- Inverter standby + conversion losses: ~7.5 W at typical load

### All-DC Redesign
- **Removed inverter entirely** — all loads are native 12 V DC, connecting directly to the battery bus
- **Scaled down lights** to 30 W 12 V DC LED (Barrina T5) — less heat, no inverter, $25 vs $120
- **Scaled down pump** to 15 W 12 V DC (Active Aqua AAPW15) — $18 vs $50, adequate for 50 GPH drip irrigation
- **Replaced DC-DC converter** with Recom R-78B5.0-2.0 ($12, 91% efficient at 7.4 W) — eliminated 12 W of continuous waste
- **Always-on base load reduced** from 22 W → **8 W** (RPi 4 config) or **2.8 W** (Pi Zero 2W config)
- **Added MPPT charge controller** (95% efficiency) for PV → battery conversion and direct load passthrough
- **Inverter loss eliminated**: 0.000 kWh confirmed in all simulations

---

## Power Budget

### Pi Zero 2W Config (Recommended — Cheapest)
| Load | Type | Power | Voltage | Run Time/day | Energy/day |
|---|---|---|---|---|---|
| Grow light (Barrina T5 12V) | Controllable | 30 W | 12 V DC | 4 h (night) | 120 Wh |
| Water pump (AAPW15) | Controllable | 15 W | 12 V DC | 0.5 h | 7.5 Wh |
| Vent actuator (Firgelli L12-P) | Controllable | 2 W avg* | 12 V DC | as needed | ~10 Wh |
| Controller (Pi Zero 2W) | Always-on | 2.0 W | 5 V via DC-DC | 24 h | 48 Wh |
| Sensors (6x I2C/1-wire) | Always-on | 0.016 W | 5 V via DC-DC | 24 h | 0.4 Wh |
| Comms (LoRa RFM95W) | Always-on | 0.3 W | 5 V via DC-DC | 24 h | 7.2 Wh |
| DC-DC converter loss | Always-on | 0.4 W | — | 24 h | 9.6 Wh |
| **Total** | | **~4.7 W base + loads** | | | **~203 Wh/day** |

### RPi 4 Config (More Compute)
| Load | Type | Power | Voltage | Run Time/day | Energy/day |
|---|---|---|---|---|---|
| Grow light (Barrina T5 12V) | Controllable | 30 W | 12 V DC | 4 h (night) | 120 Wh |
| Water pump (AAPW15) | Controllable | 15 W | 12 V DC | 0.5 h | 7.5 Wh |
| Vent actuator (Firgelli L12-P) | Controllable | 2 W avg* | 12 V DC | as needed | ~10 Wh |
| Controller (RPi 4) | Always-on | 7.0 W | 5 V via DC-DC | 24 h | 168 Wh |
| Sensors (6x I2C/1-wire) | Always-on | 0.016 W | 5 V via DC-DC | 24 h | 0.4 Wh |
| Comms (LoRa RFM95W) | Always-on | 0.3 W | 5 V via DC-DC | 24 h | 7.2 Wh |
| DC-DC converter loss | Always-on | 0.9 W | — | 24 h | 21.6 Wh |
| **Total** | | **~10.2 W base + loads** | | | **~335 Wh/day** |

\* Vent actuator draws 6 W peak for ~4 s during movement; average over 30-min timestep is negligible. The 2 W figure is a conservative continuous equivalent including ESP32-C3 controller idle draw.

The Pi Zero 2W cuts daily energy demand by **~40%** (203 Wh/day vs 335 Wh/day), enabling a **30Ah battery instead of 50Ah** — the single most impactful design decision.

### Why No Inverter Works
- **Grow light**: 30 W 12 V DC LED (Barrina T5) — no conversion needed
- **Water pump**: 15 W 12 V DC (Active Aqua AAPW15) — no conversion needed
- **Vent actuator**: 12 V DC linear actuator (Firgelli L12-P) — no conversion needed
- **Controller/sensors/comms**: 5 V via dedicated Recom R-78B5.0-2.0 DC-DC converter ($12, 91% eff)

The old design assumed all "grow lights" and "pumps" are 120 V AC appliances requiring an inverter. In practice, 12 V DC versions of all these components exist at similar or lower prices.

---

## Sizing Optimization Results

We simulated winter days at hourly resolution with 2-day lookahead, 3 weather scenarios. Two hardware configurations were evaluated:

### Minimum Build (Recommended): 384 Wh / 100 Wp / Pi Zero 2W

**Rule-Based — 10-Day Simulation**

| Metric | Value | Target |
|---|---|---|
| Min SOC | **11.0%** | > 5% ✓ |
| Mean SOC | 82.0% | — |
| Lights | 40.0 h (4.0 h/day) | 40 h ✓ |
| Pump | 10.0 h (1.0 h/day) | 5 h ✓ |
| Temp in band | 29.6% | — |
| PV / day | 534 Wh | — |
| Load / day | **226 Wh** | — |
| DC-DC loss / day | **10.5 Wh** | — |

**Deterministic MPC — 5-Day Simulation**

| Metric | Value | Target |
|---|---|---|
| Min SOC | **6.9%** | > 5% ✓ |
| Mean SOC | 78.1% | — |
| Lights | 20.0 h (4.0 h/day) | 20 h ✓ |
| Pump | 5.0 h (1.0 h/day) | 2.5 h ✓ |
| Temp in band | **49.2%** | — |

### Standard Build: 640 Wh / 100 Wp / RPi 4 — 10-Day Controller Comparison

| Controller | Min SOC | Temp in Band | Lights | Pump | PV/day | Load/day | Inv Loss |
|---|---|---|---|---|---|---|---|
| **MPC** | **17.0%** | **55.0%** | 4.0 h ✓ | 1.0 h | 526 Wh | 354 Wh | **0 Wh** |
| Deterministic | 17.0% | 48.3% | 4.0 h ✓ | 1.0 h | 526 Wh | 356 Wh | 0 Wh |
| Rule-Based | 19.5% | 31.7% | 4.0 h ✓ | 1.0 h | 534 Wh | 356 Wh | 0 Wh |
| RL (PPO) | 39.2% | 12.1% | 0.0 h ✗ | 0.5 h | 534 Wh | 219 Wh | 0 Wh |
| MILP | 0.0% ✗ | 34.6% | 2.8 h | 2.1 h | 412 Wh | 336 Wh | 0 Wh |

**Key findings:**
- **MPC is the best controller** — meets all load targets, maintains 17% minimum SOC, best temperature compliance (55%)
- **Rule-based is safest** — 19.5% min SOC, hits all load targets, but temperature control is worse (31.7%)
- **Pi Zero 2W cuts 40% of daily load** (226 Wh/day vs 354 Wh/day) vs RPi 4
- **RL needs retraining** for the Pi Zero config — the 200k-step PPO agent was trained at 7W
- **MILP still fails** with RPi 4 — battery hits 0% SOC by day 3
- **All controllers confirm 0 Wh inverter loss**

### Battery Size Comparison (Rule, 3-day winter, 100 W PV + Pi Zero 2W)

| Battery | Cost | Min SOC | Avg SOC | Lights/day | Viable? |
|---|---|---|---|---|---|
| LiTime 12V 30Ah (384 Wh) | **$187** | **11.0%** | 82.0% | 4.0 h ✓ | **Yes** |
| LiTime 12V 30Ah (384 Wh) | $187 | 0.0%* | — | — | **No (RPi 4)** |
| DL+ 12V 50Ah (640 Wh) | $282 | 17.1% | 71% | 4.0 h | Yes |
| DL+ 12V 50Ah (640 Wh) | $282 | 49.2% | 88% | 4.0 h | Yes (Pi Zero) |

\* 20Ah (256 Wh) with Pi Zero hits 0% — 30Ah is the minimum.

### Cost Comparison by Config

| Config | Battery | Controller | Panel | Energy Cost | Full BOM |
|---|---|---|---|---|---|
| **Minimum (Recommended)** | 30Ah/$100 | Pi Zero 2W/$35 | HQST 100W/$75 | **$187** | **~$480** |
| Standard (RPi 4) | 50Ah/$180 | RPi 4/$65 | Renogy 100W/$90 | $282 | ~$590 |
| Previous (Larger) | 100Ah/$280 | RPi 4/$65 | Renogy 200W/$170 | $570 | ~$880 |

### Previous (Larger) Winning Combo
Before the all-DC redesign, the size optimizer found:
- **Ren12V100Ah (1280 Wh)** + **Ren200W (200 Wp)** + **Victron 300 W inverter** = **$570**
- RL ranked #1 across all hardware combos in a 3-day sweep

The all-DC redesign + Pi Zero swap cut the cost by **71% ($570 → $187)** while maintaining viability.

---

## System Architecture

### Digital Twin

The simulation couples four physical domains:

1. **Electrical** — PV generation (temperature-corrected panel model + MPPT efficiency), battery SOC (LiFePO4 with charge/discharge limits, temperature-dependent, cycle-based degradation), DC-DC conversion (efficiency curve lookup for the Recom converter), and DC-native loads

2. **Thermal** — Lumped-capacitance greenhouse model with:
   - **Solar gain**: ~1.9 m² effective aperture, modulated by GHI
   - **Infiltration**: 0.5 ACH through glazing
   - **Ventilation**: 48 ACH when vent is open (~56 W/K conductance)
   - **Longwave radiation**: 2.0 W/K to sky (T_sky = T_amb − 10°C)
   - **Ground coupling**: 1.5 W/K to 15°C soil
   - **Internal gains**: lights + pump + controller heat
   - **Thermal mass**: air (4,221 J/K) + structure (80,000 J/K) + soil (267,600 J/K) = **352,000 J/K total**

3. **Control** — Five controllers competing to schedule lights, pump, and vent:
   - **Stochastic MPC**: CasADi + IPOPT NLP, 48-step horizon, 3 weather scenarios, continuous relaxation with binary snapping
   - **Deterministic MPC**: Single-scenario version of the same solver
   - **MILP**: Mixed-integer linear program via PuLP/CBC (binary control signals natively)
   - **Rule-Based**: Threshold-based scheduling (lights at night, pump once daily, vent at T > 80°F)
   - **RL (PPO)**: Stable-Baselines3 PPO with 15-dim observation (SOC, temp, time features, weather, pump/lights remaining, 3 forecast features)

4. **Weather** — Historical or GP-generated scenarios with realistic forecast uncertainty (random-walk noise: ~1°C temp error at 1h, ~3°C at 12h; ~14% GHI error at 1h, ~50% at 12h)

### Observation Space (RL)

15-dimensional vector:
1. SOC (normalized 0–1)
2. Temperature (normalized °C)
3. Hour sin/cos (2 features)
4. Is dark (binary)
5. GHI (normalized)
6. Ambient temperature (normalized)
7. Pump remaining steps
8. Lights remaining steps
9. Vent state
10. `temp_forecast_min_12h` — minimum temperature in next 12 hours
11. `ghi_forecast_total_24h` — total GHI in next 24 hours
12. `rem_dark_frac` — fraction of night remaining (0 = dawn)

### Action Space

5-dimensional binary (snapped at threshold 0.5):
- `u[0]` — Lights on/off
- `u[1]` — Pump on/off
- `u[2:5]` — Always-on loads (controller, sensors, comms) — forced to 1
- `u_vent` — Vent open/close (separate output)

---

## Repository Structure

```
greenhouse_digital_twin/
├── README.md                         ← This file
├── AGENTS.md                         ← Agent notes / progress tracking
├── python/
│   ├── run_twin.py                   ← Main simulation entry point, PythonPlant, run(), compare_controllers(), plotting
│   ├── real_parts.py                 ← Complete parts database with prices, specs, URLs
│   ├── size_optimizer.py             ← Sizing sweep across batteries/PV/controllers
│   ├── optimizer/
│   │   ├── stochastic_mpc.py         ← MPC solver (CasADi + IPOPT), constants, power model
│   │   ├── rl_optimizer.py           ← PPO training, inference wrapper, GreenhouseEnv
│   │   ├── deterministic_mpc.py      ← Single-scenario MPC
│   │   ├── milp_optimizer.py         ← MILP solver (PuLP/CBC)
│   │   └── rule_optimizer.py         ← Threshold-based rule controller
│   ├── weather/
│   │   ├── historical_weather.py     ← NOAA-ish weather data loader, make_noisy_forecast()
│   │   ├── scenario_generator.py     ← GP-based scenario generation, clear_sky_ghi()
│   │   └── weather_cache/            ← Cached weather data files
│   ├── sensors/
│   │   ├── state_estimator.py        ← Kalman filter, sensor configs
│   │   └── kalman.py                 ← Extended Kalman filter implementation
│   ├── results/
│   │   ├── 640Wh_100Wp_allDC/        ← 3-day MPC simulation for cheapest build
│   │   ├── 640Wh_100Wp_5d_mpc/       ← 5-day MPC simulation
│   │   ├── 640Wh_100Wp_5d_det/       ← 5-day deterministic MPC
│   │   ├── 640Wh_100Wp_10d/          ← 10-day comparison (MILP, rule, RL)
│   │   │   ├── milp/                 ←   MILP results
│   │   │   ├── rule/                 ←   Rule-based results
│   │   │   └── rl/                   ←   RL (PPO) results
│   │   ├── 1280Wh_100Wp_allDC/       ← 3-day MPC, 1280 Wh battery
│   │   ├── Ren12V100Ah_100Wp_allDC/  ← 3-day MPC, Renogy 12V 100Ah
│   │   └── rl_models/
│   │       └── ppo_greenhouse.zip    ← Trained PPO model
│   └── results/ [project root]
│       ├── rl_models/                ← RL model storage
│       └── sizing_sweep/             ← Combined sweep CSVs/plots
└── run_twin.py                       ← Root-level copy (mirrors python/run_twin.py)
```

---

## Usage

### Run a simulation

```bash
cd python

# MPC, 10 days, hourly timesteps, 2-day horizon, 640 Wh / 100 Wp
python run_twin.py --days 10 --dt 60 --horizon 48 \
    --battery-wh 640 --pv-wpeak 100 --inverter-w 0 \
    --controller mpc

# Rule-based, same hardware
python run_twin.py --days 10 --dt 60 \
    --battery-wh 640 --pv-wpeak 100 --inverter-w 0 \
    --controller rule

# RL (PPO), same hardware
python run_twin.py --days 10 --dt 60 \
    --battery-wh 640 --pv-wpeak 100 --inverter-w 0 \
    --controller rl
```

### Compare all controllers

```bash
python -c "
from pathlib import Path
from run_twin import compare_controllers
compare_controllers(
    sim_days=10, dt_minutes=60,
    battery_wh=640.0, pv_wpeak=100.0, inverter_w=0.0,
    n_scenarios=3, horizon_steps=48,
    results_dir=Path('results/my_comparison'),
)
"
```

### Train a new RL agent

```bash
# Train PPO for 200k steps on 640 Wh / 100 Wp all-DC
python -m optimizer.rl_optimizer --train --dt 60 \
    --battery 640 --pv 100 --inverter 0 --timesteps 200000

# Or from Python:
python -c "
from optimizer.rl_optimizer import train
train(total_timesteps=200000, dt_minutes=60,
      battery_wh=640.0, pv_wpeak=100.0, inverter_w=0.0)
"
```

### Run sizing sweep

```bash
# Quick sweep: all 12V batteries × 100 W PV (all-DC)
python size_optimizer.py --quick --sim-days 3 --dt 60

# Full comparison sweep (all controllers × batteries)
python size_optimizer.py --compare --quick --sim-days 5 --dt 60
```

### MATLAB/Simulink plant

```bash
# Requires MATLAB with Simulink and the greenhouse_twin_model.slx
python size_optimizer.py --quick --matlab --sim-days 10 --dt 60 --horizon 48
```

---

## Key Physics Constants

| Parameter | Value | Description |
|---|---|---|
| Greenhouse volume | 3.5 m³ | 6×4 ft × 5 ft avg height |
| Floor area | 2.23 m² | 6×4 ft |
| Thermal mass (total) | 352,000 J/K | Air 4,221 + structure 80,000 + soil 267,600 |
| Solar gain coefficient | 1.90 m² | Effective aperture × glazing transmissivity |
| Infiltration | 0.59 W/K | 0.5 ACH |
| Ventilation (open) | 56.3 W/K | 48 ACH |
| Longwave radiation | 2.0 W/K | To T_sky = T_amb − 10°C |
| Ground coupling | 1.5 W/K | To 15°C soil |
| PV temp coefficient | −0.004 /°C | Pmax derating |
| NOCT | 45°C | Nominal operating cell temperature |
| MPPT efficiency | 95% | Charge controller conversion |
| Battery round-trip | 95% | LiFePO4 |
| DC-DC efficiency | 90% (RPi 4) / 87% (Pi Zero) | Recom R-78B5.0-2.0 at 7.4 W / 2.4 W load |
| Temperature target | 26.67°C (80°F) | Comfort band ±3°C (±5.4°F) |
| SOC hard floor | 5% | System shutdown below this |
| SOC target | 55% | Preferred operating point |
| Pump duration | 30 min | Single contiguous block |
| Lights duration | 4 h | Nighttime only |

---

## Thermal Model

The greenhouse temperature evolves as:

```
dT/dt = (Q_solar + Q_infil + Q_vent + Q_rad + Q_int + Q_gnd) / C_total

Q_solar = solar_gain × GHI × is_day
Q_infil = infiltration × (T_amb − T_inside)
Q_vent  = ventilation × u_vent × (T_amb − T_inside)
Q_rad   = −radiation × max(T_inside − T_sky, 0) × is_night
Q_int   = lights_power × u_lights + pump_power × u_pump + always_on
Q_gnd   = ground × (T_ground − T_inside)

C_total = C_air + C_structure + C_soil
```

The model intentionally does not clip T_inside to T_amb + 10°C — the physics naturally capture all heat loss mechanisms (ventilation, infiltration, radiation, ground coupling). This allows accurate simulation of hot still days where a greenhouse can significantly exceed ambient temperature.

---

## Electrical Model

### Power Flow (All-DC Architecture, Minimum Build)

```
PV panel (100 Wp) → MPPT (95%) → 12 V Battery Bus
                                        │
                          ┌─────────────┼─────────────┐
                          │             │             │
                    30 W Light     15 W Pump     2 W Vent
                    (12 V DC)      (12 V DC)     (12 V DC)
                          │
                     Recom R-78B5.0-2.0 (87% eff at 2.4 W)
                          │
                     ┌────┴────┐
                     │         │
                 Pi Zero 2W  LoRa + Sensors
                  (2.0 W)    (0.4 W combined)
```

The battery draw at each timestep:

```python
p_load_12v = lights × 30 W + pump × 15 W + vent × 2 W
p_dcdc_out = 2.4 W  # Pi Zero + sensors + comms (5 V side)
dcdc_eff = dc_dc_efficiency(p_dcdc_out)  # ~87% at 24% load
p_dcdc_loss = p_dcdc_out / dcdc_eff - p_dcdc_out  # ~0.37 W
p_dcdc_input = p_dcdc_out + p_dcdc_loss  # ~2.77 W from battery

total_battery_draw = p_load_12v + p_dcdc_input
net_battery = pv_power - total_battery_draw
```

For the RPi 4:
```python
p_dcdc_out = 7.4 W   # RPi 4 + sensors + comms (5 V side)
dcdc_eff = 0.90      # ~90% at 74% load
p_dcdc_loss ≈ 0.82 W
p_dcdc_input ≈ 8.22 W
```

Inverter loss: **zero** — all loads are native 12 V DC.

---

## Battery Degradation Model

State of health tracks capacity fade linearly with throughput:

```
cycles = (cumulative_charge_Wh + cumulative_discharge_Wh) / (2 × rated_Wh)
SoH = 1.0 − 0.20 × cycles / cycle_life_to_80pct
```

Default `cycle_life_to_80pct = 3000` (Dakota Lithium LiFePO4).

---

## Cost Model

The sizing optimizer computes total system cost as:

```
total_cost = battery_cost + pv_cost + DC_DC_CONVERTER_COST ($12)
```

No inverter cost — all loads are 12 V DC native. The DC-DC converter at $12 is the only power conversion component required (12 V battery → 5 V for controller/sensors/comms).

For the full BOM including all loads, sensors, and controls:

**Minimum build (Pi Zero 2W + 30Ah):**
```
Full build ≈ battery + PV + DC-DC + light + pump + vent + Pi Zero + sensors + comms + MPPT + wiring
           ≈ $100 + $75 + $12 + $25 + $18 + $45 + $35 + $50 + $20 + $60 + $25 = ~$480
```

**Standard build (RPi 4 + 50Ah):**
```
Full build ≈ battery + PV + DC-DC + light + pump + vent + RPi 4 + sensors + comms + MPPT + wiring
           ≈ $180 + $90 + $12 + $25 + $18 + $45 + $65 + $50 + $20 + $60 + $25 = ~$590
```

---

## License / Notes

This project was built for educational and research purposes. All part numbers, prices, and specifications are for actual off-the-shelf products available from major distributors as of 2026. Prices are approximate and will vary.

Every component in `python/real_parts.py` includes a product URL where specifications can be verified against manufacturer datasheets.
