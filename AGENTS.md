# Agent Context

# Agent Context

## Goal
Build and integrate an RL optimizer for the greenhouse digital twin with realistic physics and dynamic load scheduling to minimize total system cost and battery depletion.

## Constraints & Preferences
- Must not impact existing optimizers (MPC, MILP, deterministic, rule-based).
- Single fixed model name `ppo_greenhouse` — always the latest, deleted on retrain.
- All training artifacts go to `results/rl_train/`; model `.zip` stays in `results/rl_models/`.

## Progress

### Done
- **DC-DC converter fix**: Replaced Victron Orion-TR 360W (44% eff at 9.6W load) with Recom R-78B5.0-2.0 15W switching regulator (~90% eff). DC-DC loss dropped from **296 → 20 Wh/day** (93% reduction).
- **Comms power fix**: LoRaWAN RFM95W changed from 2.5W continuous to 0.3W average (duty-cycled). Always-on base load dropped from 22W to 8W.
- **Dynamic lights scheduling**: Added `lights_benefit_score()` that considers SOC, temperature, GHI, and darkness. `compute_reward()` scales lights reward by benefit and penalises wasteful lighting. Hard override at SOC < 15% or not dark.
- **Retrained RL (physics update + dynamic lights)**: Three training runs of 1.5M steps each — first on corrected DC-DC/comms physics, then with dynamic lights reward, then with 15% SOC floor. Each ~11-13 min.
- **Size optimizer sweep**: Winning combo: **Ren12V100Ah (1280Wh) + Ren200W + Vic300W** — RL ranks #1 across all hardware combos.
- **Inference override bug fix**: Was using Kalman `soc_est` (lags true SOC 3-8%) instead of `true_state[0]`. Lights stayed on past 15% floor, causing SOC to drop to 5.5%.
- **Weather forecast awareness**: Added `build_weather_features()` that computes 3 forecast features from upcoming weather: min temp over next 12h, total GHI over next 24h, and remaining dark hours. Added to RL observation (12-dim → 15-dim). Trained 2.5M steps with rebalanced pump reward.
- **Forecast uncertainty**: Added `make_noisy_forecast()` that generates weather forecasts with random-walk noise (sigma ~ sqrt(t)). Temperature uncertainty: ~1°C at 1h, ~3°C at 12h. GHI uncertainty: ~14% at 1h, ~50% at 12h. RL forecast features now read from noisy forecast arrays. Both MPC (via perturbed scenarios) and RL (via noisy forecast features) operate under realistic forecast uncertainty instead of perfect knowledge.
- **Forecast-aware 10-day RL validation**: Total load **411 Wh/day** (down from 415). Lights vary per day (0.08h–4.0h). Pump 100% compliant. Temp in band improved from 0–58% to 33–86%. Min SOC = 7.4% (base-load drain on dark days).
- **3-day controller sweep (forecast uncertainty)**: RL ranked #1 across MPC, rule, and RL. RL: 8.3h lights (69%), 100% pump, 66.7% temp in band, 523 Wh/day. MPC: 11.2h lights (93%), 100% pump, 60.1% temp in band, 620 Wh/day. Rule: 9.2h lights (76%), 33% pump, 48.6% temp in band.

### In Progress
- **Deployed greenhouse dashboard**: Website on Netlify, API on Cloudflare Worker, D1 for history. Daemon runs on user's always-on computer.

### Blocked
- (none)

## Key Decisions
- **DC-DC converter sourced from real parts**: Recom R-78B5.0-2.0 ($12, 10W rated, ~91% peak) replaces massively-oversized 360W Victron Orion-TR ($160).
- **LoRa comms average 0.3W**: 2.5W is TX peak only. Idle <25 mW; TX at <1% duty cycle brings average below 0.5W.
- **Lights benefit score [0,1]**: Multiplies SOC factor × temperature factor × solar factor. SOC factor ramps from 0 (15%) to 1 (50%+). Hard override: lights=0 when SOC<15%, not dark, GHI>50, or temp>30°C.
- **Safety override at SOC < 15% (true SOC)**: Applied in both `GreenhouseEnv.step()` and `run_twin.py` RL inference. Uses `true_state[0]` not `soc_est` to match training environment behavior.
- **Forecast features in RL observation**: 3 features — `temp_forecast_min_12h` (min temp over next half-day), `ghi_forecast_total_24h` (total solar irradiance forecast), `rem_dark_frac` (remaining night fraction). Normalized in [0,1]. Computed from the pre-sampled weather arrays at each step.
- **Forecast uncertainty**: Added `make_noisy_forecast()` in `historical_weather.py`. Generates random-walk noise (sigma ~ sqrt(t)) added to true weather to simulate realistic forecast divergence. Temperature: ~0.25°C/step → ~3°C at 12h. GHI: ~0.04 log-ratio/step → ~50% error at 12h. RL reads forecast features from noisy arrays; MPC uses perturbed scenarios centered on true weather via `base_data`.
- **Pump reward rebalanced**: `R_PUMP` increased from 1.0 to 5.0. Terminal pump reward changed to match lights formula (max 4.0, smooth partial credit). Previously pump (~0.6/day) was 9× less rewarding than lights (~5.4/day), causing pump neglect in high-dim observation.
- **Quick-mode sizing sweep**: Full sweep (420 sims × scenarios) would take hours. Quick mode (2 batteries × 2 PV × 1 inverter × 5 controllers = 20 combos) matches ranking for dominant factor (cost).

## Next Steps
- Run full `--all` sizing sweep overnight for definitive hardware ranking including 400W panel and 1000W inverter options.
- Add larger battery (2560Wh) to see if min SOC issue resolves.
- Considering MPC-based lights scheduling as alternative to RL for comparison.

## Critical Context
- **DC-DC loss was the dominant waste**: Old 360W converter at 44% efficiency wasted 12W continuously. New 15W converter at 90% wastes 0.9W. Single change reduced total system load by 33%.
- **Base load now 8W (was 22W)**: RPi4 7W + sensors 0.1W + LoRa 0.3W + DC-DC loss 0.9W ≈ 8.3W.
- **Min SOC 7.4% is from always-on drain, not lights**: On dark/cold days with minimal solar, even zero lights doesn't prevent SOC drop below 10%. The 8W always-on base load (RPi+sensors+comms) drains ~6%/day. Only fix is larger battery. The 15% lights override is still working correctly — lights turn off at 15% and only base load continues draining.
- **Temperature control improved with forecast**: Temp in band went from 0-58% to 33-86%. Forecast-aware agent better anticipates temperature trends and schedules lights+venting accordingly.
- **Forecast features encode actionable info**: `temp_forecast_min` tells agent if it'll get colder (lights defer later). `ghi_tomorrow_total` tells agent if solar recharge is coming (consume vs conserve). `rem_dark_frac` creates urgency near dawn.
- **Kalman filter SOC estimate lags true SOC by 3-8%**: Critical for override logic. Always use `true_state[0]` for safety overrides, not `soc_est`.
- **MATLAB engine available**: `python -m optimizer.rl_optimizer --matlab-finetune` for optional Simulink fine-tuning.

## Relevant Files
- `python/optimizer/rl_optimizer.py`: `lights_benefit_score()`, `compute_reward()`, `build_weather_features()`, `build_observation()` (15-dim), `GreenhouseEnv.step()` with forecast-aware obs.
- `python/run_twin.py`: RL inference with forecast features from `build_weather_features()`, true-SOC-based dynamic lights override.
- `python/optimizer/stochastic_mpc.py`: P_COMMS=0.3W, DC_DC_EFF_CURVE for 15W regulator.
- `python/real_parts.py`: Recom R-78B5.0-2.0 10W DC-DC converter.
- `python/size_optimizer.py`: Quick-mode with correct battery filters, absolute RL model path.
- `results/rl_models/ppo_greenhouse.zip`: Latest retrained model (forecast-aware, 15-dim obs, balanced pump reward).
- `results/1280Wh_200Wp_300W/`: Winning combo directory — twin_log.csv, twin_results.png.
