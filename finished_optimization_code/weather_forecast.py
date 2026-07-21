"""
weather_forecast.py — Weather forecast and scenario generation.

Uses the Open-Meteo free API for real 2-day weather forecasts,
then generates perturbed scenarios for stochastic MPC.
"""

import numpy as np
import requests
from typing import Optional
from dataclasses import dataclass

from config import LATITUDE, LONGITUDE, TIMEZONE, DT_S, HORIZON_STEPS, N_SCENARIOS, SUNRISE, SUNSET


@dataclass
class WeatherForecast:
    """Hourly weather forecast arrays."""
    time_s: np.ndarray       # relative time (seconds)
    ghi: np.ndarray          # W/m² global horizontal irradiance
    temperature: np.ndarray  # °C ambient temperature
    humidity: np.ndarray     # % relative humidity
    wind_speed: np.ndarray   # m/s
    cloud_cover: np.ndarray  # % cloud cover (0-100)
    direct_radiation: np.ndarray  # W/m² direct normal irradiance
    precipitation: np.ndarray     # mm/h rain amount
    precip_prob: np.ndarray       # % probability of precipitation

    def daily_solar_kwh(self, pv_wpeak: float = 100.0, mppt_eff: float = 0.95) -> np.ndarray:
        """Estimate daily PV yield in kWh from hourly GHI.

        Uses: P_pv = GHI × (pv_wpeak/1000) × mppt_eff × temp_derating
        Returns one kWh value per day in the forecast.
        """
        from config import DT_H
        h_per_day = int(24 / DT_H)
        n_days = max(1, len(self.ghi) // h_per_day)
        daily = np.zeros(n_days)
        for d in range(n_days):
            start = d * h_per_day
            end = min(start + h_per_day, len(self.ghi))
            ghi_day = self.ghi[start:end]
            temp_day = self.temperature[start:end]
            # Temperature derating: panel loses ~0.4%/°C above 25°C
            temp_factor = np.maximum(1.0 - 0.004 * (temp_day - 25.0), 0.7)
            power_w = ghi_day * (pv_wpeak / 1000.0) * mppt_eff * temp_factor
            daily[d] = np.sum(power_w * DT_H) / 1000.0  # Wh → kWh
        return daily


def fetch_open_meteo_forecast(
    horizon_hours: int = 48,
    lat: float = LATITUDE,
    lon: float = LONGITUDE,
    tz: str = TIMEZONE,
) -> Optional[WeatherForecast]:
    """Fetch hourly forecast from Open-Meteo API.

    Returns WeatherForecast or None if the request fails.
    Open-Meteo is free, no API key needed.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,shortwave_radiation,wind_speed_10m,cloud_cover,direct_radiation,precipitation,precipitation_probability",
        "forecast_days": max(3, horizon_hours // 24 + 1),
        "timezone": tz,
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as e:
        print(f"[weather] Open-Meteo request failed: {e}")
        return None

    hourly = data.get("hourly", {})
    temps_c = np.array(hourly.get("temperature_2m", []), dtype=float)
    humidity = np.array(hourly.get("relative_humidity_2m", []), dtype=float)
    ghi = np.array(hourly.get("shortwave_radiation", []), dtype=float)
    wind = np.array(hourly.get("wind_speed_10m", []), dtype=float)
    cloud = np.array(hourly.get("cloud_cover", []), dtype=float)
    direct = np.array(hourly.get("direct_radiation", []), dtype=float)
    precip = np.array(hourly.get("precipitation", []), dtype=float)
    precip_prob = np.array(hourly.get("precipitation_probability", []), dtype=float)

    if len(temps_c) == 0 or len(ghi) == 0:
        print("[weather] Open-Meteo returned empty hourly data")
        return None

    n_hours = min(horizon_hours, len(temps_c))
    temps_c = temps_c[:n_hours]
    humidity = humidity[:n_hours] if len(humidity) >= n_hours else np.full(n_hours, 50.0)
    ghi = ghi[:n_hours]
    wind = wind[:n_hours]
    cloud = cloud[:n_hours] if len(cloud) >= n_hours else np.full(n_hours, 50.0)
    direct = direct[:n_hours] if len(direct) >= n_hours else ghi.copy()
    precip = precip[:n_hours] if len(precip) >= n_hours else np.zeros(n_hours)
    precip_prob = precip_prob[:n_hours] if len(precip_prob) >= n_hours else np.zeros(n_hours)

    n_steps = int(n_hours * 3600.0 / DT_S)
    time_s = np.arange(n_steps, dtype=float) * DT_S

    t_amb = np.interp(time_s, np.arange(n_hours) * 3600.0, temps_c)
    hum = np.interp(time_s, np.arange(n_hours) * 3600.0, humidity)
    ghi_r = np.interp(time_s, np.arange(n_hours) * 3600.0, ghi)
    wind_r = np.interp(time_s, np.arange(n_hours) * 3600.0, wind)
    cloud_r = np.interp(time_s, np.arange(n_hours) * 3600.0, cloud)
    direct_r = np.interp(time_s, np.arange(n_hours) * 3600.0, direct)
    precip_r = np.interp(time_s, np.arange(n_hours) * 3600.0, precip)
    precip_prob_r = np.interp(time_s, np.arange(n_hours) * 3600.0, precip_prob)

    result = WeatherForecast(
        time_s=time_s[:HORIZON_STEPS],
        ghi=ghi_r[:HORIZON_STEPS],
        temperature=t_amb[:HORIZON_STEPS],
        humidity=hum[:HORIZON_STEPS],
        wind_speed=wind_r[:HORIZON_STEPS],
        cloud_cover=cloud_r[:HORIZON_STEPS],
        direct_radiation=direct_r[:HORIZON_STEPS],
        precipitation=precip_r[:HORIZON_STEPS],
        precip_prob=precip_prob_r[:HORIZON_STEPS],
    )

    # Log daily solar estimates
    daily_kwh = result.daily_solar_kwh()
    for d, kwh in enumerate(daily_kwh):
        print(f"  [weather] Day {d}: estimated PV yield {kwh:.2f} kWh")

    return result


def _ar1_noise(length: int, rho: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """AR(1) process: x[t] = rho * x[t-1] + sigma * z[t]."""
    z = rng.normal(0, 1, size=length)
    x = np.zeros(length)
    x[0] = sigma / np.sqrt(1.0 - rho * rho) * z[0]
    for t in range(1, length):
        x[t] = rho * x[t - 1] + sigma * z[t]
    return x


def generate_scenarios(
    forecast: WeatherForecast,
    n_scenarios: int = N_SCENARIOS,
    rng_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate perturbed weather scenarios from a base forecast.

    Cloud-aware: uses cloud_cover to create realistic GHI perturbations.
    High cloud cover → scenarios favor low GHI (cloudy persist).
    Low cloud cover → scenarios favor high GHI (clear persist).

    Returns:
        ghi_scenarios:   (S, H) GHI scenarios in W/m²
        temp_scenarios:  (S, H) temperature scenarios in °C
        wind_scenarios:  (S, H) wind speed scenarios in m/s

    Uncertainty grows with forecast horizon (AR(1) perturbation).
    """
    H = len(forecast.ghi)
    rng = np.random.default_rng(rng_seed)

    RHO_SOLAR = 0.85
    SIGMA_SOLAR = 0.25    # 25% log-ratio std for GHI (higher for 4-day horizon)
    RHO_TEMP = 0.95
    SIGMA_TEMP = 0.7      # °C std for temperature (higher for 4-day horizon)

    # Cloud-aware GHI scaling: high cloud → lower GHI base
    cloud_frac = np.clip(forecast.cloud_cover / 100.0, 0.0, 1.0)
    cloud_ghi_base = 1.2 - 0.9 * cloud_frac  # 0% clouds→1.2, 100%→0.3

    ghi_scenarios = np.zeros((n_scenarios, H))
    temp_scenarios = np.zeros((n_scenarios, H))
    wind_scenarios = np.zeros((n_scenarios, H))

    for i in range(n_scenarios):
        log_eps = _ar1_noise(H, RHO_SOLAR, SIGMA_SOLAR, rng)
        eps_solar = np.clip(np.exp(log_eps), 0.3, 1.7)

        delta_temp = _ar1_noise(H, RHO_TEMP, SIGMA_TEMP, rng)

        day_mask = forecast.ghi > 10.0
        ghi_pert = forecast.ghi.copy()
        ghi_pert[day_mask] = forecast.ghi[day_mask] * cloud_ghi_base[day_mask] * eps_solar[day_mask]
        ghi_pert = np.clip(ghi_pert, 0.0, 1200.0)

        temp_pert = forecast.temperature + delta_temp
        wind_pert = np.maximum(forecast.wind_speed + rng.normal(0, 0.5, H), 0.0)

        ghi_scenarios[i] = ghi_pert
        temp_scenarios[i] = temp_pert
        wind_scenarios[i] = wind_pert

    return ghi_scenarios, temp_scenarios, wind_scenarios


def get_weather_scenarios(
    horizon_hours: int = 96,
    n_scenarios: int = N_SCENARIOS,
    rng_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, WeatherForecast]:
    """High-level: fetch forecast, generate scenarios.

    Returns (ghi_scenarios, temp_scenarios, wind_scenarios, forecast) where
    scenarios are each (S, H) and forecast has daily_solar_kwh().
    Falls back to synthetic clear-sky + noise if API is unavailable.
    """
    forecast = fetch_open_meteo_forecast(horizon_hours)

    if forecast is not None:
        daily_kwh = forecast.daily_solar_kwh()
        rain_hours = int(np.sum(forecast.precipitation > 0.1))
        max_prob = forecast.precip_prob.max() if len(forecast.precip_prob) > 0 else 0
        print(f"[weather] Got forecast: {len(forecast.ghi)} steps, "
              f"temp range [{forecast.temperature.min():.1f}, "
              f"{forecast.temperature.max():.1f}]°C, "
              f"peak GHI {forecast.ghi.max():.0f} W/m², "
              f"cloud avg {forecast.cloud_cover.mean():.0f}%, "
              f"rain {rain_hours}h (max prob {max_prob:.0f}%), "
              f"solar/day {[f'{k:.2f}' for k in daily_kwh]} kWh")
        ghi_s, temp_s, wind_s = generate_scenarios(forecast, n_scenarios, rng_seed)
        return ghi_s, temp_s, wind_s, forecast

    print("[weather] API unavailable, falling back to synthetic forecast")
    ghi_s, temp_s, wind_s = _synthetic_scenarios(horizon_hours, n_scenarios, rng_seed)
    # Create a minimal forecast for synthetic case
    H = ghi_s.shape[1]
    dummy = WeatherForecast(
        time_s=np.arange(H) * DT_S,
        ghi=ghi_s[0], temperature=temp_s[0],
        humidity=np.full(H, 50.0), wind_speed=wind_s[0],
        cloud_cover=np.full(H, 50.0), direct_radiation=ghi_s[0] * 0.6,
        precipitation=np.zeros(H), precip_prob=np.zeros(H),
    )
    return ghi_s, temp_s, wind_s, dummy


def _synthetic_scenarios(
    horizon_hours: int, n_scenarios: int, rng_seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fallback: synthetic clear-sky GHI + sinusoidal temperature."""
    n_steps = int(horizon_hours * 3600.0 / DT_S)
    H = min(n_steps, HORIZON_STEPS)
    time_s = np.arange(H, dtype=float) * DT_S

    ghi_base = np.zeros(H)
    temp_base = np.zeros(H)
    for t in range(H):
        hour = (time_s[t] / 3600.0) % 24.0
        if SUNRISE <= hour <= SUNSET:
            angle = np.pi * (hour - SUNRISE) / (SUNSET - SUNRISE)
            ghi_base[t] = 800.0 * np.sin(angle)
        temp_base[t] = 18.0 + 8.0 * np.sin(2.0 * np.pi * (hour - 14.0) / 24.0)

    rng = np.random.default_rng(rng_seed)
    ghi_scen = np.zeros((n_scenarios, H))
    temp_scen = np.zeros((n_scenarios, H))
    wind_scen = np.zeros((n_scenarios, H))

    for i in range(n_scenarios):
        eps = np.clip(np.exp(_ar1_noise(H, 0.85, 0.20, rng)), 0.3, 1.7)
        day_mask = ghi_base > 10.0
        g = ghi_base.copy()
        g[day_mask] = ghi_base[day_mask] * eps[day_mask]
        ghi_scen[i] = np.clip(g, 0, 1200)
        temp_scen[i] = temp_base + _ar1_noise(H, 0.95, 0.5, rng)
        wind_scen[i] = np.maximum(3.0 + rng.normal(0, 0.8, H), 0.0)

    return ghi_scen, temp_scen, wind_scen



