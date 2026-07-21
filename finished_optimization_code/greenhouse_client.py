"""
greenhouse_client.py — HTTP client for the Cloudflare Worker endpoint.

GET  / → current greenhouse state (sensors, battery, controls)
PUT  / → update control commands (lights, vent, pump)
"""

import time
import numpy as np
import requests
from typing import Optional
from config import ENDPOINT_URL, ENDPOINT_PASSWORD


class GreenhouseState:
    """Parsed state from a GET response."""

    def __init__(self, data: dict):
        self.raw = data
        self.lights: bool = data.get("lights", False)
        self.vent: bool = data.get("vent", False)
        self.pump: float = float(data.get("pump", 0))
        self.pump_timestamp: int = int(data.get("pumpTimestamp", 0))

        self.soil: list[int] = data.get("soil", [99, 99, 99, 99, 99, 99])
        self.soil1: int = self.soil[0] if len(self.soil) > 0 else 99
        self.soil2: int = self.soil[1] if len(self.soil) > 1 else 99

        self.temperature: float = float(data.get("temperature", 0))
        self.humidity: float = float(data.get("humidity", 0))

        self.picture_url: str = data.get("picture", "")
        self.picture_timestamp: int = int(data.get("pictureTimestamp", 0))
        self.take_picture: bool = data.get("takePicture", False)

        batt = data.get("battery", {})
        batt_in = batt.get("in", {})
        batt_out = batt.get("out", {})
        self.battery_voltage_in: float = float(batt_in.get("voltage", 0))
        self.battery_amps_in: float = float(batt_in.get("amps", 0))
        self.battery_voltage_out: float = float(batt_out.get("voltage", 0))
        self.battery_amps_out: float = float(batt_out.get("amps", 0))

        self.soc: float = float(data.get("soc", 0))

        # Plant state set by user from website
        self.plant_state_str: str = data.get("plantState", "GROWING")

        self.timestamp_ms: int = int(data.get("timestamp", int(time.time() * 1000)))

    @property
    def soil_moisture_pct(self) -> float:
        """Average soil moisture as percentage (0=dry, 100=wet).

        Capacitive sensor: raw 0-100 scale from ESP32.
        First two values in soil[] are real sensors.
        """
        vals = [s for s in [self.soil1, self.soil2] if s != 99]
        if not vals:
            return 50.0  # unknown → assume moderate
        return float(np.clip(np.mean(vals), 0, 100))

    def __repr__(self):
        return (
            f"GreenhouseState(soc={self.soc:.1f}%, temp={self.temperature:.1f}F, "
            f"soil=[{self.soil1},{self.soil2}], lights={self.lights}, "
            f"vent={self.vent}, pump={self.pump:.1f}min)"
        )


def get_greenhouse_state(timeout_s: float = 10.0) -> Optional[GreenhouseState]:
    """GET the current state from the Cloudflare Worker."""
    try:
        resp = requests.get(ENDPOINT_URL, timeout=timeout_s)
        resp.raise_for_status()
        data = resp.json()
        return GreenhouseState(data)
    except requests.RequestException as e:
        print(f"[client] GET failed: {e}")
        return None
    except (ValueError, KeyError) as e:
        print(f"[client] GET parse error: {e}")
        return None


def put_commands(
    lights: Optional[bool] = None,
    vent: Optional[bool] = None,
    pump: Optional[float] = None,
    pump_timestamp: Optional[int] = None,
    plant_state: Optional[str] = None,
    temperature_f: Optional[float] = None,
    humidity: Optional[float] = None,
    mpc_result: Optional[dict] = None,
    take_picture: Optional[bool] = None,
    timeout_s: float = 10.0,
) -> bool:
    """PUT control commands to the Cloudflare Worker.

    Only the fields provided are sent. The Worker merges them into the
    existing state (partial update).
    """
    payload = {"password": ENDPOINT_PASSWORD}

    if lights is not None:
        payload["lights"] = lights
    if vent is not None:
        payload["vent"] = vent
    if pump is not None:
        payload["pump"] = round(pump, 2)
    if pump_timestamp is not None:
        payload["pumpTimestamp"] = pump_timestamp
    if plant_state is not None:
        payload["plantState"] = plant_state
    if temperature_f is not None:
        payload["temperature"] = round(temperature_f, 1)
    if humidity is not None:
        payload["humidity"] = round(humidity, 1)
    if mpc_result is not None:
        payload["mpc"] = mpc_result
    if take_picture is not None:
        payload["takePicture"] = take_picture

    try:
        resp = requests.put(ENDPOINT_URL, json=payload, timeout=timeout_s)
        resp.raise_for_status()
        return True
    except requests.RequestException as e:
        print(f"[client] PUT failed: {e}")
        return False
