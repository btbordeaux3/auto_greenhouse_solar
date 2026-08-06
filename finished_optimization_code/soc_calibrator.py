"""
soc_calibrator.py — One-time battery SOC calibration via coulomb counting.

When the ESP32 SOC sensor reads an incorrect value (e.g. 0% when battery
is actually ~50%), this calibrator:

1. Assumes a starting SOC (default 50%).
2. Tracks relative changes via coulomb counting (battery current × voltage).
3. Once the tracked SOC hits a physical endpoint (≤1% or ≥100%), it locks in
   as the real calibrated value.
4. After calibration, trusts the ESP32 SOC directly.

Persists state to a JSON file so calibration survives daemon restarts.
Only needs to run once — after calibration, the file records it as done.
"""

import os
import json
import time
from dataclasses import dataclass
from typing import Optional

from config import BATTERY_Wh, BATTERY_NOM_V, BATTERY_EFF, DATA_DIR


CALIBRATION_FILE = os.path.join(DATA_DIR, "soc_calibration.json")


@dataclass
class CalibratorState:
    """Internal state for the SOC calibrator."""
    calibrated: bool = False          # True once we've locked to a physical endpoint
    calibrated_soc: float = 50.0      # Final calibrated SOC (0-100%), only valid when calibrated=True
    tracking_soc: float = 50.0        # Current tracked SOC (0-100%)
    last_update_ts: float = 0.0       # Timestamp of last update (epoch seconds)
    locked_at: str = ""               # "low" (≤1%) or "high" (≥100%)


class SoCCalibrator:
    """One-time SOC calibration via coulomb counting.

    Usage:
        cal = SoCCalibrator(initial_soc=50.0)
        soc_pct = cal.update(soc_raw_pct, battery_amps_in, battery_amps_out,
                            battery_voltage_out, dt_s)
        # soc_pct is the calibrated or raw SOC to use

    When not yet calibrated, returns tracked_soc (coulomb-counted).
    When calibrated, returns the locked-in value.
    When the ESP32 SOC is available and calibrated, can switch to trusting it.
    """

    def __init__(self, initial_soc: float = 50.0):
        self.state = CalibratorState()
        self._battery_cap_J = BATTERY_Wh * 3600.0  # total capacity in Joules
        self._initial_soc = initial_soc
        self._load()

    def _load(self):
        """Load calibration state from disk if available."""
        if not os.path.exists(CALIBRATION_FILE):
            self.state = CalibratorState(tracking_soc=self._initial_soc)
            return

        try:
            with open(CALIBRATION_FILE, "r") as f:
                data = json.load(f)
            self.state = CalibratorState(
                calibrated=data.get("calibrated", False),
                calibrated_soc=data.get("calibrated_soc", self._initial_soc),
                tracking_soc=data.get("tracking_soc", self._initial_soc),
                last_update_ts=data.get("last_update_ts", 0.0),
                locked_at=data.get("locked_at", ""),
            )
        except (json.JSONDecodeError, KeyError):
            self.state = CalibratorState(tracking_soc=self._initial_soc)

    def _save(self):
        """Persist calibration state to disk."""
        os.makedirs(os.path.dirname(CALIBRATION_FILE) or ".", exist_ok=True)
        data = {
            "calibrated": self.state.calibrated,
            "calibrated_soc": self.state.calibrated_soc,
            "tracking_soc": self.state.tracking_soc,
            "last_update_ts": self.state.last_update_ts,
            "locked_at": self.state.locked_at,
        }
        with open(CALIBRATION_FILE, "w") as f:
            json.dump(data, f, indent=2)

    def update(
        self,
        soc_raw_pct: float,
        battery_amps_in: float,
        battery_amps_out: float,
        battery_voltage_out: float,
        dt_s: float,
    ) -> float:
        """Update SOC estimate and return the value to use.

        Args:
            soc_raw_pct: ESP32-reported SOC (0-100%).
            battery_amps_in: Current flowing INTO the battery (charging). Amps.
            battery_amps_out: Current flowing OUT of the battery (discharging). Amps.
            battery_voltage_out: Battery voltage at output. Volts.
            dt_s: Time since last update. Seconds.

        Returns:
            SOC as a percentage (0-100) to use for optimization.
        """
        # If already calibrated, just return the locked value
        if self.state.calibrated:
            return self.state.calibrated_soc

        # Compute net energy change (Joules)
        # Charging: energy_in = V * I_in * efficiency
        # Discharging: energy_out = V * I_out / efficiency
        energy_charge_J = battery_voltage_out * battery_amps_in * BATTERY_EFF * dt_s
        energy_discharge_J = battery_voltage_out * battery_amps_out / BATTERY_EFF * dt_s
        net_energy_J = energy_charge_J - energy_discharge_J

        # Convert to SOC change (percentage)
        soc_change_pct = (net_energy_J / self._battery_cap_J) * 100.0

        # Update tracking SOC
        self.state.tracking_soc += soc_change_pct

        # Clamp to physical range
        self.state.tracking_soc = max(0.0, min(100.0, self.state.tracking_soc))
        self.state.last_update_ts = time.time()

        # Check if we've hit a physical endpoint → lock in
        if self.state.tracking_soc <= 1.0:
            self.state.calibrated = True
            self.state.calibrated_soc = 0.0
            self.state.locked_at = "low"
            print(f"[soc-cal] CALIBRATED! SOC hit {self.state.tracking_soc:.1f}% — "
                  f"locking at 0% (empty). Future readings will use ESP32 SOC.")
            self._save()
            return 0.0

        if self.state.tracking_soc >= 100.0:
            self.state.calibrated = True
            self.state.calibrated_soc = 100.0
            self.state.locked_at = "high"
            print(f"[soc-cal] CALIBRATED! SOC hit {self.state.tracking_soc:.1f}% — "
                  f"locking at 100% (full). Future readings will use ESP32 SOC.")
            self._save()
            return 100.0

        # Save periodically (not every call — avoid disk thrashing)
        if self.state.last_update_ts % 300 < dt_s:
            self._save()

        return self.state.tracking_soc

    @property
    def is_calibrated(self) -> bool:
        return self.state.calibrated

    def get_status(self) -> dict:
        """Return current calibration status for logging."""
        return {
            "calibrated": self.state.calibrated,
            "tracking_soc_pct": round(self.state.tracking_soc, 1),
            "locked_at": self.state.locked_at,
        }

    def reset(self, initial_soc: float = 50.0):
        """Manually reset calibration (for testing or re-calibration)."""
        self.state = CalibratorState(tracking_soc=initial_soc)
        self._save()
        print(f"[soc-cal] Reset. Tracking SOC = {initial_soc:.1f}%")
