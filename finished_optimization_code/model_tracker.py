"""
model_tracker.py — Track MPC prediction accuracy and calibrate model.

Logs per-cycle prediction errors (predicted vs actual SOC/temp/load)
and computes daily summaries to improve the model over time.

Compares the MPC's 1-step-ahead prediction (what it expected to happen
in the next 15 min) against the actual observation 15 min later.
"""

import os
import csv
import json
import time
import sqlite3
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Optional

from config import DATA_DIR, BATTERY_Wh, P_ESP32, P_LIGHTS, P_PUMP


MODEL_ERRORS_CSV = os.path.join(DATA_DIR, "model_errors.csv")
MODEL_CALIBRATION_JSON = os.path.join(DATA_DIR, "model_calibration.json")

MODEL_ERROR_FIELDS = [
    "timestamp_ms", "datetime_utc",
    "soc_observed", "soc_predicted_1step", "soc_error",
    "temp_observed_c", "temp_predicted_1step_c", "temp_error",
    "pump_dispatched_min", "pump_planned_total_min",
    "lights_dispatched_h", "lights_planned_total_h",
    "battery_amps_avg", "load_predicted_w",
]


class ModelTracker:
    """Tracks MPC 1-step prediction accuracy and suggests model adjustments."""

    PENDING_PATH = os.path.join(DATA_DIR, "model_pending_prediction.json")

    def __init__(self, db_path: str = "data/greenhouse.db"):
        self.db_path = db_path
        self._prev_predicted_soc_1step: Optional[float] = None
        self._prev_predicted_temp_1step: Optional[float] = None
        self._prev_pump_planned_total: float = 0.0
        self._prev_lights_planned_total: float = 0.0
        self._prev_pump_dispatched: float = 0.0
        self._prev_lights_dispatched: float = 0.0
        os.makedirs(DATA_DIR, exist_ok=True)
        self._load_pending()

    def store_prediction(
        self,
        soc_predicted_1step: float,
        temp_predicted_1step_c: float,
        pump_planned_total_min: float,
        lights_planned_total_h: float,
        pump_dispatched_min: float,
        lights_dispatched_h: float,
    ):
        """Store this cycle's 1-step prediction for comparison next cycle.

        soc_predicted_1step / temp_predicted_1step_c are what the MPC
        trajectory predicts for step index 1 (the next 15 min).
        """
        self._prev_predicted_soc_1step = soc_predicted_1step
        self._prev_predicted_temp_1step_c = temp_predicted_1step_c
        self._prev_pump_planned_total = pump_planned_total_min
        self._prev_lights_planned_total = lights_planned_total_h
        self._prev_pump_dispatched = pump_dispatched_min
        self._prev_lights_dispatched = lights_dispatched_h
        self._save_pending()

    def _save_pending(self):
        """Persist prediction to disk so it survives daemon restarts."""
        data = {
            "predicted_soc_1step": self._prev_predicted_soc_1step,
            "predicted_temp_1step_c": self._prev_predicted_temp_1step_c,
            "pump_planned_total_min": self._prev_pump_planned_total,
            "lights_planned_total_h": self._prev_lights_planned_total,
            "pump_dispatched_min": self._prev_pump_dispatched,
            "lights_dispatched_h": self._prev_lights_dispatched,
        }
        with open(self.PENDING_PATH, "w") as f:
            json.dump(data, f)

    def _load_pending(self):
        """Load previous prediction from disk if available."""
        if not os.path.exists(self.PENDING_PATH):
            return
        try:
            with open(self.PENDING_PATH, "r") as f:
                data = json.load(f)
            self._prev_predicted_soc_1step = data.get("predicted_soc_1step")
            self._prev_predicted_temp_1step_c = data.get("predicted_temp_1step_c")
            self._prev_pump_planned_total = data.get("pump_planned_total_min", 0.0)
            self._prev_lights_planned_total = data.get("lights_planned_total_h", 0.0)
            self._prev_pump_dispatched = data.get("pump_dispatched_min", 0.0)
            self._prev_lights_dispatched = data.get("lights_dispatched_h", 0.0)
        except (json.JSONDecodeError, KeyError):
            pass  # corrupted file, start fresh

    def log_cycle_error(
        self,
        soc_observed: float,
        temp_observed_c: float,
        battery_amps_avg: float = 0.0,
    ):
        """Compare current observation to previous 1-step prediction.

        Call at the START of a new cycle, before solving the MPC.
        """
        if self._prev_predicted_soc_1step is None or self._prev_predicted_temp_1step_c is None:
            return None

        soc_error = soc_observed - self._prev_predicted_soc_1step
        temp_error = temp_observed_c - self._prev_predicted_temp_1step_c

        # Estimate actual load from battery current
        load_w_predicted = P_ESP32 + (
            P_LIGHTS if self._prev_lights_dispatched > 0 else 0
        ) + (P_PUMP * self._prev_pump_dispatched / 15.0 if self._prev_pump_dispatched > 0 else 0)

        now_utc = datetime.now(timezone.utc)
        row = {
            "timestamp_ms": int(time.time() * 1000),
            "datetime_utc": now_utc.strftime("%Y-%m-%d %H:%M:%S"),
            "soc_observed": round(soc_observed * 100, 2),
            "soc_predicted_1step": round(self._prev_predicted_soc_1step * 100, 2),
            "soc_error": round(soc_error * 100, 2),
            "temp_observed_c": round(temp_observed_c, 2),
            "temp_predicted_1step_c": round(self._prev_predicted_temp_1step_c, 2),
            "temp_error": round(temp_error, 2),
            "pump_dispatched_min": round(self._prev_pump_dispatched, 2),
            "pump_planned_total_min": round(self._prev_pump_planned_total, 2),
            "lights_dispatched_h": round(self._prev_lights_dispatched, 2),
            "lights_planned_total_h": round(self._prev_lights_planned_total, 2),
            "battery_amps_avg": round(battery_amps_avg, 4),
            "load_predicted_w": round(load_w_predicted, 2),
        }
        self._write_csv_row(row)
        return row

    def _write_csv_row(self, row: dict):
        """Append a row to model_errors.csv."""
        write_header = not os.path.exists(MODEL_ERRORS_CSV)
        with open(MODEL_ERRORS_CSV, "a", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=MODEL_ERROR_FIELDS, extrasaction="ignore"
            )
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def compute_daily_summary(self, n_days: int = 7) -> dict:
        """Compute average prediction errors over the last N days."""
        if not os.path.exists(MODEL_ERRORS_CSV):
            return {"status": "no data"}

        rows = []
        with open(MODEL_ERRORS_CSV) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        if not rows:
            return {"status": "no data"}

        cutoff = datetime.now(timezone.utc) - timedelta(days=n_days)
        recent = []
        for row in rows:
            try:
                ts = datetime.strptime(row["datetime_utc"], "%Y-%m-%d %H:%M:%S")
                ts = ts.replace(tzinfo=timezone.utc)
                if ts >= cutoff:
                    recent.append(row)
            except (ValueError, KeyError):
                continue

        if len(recent) < 2:
            return {"status": f"only {len(recent)} data points, need >= 2"}

        soc_errors = [float(r["soc_error"]) for r in recent]
        temp_errors = [float(r["temp_error"]) for r in recent]

        soc_mae = float(np.mean(np.abs(soc_errors)))
        soc_bias = float(np.mean(soc_errors))
        temp_mae = float(np.mean(np.abs(temp_errors)))
        temp_bias = float(np.mean(temp_errors))

        summary = {
            "n_cycles": len(recent),
            "soc_mae_pct": round(soc_mae, 2),
            "soc_bias_pct": round(soc_bias, 2),
            "temp_mae_c": round(temp_mae, 2),
            "temp_bias_c": round(temp_bias, 2),
            "suggestions": [],
        }

        if abs(soc_bias) > 1.0:
            if soc_bias > 0:
                summary["suggestions"].append(
                    f"SOC consistently under-predicted by {soc_bias:.1f}%. "
                    "Battery efficiency may be higher than modeled, or load is lower. "
                    "Consider increasing BATTERY_EFF or decreasing P_ESP32."
                )
            else:
                summary["suggestions"].append(
                    f"SOC consistently over-predicted by {abs(soc_bias):.1f}%. "
                    "Battery efficiency may be lower than modeled, or load is higher. "
                    "Consider decreasing BATTERY_EFF or increasing P_ESP32."
                )

        if abs(temp_bias) > 1.0:
            if temp_bias > 0:
                summary["suggestions"].append(
                    f"Temperature consistently under-predicted by {temp_bias:.1f}°C. "
                    "Solar gain may be higher or ventilation lower than modeled. "
                    "Consider increasing ALPHA_SOLAR or decreasing ALPHA_VENT."
                )
            else:
                summary["suggestions"].append(
                    f"Temperature consistently over-predicted by {abs(temp_bias):.1f}°C. "
                    "Solar gain may be lower or ventilation higher than modeled. "
                    "Consider decreasing ALPHA_SOLAR or increasing ALPHA_VENT."
                )

        if soc_mae > 3.0:
            summary["suggestions"].append(
                f"SOC 1-step error is high (MAE={soc_mae:.1f}%). "
                "Check battery capacity (BATTERY_Wh) and charge/discharge limits."
            )

        if temp_mae > 2.0:
            summary["suggestions"].append(
                f"Temperature 1-step error is high (MAE={temp_mae:.1f}°C). "
                "Thermal model may need retuning. Check ACH_VENT, ACH_INFIL, ALPHA_SOLAR."
            )

        if not summary["suggestions"]:
            summary["suggestions"].append("Model is tracking well. No adjustments needed.")

        return summary

    def load_calibration(self) -> dict:
        """Load saved calibration adjustments."""
        if os.path.exists(MODEL_CALIBRATION_JSON):
            with open(MODEL_CALIBRATION_JSON) as f:
                return json.load(f)
        return {}

    def save_calibration(self, adjustments: dict):
        """Save calibration adjustments to disk."""
        with open(MODEL_CALIBRATION_JSON, "w") as f:
            json.dump(adjustments, f, indent=2)


def print_daily_summary(n_days: int = 7):
    """Print model error summary."""
    tracker = ModelTracker()
    summary = tracker.compute_daily_summary(n_days)

    print(f"\n{'='*60}")
    print(f"  Model Accuracy Summary (last {n_days} days)")
    print(f"{'='*60}")

    if summary.get("status"):
        print(f"  {summary['status']}")
        return

    print(f"  Cycles analyzed: {summary['n_cycles']}")
    print(f"  SOC error:   MAE={summary['soc_mae_pct']:.2f}%  bias={summary['soc_bias_pct']:+.2f}%")
    print(f"  Temp error:  MAE={summary['temp_mae_c']:.2f}°C  bias={summary['temp_bias_c']:+.2f}°C")
    print()
    print("  Suggestions:")
    for s in summary["suggestions"]:
        print(f"    - {s}")
    print()


if __name__ == "__main__":
    print_daily_summary()
