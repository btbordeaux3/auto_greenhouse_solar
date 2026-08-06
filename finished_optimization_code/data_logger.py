"""
data_logger.py — Dual-format logging for analysis and RL training.

SQLite: structured queries, full raw JSON storage
CSV: quick pandas loading for RL training
"""

import os
import json
import time
import sqlite3
import csv
import numpy as np
from datetime import datetime, timezone
from typing import Optional

from config import DATA_DIR, LOG_DB, LOG_CSV_DIR


class DataLogger:
    """Logs every optimization cycle to SQLite + CSV."""

    def __init__(self, db_path: str = LOG_DB, csv_dir: str = LOG_CSV_DIR):
        self.db_path = db_path
        self.csv_dir = csv_dir
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        os.makedirs(csv_dir, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute("""
            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_ms INTEGER NOT NULL,
                soc REAL,
                temperature REAL,
                humidity REAL,
                soil1 INTEGER,
                soil2 INTEGER,
                battery_voltage_in REAL,
                battery_amps_in REAL,
                battery_voltage_out REAL,
                battery_amps_out REAL,
                picture_url TEXT,
                picture_timestamp INTEGER,
                plant_state TEXT,
                plant_green_ratio REAL,
                plant_area_pct REAL,
                weather_source TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_ms INTEGER NOT NULL,
                lights INTEGER,
                vent INTEGER,
                pump_minutes REAL,
                pump_timestamp INTEGER,
                take_picture INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS system_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_ms INTEGER NOT NULL,
                raw_json TEXT,
                mpc_solve_time_ms REAL,
                mpc_status TEXT,
                mpc_soc_end REAL,
                mpc_temp_end REAL,
                mpc_pump_total_min REAL,
                mpc_lights_total_h REAL,
                mpc_cost REAL,
                cycle_duration_ms REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS weather_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_ms INTEGER NOT NULL,
                forecast_step INTEGER,
                ghi_forecast REAL,
                temp_forecast REAL,
                wind_forecast REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()

    def log_observation(
        self,
        state,
        plant_result: dict,
        weather_source: str = "open-meteo",
    ):
        """Log a greenhouse state observation."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        plant_state_str = ""
        green_ratio = 0.0
        area_pct = 0.0
        if plant_result:
            ps = plant_result.get("state", "")
            plant_state_str = ps.value if hasattr(ps, "value") else str(ps)
            green_ratio = plant_result.get("green_ratio", 0.0)
            area_pct = plant_result.get("plant_area_pct", 0.0)

        c.execute("""
            INSERT INTO observations (
                timestamp_ms, soc, temperature, humidity,
                soil1, soil2,
                battery_voltage_in, battery_amps_in,
                battery_voltage_out, battery_amps_out,
                picture_url, picture_timestamp,
                plant_state, plant_green_ratio, plant_area_pct,
                weather_source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            state.timestamp_ms if hasattr(state, "timestamp_ms") else int(time.time() * 1000),
            state.soc if hasattr(state, "soc") else 0.0,
            state.temperature if hasattr(state, "temperature") else 0.0,
            state.humidity if hasattr(state, "humidity") else 0.0,
            state.soil1 if hasattr(state, "soil1") else 99,
            state.soil2 if hasattr(state, "soil2") else 99,
            state.battery_voltage_in if hasattr(state, "battery_voltage_in") else 0.0,
            state.battery_amps_in if hasattr(state, "battery_amps_in") else 0.0,
            state.battery_voltage_out if hasattr(state, "battery_voltage_out") else 0.0,
            state.battery_amps_out if hasattr(state, "battery_amps_out") else 0.0,
            state.picture_url if hasattr(state, "picture_url") else "",
            state.picture_timestamp if hasattr(state, "picture_timestamp") else 0,
            plant_state_str,
            green_ratio,
            area_pct,
            weather_source,
        ))

        conn.commit()
        conn.close()

    def log_action(
        self,
        lights: bool,
        vent: bool,
        pump_minutes: float,
        pump_timestamp: int,
        take_picture: bool = False,
    ):
        """Log the control actions dispatched to the greenhouse."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute("""
            INSERT INTO actions (
                timestamp_ms, lights, vent, pump_minutes,
                pump_timestamp, take_picture
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            int(time.time() * 1000),
            int(lights),
            int(vent),
            round(pump_minutes, 2),
            pump_timestamp,
            int(take_picture),
        ))

        conn.commit()
        conn.close()

    def log_system(
        self,
        raw_json: dict,
        mpc_result=None,
        cycle_duration_ms: float = 0.0,
    ):
        """Log the full raw JSON and MPC result."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        mpc_time = 0.0
        mpc_status = ""
        soc_end = 0.0
        temp_end = 0.0
        pump_total = 0.0
        lights_total = 0.0
        mpc_cost = 0.0

        if mpc_result is not None:
            mpc_time = mpc_result.solve_time_ms
            mpc_status = mpc_result.status
            soc_end = float(mpc_result.soc_trajectory[-1, 0]) if mpc_result.soc_trajectory.size > 0 else 0.0
            temp_end = float(mpc_result.temp_trajectory[-1]) if len(mpc_result.temp_trajectory) > 0 else 0.0
            pump_total = float(np.sum(mpc_result.pump_plan))
            lights_total = float(np.sum(mpc_result.lights_plan)) * 0.25
            mpc_cost = mpc_result.expected_cost

        c.execute("""
            INSERT INTO system_log (
                timestamp_ms, raw_json,
                mpc_solve_time_ms, mpc_status,
                mpc_soc_end, mpc_temp_end,
                mpc_pump_total_min, mpc_lights_total_h,
                mpc_cost, cycle_duration_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            int(time.time() * 1000),
            json.dumps(raw_json),
            round(mpc_time, 1),
            mpc_status,
            round(soc_end, 4),
            round(temp_end, 2),
            round(pump_total, 2),
            round(lights_total, 2),
            round(mpc_cost, 4),
            round(cycle_duration_ms, 1),
        ))

        conn.commit()
        conn.close()

    def log_weather_scenario(
        self,
        ghi_scenarios,
        temp_scenarios,
        step_idx: int = 0,
    ):
        """Log the weather scenario data used for this MPC solve."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        ghi_mean = np.mean(ghi_scenarios, axis=0)
        temp_mean = np.mean(temp_scenarios, axis=0)

        for t in range(min(len(ghi_mean), 192)):
            c.execute("""
                INSERT INTO weather_data (
                    timestamp_ms, forecast_step,
                    ghi_forecast, temp_forecast
                ) VALUES (?, ?, ?, ?)
            """, (
                int(time.time() * 1000),
                t,
                round(float(ghi_mean[t]), 2),
                round(float(temp_mean[t]), 2),
            ))

        conn.commit()
        conn.close()

    def log_csv_row(self, row: dict):
        """Append a row to the daily CSV file."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        csv_path = os.path.join(self.csv_dir, f"session_{today}.csv")

        write_header = not os.path.exists(csv_path)

        fieldnames = [
            "timestamp_ms", "datetime_utc", "soc", "soc_raw", "soc_calibrated",
            "temperature_f", "temperature_c",
            "humidity", "soil1", "soil2", "soil_moisture", "soil_mpc_end",
            "battery_voltage", "battery_amps_in", "battery_amps_out",
            "plant_state", "green_ratio",
            "lights", "vent", "pump_minutes",
            "mpc_soc_end", "mpc_temp_end", "mpc_solve_ms", "mpc_status",
            "pv_estimate_w", "load_estimate_w",
            "harvest_alert",
        ]

        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def get_recent_observations(self, n: int = 100) -> list[dict]:
        """Fetch recent observations from SQLite for analysis."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        c.execute("""
            SELECT * FROM observations
            ORDER BY timestamp_ms DESC
            LIMIT ?
        """, (n,))

        rows = [dict(row) for row in c.fetchall()]
        conn.close()
        return rows

    def get_observation_count(self) -> int:
        """Total number of observations logged."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM observations")
        count = c.fetchone()[0]
        conn.close()
        return count

    def get_today_totals(self) -> dict:
        """Get today's total pump minutes and lights hours from actions log."""
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        start_ms = int(datetime.strptime(today, "%Y-%m-%d").replace(
            tzinfo=timezone.utc).timestamp() * 1000)
        end_ms = start_ms + 86400000  # end of day

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute("""
            SELECT COALESCE(SUM(pump_minutes), 0)
            FROM actions
            WHERE timestamp_ms >= ? AND timestamp_ms < ?
        """, (start_ms, end_ms))
        pump_total = float(c.fetchone()[0])

        c.execute("""
            SELECT COALESCE(SUM(lights), 0)
            FROM actions
            WHERE timestamp_ms >= ? AND timestamp_ms < ?
        """, (start_ms, end_ms))
        # Each action row covers 15 min; lights is 0/1, so sum * 0.25h
        lights_total_h = float(c.fetchone()[0]) * 0.25

        conn.close()
        return {"pump_min": pump_total, "lights_h": lights_total_h}
