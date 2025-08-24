"""Stateful Feature Engineering Pipeline.

Computes 17-dimensional behavioral feature vectors from streaming telemetry events:
- 5 instantaneous raw sensor channels
- 5 rolling window aggregates (5-minute and 30-minute mean, std, max)
- 3 first-order differential deltas (temperature_delta, pressure_delta, rpm_delta)
- 3 dynamic z-scores normalized against rolling baselines
- 1 multi-sensor velocity vector norm
"""

import collections
import math
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Tuple

FEATURE_NAMES: List[str] = [
    "temperature",
    "pressure",
    "vibration",
    "rpm",
    "voltage",
    "rolling_mean_5m_temp",
    "rolling_std_5m_vibr",
    "rolling_mean_30m_temp",
    "rolling_max_30m_press",
    "rolling_mean_30m_rpm",
    "temperature_delta",
    "pressure_delta",
    "rpm_delta",
    "temperature_zscore",
    "vibration_zscore",
    "pressure_zscore",
    "sensor_velocity",
]


class DeviceFeatureBuffer:
    """Maintains stateful rolling windows for a single device_id."""

    def __init__(self, window_5m_seconds: int = 300, window_30m_seconds: int = 1800):
        self.window_5m_sec = window_5m_seconds
        self.window_30m_sec = window_30m_seconds

        # Deque of (timestamp_epoch: float, values: Dict[str, float])
        self.history_5m: Deque[Tuple[float, Dict[str, float]]] = collections.deque()
        self.history_30m: Deque[Tuple[float, Dict[str, float]]] = collections.deque()

        # Previous reading for delta and velocity calculations
        self.prev_timestamp: Optional[float] = None
        self.prev_values: Optional[Dict[str, float]] = None

    def update(self, current_ts: float, values: Dict[str, float]) -> None:
        """Append new sample and evict expired records beyond windows."""
        self.history_5m.append((current_ts, values))
        self.history_30m.append((current_ts, values))

        # Evict from 5m window
        cutoff_5m = current_ts - self.window_5m_sec
        while self.history_5m and self.history_5m[0][0] < cutoff_5m:
            self.history_5m.popleft()

        # Evict from 30m window
        cutoff_30m = current_ts - self.window_30m_sec
        while self.history_30m and self.history_30m[0][0] < cutoff_30m:
            self.history_30m.popleft()


class StatefulFeatureEngine:
    """Calculates enriched feature vectors for industrial anomaly detection."""

    def __init__(self, window_5m_seconds: int = 300, window_30m_seconds: int = 1800):
        self.window_5m_sec = window_5m_seconds
        self.window_30m_sec = window_30m_seconds
        self.buffers: Dict[str, DeviceFeatureBuffer] = {}

    def _get_or_create_buffer(self, device_id: str) -> DeviceFeatureBuffer:
        if device_id not in self.buffers:
            self.buffers[device_id] = DeviceFeatureBuffer(
                window_5m_seconds=self.window_5m_sec, window_30m_seconds=self.window_30m_sec
            )
        return self.buffers[device_id]

    def compute_features(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a validated raw telemetry packet into a 17-feature vector."""
        device_id = str(event["device_id"])
        buffer = self._get_or_create_buffer(device_id)

        # Parse timestamp
        raw_ts = str(event["timestamp"])
        dt = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
        current_epoch = dt.timestamp()

        # Extract current numerical channels
        curr_vals = {
            "temperature": float(event["temperature"]),
            "pressure": float(event["pressure"]),
            "vibration": float(event["vibration"]),
            "rpm": float(event["rpm"]),
            "voltage": float(event["voltage"]),
        }

        # Update rolling buffer
        buffer.update(current_epoch, curr_vals)

        # 1. 5-minute rolling calculations
        temp_5m = [item[1]["temperature"] for item in buffer.history_5m]
        vibr_5m = [item[1]["vibration"] for item in buffer.history_5m]

        mean_5m_temp = sum(temp_5m) / len(temp_5m)
        std_5m_vibr = self._std(vibr_5m)

        # 2. 30-minute rolling calculations
        temp_30m = [item[1]["temperature"] for item in buffer.history_30m]
        press_30m = [item[1]["pressure"] for item in buffer.history_30m]
        rpm_30m = [item[1]["rpm"] for item in buffer.history_30m]

        mean_30m_temp = sum(temp_30m) / len(temp_30m)
        max_30m_press = max(press_30m)
        mean_30m_rpm = sum(rpm_30m) / len(rpm_30m)

        std_30m_temp = self._std(temp_30m)
        std_30m_vibr = self._std([item[1]["vibration"] for item in buffer.history_30m])
        std_30m_press = self._std(press_30m)

        # 3. Differentials and Deltas
        if buffer.prev_values is not None:
            temp_delta = curr_vals["temperature"] - buffer.prev_values["temperature"]
            press_delta = curr_vals["pressure"] - buffer.prev_values["pressure"]
            rpm_delta = curr_vals["rpm"] - buffer.prev_values["rpm"]
            time_elapsed = current_epoch - (buffer.prev_timestamp or current_epoch)
            delta_t = time_elapsed if time_elapsed > 0.05 else 1.0
            sensor_velocity = math.sqrt(
                (temp_delta / delta_t) ** 2
                + (press_delta / delta_t) ** 2
                + ((curr_vals["vibration"] - buffer.prev_values["vibration"]) / delta_t) ** 2
            )
        else:
            temp_delta = 0.0
            press_delta = 0.0
            rpm_delta = 0.0
            sensor_velocity = 0.0

        # Update previous state
        buffer.prev_timestamp = current_epoch
        buffer.prev_values = curr_vals

        # 4. Standardized Z-Scores
        eps = 1e-6
        temp_zscore = (curr_vals["temperature"] - mean_30m_temp) / (std_30m_temp + eps)
        mean_30m_vibr = sum(item[1]["vibration"] for item in buffer.history_30m) / len(buffer.history_30m)
        vibr_zscore = (curr_vals["vibration"] - mean_30m_vibr) / (std_30m_vibr + eps)
        mean_30m_press = sum(press_30m) / len(press_30m)
        press_zscore = (curr_vals["pressure"] - mean_30m_press) / (std_30m_press + eps)

        # Build feature dictionary
        features: Dict[str, float] = {
            "temperature": curr_vals["temperature"],
            "pressure": curr_vals["pressure"],
            "vibration": curr_vals["vibration"],
            "rpm": curr_vals["rpm"],
            "voltage": curr_vals["voltage"],
            "rolling_mean_5m_temp": round(mean_5m_temp, 3),
            "rolling_std_5m_vibr": round(std_5m_vibr, 4),
            "rolling_mean_30m_temp": round(mean_30m_temp, 3),
            "rolling_max_30m_press": round(max_30m_press, 2),
            "rolling_mean_30m_rpm": round(mean_30m_rpm, 1),
            "temperature_delta": round(temp_delta, 3),
            "pressure_delta": round(press_delta, 3),
            "rpm_delta": round(rpm_delta, 1),
            "temperature_zscore": round(temp_zscore, 4),
            "vibration_zscore": round(vibr_zscore, 4),
            "pressure_zscore": round(press_zscore, 4),
            "sensor_velocity": round(sensor_velocity, 4),
        }

        # Vector in exact canonical order
        feature_vector = [features[k] for k in FEATURE_NAMES]

        return {
            "device_id": device_id,
            "timestamp": raw_ts,
            "feature_dict": features,
            "feature_vector": feature_vector,
            "feature_names": FEATURE_NAMES,
        }

    @staticmethod
    def _std(values: List[float]) -> float:
        """Sample standard deviation calculation."""
        n = len(values)
        if n < 2:
            return 0.0
        mean = sum(values) / n
        variance = sum((x - mean) ** 2 for x in values) / (n - 1)
        return math.sqrt(max(0.0, variance))
