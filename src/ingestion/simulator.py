"""Industrial Telemetry Generator & Anomaly Simulator.

Simulates multi-sensor streaming telemetry for manufacturing machinery,
including realistic Gaussian noise, diurnal thermal cycles, and 4 synthetic failure modes:
1. THERMAL_SHOCK: Extreme temperature spike with rapid pressure loss
2. BEARING_WEAR: Progressive high-frequency vibration surge
3. MOTOR_STALL: Sudden RPM collapse and voltage drop
4. SENSOR_FAULT: Out-of-bounds values and nulls designed to trigger Great Expectations DLQ quarantine
"""

import math
import random
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional


class AnomalyType(str, Enum):
    NONE = "NONE"
    THERMAL_SHOCK = "THERMAL_SHOCK"
    BEARING_WEAR = "BEARING_WEAR"
    MOTOR_STALL = "MOTOR_STALL"
    SENSOR_FAULT = "SENSOR_FAULT"


class TelemetrySimulator:
    """Stateful physics-based telemetry simulator for industrial machinery."""

    def __init__(
        self,
        device_id: str = "MACHINE_042",
        sampling_interval_seconds: float = 1.0,
        anomaly_probability: float = 0.05,
    ):
        self.device_id = device_id
        self.interval = sampling_interval_seconds
        self.anomaly_prob = anomaly_probability
        self.step_count = 0

        # State baselines
        self.base_temp = 80.0
        self.base_press = 110.0
        self.base_vibr = 4.5
        self.base_rpm = 3200.0
        self.base_volt = 230.0

        # Active anomaly state
        self.active_anomaly: AnomalyType = AnomalyType.NONE
        self.anomaly_remaining_steps: int = 0

    def generate_event(
        self, forced_anomaly: Optional[AnomalyType] = None, timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Generate a single telemetry packet with realistic physics or anomaly."""
        self.step_count += 1
        now = timestamp or datetime.now(timezone.utc)
        iso_time = now.isoformat()

        # Handle or trigger anomaly
        if forced_anomaly is not None:
            self.active_anomaly = forced_anomaly
            self.anomaly_remaining_steps = 1
        elif self.anomaly_remaining_steps > 0:
            self.anomaly_remaining_steps -= 1
            if self.anomaly_remaining_steps <= 0:
                self.active_anomaly = AnomalyType.NONE
        elif random.random() < self.anomaly_prob:
            self.active_anomaly = random.choice(
                [
                    AnomalyType.THERMAL_SHOCK,
                    AnomalyType.BEARING_WEAR,
                    AnomalyType.MOTOR_STALL,
                    AnomalyType.SENSOR_FAULT,
                ]
            )
            self.anomaly_remaining_steps = random.randint(3, 8)

        # Baseline harmonic variations (simulate cyclic operating load)
        t_phase = self.step_count * 0.05
        harmonic_load = math.sin(t_phase) * 3.0

        temp = self.base_temp + harmonic_load + random.gauss(0, 1.2)
        press = self.base_press + (harmonic_load * 1.5) + random.gauss(0, 2.0)
        vibr = self.base_vibr + random.gauss(0, 0.4)
        rpm = self.base_rpm + (harmonic_load * 25.0) + random.gauss(0, 15.0)
        volt = self.base_volt + random.gauss(0, 1.0)

        # Apply specific anomaly distortions
        if self.active_anomaly == AnomalyType.THERMAL_SHOCK:
            temp += random.uniform(45.0, 65.0)  # Reaches 125 - 145 C
            press -= random.uniform(40.0, 60.0)  # Drops to 50 - 70 psi
        elif self.active_anomaly == AnomalyType.BEARING_WEAR:
            vibr += random.uniform(20.0, 35.0)  # Reaches 25 - 40 mm/s
            temp += random.uniform(10.0, 20.0)  # Friction heat
        elif self.active_anomaly == AnomalyType.MOTOR_STALL:
            rpm = max(100.0, rpm - random.uniform(1800.0, 2400.0))  # Sudden RPM drop
            volt -= random.uniform(25.0, 45.0)  # Voltage sag
        elif self.active_anomaly == AnomalyType.SENSOR_FAULT:
            # Inject deliberately malformed or impossible values
            fault_mode = random.choice(["extreme_temp", "negative_press", "extreme_rpm", "null_channel"])
            if fault_mode == "extreme_temp":
                temp = 285.0  # Impossible high temp (> 150 C limit)
            elif fault_mode == "negative_press":
                press = -45.0  # Impossible negative pressure (< 0 psi limit)
            elif fault_mode == "extreme_rpm":
                rpm = 15000.0  # Impossible RPM (> 10000 limit)
            elif fault_mode == "null_channel":
                # Returns invalid null value to trigger Great Expectations null check
                return {
                    "device_id": self.device_id,
                    "timestamp": iso_time,
                    "temperature": None,
                    "pressure": round(press, 2),
                    "vibration": round(vibr, 2),
                    "rpm": round(rpm, 1),
                    "voltage": round(volt, 2),
                    "_anomaly_type": AnomalyType.SENSOR_FAULT.value,
                }

        payload: Dict[str, Any] = {
            "device_id": self.device_id,
            "timestamp": iso_time,
            "temperature": round(temp, 2),
            "pressure": round(press, 2),
            "vibration": round(vibr, 2),
            "rpm": round(rpm, 1),
            "voltage": round(volt, 2),
            "_anomaly_type": self.active_anomaly.value,
        }
        return payload
