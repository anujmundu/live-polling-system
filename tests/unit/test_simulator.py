"""Unit tests for industrial telemetry simulator."""

from src.ingestion.simulator import AnomalyType, TelemetrySimulator


def test_simulator_normal_generation():
    sim = TelemetrySimulator(device_id="TEST_MACHINE_01", anomaly_probability=0.0)
    event = sim.generate_event()

    assert event["device_id"] == "TEST_MACHINE_01"
    assert "timestamp" in event
    assert 60.0 <= event["temperature"] <= 100.0
    assert 80.0 <= event["pressure"] <= 140.0
    assert 1.0 <= event["vibration"] <= 15.0
    assert 2500.0 <= event["rpm"] <= 4000.0
    assert 215.0 <= event["voltage"] <= 245.0
    assert event["_anomaly_type"] == AnomalyType.NONE.value


def test_simulator_forced_thermal_shock():
    sim = TelemetrySimulator()
    event = sim.generate_event(forced_anomaly=AnomalyType.THERMAL_SHOCK)
    assert event["_anomaly_type"] == AnomalyType.THERMAL_SHOCK.value
    assert event["temperature"] > 115.0  # Thermal shock elevates temperature


def test_simulator_forced_bearing_wear():
    sim = TelemetrySimulator()
    event = sim.generate_event(forced_anomaly=AnomalyType.BEARING_WEAR)
    assert event["_anomaly_type"] == AnomalyType.BEARING_WEAR.value
    assert event["vibration"] > 20.0  # Bearing wear elevates vibration
