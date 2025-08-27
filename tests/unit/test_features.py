"""Unit tests for stateful feature pipeline."""

from datetime import datetime, timedelta, timezone
from src.features.feature_pipeline import FEATURE_NAMES, StatefulFeatureEngine


def test_feature_engine_computes_17_dimensions():
    engine = StatefulFeatureEngine()
    event = {
        "device_id": "TURBINE_A",
        "timestamp": "2026-09-26T14:20:00Z",
        "temperature": 85.0,
        "pressure": 110.0,
        "vibration": 4.5,
        "rpm": 3200.0,
        "voltage": 230.0,
    }

    res = engine.compute_features(event)
    assert len(res["feature_vector"]) == 17
    assert res["feature_names"] == FEATURE_NAMES
    assert res["device_id"] == "TURBINE_A"


def test_feature_engine_rolling_and_deltas():
    engine = StatefulFeatureEngine()
    base_time = datetime(2026, 9, 26, 14, 0, 0, tzinfo=timezone.utc)

    # First event
    ev1 = {
        "device_id": "TURBINE_B",
        "timestamp": base_time.isoformat(),
        "temperature": 80.0,
        "pressure": 100.0,
        "vibration": 3.0,
        "rpm": 3000.0,
        "voltage": 230.0,
    }
    res1 = engine.compute_features(ev1)
    assert res1["feature_dict"]["temperature_delta"] == 0.0

    # Second event 1 minute later with temperature jump
    ev2 = {
        "device_id": "TURBINE_B",
        "timestamp": (base_time + timedelta(minutes=1)).isoformat(),
        "temperature": 88.0,
        "pressure": 102.0,
        "vibration": 3.5,
        "rpm": 3050.0,
        "voltage": 229.0,
    }
    res2 = engine.compute_features(ev2)
    assert res2["feature_dict"]["temperature_delta"] == 8.0
    assert res2["feature_dict"]["rolling_mean_5m_temp"] == 84.0  # (80 + 88) / 2
    assert res2["feature_dict"]["sensor_velocity"] > 0.0
