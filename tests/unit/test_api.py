"""Integration tests for FastAPI inference endpoints."""

import pytest
from fastapi.testclient import TestClient
from api.main import app, serving_ctx


@pytest.fixture(scope="module")
def client():
    # Explicitly ensure champion model is loaded
    serving_ctx.load_champion_model()
    with TestClient(app) as c:
        yield c


def test_api_health_endpoint(client):
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ["HEALTHY", "DEGRADED"]
    assert data["model_type"] == "ISOLATION_FOREST"
    assert data["service_name"] == "mlops-anomaly-detection"


def test_api_predict_valid_normal_event(client):
    payload = {
        "device_id": "MACHINE_042",
        "timestamp": "2026-09-26T14:20:00Z",
        "temperature": 82.5,
        "pressure": 110.0,
        "vibration": 4.2,
        "rpm": 3200.0,
        "voltage": 230.0,
    }
    resp = client.post("/v1/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["device_id"] == "MACHINE_042"
    assert "is_anomaly" in data
    assert 0.0 <= data["anomaly_score"] <= 1.0
    assert len(data["contributing_features"]) == 3
    assert data["inference_latency_ms"] > 0


def test_api_predict_quarantines_corrupt_payload(client):
    # Negative pressure is physically impossible and must be rejected by data quality firewall
    corrupt_payload = {
        "device_id": "MACHINE_042",
        "timestamp": "2026-09-26T14:20:00Z",
        "temperature": 82.5,
        "pressure": -40.0,
        "vibration": 4.2,
        "rpm": 3200.0,
        "voltage": 230.0,
    }
    resp = client.post("/v1/predict", json=corrupt_payload)
    assert resp.status_code == 422
    err_data = resp.json()
    assert "quarantine" in err_data["detail"]


def test_api_metrics_endpoint(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "anomaly_predictions_total" in resp.text
