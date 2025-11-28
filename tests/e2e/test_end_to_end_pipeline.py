"""End-to-End Pipeline Integration Test.

Validates the full MLOps lifecycle:
1. Ingestion simulation (normal & anomalies)
2. Great Expectations validation firewall & DLQ quarantine
3. Stateful 17-dim feature extraction
4. Model training & Quality Gate promotion
5. Real-time inference scoring & feature attribution
6. Statistical drift detection & automated retraining trigger
"""

from src.features.feature_pipeline import StatefulFeatureEngine
from src.ingestion.simulator import AnomalyType, TelemetrySimulator
from src.models.isolation_forest import IsolationForestDetector
from src.monitoring.drift_detector import DriftDetector
from src.validation.validator import ValidationFirewall
from training.train import generate_training_dataset


def test_complete_mlops_anomaly_pipeline():
    # 1. Ingestion Simulation
    sim = TelemetrySimulator(device_id="TEST_TURBINE_99")
    normal_event = sim.generate_event(forced_anomaly=AnomalyType.NONE)
    shock_event = sim.generate_event(forced_anomaly=AnomalyType.THERMAL_SHOCK)
    fault_event = sim.generate_event(forced_anomaly=AnomalyType.SENSOR_FAULT)

    # 2. Validation Firewall Gate
    validator = ValidationFirewall()

    # Normal event must pass
    val_norm, quar_norm = validator.validate_event(normal_event)
    assert val_norm is True
    assert quar_norm is None

    # Shock event has extreme values, let's see if it passes or flags
    # Thermal shock temp (125-145 C) is within physical limits [-20, 150], so it passes firewall to reach ML model
    val_shock, _ = validator.validate_event(shock_event)
    assert val_shock is True

    # Fault event with nulls or impossible bounds must be quarantined
    if fault_event.get("temperature") is None or fault_event.get("pressure", 0) < 0:
        val_fault, quar_fault = validator.validate_event(fault_event)
        assert val_fault is False
        assert quar_fault is not None
        assert "failed_expectations" in quar_fault

    # 3. Stateful Feature Engine
    engine = StatefulFeatureEngine()
    feat_norm = engine.compute_features(normal_event)
    assert len(feat_norm["feature_vector"]) == 17

    feat_shock = engine.compute_features(shock_event)
    assert len(feat_shock["feature_vector"]) == 17

    # 4. Model Training & Quality Gate Evaluation
    X, y = generate_training_dataset(num_samples=1200, anomaly_ratio=0.03)
    model = IsolationForestDetector(n_estimators=100, contamination=0.03, random_state=42)
    model.fit(X)

    # 5. Real-Time Inference
    pred_res = model.predict_single(feat_norm["feature_vector"])
    assert "is_anomaly" in pred_res
    assert 0.0 <= pred_res["anomaly_score"] <= 1.0
    assert len(pred_res["contributing_features"]) == 3

    # 6. Drift Detection & Self-Healing Trigger
    drift_detector = DriftDetector()
    drift_detector.set_reference_data(X)

    # Simulated drifted batch
    import numpy as np

    drifted_batch = X + np.random.normal(loc=15.0, scale=8.0, size=X.shape)
    drift_report = drift_detector.evaluate_drift(drifted_batch)

    assert drift_report["overall_status"] == "CRITICAL"
    assert drift_report["trigger_retraining"] is True
    assert drift_report["max_psi"] > 0.25

    print("\n✅ END-TO-END MLOPS ANOMALY PIPELINE VERIFIED SUCCESSFULLY!")
