"""Unit tests for statistical drift detection."""

import numpy as np
from src.monitoring.drift_detector import DriftDetector


def test_drift_detector_normal_distribution():
    detector = DriftDetector()
    np.random.seed(42)

    # Reference data: standard normal across 17 features
    ref = np.random.normal(loc=0.0, scale=1.0, size=(1000, 17))
    detector.set_reference_data(ref)

    # Current data from same distribution
    curr = np.random.normal(loc=0.0, scale=1.0, size=(1000, 17))
    report = detector.evaluate_drift(curr)

    assert report["overall_status"] == "NORMAL"
    assert report["trigger_retraining"] is False
    assert report["max_psi"] < 0.10


def test_drift_detector_severe_drift_triggers_retraining():
    detector = DriftDetector()
    np.random.seed(42)

    # Reference baseline
    ref = np.random.normal(loc=50.0, scale=5.0, size=(1000, 17))
    detector.set_reference_data(ref)

    # Drastic distribution shift (simulating worn machine or seasonal drift)
    drifted = np.random.normal(loc=85.0, scale=15.0, size=(1000, 17))
    report = detector.evaluate_drift(drifted)

    assert report["overall_status"] == "CRITICAL"
    assert report["trigger_retraining"] is True
    assert report["max_psi"] > 0.25
