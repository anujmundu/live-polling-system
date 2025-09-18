"""Unit tests for Model Quality Gate."""

from training.evaluate import ModelQualityGate


def test_quality_gate_passes_healthy_metrics():
    gate = ModelQualityGate(
        min_precision=0.90,
        min_recall=0.88,
        max_fpr=0.025,
        max_p99_latency_ms=15.0,
        max_model_size_mb=150.0,
    )
    candidate = {
        "precision": 0.94,
        "recall": 0.91,
        "f1_score": 0.925,
        "false_positive_rate": 0.018,
        "p99_latency_ms": 6.5,
        "model_size_mb": 12.0,
    }
    passed, scorecard = gate.evaluate(candidate)
    assert passed is True
    assert scorecard["approved_for_production"] is True


def test_quality_gate_rejects_high_false_positive_rate():
    gate = ModelQualityGate(max_fpr=0.025)
    candidate = {
        "precision": 0.94,
        "recall": 0.91,
        "f1_score": 0.925,
        "false_positive_rate": 0.045,  # Exceeds 2.5% max
        "p99_latency_ms": 6.5,
        "model_size_mb": 12.0,
    }
    passed, scorecard = gate.evaluate(candidate)
    assert passed is False
    assert scorecard["checks"]["false_positive_rate"]["passed"] is False


def test_quality_gate_rejects_high_latency():
    gate = ModelQualityGate(max_p99_latency_ms=15.0)
    candidate = {
        "precision": 0.94,
        "recall": 0.91,
        "f1_score": 0.925,
        "false_positive_rate": 0.015,
        "p99_latency_ms": 28.5,  # Exceeds 15ms SLA
        "model_size_mb": 12.0,
    }
    passed, scorecard = gate.evaluate(candidate)
    assert passed is False
    assert scorecard["checks"]["p99_latency_ms"]["passed"] is False
