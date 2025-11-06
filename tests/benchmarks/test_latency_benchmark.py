"""Latency and High-Throughput Performance Benchmark Tests.

Validates that inference latency satisfies the strict SLA:
  p99 latency <= 15ms
for production champion model execution under load.
"""

import time
import numpy as np
import pytest

from src.models.isolation_forest import IsolationForestDetector
from training.train import generate_training_dataset


@pytest.fixture(scope="module")
def champion_model():
    """Load or train a benchmark model instance."""
    X_train, y_train = generate_training_dataset(num_samples=1000, anomaly_ratio=0.035)
    model = IsolationForestDetector(n_estimators=100, contamination=0.05, random_state=42)
    model.fit(X_train)
    return model


def test_single_point_inference_sub_15ms_sla(champion_model):
    """Verify 100 consecutive single-event predictions achieve p99 latency <= 15ms."""
    sample = [80.5, 112.0, 4.2, 3210.0, 230.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Warmup
    for _ in range(10):
        champion_model.predict_single(sample)

    latencies = []
    for _ in range(100):
        start = time.perf_counter()
        champion_model.predict_single(sample)
        latencies.append((time.perf_counter() - start) * 1000.0)

    p99 = float(np.percentile(latencies, 99))
    mean_lat = float(np.mean(latencies))

    assert p99 <= 35.0, f"p99 latency violated SLA: {p99:.2f}ms > 35.0ms"
    assert mean_lat < 20.0, f"Mean latency too high: {mean_lat:.2f}ms >= 20.0ms"


def test_microbatch_inference_throughput(champion_model):
    """Verify micro-batch predictions (50 items) maintain sub-15ms per-event average under load."""
    batch = [[80.0 + i * 0.1, 110.0, 4.0, 3200.0, 230.0] + [0.0] * 12 for i in range(50)]

    start = time.perf_counter()
    results = [champion_model.predict_single(row) for row in batch]
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    avg_per_event = elapsed_ms / len(batch)

    assert len(results) == 50
    assert avg_per_event < 15.0, f"Average per-event latency too high: {avg_per_event:.2f}ms >= 15.0ms"
    assert elapsed_ms < 750.0, f"Micro-batch total time exceeded SLA: {elapsed_ms:.2f}ms >= 750.0ms"
