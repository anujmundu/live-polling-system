"""Training Pipeline & Model Registration.

Generates representative industrial training data with simulated failures,
computes 17-dimensional stateful features, trains Isolation Forest detector,
computes precision, recall, FPR, and latency metrics, evaluates against
Model Quality Gate, and logs run to MLflow.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from src.config import settings
from src.features.feature_pipeline import StatefulFeatureEngine
from src.ingestion.simulator import TelemetrySimulator
from src.models.isolation_forest import IsolationForestDetector
from training.evaluate import ModelQualityGate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("TrainingPipeline")


def generate_training_dataset(num_samples: int = 2500, anomaly_ratio: float = 0.04) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic telemetry data and extract 17-dimensional feature vectors."""
    logger.info(f"Generating synthetic training dataset: {num_samples} samples (anomaly_ratio={anomaly_ratio})...")
    simulator = TelemetrySimulator(anomaly_probability=anomaly_ratio)
    engine = StatefulFeatureEngine()

    feature_matrix: List[List[float]] = []
    labels: List[int] = []  # 0 = normal, 1 = anomaly

    start_time = datetime.now(timezone.utc) - timedelta(minutes=num_samples)

    for i in range(num_samples):
        current_time = start_time + timedelta(seconds=i * 60)
        event = simulator.generate_event(timestamp=current_time)

        # Skip sensor faults that violate physical boundaries (these are caught by Great Expectations DLQ)
        temp_val = event.get("temperature")
        press_val = event.get("pressure")
        rpm_val = event.get("rpm")
        if (
            temp_val is None
            or press_val is None
            or rpm_val is None
            or press_val < 0.0
            or press_val > 300.0
            or temp_val < 0.0
            or temp_val > 150.0
            or rpm_val < 0.0
            or rpm_val > 10000.0
        ):
            continue

        feature_record = engine.compute_features(event)
        feature_matrix.append(feature_record["feature_vector"])

        # Ground truth label
        anomaly_type = event.get("_anomaly_type", "NONE")
        labels.append(1 if anomaly_type != "NONE" else 0)

    X = np.array(feature_matrix, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    logger.info(f"Dataset generated: X shape={X.shape}, anomalies={np.sum(y)} ({np.mean(y)*100:.2f}%)")
    return X, y


def is_mlflow_available(uri: str) -> bool:
    """Fast socket probe to check if MLflow server is reachable within 500ms."""
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(uri)
    host = parsed.hostname or "localhost"
    port = parsed.port or 5000
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def benchmark_latency(model: IsolationForestDetector, X_test: np.ndarray, n_trials: int = 200) -> Tuple[float, float]:
    """Measure per-sample inference latency (p95 and p99 in ms)."""
    # Warmup
    for i in range(min(10, len(X_test))):
        _ = model.predict_single(X_test[i].tolist())

    latencies = []
    sample_indices = np.random.choice(len(X_test), size=min(n_trials, len(X_test)), replace=True)

    for idx in sample_indices:
        vec = X_test[idx].tolist()
        t0 = time.perf_counter()
        _ = model.predict_single(vec)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    p95 = float(np.percentile(latencies, 95))
    p99 = float(np.percentile(latencies, 99))
    return p95, p99


def run_training_pipeline(
    output_model_path: str = "models/champion_isolation_forest.joblib", auto_promote: bool = True
) -> Dict[str, Any]:
    """Execute end-to-end training, benchmarking, quality gating, and artifact saving."""
    X, y = generate_training_dataset(num_samples=3000, anomaly_ratio=0.035)

    # 80/20 train/test split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Dynamically calibrate contamination factor to match observed anomaly prevalence
    observed_contamination = max(0.01, min(0.3, float(np.mean(y_train)) * 1.05))
    logger.info(f"Fitting Isolation Forest Champion model (contamination={observed_contamination:.4f})...")
    model = IsolationForestDetector(n_estimators=100, contamination=observed_contamination, random_state=42)
    model.fit(X_train)

    # Evaluate on test set
    predictions = []
    for i in range(len(X_test)):
        res = model.predict_single(X_test[i].tolist())
        predictions.append(1 if res["is_anomaly"] else 0)
    y_pred = np.array(predictions)

    # Compute classification metrics
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / max(1, (fp + tn)))

    # Measure inference latency
    p95_lat, p99_lat = benchmark_latency(model, X_test)

    # Measure artifact size
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    temp_path = output_model_path + ".tmp"
    model.save(temp_path)
    model_size_mb = float(os.path.getsize(temp_path) / (1024 * 1024))

    candidate_metrics = {
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "false_positive_rate": fpr,
        "p95_latency_ms": p95_lat,
        "p99_latency_ms": p99_lat,
        "model_size_mb": model_size_mb,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }

    logger.info(f"Evaluation Metrics: {json.dumps(candidate_metrics, indent=2)}")

    # Evaluate against Model Quality Gate
    gate = ModelQualityGate()
    passed, scorecard = gate.evaluate(candidate_metrics)

    # Log to MLflow if tracking server is online
    if is_mlflow_available(settings.mlflow_tracking_uri):
        try:
            import mlflow

            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            mlflow.set_experiment(settings.mlflow_experiment_name)

            with mlflow.start_run(run_name=f"isolation_forest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                mlflow.log_params(
                    {
                        "model_type": "IsolationForest",
                        "n_estimators": model.n_estimators,
                        "contamination": model.contamination,
                        "n_features": len(model.feature_names),
                    }
                )
                mlflow.log_metrics(
                    {
                        "precision": prec,
                        "recall": rec,
                        "f1_score": f1,
                        "false_positive_rate": fpr,
                        "p99_latency_ms": p99_lat,
                    }
                )
                mlflow.log_dict(scorecard, "model_quality_scorecard.json")
                logger.info("Successfully logged metrics to MLflow Tracking Server.")
        except Exception as exc:
            logger.warning(f"MLflow logging error ({exc}). Continuing local promotion.")
    else:
        logger.info("MLflow server not active at %s. Skipped remote tracking.", settings.mlflow_tracking_uri)

    # Promotion decision: approve if quality gate passes, or bootstrap initial baseline
    is_initial_bootstrap = not os.path.exists(output_model_path) or "--bootstrap" in sys.argv
    if (passed or is_initial_bootstrap) and auto_promote:
        if os.path.exists(output_model_path):
            os.remove(output_model_path)
        os.rename(temp_path, output_model_path)
        logger.info(f"Model {'APPROVED' if passed else 'BOOTSTRAPPED'} and saved as Champion at: {output_model_path}")
    else:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        logger.warning("Model failed quality gate or promotion was disabled. Artifact discarded.")

    return {
        "candidate_metrics": candidate_metrics,
        "quality_gate_passed": passed,
        "scorecard": scorecard,
        "promoted_artifact_path": output_model_path if passed else None,
    }


if __name__ == "__main__":
    run_training_pipeline()
