"""Automated Retraining and Self-Healing Pipeline.

Triggered event-driven when data drift is detected (PSI > 0.25) or via scheduled cron.
Retrains the anomaly detection model, runs Model Quality Gate validation,
and executes a zero-downtime hot-swap on serving pods via the reload webhook.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
import numpy as np

from src.config import settings
from training.evaluate import ModelQualityGate
from training.train import benchmark_latency, generate_training_dataset
from src.models.isolation_forest import IsolationForestDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("RetrainingPipeline")


def execute_retraining_workflow(
    lookback_samples: int = 4000,
    model_output_path: str = "models/champion_isolation_forest.joblib",
    api_endpoint: Optional[str] = None,
    force_promote: bool = False,
) -> Dict[str, Any]:
    """Run closed-loop retraining, evaluation, and hot-swap."""
    logger.info("==================================================================")
    logger.info("🚀 Initiating Automated Closed-Loop Model Retraining Pipeline")
    logger.info("==================================================================")

    # 1. Fetch current Champion metrics for comparative Quality Gate
    champion_metrics: Optional[Dict[str, Any]] = None
    if os.path.exists(model_output_path):
        try:
            current_champion = IsolationForestDetector.load(model_output_path)
            # Benchmark current champion on a validation slice
            X_val, y_val = generate_training_dataset(num_samples=800, anomaly_ratio=0.035)
            preds = [1 if current_champion.predict_single(x.tolist())["is_anomaly"] else 0 for x in X_val]
            from sklearn.metrics import f1_score
            champ_f1 = float(f1_score(y_val, preds, zero_division=0))
            champion_metrics = {"f1_score": champ_f1}
            logger.info(f"Loaded existing Champion model baseline (F1 = {champ_f1:.4f})")
        except Exception as exc:
            logger.warning(f"Could not benchmark incumbent champion ({exc}). Proceeding without comparative gate.")

    # 2. Extract recent validated dataset reflecting updated operating regime
    logger.info(f"Extracting recent telemetry window ({lookback_samples} samples)...")
    X, y = generate_training_dataset(num_samples=lookback_samples, anomaly_ratio=0.035)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 3. Train Candidate Isolation Forest
    observed_contamination = max(0.01, min(0.3, float(np.mean(y_train)) * 1.05))
    logger.info(f"Training Candidate Isolation Forest model (contamination={observed_contamination:.4f})...")
    candidate_model = IsolationForestDetector(n_estimators=100, contamination=observed_contamination, random_state=int(time.time()))
    candidate_model.fit(X_train)

    # 4. Evaluate Candidate Performance
    from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

    preds = [1 if candidate_model.predict_single(x.tolist())["is_anomaly"] else 0 for x in X_test]
    prec = float(precision_score(y_test, preds, zero_division=0))
    rec = float(recall_score(y_test, preds, zero_division=0))
    f1 = float(f1_score(y_test, preds, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
    fpr = float(fp / max(1, (fp + tn)))

    p95_lat, p99_lat = benchmark_latency(candidate_model, X_test)

    temp_cand_path = model_output_path + ".cand"
    candidate_model.save(temp_cand_path)
    model_size_mb = float(os.path.getsize(temp_cand_path) / (1024 * 1024))

    candidate_metrics = {
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "false_positive_rate": fpr,
        "p99_latency_ms": p99_lat,
        "model_size_mb": model_size_mb,
    }

    # 5. Programmatic Model Quality Gate
    gate = ModelQualityGate()
    passed, scorecard = gate.evaluate(candidate_metrics, champion_metrics=champion_metrics)

    if not passed and not force_promote:
        if os.path.exists(temp_cand_path):
            os.remove(temp_cand_path)
        logger.error("❌ RETRAINING FAILED: Candidate model did not satisfy Model Quality Gate.")
        return {
            "status": "REJECTED",
            "candidate_metrics": candidate_metrics,
            "scorecard": scorecard,
        }

    # 6. Promote Candidate to Production Champion
    if os.path.exists(model_output_path):
        os.remove(model_output_path)
    os.rename(temp_cand_path, model_output_path)
    logger.info(f"✅ Model APPROVED: Promoted candidate to Champion at {model_output_path}")

    # 7. Hot-Swap Model on Live Serving Pods
    target_host = "127.0.0.1" if settings.api_host in ["0.0.0.0", "::"] else settings.api_host
    base_url = (api_endpoint or f"http://{target_host}:{settings.api_port}").rstrip("/")
    reload_url = base_url if base_url.endswith("/v1/model/reload") else f"{base_url}/v1/model/reload"
    reload_success = False
    try:
        resp = httpx.post(reload_url, timeout=5.0)
        if resp.status_code == 200:
            logger.info(f"🔄 Zero-Downtime Hot-Swap Triggered on Inference API: {resp.json()}")
            reload_success = True
        else:
            logger.warning(f"Inference API reload returned non-200: {resp.status_code}")
    except Exception as exc:
        logger.info(f"Inference API not running or unreachable ({exc}). Model ready on disk for next pod startup.")

    return {
        "status": "PROMOTED",
        "candidate_metrics": candidate_metrics,
        "scorecard": scorecard,
        "promoted_path": model_output_path,
        "hot_swap_triggered": reload_success,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Model Retraining Pipeline")
    parser.add_argument("--lookback-samples", type=int, default=3000, help="Number of telemetry samples")
    parser.add_argument("--force-promote", action="store_true", help="Force promotion regardless of gate")
    parser.add_argument("--api-endpoint", type=str, default=None, help="Inference API URL (e.g. http://127.0.0.1:8000)")
    args = parser.parse_args()

    execute_retraining_workflow(
        lookback_samples=args.lookback_samples,
        force_promote=args.force_promote,
        api_endpoint=args.api_endpoint,
    )
