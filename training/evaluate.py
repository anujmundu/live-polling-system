"""Automated Model Quality Gate (MQG).

Enforces production promotion criteria:
- Precision >= 0.90
- Recall >= 0.88
- False Positive Rate (FPR) <= 0.025 (2.5%)
- p99 Inference Latency <= 15.0 ms
- Model Artifact Size <= 150.0 MB
- F1-Score Improvement >= 0.015 over incumbent Champion
"""

import json
import logging
from typing import Any, Dict, Optional, Tuple

from src.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ModelQualityGate")


class ModelQualityGate:
    """Enforces strict production deployment gates on newly trained models."""

    def __init__(
        self,
        min_precision: Optional[float] = None,
        min_recall: Optional[float] = None,
        max_fpr: Optional[float] = None,
        max_p99_latency_ms: Optional[float] = None,
        max_model_size_mb: Optional[float] = None,
        f1_improvement_threshold: Optional[float] = None,
    ):
        self.min_precision = min_precision or settings.gate_min_precision
        self.min_recall = min_recall or settings.gate_min_recall
        self.max_fpr = max_fpr or settings.gate_max_fpr
        self.max_p99_latency_ms = max_p99_latency_ms or settings.gate_max_p99_latency_ms
        self.max_model_size_mb = max_model_size_mb or settings.gate_max_model_size_mb
        self.f1_improvement_threshold = f1_improvement_threshold or settings.gate_f1_improvement_threshold

    def evaluate(
        self,
        candidate_metrics: Dict[str, Any],
        champion_metrics: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Evaluate candidate model against absolute and comparative gates.

        Returns:
            Tuple of (passed: bool, scorecard: Dict[str, Any])
        """
        checks: Dict[str, Dict[str, Any]] = {}
        all_passed = True

        # 1. Precision Gate
        cand_prec = float(candidate_metrics.get("precision", 0.0))
        prec_pass = cand_prec >= self.min_precision
        checks["precision"] = {
            "required": f">= {self.min_precision}",
            "observed": round(cand_prec, 4),
            "passed": prec_pass,
        }
        if not prec_pass:
            all_passed = False

        # 2. Recall Gate
        cand_rec = float(candidate_metrics.get("recall", 0.0))
        rec_pass = cand_rec >= self.min_recall
        checks["recall"] = {
            "required": f">= {self.min_recall}",
            "observed": round(cand_rec, 4),
            "passed": rec_pass,
        }
        if not rec_pass:
            all_passed = False

        # 3. False Positive Rate (FPR) Gate
        cand_fpr = float(candidate_metrics.get("false_positive_rate", 1.0))
        fpr_pass = cand_fpr <= self.max_fpr
        checks["false_positive_rate"] = {
            "required": f"<= {self.max_fpr}",
            "observed": round(cand_fpr, 4),
            "passed": fpr_pass,
        }
        if not fpr_pass:
            all_passed = False

        # 4. Latency Gate (p99)
        cand_lat = float(candidate_metrics.get("p99_latency_ms", 999.0))
        lat_pass = cand_lat <= self.max_p99_latency_ms
        checks["p99_latency_ms"] = {
            "required": f"<= {self.max_p99_latency_ms} ms",
            "observed": round(cand_lat, 2),
            "passed": lat_pass,
        }
        if not lat_pass:
            all_passed = False

        # 5. Model Size Gate
        cand_size = float(candidate_metrics.get("model_size_mb", 999.0))
        size_pass = cand_size <= self.max_model_size_mb
        checks["model_size_mb"] = {
            "required": f"<= {self.max_model_size_mb} MB",
            "observed": round(cand_size, 2),
            "passed": size_pass,
        }
        if not size_pass:
            all_passed = False

        # 6. Champion vs Challenger Comparative Gate
        f1_cand = float(candidate_metrics.get("f1_score", 0.0))
        if champion_metrics is not None and "f1_score" in champion_metrics:
            f1_champ = float(champion_metrics["f1_score"])
            f1_gain = f1_cand - f1_champ
            comp_pass = f1_gain >= self.f1_improvement_threshold
            checks["f1_improvement_over_champion"] = {
                "required_gain": f">= {self.f1_improvement_threshold}",
                "observed_gain": round(f1_gain, 4),
                "champion_f1": round(f1_champ, 4),
                "candidate_f1": round(f1_cand, 4),
                "passed": comp_pass,
            }
            if not comp_pass:
                all_passed = False
        else:
            # Baseline case: No existing champion
            checks["f1_improvement_over_champion"] = {
                "note": "No active champion. Baseline promotion allowed.",
                "candidate_f1": round(f1_cand, 4),
                "passed": True,
            }

        scorecard = {
            "approved_for_production": all_passed,
            "summary": (
                "PASSED: Candidate approved for Champion promotion."
                if all_passed
                else "REJECTED: Candidate failed one or more Quality Gates."
            ),
            "checks": checks,
        }

        if all_passed:
            logger.info("Model Quality Gate: APPROVED for production deployment.")
        else:
            logger.warning(f"Model Quality Gate: REJECTED. Summary: {json.dumps(checks, indent=2)}")

        return all_passed, scorecard
