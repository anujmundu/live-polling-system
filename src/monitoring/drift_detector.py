"""Continuous Statistical Drift Detector.

Monitors real-time feature distributions against baseline reference distributions
using Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) tests.
Emits CRITICAL status and triggers automated retraining when PSI > 0.25.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats

from src.config import settings
from src.features.feature_pipeline import FEATURE_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("DriftDetector")


class DriftDetector:
    """Detects covariate and feature distribution shifts in streaming telemetry."""

    def __init__(
        self,
        psi_warning: Optional[float] = None,
        psi_critical: Optional[float] = None,
        ks_p_value_threshold: Optional[float] = None,
        num_quantiles: int = 10,
    ):
        self.psi_warning = psi_warning or settings.drift_psi_threshold_warning
        self.psi_critical = psi_critical or settings.drift_psi_threshold_critical
        self.ks_p_threshold = ks_p_value_threshold or settings.drift_ks_p_value_threshold
        self.num_quantiles = num_quantiles
        self.feature_names: List[str] = FEATURE_NAMES

        self.reference_data: Optional[np.ndarray] = None
        self.reference_quantiles: Dict[int, np.ndarray] = {}

    def set_reference_data(self, reference_matrix: np.ndarray) -> None:
        """Establish baseline distribution from verified training feature matrix."""
        self.reference_data = np.asarray(reference_matrix, dtype=np.float64)
        n_features = self.reference_data.shape[1]

        # Precompute quantile bin boundaries for each feature
        self.reference_quantiles.clear()
        percentiles = np.linspace(0, 100, self.num_quantiles + 1)
        for f_idx in range(n_features):
            col_data = self.reference_data[:, f_idx]
            bins = np.percentile(col_data, percentiles)
            # Ensure strictly monotonically increasing bin edges
            bins[0] = -np.inf
            bins[-1] = np.inf
            for i in range(1, len(bins) - 1):
                if bins[i] <= bins[i - 1]:
                    bins[i] = bins[i - 1] + 1e-5
            self.reference_quantiles[f_idx] = bins

        logger.info(f"Reference baseline set: {len(reference_matrix)} samples across {n_features} features.")

    def calculate_psi(self, expected: np.ndarray, actual: np.ndarray, bins: np.ndarray) -> float:
        """Compute Population Stability Index (PSI) across specified quantile bins."""
        eps = 1e-4

        expected_counts, _ = np.histogram(expected, bins=bins)
        actual_counts, _ = np.histogram(actual, bins=bins)

        expected_pct = (expected_counts / len(expected)) + eps
        actual_pct = (actual_counts / len(actual)) + eps

        psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return float(max(0.0, psi_value))

    def evaluate_drift(self, current_matrix: np.ndarray) -> Dict[str, Any]:
        """Compare current inference feature window against reference baseline."""
        if self.reference_data is None or not self.reference_quantiles:
            raise RuntimeError("Reference baseline not set. Call set_reference_data first.")

        current = np.asarray(current_matrix, dtype=np.float64)
        n_features = min(current.shape[1], len(self.feature_names))

        feature_reports: List[Dict[str, Any]] = []
        max_psi = 0.0
        drifted_features_count = 0

        for f_idx in range(n_features):
            feat_name = self.feature_names[f_idx]
            ref_col = self.reference_data[:, f_idx]
            curr_col = current[:, f_idx]
            bins = self.reference_quantiles[f_idx]

            # 1. Population Stability Index
            psi = self.calculate_psi(ref_col, curr_col, bins)
            max_psi = max(max_psi, psi)

            # 2. Two-sample Kolmogorov-Smirnov Test
            ks_stat, ks_p = stats.ks_2samp(ref_col, curr_col)

            # Drift Severity per feature
            if psi > self.psi_critical:
                feat_status = "CRITICAL"
                drifted_features_count += 1
            elif psi >= self.psi_warning or (ks_p < self.ks_p_threshold and ks_stat > 0.15):
                feat_status = "WARNING"
                drifted_features_count += 1
            else:
                feat_status = "NORMAL"

            feature_reports.append(
                {
                    "feature": feat_name,
                    "psi": round(psi, 4),
                    "ks_statistic": round(float(ks_stat), 4),
                    "ks_p_value": round(float(ks_p), 6),
                    "status": feat_status,
                }
            )

        # Overall System Status
        if max_psi > self.psi_critical or drifted_features_count >= 3:
            overall_status = "CRITICAL"
            trigger_retraining = True
        elif max_psi >= self.psi_warning or drifted_features_count >= 1:
            overall_status = "WARNING"
            trigger_retraining = False
        else:
            overall_status = "NORMAL"
            trigger_retraining = False

        report = {
            "overall_status": overall_status,
            "trigger_retraining": trigger_retraining,
            "max_psi": round(max_psi, 4),
            "drifted_features_count": drifted_features_count,
            "sample_size": len(current),
            "features": feature_reports,
        }

        if trigger_retraining:
            logger.critical(
                f"🚨 SEVERE DATA DRIFT DETECTED | Max PSI: {max_psi:.4f} | " "Triggering automated retraining pipeline!"
            )
        elif overall_status == "WARNING":
            logger.warning(f"⚠️ Moderate drift warning | Max PSI: {max_psi:.4f}")

        return report
