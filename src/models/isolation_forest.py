"""Production Champion Model: Isolation Forest Anomaly Detector.

Ensemble of isolation trees partitioning multi-dimensional feature space.
Anomalous points isolate at shorter tree depths.
Provides sub-3ms inference latency with high precision.
"""

import os
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.features.feature_pipeline import FEATURE_NAMES


class IsolationForestDetector:
    """Production Champion Anomaly Detector using Isolation Forest."""

    def __init__(
        self,
        n_estimators: int = 150,
        contamination: float = 0.03,
        max_samples: Union[int, float, str] = "auto",
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_samples = max_samples
        self.random_state = random_state

        self.model: Optional[IsolationForest] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = FEATURE_NAMES
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        """Train Isolation Forest on baseline historical feature vectors."""
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X_scaled)
        self.is_fitted = True
        return self

    def predict_single(self, feature_vector: List[float]) -> Dict[str, Any]:
        """Perform real-time synchronous inference on a single 17-dim feature vector.

        Returns:
            Dict containing is_anomaly (bool), anomaly_score (0.0 to 1.0),
            and top 3 contributing feature names.
        """
        if not self.is_fitted or self.model is None or self.scaler is None:
            raise RuntimeError("Model is not fitted. Train or load weights before inference.")

        X = np.array(feature_vector, dtype=np.float32).reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        # Raw decision function: negative values indicate anomalies (< 0.0)
        raw_score = float(self.model.decision_function(X_scaled)[0])

        # Fast decision: avoid redundant tree traversal (scikit-learn predict is raw_score < 0)
        is_anomaly = bool(raw_score < 0.0)

        # Normalize score into [0.0, 1.0] where 1.0 = highly anomalous
        # Typical raw decision function falls between -0.5 and 0.5
        normalized_score = float(np.clip(0.5 - raw_score, 0.0, 1.0))

        # Explainability: identify top 3 features deviating the most from standard normal
        feature_deviations = np.abs(X_scaled[0])
        top_indices = np.argsort(feature_deviations)[::-1][:3]
        contributing_features = [
            {
                "feature": self.feature_names[i],
                "deviation_sigma": round(float(feature_deviations[i]), 2),
                "value": round(float(feature_vector[i]), 3),
            }
            for i in top_indices
        ]

        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": round(normalized_score, 4),
            "raw_decision_score": round(raw_score, 4),
            "contributing_features": contributing_features,
            "model_type": "ISOLATION_FOREST",
        }

    def save(self, filepath: str) -> None:
        """Persist model and scaler artifacts to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        artifact = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "params": {
                "n_estimators": self.n_estimators,
                "contamination": self.contamination,
                "max_samples": self.max_samples,
                "random_state": self.random_state,
            },
        }
        joblib.dump(artifact, filepath)

    @classmethod
    def load(cls, filepath: str) -> "IsolationForestDetector":
        """Load model and scaler artifacts from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model artifact not found at {filepath}")

        artifact = joblib.load(filepath)
        instance = cls(**artifact["params"])
        instance.model = artifact["model"]
        instance.scaler = artifact["scaler"]
        instance.feature_names = artifact.get("feature_names", FEATURE_NAMES)
        instance.is_fitted = True
        return instance
