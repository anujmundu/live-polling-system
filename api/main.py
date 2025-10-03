"""Production Real-Time Inference Microservice (FastAPI).

Provides low-latency anomaly scoring, validation gating, Prometheus metrics,
and hot-reload capability for zero-downtime model promotion.
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Response, status
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from src.config import settings
from src.features.feature_pipeline import StatefulFeatureEngine
from src.models.isolation_forest import IsolationForestDetector
from src.validation.validator import ValidationFirewall

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("InferenceService")

# ==============================================================================
# Prometheus Observability Metrics
# ==============================================================================
REQUEST_COUNT = Counter("api_requests_total", "Total HTTP requests to the inference service", ["endpoint", "status"])
PREDICTIONS_TOTAL = Counter("anomaly_predictions_total", "Total individual telemetry records evaluated for anomalies")
ANOMALIES_DETECTED = Counter(
    "anomalies_detected_total", "Total industrial anomalies flagged by the champion model", ["device_id"]
)
VALIDATION_FAILURES = Counter("validation_failures_total", "Total telemetry records failing data quality firewall")
INFERENCE_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Distribution of single-point inference latency in seconds",
    buckets=[0.001, 0.003, 0.005, 0.010, 0.015, 0.025, 0.050, 0.100],
)


# ==============================================================================
# Pydantic v2 Schemas
# ==============================================================================
class TelemetryPayload(BaseModel):
    """Raw sensor telemetry payload."""

    device_id: str = Field(..., examples=["MACHINE_042"], min_length=3, max_length=32)
    timestamp: str = Field(..., examples=["2026-09-26T14:20:00Z"])
    temperature: float = Field(..., examples=[87.4], description="Operating temp in °C")
    pressure: float = Field(..., examples=[104.2], description="Pressure in psi")
    vibration: float = Field(..., examples=[8.91], description="Vibration in mm/s")
    rpm: float = Field(..., examples=[3120.0], description="Revolutions per minute")
    voltage: float = Field(..., examples=[228.7], description="AC supply voltage")


class FeatureAttribution(BaseModel):
    feature: str
    deviation_sigma: float
    value: float


class PredictionResponse(BaseModel):
    """Inference output contract."""

    device_id: str
    timestamp: str
    is_anomaly: bool
    anomaly_score: float = Field(..., ge=0.0, le=1.0)
    model_version: str
    model_type: str
    contributing_features: List[FeatureAttribution]
    inference_latency_ms: float


class BatchPredictionRequest(BaseModel):
    events: List[TelemetryPayload]


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    processed_count: int
    anomalies_count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str
    uptime_seconds: float
    service_name: str


# ==============================================================================
# Model Serving Context & State
# ==============================================================================
class ServingContext:
    def __init__(self):
        self.model: Optional[IsolationForestDetector] = None
        self.validator = ValidationFirewall()
        self.feature_engine = StatefulFeatureEngine()
        self.model_path = "models/champion_isolation_forest.joblib"
        self.model_version = "v1.0.0"
        self.start_time = time.time()

    def load_champion_model(self) -> bool:
        """Load or bootstrap champion model."""
        if os.path.exists(self.model_path):
            try:
                self.model = IsolationForestDetector.load(self.model_path)
                mtime = int(os.path.getmtime(self.model_path))
                self.model_version = f"v{mtime}"
                logger.info(f"Loaded Champion Isolation Forest [{self.model_version}] from '{self.model_path}'")
                return True
            except Exception as exc:
                logger.error(f"Failed to load model from {self.model_path}: {exc}")

        # If artifact not on disk, train in-memory baseline
        logger.info("No persisted model found. Training initial baseline model...")
        try:
            from training.train import generate_training_dataset

            X, _ = generate_training_dataset(num_samples=1500, anomaly_ratio=0.03)
            self.model = IsolationForestDetector(n_estimators=100, contamination=0.03)
            self.model.fit(X)
            self.model.save(self.model_path)
            logger.info("Trained and persisted initial baseline Champion model.")
            return True
        except Exception as exc:
            logger.error(f"Failed to bootstrap initial model: {exc}")
            return False


serving_ctx = ServingContext()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for FastAPI initialization."""
    logger.info("Initializing Real-Time Inference Microservice...")
    serving_ctx.load_champion_model()
    yield
    logger.info("Shutting down Real-Time Inference Microservice...")


app = FastAPI(
    title="Real-Time Industrial Anomaly Detection Service",
    description="Low-latency MLOps inference microservice for multi-sensor IoT telemetry.",
    version="1.0.0",
    lifespan=lifespan,
)


def dispatch_anomaly_alert(prediction: PredictionResponse) -> None:
    """Asynchronous background alert handler."""
    logger.warning(
        f"🚨 ANOMALY ALERT DISPATCHED | Device: {prediction.device_id} | "
        f"Score: {prediction.anomaly_score} | Top Sensor: {prediction.contributing_features[0].feature if prediction.contributing_features else 'N/A'}"
    )


# ==============================================================================
# API Endpoints
# ==============================================================================
@app.get("/")
async def root():
    """Welcome and platform service discovery."""
    return {
        "service": "Real-Time Industrial Anomaly Detection Service",
        "system_designation": "IT-ADS",
        "version": "1.0.0",
        "status": "OPERATIONAL",
        "champion_model": "Isolation Forest",
        "endpoints": {
            "docs": "/docs",
            "health": "/v1/health",
            "predict": "/v1/predict",
            "batch_predict": "/v1/predict/batch",
            "model_reload": "/v1/model/reload",
            "metrics": "/metrics",
        },
    }


@app.get("/v1/health", response_model=HealthResponse)
async def health_check():
    """Kubernetes liveness and readiness probe."""
    uptime = time.time() - serving_ctx.start_time
    is_ready = serving_ctx.model is not None and serving_ctx.model.is_fitted
    return HealthResponse(
        status="HEALTHY" if is_ready else "DEGRADED",
        model_loaded=is_ready,
        model_type="ISOLATION_FOREST",
        uptime_seconds=round(uptime, 1),
        service_name=settings.service_name,
    )


@app.post("/v1/predict", response_model=PredictionResponse)
async def predict_single(payload: TelemetryPayload, background_tasks: BackgroundTasks):
    """Synchronous single-event anomaly prediction."""
    t0 = time.perf_counter()
    event_dict = payload.model_dump()

    # 1. Data Validation Firewall Gate
    is_valid, quarantine = serving_ctx.validator.validate_event(event_dict)
    if not is_valid:
        VALIDATION_FAILURES.inc()
        REQUEST_COUNT.labels(endpoint="/v1/predict", status="422").inc()
        raise HTTPException(
            status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
            detail={"error": "Payload failed data quality gate", "quarantine": quarantine},
        )

    # 2. Stateful Feature Extraction
    features_record = serving_ctx.feature_engine.compute_features(event_dict)
    feature_vector = features_record["feature_vector"]

    # 3. Model Inference
    if serving_ctx.model is None or not serving_ctx.model.is_fitted:
        REQUEST_COUNT.labels(endpoint="/v1/predict", status="503").inc()
        raise HTTPException(status_code=503, detail="Model is not ready for inference")

    result = serving_ctx.model.predict_single(feature_vector)
    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000.0

    # Record Prometheus metrics
    PREDICTIONS_TOTAL.inc()
    INFERENCE_LATENCY.observe(t1 - t0)
    REQUEST_COUNT.labels(endpoint="/v1/predict", status="200").inc()

    if result["is_anomaly"]:
        ANOMALIES_DETECTED.labels(device_id=payload.device_id).inc()

    response = PredictionResponse(
        device_id=payload.device_id,
        timestamp=payload.timestamp,
        is_anomaly=result["is_anomaly"],
        anomaly_score=result["anomaly_score"],
        model_version=serving_ctx.model_version,
        model_type=result["model_type"],
        contributing_features=result["contributing_features"],
        inference_latency_ms=round(latency_ms, 2),
    )

    # Trigger async alert if anomalous
    if response.is_anomaly:
        background_tasks.add_task(dispatch_anomaly_alert, response)

    return response


@app.post("/v1/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(batch: BatchPredictionRequest, background_tasks: BackgroundTasks):
    """High-throughput micro-batch inference."""
    predictions = []
    anomalies_count = 0

    for item in batch.events:
        try:
            pred = await predict_single(item, background_tasks)
            predictions.append(pred)
            if pred.is_anomaly:
                anomalies_count += 1
        except HTTPException:
            continue

    return BatchPredictionResponse(
        predictions=predictions,
        processed_count=len(predictions),
        anomalies_count=anomalies_count,
    )


@app.post("/v1/model/reload")
async def reload_model():
    """Hot-reload champion model artifact with zero downtime."""
    success = serving_ctx.load_champion_model()
    if not success:
        raise HTTPException(status_code=500, detail="Failed to reload model from disk")
    serving_ctx.model_version = f"v{int(time.time())}"
    return {
        "status": "RELOADED",
        "new_version": serving_ctx.model_version,
        "reloaded_at": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/v1/model/reload")
async def get_reload_info():
    """Information on hot-reloading the champion model."""
    return {
        "status": "READY",
        "current_version": serving_ctx.model_version,
        "action": "Send HTTP POST to /v1/model/reload to hot-swap champion model artifact",
    }


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Silence browser favicon requests."""
    return Response(status_code=204)


@app.get("/metrics")
async def prometheus_metrics():
    """Expose Prometheus formatted metrics for scraping."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
