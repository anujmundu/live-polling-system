"""Central configuration management for the MLOps Anomaly Detection Platform.

Adheres to 12-factor principles using Pydantic Settings.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global configuration settings for the entire platform."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore")

    # Environment & Service
    environment: str = Field(
        default="development", description="Runtime environment: development | staging | production"
    )
    service_name: str = Field(default="mlops-anomaly-detection", description="Microservice identifier")
    debug: bool = Field(default=False, description="Debug mode flag")

    # Kafka Streaming Configuration
    kafka_bootstrap_servers: str = Field(default="localhost:9092", description="Kafka broker address list")
    kafka_topic_raw: str = Field(default="sensor.telemetry.raw", description="Raw sensor telemetry topic")
    kafka_topic_validated: str = Field(default="sensor.telemetry.validated", description="Validated telemetry topic")
    kafka_topic_quarantine: str = Field(default="sensor.telemetry.quarantine", description="Dead Letter Queue (DLQ)")
    kafka_consumer_group: str = Field(default="mlops-anomaly-consumers", description="Consumer group id")
    kafka_auto_offset_reset: str = Field(default="latest", description="Offset reset strategy")

    # Database & Storage
    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/mlops_anomaly_db",
        description="PostgreSQL connection string",
    )
    minio_endpoint: str = Field(default="localhost:9000", description="MinIO S3 endpoint")
    minio_access_key: str = Field(default="minioadmin", description="MinIO access key")
    minio_secret_key: str = Field(default="minioadmin", description="MinIO secret key")
    minio_bucket: str = Field(default="telemetry-lakehouse", description="S3 bucket for offline feature storage")

    # MLOps & Experiment Tracking (MLflow)
    mlflow_tracking_uri: str = Field(default="http://localhost:5000", description="MLflow tracking server URI")
    mlflow_experiment_name: str = Field(default="realtime-anomaly-detection", description="MLflow experiment name")
    mlflow_model_name: str = Field(default="IndustrialAnomalyDetector", description="Registered model name")

    # Model Quality Gates
    gate_min_precision: float = Field(default=0.90, description="Minimum acceptable precision")
    gate_min_recall: float = Field(default=0.88, description="Minimum acceptable recall")
    gate_max_fpr: float = Field(default=0.025, description="Maximum false positive rate (2.5%)")
    gate_max_p99_latency_ms: float = Field(default=15.0, description="Maximum allowed p99 latency in ms")
    gate_max_model_size_mb: float = Field(default=150.0, description="Maximum allowed model size in MB")
    gate_f1_improvement_threshold: float = Field(default=0.015, description="Required F1 gain to unseat champion")

    # Drift Detection Thresholds
    drift_psi_threshold_warning: float = Field(default=0.10, description="PSI warning threshold")
    drift_psi_threshold_critical: float = Field(default=0.25, description="PSI critical retraining threshold")
    drift_ks_p_value_threshold: float = Field(default=0.01, description="KS test p-value threshold")
    drift_window_sample_size: int = Field(default=1000, description="Inference window size for drift calculation")

    # Inference API
    api_host: str = Field(default="0.0.0.0", description="FastAPI host")
    api_port: int = Field(default=8000, description="FastAPI port")
    api_workers: int = Field(default=4, description="Uvicorn workers")


# Global singleton instance
settings = Settings()
