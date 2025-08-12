"""High-throughput Kafka Telemetry Producer.

Publishes simulated industrial sensor packets to topic 'sensor.telemetry.raw'.
Supports configurable throughput rates, graceful degradation, and offline local streaming.
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import settings
from src.ingestion.simulator import TelemetrySimulator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("KafkaProducer")


class TelemetryKafkaProducer:
    """Publishes streaming telemetry events to Apache Kafka."""

    def __init__(
        self,
        bootstrap_servers: Optional[str] = None,
        topic: Optional[str] = None,
        mock_mode: bool = False,
    ):
        self.bootstrap_servers = bootstrap_servers or settings.kafka_bootstrap_servers
        self.topic = topic or settings.kafka_topic_raw
        self.mock_mode = mock_mode
        self.producer = None

        if not self.mock_mode:
            try:
                from kafka import KafkaProducer

                self.producer = KafkaProducer(
                    bootstrap_servers=self.bootstrap_servers.split(","),
                    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                    key_serializer=lambda k: str(k).encode("utf-8"),
                    acks="all",
                    retries=3,
                    max_in_flight_requests_per_connection=1,
                )
                logger.info(f"Connected to Kafka broker at {self.bootstrap_servers}, publishing to '{self.topic}'")
            except Exception as exc:
                logger.warning(
                    f"Failed to connect to Kafka at {self.bootstrap_servers} ({exc}). "
                    "Falling back to Mock / Local Dispatch Mode."
                )
                self.mock_mode = True

    def publish_event(self, event: Dict[str, Any], callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> bool:
        """Publish a single telemetry event to Kafka or mock callback."""
        device_id = event.get("device_id", "UNKNOWN")

        if self.mock_mode or self.producer is None:
            if callback:
                callback(event)
            return True

        try:
            self.producer.send(self.topic, key=device_id, value=event)
            self.producer.flush(timeout=5)
            return True
        except Exception as exc:
            logger.error(f"Failed to publish event for device {device_id}: {exc}")
            return False

    def stream_telemetry(
        self,
        duration_seconds: Optional[int] = None,
        events_per_second: float = 1.0,
        anomaly_probability: float = 0.05,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        """Continuously stream simulated telemetry events."""
        simulator = TelemetrySimulator(anomaly_probability=anomaly_probability)
        start_time = time.time()
        delay = 1.0 / max(0.1, events_per_second)
        events_sent = 0

        logger.info(f"Starting telemetry stream: rate={events_per_second} eps, anomaly_prob={anomaly_probability}")

        try:
            while True:
                if duration_seconds and (time.time() - start_time) >= duration_seconds:
                    logger.info(f"Completed streaming for {duration_seconds} seconds. Total events: {events_sent}")
                    break

                event = simulator.generate_event()
                self.publish_event(event, callback=callback)
                events_sent += 1

                if events_sent % 100 == 0:
                    logger.info(f"Stream progress: sent {events_sent} telemetry events...")

                time.sleep(delay)
        except KeyboardInterrupt:
            logger.info("Streaming interrupted by user.")
        finally:
            self.close()

    def close(self) -> None:
        """Flush and close producer connection."""
        if self.producer:
            try:
                self.producer.flush()
                self.producer.close()
                logger.info("Kafka producer successfully closed.")
            except Exception as exc:
                logger.warning(f"Error closing Kafka producer: {exc}")


if __name__ == "__main__":
    import argparse
    import httpx

    parser = argparse.ArgumentParser(description="Industrial Telemetry Stream Simulator & Producer")
    parser.add_argument("--duration", type=int, default=15, help="Streaming duration in seconds (0 for infinite)")
    parser.add_argument("--rate", type=float, default=1.0, help="Events per second")
    parser.add_argument("--anomaly-prob", type=float, default=0.15, help="Anomaly injection probability (0.0 to 1.0)")
    parser.add_argument("--live", action="store_true", help="Connect to live Kafka broker instead of local mock")
    parser.add_argument(
        "--api-url",
        type=str,
        default=None,
        help="Optional inference API URL to forward events for real-time scoring (e.g. http://127.0.0.1:8000/v1/predict)",
    )
    args = parser.parse_args()

    producer = TelemetryKafkaProducer(mock_mode=not args.live)
    duration = None if args.duration <= 0 else args.duration

    http_client = httpx.Client(timeout=3.0) if args.api_url else None

    def fmt_val(v: Any, width: int = 6) -> str:
        if v is None:
            return f"{'NULL':>{width}}"
        if isinstance(v, (int, float)):
            return f"{v:>{width}.2f}" if isinstance(v, float) else f"{v:>{width}}"
        return f"{str(v):>{width}}"

    def print_event(e: Dict[str, Any]) -> None:
        anom = e.get("_anomaly_type", "NONE")
        tag = "[ANOMALY]" if anom != "NONE" else "[NORMAL] "
        temp = fmt_val(e.get("temperature"), 6)
        press = fmt_val(e.get("pressure"), 6)
        vibr = fmt_val(e.get("vibration"), 5)
        rpm = fmt_val(e.get("rpm"), 6)

        extra_info = ""
        if http_client and args.api_url:
            payload = {k: v for k, v in e.items() if not k.startswith("_")}
            try:
                resp = http_client.post(args.api_url, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    is_anom = data.get("is_anomaly")
                    score = data.get("anomaly_score", 0.0)
                    lat = data.get("inference_latency_ms", 0.0)
                    model_tag = "ANOMALY" if is_anom else "NORMAL"
                    extra_info = f" -> Model: [{model_tag} score={score:.3f} | {lat:.1f}ms]"
                elif resp.status_code == 422:
                    extra_info = " -> Gate: [QUARANTINED (DLQ 422)]"
                else:
                    extra_info = f" -> API HTTP {resp.status_code}"
            except Exception as exc:
                extra_info = f" -> API Error ({type(exc).__name__})"

        print(
            f"{tag} [{e.get('device_id')}] Status={anom:<14} | Temp={temp}C | Press={press}psi | Vibr={vibr}mm/s | RPM={rpm}{extra_info}"
        )

    try:
        producer.stream_telemetry(
            duration_seconds=duration,
            events_per_second=args.rate,
            anomaly_probability=args.anomaly_prob,
            callback=print_event,
        )
    finally:
        if http_client:
            http_client.close()
