"""Unit tests for Kafka telemetry producer."""

from src.ingestion.kafka_producer import TelemetryKafkaProducer
from src.ingestion.simulator import TelemetrySimulator


def test_kafka_producer_mock_mode_publish():
    producer = TelemetryKafkaProducer(mock_mode=True)
    assert producer.mock_mode is True
    assert producer.producer is None

    received = []

    def callback(evt):
        received.append(evt)

    sim = TelemetrySimulator()
    event = sim.generate_event()

    success = producer.publish_event(event, callback=callback)
    assert success is True
    assert len(received) == 1
    assert received[0]["device_id"] == event["device_id"]


def test_kafka_producer_stream_telemetry():
    producer = TelemetryKafkaProducer(mock_mode=True)
    events = []

    producer.stream_telemetry(
        duration_seconds=0.2,
        events_per_second=20.0,
        anomaly_probability=0.1,
        callback=lambda e: events.append(e),
    )

    assert len(events) >= 1
    producer.close()


def test_kafka_producer_close_idempotent():
    producer = TelemetryKafkaProducer(mock_mode=True)
    # Closing mock producer should be clean and not raise
    producer.close()
    assert producer.mock_mode is True
