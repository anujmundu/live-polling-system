"""Data Validation Firewall and Quarantine Dead Letter Queue (DLQ) Handler.

Enforces schema contracts, non-null assertions, and physical operating ranges
on streaming sensor telemetry before feature extraction or inference.
Invalid events are routed to 'sensor.telemetry.quarantine'.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ValidationFirewall")


# Physical Operating Bounds Specification
PHYSICAL_LIMITS = {
    "temperature": {"min": -20.0, "max": 150.0, "unit": "°C"},
    "pressure": {"min": 0.0, "max": 300.0, "unit": "psi"},
    "vibration": {"min": 0.0, "max": 50.0, "unit": "mm/s"},
    "rpm": {"min": 0.0, "max": 10000.0, "unit": "RPM"},
    "voltage": {"min": 180.0, "max": 260.0, "unit": "V"},
}

REQUIRED_COLUMNS = ["device_id", "timestamp", "temperature", "pressure", "vibration", "rpm", "voltage"]


class ValidationFirewall:
    """Evaluates telemetry events against strict industrial data quality gates."""

    def __init__(self, dlq_topic: Optional[str] = None):
        self.dlq_topic = dlq_topic or settings.kafka_topic_quarantine
        self.validation_count = 0
        self.passed_count = 0
        self.quarantine_count = 0

    def validate_event(self, event: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Validate a single telemetry event.

        Returns:
            Tuple of (is_valid: bool, quarantine_envelope: Optional[Dict])
        """
        self.validation_count += 1
        failed_expectations: List[Dict[str, Any]] = []

        # 1. Schema & Required Columns Check
        for col in REQUIRED_COLUMNS:
            if col not in event:
                failed_expectations.append(
                    {
                        "expectation_type": "expect_table_columns_to_match_ordered_list",
                        "column": col,
                        "observed_value": None,
                        "rule": "Column must exist",
                        "error_message": f"Missing required column: '{col}'",
                    }
                )

        # If columns missing, fail immediately
        if failed_expectations:
            envelope = self._build_quarantine_envelope(event, failed_expectations)
            self.quarantine_count += 1
            return False, envelope

        # 2. Non-Null Checks
        for col in REQUIRED_COLUMNS:
            val = event.get(col)
            if val is None:
                failed_expectations.append(
                    {
                        "expectation_type": "expect_column_values_to_not_be_null",
                        "column": col,
                        "observed_value": None,
                        "rule": "nullable = false",
                        "error_message": f"Null value encountered in critical channel '{col}'",
                    }
                )

        # If any nulls, cannot safely perform numerical range checks
        if failed_expectations:
            envelope = self._build_quarantine_envelope(event, failed_expectations)
            self.quarantine_count += 1
            return False, envelope

        # 3. Timestamp Freshness and ISO-8601 Validity
        raw_ts = str(event["timestamp"])
        try:
            # Parse ISO timestamp
            parsed_ts = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
            # Ensure timezone-aware
            if parsed_ts.tzinfo is None:
                parsed_ts = parsed_ts.replace(tzinfo=timezone.utc)
        except Exception as exc:
            failed_expectations.append(
                {
                    "expectation_type": "expect_column_values_to_be_dateutil_parseable",
                    "column": "timestamp",
                    "observed_value": raw_ts,
                    "rule": "valid ISO-8601 UTC datetime",
                    "error_message": f"Malformed timestamp '{raw_ts}': {exc}",
                }
            )

        # 4. Sensor Value Physics Limits
        for channel, limits in PHYSICAL_LIMITS.items():
            val = event.get(channel)
            try:
                num_val = float(val)
                min_val, max_val = limits["min"], limits["max"]
                if not (min_val <= num_val <= max_val):
                    failed_expectations.append(
                        {
                            "expectation_type": "expect_column_values_to_be_between",
                            "column": channel,
                            "observed_value": num_val,
                            "rule": f"min: {min_val}, max: {max_val} {limits['unit']}",
                            "error_message": (
                                f"Value {num_val} in '{channel}' exceeds physical boundary "
                                f"[{min_val}, {max_val}] {limits['unit']}"
                            ),
                        }
                    )
            except (ValueError, TypeError):
                failed_expectations.append(
                    {
                        "expectation_type": "expect_column_values_to_be_of_type",
                        "column": channel,
                        "observed_value": val,
                        "rule": "numeric float/int",
                        "error_message": f"Channel '{channel}' contains non-numeric value: {val}",
                    }
                )

        # Decision
        if failed_expectations:
            self.quarantine_count += 1
            envelope = self._build_quarantine_envelope(event, failed_expectations)
            logger.warning(
                f"Validation failed for device {event.get('device_id')}: "
                f"{len(failed_expectations)} violations. Quarantined."
            )
            return False, envelope

        self.passed_count += 1
        return True, None

    def _build_quarantine_envelope(
        self, event: Dict[str, Any], failed_expectations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Construct Dead Letter Queue forensic packaging."""
        return {
            "quarantine_id": f"urn:uuid:{uuid.uuid4()}",
            "quarantined_at": datetime.now(timezone.utc).isoformat(),
            "source_topic": settings.kafka_topic_raw,
            "failed_expectations": failed_expectations,
            "raw_payload": event,
        }

    def get_metrics(self) -> Dict[str, int]:
        """Return data quality metrics."""
        return {
            "total_validated": self.validation_count,
            "passed": self.passed_count,
            "quarantined": self.quarantine_count,
        }
