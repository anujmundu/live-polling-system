"""Data Contract & Validation tests enforcing Great Expectations quality rules."""

import pytest
from src.validation.validator import ValidationFirewall


@pytest.fixture
def validator():
    return ValidationFirewall()


@pytest.fixture
def valid_event():
    return {
        "device_id": "MACHINE_042",
        "timestamp": "2026-09-26T14:20:00Z",
        "temperature": 87.4,
        "pressure": 104.2,
        "vibration": 8.91,
        "rpm": 3120.0,
        "voltage": 228.7,
    }


def test_validation_passes_valid_payload(validator, valid_event):
    is_valid, quarantine = validator.validate_event(valid_event)
    assert is_valid is True
    assert quarantine is None


def test_validation_catches_missing_required_column(validator, valid_event):
    corrupt = dict(valid_event)
    del corrupt["temperature"]
    is_valid, quarantine = validator.validate_event(corrupt)

    assert is_valid is False
    assert quarantine is not None
    assert "failed_expectations" in quarantine
    assert any(f["column"] == "temperature" for f in quarantine["failed_expectations"])


def test_validation_catches_null_values(validator, valid_event):
    corrupt = dict(valid_event)
    corrupt["pressure"] = None
    is_valid, quarantine = validator.validate_event(corrupt)

    assert is_valid is False
    assert quarantine is not None
    assert any("Null value" in f["error_message"] for f in quarantine["failed_expectations"])


def test_validation_catches_out_of_bounds_temperature(validator, valid_event):
    corrupt = dict(valid_event)
    corrupt["temperature"] = 285.0  # Physical maximum is 150 C
    is_valid, quarantine = validator.validate_event(corrupt)

    assert is_valid is False
    assert quarantine is not None
    assert any("exceeds physical boundary" in f["error_message"] for f in quarantine["failed_expectations"])


def test_validation_catches_negative_pressure(validator, valid_event):
    corrupt = dict(valid_event)
    corrupt["pressure"] = -25.0  # Physical minimum is 0 psi
    is_valid, quarantine = validator.validate_event(corrupt)

    assert is_valid is False
    assert quarantine is not None
