"""Test module verifying all core dependencies and versions."""

import pydantic
import fastapi
import sklearn
import mlflow
import great_expectations
import kafka
import sqlalchemy


def test_dependency_imports():
    """Verify that all production libraries import cleanly without errors."""
    assert pydantic.__version__ is not None
    assert fastapi.__version__ is not None
    assert sklearn.__version__ is not None
    assert mlflow.__version__ is not None
    assert great_expectations.__version__ is not None
    assert kafka.__version__ is not None
    assert sqlalchemy.__version__ is not None
    print(f"Verified Pydantic v{pydantic.__version__}")
    print(f"Verified FastAPI v{fastapi.__version__}")
    print(f"Verified Scikit-Learn v{sklearn.__version__}")
    print(f"Verified MLflow v{mlflow.__version__}")
    print(f"Verified Great Expectations v{great_expectations.__version__}")
    print(f"Verified Kafka-Python-NG v{kafka.__version__}")
    print(f"Verified SQLAlchemy v{sqlalchemy.__version__}")
    print("ALL CORE MLOPS DEPENDENCIES VERIFIED AND FUNCTIONAL!")


if __name__ == "__main__":
    test_dependency_imports()
