"""
Tests for storage layer, SQLite default, optional PostgreSQL fallback, and experiment registry.
"""

from pathlib import Path
import pytest

from src.storage.database import DatabaseManager
from src.storage.experiments import ExperimentRegistry


def test_sqlite_database_and_experiment_registry(tmp_path: Path):
    """Test full cycle of experiment creation, insertion, and retrieval in SQLite."""
    db_file = tmp_path / "test_experiments.db"
    db_mgr = DatabaseManager(sqlite_path=db_file)
    assert db_file.is_file()
    assert db_mgr.is_postgres is False

    registry = ExperimentRegistry(db_mgr)
    exp_id = registry.record_experiment(
        model_architecture="ssdlite320_mobilenet_v3_large",
        model_format="INT8_ONNX",
        serialized_size_mb=4.08,
        ap_50=0.625,
        map_50_95=0.412,
        precision=0.78,
        recall=0.71,
        median_latency_ms=28.5,
        throughput_fps=35.1,
        quantization_method="PTQ Dynamic QInt8",
    )

    assert exp_id.startswith("EXP-")

    # Fetch
    records = registry.list_experiments()
    assert len(records) == 1
    rec = records[0]
    assert rec["experiment_id"] == exp_id
    assert rec["serialized_size_mb"] == 4.08
    assert rec["ap_50"] == 0.625

    # DataFrame conversion
    df = registry.get_experiments_dataframe()
    assert len(df) == 1
    assert "experiment_id" in df.columns


def test_postgres_fallback_on_unreachable_db(tmp_path: Path):
    """Test that an invalid or unreachable PostgreSQL DATABASE_URL gracefully falls back to local SQLite."""
    db_file = tmp_path / "fallback.db"
    # Provide dummy invalid postgres URL
    unreachable_url = "postgresql://invalid_user:invalid_pass@127.0.0.1:54339/nonexistent_db"

    db_mgr = DatabaseManager(sqlite_path=db_file, database_url=unreachable_url)
    assert db_mgr.is_postgres is False
    assert db_file.is_file()  # SQLite initialized as fallback!
