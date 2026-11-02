"""
Tests for configuration loading, dataclass defaults, and environment variable overrides.
"""

import os
from pathlib import Path
import pytest
from src.config import AppConfig, load_config


def test_default_config_loading(tmp_path: Path):
    """Test loading configuration with default values."""
    cfg = load_config()
    assert isinstance(cfg, AppConfig)
    assert cfg.project.name == "lunar-crater-quantization"
    assert cfg.model.architecture == "ssdlite320_mobilenet_v3_large"
    assert cfg.training.epochs > 0
    assert cfg.training.batch_size > 0
    assert cfg.paths.data_dir.is_dir() or str(cfg.paths.data_dir).endswith("data")


def test_environment_variable_overrides(monkeypatch):
    """Test environment variable overrides for device, batch size, and epochs."""
    monkeypatch.setenv("DEFAULT_DEVICE", "cpu")
    monkeypatch.setenv("DEFAULT_BATCH_SIZE", "8")
    monkeypatch.setenv("DEFAULT_EPOCHS", "12")
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/testdb")

    cfg = load_config()
    assert cfg.training.device == "cpu"
    assert cfg.training.batch_size == 8
    assert cfg.training.epochs == 12
    assert cfg.database_url == "postgresql://user:pass@localhost:5432/testdb"


def test_missing_optional_env_vars(monkeypatch):
    """Test clean fallback when optional environment variables are absent."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("DEFAULT_DEVICE", raising=False)

    cfg = load_config()
    assert cfg.database_url is None
    assert cfg.training.device in ["cpu", "cuda"]
