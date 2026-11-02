"""
Experiment tracker and registry.
Records complete scientific provenance for training, quantization, and evaluation runs.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from .database import DatabaseManager


class ExperimentRegistry:
    """Manages logging and querying research experiments."""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def record_experiment(
        self,
        model_architecture: str,
        model_format: str,
        serialized_size_mb: float,
        ap_50: Optional[float] = None,
        map_50_95: Optional[float] = None,
        precision: Optional[float] = None,
        recall: Optional[float] = None,
        median_latency_ms: Optional[float] = None,
        p95_latency_ms: Optional[float] = None,
        throughput_fps: Optional[float] = None,
        quantization_method: Optional[str] = None,
        dataset_manifest_hash: Optional[str] = None,
        checkpoint_sha256: Optional[str] = None,
        hardware_platform: Optional[str] = None,
        raw_metrics: Optional[Dict[str, Any]] = None,
        raw_benchmark: Optional[Dict[str, Any]] = None,
        experiment_id: Optional[str] = None,
    ) -> str:
        """Create and store a research experiment record."""
        exp_id = experiment_id or f"EXP-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        timestamp = datetime.now(timezone.utc).isoformat()

        record = {
            "experiment_id": exp_id,
            "timestamp": timestamp,
            "model_architecture": model_architecture,
            "dataset_manifest_hash": dataset_manifest_hash or "",
            "checkpoint_sha256": checkpoint_sha256 or "",
            "quantization_method": quantization_method or "None (FP32)",
            "model_format": model_format,
            "serialized_size_mb": round(float(serialized_size_mb), 2),
            "ap_50": round(float(ap_50), 4) if ap_50 is not None else None,
            "map_50_95": round(float(map_50_95), 4) if map_50_95 is not None else None,
            "precision": round(float(precision), 4) if precision is not None else None,
            "recall": round(float(recall), 4) if recall is not None else None,
            "median_latency_ms": round(float(median_latency_ms), 2) if median_latency_ms is not None else None,
            "p95_latency_ms": round(float(p95_latency_ms), 2) if p95_latency_ms is not None else None,
            "throughput_fps": round(float(throughput_fps), 1) if throughput_fps is not None else None,
            "hardware_platform": hardware_platform or "",
            "metrics_json": json.dumps(raw_metrics or {}),
            "benchmark_json": json.dumps(raw_benchmark or {}),
        }

        self.db.insert_experiment(record)
        return exp_id

    def list_experiments(self) -> List[Dict[str, Any]]:
        """Return all logged experiments."""
        return self.db.fetch_all_experiments()

    def get_experiments_dataframe(self) -> pd.DataFrame:
        """Return experiments as a Pandas DataFrame for Streamlit display."""
        records = self.list_experiments()
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        return df
