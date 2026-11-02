"""
Storage and database abstraction layer.
Provides lightweight SQLite storage by default, with optional PostgreSQL connection support.
"""

import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..utils.logging import get_logger

logger = get_logger("storage_database")


class DatabaseManager:
    """
    Unified database manager.
    Defaults to local SQLite database in artifacts directory.
    If DATABASE_URL environment variable is provided, attempts PostgreSQL connection via SQLAlchemy.
    """

    def __init__(self, sqlite_path: Path, database_url: Optional[str] = None):
        self.sqlite_path = Path(sqlite_path)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.database_url = database_url or os.environ.get("DATABASE_URL")
        self.is_postgres = False
        self._init_storage()

    def _init_storage(self) -> None:
        """Initialize tables in SQLite or PostgreSQL."""
        if self.database_url and self.database_url.strip().startswith("postgresql"):
            try:
                import sqlalchemy
                from sqlalchemy import create_engine

                self.engine = create_engine(self.database_url, pool_pre_ping=True)
                # Verify connection
                with self.engine.connect() as conn:
                    conn.execute(sqlalchemy.text("SELECT 1"))
                self.is_postgres = True
                logger.info("Connected successfully to PostgreSQL database.")
                self._create_pg_tables()
                return
            except Exception as e:
                # Mask credentials in logs
                logger.warning(f"Could not connect to configured PostgreSQL database: {type(e).__name__}. Falling back to SQLite.")
                self.is_postgres = False

        self._create_sqlite_tables()
        logger.info(f"Using local SQLite database at: {self.sqlite_path}")

    def _create_sqlite_tables(self) -> None:
        """Create experiments schema in SQLite."""
        with sqlite3.connect(self.sqlite_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    model_architecture TEXT NOT NULL,
                    dataset_manifest_hash TEXT,
                    checkpoint_sha256 TEXT,
                    quantization_method TEXT,
                    model_format TEXT,
                    serialized_size_mb REAL,
                    ap_50 REAL,
                    map_50_95 REAL,
                    precision REAL,
                    recall REAL,
                    median_latency_ms REAL,
                    p95_latency_ms REAL,
                    throughput_fps REAL,
                    hardware_platform TEXT,
                    metrics_json TEXT,
                    benchmark_json TEXT
                )
            """)
            conn.commit()

    def _create_pg_tables(self) -> None:
        """Create experiments schema in PostgreSQL."""
        import sqlalchemy
        with self.engine.connect() as conn:
            conn.execute(sqlalchemy.text("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id VARCHAR(64) PRIMARY KEY,
                    timestamp VARCHAR(32) NOT NULL,
                    model_architecture VARCHAR(64) NOT NULL,
                    dataset_manifest_hash VARCHAR(64),
                    checkpoint_sha256 VARCHAR(64),
                    quantization_method VARCHAR(64),
                    model_format VARCHAR(32),
                    serialized_size_mb REAL,
                    ap_50 REAL,
                    map_50_95 REAL,
                    precision REAL,
                    recall REAL,
                    median_latency_ms REAL,
                    p95_latency_ms REAL,
                    throughput_fps REAL,
                    hardware_platform TEXT,
                    metrics_json TEXT,
                    benchmark_json TEXT
                )
            """))
            conn.commit()

    def insert_experiment(self, record: Dict[str, Any]) -> None:
        """Insert experiment record into active database backend."""
        fields = [
            "experiment_id", "timestamp", "model_architecture", "dataset_manifest_hash",
            "checkpoint_sha256", "quantization_method", "model_format", "serialized_size_mb",
            "ap_50", "map_50_95", "precision", "recall", "median_latency_ms",
            "p95_latency_ms", "throughput_fps", "hardware_platform", "metrics_json", "benchmark_json"
        ]
        values = [record.get(f) for f in fields]

        if self.is_postgres:
            import sqlalchemy
            placeholders = ", ".join([f":{f}" for f in fields])
            sql = f"INSERT INTO experiments ({', '.join(fields)}) VALUES ({placeholders}) ON CONFLICT (experiment_id) DO UPDATE SET timestamp = EXCLUDED.timestamp"
            with self.engine.connect() as conn:
                conn.execute(sqlalchemy.text(sql), {f: record.get(f) for f in fields})
                conn.commit()
        else:
            placeholders = ", ".join(["?"] * len(fields))
            sql = f"INSERT OR REPLACE INTO experiments ({', '.join(fields)}) VALUES ({placeholders})"
            with sqlite3.connect(self.sqlite_path) as conn:
                cursor = conn.cursor()
                cursor.execute(sql, values)
                conn.commit()

    def fetch_all_experiments(self) -> List[Dict[str, Any]]:
        """Fetch all experiment records ordered by timestamp descending."""
        if self.is_postgres:
            import sqlalchemy
            with self.engine.connect() as conn:
                res = conn.execute(sqlalchemy.text("SELECT * FROM experiments ORDER BY timestamp DESC"))
                cols = res.keys()
                return [dict(zip(cols, row)) for row in res.fetchall()]
        else:
            with sqlite3.connect(self.sqlite_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM experiments ORDER BY timestamp DESC")
                return [dict(row) for row in cursor.fetchall()]
