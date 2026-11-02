"""
Experiment Registry Page — View all logged experiments from SQLite/PostgreSQL.
"""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import load_config
from src.storage.database import DatabaseManager
from src.storage.experiments import ExperimentRegistry


def render_experiment_registry() -> None:
    cfg = load_config()

    st.markdown(
        """
        <h1 style="
            background: linear-gradient(135deg, #00e5ff, #b388ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
        ">🗄️ Experiment Registry</h1>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Complete scientific provenance — every training, quantization, and evaluation run")
    st.markdown("---")

    db = DatabaseManager(cfg.paths.experiments_db, cfg.database_url)
    registry = ExperimentRegistry(db)

    experiments = registry.list_experiments()

    if not experiments:
        st.info("No experiments recorded yet. Run the benchmark pipeline to populate the registry.")
        return

    # ── Summary Metrics ──────────────────────────────────────────────
    st.markdown("### 📊 Registry Summary")
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.metric("Total Experiments", len(experiments))
    with mc2:
        architectures = set(e.get("model_architecture", "") for e in experiments)
        st.metric("Architectures", len(architectures))
    with mc3:
        quant_methods = set(e.get("quantization_method", "") for e in experiments)
        st.metric("Quant. Methods", len(quant_methods))
    with mc4:
        platforms = set(e.get("hardware_platform", "")[:20] for e in experiments if e.get("hardware_platform"))
        st.metric("Platforms", len(platforms) if platforms else 1)

    st.markdown("---")

    # ── Experiments Table ────────────────────────────────────────────
    st.markdown("### 📋 All Experiments")

    df = pd.DataFrame(experiments)

    # Select display columns
    display_cols = [
        "experiment_id", "timestamp", "model_architecture", "model_format",
        "quantization_method", "serialized_size_mb", "ap_50", "map_50_95",
        "precision", "recall", "median_latency_ms", "p95_latency_ms", "throughput_fps",
    ]
    available_cols = [c for c in display_cols if c in df.columns]
    df_display = df[available_cols].copy()

    # Format numeric columns
    for col in ["serialized_size_mb", "ap_50", "map_50_95", "precision", "recall",
                "median_latency_ms", "p95_latency_ms", "throughput_fps"]:
        if col in df_display.columns:
            df_display[col] = pd.to_numeric(df_display[col], errors="coerce")

    st.dataframe(df_display, use_container_width=True, hide_index=True)

    # ── Detailed Experiment View ─────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔍 Experiment Detail View")

    exp_ids = [e["experiment_id"] for e in experiments]
    selected_exp = st.selectbox("Select Experiment", exp_ids)

    if selected_exp:
        exp = next(e for e in experiments if e["experiment_id"] == selected_exp)

        detail_cols = st.columns(3)

        with detail_cols[0]:
            st.markdown("**Identification**")
            st.markdown(f"- **ID:** `{exp['experiment_id']}`")
            st.markdown(f"- **Timestamp:** {exp['timestamp']}")
            st.markdown(f"- **Architecture:** {exp['model_architecture']}")
            st.markdown(f"- **Format:** {exp['model_format']}")

        with detail_cols[1]:
            st.markdown("**Performance Metrics**")
            st.markdown(f"- **mAP@0.50:** {exp.get('ap_50', 'N/A')}")
            st.markdown(f"- **mAP@0.50:0.95:** {exp.get('map_50_95', 'N/A')}")
            st.markdown(f"- **Precision:** {exp.get('precision', 'N/A')}")
            st.markdown(f"- **Recall:** {exp.get('recall', 'N/A')}")

        with detail_cols[2]:
            st.markdown("**Deployment Metrics**")
            st.markdown(f"- **Size:** {exp.get('serialized_size_mb', 'N/A')} MB")
            st.markdown(f"- **Median Latency:** {exp.get('median_latency_ms', 'N/A')} ms")
            st.markdown(f"- **P95 Latency:** {exp.get('p95_latency_ms', 'N/A')} ms")
            st.markdown(f"- **Throughput:** {exp.get('throughput_fps', 'N/A')} FPS")

        # Raw metrics JSON
        with st.expander("📦 Raw Metrics JSON"):
            metrics_str = exp.get("metrics_json", "{}")
            try:
                st.json(json.loads(metrics_str) if isinstance(metrics_str, str) else metrics_str)
            except (json.JSONDecodeError, TypeError):
                st.text(str(metrics_str))

        with st.expander("📦 Raw Benchmark JSON"):
            bench_str = exp.get("benchmark_json", "{}")
            try:
                st.json(json.loads(bench_str) if isinstance(bench_str, str) else bench_str)
            except (json.JSONDecodeError, TypeError):
                st.text(str(bench_str))

    # ── Storage Backend Info ─────────────────────────────────────────
    st.markdown("---")
    with st.expander("🗄️ Storage Backend Info"):
        st.markdown(f"**Backend:** {'PostgreSQL' if db.is_postgres else 'SQLite (Local)'}")
        st.markdown(f"**Database Path:** `{cfg.paths.experiments_db}`")
        st.markdown(f"**Total Records:** {len(experiments)}")


if __name__ == "__main__":
    render_experiment_registry()

