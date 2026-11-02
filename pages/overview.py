"""
Overview Page — Project Mission, Key Metrics, and Research Context.
"""

import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from src.config import load_config


def render_overview() -> None:
    cfg = load_config()

    # ── Hero Header ──────────────────────────────────────────────────
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #0d1b2a 0%, #1b2838 40%, #0d1b2a 100%);
            border: 1px solid rgba(0, 229, 255, 0.2);
            border-radius: 16px;
            padding: 40px 48px;
            margin-bottom: 32px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.05);
        ">
            <h1 style="
                font-size: 2.2rem;
                margin: 0 0 8px 0;
                background: linear-gradient(135deg, #00e5ff, #b388ff);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-weight: 800;
                letter-spacing: -0.02em;
            ">🌑 Micro-Quantized CNN for Onboard Crater Detection</h1>
            <p style="
                font-size: 1.1rem;
                color: #8892a4;
                margin: 0 0 20px 0;
                font-weight: 400;
            ">Chandrayaan-Inspired Lunar Research — FP32 → INT8 Post-Training Quantization Study</p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap;">
                <span style="background: rgba(0, 229, 255, 0.12); color: #00e5ff; padding: 4px 14px;
                       border-radius: 20px; font-size: 0.8rem; font-weight: 600; border: 1px solid rgba(0, 229, 255, 0.3);">
                    SSDLite320 + MobileNetV3-Large
                </span>
                <span style="background: rgba(179, 136, 255, 0.12); color: #b388ff; padding: 4px 14px;
                       border-radius: 20px; font-size: 0.8rem; font-weight: 600; border: 1px solid rgba(179, 136, 255, 0.3);">
                    ONNX Runtime Dynamic PTQ
                </span>
                <span style="background: rgba(0, 230, 118, 0.12); color: #00e676; padding: 4px 14px;
                       border-radius: 20px; font-size: 0.8rem; font-weight: 600; border: 1px solid rgba(0, 230, 118, 0.3);">
                    LRO NAC Dataset
                </span>
                <span style="background: rgba(255, 183, 0, 0.12); color: #ffb700; padding: 4px 14px;
                       border-radius: 20px; font-size: 0.8rem; font-weight: 600; border: 1px solid rgba(255, 183, 0, 0.3);">
                    INT8 • 71% Storage Reduction
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Key Metrics Row ──────────────────────────────────────────────
    # Load quantization report if available
    int8_meta_path = cfg.paths.quantized_dir / "crater_detector_int8.json"
    quant_report = {}
    if int8_meta_path.is_file():
        with open(int8_meta_path, "r") as f:
            quant_report = json.load(f)

    # Load benchmark report if available
    benchmark_files = sorted(cfg.paths.benchmarks_dir.glob("benchmark_comparison_*.json"))
    bench_report = {}
    if benchmark_files:
        with open(benchmark_files[-1], "r") as f:
            bench_report = json.load(f)

    col1, col2, col3, col4 = st.columns(4)

    fp32_size = quant_report.get("fp32_model_size_mb", 14.29)
    int8_size = quant_report.get("int8_model_size_mb", 4.10)
    reduction = quant_report.get("percentage_size_reduction", 71.3)

    with col1:
        st.metric("FP32 Model Size", f"{fp32_size:.2f} MB", delta=None)
    with col2:
        st.metric("INT8 Model Size", f"{int8_size:.2f} MB", delta=f"-{reduction:.1f}%")
    with col3:
        fp32_lat = bench_report.get("fp32_metrics", {}).get("median_latency_ms", 0)
        int8_lat = bench_report.get("int8_metrics", {}).get("median_latency_ms", 0)
        st.metric("FP32 Median Latency", f"{fp32_lat:.1f} ms")
    with col4:
        st.metric("INT8 Median Latency", f"{int8_lat:.1f} ms")

    st.markdown("---")

    # ── Research Context ─────────────────────────────────────────────
    left, right = st.columns([3, 2])

    with left:
        st.markdown("### 📋 Research Abstract")
        st.markdown(
            """
            This project investigates whether a **lightweight object-detection CNN**
            (SSDLite320 with MobileNetV3-Large backbone) can be post-training quantized
            from **FP32 to INT8** to reduce model storage and inference footprint while
            retaining useful crater-detection performance — suitable for
            **onboard deployment on radiation-tolerant embedded processors** in
            Chandrayaan-class lunar orbiters.

            **Key Research Questions:**
            1. How much storage reduction does INT8 dynamic quantization achieve on an SSD-family detector?
            2. Does quantization preserve detection accuracy within acceptable limits?
            3. What precision mode results from dynamic PTQ on coordinate-sensitive detection heads?

            **Dataset:** LRO Narrow Angle Camera (NAC) imagery from the Von Kármán crater region
            (Chang'e-4 landing site), with expert-annotated crater catalogs.
            """,
        )

    with right:
        st.markdown("### 🎯 Quantization Pipeline")
        st.markdown(
            """
            ```
            ┌─────────────────────┐
            │  PyTorch SSDLite320 │
            │  + MobileNetV3-Large│
            │  (FP32 Weights)     │
            └────────┬────────────┘
                     │ TorchScript Trace
                     ▼
            ┌─────────────────────┐
            │   ONNX FP32 Model   │
            │   (~14.29 MB)       │
            └────────┬────────────┘
                     │ ONNX Runtime
                     │ Dynamic PTQ
                     ▼
            ┌─────────────────────┐
            │   ONNX INT8 Model   │
            │   (~4.10 MB)        │
            │   -71.3% Storage    │
            └─────────────────────┘
            ```
            """,
        )

    st.markdown("---")

    # ── Size Comparison Gauge ────────────────────────────────────────
    st.markdown("### 📦 Model Size Comparison")

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=[fp32_size],
        y=["FP32 Baseline"],
        orientation="h",
        marker=dict(
            color="rgba(0, 180, 216, 0.8)",
            line=dict(color="rgba(0, 229, 255, 0.9)", width=1),
        ),
        text=[f"{fp32_size:.2f} MB"],
        textposition="inside",
        textfont=dict(size=14, family="JetBrains Mono", color="#ffffff"),
        name="FP32",
    ))

    fig.add_trace(go.Bar(
        x=[int8_size],
        y=["INT8 Quantized"],
        orientation="h",
        marker=dict(
            color="rgba(0, 230, 118, 0.8)",
            line=dict(color="rgba(0, 230, 118, 0.9)", width=1),
        ),
        text=[f"{int8_size:.2f} MB (-{reduction:.1f}%)"],
        textposition="inside",
        textfont=dict(size=14, family="JetBrains Mono", color="#ffffff"),
        name="INT8",
    ))

    fig.update_layout(
        height=160,
        margin=dict(l=0, r=40, t=10, b=10),
        plot_bgcolor="rgba(26, 32, 53, 0.6)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        font=dict(color="#e0e6ed", family="Inter"),
        xaxis=dict(
            title="Model Size (MB)",
            gridcolor="rgba(42, 58, 92, 0.5)",
            zerolinecolor="rgba(42, 58, 92, 0.5)",
        ),
        yaxis=dict(gridcolor="rgba(42, 58, 92, 0.5)"),
        showlegend=False,
        barmode="group",
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── Dataset Provenance ───────────────────────────────────────────
    st.markdown("### 🛰️ Dataset & Provenance")
    prov_cols = st.columns(3)

    with prov_cols[0]:
        st.markdown(
            """
            **Instrument**
            - Lunar Reconnaissance Orbiter (LRO)
            - Narrow Angle Camera (NAC)
            - ~0.5 m/pixel resolution
            """
        )

    with prov_cols[1]:
        st.markdown(
            """
            **Target Region**
            - Von Kármán Crater
            - Chang'e-4 Landing Zone
            - 45°-46°S, 176.4°-178.8°E
            """
        )

    with prov_cols[2]:
        # Load split manifest if available
        split_path = cfg.paths.splits_dir / "split_manifest.json"
        if split_path.is_file():
            with open(split_path, "r") as f:
                split_data = json.load(f)
            train_n = len(split_data["splits"]["train"])
            val_n = len(split_data["splits"]["val"])
            test_n = len(split_data["splits"]["test"])
            total_craters = split_data.get("total_annotations", "~23k")
        else:
            train_n, val_n, test_n = "—", "—", "—"
            total_craters = "~23k"

        st.markdown(
            f"""
            **Split Statistics**
            - Train: {train_n} tiles
            - Validation: {val_n} tiles
            - Test: {test_n} tiles
            - Total craters: {total_craters}
            """
        )


if __name__ == "__main__":
    render_overview()
