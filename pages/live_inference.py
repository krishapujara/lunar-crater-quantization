"""
Live Inference Page — Upload lunar surface images and run FP32 vs INT8 detection.
"""

import io
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from src.config import load_config


def render_live_inference() -> None:
    cfg = load_config()

    st.markdown(
        """
        <h1 style="
            background: linear-gradient(135deg, #00e5ff, #b388ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
        ">🔬 Live Crater Detection Inference</h1>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Upload a lunar surface image or select a test sample to compare FP32 vs INT8 detection side-by-side")
    st.markdown("---")

    fp32_onnx = cfg.paths.quantized_dir / "crater_detector_fp32.onnx"
    int8_onnx = cfg.paths.quantized_dir / "crater_detector_int8.onnx"

    if not fp32_onnx.is_file() or not int8_onnx.is_file():
        st.error("⚠️ Model artifacts not found. Please ensure crater_detector_fp32.onnx and crater_detector_int8.onnx exist in artifacts/quantized/.")
        return

    # Lazy load detectors
    @st.cache_resource
    def load_detectors():
        from src.models.inference import CraterDetector
        fp32 = CraterDetector(fp32_onnx)
        int8 = CraterDetector(int8_onnx)
        return fp32, int8

    fp32_det, int8_det = load_detectors()

    # ── Input Selection ──────────────────────────────────────────────
    input_method = st.radio(
        "Image Source",
        ["📁 Upload Image", "🗂️ Use Test Sample"],
        horizontal=True,
    )

    pil_image = None

    if input_method == "📁 Upload Image":
        uploaded = st.file_uploader(
            "Upload a lunar surface image (PNG, JPG, TIFF)",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
        )
        if uploaded:
            pil_image = Image.open(uploaded).convert("RGB")
    else:
        # Search for available lunar images
        sample_images = []
        if cfg.paths.samples_dir.is_dir():
            sample_images += sorted(list(cfg.paths.samples_dir.glob("*.png")) + list(cfg.paths.samples_dir.glob("*.jpg")))
        if not sample_images and cfg.paths.raw_data_dir.is_dir():
            sample_images += sorted(list(cfg.paths.raw_data_dir.rglob("*.png")) + list(cfg.paths.raw_data_dir.rglob("*.jpg")))[:15]
        if not sample_images and cfg.paths.data_dir.is_dir():
            sample_images += sorted(list(cfg.paths.data_dir.rglob("*.png")) + list(cfg.paths.data_dir.rglob("*.jpg")))[:10]

        if sample_images:
            selected_file = st.selectbox(
                "Select Test Lunar Tile",
                options=sample_images,
                format_func=lambda p: Path(p).name,
            )
            if selected_file:
                pil_image = Image.open(selected_file).convert("RGB")
                st.caption(f"Loaded: `{Path(selected_file).name}` ({pil_image.width} × {pil_image.height} px)")
        else:
            st.warning("No sample images found in data directory.")

    if pil_image is None:
        st.info("ℹ️ Please select or upload an image above to begin real-time inference.")
        return

    # ── Configuration Controls ───────────────────────────────────────
    with st.expander("⚙️ Detection Parameters", expanded=False):
        conf_threshold = st.slider("Confidence Threshold", min_value=0.05, max_value=0.95, value=0.25, step=0.05)
        show_labels = st.checkbox("Show Labels & Diameters", value=True)

    # ── Run Inference ────────────────────────────────────────────────
    st.markdown("### 🎯 Detection Results")

    col_fp32, col_int8 = st.columns(2)

    with col_fp32:
        st.markdown(
            '<div style="text-align:center; padding:8px; background:rgba(0,180,216,0.12); '
            'border-radius:8px; border:1px solid rgba(0,180,216,0.3); margin-bottom:12px;">'
            '<b style="color:#00b4d8;">FP32 ONNX Baseline</b></div>',
            unsafe_allow_html=True,
        )
        fp32_result = fp32_det.detect(pil_image, confidence_threshold=conf_threshold)
        fp32_viz = fp32_det.render_detections(pil_image, fp32_result, box_color="#00DDFF", show_labels=show_labels)
        st.image(fp32_viz, use_container_width=True)

        st.markdown(f"""
        | Metric | Value |
        |---|---|
        | **Detections** | **{fp32_result.num_detections} craters** |
        | **Pure Inference** | **{fp32_result.inference_time_ms:.1f} ms** |
        | **Preprocessing** | {fp32_result.preprocessing_time_ms:.1f} ms |
        | **Postprocessing** | {fp32_result.postprocessing_time_ms:.1f} ms |
        | **End-to-End Total** | {fp32_result.total_time_ms:.1f} ms |
        """)

    with col_int8:
        st.markdown(
            '<div style="text-align:center; padding:8px; background:rgba(0,230,118,0.12); '
            'border-radius:8px; border:1px solid rgba(0,230,118,0.3); margin-bottom:12px;">'
            '<b style="color:#00e676;">INT8 ONNX Quantized</b></div>',
            unsafe_allow_html=True,
        )
        int8_result = int8_det.detect(pil_image, confidence_threshold=conf_threshold)
        int8_viz = int8_det.render_detections(pil_image, int8_result, box_color="#00FF88", show_labels=show_labels)
        st.image(int8_viz, use_container_width=True)

        st.markdown(f"""
        | Metric | Value |
        |---|---|
        | **Detections** | **{int8_result.num_detections} craters** |
        | **Pure Inference** | **{int8_result.inference_time_ms:.1f} ms** |
        | **Preprocessing** | {int8_result.preprocessing_time_ms:.1f} ms |
        | **Postprocessing** | {int8_result.postprocessing_time_ms:.1f} ms |
        | **End-to-End Total** | {int8_result.total_time_ms:.1f} ms |
        """)

    # ── Detection Records Table ──────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 Structured Detection Records")

    det_tab1, det_tab2 = st.tabs(["FP32 Detections Table", "INT8 Detections Table"])

    with det_tab1:
        if fp32_result.num_detections > 0:
            import pandas as pd
            df_fp32 = pd.DataFrame(fp32_result.to_dict_list())
            st.dataframe(df_fp32, use_container_width=True, hide_index=True)
        else:
            st.info("No craters detected above the confidence threshold.")

    with det_tab2:
        if int8_result.num_detections > 0:
            import pandas as pd
            df_int8 = pd.DataFrame(int8_result.to_dict_list())
            st.dataframe(df_int8, use_container_width=True, hide_index=True)
        else:
            st.info("No craters detected above the confidence threshold.")

    # ── Timing Comparison Chart ──────────────────────────────────────
    st.markdown("### ⏱️ Latency Breakdown Comparison")

    timing_labels = ["Preprocessing", "Model Inference", "Postprocessing", "Total End-to-End"]
    fp32_times = [
        fp32_result.preprocessing_time_ms,
        fp32_result.inference_time_ms,
        fp32_result.postprocessing_time_ms,
        fp32_result.total_time_ms,
    ]
    int8_times = [
        int8_result.preprocessing_time_ms,
        int8_result.inference_time_ms,
        int8_result.postprocessing_time_ms,
        int8_result.total_time_ms,
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="FP32 Baseline",
        x=timing_labels,
        y=fp32_times,
        marker=dict(color="rgba(0, 180, 216, 0.8)"),
        text=[f"{t:.1f} ms" for t in fp32_times],
        textposition="outside",
        textfont=dict(family="JetBrains Mono"),
    ))
    fig.add_trace(go.Bar(
        name="INT8 Quantized",
        x=timing_labels,
        y=int8_times,
        marker=dict(color="rgba(0, 230, 118, 0.8)"),
        text=[f"{t:.1f} ms" for t in int8_times],
        textposition="outside",
        textfont=dict(family="JetBrains Mono"),
    ))

    fig.update_layout(
        barmode="group",
        height=350,
        margin=dict(l=40, r=40, t=30, b=40),
        plot_bgcolor="rgba(26, 32, 53, 0.6)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        font=dict(color="#e0e6ed", family="Inter"),
        yaxis_title="Time (ms)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(26, 32, 53, 0.8)",
        ),
    )
    fig.update_xaxes(gridcolor="rgba(42, 58, 92, 0.5)")
    fig.update_yaxes(gridcolor="rgba(42, 58, 92, 0.5)")

    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    render_live_inference()
