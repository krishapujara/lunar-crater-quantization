"""
Architecture Details Page — Model architecture, ONNX operator analysis, and quantization details.
"""

import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from src.config import load_config


def render_architecture() -> None:
    cfg = load_config()

    st.markdown(
        """
        <h1 style="
            background: linear-gradient(135deg, #00e5ff, #b388ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
        ">📐 Model Architecture & Quantization Details</h1>
        """,
        unsafe_allow_html=True,
    )
    st.caption("SSDLite320 architecture breakdown, ONNX operator analysis, and INT8 precision mapping")
    st.markdown("---")

    # ── Architecture Overview ────────────────────────────────────────
    st.markdown("### 🏗️ SSDLite320 + MobileNetV3-Large Architecture")

    arch_cols = st.columns([3, 2])

    with arch_cols[0]:
        st.markdown(
            """
            | Component | Specification |
            |-----------|---------------|
            | **Detection Head** | SSD (Single Shot Detector) with Lite depthwise convolutions |
            | **Backbone** | MobileNetV3-Large (ImageNet pre-trained) |
            | **Input Resolution** | 320 × 320 × 3 (RGB) |
            | **Anchor Generator** | Multi-scale default box priors |
            | **NMS** | Non-Maximum Suppression (IoU-based) |
            | **Output Classes** | 2 (Background, Crater) |
            | **Total Parameters** | ~3.71 Million |
            | **FP32 Size** | ~14.15 MB (PyTorch) / ~14.29 MB (ONNX) |
            | **INT8 Size** | ~4.10 MB (ONNX Runtime) |
            """
        )

    with arch_cols[1]:
        st.markdown(
            """
            **Design Rationale for Lunar Missions:**

            - **MobileNetV3** uses inverted residuals with
              squeeze-and-excitation — optimized for low-power
              compute (ARM Cortex, DSP architectures)
            - **SSD Lite** replaces standard convolutions with
              depthwise separable convolutions — ~10x fewer
              FLOPs per detection layer
            - **320×320 input** is a practical tile size for
              segmenting large LRO NAC images into
              inference-ready patches
            """
        )

    st.markdown("---")

    # ── Network Architecture Diagram ─────────────────────────────────
    st.markdown("### 🔗 Network Data Flow")

    st.markdown(
        """
        ```
        Input Image (320×320×3)
              │
              ▼
        ┌─────────────────────────────────────────────────┐
        │           MobileNetV3-Large Backbone            │
        │  ┌────────┐  ┌────────┐  ┌────────┐  ┌──────┐ │
        │  │ Block 1│→ │ Block 2│→ │Block 3 │→ │Blk N │ │
        │  │ InvRes │  │ InvRes │  │InvRes  │  │InvRes│ │
        │  │ +SE    │  │ +SE    │  │+SE     │  │+SE   │ │
        │  └────────┘  └────────┘  └────────┘  └──────┘ │
        │      ↓            ↓            ↓          ↓    │
        │  Feature Maps at Multiple Scales (C3, C4, C5+) │
        └────────────────────┬────────────────────────────┘
                             │
                             ▼
        ┌─────────────────────────────────────────────────┐
        │          SSD Lite Detection Head                 │
        │  ┌───────────────┐  ┌───────────────┐          │
        │  │Classification │  │  Box Regress  │          │
        │  │  DW-Sep Conv  │  │  DW-Sep Conv  │          │
        │  │  → Cls Logits │  │  → BBox Δ     │          │
        │  └───────────────┘  └───────────────┘          │
        └────────────────────┬────────────────────────────┘
                             │
                             ▼
        ┌─────────────────────────────────────────────────┐
        │         Post-Processing Pipeline                │
        │  Anchor Decoding → Score Thresholding → NMS     │
        │  → [boxes, scores, labels] per image            │
        └─────────────────────────────────────────────────┘
        ```
        """
    )

    st.markdown("---")

    # ── ONNX Operator Analysis ───────────────────────────────────────
    st.markdown("### 🔧 ONNX Operator Analysis")

    int8_meta_path = cfg.paths.quantized_dir / "crater_detector_int8.json"
    fp32_meta_path = cfg.paths.quantized_dir / "crater_detector_fp32.json"

    if int8_meta_path.is_file():
        with open(int8_meta_path, "r") as f:
            int8_report = json.load(f)

        # Show operator comparison
        fp32_ops = int8_report.get("initial_operators", {})
        int8_ops = int8_report.get("quantized_operators", {})

        if fp32_ops or int8_ops:
            op_tab1, op_tab2 = st.tabs(["FP32 Operators", "INT8 Operators"])

            with op_tab1:
                if fp32_ops:
                    # Sort by count descending
                    sorted_ops = sorted(fp32_ops.items(), key=lambda x: x[1], reverse=True)
                    op_names = [op for op, _ in sorted_ops]
                    op_counts = [cnt for _, cnt in sorted_ops]

                    fig = go.Figure(go.Bar(
                        x=op_counts,
                        y=op_names,
                        orientation="h",
                        marker=dict(
                            color="rgba(0, 180, 216, 0.7)",
                            line=dict(color="#00b4d8", width=1),
                        ),
                        text=op_counts,
                        textposition="outside",
                        textfont=dict(family="JetBrains Mono", size=11),
                    ))
                    fig.update_layout(
                        height=max(300, len(op_names) * 28),
                        margin=dict(l=120, r=60, t=20, b=20),
                        plot_bgcolor="rgba(26, 32, 53, 0.6)",
                        paper_bgcolor="rgba(0, 0, 0, 0)",
                        font=dict(color="#e0e6ed", family="Inter"),
                        xaxis=dict(title="Count", gridcolor="rgba(42, 58, 92, 0.5)"),
                        yaxis=dict(gridcolor="rgba(42, 58, 92, 0.5)", autorange="reversed"),
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with op_tab2:
                if int8_ops:
                    sorted_ops = sorted(int8_ops.items(), key=lambda x: x[1], reverse=True)
                    op_names = [op for op, _ in sorted_ops]
                    op_counts = [cnt for _, cnt in sorted_ops]

                    # Color-code quantized ops
                    colors = []
                    for op in op_names:
                        if "Quantize" in op or "Integer" in op or "QLinear" in op:
                            colors.append("rgba(0, 230, 118, 0.8)")
                        elif op in ["Conv", "MatMul", "Gemm"]:
                            colors.append("rgba(255, 183, 0, 0.8)")
                        else:
                            colors.append("rgba(0, 180, 216, 0.6)")

                    fig = go.Figure(go.Bar(
                        x=op_counts,
                        y=op_names,
                        orientation="h",
                        marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.2)", width=1)),
                        text=op_counts,
                        textposition="outside",
                        textfont=dict(family="JetBrains Mono", size=11),
                    ))
                    fig.update_layout(
                        height=max(300, len(op_names) * 28),
                        margin=dict(l=160, r=60, t=20, b=20),
                        plot_bgcolor="rgba(26, 32, 53, 0.6)",
                        paper_bgcolor="rgba(0, 0, 0, 0)",
                        font=dict(color="#e0e6ed", family="Inter"),
                        xaxis=dict(title="Count", gridcolor="rgba(42, 58, 92, 0.5)"),
                        yaxis=dict(gridcolor="rgba(42, 58, 92, 0.5)", autorange="reversed"),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown(
                        """
                        <div style="display: flex; gap: 16px; flex-wrap: wrap; margin-top: 8px;">
                            <span style="display:flex;align-items:center;gap:6px;">
                                <span style="width:14px;height:14px;background:rgba(0,230,118,0.8);border-radius:3px;display:inline-block;"></span>
                                <span style="font-size:0.85rem;color:#8892a4;">Quantization Ops (new in INT8)</span>
                            </span>
                            <span style="display:flex;align-items:center;gap:6px;">
                                <span style="width:14px;height:14px;background:rgba(255,183,0,0.8);border-radius:3px;display:inline-block;"></span>
                                <span style="font-size:0.85rem;color:#8892a4;">Compute-Heavy Ops (quantized weights)</span>
                            </span>
                            <span style="display:flex;align-items:center;gap:6px;">
                                <span style="width:14px;height:14px;background:rgba(0,180,216,0.6);border-radius:3px;display:inline-block;"></span>
                                <span style="font-size:0.85rem;color:#8892a4;">Unquantized Ops (preserved FP32/INT64)</span>
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
    else:
        st.warning("INT8 quantization metadata not found. Run quantization pipeline first.")

    st.markdown("---")

    # ── Quantization Details ─────────────────────────────────────────
    st.markdown("### 🧪 Quantization Methodology")

    quant_cols = st.columns(2)

    with quant_cols[0]:
        st.markdown(
            """
            **Post-Training Dynamic Quantization (PTQ)**

            | Parameter | Value |
            |-----------|-------|
            | **Framework** | ONNX Runtime |
            | **Method** | `quantize_dynamic()` |
            | **Weight Precision** | INT8 (QInt8) |
            | **Activation Precision** | FP32 (runtime quantize/dequantize) |
            | **Target Ops** | Conv, MatMul, Gemm |
            | **ONNX Opset** | 14 |
            """
        )

    with quant_cols[1]:
        st.markdown(
            """
            **Operator Precision Mapping**

            | Operator Category | Precision |
            |-------------------|-----------|
            | Convolution weights | **INT8** |
            | Linear/MatMul weights | **INT8** |
            | Activations | FP32 (dynamic) |
            | Batch Normalization | FP32 |
            | NMS/Box Clipping | FP32/INT64 |
            | Anchor Generation | FP32 |
            | Coordinate Transforms | FP32 |
            """
        )

    st.markdown("---")

    # ── Deployment Context ───────────────────────────────────────────
    st.markdown("### 🛰️ Target Deployment Platform")

    st.markdown(
        """
        | Specification | Value |
        |---------------|-------|
        | **Target Processor** | Radiation-Tolerant Embedded CPU (e.g., LEON4-FT, GR740) |
        | **Alternative** | ARM Cortex-R/M class DSP with INT8 instruction support |
        | **Memory Budget** | < 8 MB model storage (SRAM/Flash) |
        | **Inference Budget** | < 500 ms per 320×320 tile |
        | **Mission Context** | Chandrayaan-class lunar orbiter / lander |
        | **Autonomy Level** | Onboard preprocessing for downlink prioritization |
        """
    )

    st.info(
        """
        **Deployment Rationale:** In deep-space missions, downlink bandwidth to Earth is 
        extremely limited (~few kbps for Chandrayaan-2). Onboard crater detection enables 
        intelligent data triage — only tiles containing scientifically interesting craters 
        are prioritized for transmission, dramatically improving science return per byte.
        
        The INT8 quantized model at **4.10 MB** fits within typical radiation-hardened 
        flash/SRAM budgets, while maintaining the same detection semantics as the FP32 baseline.
        """
    )


if __name__ == "__main__":
    render_architecture()

