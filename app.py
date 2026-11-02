"""
Micro-Quantized CNN for Onboard Crater Detection — Chandrayaan-Inspired Lunar Research
Main Streamlit Research Application & Dashboard Entrypoint.
"""

import sys
from pathlib import Path

# Fix Streamlit module watcher compatibility with PyTorch torch.classes
try:
    import torch
    if hasattr(torch, "classes"):
        torch.classes.__path__ = []
except Exception:
    pass

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

# ── Page Config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Lunar Crater Detection — Micro-Quantized CNN Research",
    page_icon="🌑",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Custom CSS ───────────────────────────────────────────────────────
def inject_custom_css() -> None:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;700&display=swap');

    :root {
        --lunar-bg: #0a0e1a;
        --lunar-surface: #111827;
        --lunar-card: #1a2035;
        --lunar-border: #2a3a5c;
        --lunar-accent: #00e5ff;
        --lunar-accent-glow: rgba(0, 229, 255, 0.25);
        --lunar-gold: #ffb700;
        --lunar-green: #00e676;
        --lunar-red: #ff5252;
        --lunar-purple: #b388ff;
        --lunar-text: #e0e6ed;
        --lunar-muted: #94a3b8;
        --lunar-gradient: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 50%, #1b2838 100%);
    }

    html, body, [data-testid="stAppViewContainer"] {
        background: var(--lunar-gradient) !important;
        color: var(--lunar-text) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Top Navigation Bar / Streamlit Header */
    header[data-testid="stHeader"] {
        background: rgba(10, 14, 26, 0.92) !important;
        backdrop-filter: blur(12px) !important;
        border-bottom: 1px solid rgba(42, 58, 92, 0.4) !important;
    }

    header[data-testid="stHeader"] svg,
    header[data-testid="stHeader"] button {
        color: var(--lunar-accent) !important;
        fill: var(--lunar-accent) !important;
    }

    /* Sidebar Core Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c121e 0%, #111827 100%) !important;
        border-right: 1px solid var(--lunar-border) !important;
    }

    [data-testid="stSidebar"] * {
        color: var(--lunar-text);
    }

    /* Navigation Bar / Multi-Page Nav Links in Sidebar */
    [data-testid="stSidebarNav"] {
        background: transparent !important;
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
    }

    [data-testid="stSidebarNav"] ul {
        list-style: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    [data-testid="stSidebarNav"] li {
        margin-bottom: 4px !important;
    }

    /* Nav Link items */
    [data-testid="stSidebarNav"] a,
    [data-testid="stSidebarNav"] [data-testid="stSidebarNavLink"],
    [data-testid="stSidebarNav"] button {
        color: #e2e8f0 !important;
        background: rgba(26, 32, 53, 0.5) !important;
        border: 1px solid rgba(42, 58, 92, 0.4) !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
        font-size: 0.92rem !important;
        font-weight: 500 !important;
        text-decoration: none !important;
        display: flex !important;
        align-items: center !important;
        transition: all 0.25s ease !important;
    }

    [data-testid="stSidebarNav"] a span,
    [data-testid="stSidebarNav"] [data-testid="stSidebarNavLink"] span {
        color: #e2e8f0 !important;
        font-weight: 500 !important;
    }

    /* Hover State for Nav Links */
    [data-testid="stSidebarNav"] a:hover,
    [data-testid="stSidebarNav"] [data-testid="stSidebarNavLink"]:hover,
    [data-testid="stSidebarNav"] button:hover {
        background: rgba(0, 229, 255, 0.12) !important;
        border-color: var(--lunar-accent) !important;
        color: #00e5ff !important;
        transform: translateX(3px) !important;
        box-shadow: 0 0 12px rgba(0, 229, 255, 0.2) !important;
    }

    [data-testid="stSidebarNav"] a:hover span,
    [data-testid="stSidebarNav"] [data-testid="stSidebarNavLink"]:hover span {
        color: #00e5ff !important;
    }

    /* Active / Selected Page State */
    [data-testid="stSidebarNav"] a[aria-current="page"],
    [data-testid="stSidebarNav"] [data-testid="stSidebarNavLink"][aria-current="page"],
    [data-testid="stSidebarNav"] [aria-selected="true"] {
        background: linear-gradient(90deg, rgba(0, 229, 255, 0.2) 0%, rgba(26, 32, 53, 0.8) 100%) !important;
        border: 1px solid var(--lunar-accent) !important;
        border-left: 4px solid var(--lunar-accent) !important;
        color: #00e5ff !important;
        font-weight: 700 !important;
        box-shadow: 0 0 16px rgba(0, 229, 255, 0.3) !important;
    }

    [data-testid="stSidebarNav"] a[aria-current="page"] span,
    [data-testid="stSidebarNav"] [data-testid="stSidebarNavLink"][aria-current="page"] span {
        color: #00e5ff !important;
        font-weight: 700 !important;
    }

    /* Sidebar Headings */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: var(--lunar-accent) !important;
        font-weight: 700 !important;
    }

    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: var(--lunar-card) !important;
        border: 1px solid var(--lunar-border) !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.35) !important;
    }

    [data-testid="stMetricLabel"] {
        color: var(--lunar-muted) !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }

    [data-testid="stMetricValue"] {
        color: var(--lunar-accent) !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }

    [data-testid="stMetricDelta"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 600 !important;
    }

    /* Data frames & Tables */
    .stDataFrame, [data-testid="stTable"] {
        border-radius: 12px !important;
        overflow: hidden !important;
        border: 1px solid var(--lunar-border) !important;
        background: var(--lunar-card) !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00b4d8 0%, #0077b6 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(0, 229, 255, 0.3) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.5rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 14px rgba(0, 119, 182, 0.4) !important;
    }

    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 22px rgba(0, 229, 255, 0.5) !important;
        border-color: var(--lunar-accent) !important;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        background: var(--lunar-card) !important;
        border: 1px solid var(--lunar-border) !important;
        border-radius: 12px !important;
    }
    </style>
    """, unsafe_allow_html=True)


inject_custom_css()

# ── Sidebar Brand Header ─────────────────────────────────────────────
st.sidebar.markdown(
    """
    <div style="padding: 10px 0 14px 0; border-bottom: 1px solid rgba(42, 58, 92, 0.5); margin-bottom: 12px;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 1.8rem;">🌑</span>
            <div>
                <div style="font-size: 1.05rem; font-weight: 800; color: #00e5ff; letter-spacing: -0.01em; line-height: 1.2;">
                    LUNAR CRATER AI
                </div>
                <div style="font-size: 0.72rem; color: #94a3b8; font-weight: 500;">
                    Micro-Quantized CNN Research
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Import Page Renderers ────────────────────────────────────────────
from pages.overview import render_overview
from pages.dataset_inspection import render_dataset_inspection
from pages.live_inference import render_live_inference
from pages.benchmarks import render_benchmarks
from pages.training_history import render_training_history
from pages.experiment_registry import render_experiment_registry
from pages.architecture import render_architecture
from pages.system_diagnostics import render_system_diagnostics

# ── Multi-Page Navigation ────────────────────────────────────────────
pages = {
    "Research & Data": [
        st.Page(render_overview, title="Overview", icon="🏠", default=True),
        st.Page(render_dataset_inspection, title="Dataset Inspection", icon="🛰️"),
    ],
    "Model Inference & Benchmarks": [
        st.Page(render_live_inference, title="Live Crater Detection", icon="🔬"),
        st.Page(render_benchmarks, title="Benchmark Results", icon="📊"),
    ],
    "Engineering & Diagnostics": [
        st.Page(render_training_history, title="Training History", icon="📈"),
        st.Page(render_experiment_registry, title="Experiment Registry", icon="🗄️"),
        st.Page(render_architecture, title="Architecture Details", icon="📐"),
        st.Page(render_system_diagnostics, title="System Diagnostics", icon="🩺"),
    ],
}

nav = st.navigation(pages)

# ── Sidebar Mission Specs Footer ─────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style="font-size: 0.75rem; color: #94a3b8; line-height: 1.6; background: rgba(26, 32, 53, 0.4); padding: 12px; border-radius: 8px; border: 1px solid rgba(42, 58, 92, 0.4);">
        <b style="color: #00e5ff;">Mission Specification</b><br>
        • <b>Target:</b> LRO NAC Lunar Imagery<br>
        • <b>Detector:</b> SSDLite320 + MobileNetV3<br>
        • <b>Precision:</b> FP32 (14.3 MB) → INT8 (4.1 MB)<br>
        • <b>Reduction:</b> <span style="color: #00e676; font-weight: 700;">-71.3% Storage</span><br>
        • <b>Runtime:</b> ONNX Runtime CPU<br>
        • <b>Air-Gap:</b> 100% Offline / Local
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Run Selected Page ────────────────────────────────────────────────
nav.run()
