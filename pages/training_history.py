"""
Training History Page — Loss curves and training metadata.
"""

import json
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.config import load_config


def render_training_history() -> None:
    cfg = load_config()

    st.markdown(
        """
        <h1 style="
            background: linear-gradient(135deg, #00e5ff, #b388ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
        ">📈 Training History</h1>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Training loss curves, checkpoint metadata, and convergence analysis")
    st.markdown("---")

    # Load best checkpoint metadata
    best_meta = cfg.paths.checkpoints_dir / "crater_detector_baseline_best.json"
    final_meta = cfg.paths.checkpoints_dir / "crater_detector_baseline_final.json"

    if not best_meta.is_file():
        st.warning("⚠️ No training metadata found. Run `python scripts/train_model.py` first.")
        return

    with open(best_meta, "r") as f:
        best_data = json.load(f)

    final_data = {}
    if final_meta.is_file():
        with open(final_meta, "r") as f:
            final_data = json.load(f)

    # ── Training Configuration ───────────────────────────────────────
    st.markdown("### ⚙️ Training Configuration")

    cfg_cols = st.columns(4)
    with cfg_cols[0]:
        st.metric("Architecture", cfg.model.architecture.split("_")[0].upper())
    with cfg_cols[1]:
        st.metric("Backbone", cfg.model.backbone)
    with cfg_cols[2]:
        st.metric("Epochs", str(cfg.training.epochs))
    with cfg_cols[3]:
        st.metric("Learning Rate", f"{cfg.training.learning_rate:.4f}")

    cfg_cols2 = st.columns(4)
    with cfg_cols2[0]:
        st.metric("Batch Size", str(cfg.training.batch_size))
    with cfg_cols2[1]:
        st.metric("Optimizer", cfg.training.optimizer.upper())
    with cfg_cols2[2]:
        st.metric("LR Scheduler", cfg.training.lr_scheduler.title())
    with cfg_cols2[3]:
        st.metric("Device", cfg.training.device.upper())

    st.markdown("---")

    # ── Checkpoint Metadata ──────────────────────────────────────────
    st.markdown("### 🏆 Best Checkpoint")

    chk_cols = st.columns(3)
    with chk_cols[0]:
        st.markdown(f"**Saved At Epoch:** {best_data.get('epoch', 'N/A')}")
        st.markdown(f"**Best Val Loss:** {best_data.get('best_val_loss', 'N/A')}")
    with chk_cols[1]:
        st.markdown(f"**Architecture:** {best_data.get('architecture', 'N/A')}")
        st.markdown(f"**Num Classes:** {best_data.get('num_classes', 'N/A')}")
    with chk_cols[2]:
        st.markdown(f"**Checkpoint SHA256:** `{best_data.get('sha256', 'N/A')[:16]}...`")

    st.markdown("---")

    # ── Loss Curves ──────────────────────────────────────────────────
    st.markdown("### 📉 Training & Validation Loss Curves")

    # Try to load training history from final checkpoint
    train_losses = best_data.get("train_losses", [])
    val_losses = best_data.get("val_losses", [])

    if not train_losses and final_data:
        train_losses = final_data.get("train_losses", [])
        val_losses = final_data.get("val_losses", [])

    if train_losses:
        epochs_range = list(range(1, len(train_losses) + 1))

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=epochs_range,
            y=train_losses,
            mode="lines+markers",
            name="Train Loss",
            line=dict(color="#00b4d8", width=3),
            marker=dict(size=8, color="#00b4d8", line=dict(color="#ffffff", width=1)),
        ))

        if val_losses:
            fig.add_trace(go.Scatter(
                x=epochs_range,
                y=val_losses,
                mode="lines+markers",
                name="Val Loss",
                line=dict(color="#ffb700", width=3, dash="dash"),
                marker=dict(size=8, color="#ffb700", symbol="diamond", line=dict(color="#ffffff", width=1)),
            ))

        # Mark best epoch
        best_epoch = best_data.get("epoch", 1)
        if val_losses and best_epoch <= len(val_losses):
            fig.add_trace(go.Scatter(
                x=[best_epoch],
                y=[val_losses[best_epoch - 1]],
                mode="markers",
                name="Best Checkpoint",
                marker=dict(size=16, color="#00e676", symbol="star", line=dict(color="#ffffff", width=2)),
            ))

        fig.update_layout(
            height=420,
            margin=dict(l=40, r=40, t=30, b=40),
            plot_bgcolor="rgba(26, 32, 53, 0.6)",
            paper_bgcolor="rgba(0, 0, 0, 0)",
            font=dict(color="#e0e6ed", family="Inter"),
            xaxis=dict(
                title="Epoch",
                gridcolor="rgba(42, 58, 92, 0.5)",
                dtick=1,
            ),
            yaxis=dict(
                title="Loss",
                gridcolor="rgba(42, 58, 92, 0.5)",
            ),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                bgcolor="rgba(26, 32, 53, 0.8)", bordercolor="rgba(42, 58, 92, 0.5)", borderwidth=1,
            ),
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Loss history not available in checkpoint metadata. Training may need to save per-epoch losses.")

    # ── Training Details ─────────────────────────────────────────────
    st.markdown("### 📝 Full Checkpoint Metadata")
    with st.expander("Best Checkpoint JSON"):
        st.json(best_data)

    if final_data:
        with st.expander("Final Checkpoint JSON"):
            st.json(final_data)

    # ── Research Notes ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📘 Training Methodology Notes")
    st.info(
        """
        **Transfer Learning Strategy:** The SSDLite320 model uses a MobileNetV3-Large backbone 
        pre-trained on ImageNet (via `MobileNet_V3_Large_Weights.DEFAULT`). The detection head 
        is randomly initialized for the crater detection task (2 classes: background + crater).

        **Data Pipeline:** Images are resized to 320×320 with proper annotation coordinate scaling. 
        The train/val/test split uses group-based stratification to prevent data leakage 
        (all craters from a single source image stay within the same split).

        **Training Note:** With only 8 epochs on a small dataset (20 tiles, CPU training), 
        the model demonstrates proof-of-concept learning. Production-quality mAP would require 
        significantly more training data, GPU-accelerated training, and data augmentation 
        (random flips, rotations, brightness jitter).
        """
    )


if __name__ == "__main__":
    render_training_history()

