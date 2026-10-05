"""
app.py
======
Interactive Streamlit Web Application for Deep Learning Based Glacial Lake Segmentation.

Powered by the centralized inference engine (src/inference.py).
Supports side-by-side exploration of:
  - DeepLabV3+ Advanced Model (ResNet-50 + ASPP)
  - ResNet34-FCN Baseline Model (ResNet-34 + 5-Stage Progressive FCN Decoder)

Usage:
    streamlit run app.py
"""

from pathlib import Path
from typing import Optional, Tuple
import io

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import streamlit as st

# Ensure repository root is in sys.path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import (
    DEFAULT_THRESHOLD,
    DEFAULT_IMAGE_SIZE,
    GlacialLakePredictor,
    ModelMetadata,
    PredictionResult,
    select_device,
)


# ---------------------------------------------------------------------------
# Default Checkpoint Paths (Resolved relative to project root)
# ---------------------------------------------------------------------------
CHECKPOINTS = {
    "DeepLabV3+ (Advanced — ResNet-50 + ASPP)": {
        "key": "deeplabv3plus",
        "path": PROJECT_ROOT / "checkpoints" / "best_amp_model.pth",
        "description": "40.3M parameters, Atrous Spatial Pyramid Pooling with low-level boundary feature fusion.",
    },
    "ResNet34-FCN (Baseline — ResNet-34 + FCN)": {
        "key": "resnet34_fcn",
        "path": PROJECT_ROOT / "resnet34_fcn" / "checkpoints" / "selected_model.pth",
        "alt_path": PROJECT_ROOT / "resnet34_fcn" / "checkpoints" / "best_model.pth",
        "description": "24.1M parameters, 5-stage progressive transposed convolution decoder.",
    },
}


# ---------------------------------------------------------------------------
# Cached Model Loader
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading segmentation model into memory...")
def get_cached_predictor(checkpoint_str: str, device_str: str) -> GlacialLakePredictor:
    """
    Load and cache the model in memory across Streamlit re-runs.

    Args:
        checkpoint_str: Absolute or relative filesystem path to .pth checkpoint.
        device_str: Target device string ('cuda', 'mps', or 'cpu').

    Returns:
        Instantiated and evaluated GlacialLakePredictor.
    """
    chk_path = Path(checkpoint_str)
    if not chk_path.is_file():
        raise FileNotFoundError(f"Checkpoint file does not exist at: {chk_path}")

    return GlacialLakePredictor(
        checkpoint_path=chk_path,
        device=device_str,
    )


def generate_probability_heatmap(prob_map: np.ndarray) -> np.ndarray:
    """Render a colored probability heatmap using viridis colormap."""
    cmap = plt.get_cmap("viridis")
    rgba = cmap(prob_map)  # (H, W, 4) in [0.0, 1.0]
    rgb = (rgba[:, :, :3] * 255.0).astype(np.uint8)
    return rgb


# ---------------------------------------------------------------------------
# Streamlit Application
# ---------------------------------------------------------------------------

def main() -> None:
    # Page setup
    st.set_page_config(
        page_title="Glacial Lake Segmentation — AI Mapping System",
        page_icon="🏔️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Header
    st.title("🏔️ Glacial Lake Detection & Segmentation")
    st.markdown(
        "**Deep Learning Based Remote Sensing Analysis for Cryospheric Water Bodies**  \n"
        "Accurate identification and boundary segmentation of high-altitude glacial lakes "
        "using Deep Transfer Learning on satellite imagery (GLID benchmark)."
    )
    st.divider()

    # -----------------------------------------------------------------------
    # Sidebar — Configuration & Model Controls
    # -----------------------------------------------------------------------
    st.sidebar.header("⚙️ Model Configuration")

    # 1. Model Selection
    model_choice = st.sidebar.selectbox(
        "Select Architecture",
        options=list(CHECKPOINTS.keys()),
        index=0,
        help="Choose between the advanced DeepLabV3+ model and the ResNet34-FCN baseline.",
    )
    model_info = CHECKPOINTS[model_choice]

    # Resolve checkpoint path
    chk_path = model_info["path"]
    if not chk_path.is_file() and "alt_path" in model_info and model_info["alt_path"].is_file():
        chk_path = model_info["alt_path"]

    # 2. Checkpoint Provenance Status
    if chk_path.is_file():
        st.sidebar.success(f"✓ Checkpoint found: `{chk_path.name}`")
    else:
        st.sidebar.error(f"⚠️ Checkpoint not found at `{chk_path}`")
        st.sidebar.info("Please verify that the trained checkpoint file is present.")

    st.sidebar.caption(model_info["description"])
    st.sidebar.markdown("---")

    # 3. Decision Threshold Slider
    st.sidebar.subheader("🎯 Decision Threshold")
    threshold = st.sidebar.slider(
        "Binary Lake Threshold (t)",
        min_value=0.10,
        max_value=0.90,
        value=DEFAULT_THRESHOLD,
        step=0.05,
        help="Probability threshold above which a pixel is classified as Glacial Lake (standard: 0.50).",
    )
    st.sidebar.caption(f"Active threshold: **t = {threshold:.2f}**")
    st.sidebar.markdown("---")

    # 4. Hardware Device Selection
    st.sidebar.subheader("💻 Inference Device")
    available_device = str(select_device())
    device_override = st.sidebar.selectbox(
        "Execution Device",
        options=["auto", "cpu", "mps", "cuda"],
        index=0,
        help="Select computation device. 'auto' selects the fastest available accelerator.",
    )
    active_device = available_device if device_override == "auto" else device_override
    st.sidebar.caption(f"Running on: `{active_device.upper()}`")

    # -----------------------------------------------------------------------
    # Main Panel — Upload & Inference
    # -----------------------------------------------------------------------
    col_upload, col_preview = st.columns([1, 1], gap="medium")

    with col_upload:
        st.subheader("1. Upload Satellite Image")
        uploaded_file = st.file_uploader(
            "Upload an optical RGB satellite image patch (PNG, JPG, JPEG, TIF)",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            help="High-altitude multi-spectral or true-color RGB satellite patch.",
        )

        st.caption("ℹ️ Uploaded images will be automatically converted to RGB and formatted to 512×512 resolution.")

    # Process and display uploaded image
    pil_image = None
    if uploaded_file is not None:
        try:
            pil_image = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"❌ Could not open image file: {e}")
            return

    with col_preview:
        st.subheader("2. Input Preview")
        if pil_image is not None:
            st.image(
                pil_image,
                caption=f"Uploaded: {uploaded_file.name} (Original size: {pil_image.size[0]}×{pil_image.size[1]})",
                use_container_width=True,
            )
        else:
            st.info("👈 Upload a satellite image patch on the left to begin.")

    # -----------------------------------------------------------------------
    # Segmentation Trigger & Execution
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.subheader("3. Segmentation Pipeline")

    run_button = st.button(
        "🚀 Run Glacial Lake Segmentation",
        type="primary",
        disabled=(pil_image is None or not chk_path.is_file()),
        use_container_width=True,
    )

    if run_button and pil_image is not None:
        # Load cached model
        try:
            predictor = get_cached_predictor(str(chk_path), active_device)
        except Exception as e:
            st.error(f"❌ Failed to load model checkpoint: {e}")
            return

        # Execute inference
        with st.spinner("Processing satellite imagery and generating semantic masks..."):
            try:
                result: PredictionResult = predictor.predict(
                    image_input=pil_image,
                    threshold=threshold,
                )
            except Exception as e:
                st.error(f"❌ Inference execution failed: {e}")
                return

        # -------------------------------------------------------------------
        # Results Display
        # -------------------------------------------------------------------
        st.success("✓ Semantic segmentation completed successfully!")

        st.subheader("4. Visual Segmentation Results")
        res_col1, res_col2, res_col3, res_col4 = st.columns(4, gap="small")

        with res_col1:
            st.markdown("**1. Satellite RGB**")
            st.image(result.processed_image, use_container_width=True, caption="512×512 Standardized Input")

        with res_col2:
            st.markdown(f"**2. Binary Mask (t={threshold:.2f})**")
            # Scale 0/1 to 0/255 for display
            mask_display = result.binary_mask * 255
            st.image(mask_display, use_container_width=True, caption="1 = Glacial Lake, 0 = Background")

        with res_col3:
            st.markdown("**3. High-Contrast Overlay**")
            st.image(result.overlay_image, use_container_width=True, caption="Cyan Highlight: Detected Lake")

        with res_col4:
            st.markdown("**4. Lake Probability Heatmap**")
            heatmap_rgb = generate_probability_heatmap(result.probability_map)
            st.image(heatmap_rgb, use_container_width=True, caption="Viridis: [0.0 (Purple) → 1.0 (Yellow)]")

        # -------------------------------------------------------------------
        # Quantitative Summary & Statistics
        # -------------------------------------------------------------------
        st.subheader("5. Quantitative Prediction Statistics")
        stats = result.statistics
        meta: ModelMetadata = predictor.metadata

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric(
                label="Glacial Lake Detected?",
                value="YES" if stats["has_lake_detected"] else "NO",
                delta=f"{stats['lake_pixels']:,} px" if stats["has_lake_detected"] else "0 px",
            )
        with m_col2:
            st.metric(
                label="Lake Surface Coverage",
                value=f"{stats['lake_percentage']:.2f}%",
                delta=f"Total: {stats['total_pixels']:,} px",
            )
        with m_col3:
            st.metric(
                label="Mean Lake Confidence",
                value=f"{stats['mean_lake_confidence'] * 100:.1f}%",
                delta=f"Max: {stats['max_lake_confidence'] * 100:.1f}%",
            )
        with m_col4:
            st.metric(
                label="Model Parameters",
                value=f"{meta.total_parameters / 1e6:.1f}M",
                delta=f"Epoch {meta.epoch}",
            )

        # Technical Provenance Expander
        with st.expander("🔍 View Technical Model & Inference Metadata"):
            st.json({
                "Architecture": meta.arch_type,
                "Model Internal Key": meta.model_name,
                "Checkpoint Path": meta.checkpoint_path,
                "Checkpoint Epoch": meta.epoch,
                "Best Validation F1": meta.best_f1,
                "Total Parameters": meta.total_parameters,
                "Trainable Parameters": meta.trainable_parameters,
                "Inference Device": meta.device,
                "Decision Threshold": result.threshold,
                "Lake Pixels Count": stats["lake_pixels"],
                "Total Pixels": stats["total_pixels"],
                "Lake Percentage": stats["lake_percentage"],
                "Mean Lake Confidence": stats["mean_lake_confidence"],
                "Max Lake Confidence": stats["max_lake_confidence"],
                "Mean Scene Confidence": stats["mean_scene_confidence"],
            })


if __name__ == "__main__":
    main()
