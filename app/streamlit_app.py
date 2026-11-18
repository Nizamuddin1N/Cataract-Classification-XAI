"""
streamlit_app.py - Simple web demo for cataract classification with Grad-CAM.
Upload a fundus image → get prediction + probability + Grad-CAM heatmap.

Run with:  streamlit run app/streamlit_app.py
"""
import os
import sys
import json
import numpy as np
import torch
from PIL import Image
import streamlit as st

# Add src to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from config import MODELS_DIR, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD
from models import build_model, get_target_layer
from dataset import get_eval_transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


@st.cache_resource
def load_model():
    """Load the best model (cached so it only loads once)."""
    info_path = os.path.join(MODELS_DIR, "best_model_info.json")
    if not os.path.exists(info_path):
        return None, None, None, None

    with open(info_path, "r") as f:
        info = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = info["model_name"]
    threshold = info["threshold"]

    model, _ = build_model(model_name, freeze_early=False)
    ckpt = torch.load(info["checkpoint"], map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    target_layer = get_target_layer(model, model_name)
    return model, model_name, threshold, target_layer


def predict_with_gradcam(model, target_layer, image_pil, threshold, device):
    """Run prediction and generate Grad-CAM."""
    # Prepare image for model
    transform = get_eval_transforms()
    img_tensor = transform(image_pil).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        logit = model(img_tensor).squeeze()
        prob = torch.sigmoid(logit).item()

    pred_class = "Cataract" if prob >= threshold else "Normal"

    # Grad-CAM
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=img_tensor, targets=None)[0, :]

    # Create overlay
    img_resized = image_pil.resize((IMG_SIZE, IMG_SIZE))
    img_np = np.array(img_resized).astype(np.float32) / 255.0
    overlay = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    return pred_class, prob, overlay, img_np


def main():
    st.set_page_config(
        page_title="Cataract Detection",
        page_icon="🏥",
        layout="wide"
    )

    st.title("🏥 Cataract Detection using Explainable AI")
    st.markdown(
        "Upload a retinal fundus image to classify it as **Cataract** or **Normal**, "
        "with a Grad-CAM heatmap showing which regions influenced the prediction."
    )

    # Load model
    model, model_name, threshold, target_layer = load_model()
    if model is None:
        st.error("❌ No trained model found! Run training first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.sidebar.markdown(f"**Model:** {model_name}")
    st.sidebar.markdown(f"**Threshold:** {threshold:.2f}")
    st.sidebar.markdown(f"**Device:** {device}")

    # Upload image
    uploaded = st.file_uploader(
        "Upload a fundus image (JPG/PNG)", type=["jpg", "jpeg", "png"]
    )

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")

        # Run prediction
        with st.spinner("Analyzing..."):
            pred_class, prob, overlay, img_np = predict_with_gradcam(
                model, target_layer, image, threshold, device
            )

        # Display results
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("Grad-CAM Overlay")
            st.image(overlay, use_container_width=True)

        with col3:
            st.subheader("Prediction")
            if pred_class == "Cataract":
                st.error(f"🔴 **{pred_class}**")
            else:
                st.success(f"🟢 **{pred_class}**")

            st.metric("Cataract Probability", f"{prob:.4f}")
            st.metric("Threshold", f"{threshold:.2f}")

            if abs(prob - threshold) < 0.05:
                st.warning("⚠️ Score is close to the boundary.")

        st.markdown("---")
        st.caption(
            "⚠️ This is a research prototype, not a clinical diagnostic tool. "
            "It does not replace professional medical evaluation."
        )


if __name__ == "__main__":
    main()

