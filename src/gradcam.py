"""
gradcam.py - Generate Grad-CAM heatmap explanations for the best model.
Shows which regions of the fundus image influenced the cataract prediction.
"""
import os
import json
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MODELS_DIR, GRADCAM_DIR, PLOTS_DIR, SPLITS_DIR, IMAGES_DIR, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD
from dataset import CataractDataset, get_eval_transforms
from models import build_model, get_target_layer
import pandas as pd


def load_best_model(device):
    """Load the best model and its threshold."""
    info_path = os.path.join(MODELS_DIR, "best_model_info.json")
    with open(info_path, "r") as f:
        info = json.load(f)

    model_name = info["model_name"]
    threshold = info["threshold"]

    model, _ = build_model(model_name, freeze_early=False)
    ckpt = torch.load(info["checkpoint"], map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    target_layer = get_target_layer(model, model_name)
    return model, model_name, threshold, target_layer


def get_original_image(img_path):
    """Load and resize image to 224x224, return as numpy float [0,1]."""
    img = Image.open(img_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return np.array(img).astype(np.float32) / 255.0


def generate_gradcam_panel(model, target_layer, img_tensor, original_np,
                           true_label, pred_prob, threshold, title, save_path):
    """
    Generate and save a 3-panel figure: Original | Heatmap | Overlay
    """
    # Grad-CAM
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=img_tensor.unsqueeze(0), targets=None)
    grayscale_cam = grayscale_cam[0, :]

    # Create overlay
    overlay = show_cam_on_image(original_np, grayscale_cam, use_rgb=True)

    # Create heatmap visualization
    heatmap = plt.cm.jet(grayscale_cam)[:, :, :3]

    # Plot 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original_np)
    axes[0].set_title("Original", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(heatmap)
    axes[1].set_title("Grad-CAM Heatmap", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay", fontsize=12)
    axes[2].axis("off")

    pred_class = "Cataract" if pred_prob >= threshold else "Normal"
    true_class = "Cataract" if true_label == 1 else "Normal"
    correct = "✓" if (pred_class == true_class) else "✗"

    fig.suptitle(
        f"{title}\n"
        f"True: {true_class} | Predicted: {pred_class} ({pred_prob:.4f}) | "
        f"Threshold: {threshold:.2f} | {correct}",
        fontsize=13, fontweight="bold", y=1.02
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    print("=" * 55)
    print("PHASE 4: GRAD-CAM EXPLANATIONS")
    print("=" * 55)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_name, threshold, target_layer = load_best_model(device)
    print(f"Best model: {model_name}, threshold: {threshold:.2f}")

    # Load test dataset
    manifest_df = pd.read_csv(os.path.join(SPLITS_DIR, "manifest.csv"))
    test_ds = CataractDataset(manifest_df, "test", transform=get_eval_transforms())

    # Get predictions for all test images
    predictions = []
    with torch.no_grad():
        for idx in range(len(test_ds)):
            img_tensor, label = test_ds[idx]
            img_tensor = img_tensor.to(device)
            logit = model(img_tensor.unsqueeze(0)).squeeze()
            prob = torch.sigmoid(logit).item()
            pred = 1 if prob >= threshold else 0
            info = test_ds.get_info(idx)
            predictions.append({
                "idx": idx,
                "true_label": int(label.item()),
                "pred_label": pred,
                "prob": prob,
                "correct": pred == int(label.item()),
                **info
            })

    pred_df = pd.DataFrame(predictions)

    # Select examples for Grad-CAM
    # 4 correct cataract, 4 correct normal, up to 4 misclassified
    correct_cat = pred_df[(pred_df["correct"]) & (pred_df["true_label"] == 1)].head(4)
    correct_norm = pred_df[(pred_df["correct"]) & (pred_df["true_label"] == 0)].head(4)
    misclassified = pred_df[~pred_df["correct"]].head(4)

    examples = [
        (correct_cat, "Correct Cataract"),
        (correct_norm, "Correct Normal"),
        (misclassified, "Misclassified"),
    ]

    count = 0
    for group_df, group_name in examples:
        for i, (_, row) in enumerate(group_df.iterrows()):
            idx = row["idx"]
            img_tensor, label = test_ds[idx]
            img_path = test_ds.get_filepath(idx)
            original_np = get_original_image(img_path)

            title = f"{group_name} - Patient {row['patient_id']}, {row['eye']} eye"
            save_name = f"gradcam_{group_name.lower().replace(' ', '_')}_{i+1}.png"
            save_path = os.path.join(GRADCAM_DIR, save_name)

            generate_gradcam_panel(
                model, target_layer,
                img_tensor.to(device), original_np,
                int(label.item()), row["prob"], threshold,
                title, save_path
            )
            print(f"  Saved: {save_name}")
            count += 1

    print(f"\n✅ Phase 4 complete! Generated {count} Grad-CAM explanations.")
    print(f"   Saved in: {GRADCAM_DIR}")


if __name__ == "__main__":
    main()

