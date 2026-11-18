"""
train.py - Training loop for binary cataract classification.
Supports ResNet50 and EfficientNet-B0 with early stopping and LR scheduling.
"""
import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    SPLITS_DIR, MODELS_DIR, PLOTS_DIR, IMAGES_DIR,
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    MAX_EPOCHS, EARLY_STOP_PATIENCE, LR_PATIENCE, LR_MIN,
    NUM_WORKERS, SEED, MODEL_NAMES
)
from dataset import CataractDataset, get_train_transforms, get_eval_transforms
from models import build_model


def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_model(model_name, manifest_df, device):
    """
    Train a single model end-to-end.

    Returns:
        history: dict with training metrics per epoch
    """
    print(f"\n{'='*55}")
    print(f"TRAINING: {model_name.upper()}")
    print(f"{'='*55}")

    # Build model
    model, _ = build_model(model_name, freeze_early=True)
    model = model.to(device)

    # Create datasets and loaders
    train_ds = CataractDataset(manifest_df, "train", transform=get_train_transforms())
    val_ds = CataractDataset(manifest_df, "val", transform=get_eval_transforms())

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    # Compute positive class weight for imbalanced data
    train_labels = manifest_df[manifest_df["split"] == "train"]["label"]
    n_normal = (train_labels == 0).sum()
    n_cataract = (train_labels == 1).sum()
    pos_weight = torch.tensor([n_normal / n_cataract], dtype=torch.float32).to(device)
    print(f"  pos_weight: {pos_weight.item():.4f}")

    # Loss, optimizer, scheduler
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=LR_PATIENCE,
        factor=0.5, min_lr=LR_MIN
    )

    # Training history
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "val_roc_auc": [], "lr": []
    }

    best_val_auc = 0.0
    best_epoch = 0
    patience_counter = 0

    start_time = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        # ── Train ──
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # ── Validate ──
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images).squeeze(1)
                loss = criterion(logits, labels)

                val_loss_sum += loss.item() * images.size(0)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total
        val_auc = roc_auc_score(all_labels, all_probs)
        current_lr = optimizer.param_groups[0]["lr"]

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_roc_auc"].append(val_auc)
        history["lr"].append(current_lr)

        # Print epoch summary
        print(f"  Epoch {epoch:2d}/{MAX_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} AUC: {val_auc:.4f} | "
              f"LR: {current_lr:.6f}")

        # Step scheduler
        scheduler.step(val_auc)

        # Early stopping check
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
            # Save best checkpoint
            ckpt_path = os.path.join(MODELS_DIR, f"{model_name}_best.pth")
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_roc_auc": val_auc,
                "model_name": model_name,
            }, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"  ⚡ Early stopping at epoch {epoch}. "
                      f"Best val AUC: {best_val_auc:.4f} at epoch {best_epoch}")
                break

    elapsed = time.time() - start_time
    print(f"\n  ✅ {model_name} done in {elapsed:.1f}s. "
          f"Best epoch: {best_epoch}, Best val AUC: {best_val_auc:.4f}")

    # Save history
    hist_path = os.path.join(MODELS_DIR, f"{model_name}_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    return history


def plot_training_curves(histories, plots_dir):
    """Plot training curves for all models on the same figure."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for model_name, hist in histories.items():
        epochs = range(1, len(hist["train_loss"]) + 1)

        # Loss
        axes[0].plot(epochs, hist["train_loss"], label=f"{model_name} train", linestyle="--")
        axes[0].plot(epochs, hist["val_loss"], label=f"{model_name} val")

        # Accuracy
        axes[1].plot(epochs, hist["train_acc"], label=f"{model_name} train", linestyle="--")
        axes[1].plot(epochs, hist["val_acc"], label=f"{model_name} val")

        # Val ROC-AUC
        axes[2].plot(epochs, hist["val_roc_auc"], label=model_name, marker="o", markersize=3)

    axes[0].set_title("Loss", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Accuracy", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].set_title("Validation ROC-AUC", fontsize=14, fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("ROC-AUC")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(plots_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Load manifest
    manifest_path = os.path.join(SPLITS_DIR, "manifest.csv")
    manifest_df = pd.read_csv(manifest_path)
    print(f"Loaded manifest: {len(manifest_df)} images")

    # Train each model
    histories = {}
    for model_name in MODEL_NAMES:
        histories[model_name] = train_one_model(model_name, manifest_df, device)

    # Plot training curves
    plot_training_curves(histories, PLOTS_DIR)
    print("\n✅ Phase 2 complete! All models trained.")


if __name__ == "__main__":
    main()

