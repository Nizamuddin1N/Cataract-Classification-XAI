"""
evaluate.py - Evaluate trained models on the test set.
Generates confusion matrices, ROC curves, and a comparison table.
"""
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve,
    classification_report
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    SPLITS_DIR, MODELS_DIR, PLOTS_DIR, OUTPUT_DIR,
    BATCH_SIZE, NUM_WORKERS, MODEL_NAMES
)
from dataset import CataractDataset, get_eval_transforms
from models import build_model


def find_best_threshold(labels, probs):
    """Find the threshold that maximizes cataract F1 on validation."""
    best_f1 = 0.0
    best_thresh = 0.5
    for t in np.arange(0.05, 0.96, 0.01):
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1 or (f1 == best_f1 and t < best_thresh):
            best_f1 = f1
            best_thresh = t
    return best_thresh, best_f1


def evaluate_model(model_name, manifest_df, device):
    """Evaluate a single model. Returns metrics dict."""
    print(f"\n{'='*55}")
    print(f"EVALUATING: {model_name.upper()}")
    print(f"{'='*55}")

    # Load model
    model, _ = build_model(model_name, freeze_early=False)
    ckpt_path = os.path.join(MODELS_DIR, f"{model_name}_best.pth")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # ── Validation pass (find threshold) ──
    val_ds = CataractDataset(manifest_df, "val", transform=get_eval_transforms())
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    val_probs, val_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits = model(images).squeeze(1)
            probs = torch.sigmoid(logits)
            val_probs.extend(probs.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_probs = np.array(val_probs)
    val_labels = np.array(val_labels)
    threshold, val_f1 = find_best_threshold(val_labels, val_probs)
    val_auc = roc_auc_score(val_labels, val_probs)
    print(f"  Validation ROC-AUC: {val_auc:.4f}")
    print(f"  Frozen threshold: {threshold:.2f} (val cataract F1: {val_f1:.4f})")

    # ── Test pass ──
    test_ds = CataractDataset(manifest_df, "test", transform=get_eval_transforms())
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    test_probs, test_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images).squeeze(1)
            probs = torch.sigmoid(logits)
            test_probs.extend(probs.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    test_probs = np.array(test_probs)
    test_labels = np.array(test_labels)
    test_preds = (test_probs >= threshold).astype(int)

    # Compute metrics
    metrics = {
        "model": model_name,
        "threshold": float(threshold),
        "val_roc_auc": float(val_auc),
        "val_f1": float(val_f1),
        "test_accuracy": float(accuracy_score(test_labels, test_preds)),
        "test_precision": float(precision_score(test_labels, test_preds, zero_division=0)),
        "test_recall": float(recall_score(test_labels, test_preds, zero_division=0)),
        "test_specificity": float(recall_score(test_labels, test_preds, pos_label=0, zero_division=0)),
        "test_f1": float(f1_score(test_labels, test_preds, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(test_labels, test_probs)),
        "test_pr_auc": float(average_precision_score(test_labels, test_probs)),
        "best_epoch": int(checkpoint["epoch"]),
    }

    # Print results
    print(f"\n  Test Results ({model_name}):")
    print(f"  {'Accuracy':<20s}: {metrics['test_accuracy']:.4f}")
    print(f"  {'Precision':<20s}: {metrics['test_precision']:.4f}")
    print(f"  {'Recall (Sensitivity)':<20s}: {metrics['test_recall']:.4f}")
    print(f"  {'Specificity':<20s}: {metrics['test_specificity']:.4f}")
    print(f"  {'F1 Score':<20s}: {metrics['test_f1']:.4f}")
    print(f"  {'ROC-AUC':<20s}: {metrics['test_roc_auc']:.4f}")
    print(f"  {'PR-AUC':<20s}: {metrics['test_pr_auc']:.4f}")

    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    print(f"\n  Confusion Matrix:")
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")

    return metrics, test_probs, test_labels, threshold


def plot_confusion_matrix(test_labels, test_preds, model_name, plots_dir):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(test_labels, test_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Normal", "Cataract"],
                yticklabels=["Normal", "Cataract"],
                annot_kws={"size": 16})
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix - {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(plots_dir, f"confusion_matrix_{model_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_roc_curves(all_results, plots_dir):
    """Plot ROC curves for all models on the same figure."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#e74c3c", "#3498db"]

    for i, (model_name, probs, labels, _) in enumerate(all_results):
        fpr, tpr, _ = roc_curve(labels, probs)
        auc_val = roc_auc_score(labels, probs)
        ax.plot(fpr, tpr, color=colors[i], lw=2,
                label=f"{model_name} (AUC = {auc_val:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves - Model Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(plots_dir, "roc_curves_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_pr_curves(all_results, plots_dir):
    """Plot Precision-Recall curves."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#e74c3c", "#3498db"]

    for i, (model_name, probs, labels, _) in enumerate(all_results):
        precision, recall, _ = precision_recall_curve(labels, probs)
        ap = average_precision_score(labels, probs)
        ax.plot(recall, precision, color=colors[i], lw=2,
                label=f"{model_name} (AP = {ap:.4f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(plots_dir, "pr_curves_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_metrics_comparison(all_metrics, plots_dir):
    """Bar chart comparing key metrics across models."""
    metrics_to_plot = ["test_accuracy", "test_precision", "test_recall",
                       "test_f1", "test_roc_auc", "test_pr_auc"]
    labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(labels))
    width = 0.35
    colors = ["#e74c3c", "#3498db"]

    for i, m in enumerate(all_metrics):
        values = [m[k] for k in metrics_to_plot]
        bars = ax.bar(x + i * width, values, width, label=m["model"], color=colors[i], edgecolor="black")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison - Test Metrics", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(plots_dir, "metrics_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    manifest_df = pd.read_csv(os.path.join(SPLITS_DIR, "manifest.csv"))

    all_metrics = []
    all_results = []

    for model_name in MODEL_NAMES:
        metrics, probs, labels, thresh = evaluate_model(model_name, manifest_df, device)
        all_metrics.append(metrics)
        all_results.append((model_name, probs, labels, thresh))

        # Plot individual confusion matrix
        preds = (probs >= thresh).astype(int)
        plot_confusion_matrix(labels, preds, model_name, PLOTS_DIR)

    # Comparative plots
    plot_roc_curves(all_results, PLOTS_DIR)
    plot_pr_curves(all_results, PLOTS_DIR)
    plot_metrics_comparison(all_metrics, PLOTS_DIR)

    # Save metrics table
    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = os.path.join(OUTPUT_DIR, "results", "test_metrics.csv")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved metrics: {metrics_path}")

    # Select best model
    best = max(all_metrics, key=lambda m: m["val_roc_auc"])
    print(f"\n🏆 Best model: {best['model']} (val ROC-AUC: {best['val_roc_auc']:.4f})")

    # Save best model info
    best_info = {
        "model_name": best["model"],
        "threshold": best["threshold"],
        "val_roc_auc": best["val_roc_auc"],
        "checkpoint": os.path.join(MODELS_DIR, f"{best['model']}_best.pth"),
    }
    with open(os.path.join(MODELS_DIR, "best_model_info.json"), "w") as f:
        json.dump(best_info, f, indent=2)

    # Print comparison table
    print(f"\n{'='*70}")
    print("FINAL COMPARISON TABLE")
    print(f"{'='*70}")
    print(metrics_df[["model", "test_accuracy", "test_precision", "test_recall",
                       "test_f1", "test_roc_auc", "test_pr_auc", "threshold"]].to_string(index=False))

    print("\n✅ Phase 3 complete! Evaluation done.")


if __name__ == "__main__":
    main()

