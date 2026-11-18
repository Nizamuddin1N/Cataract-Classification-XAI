"""
prepare_data.py - Reads ODIR-5K Excel, extracts cataract/normal eyes,
performs patient-level train/val/test split, and saves the manifest CSV.
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    EXCEL_PATH, IMAGES_DIR, SPLITS_DIR, PLOTS_DIR,
    SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
)


def build_eye_level_dataframe(excel_path):
    """
    Read the ODIR Excel workbook (1 row per patient) and create
    an eye-level dataframe (1 row per eye image).
    Only keep eyes with keyword exactly 'cataract' or 'normal fundus'.
    """
    df = pd.read_excel(excel_path)
    print(f"Loaded {len(df)} patients from Excel.")

    records = []
    for _, row in df.iterrows():
        patient_id = row["ID"]
        age = row["Patient Age"]
        sex = row["Patient Sex"]

        # Process left eye
        left_file = row["Left-Fundus"]
        left_kw = str(row["Left-Diagnostic Keywords"]).strip().lower()
        if left_kw == "cataract":
            records.append({
                "patient_id": patient_id, "age": age, "sex": sex,
                "filename": left_file, "eye": "left",
                "keyword": left_kw, "label": 1  # 1 = cataract
            })
        elif left_kw == "normal fundus":
            records.append({
                "patient_id": patient_id, "age": age, "sex": sex,
                "filename": left_file, "eye": "left",
                "keyword": left_kw, "label": 0  # 0 = normal
            })

        # Process right eye
        right_file = row["Right-Fundus"]
        right_kw = str(row["Right-Diagnostic Keywords"]).strip().lower()
        if right_kw == "cataract":
            records.append({
                "patient_id": patient_id, "age": age, "sex": sex,
                "filename": right_file, "eye": "right",
                "keyword": right_kw, "label": 1
            })
        elif right_kw == "normal fundus":
            records.append({
                "patient_id": patient_id, "age": age, "sex": sex,
                "filename": right_file, "eye": "right",
                "keyword": right_kw, "label": 0
            })

    eye_df = pd.DataFrame(records)
    print(f"Eye-level records: {len(eye_df)}")
    print(f"  Cataract: {(eye_df['label'] == 1).sum()}")
    print(f"  Normal:   {(eye_df['label'] == 0).sum()}")
    return eye_df


def verify_images_exist(eye_df, images_dir):
    """Check that every image file exists. Drop missing ones."""
    exists = []
    missing = 0
    for _, row in eye_df.iterrows():
        path = os.path.join(images_dir, row["filename"])
        if os.path.isfile(path):
            exists.append(True)
        else:
            exists.append(False)
            missing += 1
    if missing > 0:
        print(f"WARNING: {missing} images not found, dropping them.")
    eye_df = eye_df[exists].reset_index(drop=True)
    return eye_df


def patient_level_split(eye_df, seed=42):
    """
    Split by PATIENT so both eyes of a patient stay in the same set.
    70% train, 15% val, 15% test.
    """
    patients = eye_df["patient_id"].unique()
    np.random.seed(seed)
    np.random.shuffle(patients)

    # First split: 70% train, 30% remaining
    train_pats, temp_pats = train_test_split(
        patients, test_size=(1 - TRAIN_RATIO), random_state=seed
    )
    # Second split: 50/50 of remaining → 15% val, 15% test
    val_pats, test_pats = train_test_split(
        temp_pats, test_size=0.5, random_state=seed
    )

    eye_df = eye_df.copy()
    eye_df["split"] = "train"
    eye_df.loc[eye_df["patient_id"].isin(val_pats), "split"] = "val"
    eye_df.loc[eye_df["patient_id"].isin(test_pats), "split"] = "test"

    # Verify no patient leaks
    train_set = set(eye_df[eye_df["split"] == "train"]["patient_id"])
    val_set = set(eye_df[eye_df["split"] == "val"]["patient_id"])
    test_set = set(eye_df[eye_df["split"] == "test"]["patient_id"])
    assert len(train_set & val_set) == 0, "Patient leak: train ∩ val"
    assert len(train_set & test_set) == 0, "Patient leak: train ∩ test"
    assert len(val_set & test_set) == 0, "Patient leak: val ∩ test"

    return eye_df


def print_split_summary(eye_df):
    """Print a nice summary table of the splits."""
    print("\n" + "=" * 55)
    print("DATASET SPLIT SUMMARY")
    print("=" * 55)
    summary = eye_df.groupby("split").agg(
        patients=("patient_id", "nunique"),
        images=("filename", "count"),
        cataract=("label", "sum"),
    ).reindex(["train", "val", "test"])
    summary["normal"] = summary["images"] - summary["cataract"]
    summary["cataract"] = summary["cataract"].astype(int)
    print(summary.to_string())
    print("=" * 55)


def plot_class_distribution(eye_df, plots_dir):
    """Plot and save class distribution charts."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Overall distribution
    counts = eye_df["label"].value_counts().sort_index()
    labels = ["Normal", "Cataract"]
    colors = ["#2ecc71", "#e74c3c"]
    axes[0].bar(labels, [counts[0], counts[1]], color=colors, edgecolor="black")
    axes[0].set_title("Overall Class Distribution", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Number of Images")
    for i, v in enumerate([counts[0], counts[1]]):
        axes[0].text(i, v + 20, str(v), ha="center", fontweight="bold")

    # Per-split distribution
    split_data = eye_df.groupby(["split", "label"]).size().unstack(fill_value=0)
    split_data = split_data.reindex(["train", "val", "test"])
    split_data.columns = ["Normal", "Cataract"]
    split_data.plot(kind="bar", ax=axes[1], color=colors, edgecolor="black")
    axes[1].set_title("Class Distribution per Split", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Number of Images")
    axes[1].set_xticklabels(["Train", "Validation", "Test"], rotation=0)
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(plots_dir, "class_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_sample_images(eye_df, images_dir, plots_dir):
    """Show sample fundus images from each class."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for row_idx, (label, title) in enumerate([(0, "Normal"), (1, "Cataract")]):
        samples = eye_df[eye_df["label"] == label].sample(4, random_state=42)
        for col_idx, (_, sample) in enumerate(samples.iterrows()):
            img_path = os.path.join(images_dir, sample["filename"])
            img = Image.open(img_path).convert("RGB")
            axes[row_idx, col_idx].imshow(img)
            axes[row_idx, col_idx].set_title(
                f"{title}\nPatient {sample['patient_id']}, {sample['eye']}",
                fontsize=10
            )
            axes[row_idx, col_idx].axis("off")

    plt.suptitle("Sample Fundus Images", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(plots_dir, "sample_images.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def main():
    print("=" * 55)
    print("PHASE 1: DATA PREPARATION")
    print("=" * 55)

    # Step 1: Build eye-level dataframe
    print("\n[Step 1] Building eye-level dataframe from Excel...")
    eye_df = build_eye_level_dataframe(EXCEL_PATH)

    # Step 2: Verify images exist
    print("\n[Step 2] Verifying image files exist...")
    eye_df = verify_images_exist(eye_df, IMAGES_DIR)
    print(f"Verified: {len(eye_df)} images found.")

    # Step 3: Patient-level split
    print("\n[Step 3] Performing patient-level train/val/test split...")
    eye_df = patient_level_split(eye_df, seed=SEED)
    print_split_summary(eye_df)

    # Step 4: Save manifest
    manifest_path = os.path.join(SPLITS_DIR, "manifest.csv")
    eye_df.to_csv(manifest_path, index=False)
    print(f"\nSaved manifest: {manifest_path}")

    # Step 5: Compute class weight for training
    train_df = eye_df[eye_df["split"] == "train"]
    n_normal = (train_df["label"] == 0).sum()
    n_cataract = (train_df["label"] == 1).sum()
    pos_weight = n_normal / n_cataract
    print(f"\nClass weight (pos_weight for BCE): {pos_weight:.4f}")
    print(f"  (Training has {n_normal} normal vs {n_cataract} cataract)")

    # Step 6: Generate plots
    print("\n[Step 6] Generating plots...")
    plot_class_distribution(eye_df, PLOTS_DIR)
    plot_sample_images(eye_df, IMAGES_DIR, PLOTS_DIR)

    print("\n✅ Phase 1 complete! Data is ready for training.")
    return eye_df


if __name__ == "__main__":
    main()

