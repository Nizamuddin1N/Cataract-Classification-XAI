"""
dataset.py - PyTorch Dataset class for loading fundus images with transforms.
"""
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import IMAGES_DIR, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD


def get_train_transforms():
    """Training transforms with mild augmentation."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms():
    """Validation/Test transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class CataractDataset(Dataset):
    """
    Custom PyTorch Dataset for cataract vs normal fundus images.

    Args:
        manifest_df: DataFrame with columns [filename, label, split, ...]
        split: 'train', 'val', or 'test'
        images_dir: Path to the images folder
        transform: torchvision transform to apply
    """

    def __init__(self, manifest_df, split, images_dir=IMAGES_DIR, transform=None):
        self.df = manifest_df[manifest_df["split"] == split].reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, row["filename"])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(row["label"], dtype=torch.float32)
        return image, label

    def get_filepath(self, idx):
        """Return the full path of image at index idx."""
        return os.path.join(self.images_dir, self.df.iloc[idx]["filename"])

    def get_info(self, idx):
        """Return metadata dict for image at index idx."""
        row = self.df.iloc[idx]
        return {
            "patient_id": row["patient_id"],
            "eye": row["eye"],
            "filename": row["filename"],
            "label": row["label"],
        }

