"""
models.py - ResNet50 and EfficientNet-B0 for binary cataract classification.
Both use ImageNet pretrained weights with a custom single-output head.
"""
import torch
import torch.nn as nn
from torchvision import models


def build_model(name, freeze_early=True):
    """
    Build a classification model for binary cataract detection.

    Args:
        name: 'resnet50' or 'efficientnet_b0'
        freeze_early: If True, freeze all layers except the last block + classifier.

    Returns:
        model: PyTorch model with 1 logit output
        target_layer: The layer to use for Grad-CAM
    """
    if name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        if freeze_early:
            # Freeze everything except layer4 and fc
            for param in model.parameters():
                param.requires_grad = False
            for param in model.layer4.parameters():
                param.requires_grad = True

        # Replace classifier: 2048 → 1 (binary logit)
        model.fc = nn.Linear(model.fc.in_features, 1)
        target_layer = model.layer4[-1]  # Last block of layer4

    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

        if freeze_early:
            # Freeze everything except the last feature block and classifier
            for param in model.parameters():
                param.requires_grad = False
            for param in model.features[-1].parameters():
                param.requires_grad = True

        # Replace classifier: 1280 → 1
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1280, 1),
        )
        target_layer = model.features[-1]  # Last feature block

    else:
        raise ValueError(f"Unknown model: {name}. Use 'resnet50' or 'efficientnet_b0'.")

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  {name}: {total:,} total params, {trainable:,} trainable ({100*trainable/total:.1f}%)")

    return model, target_layer


def get_target_layer(model, name):
    """Return the Grad-CAM target layer for a given model."""
    if name == "resnet50":
        return model.layer4[-1]
    elif name == "efficientnet_b0":
        return model.features[-1]
    else:
        raise ValueError(f"Unknown model: {name}")

