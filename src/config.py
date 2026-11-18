"""
config.py - Central configuration for the Cataract Classification project.
All paths, hyperparameters, and settings are defined here.
"""
import os

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset paths
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "ODIR-5K")
EXCEL_PATH = os.path.join(DATA_DIR, "ODIR-5K", "ODIR-5K", "data.xlsx")
IMAGES_DIR = os.path.join(DATA_DIR, "preprocessed_images")  # 512x512 pre-resized

# Output paths
SPLITS_DIR = os.path.join(PROJECT_ROOT, "data", "splits")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
GRADCAM_DIR = os.path.join(OUTPUT_DIR, "gradcam")

# Ensure directories exist
for d in [SPLITS_DIR, MODELS_DIR, PLOTS_DIR, GRADCAM_DIR]:
    os.makedirs(d, exist_ok=True)

# ──────────────────────────────────────────────
# Data settings
# ──────────────────────────────────────────────
IMG_SIZE = 224                 # Resize all images to 224x224
SEED = 42                     # Reproducibility seed
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ──────────────────────────────────────────────
# Training hyperparameters
# ──────────────────────────────────────────────
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
MAX_EPOCHS = 15
EARLY_STOP_PATIENCE = 5       # Stop if val ROC-AUC doesn't improve
LR_PATIENCE = 2               # Halve LR after 2 stalled epochs
LR_MIN = 1e-6
NUM_WORKERS = 0                # Windows compatibility

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ──────────────────────────────────────────────
# Model names to train
# ──────────────────────────────────────────────
MODEL_NAMES = ["resnet50", "efficientnet_b0"]
