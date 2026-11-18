# Cataract Classification from Retinal Fundus Images using Explainable AI

A deep learning system that classifies retinal fundus images as **Cataract** or **Normal** using transfer learning (ResNet50, EfficientNet-B0) with **Grad-CAM** heatmaps to explain model decisions. Built on the **ODIR-5K** dataset.

---

## Project Overview

| Item | Detail |
|------|--------|
| **Title** | Cataract Classification using Explainable AI |
| **Domain** | Medical Image Classification, Explainable AI (XAI) |
| **Dataset** | ODIR-5K (Ocular Disease Intelligent Recognition) |
| **Models** | ResNet50, EfficientNet-B0 (ImageNet pretrained) |
| **Explainability** | Grad-CAM heatmaps |
| **Demo** | Streamlit web application |
| **Language** | Python 3.12 |
| **Framework** | PyTorch 2.6 + CUDA 12.4 |

---

## Project Structure

```
ML_DL_project/
│
├── data/
│   ├── ODIR-5K/                        # ODIR-5K dataset
│   │   ├── preprocessed_images/        # 512×512 fundus images
│   │   ├── ODIR-5K/ODIR-5K/data.xlsx   # Labels spreadsheet
│   │   └── full_df.csv                 # Flattened metadata
│   └── splits/
│       └── manifest.csv                # Train/val/test split with labels
│
├── src/
│   ├── config.py               # All paths and hyperparameters
│   ├── prepare_data.py         # Data preparation (label extraction, splitting)
│   ├── dataset.py              # PyTorch Dataset class with transforms
│   ├── models.py               # ResNet50 & EfficientNet-B0 model definitions
│   ├── train.py                # Training loop with early stopping
│   ├── evaluate.py             # Test evaluation, metrics, plots
│   └── gradcam.py              # Grad-CAM heatmap generation
│
├── app/
│   └── streamlit_app.py        # Interactive demo web application
│
├── outputs/
│   ├── models/                 # Saved model checkpoints (.pth)
│   │   ├── resnet50_best.pth
│   │   ├── efficientnet_b0_best.pth
│   │   └── best_model_info.json
│   ├── plots/                  # All generated plots
│   │   ├── class_distribution.png
│   │   ├── sample_images.png
│   │   ├── training_curves.png
│   │   ├── confusion_matrix_resnet50.png
│   │   ├── confusion_matrix_efficientnet_b0.png
│   │   ├── roc_curves_comparison.png
│   │   ├── pr_curves_comparison.png
│   │   └── metrics_comparison.png
│   ├── gradcam/                # Grad-CAM heatmap images
│   │   ├── gradcam_correct_cataract_1-4.png
│   │   ├── gradcam_correct_normal_1-4.png
│   │   └── gradcam_misclassified_1-4.png
│   └── results/
│       └── test_metrics.csv    # All test metrics
│
├── report/
│   └── main.tex                # LaTeX project report
│
├── requirements.txt
├── main.py                     # Master script to run all phases
└── README.md
```

---

## How to Run

### Prerequisites
- Python 3.12+
- NVIDIA GPU with CUDA (optional, works on CPU too)
- ODIR-5K dataset from [Kaggle](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

For GPU support, install CUDA-enabled PyTorch:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### Step 2: Run All Phases
```bash
python main.py
```

Or run phases individually:
```bash
python -m src.prepare_data     # Phase 1: Data preparation
python -m src.train            # Phase 2: Model training
python -m src.evaluate         # Phase 3: Evaluation
python -m src.gradcam          # Phase 4: Grad-CAM explanations
```

### Step 3: Launch Demo App
```bash
streamlit run app/streamlit_app.py
```

---

## Dataset Summary

The **ODIR-5K** dataset contains 7,000 fundus images (3,500 patients × 2 eyes). After filtering for eyes with exact keywords "cataract" or "normal fundus":

| Split | Patients | Images | Cataract | Normal |
|-------|----------|--------|----------|--------|
| Train | 1,365 | 2,175 | 186 | 1,989 |
| Validation | 293 | 455 | 31 | 424 |
| Test | 293 | 448 | 45 | 403 |
| **Total** | **1,951** | **3,078** | **262** | **2,816** |

---

## Methodology

1. **Data Preparation**: Eye-level labels extracted from diagnostic keywords. Patient-level 70/15/15 split (no data leakage between eyes of the same patient).
2. **Preprocessing**: Resize to 224×224, ImageNet normalization. Training augmentation: random flip, rotation (±10°), color jitter.
3. **Models**: Transfer learning — ImageNet pretrained ResNet50 and EfficientNet-B0 with frozen early layers and custom binary classification head.
4. **Training**: Weighted BCE loss (pos_weight=10.69 for class imbalance), AdamW optimizer (lr=1e-4), ReduceLROnPlateau scheduler, early stopping by validation ROC-AUC.
5. **Evaluation**: Threshold frozen on validation set (maximize cataract F1), then applied once to held-out test set.
6. **Explainability**: Grad-CAM heatmaps on the best model showing which image regions influenced predictions.

---

## Results

### Test Set Performance (448 images: 45 cataract, 403 normal)

| Metric | ResNet50 | EfficientNet-B0 |
|--------|----------|-----------------|
| **Accuracy** | 96.88% | 95.31% |
| **Precision** | 87.80% | 72.22% |
| **Recall (Sensitivity)** | 80.00% | 86.67% |
| **Specificity** | 98.76% | 96.28% |
| **F1 Score** | 83.72% | 78.79% |
| **ROC-AUC** | 0.9676 | 0.9848 |
| **PR-AUC** | 0.8943 | 0.8635 |
| **Threshold** | 0.89 | 0.88 |
| **Best Epoch** | 1 | 12 |

### Confusion Matrices

| | ResNet50 | EfficientNet-B0 |
|--|----------|-----------------|
| True Normal (TN) | 398 | 388 |
| False Cataract (FP) | 5 | 15 |
| Missed Cataract (FN) | 9 | 6 |
| True Cataract (TP) | 36 | 39 |

**Best model selected**: ResNet50 (highest validation ROC-AUC = 0.9953)

---

## Technologies Used

| Component | Technology |
|-----------|-----------|
| Language | Python 3.12 |
| Deep Learning | PyTorch 2.6, torchvision |
| Data Science | NumPy, Pandas, scikit-learn |
| Image Processing | Pillow, OpenCV |
| Visualization | Matplotlib, Seaborn |
| Explainable AI | pytorch-grad-cam |
| Demo App | Streamlit |
| GPU | NVIDIA RTX 2050, CUDA 12.4 |

---

## License

This project is for academic purposes only. The ODIR-5K dataset is subject to its own license terms.
