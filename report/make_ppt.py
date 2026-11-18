"""
Generate a simple 7-slide PPT for the Cataract Classification project.
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
import os

PROJECT = r"c:\Users\Nizamuddin2004\OneDrive\Desktop\ML_DL_project"
PLOTS = os.path.join(PROJECT, "outputs", "plots")
GRADCAM = os.path.join(PROJECT, "outputs", "gradcam")
OUTPUT_PPT = os.path.join(PROJECT, "report", "presentation.pptx")

# Colors
WHITE = RGBColor(255, 255, 255)
BLACK = RGBColor(0, 0, 0)
DARK_GRAY = RGBColor(50, 50, 50)
MEDIUM_GRAY = RGBColor(100, 100, 100)
LIGHT_BG = RGBColor(245, 245, 245)
ACCENT = RGBColor(25, 60, 120)  # Dark navy blue

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)


def add_bg(slide, color=WHITE):
    """Set slide background color."""
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_text(slide, left, top, width, height, text, font_size=18, bold=False,
             color=DARK_GRAY, alignment=PP_ALIGN.LEFT, font_name="Calibri"):
    """Add a text box to a slide."""
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.font.name = font_name
    p.alignment = alignment
    return tf


def add_bullet_text(slide, left, top, width, height, items, font_size=16,
                    color=DARK_GRAY, spacing=Pt(8)):
    """Add bulleted text."""
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = item
        p.font.size = Pt(font_size)
        p.font.color.rgb = color
        p.font.name = "Calibri"
        p.space_after = spacing
        p.level = 0
    return tf


def add_line(slide, left, top, width, color=ACCENT, thickness=Pt(2)):
    """Add a horizontal line."""
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(left), Inches(top), Inches(width), thickness
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()


# ════════════════════════════════════════════════════════════
# SLIDE 1: TITLE PAGE
# ════════════════════════════════════════════════════════════
slide1 = prs.slides.add_slide(prs.slide_layouts[6])  # Blank
add_bg(slide1, WHITE)

# University name at top
add_text(slide1, 0, 0.3, 13.333, 0.5, "FACULTY OF TECHNOLOGY, UNIVERSITY OF DELHI",
         font_size=16, bold=True, color=ACCENT, alignment=PP_ALIGN.CENTER)
add_text(slide1, 0, 0.7, 13.333, 0.4, "NEW DELHI – 110007",
         font_size=13, color=MEDIUM_GRAY, alignment=PP_ALIGN.CENTER)

add_line(slide1, 1.5, 1.2, 10.333, ACCENT)

# Subject
add_text(slide1, 0, 1.5, 13.333, 0.5, "ML AND DL FOR 3D DATA — Project Presentation",
         font_size=15, color=MEDIUM_GRAY, alignment=PP_ALIGN.CENTER)

# Title
add_text(slide1, 1, 2.3, 11.333, 1.2,
         "Cataract Classification from Retinal\nFundus Images using Explainable AI",
         font_size=32, bold=True, color=ACCENT, alignment=PP_ALIGN.CENTER)

add_line(slide1, 3, 3.9, 7.333, ACCENT)

# Students
add_text(slide1, 1.5, 4.3, 5, 0.4, "Submitted By:", font_size=14, bold=True, color=ACCENT)
students = [
    "Avi Pandey (23293916114)",
    "Nizamuddin (23293916132)",
    "Ayush Kumar Singh (23293916072)",
    "Animesh Kumar (23293916065)",
    "Anish Kumar (23293916096)",
]
add_bullet_text(slide1, 1.5, 4.7, 5, 2.2, students, font_size=13, color=DARK_GRAY, spacing=Pt(4))

# Guide
add_text(slide1, 7.5, 4.3, 5, 0.4, "Submitted To:", font_size=14, bold=True, color=ACCENT)
add_bullet_text(slide1, 7.5, 4.7, 5, 1, ["Mr. Sumit Kumar"], font_size=13, color=DARK_GRAY)

# Footer
add_text(slide1, 0, 6.8, 13.333, 0.4,
         "B.Tech. CSE  |  Semester VII  |  CSE B  |  2026–2027",
         font_size=12, color=MEDIUM_GRAY, alignment=PP_ALIGN.CENTER)


# ════════════════════════════════════════════════════════════
# SLIDE 2: INTRODUCTION & OBJECTIVES
# ════════════════════════════════════════════════════════════
slide2 = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide2, WHITE)

add_text(slide2, 0.6, 0.3, 12, 0.6, "Introduction & Objectives", font_size=28, bold=True, color=ACCENT)
add_line(slide2, 0.6, 0.85, 5, ACCENT)

# Problem
add_text(slide2, 0.6, 1.2, 6, 0.4, "Problem", font_size=18, bold=True, color=ACCENT)
add_bullet_text(slide2, 0.6, 1.6, 6, 2.5, [
    "Cataract is a leading cause of avoidable blindness worldwide.",
    "Manual screening by ophthalmologists is time-consuming and resource-intensive.",
    "Fundus cameras capture retinal images through the lens — cataract appears as haze, reduced contrast, and uneven illumination.",
    "Need: An automated, explainable AI system for cataract screening.",
], font_size=14, color=DARK_GRAY, spacing=Pt(6))

# Objectives
add_text(slide2, 0.6, 3.8, 6, 0.4, "Objectives", font_size=18, bold=True, color=ACCENT)
add_bullet_text(slide2, 0.6, 4.2, 6, 3, [
    "Build a binary classifier (Cataract vs Normal) from ODIR-5K fundus images.",
    "Train and compare ResNet50 and EfficientNet-B0 using transfer learning.",
    "Handle severe class imbalance (8.5% cataract) with weighted loss.",
    "Apply Grad-CAM to visually explain model predictions.",
    "Develop a Streamlit web demo for interactive classification.",
], font_size=14, color=DARK_GRAY, spacing=Pt(6))

# Sample images on right
try:
    slide2.shapes.add_picture(
        os.path.join(PLOTS, "sample_images.png"),
        Inches(7.2), Inches(1.2), Inches(5.5)
    )
except:
    pass


# ════════════════════════════════════════════════════════════
# SLIDE 3: DATASET & METHODOLOGY
# ════════════════════════════════════════════════════════════
slide3 = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide3, WHITE)

add_text(slide3, 0.6, 0.3, 12, 0.6, "Dataset & Methodology", font_size=28, bold=True, color=ACCENT)
add_line(slide3, 0.6, 0.85, 5, ACCENT)

# Dataset info
add_text(slide3, 0.6, 1.2, 5.5, 0.4, "ODIR-5K Dataset", font_size=18, bold=True, color=ACCENT)
add_bullet_text(slide3, 0.6, 1.6, 5.5, 2, [
    "7,000 fundus images from 3,500 patients (left + right eye).",
    "Filtered to 3,078 images: 262 cataract, 2,816 normal.",
    "Patient-level 70/15/15 split (no data leakage).",
    "Train: 2,175  |  Val: 455  |  Test: 448 images.",
], font_size=14, color=DARK_GRAY, spacing=Pt(6))

# Methodology
add_text(slide3, 0.6, 3.5, 5.5, 0.4, "Methodology", font_size=18, bold=True, color=ACCENT)
add_bullet_text(slide3, 0.6, 3.9, 5.5, 3, [
    "Preprocessing: Resize to 224×224, ImageNet normalization.",
    "Augmentation: Random flip, rotation (±10°), color jitter.",
    "Models: ResNet50 and EfficientNet-B0 (ImageNet pretrained).",
    "Loss: Weighted BCE (pos_weight = 10.69 for imbalance).",
    "Optimizer: AdamW (lr=1e-4), early stopping on val ROC-AUC.",
    "Threshold: Frozen on validation set (maximize cataract F1).",
], font_size=14, color=DARK_GRAY, spacing=Pt(6))

# Class distribution image on right
try:
    slide3.shapes.add_picture(
        os.path.join(PLOTS, "class_distribution.png"),
        Inches(6.8), Inches(1.2), Inches(6)
    )
except:
    pass


# ════════════════════════════════════════════════════════════
# SLIDE 4: TRAINING RESULTS
# ════════════════════════════════════════════════════════════
slide4 = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide4, WHITE)

add_text(slide4, 0.6, 0.3, 12, 0.6, "Training Results", font_size=28, bold=True, color=ACCENT)
add_line(slide4, 0.6, 0.85, 5, ACCENT)

# Summary text
add_bullet_text(slide4, 0.6, 1.2, 5, 2.5, [
    "ResNet50: Best at epoch 1, early-stopped at epoch 6. Val AUC = 0.9953.",
    "EfficientNet-B0: Best at epoch 12, ran 15 epochs. Val AUC = 0.9893.",
    "ResNet50 converges fast due to 15M trainable params in layer4.",
    "EfficientNet-B0 trains more stably with only 0.41M trainable params.",
], font_size=14, color=DARK_GRAY, spacing=Pt(8))

# Training curves image
try:
    slide4.shapes.add_picture(
        os.path.join(PLOTS, "training_curves.png"),
        Inches(0.6), Inches(3.3), Inches(12)
    )
except:
    pass


# ════════════════════════════════════════════════════════════
# SLIDE 5: TEST RESULTS & COMPARISON
# ════════════════════════════════════════════════════════════
slide5 = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide5, WHITE)

add_text(slide5, 0.6, 0.3, 12, 0.6, "Test Results & Model Comparison", font_size=28, bold=True, color=ACCENT)
add_line(slide5, 0.6, 0.85, 5, ACCENT)

# Results table as text
add_text(slide5, 0.6, 1.1, 6, 0.3, "Test Set: 448 images (45 cataract, 403 normal)",
         font_size=13, color=MEDIUM_GRAY)

# Table
table_data = [
    ["Metric", "ResNet50", "EfficientNet-B0"],
    ["Accuracy", "96.88%", "95.31%"],
    ["Precision", "87.80%", "72.22%"],
    ["Recall", "80.00%", "86.67%"],
    ["Specificity", "98.76%", "96.28%"],
    ["F1 Score", "83.72%", "78.79%"],
    ["ROC-AUC", "0.9676", "0.9848"],
    ["PR-AUC", "0.8943", "0.8635"],
]

table = slide5.shapes.add_table(len(table_data), 3, Inches(0.6), Inches(1.5), Inches(5), Inches(3.5)).table
table.columns[0].width = Inches(1.8)
table.columns[1].width = Inches(1.6)
table.columns[2].width = Inches(1.6)

for row_idx, row_data in enumerate(table_data):
    for col_idx, cell_text in enumerate(row_data):
        cell = table.cell(row_idx, col_idx)
        cell.text = cell_text
        for paragraph in cell.text_frame.paragraphs:
            paragraph.font.size = Pt(12)
            paragraph.font.name = "Calibri"
            paragraph.alignment = PP_ALIGN.CENTER
            if row_idx == 0:
                paragraph.font.bold = True
                paragraph.font.color.rgb = WHITE
            else:
                paragraph.font.color.rgb = DARK_GRAY
        if row_idx == 0:
            cell.fill.solid()
            cell.fill.fore_color.rgb = ACCENT

# Confusion matrices + ROC on right
try:
    slide5.shapes.add_picture(
        os.path.join(PLOTS, "confusion_matrix_resnet50.png"),
        Inches(6.2), Inches(1.1), Inches(3.3)
    )
    slide5.shapes.add_picture(
        os.path.join(PLOTS, "roc_curves_comparison.png"),
        Inches(9.5), Inches(1.1), Inches(3.3)
    )
    slide5.shapes.add_picture(
        os.path.join(PLOTS, "confusion_matrix_efficientnet_b0.png"),
        Inches(6.2), Inches(4.2), Inches(3.3)
    )
    slide5.shapes.add_picture(
        os.path.join(PLOTS, "metrics_comparison.png"),
        Inches(9.5), Inches(4.2), Inches(3.3)
    )
except:
    pass


# ════════════════════════════════════════════════════════════
# SLIDE 6: GRAD-CAM EXPLANATIONS
# ════════════════════════════════════════════════════════════
slide6 = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide6, WHITE)

add_text(slide6, 0.6, 0.3, 12, 0.6, "Explainable AI — Grad-CAM Heatmaps", font_size=28, bold=True, color=ACCENT)
add_line(slide6, 0.6, 0.85, 5, ACCENT)

add_bullet_text(slide6, 0.6, 1.1, 12, 0.8, [
    "Grad-CAM highlights which image regions most influenced the model's prediction.",
    "Each panel: Original fundus image → Grad-CAM heatmap → Overlay on the original.",
], font_size=13, color=DARK_GRAY, spacing=Pt(4))

# Grad-CAM images — 2 cataract, 2 normal
try:
    slide6.shapes.add_picture(
        os.path.join(GRADCAM, "gradcam_correct_cataract_1.png"),
        Inches(0.3), Inches(2.0), Inches(6.3)
    )
    slide6.shapes.add_picture(
        os.path.join(GRADCAM, "gradcam_correct_cataract_2.png"),
        Inches(6.8), Inches(2.0), Inches(6.3)
    )
    slide6.shapes.add_picture(
        os.path.join(GRADCAM, "gradcam_correct_normal_1.png"),
        Inches(0.3), Inches(4.7), Inches(6.3)
    )
    slide6.shapes.add_picture(
        os.path.join(GRADCAM, "gradcam_correct_normal_2.png"),
        Inches(6.8), Inches(4.7), Inches(6.3)
    )
except:
    pass


# ════════════════════════════════════════════════════════════
# SLIDE 7: CONCLUSION & FUTURE SCOPE
# ════════════════════════════════════════════════════════════
slide7 = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide7, WHITE)

add_text(slide7, 0.6, 0.3, 12, 0.6, "Conclusion & Future Scope", font_size=28, bold=True, color=ACCENT)
add_line(slide7, 0.6, 0.85, 5, ACCENT)

add_text(slide7, 0.6, 1.2, 6, 0.4, "Conclusion", font_size=18, bold=True, color=ACCENT)
add_bullet_text(slide7, 0.6, 1.6, 6, 2.5, [
    "ResNet50 selected as deployed model (val ROC-AUC = 0.9953).",
    "Test accuracy: 96.88%, F1: 83.72%, only 5 false positives out of 403 normal eyes.",
    "Grad-CAM confirms the model focuses on clinically relevant fundus regions (central haze, contrast changes).",
    "Patient-level splitting and weighted loss ensure methodological rigor.",
    "Streamlit app provides interactive classification with real-time Grad-CAM overlay.",
], font_size=14, color=DARK_GRAY, spacing=Pt(6))

add_text(slide7, 0.6, 4.2, 6, 0.4, "Future Scope", font_size=18, bold=True, color=ACCENT)
add_bullet_text(slide7, 0.6, 4.6, 6, 2.5, [
    "Multi-class: Extend to glaucoma, diabetic retinopathy, AMD.",
    "Cataract grading: Classify severity (mild / moderate / severe).",
    "External validation on different hospital fundus cameras.",
    "Mobile deployment via ONNX / TFLite conversion.",
], font_size=14, color=DARK_GRAY, spacing=Pt(6))

# References
add_text(slide7, 7, 1.2, 5.5, 0.4, "Key References", font_size=18, bold=True, color=ACCENT)
add_bullet_text(slide7, 7, 1.6, 5.8, 3.5, [
    "[1] He et al., \"Deep Residual Learning,\" CVPR 2016.",
    "[2] Tan & Le, \"EfficientNet,\" ICML 2019.",
    "[3] Selvaraju et al., \"Grad-CAM,\" ICCV 2017.",
    "[4] Li et al., \"ODIR Benchmark,\" arXiv 2021.",
    "[5] Loshchilov & Hutter, \"AdamW,\" ICLR 2019.",
], font_size=12, color=MEDIUM_GRAY, spacing=Pt(4))

# Tech stack
add_text(slide7, 7, 4.2, 5.5, 0.4, "Technologies Used", font_size=18, bold=True, color=ACCENT)
add_bullet_text(slide7, 7, 4.6, 5.5, 2.5, [
    "Python 3.12, PyTorch 2.6, CUDA 12.4",
    "torchvision, scikit-learn, Matplotlib",
    "pytorch-grad-cam, Streamlit",
    "Dataset: ODIR-5K (Kaggle)",
], font_size=12, color=MEDIUM_GRAY, spacing=Pt(4))

# Thank you
add_text(slide7, 0, 6.6, 13.333, 0.6, "Thank You",
         font_size=22, bold=True, color=ACCENT, alignment=PP_ALIGN.CENTER)


# ── Save ──
prs.save(OUTPUT_PPT)
print(f"PPT saved: {OUTPUT_PPT}")

