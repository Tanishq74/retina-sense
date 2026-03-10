#!/usr/bin/env python3
"""
RetinaSense v3.0 -- Phase 1A: Rich Evaluation Dashboard
========================================================
Standalone script that loads the trained ViT model, runs inference on the
full test set (1,281 images), and produces publication-quality evaluation
plots plus a structured metrics JSON report.

Outputs (all written to outputs_v3/evaluation/):
  - confusion_matrix.png
  - roc_curves_per_class.png
  - precision_recall_curves.png
  - calibration_reliability.png
  - confidence_histograms.png
  - error_analysis_by_source.png
  - metrics_report.json

Usage:
  python eval_dashboard.py
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from PIL import Image
from collections import OrderedDict

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    log_loss,
)

# ================================================================
# CONFIGURATION
# ================================================================
BASE_DIR    = '/teamspace/studios/this_studio'
OUTPUT_DIR  = os.path.join(BASE_DIR, 'outputs_v3')
EVAL_DIR    = os.path.join(OUTPUT_DIR, 'evaluation')
os.makedirs(EVAL_DIR, exist_ok=True)

MODEL_PATH       = os.path.join(OUTPUT_DIR, 'best_model.pth')
THRESHOLDS_PATH  = os.path.join(OUTPUT_DIR, 'thresholds.json')
TEMPERATURE_PATH = os.path.join(OUTPUT_DIR, 'temperature.json')
TEST_CSV         = os.path.join(BASE_DIR, 'data', 'test_split.csv')
NORM_STATS_PATH  = os.path.join(BASE_DIR, 'data', 'fundus_norm_stats.json')

NUM_CLASSES = 5
IMG_SIZE    = 224
DROPOUT     = 0.3
BATCH_SIZE  = 32

CLASS_NAMES = ['Normal', 'Diabetes/DR', 'Glaucoma', 'Cataract', 'AMD']

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Publication style defaults
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
    'font.family': 'sans-serif',
})

print('=' * 65)
print('   RetinaSense v3.0 -- Phase 1A: Evaluation Dashboard')
print('=' * 65)
print(f'  Device  : {DEVICE}')
if torch.cuda.is_available():
    print(f'  GPU     : {torch.cuda.get_device_name(0)}')
print(f'  Output  : {EVAL_DIR}')
print('=' * 65)


# ================================================================
# LOAD NORMALISATION STATS
# ================================================================
if os.path.exists(NORM_STATS_PATH):
    with open(NORM_STATS_PATH) as f:
        norm_stats = json.load(f)
    NORM_MEAN = norm_stats['mean_rgb']
    NORM_STD  = norm_stats['std_rgb']
    print(f'  Fundus norm stats loaded: mean={[round(v, 4) for v in NORM_MEAN]}, '
          f'std={[round(v, 4) for v in NORM_STD]}')
else:
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD  = [0.229, 0.224, 0.225]
    print('  Using ImageNet normalisation fallback')


# ================================================================
# MODEL ARCHITECTURE (mirrors retinasense_v3.py / gradcam_v3.py)
# ================================================================
class MultiTaskViT(nn.Module):
    """ViT-Base-Patch16-224 with disease + severity heads."""

    def __init__(self, n_disease=NUM_CLASSES, n_severity=5, drop=DROPOUT):
        super().__init__()
        self.backbone = timm.create_model(
            'vit_base_patch16_224', pretrained=False, num_classes=0
        )
        feat = 768  # CLS token dimension
        self.drop = nn.Dropout(drop)
        self.disease_head = nn.Sequential(
            nn.Linear(feat, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, n_disease),
        )
        self.severity_head = nn.Sequential(
            nn.Linear(feat, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, n_severity),
        )

    def forward(self, x):
        f = self.backbone(x)   # (B, 768) CLS token features
        f = self.drop(f)
        return self.disease_head(f), self.severity_head(f)


# ================================================================
# LOAD MODEL + CALIBRATION ARTIFACTS
# ================================================================
print('\nLoading model...')
model = MultiTaskViT().to(DEVICE)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f'  Loaded: {MODEL_PATH}')
print(f'  Checkpoint epoch: {ckpt.get("epoch", "?") + 1}  '
      f'val_acc={ckpt.get("val_acc", 0):.2f}%')

with open(THRESHOLDS_PATH) as f:
    thr_data = json.load(f)
THRESHOLDS = thr_data['thresholds']

with open(TEMPERATURE_PATH) as f:
    temp_data = json.load(f)
TEMPERATURE = temp_data['temperature']

print(f'  Temperature T = {TEMPERATURE:.4f}')
print(f'  Thresholds    = {[round(t, 3) for t in THRESHOLDS]}')


# ================================================================
# DATASET
# ================================================================
class TestDataset(Dataset):
    """
    Test dataset that loads from preprocessed .npy cache (fast path).
    Falls back to on-the-fly preprocessing if cache is missing.
    """

    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Try cache path first
        cache_fp = row.get('cache_path', '')
        img = None

        if cache_fp and os.path.exists(cache_fp):
            try:
                img = np.load(cache_fp)
            except Exception:
                img = None

        # Fallback: on-the-fly preprocessing
        if img is None:
            image_path = row['image_path']
            if not os.path.isabs(image_path):
                clean = image_path
                while clean.startswith('./') or clean.startswith('.//'):
                    clean = clean[2:] if clean.startswith('./') else clean[3:]
                image_path = os.path.join(BASE_DIR, clean)

            source = row.get('source', 'ODIR')
            try:
                if source == 'APTOS':
                    img = self._ben_graham(image_path)
                else:
                    img = self._clahe_preprocess(image_path)
            except Exception:
                img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

        img_tensor = self.transform(img)
        disease_lbl = int(row['disease_label'])
        source = row.get('source', 'unknown')
        return img_tensor, disease_lbl, source

    @staticmethod
    def _ben_graham(path, sz=IMG_SIZE, sigma=10):
        raw = cv2.imread(path)
        if raw is None:
            raw = np.array(Image.open(path).convert('RGB'))
            raw = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        raw = cv2.resize(raw, (sz, sz))
        raw = cv2.addWeighted(raw, 4, cv2.GaussianBlur(raw, (0, 0), sigma), -4, 128)
        mask = np.zeros(raw.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (sz // 2, sz // 2), int(sz * 0.48), 255, -1)
        return cv2.bitwise_and(raw, raw, mask=mask)

    @staticmethod
    def _clahe_preprocess(path, sz=IMG_SIZE):
        raw = cv2.imread(path)
        if raw is None:
            raw = np.array(Image.open(path).convert('RGB'))
            raw = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
        raw = cv2.resize(raw, (sz, sz))
        lab = cv2.cvtColor(raw, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        raw = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)


val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])

print('\nLoading test set...')
test_df = pd.read_csv(TEST_CSV)
print(f'  Test samples: {len(test_df)}')
print(f'  Sources     : {sorted(test_df["source"].unique())}')
print(f'  Class dist  : {test_df["disease_label"].value_counts().sort_index().to_dict()}')

test_ds = TestDataset(test_df, val_transform)
test_loader = DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=4, pin_memory=True,
)


# ================================================================
# INFERENCE
# ================================================================
print('\nRunning inference on full test set...')
all_logits = []
all_labels = []
all_sources = []

with torch.no_grad():
    for imgs, labels, sources in test_loader:
        imgs = imgs.to(DEVICE)
        disease_logits, _ = model(imgs)
        all_logits.append(disease_logits.cpu())
        all_labels.extend(labels.numpy().tolist())
        all_sources.extend(sources)

all_logits = torch.cat(all_logits, dim=0)  # (N, 5)
all_labels = np.array(all_labels)
all_sources = np.array(all_sources)
N = len(all_labels)
print(f'  Inference complete: {N} samples')

# Temperature-scaled probabilities
probs_calibrated = F.softmax(all_logits / TEMPERATURE, dim=1).numpy()  # (N, 5)
probs_uncalibrated = F.softmax(all_logits, dim=1).numpy()

# Predictions: argmax of calibrated probabilities
preds = np.argmax(probs_calibrated, axis=1)
confidences = np.max(probs_calibrated, axis=1)

correct_mask = (preds == all_labels)
acc = accuracy_score(all_labels, preds)
print(f'  Overall accuracy: {acc:.4f} ({int(acc * N)}/{N})')


# ================================================================
# 1. CONFUSION MATRIX
# ================================================================
print('\n[1/7] Confusion matrix...')
cm = confusion_matrix(all_labels, preds, labels=list(range(NUM_CLASSES)))
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(
    cm_norm, annot=True, fmt='.2f', cmap='Blues',
    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
    linewidths=0.5, linecolor='white',
    cbar_kws={'label': 'Proportion', 'shrink': 0.8},
    ax=ax, vmin=0, vmax=1,
)
# Overlay raw counts in smaller font
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        ax.text(j + 0.5, i + 0.72, f'(n={cm[i, j]})',
                ha='center', va='center', fontsize=7, color='gray')

ax.set_xlabel('Predicted Class')
ax.set_ylabel('True Class')
ax.set_title('Normalized Confusion Matrix (Test Set)')
fig.tight_layout()
fig.savefig(os.path.join(EVAL_DIR, 'confusion_matrix.png'))
plt.close(fig)
print('  Saved confusion_matrix.png')


# ================================================================
# 2. ROC CURVES PER CLASS
# ================================================================
print('[2/7] ROC curves...')
fig, ax = plt.subplots(figsize=(7, 6))
colors = sns.color_palette('tab10', NUM_CLASSES)
all_fpr_tpr = {}
macro_auc_list = []

for i in range(NUM_CLASSES):
    y_true_bin = (all_labels == i).astype(int)
    y_score = probs_calibrated[:, i]
    fpr, tpr, _ = roc_curve(y_true_bin, y_score)
    roc_auc = auc(fpr, tpr)
    macro_auc_list.append(roc_auc)
    all_fpr_tpr[i] = (fpr, tpr)
    ax.plot(fpr, tpr, color=colors[i], lw=2,
            label=f'{CLASS_NAMES[i]} (AUC={roc_auc:.3f})')

# Macro average ROC
mean_fpr = np.linspace(0, 1, 200)
mean_tpr = np.zeros_like(mean_fpr)
for i in range(NUM_CLASSES):
    mean_tpr += np.interp(mean_fpr, all_fpr_tpr[i][0], all_fpr_tpr[i][1])
mean_tpr /= NUM_CLASSES
macro_auc = auc(mean_fpr, mean_tpr)
ax.plot(mean_fpr, mean_tpr, 'k--', lw=2.5,
        label=f'Macro-average (AUC={macro_auc:.3f})')
ax.plot([0, 1], [0, 1], 'k:', lw=1, alpha=0.4)

ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('One-vs-Rest ROC Curves (Calibrated)')
ax.legend(loc='lower right', framealpha=0.9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(EVAL_DIR, 'roc_curves_per_class.png'))
plt.close(fig)
print('  Saved roc_curves_per_class.png')


# ================================================================
# 3. PRECISION-RECALL CURVES
# ================================================================
print('[3/7] Precision-recall curves...')
fig, ax = plt.subplots(figsize=(7, 6))

for i in range(NUM_CLASSES):
    y_true_bin = (all_labels == i).astype(int)
    y_score = probs_calibrated[:, i]
    prec, rec, _ = precision_recall_curve(y_true_bin, y_score)
    ap = average_precision_score(y_true_bin, y_score)
    ax.plot(rec, prec, color=colors[i], lw=2,
            label=f'{CLASS_NAMES[i]} (AP={ap:.3f})')

# Add prevalence baselines
prevalences = np.bincount(all_labels, minlength=NUM_CLASSES) / N
for i in range(NUM_CLASSES):
    ax.axhline(y=prevalences[i], color=colors[i], ls=':', alpha=0.3)

ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.05])
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curves (Calibrated)')
ax.legend(loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(EVAL_DIR, 'precision_recall_curves.png'))
plt.close(fig)
print('  Saved precision_recall_curves.png')


# ================================================================
# 4. CALIBRATION RELIABILITY DIAGRAM
# ================================================================
print('[4/7] Calibration reliability diagram...')
n_bins = 10
bin_edges = np.linspace(0, 1, n_bins + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Compute calibration for both calibrated and uncalibrated probabilities
def compute_calibration(confidences_arr, correct_arr, bin_edges):
    """Compute per-bin accuracy and average confidence."""
    bin_accs = []
    bin_confs = []
    bin_counts = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences_arr > lo) & (confidences_arr <= hi)
        if mask.sum() == 0:
            bin_accs.append(np.nan)
            bin_confs.append(np.nan)
            bin_counts.append(0)
        else:
            bin_accs.append(correct_arr[mask].mean())
            bin_confs.append(confidences_arr[mask].mean())
            bin_counts.append(int(mask.sum()))
    return np.array(bin_accs), np.array(bin_confs), np.array(bin_counts)

conf_calib = np.max(probs_calibrated, axis=1)
conf_uncalib = np.max(probs_uncalibrated, axis=1)

bin_accs_cal, bin_confs_cal, bin_counts_cal = compute_calibration(
    conf_calib, correct_mask.astype(float), bin_edges)
bin_accs_uncal, bin_confs_uncal, bin_counts_uncal = compute_calibration(
    conf_uncalib, correct_mask.astype(float), bin_edges)

# ECE
ece_cal = np.nansum(
    np.abs(bin_accs_cal - bin_confs_cal) * bin_counts_cal) / N
ece_uncal = np.nansum(
    np.abs(bin_accs_uncal - bin_confs_uncal) * bin_counts_uncal) / N

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax_idx, (b_accs, b_confs, b_counts, ece_val, title_suffix) in enumerate([
    (bin_accs_cal, bin_confs_cal, bin_counts_cal, ece_cal, 'Calibrated'),
    (bin_accs_uncal, bin_confs_uncal, bin_counts_uncal, ece_uncal, 'Uncalibrated'),
]):
    ax = axes[ax_idx]
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Perfectly calibrated')
    # Bar chart of bin accuracy
    valid = ~np.isnan(b_accs)
    bar_color = '#4C72B0' if ax_idx == 0 else '#DD8452'
    ax.bar(bin_centers[valid], b_accs[valid], width=0.08,
           alpha=0.7, color=bar_color, edgecolor='black', linewidth=0.5,
           label=f'Model (ECE={ece_val:.4f})')
    # Gap shading
    for j in range(n_bins):
        if valid[j]:
            lo_val = min(b_accs[j], b_confs[j])
            hi_val = max(b_accs[j], b_confs[j])
            ax.fill_between(
                [bin_centers[j] - 0.04, bin_centers[j] + 0.04],
                lo_val, hi_val, alpha=0.15, color='red')
    # Sample counts on top
    for j in range(n_bins):
        if valid[j] and b_counts[j] > 0:
            ax.text(bin_centers[j], b_accs[j] + 0.03,
                    str(b_counts[j]), ha='center', va='bottom', fontsize=7)

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.1])
    ax.set_xlabel('Mean Predicted Confidence')
    ax.set_ylabel('Fraction of Correct Predictions')
    ax.set_title(f'Reliability Diagram ({title_suffix})')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(EVAL_DIR, 'calibration_reliability.png'))
plt.close(fig)
print(f'  Saved calibration_reliability.png  (ECE_cal={ece_cal:.4f}, ECE_uncal={ece_uncal:.4f})')


# ================================================================
# 5. CONFIDENCE HISTOGRAMS
# ================================================================
print('[5/7] Confidence histograms...')
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Correct vs Incorrect
for ax_idx, (mask, label, color) in enumerate([
    (correct_mask, 'Correct', '#2ca02c'),
    (~correct_mask, 'Incorrect', '#d62728'),
]):
    axes[0].hist(confidences[mask], bins=30, alpha=0.65, color=color,
                 label=f'{label} (n={mask.sum()})', edgecolor='black', linewidth=0.3)

axes[0].set_xlabel('Prediction Confidence')
axes[0].set_ylabel('Count')
axes[0].set_title('Confidence Distribution: Correct vs Incorrect')
axes[0].legend(loc='upper left', framealpha=0.9)
axes[0].axvline(x=np.median(confidences[correct_mask]), color='#2ca02c',
                ls='--', alpha=0.6, label='_nolegend_')
axes[0].axvline(x=np.median(confidences[~correct_mask]), color='#d62728',
                ls='--', alpha=0.6, label='_nolegend_')
axes[0].grid(True, alpha=0.3, axis='y')

# Per-class confidence
for i in range(NUM_CLASSES):
    cls_mask = (all_labels == i)
    axes[1].hist(confidences[cls_mask], bins=20, alpha=0.5, color=colors[i],
                 label=f'{CLASS_NAMES[i]} (n={cls_mask.sum()})',
                 edgecolor='black', linewidth=0.3)

axes[1].set_xlabel('Prediction Confidence')
axes[1].set_ylabel('Count')
axes[1].set_title('Confidence Distribution by True Class')
axes[1].legend(loc='upper left', framealpha=0.9, fontsize=9)
axes[1].grid(True, alpha=0.3, axis='y')

fig.tight_layout()
fig.savefig(os.path.join(EVAL_DIR, 'confidence_histograms.png'))
plt.close(fig)
print('  Saved confidence_histograms.png')


# ================================================================
# 6. ERROR ANALYSIS BY SOURCE
# ================================================================
print('[6/7] Error analysis by source...')
sources_unique = sorted(np.unique(all_sources))
n_sources = len(sources_unique)

# Build accuracy per (source, class) pair
source_class_acc = {}
source_class_n = {}
for src in sources_unique:
    for cls_idx in range(NUM_CLASSES):
        mask = (all_sources == src) & (all_labels == cls_idx)
        n_cls = mask.sum()
        if n_cls > 0:
            acc_sc = (preds[mask] == all_labels[mask]).mean()
        else:
            acc_sc = np.nan
        source_class_acc[(src, cls_idx)] = acc_sc
        source_class_n[(src, cls_idx)] = int(n_cls)

# Also overall accuracy per source
source_overall_acc = {}
for src in sources_unique:
    mask = (all_sources == src)
    source_overall_acc[src] = accuracy_score(all_labels[mask], preds[mask])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left panel: grouped bar chart of per-class accuracy by source
x = np.arange(NUM_CLASSES)
bar_width = 0.8 / max(n_sources, 1)
source_colors = sns.color_palette('Set2', n_sources)

for s_idx, src in enumerate(sources_unique):
    accs = [source_class_acc[(src, c)] for c in range(NUM_CLASSES)]
    counts = [source_class_n[(src, c)] for c in range(NUM_CLASSES)]
    offset = (s_idx - n_sources / 2 + 0.5) * bar_width
    bars = axes[0].bar(x + offset, accs, bar_width * 0.9,
                       label=f'{src} (n={sum(counts)})',
                       color=source_colors[s_idx], edgecolor='black', linewidth=0.5)
    # Annotate sample counts
    for j, (b, n_val) in enumerate(zip(bars, counts)):
        if n_val > 0 and not np.isnan(accs[j]):
            axes[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                         str(n_val), ha='center', va='bottom', fontsize=7)

axes[0].set_xticks(x)
axes[0].set_xticklabels(CLASS_NAMES, rotation=15, ha='right')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Per-Class Accuracy by Data Source')
axes[0].set_ylim([0, 1.15])
axes[0].legend(loc='upper right', framealpha=0.9)
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].axhline(y=acc, color='black', ls='--', alpha=0.4, lw=1)
axes[0].text(NUM_CLASSES - 0.5, acc + 0.02, f'Overall: {acc:.3f}',
             ha='right', fontsize=9, alpha=0.6)

# Right panel: confusion breakdown -- most common misclassifications per source
error_data = []
for src in sources_unique:
    src_mask = (all_sources == src) & (~correct_mask)
    if src_mask.sum() == 0:
        continue
    for true_cls in range(NUM_CLASSES):
        for pred_cls in range(NUM_CLASSES):
            if true_cls == pred_cls:
                continue
            pair_mask = src_mask & (all_labels == true_cls) & (preds == pred_cls)
            cnt = pair_mask.sum()
            if cnt > 0:
                error_data.append({
                    'Source': src,
                    'Error': f'{CLASS_NAMES[true_cls][:3]}>{CLASS_NAMES[pred_cls][:3]}',
                    'Count': int(cnt),
                })

if error_data:
    err_df = pd.DataFrame(error_data)
    # Top 10 error types
    top_errors = (err_df.groupby('Error')['Count'].sum()
                  .sort_values(ascending=False).head(10).index.tolist())
    err_df_top = err_df[err_df['Error'].isin(top_errors)]
    pivot = err_df_top.pivot_table(index='Error', columns='Source',
                                   values='Count', aggfunc='sum', fill_value=0)
    # Reorder by total count
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=True).index]
    pivot.plot(kind='barh', stacked=True, ax=axes[1],
               color=source_colors[:n_sources], edgecolor='black', linewidth=0.5)
    axes[1].set_xlabel('Error Count')
    axes[1].set_title('Top Misclassification Patterns by Source')
    axes[1].legend(loc='lower right', framealpha=0.9)
    axes[1].grid(True, alpha=0.3, axis='x')
else:
    axes[1].text(0.5, 0.5, 'No errors to display', ha='center', va='center',
                 transform=axes[1].transAxes, fontsize=14)
    axes[1].set_title('Top Misclassification Patterns by Source')

fig.tight_layout()
fig.savefig(os.path.join(EVAL_DIR, 'error_analysis_by_source.png'))
plt.close(fig)
print('  Saved error_analysis_by_source.png')


# ================================================================
# 7. METRICS REPORT (JSON)
# ================================================================
print('[7/7] Metrics report...')

# Classification report as dict
cls_report = classification_report(
    all_labels, preds, target_names=CLASS_NAMES,
    output_dict=True, zero_division=0)

# Per-class AUC and AP
per_class_auc = {}
per_class_ap = {}
for i in range(NUM_CLASSES):
    y_bin = (all_labels == i).astype(int)
    y_score = probs_calibrated[:, i]
    fpr_i, tpr_i, _ = roc_curve(y_bin, y_score)
    per_class_auc[CLASS_NAMES[i]] = float(auc(fpr_i, tpr_i))
    per_class_ap[CLASS_NAMES[i]] = float(average_precision_score(y_bin, y_score))

# Build the full report
try:
    ll = float(log_loss(all_labels, probs_calibrated))
except Exception:
    ll = None

metrics_report = OrderedDict([
    ('n_test_samples', int(N)),
    ('overall_accuracy', float(acc)),
    ('balanced_accuracy', float(balanced_accuracy_score(all_labels, preds))),
    ('macro_f1', float(f1_score(all_labels, preds, average='macro', zero_division=0))),
    ('weighted_f1', float(f1_score(all_labels, preds, average='weighted', zero_division=0))),
    ('cohen_kappa', float(cohen_kappa_score(all_labels, preds))),
    ('matthews_corrcoef', float(matthews_corrcoef(all_labels, preds))),
    ('log_loss', ll),
    ('macro_auc', float(np.mean(list(per_class_auc.values())))),
    ('ece_calibrated', float(ece_cal)),
    ('ece_uncalibrated', float(ece_uncal)),
    ('temperature', float(TEMPERATURE)),
    ('thresholds', THRESHOLDS),
    ('per_class_metrics', {}),
    ('per_class_auc', per_class_auc),
    ('per_class_ap', per_class_ap),
    ('confusion_matrix_raw', cm.tolist()),
    ('confusion_matrix_normalized', np.round(cm_norm, 4).tolist()),
    ('source_accuracy', {src: float(v) for src, v in source_overall_acc.items()}),
    ('source_class_counts', {
        src: {CLASS_NAMES[c]: source_class_n[(src, c)]
              for c in range(NUM_CLASSES)}
        for src in sources_unique
    }),
    ('class_names', CLASS_NAMES),
])

# Per-class from classification_report
for i, name in enumerate(CLASS_NAMES):
    metrics_report['per_class_metrics'][name] = {
        'precision': float(cls_report[name]['precision']),
        'recall': float(cls_report[name]['recall']),
        'f1-score': float(cls_report[name]['f1-score']),
        'support': int(cls_report[name]['support']),
        'auc': per_class_auc[name],
        'average_precision': per_class_ap[name],
    }

report_path = os.path.join(EVAL_DIR, 'metrics_report.json')
with open(report_path, 'w') as f:
    json.dump(metrics_report, f, indent=2)
print(f'  Saved metrics_report.json')


# ================================================================
# SUMMARY
# ================================================================
print('\n' + '=' * 65)
print('  EVALUATION DASHBOARD COMPLETE')
print('=' * 65)
print(f'  Overall Accuracy    : {acc:.4f}')
print(f'  Balanced Accuracy   : {metrics_report["balanced_accuracy"]:.4f}')
print(f'  Macro F1            : {metrics_report["macro_f1"]:.4f}')
print(f'  Cohen Kappa         : {metrics_report["cohen_kappa"]:.4f}')
print(f'  Macro AUC           : {metrics_report["macro_auc"]:.4f}')
print(f'  ECE (calibrated)    : {ece_cal:.4f}')
print(f'  ECE (uncalibrated)  : {ece_uncal:.4f}')
print(f'\n  Per-class AUC:')
for name, val in per_class_auc.items():
    print(f'    {name:15s} : {val:.4f}')
print(f'\n  Source accuracy:')
for src, val in source_overall_acc.items():
    print(f'    {src:10s} : {val:.4f}')
print(f'\n  All outputs in: {EVAL_DIR}/')
output_files = [
    'confusion_matrix.png',
    'roc_curves_per_class.png',
    'precision_recall_curves.png',
    'calibration_reliability.png',
    'confidence_histograms.png',
    'error_analysis_by_source.png',
    'metrics_report.json',
]
for fname in output_files:
    fpath = os.path.join(EVAL_DIR, fname)
    exists = os.path.exists(fpath)
    size_kb = os.path.getsize(fpath) / 1024 if exists else 0
    status = f'{size_kb:.0f} KB' if exists else 'MISSING'
    print(f'    [{status:>8s}] {fname}')
print('=' * 65)
