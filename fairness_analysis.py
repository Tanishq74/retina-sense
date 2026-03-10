#!/usr/bin/env python3
"""
RetinaSense v3.0 — Phase 1D: Fairness & Domain Robustness Analysis
===================================================================
Evaluates model performance across data sources (APTOS, ODIR, REFUGE2)
to quantify domain gap and identify fairness concerns.

Outputs (saved to outputs_v3/fairness/):
  - performance_by_source.png   : grouped bar chart of metrics per class per source
  - calibration_by_source.png   : reliability diagrams by source
  - confusion_matrix_aptos.png  : confusion matrix for APTOS subset
  - confusion_matrix_odir.png   : confusion matrix for ODIR subset
  - confidence_by_source.png    : violin plots of prediction confidence
  - error_patterns.png          : most common misclassification pairs by source
  - domain_gap_report.json      : full quantitative report
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
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from collections import Counter, defaultdict

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import timm

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from scipy import stats

# ================================================================
# CONFIGURATION
# ================================================================
BASE_DIR = '/teamspace/studios/this_studio'
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs_v3')
FAIRNESS_DIR = os.path.join(OUTPUT_DIR, 'fairness')
os.makedirs(FAIRNESS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(OUTPUT_DIR, 'best_model.pth')
TEST_CSV = os.path.join(BASE_DIR, 'data', 'test_split.csv')
NORM_STATS_PATH = os.path.join(BASE_DIR, 'data', 'fundus_norm_stats.json')
TEMPERATURE_PATH = os.path.join(OUTPUT_DIR, 'temperature.json')

NUM_CLASSES = 5
IMG_SIZE = 224
DROPOUT = 0.3
BATCH_SIZE = 32

CLASS_NAMES = ['Normal', 'DR', 'Glaucoma', 'Cataract', 'AMD']

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('=' * 65)
print('   RetinaSense v3.0 -- Phase 1D: Fairness & Domain Robustness')
print('=' * 65)
print(f'  Device   : {DEVICE}')
if torch.cuda.is_available():
    print(f'  GPU      : {torch.cuda.get_device_name(0)}')
print(f'  Output   : {FAIRNESS_DIR}')
print('=' * 65)

# ================================================================
# LOAD NORMALISATION STATS
# ================================================================
if os.path.exists(NORM_STATS_PATH):
    with open(NORM_STATS_PATH) as f:
        norm_stats = json.load(f)
    NORM_MEAN = norm_stats['mean_rgb']
    NORM_STD = norm_stats['std_rgb']
    print(f'  Fundus norm: mean={[round(v, 4) for v in NORM_MEAN]}, '
          f'std={[round(v, 4) for v in NORM_STD]}')
else:
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]
    print('  Using ImageNet normalisation fallback')

# Load temperature
with open(TEMPERATURE_PATH) as f:
    temp_data = json.load(f)
TEMPERATURE = temp_data['temperature']
print(f'  Temperature T = {TEMPERATURE:.4f}')


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
        feat = 768

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
        f = self.backbone(x)
        f = self.drop(f)
        return self.disease_head(f), self.severity_head(f)


# ================================================================
# LOAD MODEL
# ================================================================
print('\nLoading model...')
model = MultiTaskViT().to(DEVICE)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f'  Loaded: {MODEL_PATH}')
print(f'  Checkpoint epoch: {ckpt.get("epoch", "?") + 1}  '
      f'val_acc={ckpt.get("val_acc", 0):.2f}%')


# ================================================================
# DATASET
# ================================================================
class FairnessDataset(Dataset):
    """Loads preprocessed cached images for inference."""

    def __init__(self, csv_path, base_dir):
        self.df = pd.read_csv(csv_path)
        self.base_dir = base_dir
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD),
        ])
        print(f'  Test set: {len(self.df)} images')
        print(f'  Sources : {dict(self.df["source"].value_counts())}')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = int(row['disease_label'])
        source = row['source']

        # Try cached .npy first
        cache_path = row.get('cache_path', '')
        if isinstance(cache_path, str) and cache_path:
            # Resolve relative path
            if not os.path.isabs(cache_path):
                clean = cache_path
                while clean.startswith('./') or clean.startswith('.//'):
                    clean = clean[2:] if clean.startswith('./') else clean[3:]
                cache_path = os.path.join(self.base_dir, clean)
            if os.path.exists(cache_path):
                img = np.load(cache_path)  # (224,224,3) uint8
                tensor = self.transform(img)
                return tensor, label, source, idx

        # Fallback: load and preprocess from source image
        image_path = row['image_path']
        if not os.path.isabs(image_path):
            clean = image_path
            while clean.startswith('./') or clean.startswith('.//'):
                clean = clean[2:] if clean.startswith('./') else clean[3:]
            image_path = os.path.join(self.base_dir, clean)

        if source == 'APTOS':
            img = self._ben_graham(image_path)
        else:
            img = self._clahe_preprocess(image_path)

        tensor = self.transform(img)
        return tensor, label, source, idx

    @staticmethod
    def _ben_graham(path, sz=IMG_SIZE, sigma=10):
        img = cv2.imread(path)
        if img is None:
            img = np.array(Image.open(path).convert('RGB'))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (sz, sz))
        img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigma), -4, 128)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (sz // 2, sz // 2), int(sz * 0.48), 255, -1)
        return cv2.bitwise_and(img, img, mask=mask)

    @staticmethod
    def _clahe_preprocess(path, sz=IMG_SIZE):
        img = cv2.imread(path)
        if img is None:
            img = np.array(Image.open(path).convert('RGB'))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (sz, sz))
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ================================================================
# RUN INFERENCE
# ================================================================
print('\nRunning inference on test set...')
dataset = FairnessDataset(TEST_CSV, BASE_DIR)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=4, pin_memory=True)

all_labels = []
all_preds = []
all_probs = []       # temperature-scaled probabilities
all_logits = []
all_sources = []
all_indices = []

with torch.no_grad():
    for batch_imgs, batch_labels, batch_sources, batch_idx in tqdm(loader, desc='Inference'):
        batch_imgs = batch_imgs.to(DEVICE)
        disease_logits, _ = model(batch_imgs)

        # Temperature scaling
        scaled_logits = disease_logits / TEMPERATURE
        probs = F.softmax(scaled_logits, dim=1)
        preds = probs.argmax(dim=1)

        all_labels.extend(batch_labels.numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())
        all_logits.extend(disease_logits.cpu().numpy().tolist())
        all_sources.extend(list(batch_sources))
        all_indices.extend(batch_idx.numpy().tolist())

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)
all_logits = np.array(all_logits)
all_sources = np.array(all_sources)
all_correct = (all_labels == all_preds).astype(int)

# Max confidence (probability of predicted class)
all_confidence = np.max(all_probs, axis=1)

print(f'\n  Total images: {len(all_labels)}')
print(f'  Overall accuracy: {accuracy_score(all_labels, all_preds):.4f}')
print(f'  Sources: {Counter(all_sources)}')


# ================================================================
# HELPER: compute metrics for a subset
# ================================================================
def compute_metrics(labels, preds, probs=None, class_names=CLASS_NAMES):
    """Compute per-class and overall metrics."""
    present_classes = sorted(set(labels))
    results = {}

    for c in range(len(class_names)):
        mask = labels == c
        n_c = mask.sum()
        if n_c == 0:
            results[class_names[c]] = {
                'n': 0, 'accuracy': None, 'precision': None,
                'recall': None, 'f1': None
            }
            continue
        # Binary: class c vs rest
        y_true_bin = (labels == c).astype(int)
        y_pred_bin = (preds == c).astype(int)

        tp = ((y_true_bin == 1) & (y_pred_bin == 1)).sum()
        fn = ((y_true_bin == 1) & (y_pred_bin == 0)).sum()
        fp = ((y_true_bin == 0) & (y_pred_bin == 1)).sum()

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        acc = (labels[mask] == preds[mask]).mean()

        results[class_names[c]] = {
            'n': int(n_c),
            'accuracy': float(round(acc, 4)),
            'precision': float(round(prec, 4)),
            'recall': float(round(rec, 4)),
            'f1': float(round(f1, 4)),
        }

    # Overall
    overall_acc = accuracy_score(labels, preds)
    present_labels = sorted(set(labels) | set(preds))
    overall_f1 = f1_score(labels, preds, labels=present_labels, average='macro', zero_division=0)
    overall_prec = precision_score(labels, preds, labels=present_labels, average='macro', zero_division=0)
    overall_rec = recall_score(labels, preds, labels=present_labels, average='macro', zero_division=0)

    results['Overall'] = {
        'n': int(len(labels)),
        'accuracy': float(round(overall_acc, 4)),
        'precision': float(round(overall_prec, 4)),
        'recall': float(round(overall_rec, 4)),
        'f1': float(round(overall_f1, 4)),
    }

    return results


# ================================================================
# SPLIT BY SOURCE
# ================================================================
sources_unique = sorted(set(all_sources))
print(f'\n  Unique sources: {sources_unique}')

source_masks = {}
source_metrics = {}
for src in sources_unique:
    mask = all_sources == src
    source_masks[src] = mask
    labels_s = all_labels[mask]
    preds_s = all_preds[mask]
    probs_s = all_probs[mask]
    metrics = compute_metrics(labels_s, preds_s, probs_s)
    source_metrics[src] = metrics
    acc = metrics['Overall']['accuracy']
    f1 = metrics['Overall']['f1']
    print(f'  {src:8s}: n={mask.sum():4d}  acc={acc:.4f}  macro-F1={f1:.4f}')


# ================================================================
# PLOT 1: Performance by Source (grouped bar chart)
# ================================================================
print('\nGenerating performance_by_source.png...')

# Focus on DR comparison (APTOS vs ODIR) + ODIR per-class
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
plt.subplots_adjust(wspace=0.35)

# Left panel: DR comparison across sources
metric_names = ['accuracy', 'f1', 'precision', 'recall']
metric_labels = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
bar_colors = {'APTOS': '#2196F3', 'ODIR': '#FF9800', 'REFUGE2': '#4CAF50'}

# DR-only comparison
ax = axes[0]
x = np.arange(len(metric_names))
bar_width = 0.25
offsets = np.arange(len(sources_unique)) - (len(sources_unique) - 1) / 2

for i, src in enumerate(sources_unique):
    dr_metrics = source_metrics[src].get('DR', {})
    vals = []
    for m in metric_names:
        v = dr_metrics.get(m, None)
        vals.append(v if v is not None else 0)
    bars = ax.bar(x + offsets[i] * bar_width, vals, bar_width,
                  label=f'{src} (n={dr_metrics.get("n", 0)})',
                  color=bar_colors.get(src, '#999999'), edgecolor='white', linewidth=0.5)
    # Value labels
    for bar, val in zip(bars, vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_xlabel('Metric', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('DR Classification: APTOS vs ODIR vs REFUGE2', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metric_labels)
ax.set_ylim(0, 1.15)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Right panel: ODIR per-class performance
ax = axes[1]
odir_metrics = source_metrics.get('ODIR', {})
classes_present = [c for c in CLASS_NAMES if odir_metrics.get(c, {}).get('n', 0) > 0]
x2 = np.arange(len(classes_present))
bar_width2 = 0.18
colors_metric = ['#1976D2', '#388E3C', '#F57C00', '#D32F2F']

for i, (m, ml) in enumerate(zip(metric_names, metric_labels)):
    vals = [odir_metrics[c][m] if odir_metrics[c][m] is not None else 0 for c in classes_present]
    bars = ax.bar(x2 + (i - 1.5) * bar_width2, vals, bar_width2,
                  label=ml, color=colors_metric[i], edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

ax.set_xlabel('Disease Class', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('ODIR Per-Class Performance', fontsize=13, fontweight='bold')
class_labels = [f'{c}\n(n={odir_metrics[c]["n"]})' for c in classes_present]
ax.set_xticks(x2)
ax.set_xticklabels(class_labels)
ax.set_ylim(0, 1.15)
ax.legend(fontsize=9, ncol=2)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.suptitle('Fairness & Domain Robustness: Performance by Source',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(FAIRNESS_DIR, 'performance_by_source.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('  Saved performance_by_source.png')


# ================================================================
# PLOT 2: Calibration by Source (reliability diagrams)
# ================================================================
print('Generating calibration_by_source.png...')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Reliability diagram helper
def reliability_diagram(labels, probs, preds, n_bins=10):
    """Compute reliability diagram data (mean predicted confidence vs actual accuracy per bin)."""
    confidences = np.max(probs, axis=1)
    correct = (labels == preds).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        mask = (confidences >= lo) & (confidences < hi) if b < n_bins - 1 \
            else (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_centers.append((lo + hi) / 2)
        bin_accs.append(correct[mask].mean())
        bin_confs.append(confidences[mask].mean())
        bin_counts.append(int(mask.sum()))

    return np.array(bin_centers), np.array(bin_accs), np.array(bin_confs), np.array(bin_counts)


# Left: all sources overlaid
ax = axes[0]
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
source_colors = {'APTOS': '#2196F3', 'ODIR': '#FF9800', 'REFUGE2': '#4CAF50'}

for src in sources_unique:
    mask = source_masks[src]
    if mask.sum() < 10:
        continue
    centers, accs, confs, counts = reliability_diagram(
        all_labels[mask], all_probs[mask], all_preds[mask], n_bins=10
    )
    ax.plot(confs, accs, 'o-', color=source_colors.get(src, '#999'),
            label=f'{src} (n={mask.sum()})', markersize=6, linewidth=2)

ax.set_xlabel('Mean Predicted Confidence', fontsize=12)
ax.set_ylabel('Actual Accuracy (Fraction Correct)', fontsize=12)
ax.set_title('Reliability Diagram by Source', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.grid(alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_aspect('equal')

# Right: ECE and histogram of confidence
ax = axes[1]
for src in sources_unique:
    mask = source_masks[src]
    if mask.sum() < 10:
        continue
    confs = np.max(all_probs[mask], axis=1)
    ax.hist(confs, bins=20, alpha=0.5, label=f'{src} (n={mask.sum()})',
            color=source_colors.get(src, '#999'), density=True, edgecolor='white')

ax.set_xlabel('Prediction Confidence', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Confidence Distribution by Source', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
fig.savefig(os.path.join(FAIRNESS_DIR, 'calibration_by_source.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('  Saved calibration_by_source.png')


# ================================================================
# PLOT 3 & 4: Confusion Matrices
# ================================================================
def plot_confusion_matrix(labels, preds, title, save_path, class_names=CLASS_NAMES):
    """Plot and save a publication-quality confusion matrix."""
    present_classes = sorted(set(labels) | set(preds))
    present_names = [class_names[c] for c in present_classes]

    cm = confusion_matrix(labels, preds, labels=present_classes)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    fig, ax = plt.subplots(figsize=(8, 7))

    # Create annotation strings with both count and percentage
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f'{cm[i, j]}\n({cm_norm[i, j]:.1%})'

    sns.heatmap(cm_norm, annot=annot, fmt='', cmap='Blues',
                xticklabels=present_names, yticklabels=present_names,
                ax=ax, vmin=0, vmax=1, linewidths=0.5, linecolor='white',
                cbar_kws={'label': 'Proportion'})

    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return cm


print('Generating confusion matrices...')

# APTOS confusion matrix (DR only, but predictions can span all classes)
aptos_mask = all_sources == 'APTOS'
if aptos_mask.sum() > 0:
    cm_aptos = plot_confusion_matrix(
        all_labels[aptos_mask], all_preds[aptos_mask],
        f'Confusion Matrix: APTOS (n={aptos_mask.sum()}, DR images only)',
        os.path.join(FAIRNESS_DIR, 'confusion_matrix_aptos.png')
    )
    print('  Saved confusion_matrix_aptos.png')

# ODIR confusion matrix (all 5 classes)
odir_mask = all_sources == 'ODIR'
if odir_mask.sum() > 0:
    cm_odir = plot_confusion_matrix(
        all_labels[odir_mask], all_preds[odir_mask],
        f'Confusion Matrix: ODIR (n={odir_mask.sum()}, all 5 classes)',
        os.path.join(FAIRNESS_DIR, 'confusion_matrix_odir.png')
    )
    print('  Saved confusion_matrix_odir.png')


# ================================================================
# PLOT 5: Confidence by Source (violin/box plots)
# ================================================================
print('Generating confidence_by_source.png...')

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Build dataframe for plotting
conf_df = pd.DataFrame({
    'Source': all_sources,
    'Confidence': all_confidence,
    'Correct': ['Correct' if c else 'Incorrect' for c in all_correct],
    'Predicted': [CLASS_NAMES[p] for p in all_preds],
    'True': [CLASS_NAMES[l] for l in all_labels],
})

# Left: violin by source and correctness
ax = axes[0]
plot_sources = [s for s in sources_unique if (all_sources == s).sum() >= 10]
conf_df_filtered = conf_df[conf_df['Source'].isin(plot_sources)]

sns.violinplot(data=conf_df_filtered, x='Source', y='Confidence', hue='Correct',
               split=True, inner='quartile', ax=ax,
               palette={'Correct': '#4CAF50', 'Incorrect': '#F44336'})

# Add means as scatter points
for i, src in enumerate(plot_sources):
    for j, corr_label in enumerate(['Correct', 'Incorrect']):
        mask_sc = (conf_df_filtered['Source'] == src) & (conf_df_filtered['Correct'] == corr_label)
        if mask_sc.sum() > 0:
            mean_val = conf_df_filtered.loc[mask_sc, 'Confidence'].mean()
            offset = -0.05 if j == 0 else 0.05
            ax.scatter(i + offset, mean_val, color='black', s=30, zorder=5, marker='D')

ax.set_title('Prediction Confidence by Source & Correctness', fontsize=13, fontweight='bold')
ax.set_ylabel('Confidence (max probability)', fontsize=12)
ax.set_xlabel('Data Source', fontsize=12)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Right: confidence by source for DR images only
ax = axes[1]
dr_df = conf_df[conf_df['True'] == 'DR']
dr_plot_sources = [s for s in sources_unique if ((all_sources == s) & (all_labels == 1)).sum() >= 5]
dr_df_filtered = dr_df[dr_df['Source'].isin(dr_plot_sources)]

if len(dr_df_filtered) > 0:
    sns.violinplot(data=dr_df_filtered, x='Source', y='Confidence', hue='Correct',
                   split=True, inner='quartile', ax=ax,
                   palette={'Correct': '#4CAF50', 'Incorrect': '#F44336'})

    ax.set_title('DR Confidence: APTOS vs ODIR', fontsize=13, fontweight='bold')
    ax.set_ylabel('Confidence (max probability)', fontsize=12)
    ax.set_xlabel('Data Source', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
fig.savefig(os.path.join(FAIRNESS_DIR, 'confidence_by_source.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('  Saved confidence_by_source.png')


# ================================================================
# PLOT 6: Error Patterns
# ================================================================
print('Generating error_patterns.png...')

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

def get_error_pairs(labels, preds, class_names, top_k=10):
    """Get most common misclassification pairs."""
    errors = []
    for true_l, pred_l in zip(labels, preds):
        if true_l != pred_l:
            errors.append(f'{class_names[true_l]} -> {class_names[pred_l]}')
    counter = Counter(errors)
    return counter.most_common(top_k)

# APTOS errors
ax = axes[0]
if aptos_mask.sum() > 0:
    aptos_errors = get_error_pairs(all_labels[aptos_mask], all_preds[aptos_mask], CLASS_NAMES, top_k=8)
    if aptos_errors:
        pairs, counts = zip(*aptos_errors)
        y_pos = np.arange(len(pairs))
        bars = ax.barh(y_pos, counts, color='#2196F3', edgecolor='white', linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(pairs, fontsize=10)
        ax.invert_yaxis()
        for bar, count in zip(bars, counts):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    str(count), va='center', fontsize=10, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No errors!', ha='center', va='center', fontsize=14,
                transform=ax.transAxes)

n_aptos_errors = (all_labels[aptos_mask] != all_preds[aptos_mask]).sum() if aptos_mask.sum() > 0 else 0
ax.set_title(f'APTOS Error Patterns\n({n_aptos_errors} errors / {aptos_mask.sum()} images)',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Count', fontsize=12)
ax.set_ylabel('True -> Predicted', fontsize=12)
ax.grid(axis='x', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ODIR errors
ax = axes[1]
if odir_mask.sum() > 0:
    odir_errors = get_error_pairs(all_labels[odir_mask], all_preds[odir_mask], CLASS_NAMES, top_k=10)
    if odir_errors:
        pairs, counts = zip(*odir_errors)
        y_pos = np.arange(len(pairs))
        # Color by severity
        pair_colors = []
        for p in pairs:
            if 'Normal' in p:
                pair_colors.append('#FF9800')
            elif 'DR' in p:
                pair_colors.append('#F44336')
            else:
                pair_colors.append('#9C27B0')
        bars = ax.barh(y_pos, counts, color=pair_colors, edgecolor='white', linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(pairs, fontsize=10)
        ax.invert_yaxis()
        for bar, count in zip(bars, counts):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    str(count), va='center', fontsize=10, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No errors!', ha='center', va='center', fontsize=14,
                transform=ax.transAxes)

n_odir_errors = (all_labels[odir_mask] != all_preds[odir_mask]).sum() if odir_mask.sum() > 0 else 0
ax.set_title(f'ODIR Error Patterns\n({n_odir_errors} errors / {odir_mask.sum()} images)',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Count', fontsize=12)
ax.set_ylabel('True -> Predicted', fontsize=12)
ax.grid(axis='x', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
fig.savefig(os.path.join(FAIRNESS_DIR, 'error_patterns.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('  Saved error_patterns.png')


# ================================================================
# COMPUTE ECE (Expected Calibration Error) per source
# ================================================================
def compute_ece(labels, probs, preds, n_bins=15):
    """Compute Expected Calibration Error."""
    confidences = np.max(probs, axis=1)
    correct = (labels == preds).astype(float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        mask = (confidences >= lo) & (confidences < hi) if b < n_bins - 1 \
            else (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / len(labels)) * abs(bin_acc - bin_conf)
    return float(ece)


# ================================================================
# STATISTICAL SIGNIFICANCE TEST
# ================================================================
print('\nRunning statistical tests...')

# Chi-squared test: is there a significant difference in accuracy between
# APTOS-DR and ODIR-DR?
aptos_dr_mask = (all_sources == 'APTOS') & (all_labels == 1)
odir_dr_mask = (all_sources == 'ODIR') & (all_labels == 1)

stat_results = {}

if aptos_dr_mask.sum() > 0 and odir_dr_mask.sum() > 0:
    aptos_dr_correct = all_correct[aptos_dr_mask].sum()
    aptos_dr_total = aptos_dr_mask.sum()
    odir_dr_correct = all_correct[odir_dr_mask].sum()
    odir_dr_total = odir_dr_mask.sum()

    # Contingency table: [[aptos_correct, aptos_wrong], [odir_correct, odir_wrong]]
    contingency = np.array([
        [aptos_dr_correct, aptos_dr_total - aptos_dr_correct],
        [odir_dr_correct, odir_dr_total - odir_dr_correct],
    ])

    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

    stat_results = {
        'test': 'chi-squared',
        'chi2_statistic': float(round(chi2, 4)),
        'p_value': float(p_value),
        'degrees_of_freedom': int(dof),
        'significant_at_005': bool(p_value < 0.05),
        'aptos_dr_accuracy': float(round(aptos_dr_correct / aptos_dr_total, 4)),
        'odir_dr_accuracy': float(round(odir_dr_correct / odir_dr_total, 4)),
        'aptos_dr_n': int(aptos_dr_total),
        'odir_dr_n': int(odir_dr_total),
    }

    print(f'  Chi-squared test (DR: APTOS vs ODIR):')
    print(f'    APTOS-DR accuracy: {aptos_dr_correct}/{aptos_dr_total} = '
          f'{aptos_dr_correct / aptos_dr_total:.4f}')
    print(f'    ODIR-DR  accuracy: {odir_dr_correct}/{odir_dr_total} = '
          f'{odir_dr_correct / odir_dr_total:.4f}')
    print(f'    chi2 = {chi2:.4f}, p = {p_value:.6f}, significant = {p_value < 0.05}')


# ================================================================
# BUILD DOMAIN GAP REPORT
# ================================================================
print('\nBuilding domain_gap_report.json...')

# Per-source ECE
source_ece = {}
for src in sources_unique:
    mask = source_masks[src]
    if mask.sum() >= 10:
        ece = compute_ece(all_labels[mask], all_probs[mask], all_preds[mask])
        source_ece[src] = round(ece, 4)

# Compute domain gap (difference in accuracy)
overall_accs = {}
for src in sources_unique:
    overall_accs[src] = source_metrics[src]['Overall']['accuracy']

domain_gap = {}
if 'APTOS' in overall_accs and 'ODIR' in overall_accs:
    # For DR specifically
    aptos_dr_acc = stat_results.get('aptos_dr_accuracy', None)
    odir_dr_acc = stat_results.get('odir_dr_accuracy', None)
    if aptos_dr_acc is not None and odir_dr_acc is not None:
        domain_gap['dr_accuracy_gap'] = round(abs(aptos_dr_acc - odir_dr_acc), 4)
        domain_gap['dr_gap_direction'] = 'APTOS > ODIR' if aptos_dr_acc > odir_dr_acc else 'ODIR > APTOS'
    domain_gap['overall_accuracy_gap'] = round(abs(overall_accs['APTOS'] - overall_accs['ODIR']), 4)
    domain_gap['overall_gap_direction'] = (
        'APTOS > ODIR' if overall_accs['APTOS'] > overall_accs['ODIR'] else 'ODIR > APTOS'
    )

# Mean confidence by source and correctness
confidence_stats = {}
for src in sources_unique:
    mask = source_masks[src]
    correct_mask = mask & (all_correct == 1)
    incorrect_mask = mask & (all_correct == 0)
    confidence_stats[src] = {
        'mean_confidence': round(float(all_confidence[mask].mean()), 4),
        'mean_confidence_correct': round(float(all_confidence[correct_mask].mean()), 4) if correct_mask.sum() > 0 else None,
        'mean_confidence_incorrect': round(float(all_confidence[incorrect_mask].mean()), 4) if incorrect_mask.sum() > 0 else None,
        'n_correct': int(correct_mask.sum()),
        'n_incorrect': int(incorrect_mask.sum()),
    }

# Key findings
findings = []

# 1. Domain gap in DR
if stat_results:
    gap = domain_gap.get('dr_accuracy_gap', 0)
    direction = domain_gap.get('dr_gap_direction', '')
    sig = stat_results.get('significant_at_005', False)
    findings.append(
        f"DR accuracy gap: {gap:.1%} ({direction}). "
        f"Chi-squared p={stat_results['p_value']:.4f} "
        f"({'statistically significant' if sig else 'not statistically significant'} at alpha=0.05)."
    )

# 2. Calibration
for src in sources_unique:
    if src in source_ece:
        findings.append(f"{src} ECE (Expected Calibration Error) = {source_ece[src]:.4f}.")

# 3. Confidence analysis
for src in sources_unique:
    cs = confidence_stats.get(src, {})
    mc_corr = cs.get('mean_confidence_correct')
    mc_incorr = cs.get('mean_confidence_incorrect')
    if mc_corr is not None and mc_incorr is not None:
        findings.append(
            f"{src}: mean confidence on correct={mc_corr:.3f}, incorrect={mc_incorr:.3f} "
            f"(separation={mc_corr - mc_incorr:.3f})."
        )

# 4. Error pattern analysis
if aptos_mask.sum() > 0:
    aptos_errors_list = get_error_pairs(all_labels[aptos_mask], all_preds[aptos_mask], CLASS_NAMES, top_k=3)
    if aptos_errors_list:
        top_err = aptos_errors_list[0]
        findings.append(
            f"APTOS top error: {top_err[0]} ({top_err[1]} instances). "
            f"Total APTOS errors: {n_aptos_errors}/{aptos_mask.sum()} "
            f"({n_aptos_errors / aptos_mask.sum():.1%})."
        )

if odir_mask.sum() > 0:
    odir_errors_list = get_error_pairs(all_labels[odir_mask], all_preds[odir_mask], CLASS_NAMES, top_k=3)
    if odir_errors_list:
        top_err = odir_errors_list[0]
        findings.append(
            f"ODIR top error: {top_err[0]} ({top_err[1]} instances). "
            f"Total ODIR errors: {n_odir_errors}/{odir_mask.sum()} "
            f"({n_odir_errors / odir_mask.sum():.1%})."
        )

# Assemble report
report = {
    'phase': '1D - Fairness & Domain Robustness Analysis',
    'model': 'RetinaSense-ViT v3 (vit_base_patch16_224)',
    'test_set_size': int(len(all_labels)),
    'temperature': TEMPERATURE,
    'source_distribution': {src: int((all_sources == src).sum()) for src in sources_unique},
    'per_source_metrics': {},
    'domain_gap': domain_gap,
    'statistical_test': stat_results,
    'calibration': {
        'ece_by_source': source_ece,
    },
    'confidence_analysis': confidence_stats,
    'key_findings': findings,
}

# Add per-source metrics (convert to serialisable format)
for src in sources_unique:
    report['per_source_metrics'][src] = source_metrics[src]

report_path = os.path.join(FAIRNESS_DIR, 'domain_gap_report.json')
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)

print(f'  Saved domain_gap_report.json')

# ================================================================
# SUMMARY
# ================================================================
print('\n' + '=' * 65)
print('  FAIRNESS ANALYSIS COMPLETE')
print('=' * 65)
print(f'\n  Output directory: {FAIRNESS_DIR}')
print(f'  Files generated:')
for fname in ['performance_by_source.png', 'calibration_by_source.png',
              'confusion_matrix_aptos.png', 'confusion_matrix_odir.png',
              'confidence_by_source.png', 'error_patterns.png',
              'domain_gap_report.json']:
    fpath = os.path.join(FAIRNESS_DIR, fname)
    exists = os.path.exists(fpath)
    size = os.path.getsize(fpath) if exists else 0
    status = f'{size / 1024:.1f} KB' if exists else 'MISSING'
    print(f'    {fname:35s} {status}')

print(f'\n  KEY FINDINGS:')
for i, finding in enumerate(findings, 1):
    print(f'    {i}. {finding}')
print('=' * 65)
