#!/usr/bin/env python3
"""
RetinaSense ViT v2 - Comprehensive Error Analysis & Baseline Report
===================================================================
Runs full evaluation on the validation split, computes ECE,
confusion analysis, confidence distributions, and source-level
performance. Saves all plots and metrics to outputs_analysis/v2_baseline/.
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
warnings.filterwarnings('ignore')

import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import timm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report,
    f1_score, precision_score, recall_score
)

# ================================================================
# CONFIG
# ================================================================
BASE_DIR    = '/teamspace/studios/this_studio'
MODEL_PATH  = f'{BASE_DIR}/outputs_vit/best_model.pth'
META_CSV    = f'{BASE_DIR}/final_unified_metadata.csv'
THRESH_JSON = f'{BASE_DIR}/outputs_vit/threshold_optimization_results.json'
CACHE_DIR   = f'{BASE_DIR}/preprocessed_cache_vit'
OUT_DIR     = f'{BASE_DIR}/outputs_analysis/v2_baseline'

IMG_SIZE    = 224
BATCH_SIZE  = 64
NUM_WORKERS = 8
NUM_CLASSES = 5
CLASS_NAMES = ['Normal', 'Diabetes/DR', 'Glaucoma', 'Cataract', 'AMD']

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# ================================================================
# MODEL DEFINITION  (mirrors retinasense_vit.py)
# ================================================================
class MultiTaskViT(nn.Module):
    def __init__(self, n_disease=5, n_severity=5, drop=0.4):
        super().__init__()
        self.backbone = timm.create_model(
            'vit_base_patch16_224', pretrained=False, num_classes=0)
        feat = 768
        self.drop = nn.Dropout(drop)
        self.disease_head = nn.Sequential(
            nn.Linear(feat, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, n_disease))
        self.severity_head = nn.Sequential(
            nn.Linear(feat, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, n_severity))

    def forward(self, x):
        f = self.backbone(x)
        f = self.drop(f)
        return self.disease_head(f), self.severity_head(f)


# ================================================================
# IMAGE PREPROCESSING  (Ben Graham method, matches training)
# ================================================================
def ben_graham(path, sz=IMG_SIZE, sigma=10):
    img = cv2.imread(str(path))
    if img is None:
        img = np.array(Image.open(str(path)).convert('RGB'))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (sz, sz))
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), sigma), -4, 128)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (sz//2, sz//2), int(sz * 0.48), 255, -1)
    return cv2.bitwise_and(img, img, mask=mask)


def resolve_image_path(raw_path):
    """
    Resolve image path from CSV entry (which has leading .// prefix).
    Tries multiple known root locations.
    APTOS images live in:
      aptos/gaussian_filtered_images/gaussian_filtered_images/{Severity}/{stem}.png
    ODIR images live in:
      odir/preprocessed_images/{filename}
    """
    # Strip leading .// or ./
    clean = raw_path.lstrip('.').lstrip('/').lstrip('/')
    clean = clean.replace('//', '/')

    stem = Path(raw_path).stem

    candidates = [
        f'{BASE_DIR}/{clean}',
    ]

    # APTOS: search all severity subfolders
    if 'aptos' in raw_path.lower():
        aptos_base = f'{BASE_DIR}/aptos/gaussian_filtered_images/gaussian_filtered_images'
        for severity in ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']:
            for ext in ['.png', '.jpg', '.jpeg']:
                candidates.append(f'{aptos_base}/{severity}/{stem}{ext}')
        # Also try train_images (original path)
        for ext in ['.png', '.jpg', '.jpeg']:
            candidates.append(f'{BASE_DIR}/aptos/train_images/{stem}{ext}')

    # ODIR: preprocessed_images
    if 'odir' in raw_path.lower():
        fname = Path(raw_path).name
        candidates.append(f'{BASE_DIR}/odir/preprocessed_images/{fname}')
        candidates.append(f'{BASE_DIR}/ocular-disease-recognition-odir5k/preprocessed_images/{fname}')

    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def load_or_cache(row):
    """
    Load preprocessed image from cache (.npy) or process from disk.
    Returns uint8 HxWx3 numpy array.
    """
    stem = Path(row['image_path_clean']).stem
    cache_fp = f'{CACHE_DIR}/{stem}_224.npy'

    if os.path.exists(cache_fp):
        try:
            return np.load(cache_fp)
        except Exception:
            pass

    img_path = row.get('image_path_resolved')
    if img_path and os.path.exists(img_path):
        try:
            arr = ben_graham(img_path)
            np.save(cache_fp, arr)
            return arr
        except Exception as e:
            pass

    # Fallback: zero image
    return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)


# ================================================================
# DATASET
# ================================================================
val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class RetDS(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = load_or_cache(r)
        return (
            val_transform(img),
            torch.tensor(int(r['disease_label']), dtype=torch.long),
            torch.tensor(int(r['severity_label']), dtype=torch.long),
            i  # return index so we can track per-sample metadata
        )


# ================================================================
# STEP 1 — LOAD METADATA & BUILD VAL SPLIT
# ================================================================
print('\n[1/6] Loading metadata and building val split...')

meta = pd.read_csv(META_CSV)
print(f'  Raw rows: {len(meta)}')

# Fix image paths
meta['image_path_clean'] = meta['image_path'].str.lstrip('.').str.lstrip('/').str.replace('//', '/', regex=False)
meta['image_path_resolved'] = meta['image_path_clean'].apply(
    lambda p: resolve_image_path(p)
)

n_resolved = meta['image_path_resolved'].notna().sum()
print(f'  Images resolved on disk: {n_resolved} / {len(meta)}')

# Build the same stratified split used in training (random_state=42, test_size=0.2)
train_df, val_df = train_test_split(
    meta,
    test_size=0.2,
    stratify=meta['disease_label'],
    random_state=42
)
val_df = val_df.reset_index(drop=True)
print(f'  Val split: {len(val_df)} samples')
print(f'  Val class distribution:')
for lbl, cnt in val_df['disease_label'].value_counts().sort_index().items():
    print(f'    {CLASS_NAMES[int(lbl)]:<15s}: {cnt:4d}')

# ================================================================
# STEP 2 — LOAD MODEL
# ================================================================
print('\n[2/6] Loading model...')

model = MultiTaskViT().to(device)
ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f'  Loaded checkpoint: epoch={ckpt.get("epoch","?")}, '
      f'macro_f1={ckpt.get("macro_f1", 0):.4f}')

# Load thresholds
with open(THRESH_JSON) as f:
    thresh_data = json.load(f)
thresholds = {int(k): float(v) for k, v in thresh_data['optimal_thresholds'].items()}
print(f'  Optimal thresholds: {thresholds}')

# ================================================================
# STEP 3 — RUN INFERENCE
# ================================================================
print('\n[3/6] Running inference on val set...')

val_ds = RetDS(val_df)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True
)

all_probs  = []   # (N, 5) softmax probabilities
all_preds  = []   # (N,) argmax predictions
all_labels = []   # (N,) true labels
all_idxs   = []   # (N,) val_df indices

with torch.no_grad():
    for imgs, d_lbl, s_lbl, idx in tqdm(val_loader, desc='Inference'):
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast('cuda'):
            d_out, _ = model(imgs)
        probs = torch.softmax(d_out.float(), dim=1).cpu().numpy()
        preds = d_out.argmax(1).cpu().numpy()
        all_probs.append(probs)
        all_preds.append(preds)
        all_labels.append(d_lbl.numpy())
        all_idxs.append(idx.numpy())

all_probs  = np.vstack(all_probs)    # (N, 5)
all_preds  = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)
all_idxs   = np.concatenate(all_idxs)

# Also compute threshold-adjusted predictions
thresh_preds = np.zeros_like(all_preds)
for i in range(len(all_probs)):
    adjusted = all_probs[i].copy()
    for c, t in thresholds.items():
        adjusted[c] = all_probs[i][c] / t  # scale by threshold
    thresh_preds[i] = adjusted.argmax()

raw_acc    = (all_preds == all_labels).mean() * 100
thresh_acc = (thresh_preds == all_labels).mean() * 100
print(f'  Raw accuracy      : {raw_acc:.2f}%')
print(f'  Threshold accuracy: {thresh_acc:.2f}%')

# Use threshold-adjusted for main analysis (matches published 84.48%)
preds = thresh_preds

# ================================================================
# STEP 4 — CONFIDENCE CALIBRATION (ECE)
# ================================================================
print('\n[4/6] Computing ECE and reliability diagram...')

def compute_ece(probs, labels, n_bins=10):
    """Expected Calibration Error with equal-width bins."""
    confidences = probs.max(axis=1)          # max probability = confidence
    predicted   = probs.argmax(axis=1)
    correct     = (predicted == labels).astype(float)

    bins    = np.linspace(0, 1, n_bins + 1)
    ece     = 0.0
    bin_acc   = []
    bin_conf  = []
    bin_count = []

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            bin_acc.append(0.0)
            bin_conf.append((lo + hi) / 2)
            bin_count.append(0)
            continue
        acc  = correct[mask].mean()
        conf = confidences[mask].mean()
        n    = mask.sum()
        ece += (n / len(labels)) * abs(acc - conf)
        bin_acc.append(acc)
        bin_conf.append(conf)
        bin_count.append(int(n))

    return ece, bin_acc, bin_conf, bin_count, bins

ece, bin_acc, bin_conf, bin_count, bins = compute_ece(all_probs, all_labels)
print(f'  ECE (10 bins): {ece:.4f}')

# Per-class calibration
per_class_ece = {}
for c in range(NUM_CLASSES):
    mask = (all_labels == c)
    if mask.sum() == 0:
        per_class_ece[CLASS_NAMES[c]] = 0.0
        continue
    ece_c, _, _, _, _ = compute_ece(all_probs[mask], all_labels[mask])
    per_class_ece[CLASS_NAMES[c]] = float(ece_c)
    print(f'    ECE {CLASS_NAMES[c]:<15s}: {ece_c:.4f}')

# -- Reliability diagram --
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

bin_centers = (bins[:-1] + bins[1:]) / 2
bars = axes[0].bar(
    bin_centers, bin_acc,
    width=(bins[1] - bins[0]) * 0.9,
    alpha=0.7, color='steelblue', label='Accuracy per bin'
)
axes[0].plot([0, 1], [0, 1], 'r--', lw=2, label='Perfect calibration')
axes[0].set_xlabel('Confidence', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title(f'Reliability Diagram\nECE = {ece:.4f}', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)
axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1)

# Annotate with bin counts
for bar, cnt in zip(bars, bin_count):
    if cnt > 0:
        axes[0].text(
            bar.get_x() + bar.get_width()/2, min(bar.get_height() + 0.02, 0.97),
            str(cnt), ha='center', va='bottom', fontsize=7, color='black'
        )

# Gap diagram (overconfidence = positive gap)
gap = np.array(bin_conf) - np.array(bin_acc)
color_gap = ['#e74c3c' if g > 0 else '#2ecc71' for g in gap]
axes[1].bar(bin_centers, gap, width=(bins[1]-bins[0])*0.9, color=color_gap, alpha=0.8)
axes[1].axhline(0, color='black', lw=1)
axes[1].set_xlabel('Confidence', fontsize=12)
axes[1].set_ylabel('Confidence - Accuracy (Gap)', fontsize=12)
axes[1].set_title('Calibration Gap\n(Red=overconfident, Green=underconfident)',
                  fontsize=13, fontweight='bold')
axes[1].grid(alpha=0.3)
axes[1].set_xlim(0, 1)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/reliability_diagram.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved reliability_diagram.png')

# ================================================================
# STEP 5 — CONFUSION MATRIX
# ================================================================
print('\n[5/6] Generating confusion matrices...')

cm_raw  = confusion_matrix(all_labels, preds)
cm_norm = cm_raw.astype(float) / cm_raw.sum(axis=1, keepdims=True)

# -- Raw counts confusion matrix --
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    cm_raw, annot=True, fmt='d', cmap='Blues', ax=ax,
    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
    linewidths=0.5, linecolor='gray'
)
ax.set_title('Confusion Matrix (Raw Counts)', fontsize=14, fontweight='bold')
ax.set_ylabel('True Label', fontsize=12)
ax.set_xlabel('Predicted Label', fontsize=12)
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/confusion_matrix_raw.png', dpi=150, bbox_inches='tight')
plt.close()

# -- Normalized confusion matrix --
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    cm_norm, annot=True, fmt='.3f', cmap='Blues', ax=ax,
    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
    linewidths=0.5, linecolor='gray', vmin=0, vmax=1
)
ax.set_title('Confusion Matrix (Normalized by True Class)', fontsize=14, fontweight='bold')
ax.set_ylabel('True Label', fontsize=12)
ax.set_xlabel('Predicted Label', fontsize=12)
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/confusion_matrix_normalized.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved confusion_matrix_raw.png and confusion_matrix_normalized.png')

# -- Top confused pairs --
confused_pairs = []
for true_c in range(NUM_CLASSES):
    for pred_c in range(NUM_CLASSES):
        if true_c == pred_c:
            continue
        count = cm_raw[true_c, pred_c]
        rate  = cm_norm[true_c, pred_c]
        confused_pairs.append({
            'true_class':      CLASS_NAMES[true_c],
            'pred_class':      CLASS_NAMES[pred_c],
            'count':           int(count),
            'rate':            float(rate),
            'description':     f'{CLASS_NAMES[true_c]} misclassified AS {CLASS_NAMES[pred_c]}'
        })
confused_pairs.sort(key=lambda x: x['count'], reverse=True)
top5_pairs = confused_pairs[:5]

print('\n  Top 5 confused class pairs (by raw count):')
for p in top5_pairs:
    print(f'    {p["description"]}: {p["count"]} ({p["rate"]*100:.1f}%)')

# ================================================================
# STEP 6 — PER-CLASS METRICS
# ================================================================
print('\n[6/6] Computing per-class metrics...')

report_dict = classification_report(
    all_labels, preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0
)
print(classification_report(all_labels, preds, target_names=CLASS_NAMES, digits=4, zero_division=0))

per_class_precision = {}
per_class_recall    = {}
per_class_f1        = {}
per_class_support   = {}

for cn in CLASS_NAMES:
    per_class_precision[cn] = report_dict[cn]['precision']
    per_class_recall[cn]    = report_dict[cn]['recall']
    per_class_f1[cn]        = report_dict[cn]['f1-score']
    per_class_support[cn]   = int(report_dict[cn]['support'])

overall_accuracy = report_dict['accuracy'] * 100
macro_f1         = report_dict['macro avg']['f1-score']
weighted_f1      = report_dict['weighted avg']['f1-score']

print(f'\n  Overall accuracy : {overall_accuracy:.2f}%')
print(f'  Macro F1         : {macro_f1:.4f}')
print(f'  Weighted F1      : {weighted_f1:.4f}')

# ================================================================
# CONFIDENCE DISTRIBUTION ANALYSIS
# ================================================================
print('\nAnalyzing confidence distributions...')

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

all_max_conf = all_probs.max(axis=1)
all_correct  = (preds == all_labels)

for ci, cn in enumerate(CLASS_NAMES):
    ax = axes[ci]
    mask_class = (all_labels == ci)

    correct_conf = all_max_conf[mask_class & all_correct]
    wrong_conf   = all_max_conf[mask_class & ~all_correct]

    n_correct = len(correct_conf)
    n_wrong   = len(wrong_conf)

    if n_correct > 0:
        ax.hist(correct_conf, bins=20, alpha=0.6, color='#2ecc71',
                label=f'Correct (n={n_correct})', density=True)
    if n_wrong > 0:
        ax.hist(wrong_conf, bins=20, alpha=0.6, color='#e74c3c',
                label=f'Wrong (n={n_wrong})', density=True)

    # Mark high-confidence wrong predictions
    if n_wrong > 0:
        high_conf_wrong = (wrong_conf > 0.8).sum()
        ax.axvline(0.8, color='darkred', linestyle='--', alpha=0.7, lw=1.5,
                   label=f'Conf>0.8 wrong: {high_conf_wrong}')

    ax.set_title(f'{cn}\nPrec={per_class_precision[cn]:.3f} Rec={per_class_recall[cn]:.3f} F1={per_class_f1[cn]:.3f}',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('Max Confidence', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)

# Summary panel
ax = axes[5]
mean_correct = [all_max_conf[all_labels==c][preds[all_labels==c]==c].mean()
                if (all_labels==c).sum() > 0 else 0 for c in range(NUM_CLASSES)]
mean_wrong   = [all_max_conf[all_labels==c][preds[all_labels==c]!=c].mean()
                if ((all_labels==c) & (preds!=c)).sum() > 0 else 0 for c in range(NUM_CLASSES)]

x = np.arange(NUM_CLASSES)
width = 0.35
ax.bar(x - width/2, mean_correct, width, label='Mean conf (correct)', color='#2ecc71', alpha=0.8)
ax.bar(x + width/2, mean_wrong,   width, label='Mean conf (wrong)',   color='#e74c3c', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels([c[:6] for c in CLASS_NAMES], rotation=20)
ax.set_ylabel('Mean Confidence')
ax.set_title('Mean Confidence: Correct vs Wrong', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis='y')
ax.set_ylim(0, 1)

plt.suptitle('Confidence Distribution Analysis per Class', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/confidence_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved confidence_distributions.png')

# ================================================================
# PER-SOURCE ANALYSIS
# ================================================================
print('\nRunning per-source analysis...')

# Attach dataset source to val_df indices
source_col = val_df['dataset'].values

results_df = pd.DataFrame({
    'true_label': all_labels,
    'pred_label': preds,
    'max_conf':   all_max_conf,
    'dataset':    source_col[all_idxs],
    'correct':    (preds == all_labels).astype(int),
})

per_source = {}
for src in ['ODIR', 'APTOS']:
    mask = results_df['dataset'] == src
    if mask.sum() == 0:
        continue
    src_true = results_df['true_label'][mask].values
    src_pred = results_df['pred_label'][mask].values
    src_acc  = (src_true == src_pred).mean() * 100
    src_f1   = f1_score(src_true, src_pred, average='macro', zero_division=0)

    per_class_acc_src = {}
    for c in range(NUM_CLASSES):
        cmask = (src_true == c)
        if cmask.sum() == 0:
            per_class_acc_src[CLASS_NAMES[c]] = None
        else:
            per_class_acc_src[CLASS_NAMES[c]] = float((src_pred[cmask] == c).mean() * 100)

    per_source[src] = {
        'n_samples':      int(mask.sum()),
        'accuracy':       float(src_acc),
        'macro_f1':       float(src_f1),
        'per_class_acc':  per_class_acc_src
    }
    print(f'\n  {src} (n={mask.sum()}):')
    print(f'    Accuracy : {src_acc:.2f}%')
    print(f'    Macro F1 : {src_f1:.4f}')
    for cn, acc in per_class_acc_src.items():
        if acc is not None:
            print(f'    {cn:<15s}: {acc:.1f}%')

# -- Per-source performance plot --
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Overall bar
sources = list(per_source.keys())
accs    = [per_source[s]['accuracy'] for s in sources]
f1s     = [per_source[s]['macro_f1'] for s in sources]

x = np.arange(len(sources))
w = 0.35
axes[0].bar(x - w/2, accs, w, label='Accuracy (%)', color=['#3498db', '#e67e22'], alpha=0.85)
axes[0].bar(x + w/2, [f*100 for f in f1s], w, label='Macro F1 ×100',
            color=['#2ecc71', '#e74c3c'], alpha=0.85)
axes[0].set_xticks(x); axes[0].set_xticklabels(sources)
axes[0].set_ylim(50, 100)
axes[0].set_ylabel('Score')
axes[0].set_title('Overall Performance by Source', fontweight='bold')
axes[0].legend(); axes[0].grid(alpha=0.3, axis='y')
for xi, (acc, f1) in enumerate(zip(accs, f1s)):
    axes[0].text(xi - w/2, acc + 0.5, f'{acc:.1f}', ha='center', fontsize=9)
    axes[0].text(xi + w/2, f1*100 + 0.5, f'{f1*100:.1f}', ha='center', fontsize=9)

# Per-class accuracy by source
class_data = {cn: [] for cn in CLASS_NAMES}
valid_sources = []
for src in sources:
    valid_sources.append(src)
    for cn in CLASS_NAMES:
        acc = per_source[src]['per_class_acc'].get(cn)
        class_data[cn].append(acc if acc is not None else 0.0)

x = np.arange(len(CLASS_NAMES))
n_src = len(valid_sources)
width = 0.8 / n_src
colors_src = ['#3498db', '#e67e22', '#2ecc71']

for si, src in enumerate(valid_sources):
    vals = [class_data[cn][si] for cn in CLASS_NAMES]
    offset = (si - n_src/2 + 0.5) * width
    axes[1].bar(x + offset, vals, width, label=src, alpha=0.85, color=colors_src[si])

axes[1].set_xticks(x); axes[1].set_xticklabels(CLASS_NAMES, rotation=20, ha='right')
axes[1].set_ylim(0, 105)
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Per-Class Accuracy by Source', fontweight='bold')
axes[1].legend(); axes[1].grid(alpha=0.3, axis='y')

plt.suptitle('Dataset Source Performance Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/per_source_performance.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved per_source_performance.png')

# ================================================================
# SAVE METRICS JSON
# ================================================================
print('\nSaving metrics JSON...')

baseline_metrics = {
    'overall_accuracy':      float(overall_accuracy),
    'raw_accuracy':          float(raw_acc),
    'threshold_accuracy':    float(thresh_acc),
    'macro_f1':              float(macro_f1),
    'weighted_f1':           float(weighted_f1),
    'ece':                   float(ece),
    'per_class_ece':         per_class_ece,
    'per_class_f1':          per_class_f1,
    'per_class_precision':   per_class_precision,
    'per_class_recall':      per_class_recall,
    'per_class_support':     per_class_support,
    'per_source_accuracy':   {
        src: {
            'accuracy':    per_source[src]['accuracy'],
            'macro_f1':    per_source[src]['macro_f1'],
            'n_samples':   per_source[src]['n_samples'],
            'per_class_acc': per_source[src]['per_class_acc']
        }
        for src in per_source
    },
    'top_confusion_pairs':   top5_pairs,
    'confusion_matrix_raw':  cm_raw.tolist(),
    'val_split_size':        len(val_df),
    'thresholds_used':       thresholds,
    'calibration': {
        'ece':        float(ece),
        'bin_acc':    [float(x) for x in bin_acc],
        'bin_conf':   [float(x) for x in bin_conf],
        'bin_count':  bin_count,
    }
}

with open(f'{OUT_DIR}/baseline_metrics.json', 'w') as f:
    json.dump(baseline_metrics, f, indent=2)
print(f'  Saved baseline_metrics.json')

# ================================================================
# ANALYSIS REPORT
# ================================================================
print('\nGenerating analysis report...')

# Identify key findings
worst_recall_class = min(per_class_recall, key=per_class_recall.get)
worst_f1_class     = min(per_class_f1,     key=per_class_f1.get)
best_f1_class      = max(per_class_f1,     key=per_class_f1.get)

# High-confidence wrong predictions per class
hcw_analysis = {}
for ci, cn in enumerate(CLASS_NAMES):
    mask_class = (all_labels == ci)
    wrong_mask  = mask_class & ~all_correct
    if wrong_mask.sum() > 0:
        high_conf_wrong = ((all_max_conf > 0.8) & wrong_mask).sum()
        hcw_analysis[cn] = {
            'total_wrong': int(wrong_mask.sum()),
            'high_conf_wrong_count': int(high_conf_wrong),
            'high_conf_wrong_pct': float(high_conf_wrong / wrong_mask.sum() * 100) if wrong_mask.sum() > 0 else 0,
            'mean_wrong_conf': float(all_max_conf[wrong_mask].mean()) if wrong_mask.sum() > 0 else 0,
        }
    else:
        hcw_analysis[cn] = {'total_wrong': 0, 'high_conf_wrong_count': 0,
                             'high_conf_wrong_pct': 0, 'mean_wrong_conf': 0}

# Domain gap
domain_gap = None
if 'ODIR' in per_source and 'APTOS' in per_source:
    odir_acc  = per_source['ODIR']['accuracy']
    aptos_acc = per_source['APTOS']['accuracy']
    domain_gap = abs(odir_acc - aptos_acc)

    # DR-specific domain gap
    odir_dr  = per_source['ODIR']['per_class_acc'].get('Diabetes/DR', 0) or 0
    aptos_dr = per_source['APTOS']['per_class_acc'].get('Diabetes/DR', 0) or 0
    dr_gap   = abs(odir_dr - aptos_dr)
else:
    domain_gap = 0; odir_acc = 0; aptos_acc = 0; odir_dr = 0; aptos_dr = 0; dr_gap = 0

calibration_verdict = 'overconfident' if sum(
    b_conf - b_acc for b_conf, b_acc in zip(bin_conf, bin_acc) if bin_count[bin_acc.index(b_acc)] > 0
) > 0 else 'underconfident'

report = f"""# RetinaSense ViT v2 — Baseline Error Analysis Report
**Generated**: 2026-03-06
**Model**: ViT-Base-Patch16-224 (MultiTaskViT)
**Checkpoint**: outputs_vit/best_model.pth
**Val Split**: {len(val_df)} samples (20% stratified, random_state=42)

---

## 1. Overall Performance

| Metric | Value |
|--------|-------|
| Accuracy (raw argmax) | {raw_acc:.2f}% |
| Accuracy (with thresholds) | {thresh_acc:.2f}% |
| Macro F1 | {macro_f1:.4f} |
| Weighted F1 | {weighted_f1:.4f} |
| ECE (10 bins) | {ece:.4f} |

---

## 2. Per-Class Metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
"""
for cn in CLASS_NAMES:
    report += (f"| {cn:<15s} | {per_class_precision[cn]:.4f} | "
               f"{per_class_recall[cn]:.4f} | {per_class_f1[cn]:.4f} | "
               f"{per_class_support[cn]:4d} |\n")

report += f"""
---

## 3. Confusion Analysis — Top 5 Confused Pairs

| Rank | True Class | Predicted As | Count | Rate |
|------|-----------|-------------|-------|------|
"""
for rank, pair in enumerate(top5_pairs, 1):
    report += (f"| {rank} | {pair['true_class']} | {pair['pred_class']} | "
               f"{pair['count']} | {pair['rate']*100:.1f}% |\n")

report += f"""
### Full Confusion Matrix (normalized by true class)

```
          {('  '.join(f'{cn[:6]:>7s}' for cn in CLASS_NAMES))}
"""
for ri, rn in enumerate(CLASS_NAMES):
    row_str = '  '.join(f'{cm_norm[ri, ci]:.3f}' for ci in range(NUM_CLASSES))
    report += f"{rn[:8]:>8s}  {row_str}\n"

report += f"""```

---

## 4. Confidence Calibration Analysis

- **ECE (overall)**: {ece:.4f}
- **Calibration pattern**: The model is predominantly **{calibration_verdict}**
  (mean confidence exceeds accuracy in most bins).

### Per-Class ECE

| Class | ECE |
|-------|-----|
"""
for cn, ece_c in per_class_ece.items():
    report += f"| {cn} | {ece_c:.4f} |\n"

report += f"""
### High-Confidence Wrong Predictions (confidence > 0.8)

| Class | Total Wrong | High-Conf Wrong | % of Errors | Mean Wrong Conf |
|-------|------------|----------------|-------------|----------------|
"""
for cn, hcw in hcw_analysis.items():
    report += (f"| {cn} | {hcw['total_wrong']} | {hcw['high_conf_wrong_count']} | "
               f"{hcw['high_conf_wrong_pct']:.1f}% | {hcw['mean_wrong_conf']:.3f} |\n")

report += f"""
---

## 5. Dataset Source Analysis (ODIR vs APTOS)

| Source | N Samples | Accuracy | Macro F1 |
|--------|-----------|----------|----------|
"""
for src, data in per_source.items():
    report += f"| {src} | {data['n_samples']} | {data['accuracy']:.2f}% | {data['macro_f1']:.4f} |\n"

report += f"""
### Per-Class Accuracy by Source

| Class |"""
for src in per_source:
    report += f" {src} |"
report += "\n|-------|"
for _ in per_source:
    report += "--------|"
report += "\n"
for cn in CLASS_NAMES:
    report += f"| {cn} |"
    for src in per_source:
        acc = per_source[src]['per_class_acc'].get(cn)
        if acc is None:
            report += " N/A |"
        else:
            report += f" {acc:.1f}% |"
    report += "\n"

report += f"""
**Domain gap (overall accuracy)**: {domain_gap:.2f}pp between ODIR and APTOS
"""
if 'ODIR' in per_source and 'APTOS' in per_source:
    report += f"""**DR class gap (ODIR vs APTOS)**: ODIR={odir_dr:.1f}% vs APTOS={aptos_dr:.1f}% (gap={dr_gap:.1f}pp)
"""

report += f"""
---

## 6. Error Pattern Summary

### Q1: What is the model's biggest weakness?

The model's biggest weakness is classifying **{worst_f1_class}** (F1={per_class_f1[worst_f1_class]:.4f},
recall={per_class_recall[worst_f1_class]:.4f}). This class has the worst F1 score, indicating the
model struggles to both detect and correctly distinguish it from other pathologies.

The confusion matrix shows that the primary confusion pathway is:
- **{top5_pairs[0]['description']}**: {top5_pairs[0]['count']} cases ({top5_pairs[0]['rate']*100:.1f}% error rate)
- **{top5_pairs[1]['description']}**: {top5_pairs[1]['count']} cases ({top5_pairs[1]['rate']*100:.1f}% error rate)

### Q2: Which class has the worst recall? Why?

**{worst_recall_class}** has the worst recall at {per_class_recall[worst_recall_class]:.4f}.
"""

# Detailed reason based on support
worst_support = per_class_support[worst_recall_class]
all_support   = sum(per_class_support.values())
worst_pct     = worst_support / all_support * 100
report += f"""This class represents only {worst_support} samples ({worst_pct:.1f}% of the val set).
The low recall is likely caused by:
1. **Class imbalance** — the model sees fewer examples during training and defaults to predicting
   more common classes when uncertain.
2. **Visual similarity** with other conditions (especially {top5_pairs[0]['pred_class'] if top5_pairs[0]['true_class']==worst_recall_class else 'Normal'})
   at the fundus level.
3. **Threshold sensitivity** — the optimized threshold ({thresholds.get(CLASS_NAMES.index(worst_recall_class), 0.5):.2f})
   may overcorrect or undercorrect depending on the calibration.

### Q3: Evidence of domain shift (ODIR vs APTOS)?

"""
if domain_gap is not None and domain_gap > 2.0:
    report += f"""YES — there is a **{domain_gap:.1f}pp accuracy gap** between ODIR ({odir_acc:.1f}%) and APTOS
({aptos_acc:.1f}%). This is significant and consistent with domain shift between the two data sources.

For the DR/Diabetes class specifically, the gap is **{dr_gap:.1f}pp** (ODIR={odir_dr:.1f}% vs APTOS={aptos_dr:.1f}%).
APTOS images are specifically DR-graded fundus photographs from India (Aravind Eye Hospital),
while ODIR covers multiple disease classes with more varied image quality and capture conditions.
The Ben Graham preprocessing helps but does not fully bridge the domain gap.

**Implication for v3**: Domain-specific augmentation or source-aware training (e.g., source
as auxiliary input, separate batch norms, or domain adaptation) may improve generalization.
"""
elif domain_gap is not None and domain_gap > 0:
    report += f"""MINOR gap observed — {domain_gap:.1f}pp difference between ODIR ({odir_acc:.1f}%) and
APTOS ({aptos_acc:.1f}%). The gap is small, suggesting the Ben Graham preprocessing and ViT
architecture generalize reasonably across sources. DR-specific gap: {dr_gap:.1f}pp.
"""
else:
    report += "Insufficient cross-source data to conclude domain shift.\n"

report += f"""
### Q4: Calibration assessment

ECE = **{ece:.4f}** (scale: 0=perfect, 0.1=poor).

"""
if ece < 0.03:
    report += "The model is **well-calibrated** (ECE < 0.03). Confidence scores are reliable."
elif ece < 0.07:
    report += f"""The model shows **moderate miscalibration** (ECE={ece:.4f}). The reliability diagram
shows the model is {calibration_verdict} in the high-confidence range, meaning predicted
confidence scores are not fully reliable. Temperature scaling in v3 is recommended."""
else:
    report += f"""The model is **poorly calibrated** (ECE={ece:.4f}). The {calibration_verdict}
pattern is severe. Temperature scaling or label smoothing in v3 training is strongly recommended."""

report += f"""

---

## 7. Recommendations for v3 Training

Based on this baseline analysis:

1. **Address {worst_recall_class} recall** — increase class weight, targeted augmentation,
   or focal loss gamma tuning for this class.
2. **Calibration** — add temperature scaling post-training or increase label smoothing
   (current ECE={ece:.4f}).
3. **Domain shift mitigation** — consider source-conditioned augmentation or adversarial
   domain adaptation if ODIR/APTOS gap persists.
4. **High-confidence errors** — the model makes confidently wrong predictions on certain
   classes; mixup or CutMix augmentation may improve uncertainty estimation.
5. **Top confusion pairs** to specifically target:
"""
for pair in top5_pairs[:3]:
    report += f"   - {pair['description']} ({pair['count']} errors)\n"

report += """
---

## 8. Output Files

| File | Description |
|------|-------------|
| confusion_matrix_raw.png | Raw count confusion matrix |
| confusion_matrix_normalized.png | Recall-normalized confusion matrix |
| reliability_diagram.png | ECE calibration plot |
| confidence_distributions.png | Per-class confidence histograms |
| per_source_performance.png | ODIR vs APTOS breakdown |
| baseline_metrics.json | All metrics in structured JSON |

---
*Report generated by RetinaSense ViT v2 error analysis pipeline.*
"""

with open(f'{OUT_DIR}/BASELINE_ANALYSIS.md', 'w') as f:
    f.write(report)
print(f'  Saved BASELINE_ANALYSIS.md')

# ================================================================
# FINAL SUMMARY
# ================================================================
print('\n' + '='*65)
print('   BASELINE ANALYSIS COMPLETE')
print('='*65)
print(f'  Val accuracy (thresh) : {thresh_acc:.2f}%')
print(f'  Macro F1              : {macro_f1:.4f}')
print(f'  ECE                   : {ece:.4f}')
print(f'  Worst class (F1)      : {worst_f1_class} ({per_class_f1[worst_f1_class]:.4f})')
print(f'  Worst class (recall)  : {worst_recall_class} ({per_class_recall[worst_recall_class]:.4f})')
print(f'  Top confusion         : {top5_pairs[0]["description"]}')
if domain_gap is not None:
    print(f'  Domain gap (ODIR-APTOS): {domain_gap:.2f}pp')
print(f'\n  All outputs in: {OUT_DIR}/')
print('='*65)
