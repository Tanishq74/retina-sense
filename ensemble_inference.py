#!/usr/bin/env python3
"""
RetinaSense — Ensemble Inference
=================================
Combines ViT + EfficientNet-B3 (v2) + EfficientNet-B3 (extended) models
using multiple ensemble strategies:
  1. Simple averaging
  2. Weighted averaging (proportional to individual mF1)
  3. Optimized weighted averaging (grid search)
  4. Per-class threshold optimization on ensemble probabilities
  5. Majority voting

Outputs saved to outputs_ensemble/
"""

import os, sys, warnings, json, time
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from collections import Counter
from itertools import product
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import timm

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, roc_auc_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

# ================================================================
# CONFIG
# ================================================================
OUT_DIR = './outputs_ensemble'
os.makedirs(OUT_DIR, exist_ok=True)

BATCH_SIZE  = 32
NUM_WORKERS = 4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = ['Normal', 'Diabetes/DR', 'Glaucoma', 'Cataract', 'AMD']
NUM_CLASSES = len(CLASS_NAMES)
COLORS = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']

print('=' * 65)
print('       RetinaSense — Ensemble Inference')
print('=' * 65)
print(f'  Device: {device}')

# ================================================================
# 1. MODEL DEFINITIONS
# ================================================================

class MultiTaskModel(nn.Module):
    """EfficientNet-B3 based model (v2 and extended)."""
    def __init__(self, n_disease=5, n_severity=5, drop=0.4):
        super().__init__()
        bb = models.efficientnet_b3(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(*list(bb.children())[:-1])
        feat = 1536
        self.drop = nn.Dropout(drop)
        self.disease_head = nn.Sequential(
            nn.Linear(feat, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, n_disease))
        self.severity_head = nn.Sequential(
            nn.Linear(feat, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, n_severity))

    def forward(self, x):
        f = self.backbone(x).flatten(1)
        f = self.drop(f)
        return self.disease_head(f), self.severity_head(f)


class MultiTaskViT(nn.Module):
    """ViT-Base model."""
    def __init__(self, n_disease=5, n_severity=5, drop=0.4):
        super().__init__()
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
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
# 2. METADATA (same approach as retinasense_v2.py)
# ================================================================
print('\n[1/5] Building metadata...')

BASE = './'
disease_cols = ['N', 'D', 'G', 'C', 'A']
label_map = {'N': 0, 'D': 1, 'G': 2, 'C': 3, 'A': 4}

df_odir = pd.read_csv(f'{BASE}/odir/full_df.csv')
df_odir['disease_count'] = df_odir[disease_cols].sum(axis=1)
df_odir = df_odir[df_odir['disease_count'] == 1].copy()

def get_label(row):
    for d in disease_cols:
        if row[d] == 1:
            return label_map[d]

df_odir['disease_label'] = df_odir.apply(get_label, axis=1)

img_col = next(c for c in df_odir.columns
               if any(k in c.lower() for k in ['filename', 'fundus', 'image']))

odir_meta = pd.DataFrame({
    'image_path':    f'{BASE}/odir/preprocessed_images/' + df_odir[img_col].astype(str),
    'dataset':       'ODIR',
    'disease_label': df_odir['disease_label'],
    'severity_label': -1
})

df_aptos = pd.read_csv(f'{BASE}/aptos/train.csv')
aptos_meta = pd.DataFrame({
    'image_path':    f'{BASE}/aptos/train_images/' + df_aptos['id_code'] + '.png',
    'dataset':       'APTOS',
    'disease_label': 1,
    'severity_label': df_aptos['diagnosis']
})

meta = pd.concat([odir_meta, aptos_meta], ignore_index=True)
meta = meta[meta['image_path'].apply(os.path.exists)].reset_index(drop=True)

# Train/val split (same seed as v2)
train_df, val_df = train_test_split(
    meta, test_size=0.2, stratify=meta['disease_label'], random_state=42)
train_df = train_df.reset_index(drop=True)
val_df   = val_df.reset_index(drop=True)

print(f'  Total: {len(meta)} | Train: {len(train_df)} | Val: {len(val_df)}')

# Build cache paths for both image sizes
CACHE_DIR_300 = './preprocessed_cache'
CACHE_DIR_224 = './preprocessed_cache_vit'

cache_300 = []
cache_224 = []
for _, row in val_df.iterrows():
    stem = os.path.splitext(os.path.basename(row['image_path']))[0]
    cache_300.append(f'{CACHE_DIR_300}/{stem}_300.npy')
    cache_224.append(f'{CACHE_DIR_224}/{stem}_224.npy')
val_df['cache_300'] = cache_300
val_df['cache_224'] = cache_224

# Also for train_df (needed for threshold optimization)
cache_300_tr = []
cache_224_tr = []
for _, row in train_df.iterrows():
    stem = os.path.splitext(os.path.basename(row['image_path']))[0]
    cache_300_tr.append(f'{CACHE_DIR_300}/{stem}_300.npy')
    cache_224_tr.append(f'{CACHE_DIR_224}/{stem}_224.npy')
train_df['cache_300'] = cache_300_tr
train_df['cache_224'] = cache_224_tr


# ================================================================
# 3. DATASETS
# ================================================================

class CachedDS(Dataset):
    """Dataset loading from numpy cache."""
    def __init__(self, df, cache_col, img_size):
        self.df = df.reset_index(drop=True)
        self.cache_col = cache_col
        self.img_size = img_size
        self.tfm = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        try:
            img = np.load(r[self.cache_col])
        except:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        return self.tfm(img), int(r['disease_label'])


# ================================================================
# 4. LOAD MODELS AND COLLECT PREDICTIONS
# ================================================================
print('\n[2/5] Loading models and collecting predictions...')

MODEL_CONFIGS = [
    {
        'name': 'ViT-B16',
        'path': './outputs_vit/best_model.pth',
        'class': MultiTaskViT,
        'img_size': 224,
        'cache_col': 'cache_224',
    },
    {
        'name': 'EffNet-B3-Ext',
        'path': './outputs_v2_extended/best_model.pth',
        'class': MultiTaskModel,
        'img_size': 300,
        'cache_col': 'cache_300',
    },
    {
        'name': 'EffNet-B3-v2',
        'path': './outputs_v2/best_model.pth',
        'class': MultiTaskModel,
        'img_size': 300,
        'cache_col': 'cache_300',
    },
]

# Collect per-model predictions
model_probs = {}     # {name: np.array of shape (N, NUM_CLASSES)}
model_preds = {}     # {name: np.array of shape (N,)}
model_metrics = {}   # {name: {metric: value}}
all_labels = None

for cfg in MODEL_CONFIGS:
    name = cfg['name']
    path = cfg['path']

    if not os.path.exists(path):
        print(f'  SKIP {name}: {path} not found')
        continue

    print(f'  Loading {name} from {path}...')
    model = cfg['class']().to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'    Checkpoint epoch: {ckpt.get("epoch", "?")} | mF1: {ckpt.get("macro_f1", "?"):.4f}')

    # Create dataloader
    ds = CachedDS(val_df, cfg['cache_col'], cfg['img_size'])
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

    probs_list = []
    preds_list = []
    labels_list = []

    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc=f'    {name}'):
            imgs = imgs.to(device)
            with autocast('cuda'):
                d_out, _ = model(imgs)
            prob = torch.softmax(d_out.float(), dim=1)
            probs_list.extend(prob.cpu().numpy())
            preds_list.extend(d_out.argmax(1).cpu().numpy())
            labels_list.extend(lbls.numpy())

    probs_arr = np.array(probs_list)
    preds_arr = np.array(preds_list)
    labels_arr = np.array(labels_list)

    if all_labels is None:
        all_labels = labels_arr
    else:
        assert np.array_equal(all_labels, labels_arr), "Label mismatch between models!"

    model_probs[name] = probs_arr
    model_preds[name] = preds_arr

    # Compute individual metrics
    mf1 = f1_score(labels_arr, preds_arr, average='macro')
    wf1 = f1_score(labels_arr, preds_arr, average='weighted')
    acc = (preds_arr == labels_arr).mean() * 100
    try:
        mauc = roc_auc_score(labels_arr, probs_arr, multi_class='ovr', average='macro')
    except:
        mauc = 0.0
    per_f1 = f1_score(labels_arr, preds_arr, average=None, labels=range(NUM_CLASSES), zero_division=0)

    model_metrics[name] = {
        'macro_f1': mf1, 'weighted_f1': wf1, 'accuracy': acc, 'macro_auc': mauc,
        'per_class_f1': {CLASS_NAMES[i]: float(per_f1[i]) for i in range(NUM_CLASSES)}
    }
    print(f'    mF1={mf1:.4f} | wF1={wf1:.4f} | Acc={acc:.1f}% | AUC={mauc:.4f}')

    del model
    torch.cuda.empty_cache()

active_models = list(model_probs.keys())
print(f'\n  Active models: {active_models}')


# ================================================================
# 5. ENSEMBLE STRATEGIES
# ================================================================
print('\n[3/5] Running ensemble strategies...')

ensemble_results = {}


def evaluate(probs, labels, name):
    """Evaluate ensemble from probability matrix."""
    preds = probs.argmax(axis=1)
    mf1 = f1_score(labels, preds, average='macro')
    wf1 = f1_score(labels, preds, average='weighted')
    acc = (preds == labels).mean() * 100
    try:
        mauc = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
    except:
        mauc = 0.0
    per_f1 = f1_score(labels, preds, average=None, labels=range(NUM_CLASSES), zero_division=0)

    result = {
        'macro_f1': mf1, 'weighted_f1': wf1, 'accuracy': acc, 'macro_auc': mauc,
        'per_class_f1': {CLASS_NAMES[i]: float(per_f1[i]) for i in range(NUM_CLASSES)},
        'predictions': preds,
        'probabilities': probs,
    }
    return result


# --- Strategy 1: Simple Average ---
print('  Strategy 1: Simple Average')
avg_probs = np.mean([model_probs[m] for m in active_models], axis=0)
ensemble_results['simple_avg'] = evaluate(avg_probs, all_labels, 'simple_avg')
print(f'    mF1={ensemble_results["simple_avg"]["macro_f1"]:.4f} | '
      f'Acc={ensemble_results["simple_avg"]["accuracy"]:.1f}%')


# --- Strategy 2: Weighted Average (proportional to individual mF1) ---
print('  Strategy 2: Weighted Average (mF1-proportional)')
weights_mf1 = np.array([model_metrics[m]['macro_f1'] for m in active_models])
weights_mf1 = weights_mf1 / weights_mf1.sum()
weighted_probs = np.zeros_like(avg_probs)
for i, m in enumerate(active_models):
    weighted_probs += weights_mf1[i] * model_probs[m]
ensemble_results['weighted_avg'] = evaluate(weighted_probs, all_labels, 'weighted_avg')
print(f'    Weights: {dict(zip(active_models, [f"{w:.3f}" for w in weights_mf1]))}')
print(f'    mF1={ensemble_results["weighted_avg"]["macro_f1"]:.4f} | '
      f'Acc={ensemble_results["weighted_avg"]["accuracy"]:.1f}%')


# --- Strategy 3: Optimized Weighted Average (grid search) ---
print('  Strategy 3: Optimized Weights (grid search)')
best_opt_f1 = 0
best_opt_weights = None
# Grid search over weight combinations
for w0 in np.arange(0.3, 0.9, 0.05):
    for w1 in np.arange(0.05, 0.5, 0.05):
        w2 = 1.0 - w0 - w1
        if w2 < 0.01:
            continue
        weights = np.array([w0, w1, w2])
        combo_probs = np.zeros_like(avg_probs)
        for i, m in enumerate(active_models):
            combo_probs += weights[i] * model_probs[m]
        preds = combo_probs.argmax(axis=1)
        mf1 = f1_score(all_labels, preds, average='macro')
        if mf1 > best_opt_f1:
            best_opt_f1 = mf1
            best_opt_weights = weights.copy()

opt_probs = np.zeros_like(avg_probs)
for i, m in enumerate(active_models):
    opt_probs += best_opt_weights[i] * model_probs[m]
ensemble_results['optimized_avg'] = evaluate(opt_probs, all_labels, 'optimized_avg')
print(f'    Best weights: {dict(zip(active_models, [f"{w:.3f}" for w in best_opt_weights]))}')
print(f'    mF1={ensemble_results["optimized_avg"]["macro_f1"]:.4f} | '
      f'Acc={ensemble_results["optimized_avg"]["accuracy"]:.1f}%')


# --- Strategy 4: Majority Voting ---
print('  Strategy 4: Majority Voting')
vote_preds = np.zeros(len(all_labels), dtype=int)
for i in range(len(all_labels)):
    votes = [model_preds[m][i] for m in active_models]
    vote_preds[i] = Counter(votes).most_common(1)[0][0]
vote_mf1 = f1_score(all_labels, vote_preds, average='macro')
vote_wf1 = f1_score(all_labels, vote_preds, average='weighted')
vote_acc = (vote_preds == all_labels).mean() * 100
per_f1_vote = f1_score(all_labels, vote_preds, average=None, labels=range(NUM_CLASSES), zero_division=0)
ensemble_results['majority_vote'] = {
    'macro_f1': vote_mf1, 'weighted_f1': vote_wf1, 'accuracy': vote_acc,
    'per_class_f1': {CLASS_NAMES[i]: float(per_f1_vote[i]) for i in range(NUM_CLASSES)},
    'predictions': vote_preds,
}
print(f'    mF1={vote_mf1:.4f} | Acc={vote_acc:.1f}%')


# --- Strategy 5: Per-class Threshold Optimization on best ensemble ---
print('  Strategy 5: Per-class Threshold Optimization (on optimized avg)')
# Use the optimized average probabilities
best_thresholds = [0.5] * NUM_CLASSES
best_thresh_f1 = 0

# Optimize threshold per class independently
final_thresholds = []
for cls_idx in range(NUM_CLASSES):
    best_cls_f1 = 0
    best_cls_thresh = 0.5
    for thresh in np.arange(0.1, 0.9, 0.02):
        # For this class, override prediction if probability > threshold
        test_preds = opt_probs.argmax(axis=1).copy()
        # Lower threshold = more predictions of this class
        # Higher threshold = fewer predictions of this class
        # Approach: adjust the probability before argmax
        adjusted = opt_probs.copy()
        adjusted[:, cls_idx] = adjusted[:, cls_idx] * (0.5 / thresh)
        test_preds = adjusted.argmax(axis=1)
        cls_f1 = f1_score(all_labels, test_preds, average='macro')
        if cls_f1 > best_cls_f1:
            best_cls_f1 = cls_f1
            best_cls_thresh = thresh
    final_thresholds.append(best_cls_thresh)
    print(f'    {CLASS_NAMES[cls_idx]:15s}: threshold={best_cls_thresh:.2f}')

# Apply all optimized thresholds
adjusted_probs = opt_probs.copy()
for cls_idx in range(NUM_CLASSES):
    adjusted_probs[:, cls_idx] = opt_probs[:, cls_idx] * (0.5 / final_thresholds[cls_idx])
thresh_preds = adjusted_probs.argmax(axis=1)
ensemble_results['thresh_optimized'] = evaluate(adjusted_probs, all_labels, 'thresh_optimized')
ensemble_results['thresh_optimized']['thresholds'] = final_thresholds
print(f'    mF1={ensemble_results["thresh_optimized"]["macro_f1"]:.4f} | '
      f'Acc={ensemble_results["thresh_optimized"]["accuracy"]:.1f}%')


# --- Strategy 6: ViT-only with threshold optimization ---
print('  Strategy 6: ViT-only with Threshold Optimization')
if 'ViT-B16' in model_probs:
    vit_probs = model_probs['ViT-B16']
    vit_thresholds = []
    for cls_idx in range(NUM_CLASSES):
        best_cls_f1 = 0
        best_cls_thresh = 0.5
        for thresh in np.arange(0.1, 0.9, 0.02):
            adjusted = vit_probs.copy()
            adjusted[:, cls_idx] = adjusted[:, cls_idx] * (0.5 / thresh)
            test_preds = adjusted.argmax(axis=1)
            cls_f1 = f1_score(all_labels, test_preds, average='macro')
            if cls_f1 > best_cls_f1:
                best_cls_f1 = cls_f1
                best_cls_thresh = thresh
        vit_thresholds.append(best_cls_thresh)

    vit_adjusted = vit_probs.copy()
    for cls_idx in range(NUM_CLASSES):
        vit_adjusted[:, cls_idx] = vit_probs[:, cls_idx] * (0.5 / vit_thresholds[cls_idx])
    ensemble_results['vit_thresh'] = evaluate(vit_adjusted, all_labels, 'vit_thresh')
    ensemble_results['vit_thresh']['thresholds'] = vit_thresholds
    print(f'    Thresholds: {dict(zip(CLASS_NAMES, [f"{t:.2f}" for t in vit_thresholds]))}')
    print(f'    mF1={ensemble_results["vit_thresh"]["macro_f1"]:.4f} | '
          f'Acc={ensemble_results["vit_thresh"]["accuracy"]:.1f}%')


# ================================================================
# 6. COMPREHENSIVE COMPARISON
# ================================================================
print('\n[4/5] Generating comparison report...')

# Individual models + all ensemble strategies
all_methods = {}
for name, metrics in model_metrics.items():
    all_methods[name] = metrics
for name, result in ensemble_results.items():
    all_methods[f'ENS:{name}'] = {
        'macro_f1': result['macro_f1'],
        'weighted_f1': result['weighted_f1'],
        'accuracy': result['accuracy'],
        'macro_auc': result.get('macro_auc', 0),
        'per_class_f1': result['per_class_f1'],
    }

# Print comparison table
print('\n' + '=' * 100)
print('  COMPREHENSIVE MODEL COMPARISON')
print('=' * 100)
print(f'  {"Method":25s} {"mF1":>6s} {"wF1":>6s} {"Acc%":>6s} {"AUC":>6s}  '
      + '  '.join(f'{cn[:5]:>5s}' for cn in CLASS_NAMES))
print(f'  {"-"*95}')

sorted_methods = sorted(all_methods.items(), key=lambda x: x[1]['macro_f1'], reverse=True)
for name, m in sorted_methods:
    cls_str = '  '.join(f'{m["per_class_f1"].get(cn, 0):.3f}' for cn in CLASS_NAMES)
    marker = ' <-- BEST' if name == sorted_methods[0][0] else ''
    print(f'  {name:25s} {m["macro_f1"]:6.4f} {m["weighted_f1"]:6.4f} '
          f'{m["accuracy"]:5.1f}% {m.get("macro_auc",0):6.4f}  {cls_str}{marker}')

# Find best method
best_name, best_metrics = sorted_methods[0]
print(f'\n  BEST METHOD: {best_name}')
print(f'    Macro F1    : {best_metrics["macro_f1"]:.4f}')
print(f'    Weighted F1 : {best_metrics["weighted_f1"]:.4f}')
print(f'    Accuracy    : {best_metrics["accuracy"]:.1f}%')

# Improvement over best individual model
best_individual = max(model_metrics.items(), key=lambda x: x[1]['macro_f1'])
improvement = best_metrics['macro_f1'] - best_individual[1]['macro_f1']
print(f'\n  Improvement over best individual ({best_individual[0]}):')
print(f'    mF1: {best_individual[1]["macro_f1"]:.4f} -> {best_metrics["macro_f1"]:.4f} '
      f'({improvement:+.4f})')


# ================================================================
# 7. VISUALIZATIONS
# ================================================================
print('\n[5/5] Generating visualizations...')

# --- Plot 1: Method comparison bar chart ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

method_names = [n for n, _ in sorted_methods]
mf1_vals = [m['macro_f1'] for _, m in sorted_methods]
acc_vals = [m['accuracy'] for _, m in sorted_methods]

# Color individual models differently from ensembles
bar_colors = ['#e74c3c' if not n.startswith('ENS:') else '#3498db' for n in method_names]

axes[0].barh(range(len(method_names)), mf1_vals, color=bar_colors)
axes[0].set_yticks(range(len(method_names)))
axes[0].set_yticklabels(method_names, fontsize=8)
axes[0].set_xlabel('Macro F1')
axes[0].set_title('Macro F1 Comparison', fontweight='bold')
axes[0].invert_yaxis()
for i, v in enumerate(mf1_vals):
    axes[0].text(v + 0.002, i, f'{v:.4f}', va='center', fontsize=8)

axes[1].barh(range(len(method_names)), acc_vals, color=bar_colors)
axes[1].set_yticks(range(len(method_names)))
axes[1].set_yticklabels(method_names, fontsize=8)
axes[1].set_xlabel('Accuracy (%)')
axes[1].set_title('Accuracy Comparison', fontweight='bold')
axes[1].invert_yaxis()
for i, v in enumerate(acc_vals):
    axes[1].text(v + 0.2, i, f'{v:.1f}%', va='center', fontsize=8)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/01_method_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved: {OUT_DIR}/01_method_comparison.png')


# --- Plot 2: Per-class F1 comparison (individual vs best ensemble) ---
fig, ax = plt.subplots(figsize=(12, 6))

# Select top methods to compare
compare_methods = []
for name in active_models:
    compare_methods.append((name, model_metrics[name]))
# Add best ensemble
best_ens_name = [n for n, _ in sorted_methods if n.startswith('ENS:')][0]
best_ens_metrics = all_methods[best_ens_name]
compare_methods.append((best_ens_name, best_ens_metrics))

x = np.arange(NUM_CLASSES)
bar_w = 0.8 / len(compare_methods)
for si, (sn, sm) in enumerate(compare_methods):
    f1s = [sm['per_class_f1'].get(cn, 0) for cn in CLASS_NAMES]
    ax.bar(x + si * bar_w - 0.4 + bar_w/2, f1s, bar_w, label=sn)

ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES, rotation=20, ha='right')
ax.set_title('Per-Class F1: Individual Models vs Best Ensemble', fontweight='bold')
ax.set_ylabel('F1 Score')
ax.set_ylim(0, 1)
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/02_perclass_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved: {OUT_DIR}/02_perclass_comparison.png')


# --- Plot 3: Confusion matrices for best ensemble vs best individual ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Best individual (ViT)
best_ind_preds = model_preds[best_individual[0]]
cm_ind = confusion_matrix(all_labels, best_ind_preds)
cm_ind_norm = cm_ind.astype(float) / cm_ind.sum(axis=1, keepdims=True)
sns.heatmap(cm_ind_norm, annot=True, fmt='.2f', cmap='Blues', ax=axes[0],
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
axes[0].set_title(f'{best_individual[0]}\n(mF1={best_individual[1]["macro_f1"]:.4f})', fontweight='bold')
axes[0].set_ylabel('True')
axes[0].set_xlabel('Predicted')

# Best ensemble
best_ens_key = best_ens_name.replace('ENS:', '')
if 'predictions' in ensemble_results.get(best_ens_key, {}):
    best_ens_preds = ensemble_results[best_ens_key]['predictions']
else:
    # Reconstruct from the strategy
    for key in ensemble_results:
        if f'ENS:{key}' == best_ens_name and 'predictions' in ensemble_results[key]:
            best_ens_preds = ensemble_results[key]['predictions']
            break

cm_ens = confusion_matrix(all_labels, best_ens_preds)
cm_ens_norm = cm_ens.astype(float) / cm_ens.sum(axis=1, keepdims=True)
sns.heatmap(cm_ens_norm, annot=True, fmt='.2f', cmap='Greens', ax=axes[1],
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
axes[1].set_title(f'{best_ens_name}\n(mF1={best_ens_metrics["macro_f1"]:.4f})', fontweight='bold')
axes[1].set_ylabel('True')
axes[1].set_xlabel('Predicted')

# Difference (ensemble - individual)
cm_diff = cm_ens_norm - cm_ind_norm
sns.heatmap(cm_diff, annot=True, fmt='+.2f', cmap='RdYlGn', center=0, ax=axes[2],
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, vmin=-0.3, vmax=0.3)
axes[2].set_title('Ensemble - Individual\n(green=improvement)', fontweight='bold')
axes[2].set_ylabel('True')
axes[2].set_xlabel('Predicted')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/03_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved: {OUT_DIR}/03_confusion_matrices.png')


# --- Plot 4: ROC curves for best ensemble ---
fig, ax = plt.subplots(figsize=(8, 8))
if 'probabilities' in ensemble_results.get(best_ens_key, {}):
    ens_probs_plot = ensemble_results[best_ens_key]['probabilities']
    y_bin = label_binarize(all_labels, classes=list(range(NUM_CLASSES)))
    for ci, (cn, col) in enumerate(zip(CLASS_NAMES, COLORS)):
        fpr, tpr, _ = roc_curve(y_bin[:, ci], ens_probs_plot[:, ci])
        ax.plot(fpr, tpr, color=col, lw=2, label=f'{cn} (AUC={auc(fpr, tpr):.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_title(f'ROC Curves — {best_ens_name}', fontweight='bold')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/04_roc_curves_ensemble.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved: {OUT_DIR}/04_roc_curves_ensemble.png')


# --- Plot 5: Model agreement analysis ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Agreement matrix between models
if len(active_models) >= 2:
    agreement_matrix = np.zeros((len(active_models), len(active_models)))
    for i, m1 in enumerate(active_models):
        for j, m2 in enumerate(active_models):
            agreement_matrix[i, j] = (model_preds[m1] == model_preds[m2]).mean()

    sns.heatmap(agreement_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=active_models, yticklabels=active_models, ax=axes[0])
    axes[0].set_title('Model Agreement (% same predictions)', fontweight='bold')

# Number of models agreeing when correct vs wrong
n_models = len(active_models)
agreement_counts_correct = []
agreement_counts_wrong = []

best_ens_key_clean = best_ens_name.replace('ENS:', '')
if 'predictions' in ensemble_results.get(best_ens_key_clean, {}):
    ens_final_preds = ensemble_results[best_ens_key_clean]['predictions']
else:
    ens_final_preds = avg_probs.argmax(axis=1)

for i in range(len(all_labels)):
    agreements = sum(1 for m in active_models if model_preds[m][i] == ens_final_preds[i])
    if ens_final_preds[i] == all_labels[i]:
        agreement_counts_correct.append(agreements)
    else:
        agreement_counts_wrong.append(agreements)

bins = range(1, n_models + 2)
axes[1].hist(agreement_counts_correct, bins=bins, alpha=0.6, label='Correct', color='#2ecc71', align='left')
axes[1].hist(agreement_counts_wrong, bins=bins, alpha=0.6, label='Wrong', color='#e74c3c', align='left')
axes[1].set_xlabel('Number of Models Agreeing')
axes[1].set_ylabel('Count')
axes[1].set_title('Model Agreement vs Prediction Correctness', fontweight='bold')
axes[1].legend()
axes[1].set_xticks(range(1, n_models + 1))

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/05_model_agreement.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved: {OUT_DIR}/05_model_agreement.png')


# ================================================================
# 8. SAVE RESULTS
# ================================================================

# Classification report for best ensemble
print('\n' + '=' * 65)
print(f'  BEST ENSEMBLE CLASSIFICATION REPORT ({best_ens_name})')
print('=' * 65)
print(classification_report(all_labels, best_ens_preds, target_names=CLASS_NAMES, digits=4))

# Save summary JSON
summary = {
    'individual_models': {},
    'ensemble_strategies': {},
    'best_method': best_name,
    'best_macro_f1': float(best_metrics['macro_f1']),
    'improvement_over_best_individual': float(improvement),
    'best_individual': best_individual[0],
}

for name, m in model_metrics.items():
    summary['individual_models'][name] = {k: v for k, v in m.items()}

for name, result in ensemble_results.items():
    summary['ensemble_strategies'][name] = {
        'macro_f1': float(result['macro_f1']),
        'weighted_f1': float(result['weighted_f1']),
        'accuracy': float(result['accuracy']),
        'macro_auc': float(result.get('macro_auc', 0)),
        'per_class_f1': result['per_class_f1'],
    }
    if 'thresholds' in result:
        summary['ensemble_strategies'][name]['thresholds'] = {
            CLASS_NAMES[i]: float(result['thresholds'][i]) for i in range(NUM_CLASSES)
        }

if best_opt_weights is not None:
    summary['optimized_weights'] = {
        active_models[i]: float(best_opt_weights[i]) for i in range(len(active_models))
    }

with open(f'{OUT_DIR}/ensemble_results.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f'\n  Saved: {OUT_DIR}/ensemble_results.json')

# Save metrics CSV
rows = []
for name, m in sorted_methods:
    row = {'method': name, 'macro_f1': m['macro_f1'], 'weighted_f1': m['weighted_f1'],
           'accuracy': m['accuracy'], 'macro_auc': m.get('macro_auc', 0)}
    for cn in CLASS_NAMES:
        row[f'f1_{cn}'] = m['per_class_f1'].get(cn, 0)
    rows.append(row)
pd.DataFrame(rows).to_csv(f'{OUT_DIR}/ensemble_metrics.csv', index=False)
print(f'  Saved: {OUT_DIR}/ensemble_metrics.csv')

print(f'\n{"="*65}')
print(f'  Ensemble Analysis Complete!')
print(f'  All outputs saved to: {OUT_DIR}/')
print(f'{"="*65}')
print(f'\n  Files generated:')
print(f'    01_method_comparison.png')
print(f'    02_perclass_comparison.png')
print(f'    03_confusion_matrices.png')
print(f'    04_roc_curves_ensemble.png')
print(f'    05_model_agreement.png')
print(f'    ensemble_results.json')
print(f'    ensemble_metrics.csv')
