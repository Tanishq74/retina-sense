#!/usr/bin/env python3
"""
Threshold Optimization for RetinaSense v2 Extended
===================================================
Applies per-class threshold optimization to the v2_extended model.
Uses the same data pipeline as retinasense_v2_extended.py for consistency.
"""

import os, warnings, json
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, classification_report
)

# ═══════════════════════════════════════════════════════════
# CONFIG (must match v2_extended training)
# ═══════════════════════════════════════════════════════════
MODEL_PATH  = './outputs_v2_extended/best_model.pth'
OUTPUT_DIR  = './outputs_v2_extended'
CACHE_DIR   = './preprocessed_cache'
BATCH_SIZE  = 64
NUM_WORKERS = 8
IMG_SIZE    = 300

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = ['Normal', 'Diabetes/DR', 'Glaucoma', 'Cataract', 'AMD']
NUM_CLASSES = len(CLASS_NAMES)

print('='*60)
print('  Threshold Optimization — RetinaSense v2 Extended')
print('='*60)
print(f'  Device: {device}')

# ═══════════════════════════════════════════════════════════
# 1. REBUILD EXACT SAME VALIDATION SET
# ═══════════════════════════════════════════════════════════
print('\n[1/5] Rebuilding validation set (same split as training)...')

BASE = './'
disease_cols = ['N','D','G','C','A']
label_map = {'N':0,'D':1,'G':2,'C':3,'A':4}

df_odir = pd.read_csv(f'{BASE}/odir/full_df.csv')
df_odir['disease_count'] = df_odir[disease_cols].sum(axis=1)
df_odir = df_odir[df_odir['disease_count']==1].copy()

def get_label(row):
    for d in disease_cols:
        if row[d]==1: return label_map[d]

df_odir['disease_label'] = df_odir.apply(get_label, axis=1)
img_col = next(c for c in df_odir.columns
               if any(k in c.lower() for k in ['filename','fundus','image']))

odir_meta = pd.DataFrame({
    'image_path':    f'{BASE}/odir/preprocessed_images/'+df_odir[img_col].astype(str),
    'dataset':       'ODIR',
    'disease_label': df_odir['disease_label'],
    'severity_label':-1
})

df_aptos = pd.read_csv(f'{BASE}/aptos/train.csv')
aptos_meta = pd.DataFrame({
    'image_path':    f'{BASE}/aptos/train_images/'+df_aptos['id_code']+'.png',
    'dataset':       'APTOS',
    'disease_label': 1,
    'severity_label':df_aptos['diagnosis']
})

meta = pd.concat([odir_meta, aptos_meta], ignore_index=True)
meta = meta[meta['image_path'].apply(os.path.exists)].reset_index(drop=True)

# Build cache paths (same as training)
cache_paths = []
for _, row in meta.iterrows():
    stem = os.path.splitext(os.path.basename(row['image_path']))[0]
    fp = f'{CACHE_DIR}/{stem}_{IMG_SIZE}.npy'
    cache_paths.append(fp)
meta['cache_path'] = cache_paths

# Same split as training (random_state=42, test_size=0.2)
_, val_df = train_test_split(
    meta, test_size=0.2, stratify=meta['disease_label'], random_state=42)

print(f'  Val samples: {len(val_df)}')
dist = val_df['disease_label'].value_counts().sort_index()
for i, cnt in dist.items():
    print(f'    {CLASS_NAMES[i]:15s}: {cnt:4d}')

# ═══════════════════════════════════════════════════════════
# 2. DATASET + MODEL
# ═══════════════════════════════════════════════════════════
print('\n[2/5] Loading model and creating dataloader...')

val_tfm = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

class RetDS(Dataset):
    def __init__(self, df, tfm):
        self.df  = df.reset_index(drop=True)
        self.tfm = tfm
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        try:    img = np.load(r['cache_path'])
        except: img = np.zeros((IMG_SIZE,IMG_SIZE,3), dtype=np.uint8)
        return (self.tfm(img),
                torch.tensor(int(r['disease_label']), dtype=torch.long))

val_ds = RetDS(val_df, val_tfm)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

class MultiTaskModel(nn.Module):
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

model = MultiTaskModel().to(device)
ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f'  Loaded checkpoint from epoch {ckpt.get("epoch", "?")}')
print(f'  Checkpoint val_acc={ckpt.get("val_acc", 0):.2f}%, macro_f1={ckpt.get("macro_f1", 0):.4f}')

# ═══════════════════════════════════════════════════════════
# 3. GET PREDICTIONS
# ═══════════════════════════════════════════════════════════
print('\n[3/5] Getting predictions...')

all_probs, all_labels = [], []
with torch.no_grad():
    for imgs, labels in tqdm(val_loader, desc='Predicting'):
        imgs = imgs.to(device, non_blocking=True)
        d_out, _ = model(imgs)
        probs = torch.softmax(d_out.float(), dim=1)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())

all_probs  = np.vstack(all_probs)
all_labels = np.concatenate(all_labels)
print(f'  Predictions: {all_probs.shape[0]} samples, {all_probs.shape[1]} classes')

# Baseline (argmax)
y_pred_baseline = np.argmax(all_probs, axis=1)
baseline_acc = 100 * (y_pred_baseline == all_labels).mean()
baseline_mf1 = f1_score(all_labels, y_pred_baseline, average='macro')
baseline_wf1 = f1_score(all_labels, y_pred_baseline, average='weighted')
try:    baseline_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
except: baseline_auc = 0.0

print(f'\n  BASELINE (argmax):')
print(f'    Accuracy:    {baseline_acc:.2f}%')
print(f'    Macro F1:    {baseline_mf1:.4f}')
print(f'    Weighted F1: {baseline_wf1:.4f}')
print(f'    Macro AUC:   {baseline_auc:.4f}')

# ═══════════════════════════════════════════════════════════
# 4. OPTIMIZE THRESHOLDS
# ═══════════════════════════════════════════════════════════
print('\n[4/5] Optimizing per-class thresholds...')

def find_optimal_threshold_ovr(y_true, y_probs_class, class_idx):
    """Find optimal threshold for one-vs-rest using F1."""
    y_binary = (y_true == class_idx).astype(int)
    thresholds = np.arange(0.05, 0.96, 0.01)
    best_f1 = 0
    best_thresh = 0.5
    for thresh in thresholds:
        y_pred = (y_probs_class >= thresh).astype(int)
        f1 = f1_score(y_binary, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh, best_f1

optimal_thresholds = {}
print(f'  {"Class":15s} {"Threshold":>10s} {"OvR F1":>8s} {"Support":>8s}')
print('  ' + '-'*45)
for ci in range(NUM_CLASSES):
    thresh, ovr_f1 = find_optimal_threshold_ovr(all_labels, all_probs[:, ci], ci)
    optimal_thresholds[ci] = thresh
    n = (all_labels == ci).sum()
    print(f'  {CLASS_NAMES[ci]:15s} {thresh:>10.3f} {ovr_f1:>8.3f} {n:>8d}')

# Apply optimized thresholds
def predict_with_thresholds(probs, thresholds):
    n = probs.shape[0]
    predictions = np.zeros(n, dtype=int)
    for i in range(n):
        p = probs[i]
        max_class = np.argmax(p)
        if p[max_class] >= thresholds[max_class]:
            predictions[i] = max_class
        else:
            sorted_cls = np.argsort(p)[::-1]
            assigned = False
            for cls in sorted_cls:
                if p[cls] >= thresholds[cls]:
                    predictions[i] = cls
                    assigned = True
                    break
            if not assigned:
                predictions[i] = max_class
    return predictions

y_pred_optimized = predict_with_thresholds(all_probs, optimal_thresholds)
opt_acc = 100 * (y_pred_optimized == all_labels).mean()
opt_mf1 = f1_score(all_labels, y_pred_optimized, average='macro')
opt_wf1 = f1_score(all_labels, y_pred_optimized, average='weighted')
try:    opt_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
except: opt_auc = 0.0

print(f'\n  OPTIMIZED (per-class thresholds):')
print(f'    Accuracy:    {opt_acc:.2f}%')
print(f'    Macro F1:    {opt_mf1:.4f}')
print(f'    Weighted F1: {opt_wf1:.4f}')
print(f'    Macro AUC:   {opt_auc:.4f}')

# ═══════════════════════════════════════════════════════════
# 5. FULL REPORT + PLOTS
# ═══════════════════════════════════════════════════════════
print('\n[5/5] Generating report and plots...')

# Per-class F1 comparison
baseline_per_f1 = f1_score(all_labels, y_pred_baseline, average=None, labels=range(NUM_CLASSES), zero_division=0)
opt_per_f1 = f1_score(all_labels, y_pred_optimized, average=None, labels=range(NUM_CLASSES), zero_division=0)

print(f'\n  {"Class":15s} {"Baseline F1":>12s} {"Optimized F1":>13s} {"Change":>8s}')
print('  ' + '-'*52)
for ci, cn in enumerate(CLASS_NAMES):
    diff = opt_per_f1[ci] - baseline_per_f1[ci]
    print(f'  {cn:15s} {baseline_per_f1[ci]:>12.4f} {opt_per_f1[ci]:>13.4f} {diff:>+8.4f}')

# Classification report
print('\n' + '='*60)
print('  OPTIMIZED CLASSIFICATION REPORT')
print('='*60)
print(classification_report(all_labels, y_pred_optimized, target_names=CLASS_NAMES, digits=4))

# Save results JSON
results = {
    'optimal_thresholds': {str(k): float(v) for k, v in optimal_thresholds.items()},
    'baseline': {
        'accuracy': float(baseline_acc),
        'macro_f1': float(baseline_mf1),
        'weighted_f1': float(baseline_wf1),
        'macro_auc': float(baseline_auc),
        'per_class_f1': {cn: float(baseline_per_f1[ci]) for ci, cn in enumerate(CLASS_NAMES)}
    },
    'optimized': {
        'accuracy': float(opt_acc),
        'macro_f1': float(opt_mf1),
        'weighted_f1': float(opt_wf1),
        'macro_auc': float(opt_auc),
        'per_class_f1': {cn: float(opt_per_f1[ci]) for ci, cn in enumerate(CLASS_NAMES)}
    }
}

json_path = f'{OUTPUT_DIR}/threshold_optimization_results.json'
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f'  Saved: {json_path}')

# Plot comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
colors_bar = ['#3498db', '#2ecc71']

# 1. Per-class F1 comparison
ax = axes[0, 0]
x = np.arange(NUM_CLASSES)
w = 0.35
ax.bar(x - w/2, baseline_per_f1, w, label='Baseline (argmax)', color=colors_bar[0], alpha=0.8)
ax.bar(x + w/2, opt_per_f1, w, label='Optimized thresholds', color=colors_bar[1], alpha=0.8)
ax.set_ylabel('F1 Score')
ax.set_title('Per-Class F1 Comparison', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 2. Overall metrics
ax = axes[0, 1]
metrics = ['Accuracy', 'Macro F1', 'Weighted F1', 'AUC-ROC']
b_vals = [baseline_acc/100, baseline_mf1, baseline_wf1, baseline_auc]
o_vals = [opt_acc/100, opt_mf1, opt_wf1, opt_auc]
x = np.arange(len(metrics))
ax.bar(x - w/2, b_vals, w, label='Baseline', color=colors_bar[0], alpha=0.8)
ax.bar(x + w/2, o_vals, w, label='Optimized', color=colors_bar[1], alpha=0.8)
ax.set_ylabel('Score')
ax.set_title('Overall Metrics Comparison', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics, rotation=45, ha='right')
ax.legend()
ax.set_ylim([0, 1])
ax.grid(axis='y', alpha=0.3)

# 3. Optimal thresholds
ax = axes[1, 0]
thresh_vals = [optimal_thresholds[i] for i in range(NUM_CLASSES)]
bars = ax.bar(CLASS_NAMES, thresh_vals, alpha=0.8, color='steelblue')
ax.axhline(y=0.5, color='red', linestyle='--', label='Default (0.5)', alpha=0.5)
ax.set_ylabel('Optimal Threshold')
ax.set_title('Optimized Thresholds per Class', fontweight='bold')
ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
ax.legend()
ax.set_ylim([0, 1])
ax.grid(axis='y', alpha=0.3)
for bar, t in zip(bars, thresh_vals):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{t:.2f}', ha='center', va='bottom', fontsize=9)

# 4. Improvement bars
ax = axes[1, 1]
improvements = opt_per_f1 - baseline_per_f1
bar_colors = ['green' if x >= 0 else 'red' for x in improvements]
bars = ax.barh(CLASS_NAMES, improvements, color=bar_colors, alpha=0.7)
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax.set_xlabel('F1 Score Change')
ax.set_title('Per-Class F1 Improvement', fontweight='bold')
ax.grid(axis='x', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, improvements)):
    x_pos = val + (0.005 if val >= 0 else -0.005)
    ha = 'left' if val >= 0 else 'right'
    ax.text(x_pos, i, f'{val:+.3f}', va='center', ha=ha, fontsize=9)

plt.suptitle(f'v2 Extended Threshold Optimization: {baseline_acc:.1f}% -> {opt_acc:.1f}% Acc | {baseline_mf1:.3f} -> {opt_mf1:.3f} mF1',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plot_path = f'{OUTPUT_DIR}/threshold_comparison.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved: {plot_path}')

# Confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, preds, title in [(axes[0], y_pred_baseline, 'Baseline (argmax)'),
                          (axes[1], y_pred_optimized, 'Optimized Thresholds')]:
    cm = confusion_matrix(all_labels, preds)
    cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_n, annot=True, fmt='.2f', cmap='Blues', ax=ax,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel('True')
    ax.set_xlabel('Predicted')
plt.tight_layout()
cm_path = f'{OUTPUT_DIR}/confusion_matrix_threshold.png'
plt.savefig(cm_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved: {cm_path}')

# Final summary
print(f'\n{"="*60}')
print(f'  THRESHOLD OPTIMIZATION SUMMARY')
print(f'{"="*60}')
print(f'  Baseline Accuracy:   {baseline_acc:.2f}%')
print(f'  Optimized Accuracy:  {opt_acc:.2f}%')
print(f'  Improvement:         {opt_acc - baseline_acc:+.2f}%')
print(f'')
print(f'  Baseline Macro F1:   {baseline_mf1:.4f}')
print(f'  Optimized Macro F1:  {opt_mf1:.4f}')
print(f'  Improvement:         {opt_mf1 - baseline_mf1:+.4f}')
print(f'{"="*60}')
