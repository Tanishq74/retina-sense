#!/usr/bin/env python3
"""
Threshold Optimization for RetinaSense v2
==========================================
Quick script to optimize classification thresholds
"""

import os, warnings
import numpy as np
import pandas as pd
from pathlib import Path
import json
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = './outputs_v2/best_model.pth'
OUTPUT_DIR = Path('./outputs_v2')
CACHE_DIR = './preprocessed_cache'
IMG_SIZE = 300
BATCH_SIZE = 64
NUM_WORKERS = 8
CLASS_NAMES = ['Normal', 'Diabetes/DR', 'Glaucoma', 'Cataract', 'AMD']

print("🎯 Threshold Optimization for RetinaSense v2")
print("="*50)
print(f"Device: {device}")

# ============================================================
# 1. DATA LOADING (same as v2)
# ============================================================
print("\n[1/3] Loading data...")
BASE = './'
disease_cols = ['N','D','G','C','A']
label_map = {'N':0,'D':1,'G':2,'C':3,'A':4}

# ODIR
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
    'image_path': f'{BASE}/odir/preprocessed_images/'+df_odir[img_col].astype(str),
    'dataset': 'ODIR',
    'disease_label': df_odir['disease_label'],
    'severity_label': -1
})

# APTOS
df_aptos = pd.read_csv(f'{BASE}/aptos/train.csv')
aptos_meta = pd.DataFrame({
    'image_path': f'{BASE}/aptos/train_images/'+df_aptos['id_code']+'.png',
    'dataset': 'APTOS',
    'disease_label': 1,
    'severity_label': df_aptos['diagnosis']
})

# Combine
meta = pd.concat([odir_meta, aptos_meta], ignore_index=True)
meta = meta[meta['image_path'].apply(os.path.exists)].reset_index(drop=True)

# Add cache paths
cache_paths = []
for _, row in meta.iterrows():
    stem = os.path.splitext(os.path.basename(row['image_path']))[0]
    cache_paths.append(f'{CACHE_DIR}/{stem}_{IMG_SIZE}.npy')
meta['cache_path'] = cache_paths

# Split (same random_state as v2)
train_df, val_df = train_test_split(
    meta, test_size=0.2, stratify=meta['disease_label'], random_state=42)

print(f"Train: {len(train_df)}, Val: {len(val_df)}")

# ============================================================
# 2. MODEL
# ============================================================
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

# Load model
print("\n[2/3] Loading model...")
model = MultiTaskModel(n_disease=5, n_severity=5, drop=0.4)
ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model = model.to(device)
model.eval()
print(f"✅ Loaded checkpoint from epoch {ckpt['epoch']}")

# ============================================================
# 3. DATASET
# ============================================================
val_tfm = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

class RetDS(Dataset):
    def __init__(self, df, tfm):
        self.df = df.reset_index(drop=True)
        self.tfm = tfm
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        try:
            img = np.load(r['cache_path'])
        except:
            img = np.zeros((IMG_SIZE,IMG_SIZE,3), dtype=np.uint8)
        return (self.tfm(img),
                torch.tensor(int(r['disease_label']), dtype=torch.long),
                torch.tensor(int(r['severity_label']), dtype=torch.long))

val_ds = RetDS(val_df, val_tfm)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

# ============================================================
# 4. GET PREDICTIONS
# ============================================================
print("\n[3/3] Getting predictions...")
all_probs, all_labels = [], []

with torch.no_grad():
    for imgs, diseases, severities in tqdm(val_loader):
        imgs = imgs.to(device, non_blocking=True)
        disease_logits, _ = model(imgs)
        probs = torch.softmax(disease_logits, dim=1)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(diseases.numpy())

y_probs = np.vstack(all_probs)
y_true = np.concatenate(all_labels)
print(f"✅ Got predictions for {len(y_true)} samples")

# ============================================================
# 5. BASELINE (argmax)
# ============================================================
y_pred_baseline = np.argmax(y_probs, axis=1)
acc_baseline = (y_true == y_pred_baseline).mean() * 100
f1_macro_baseline = f1_score(y_true, y_pred_baseline, average='macro', zero_division=0)
f1_weighted_baseline = f1_score(y_true, y_pred_baseline, average='weighted', zero_division=0)

print("\n" + "="*50)
print("BASELINE (argmax)")
print("="*50)
print(f"Accuracy:    {acc_baseline:.2f}%")
print(f"Macro F1:    {f1_macro_baseline:.3f}")
print(f"Weighted F1: {f1_weighted_baseline:.3f}")

f1_per_class_baseline = f1_score(y_true, y_pred_baseline, average=None, zero_division=0)
print("\nPer-class F1:")
for i, (name, f1) in enumerate(zip(CLASS_NAMES, f1_per_class_baseline)):
    support = (y_true == i).sum()
    print(f"  {name:15s}: {f1:.3f}  (n={support})")

# ============================================================
# 6. OPTIMIZE THRESHOLDS
# ============================================================
print("\n" + "="*50)
print("THRESHOLD OPTIMIZATION")
print("="*50)

def find_best_threshold(y_true, y_probs, class_idx):
    """Find optimal threshold for one-vs-rest"""
    y_binary = (y_true == class_idx).astype(int)
    thresholds = np.arange(0.05, 0.96, 0.01)
    best_f1, best_thresh = 0, 0.5

    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        f1 = f1_score(y_binary, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    return best_thresh, best_f1

optimal_thresholds = {}
for i in range(5):
    best_thresh, best_f1 = find_best_threshold(y_true, y_probs[:, i], i)
    optimal_thresholds[i] = best_thresh
    n_samples = (y_true == i).sum()
    print(f"  {CLASS_NAMES[i]:15s}: threshold={best_thresh:.3f}, F1={best_f1:.3f}, n={n_samples}")

# ============================================================
# 7. PREDICT WITH OPTIMIZED THRESHOLDS
# ============================================================
def predict_with_thresholds(y_probs, thresholds):
    """Apply per-class thresholds"""
    n_samples = y_probs.shape[0]
    predictions = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        probs = y_probs[i]
        max_class = np.argmax(probs)
        max_prob = probs[max_class]

        # If max class exceeds its threshold, predict it
        if max_prob >= thresholds[max_class]:
            predictions[i] = max_class
        else:
            # Try other classes in order of probability
            sorted_classes = np.argsort(probs)[::-1]
            assigned = False
            for cls in sorted_classes:
                if probs[cls] >= thresholds[cls]:
                    predictions[i] = cls
                    assigned = True
                    break
            if not assigned:
                predictions[i] = max_class

    return predictions

y_pred_optimized = predict_with_thresholds(y_probs, optimal_thresholds)

# ============================================================
# 8. EVALUATE OPTIMIZED
# ============================================================
acc_optimized = (y_true == y_pred_optimized).mean() * 100
f1_macro_optimized = f1_score(y_true, y_pred_optimized, average='macro', zero_division=0)
f1_weighted_optimized = f1_score(y_true, y_pred_optimized, average='weighted', zero_division=0)

print("\n" + "="*50)
print("OPTIMIZED")
print("="*50)
print(f"Accuracy:    {acc_optimized:.2f}%")
print(f"Macro F1:    {f1_macro_optimized:.3f}")
print(f"Weighted F1: {f1_weighted_optimized:.3f}")

f1_per_class_optimized = f1_score(y_true, y_pred_optimized, average=None, zero_division=0)
print("\nPer-class F1:")
for i, (name, f1) in enumerate(zip(CLASS_NAMES, f1_per_class_optimized)):
    support = (y_true == i).sum()
    delta = f1 - f1_per_class_baseline[i]
    print(f"  {name:15s}: {f1:.3f}  ({delta:+.3f})  n={support}")

# ============================================================
# 9. SUMMARY
# ============================================================
print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print(f"Macro F1:   {f1_macro_baseline:.3f} → {f1_macro_optimized:.3f}  ({f1_macro_optimized - f1_macro_baseline:+.3f})")
print(f"Accuracy:   {acc_baseline:.2f}% → {acc_optimized:.2f}%  ({acc_optimized - acc_baseline:+.2f}%)")

# Save results
results = {
    'optimal_thresholds': {str(k): float(v) for k, v in optimal_thresholds.items()},
    'baseline': {
        'accuracy': float(acc_baseline),
        'macro_f1': float(f1_macro_baseline),
        'weighted_f1': float(f1_weighted_baseline),
        'per_class_f1': {CLASS_NAMES[i]: float(f1) for i, f1 in enumerate(f1_per_class_baseline)}
    },
    'optimized': {
        'accuracy': float(acc_optimized),
        'macro_f1': float(f1_macro_optimized),
        'weighted_f1': float(f1_weighted_optimized),
        'per_class_f1': {CLASS_NAMES[i]: float(f1) for i, f1 in enumerate(f1_per_class_optimized)}
    }
}

output_json = OUTPUT_DIR / 'threshold_optimization_results.json'
with open(output_json, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n✅ Results saved to {output_json}")

# ============================================================
# 10. PLOT
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Per-class F1 comparison
ax = axes[0]
x = np.arange(len(CLASS_NAMES))
width = 0.35
ax.bar(x - width/2, f1_per_class_baseline, width, label='Baseline', alpha=0.8)
ax.bar(x + width/2, f1_per_class_optimized, width, label='Optimized', alpha=0.8)
ax.set_ylabel('F1 Score')
ax.set_title('Per-Class F1 Comparison')
ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Optimal thresholds
ax = axes[1]
thresholds_list = [optimal_thresholds[i] for i in range(5)]
bars = ax.bar(CLASS_NAMES, thresholds_list, alpha=0.8)
ax.axhline(y=0.5, color='red', linestyle='--', label='Default', alpha=0.5)
ax.set_ylabel('Optimal Threshold')
ax.set_title('Optimized Thresholds')
ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
ax.legend()
ax.set_ylim([0, 1])
ax.grid(axis='y', alpha=0.3)
for bar, thresh in zip(bars, thresholds_list):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{thresh:.2f}', ha='center', va='bottom', fontsize=9)

# Improvement
ax = axes[2]
improvements = [f1_per_class_optimized[i] - f1_per_class_baseline[i] for i in range(5)]
colors = ['green' if x >= 0 else 'red' for x in improvements]
bars = ax.barh(CLASS_NAMES, improvements, color=colors, alpha=0.7)
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax.set_xlabel('F1 Change')
ax.set_title('Per-Class F1 Improvement')
ax.grid(axis='x', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, improvements)):
    x_pos = val + (0.005 if val > 0 else -0.005)
    ha = 'left' if val > 0 else 'right'
    ax.text(x_pos, i, f'{val:+.3f}', va='center', ha=ha, fontsize=9)

plt.tight_layout()
plot_path = OUTPUT_DIR / 'threshold_comparison.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"📊 Comparison plot saved to {plot_path}")

print("\n✅ Threshold optimization complete!")
