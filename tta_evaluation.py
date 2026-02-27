#!/usr/bin/env python3
"""
Test-Time Augmentation (TTA) for RetinaSense v2
================================================
Averages predictions over multiple augmented versions of each image
"""

import os, warnings
import numpy as np
import pandas as pd
from pathlib import Path
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
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
THRESHOLDS_PATH = './outputs_v2/threshold_optimization_results.json'
OUTPUT_DIR = Path('./outputs_v2')
CACHE_DIR = './preprocessed_cache'
IMG_SIZE = 300
BATCH_SIZE = 32  # Smaller for TTA (more memory needed)
NUM_WORKERS = 8
CLASS_NAMES = ['Normal', 'Diabetes/DR', 'Glaucoma', 'Cataract', 'AMD']
NUM_TTA = 8  # Number of augmentations per image

print("🔮 Test-Time Augmentation for RetinaSense v2")
print("="*50)
print(f"Device: {device}")
print(f"TTA augmentations: {NUM_TTA}")

# ============================================================
# 1. DATA LOADING
# ============================================================
print("\n[1/4] Loading data...")
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

# Split
train_df, val_df = train_test_split(
    meta, test_size=0.2, stratify=meta['disease_label'], random_state=42)

print(f"Val samples: {len(val_df)}")

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

print("\n[2/4] Loading model...")
model = MultiTaskModel(n_disease=5, n_severity=5, drop=0.4)
ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model = model.to(device)
model.eval()
print(f"✅ Loaded checkpoint from epoch {ckpt['epoch']}")

# Load optimal thresholds if available
optimal_thresholds = None
if Path(THRESHOLDS_PATH).exists():
    with open(THRESHOLDS_PATH, 'r') as f:
        results = json.load(f)
        optimal_thresholds = {int(k): float(v) for k, v in results['optimal_thresholds'].items()}
    print(f"✅ Loaded optimal thresholds: {optimal_thresholds}")

# ============================================================
# 3. TTA TRANSFORMS
# ============================================================
print("\n[3/4] Setting up TTA transforms...")

# Base normalization
normalize = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])

# TTA augmentations
tta_transforms = [
    # 1. Original
    transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        normalize,
    ]),
    # 2. Horizontal flip
    transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        normalize,
    ]),
    # 3. Vertical flip
    transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(),
        normalize,
    ]),
    # 4. Both flips
    transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(),
        normalize,
    ]),
    # 5. Rotate 90
    transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation((90, 90)),
        transforms.ToTensor(),
        normalize,
    ]),
    # 6. Rotate 180
    transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation((180, 180)),
        transforms.ToTensor(),
        normalize,
    ]),
    # 7. Rotate 270
    transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation((270, 270)),
        transforms.ToTensor(),
        normalize,
    ]),
    # 8. Small brightness adjustment
    transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(brightness=0.1),
        transforms.ToTensor(),
        normalize,
    ]),
]

print(f"TTA augmentations: {len(tta_transforms)}")

# ============================================================
# 4. EVALUATION WITHOUT TTA (baseline)
# ============================================================
print("\n[4/4] Running evaluation...")

val_tfm_base = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    normalize,
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

# Baseline (no TTA)
print("\n→ Baseline (no TTA):")
val_ds_base = RetDS(val_df, val_tfm_base)
val_loader_base = DataLoader(val_ds_base, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

all_probs_base = []
all_labels = []

with torch.no_grad():
    for imgs, diseases, _ in tqdm(val_loader_base, desc="Baseline"):
        imgs = imgs.to(device, non_blocking=True)
        disease_logits, _ = model(imgs)
        probs = torch.softmax(disease_logits, dim=1)
        all_probs_base.append(probs.cpu().numpy())
        all_labels.append(diseases.numpy())

y_probs_base = np.vstack(all_probs_base)
y_true = np.concatenate(all_labels)

# Baseline with argmax
y_pred_base_argmax = np.argmax(y_probs_base, axis=1)
acc_base_argmax = (y_true == y_pred_base_argmax).mean() * 100
f1_base_argmax = f1_score(y_true, y_pred_base_argmax, average='macro', zero_division=0)

print(f"  Argmax: Acc={acc_base_argmax:.2f}%, F1={f1_base_argmax:.3f}")

# Baseline with optimal thresholds
if optimal_thresholds:
    def predict_with_thresholds(y_probs, thresholds):
        n_samples = y_probs.shape[0]
        predictions = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            probs = y_probs[i]
            max_class = np.argmax(probs)
            if probs[max_class] >= thresholds[max_class]:
                predictions[i] = max_class
            else:
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

    y_pred_base_thresh = predict_with_thresholds(y_probs_base, optimal_thresholds)
    acc_base_thresh = (y_true == y_pred_base_thresh).mean() * 100
    f1_base_thresh = f1_score(y_true, y_pred_base_thresh, average='macro', zero_division=0)
    print(f"  Optimal thresholds: Acc={acc_base_thresh:.2f}%, F1={f1_base_thresh:.3f}")

# ============================================================
# 5. EVALUATION WITH TTA
# ============================================================
print("\n→ With TTA:")

# For each sample, we'll average predictions across all augmentations
all_probs_tta = np.zeros_like(y_probs_base)

for tta_idx, tfm in enumerate(tta_transforms):
    print(f"  Aug {tta_idx+1}/{len(tta_transforms)}...", end=" ")
    val_ds_tta = RetDS(val_df, tfm)
    val_loader_tta = DataLoader(val_ds_tta, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=NUM_WORKERS, pin_memory=True)

    batch_probs = []
    with torch.no_grad():
        for imgs, _, _ in val_loader_tta:
            imgs = imgs.to(device, non_blocking=True)
            disease_logits, _ = model(imgs)
            probs = torch.softmax(disease_logits, dim=1)
            batch_probs.append(probs.cpu().numpy())

    aug_probs = np.vstack(batch_probs)
    all_probs_tta += aug_probs
    print(f"✓")

# Average across augmentations
y_probs_tta = all_probs_tta / len(tta_transforms)

# TTA with argmax
y_pred_tta_argmax = np.argmax(y_probs_tta, axis=1)
acc_tta_argmax = (y_true == y_pred_tta_argmax).mean() * 100
f1_tta_argmax = f1_score(y_true, y_pred_tta_argmax, average='macro', zero_division=0)

print(f"\n  Argmax: Acc={acc_tta_argmax:.2f}%, F1={f1_tta_argmax:.3f}")

# TTA with optimal thresholds
if optimal_thresholds:
    y_pred_tta_thresh = predict_with_thresholds(y_probs_tta, optimal_thresholds)
    acc_tta_thresh = (y_true == y_pred_tta_thresh).mean() * 100
    f1_tta_thresh = f1_score(y_true, y_pred_tta_thresh, average='macro', zero_division=0)
    print(f"  Optimal thresholds: Acc={acc_tta_thresh:.2f}%, F1={f1_tta_thresh:.3f}")

# ============================================================
# 6. PER-CLASS ANALYSIS
# ============================================================
print("\n" + "="*50)
print("PER-CLASS F1 SCORES")
print("="*50)

f1_base_argmax_per_class = f1_score(y_true, y_pred_base_argmax, average=None, zero_division=0)
f1_tta_argmax_per_class = f1_score(y_true, y_pred_tta_argmax, average=None, zero_division=0)

if optimal_thresholds:
    f1_base_thresh_per_class = f1_score(y_true, y_pred_base_thresh, average=None, zero_division=0)
    f1_tta_thresh_per_class = f1_score(y_true, y_pred_tta_thresh, average=None, zero_division=0)

    print(f"{'Class':<15} {'Base':<8} {'Base+Thr':<10} {'TTA':<8} {'TTA+Thr':<10} {'Best Δ'}")
    print("-"*70)

    for i, name in enumerate(CLASS_NAMES):
        support = (y_true == i).sum()
        improvement = f1_tta_thresh_per_class[i] - f1_base_argmax_per_class[i]
        print(f"{name:<15} {f1_base_argmax_per_class[i]:.3f}    "
              f"{f1_base_thresh_per_class[i]:.3f}      "
              f"{f1_tta_argmax_per_class[i]:.3f}    "
              f"{f1_tta_thresh_per_class[i]:.3f}      "
              f"{improvement:+.3f}  (n={support})")
else:
    print(f"{'Class':<15} {'Baseline':<10} {'TTA':<10} {'Δ'}")
    print("-"*50)

    for i, name in enumerate(CLASS_NAMES):
        support = (y_true == i).sum()
        improvement = f1_tta_argmax_per_class[i] - f1_base_argmax_per_class[i]
        print(f"{name:<15} {f1_base_argmax_per_class[i]:.3f}      "
              f"{f1_tta_argmax_per_class[i]:.3f}      "
              f"{improvement:+.3f}  (n={support})")

# ============================================================
# 7. SUMMARY
# ============================================================
print("\n" + "="*50)
print("SUMMARY")
print("="*50)

if optimal_thresholds:
    print(f"Baseline (argmax):           Acc={acc_base_argmax:.2f}%, F1={f1_base_argmax:.3f}")
    print(f"Baseline (opt thresholds):   Acc={acc_base_thresh:.2f}%, F1={f1_base_thresh:.3f}")
    print(f"TTA (argmax):                Acc={acc_tta_argmax:.2f}%, F1={f1_tta_argmax:.3f}")
    print(f"TTA (opt thresholds):        Acc={acc_tta_thresh:.2f}%, F1={f1_tta_thresh:.3f} ⭐")
    print(f"\nBest improvement: +{f1_tta_thresh - f1_base_argmax:.3f} F1")
    print(f"                  +{acc_tta_thresh - acc_base_argmax:.2f}% accuracy")
else:
    print(f"Baseline: Acc={acc_base_argmax:.2f}%, F1={f1_base_argmax:.3f}")
    print(f"TTA:      Acc={acc_tta_argmax:.2f}%, F1={f1_tta_argmax:.3f}")
    print(f"\nImprovement: +{f1_tta_argmax - f1_base_argmax:.3f} F1")
    print(f"             +{acc_tta_argmax - acc_base_argmax:.2f}% accuracy")

# Save results
results = {
    'baseline_argmax': {
        'accuracy': float(acc_base_argmax),
        'macro_f1': float(f1_base_argmax),
        'per_class_f1': {CLASS_NAMES[i]: float(f1) for i, f1 in enumerate(f1_base_argmax_per_class)}
    },
    'tta_argmax': {
        'accuracy': float(acc_tta_argmax),
        'macro_f1': float(f1_tta_argmax),
        'per_class_f1': {CLASS_NAMES[i]: float(f1) for i, f1 in enumerate(f1_tta_argmax_per_class)}
    }
}

if optimal_thresholds:
    results['baseline_thresh'] = {
        'accuracy': float(acc_base_thresh),
        'macro_f1': float(f1_base_thresh),
        'per_class_f1': {CLASS_NAMES[i]: float(f1) for i, f1 in enumerate(f1_base_thresh_per_class)}
    }
    results['tta_thresh'] = {
        'accuracy': float(acc_tta_thresh),
        'macro_f1': float(f1_tta_thresh),
        'per_class_f1': {CLASS_NAMES[i]: float(f1) for i, f1 in enumerate(f1_tta_thresh_per_class)}
    }

output_json = OUTPUT_DIR / 'tta_results.json'
with open(output_json, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n✅ Results saved to {output_json}")

# ============================================================
# 8. PLOT
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Per-class F1 comparison
ax = axes[0]
x = np.arange(len(CLASS_NAMES))
width = 0.2

if optimal_thresholds:
    ax.bar(x - 1.5*width, f1_base_argmax_per_class, width, label='Base (argmax)', alpha=0.8)
    ax.bar(x - 0.5*width, f1_base_thresh_per_class, width, label='Base (thresh)', alpha=0.8)
    ax.bar(x + 0.5*width, f1_tta_argmax_per_class, width, label='TTA (argmax)', alpha=0.8)
    ax.bar(x + 1.5*width, f1_tta_thresh_per_class, width, label='TTA (thresh)', alpha=0.8)
else:
    width = 0.35
    ax.bar(x - width/2, f1_base_argmax_per_class, width, label='Baseline', alpha=0.8)
    ax.bar(x + width/2, f1_tta_argmax_per_class, width, label='TTA', alpha=0.8)

ax.set_ylabel('F1 Score')
ax.set_title('Per-Class F1: Baseline vs TTA')
ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Overall metrics
ax = axes[1]

if optimal_thresholds:
    methods = ['Base\n(argmax)', 'Base\n(thresh)', 'TTA\n(argmax)', 'TTA\n(thresh)']
    f1_scores = [f1_base_argmax, f1_base_thresh, f1_tta_argmax, f1_tta_thresh]
    accuracies = [acc_base_argmax/100, acc_base_thresh/100, acc_tta_argmax/100, acc_tta_thresh/100]
else:
    methods = ['Baseline', 'TTA']
    f1_scores = [f1_base_argmax, f1_tta_argmax]
    accuracies = [acc_base_argmax/100, acc_tta_argmax/100]

x = np.arange(len(methods))
width = 0.35

ax.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
ax.bar(x + width/2, f1_scores, width, label='Macro F1', alpha=0.8)

ax.set_ylabel('Score')
ax.set_title('Overall Metrics Comparison')
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend()
ax.set_ylim([0, 1])
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plot_path = OUTPUT_DIR / 'tta_comparison.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"📊 Comparison plot saved to {plot_path}")

print("\n✅ TTA evaluation complete!")
