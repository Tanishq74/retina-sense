#!/usr/bin/env python3
"""
RetinaSense — Comprehensive Data Analysis
==========================================
Analyzes dataset characteristics, model errors, preprocessing,
and augmentation effectiveness. Generates a full report with
visualizations and recommendations.

Outputs saved to outputs_analysis/
"""

import os, sys, warnings, json, time
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageStat, ImageFilter
from tqdm import tqdm
from collections import Counter, defaultdict
from io import StringIO
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, roc_auc_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

# ================================================================
# CONFIG
# ================================================================
OUT_DIR   = './outputs_analysis'
SAVE_DIR  = './outputs_v2'
CACHE_DIR = './preprocessed_cache'
os.makedirs(OUT_DIR, exist_ok=True)

IMG_SIZE    = 300
BATCH_SIZE  = 32
NUM_WORKERS = 4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = ['Normal', 'Diabetes/DR', 'Glaucoma', 'Cataract', 'AMD']
NUM_CLASSES = len(CLASS_NAMES)
COLORS = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']

print('=' * 65)
print('       RetinaSense — Comprehensive Data Analysis')
print('=' * 65)
print(f'  Device: {device}')

report_lines = []
def log(msg=''):
    print(msg)
    report_lines.append(msg)

# ================================================================
# 1. METADATA (same approach as retinasense_v2.py)
# ================================================================
log('\n[1/6] Building metadata...')

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

# Build cache paths (same logic as v2)
cache_paths = []
for _, row in meta.iterrows():
    stem = os.path.splitext(os.path.basename(row['image_path']))[0]
    fp   = f'{CACHE_DIR}/{stem}_{IMG_SIZE}.npy'
    cache_paths.append(fp)
meta['cache_path'] = cache_paths

# Train/val split (same seed as v2)
train_df, val_df = train_test_split(
    meta, test_size=0.2, stratify=meta['disease_label'], random_state=42)
train_df = train_df.reset_index(drop=True)
val_df   = val_df.reset_index(drop=True)

log(f'  Total samples : {len(meta)}')
log(f'  Train samples : {len(train_df)}')
log(f'  Val samples   : {len(val_df)}')

# ================================================================
# 2. CLASS DISTRIBUTION ANALYSIS
# ================================================================
log('\n' + '=' * 65)
log('  SECTION 1: CLASS DISTRIBUTION ANALYSIS')
log('=' * 65)

# Overall distribution
dist_all = meta['disease_label'].value_counts().sort_index()
log('\n--- Overall Distribution ---')
for i, cnt in dist_all.items():
    log(f'  {CLASS_NAMES[i]:15s}: {cnt:5d}  ({100*cnt/len(meta):.1f}%)')

# Train/Val split distribution
log('\n--- Train/Val Split ---')
dist_train = train_df['disease_label'].value_counts().sort_index()
dist_val   = val_df['disease_label'].value_counts().sort_index()
log(f'  {"Class":15s} {"Train":>6s} {"Val":>6s} {"Train%":>7s} {"Val%":>7s}')
log(f'  {"-"*43}')
for i in range(NUM_CLASSES):
    tr = dist_train.get(i, 0)
    vl = dist_val.get(i, 0)
    log(f'  {CLASS_NAMES[i]:15s} {tr:6d} {vl:6d} {100*tr/len(train_df):6.1f}% {100*vl/len(val_df):6.1f}%')

# Dataset source distribution
log('\n--- Dataset Source per Class ---')
log(f'  {"Class":15s} {"ODIR":>6s} {"APTOS":>6s}')
log(f'  {"-"*30}')
for i in range(NUM_CLASSES):
    cls_meta = meta[meta['disease_label'] == i]
    odir_cnt  = len(cls_meta[cls_meta['dataset'] == 'ODIR'])
    aptos_cnt = len(cls_meta[cls_meta['dataset'] == 'APTOS'])
    log(f'  {CLASS_NAMES[i]:15s} {odir_cnt:6d} {aptos_cnt:6d}')

# Imbalance ratio
max_cls = dist_all.max()
min_cls = dist_all.min()
log(f'\n  Imbalance ratio (max/min): {max_cls/min_cls:.1f}x')
log(f'  Majority class: {CLASS_NAMES[dist_all.idxmax()]} ({max_cls})')
log(f'  Minority class: {CLASS_NAMES[dist_all.idxmin()]} ({min_cls})')

# APTOS severity distribution
log('\n--- APTOS Severity Distribution (within Diabetes/DR class) ---')
aptos_sev = meta[meta['dataset'] == 'APTOS']['severity_label'].value_counts().sort_index()
sev_names = {0: 'No DR', 1: 'Mild', 2: 'Moderate', 3: 'Severe', 4: 'Proliferative'}
for s, cnt in aptos_sev.items():
    log(f'  {sev_names.get(s, f"Sev {s}"):15s}: {cnt:5d}  ({100*cnt/len(aptos_sev)*aptos_sev.count()/cnt:.1f}%)')

# --- Plot: Class distribution ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Bar chart
bars = axes[0].bar(CLASS_NAMES, [dist_all.get(i, 0) for i in range(NUM_CLASSES)], color=COLORS)
axes[0].set_title('Class Distribution (Overall)', fontweight='bold')
axes[0].set_ylabel('Count')
for bar, cnt in zip(bars, [dist_all.get(i, 0) for i in range(NUM_CLASSES)]):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                 str(cnt), ha='center', va='bottom', fontsize=9)
axes[0].tick_params(axis='x', rotation=30)

# Train/val side-by-side
x = np.arange(NUM_CLASSES)
w = 0.35
axes[1].bar(x - w/2, [dist_train.get(i, 0) for i in range(NUM_CLASSES)], w, label='Train', color='#3498db')
axes[1].bar(x + w/2, [dist_val.get(i, 0) for i in range(NUM_CLASSES)], w, label='Val', color='#e74c3c')
axes[1].set_xticks(x)
axes[1].set_xticklabels(CLASS_NAMES, rotation=30)
axes[1].set_title('Train vs Val Distribution', fontweight='bold')
axes[1].set_ylabel('Count')
axes[1].legend()

# Dataset source stacked
odir_counts  = [len(meta[(meta['disease_label']==i) & (meta['dataset']=='ODIR')]) for i in range(NUM_CLASSES)]
aptos_counts = [len(meta[(meta['disease_label']==i) & (meta['dataset']=='APTOS')]) for i in range(NUM_CLASSES)]
axes[2].bar(CLASS_NAMES, odir_counts, label='ODIR', color='#2ecc71')
axes[2].bar(CLASS_NAMES, aptos_counts, bottom=odir_counts, label='APTOS', color='#f39c12')
axes[2].set_title('Dataset Source per Class', fontweight='bold')
axes[2].set_ylabel('Count')
axes[2].legend()
axes[2].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/01_class_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
log(f'\n  Saved: {OUT_DIR}/01_class_distribution.png')


# ================================================================
# 3. IMAGE QUALITY METRICS PER CLASS
# ================================================================
log('\n' + '=' * 65)
log('  SECTION 2: IMAGE QUALITY METRICS PER CLASS')
log('=' * 65)

# Sample images per class for quality analysis (use all for small dataset)
MAX_SAMPLES_PER_CLASS = 500

quality_data = []
log('\n  Computing image quality metrics...')

for cls_idx in range(NUM_CLASSES):
    cls_df = meta[meta['disease_label'] == cls_idx]
    sample_df = cls_df.sample(n=min(MAX_SAMPLES_PER_CLASS, len(cls_df)), random_state=42)

    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df),
                       desc=f'  {CLASS_NAMES[cls_idx]}'):
        try:
            img = cv2.imread(row['image_path'])
            if img is None:
                continue
            h, w = img.shape[:2]
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Brightness (mean of grayscale)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            brightness = float(np.mean(gray))

            # Contrast (std of grayscale)
            contrast = float(np.std(gray))

            # Sharpness (Laplacian variance)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

            # Color channel means
            b_mean = float(np.mean(img[:, :, 0]))
            g_mean = float(np.mean(img[:, :, 1]))
            r_mean = float(np.mean(img[:, :, 2]))

            # Color channel stds
            b_std = float(np.std(img[:, :, 0]))
            g_std = float(np.std(img[:, :, 1]))
            r_std = float(np.std(img[:, :, 2]))

            quality_data.append({
                'class': cls_idx,
                'class_name': CLASS_NAMES[cls_idx],
                'dataset': row['dataset'],
                'width': w,
                'height': h,
                'brightness': brightness,
                'contrast': contrast,
                'sharpness': sharpness,
                'r_mean': r_mean, 'g_mean': g_mean, 'b_mean': b_mean,
                'r_std': r_std, 'g_std': g_std, 'b_std': b_std,
            })
        except Exception as e:
            continue

qdf = pd.DataFrame(quality_data)
log(f'  Analyzed {len(qdf)} images')

# Summary table
log('\n--- Image Quality Summary per Class ---')
log(f'  {"Class":15s} {"Brightness":>10s} {"Contrast":>10s} {"Sharpness":>10s} '
    f'{"W(mean)":>8s} {"H(mean)":>8s}')
log(f'  {"-"*62}')

for cls_idx in range(NUM_CLASSES):
    cq = qdf[qdf['class'] == cls_idx]
    log(f'  {CLASS_NAMES[cls_idx]:15s} '
        f'{cq["brightness"].mean():10.1f} '
        f'{cq["contrast"].mean():10.1f} '
        f'{cq["sharpness"].mean():10.1f} '
        f'{cq["width"].mean():8.0f} '
        f'{cq["height"].mean():8.0f}')

# Color channel stats
log('\n--- Color Channel Means per Class ---')
log(f'  {"Class":15s} {"R mean":>8s} {"G mean":>8s} {"B mean":>8s} '
    f'{"R std":>8s} {"G std":>8s} {"B std":>8s}')
log(f'  {"-"*62}')
for cls_idx in range(NUM_CLASSES):
    cq = qdf[qdf['class'] == cls_idx]
    log(f'  {CLASS_NAMES[cls_idx]:15s} '
        f'{cq["r_mean"].mean():8.1f} {cq["g_mean"].mean():8.1f} {cq["b_mean"].mean():8.1f} '
        f'{cq["r_std"].mean():8.1f} {cq["g_std"].mean():8.1f} {cq["b_std"].mean():8.1f}')

# ODIR vs APTOS quality comparison
log('\n--- ODIR vs APTOS Quality Comparison ---')
for ds in ['ODIR', 'APTOS']:
    dq = qdf[qdf['dataset'] == ds]
    if len(dq) > 0:
        log(f'  {ds:6s}: brightness={dq["brightness"].mean():.1f}, '
            f'contrast={dq["contrast"].mean():.1f}, '
            f'sharpness={dq["sharpness"].mean():.1f}, '
            f'mean_size={dq["width"].mean():.0f}x{dq["height"].mean():.0f}')

# Identify outliers (images with extreme values)
log('\n--- Potential Outliers ---')
for metric in ['brightness', 'contrast', 'sharpness']:
    q1 = qdf[metric].quantile(0.01)
    q99 = qdf[metric].quantile(0.99)
    outliers = qdf[(qdf[metric] < q1) | (qdf[metric] > q99)]
    log(f'  {metric:12s}: {len(outliers)} outliers (outside 1st-99th percentile)')
    for cls_idx in range(NUM_CLASSES):
        cnt = len(outliers[outliers['class'] == cls_idx])
        if cnt > 0:
            log(f'    {CLASS_NAMES[cls_idx]:15s}: {cnt}')

# --- Plot: Quality metrics ---
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Brightness distribution per class
for cls_idx in range(NUM_CLASSES):
    cq = qdf[qdf['class'] == cls_idx]
    axes[0, 0].hist(cq['brightness'], bins=30, alpha=0.5, label=CLASS_NAMES[cls_idx],
                     color=COLORS[cls_idx])
axes[0, 0].set_title('Brightness Distribution', fontweight='bold')
axes[0, 0].legend(fontsize=8)
axes[0, 0].set_xlabel('Mean Brightness')

# Contrast distribution per class
for cls_idx in range(NUM_CLASSES):
    cq = qdf[qdf['class'] == cls_idx]
    axes[0, 1].hist(cq['contrast'], bins=30, alpha=0.5, label=CLASS_NAMES[cls_idx],
                     color=COLORS[cls_idx])
axes[0, 1].set_title('Contrast Distribution', fontweight='bold')
axes[0, 1].legend(fontsize=8)
axes[0, 1].set_xlabel('Contrast (Std Dev)')

# Sharpness distribution per class (log scale)
for cls_idx in range(NUM_CLASSES):
    cq = qdf[qdf['class'] == cls_idx]
    sharpness_log = np.log1p(cq['sharpness'])
    axes[0, 2].hist(sharpness_log, bins=30, alpha=0.5, label=CLASS_NAMES[cls_idx],
                     color=COLORS[cls_idx])
axes[0, 2].set_title('Sharpness Distribution (log)', fontweight='bold')
axes[0, 2].legend(fontsize=8)
axes[0, 2].set_xlabel('Log(1 + Sharpness)')

# Image size scatter
for cls_idx in range(NUM_CLASSES):
    cq = qdf[qdf['class'] == cls_idx]
    axes[1, 0].scatter(cq['width'], cq['height'], alpha=0.3, s=10,
                        label=CLASS_NAMES[cls_idx], color=COLORS[cls_idx])
axes[1, 0].set_title('Image Size Distribution', fontweight='bold')
axes[1, 0].set_xlabel('Width')
axes[1, 0].set_ylabel('Height')
axes[1, 0].legend(fontsize=8)

# Boxplot: brightness per class
bp_data = [qdf[qdf['class'] == i]['brightness'].values for i in range(NUM_CLASSES)]
bplot = axes[1, 1].boxplot(bp_data, labels=CLASS_NAMES, patch_artist=True)
for patch, color in zip(bplot['boxes'], COLORS):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
axes[1, 1].set_title('Brightness by Class', fontweight='bold')
axes[1, 1].tick_params(axis='x', rotation=30)

# Boxplot: contrast per class
bp_data2 = [qdf[qdf['class'] == i]['contrast'].values for i in range(NUM_CLASSES)]
bplot2 = axes[1, 2].boxplot(bp_data2, labels=CLASS_NAMES, patch_artist=True)
for patch, color in zip(bplot2['boxes'], COLORS):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
axes[1, 2].set_title('Contrast by Class', fontweight='bold')
axes[1, 2].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/02_image_quality.png', dpi=150, bbox_inches='tight')
plt.close()
log(f'  Saved: {OUT_DIR}/02_image_quality.png')


# ================================================================
# 4. AUGMENTATION ANALYSIS
# ================================================================
log('\n' + '=' * 65)
log('  SECTION 3: AUGMENTATION ANALYSIS')
log('=' * 65)

# Current augmentations from v2
log('\n--- Current Augmentation Pipeline (v2) ---')
log('  1. RandomHorizontalFlip (p=0.5)')
log('  2. RandomVerticalFlip (p=0.3)')
log('  3. RandomRotation (20 degrees)')
log('  4. RandomAffine (translate=0.05, scale=0.95-1.05)')
log('  5. ColorJitter (brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02)')
log('  6. RandomErasing (p=0.2)')

# Visualize augmentations on sample images
train_tfm = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.2),
])

val_tfm = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

# Show augmented versions of one sample per class
fig, axes = plt.subplots(NUM_CLASSES, 6, figsize=(18, 3 * NUM_CLASSES))
fig.suptitle('Augmentation Visualization (Original + 5 Augmented)', fontsize=14, fontweight='bold')

for cls_idx in range(NUM_CLASSES):
    cls_df = meta[meta['disease_label'] == cls_idx]
    sample_row = cls_df.iloc[0]
    cache_path = sample_row['cache_path']

    try:
        img = np.load(cache_path)
    except:
        img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

    # Original
    orig_tensor = val_tfm(img)
    axes[cls_idx, 0].imshow(orig_tensor.permute(1, 2, 0).clamp(0, 1))
    axes[cls_idx, 0].set_title(f'{CLASS_NAMES[cls_idx]}\n(Original)', fontsize=8)
    axes[cls_idx, 0].axis('off')

    # 5 augmented versions
    for aug_i in range(1, 6):
        aug_tensor = train_tfm(img)
        axes[cls_idx, aug_i].imshow(aug_tensor.permute(1, 2, 0).clamp(0, 1))
        axes[cls_idx, aug_i].set_title(f'Aug {aug_i}', fontsize=8)
        axes[cls_idx, aug_i].axis('off')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/03_augmentation_samples.png', dpi=150, bbox_inches='tight')
plt.close()
log(f'  Saved: {OUT_DIR}/03_augmentation_samples.png')

# Augmentation recommendations
log('\n--- Augmentation Recommendations ---')
for cls_idx in range(NUM_CLASSES):
    cnt = dist_all.get(cls_idx, 0)
    ratio = cnt / max_cls
    log(f'\n  {CLASS_NAMES[cls_idx]} ({cnt} samples, {ratio:.2f}x of majority):')
    if ratio < 0.15:
        log(f'    -> CRITICAL minority class. Recommendations:')
        log(f'       - Use stronger augmentation (higher rotation, more color jitter)')
        log(f'       - Apply Mixup/CutMix with same-class samples')
        log(f'       - Consider synthetic generation or aggressive oversampling')
        log(f'       - Use class-specific augmentation policy')
    elif ratio < 0.3:
        log(f'    -> Minority class. Recommendations:')
        log(f'       - Moderate augmentation boost (increase rotation to 30deg)')
        log(f'       - Use Mixup with alpha=0.4 for this class')
        log(f'       - Weighted sampling to increase exposure')
    elif ratio < 0.6:
        log(f'    -> Moderate class. Recommendations:')
        log(f'       - Standard augmentation is sufficient')
        log(f'       - Light Mixup can help (alpha=0.2)')
    else:
        log(f'    -> Majority or near-majority class. Recommendations:')
        log(f'       - Standard augmentation')
        log(f'       - Consider undersampling or reducing weight if dominating')


# ================================================================
# 5. ERROR ANALYSIS (load best v2 model)
# ================================================================
log('\n' + '=' * 65)
log('  SECTION 4: ERROR ANALYSIS')
log('=' * 65)

MODEL_PATH = f'{SAVE_DIR}/best_model.pth'
if not os.path.exists(MODEL_PATH):
    log(f'  WARNING: Model not found at {MODEL_PATH}. Skipping error analysis.')
else:
    log(f'\n  Loading model from {MODEL_PATH}...')

    # Recreate model architecture (same as v2)
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
    log(f'  Model loaded (epoch {ckpt.get("epoch", "?")})')
    log(f'  Checkpoint macro-F1: {ckpt.get("macro_f1", 0):.4f}')

    # Dataset for validation
    class SimpleDS(Dataset):
        def __init__(self, df):
            self.df = df.reset_index(drop=True)
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
                img = np.load(r['cache_path'])
            except:
                img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            return self.tfm(img), int(r['disease_label']), i

    val_ds = SimpleDS(val_df)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    # Collect predictions
    all_preds = []
    all_labels = []
    all_probs = []
    all_indices = []

    with torch.no_grad():
        for imgs, labels, indices in tqdm(val_loader, desc='  Evaluating'):
            imgs = imgs.to(device)
            d_out, _ = model(imgs)
            probs = torch.softmax(d_out.float(), dim=1)
            all_preds.extend(d_out.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            all_indices.extend(indices.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    all_indices = np.array(all_indices)

    # Classification report
    log('\n--- Classification Report ---')
    report_str = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4)
    log(report_str)

    mf1 = f1_score(all_labels, all_preds, average='macro')
    wf1 = f1_score(all_labels, all_preds, average='weighted')
    try:
        mauc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    except:
        mauc = 0.0
    log(f'  Macro F1    : {mf1:.4f}')
    log(f'  Weighted F1 : {wf1:.4f}')
    log(f'  Macro AUC   : {mauc:.4f}')

    # Confusion matrix analysis
    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    log('\n--- Confusion Matrix (Raw) ---')
    header = f'  {"":15s} ' + ' '.join(f'{cn[:6]:>6s}' for cn in CLASS_NAMES)
    log(header)
    for i in range(NUM_CLASSES):
        row_str = f'  {CLASS_NAMES[i]:15s} ' + ' '.join(f'{cm[i,j]:6d}' for j in range(NUM_CLASSES))
        log(row_str)

    log('\n--- Confusion Matrix (Normalized) ---')
    log(header)
    for i in range(NUM_CLASSES):
        row_str = f'  {CLASS_NAMES[i]:15s} ' + ' '.join(f'{cm_norm[i,j]:6.2f}' for j in range(NUM_CLASSES))
        log(row_str)

    # Most confused class pairs
    log('\n--- Most Commonly Confused Class Pairs ---')
    confusions = []
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i != j and cm[i, j] > 0:
                confusions.append((CLASS_NAMES[i], CLASS_NAMES[j], cm[i, j], cm_norm[i, j]))
    confusions.sort(key=lambda x: x[2], reverse=True)
    for true_cls, pred_cls, count, rate in confusions[:10]:
        log(f'  {true_cls:15s} -> {pred_cls:15s}: {count:4d} ({rate:.1%})')

    # Hardest samples (lowest confidence correct predictions)
    correct_mask = all_preds == all_labels
    correct_indices = np.where(correct_mask)[0]
    correct_conf = all_probs[correct_indices, all_labels[correct_indices]]
    hardest_order = np.argsort(correct_conf)

    log('\n--- Hardest Correct Predictions (Lowest Confidence) ---')
    log(f'  {"Index":>6s} {"True Class":>15s} {"Confidence":>11s} {"Dataset":>8s}')
    for i in range(min(20, len(hardest_order))):
        idx = correct_indices[hardest_order[i]]
        orig_idx = all_indices[idx]
        conf = correct_conf[hardest_order[i]]
        row = val_df.iloc[orig_idx]
        log(f'  {orig_idx:6d} {CLASS_NAMES[all_labels[idx]]:>15s} {conf:11.4f} {row["dataset"]:>8s}')

    # Failed predictions on minority classes
    log('\n--- Failed Predictions on Minority Classes ---')
    minority_classes = [i for i in range(NUM_CLASSES) if dist_all.get(i, 0) < dist_all.median()]
    for cls_idx in minority_classes:
        cls_mask = all_labels == cls_idx
        cls_correct = (all_preds[cls_mask] == cls_idx).sum()
        cls_total = cls_mask.sum()
        cls_wrong = cls_total - cls_correct
        if cls_total > 0:
            log(f'\n  {CLASS_NAMES[cls_idx]}: {cls_wrong}/{cls_total} failed ({100*cls_wrong/cls_total:.1f}%)')
            # What are they being predicted as?
            wrong_mask = cls_mask & (all_preds != cls_idx)
            wrong_preds = all_preds[wrong_mask]
            pred_counts = Counter(wrong_preds)
            for pred_cls, cnt in pred_counts.most_common():
                log(f'    Misclassified as {CLASS_NAMES[pred_cls]:15s}: {cnt}')

            # Confidence analysis on wrong predictions
            wrong_indices_local = np.where(wrong_mask)[0]
            if len(wrong_indices_local) > 0:
                wrong_conf = all_probs[wrong_indices_local, cls_idx]
                log(f'    True-class confidence on errors: mean={wrong_conf.mean():.3f}, '
                    f'std={wrong_conf.std():.3f}')

    # --- Plot: Error analysis ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Confusion matrix heatmap
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=axes[0, 0],
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    axes[0, 0].set_title('Normalized Confusion Matrix', fontweight='bold')
    axes[0, 0].set_ylabel('True Label')
    axes[0, 0].set_xlabel('Predicted Label')

    # Raw confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=axes[0, 1],
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    axes[0, 1].set_title('Raw Confusion Matrix', fontweight='bold')
    axes[0, 1].set_ylabel('True Label')
    axes[0, 1].set_xlabel('Predicted Label')

    # Per-class accuracy
    per_class_acc = [cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0 for i in range(NUM_CLASSES)]
    bars = axes[1, 0].bar(CLASS_NAMES, per_class_acc, color=COLORS)
    axes[1, 0].set_title('Per-Class Accuracy', fontweight='bold')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_ylim(0, 1)
    for bar, acc in zip(bars, per_class_acc):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                         f'{acc:.2f}', ha='center', fontsize=9)
    axes[1, 0].tick_params(axis='x', rotation=30)

    # Confidence distribution for correct vs wrong
    wrong_mask_all = ~correct_mask
    if correct_mask.sum() > 0 and wrong_mask_all.sum() > 0:
        correct_max_conf = all_probs[correct_mask].max(axis=1)
        wrong_max_conf   = all_probs[wrong_mask_all].max(axis=1)
        axes[1, 1].hist(correct_max_conf, bins=30, alpha=0.6, label='Correct', color='#2ecc71')
        axes[1, 1].hist(wrong_max_conf, bins=30, alpha=0.6, label='Wrong', color='#e74c3c')
        axes[1, 1].set_title('Prediction Confidence Distribution', fontweight='bold')
        axes[1, 1].set_xlabel('Max Probability')
        axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/04_error_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    log(f'  Saved: {OUT_DIR}/04_error_analysis.png')

    # ROC curves
    fig, ax = plt.subplots(figsize=(8, 8))
    y_bin = label_binarize(all_labels, classes=list(range(NUM_CLASSES)))
    for ci, (cn, col) in enumerate(zip(CLASS_NAMES, COLORS)):
        fpr, tpr, _ = roc_curve(y_bin[:, ci], all_probs[:, ci])
        ax.plot(fpr, tpr, color=col, lw=2, label=f'{cn} (AUC={auc(fpr, tpr):.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_title('ROC Curves per Class', fontweight='bold')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/05_roc_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    log(f'  Saved: {OUT_DIR}/05_roc_curves.png')


# ================================================================
# 6. PREPROCESSING ANALYSIS
# ================================================================
log('\n' + '=' * 65)
log('  SECTION 5: PREPROCESSING ANALYSIS')
log('=' * 65)

def ben_graham(path, sz=IMG_SIZE, sigma=10):
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


# Visualize before/after preprocessing for each class
fig, axes = plt.subplots(NUM_CLASSES, 4, figsize=(16, 4 * NUM_CLASSES))
fig.suptitle('Ben Graham Preprocessing: Before vs After per Class', fontsize=14, fontweight='bold')

preprocessing_stats = []

for cls_idx in range(NUM_CLASSES):
    cls_df = meta[meta['disease_label'] == cls_idx]
    # Pick a sample with an existing original image
    sample = cls_df.iloc[0]

    # Original image
    try:
        orig = cv2.imread(sample['image_path'])
        if orig is not None:
            orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
            orig_resized = cv2.resize(orig_rgb, (IMG_SIZE, IMG_SIZE))
        else:
            orig_resized = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    except:
        orig_resized = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

    # Preprocessed image
    try:
        preprocessed = ben_graham(sample['image_path'])
    except:
        preprocessed = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

    # Display
    axes[cls_idx, 0].imshow(orig_resized)
    axes[cls_idx, 0].set_title(f'{CLASS_NAMES[cls_idx]}\nOriginal', fontsize=9)
    axes[cls_idx, 0].axis('off')

    axes[cls_idx, 1].imshow(preprocessed)
    axes[cls_idx, 1].set_title(f'{CLASS_NAMES[cls_idx]}\nBen Graham', fontsize=9)
    axes[cls_idx, 1].axis('off')

    # Difference image
    diff = np.abs(orig_resized.astype(float) - preprocessed.astype(float))
    diff_norm = (diff / diff.max() * 255).astype(np.uint8) if diff.max() > 0 else diff.astype(np.uint8)
    axes[cls_idx, 2].imshow(diff_norm)
    axes[cls_idx, 2].set_title('Difference', fontsize=9)
    axes[cls_idx, 2].axis('off')

    # Histograms comparison
    for ch, color in enumerate(['r', 'g', 'b']):
        axes[cls_idx, 3].hist(orig_resized[:, :, ch].ravel(), bins=50, alpha=0.3,
                               color=color, label=f'Orig {color.upper()}')
        axes[cls_idx, 3].hist(preprocessed[:, :, ch].ravel(), bins=50, alpha=0.3,
                               color=color, linestyle='--', label=f'BG {color.upper()}')
    axes[cls_idx, 3].set_title('Channel Histograms', fontsize=9)
    if cls_idx == 0:
        axes[cls_idx, 3].legend(fontsize=6)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/06_preprocessing_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
log(f'  Saved: {OUT_DIR}/06_preprocessing_comparison.png')

# Compare preprocessing effect across classes
log('\n--- Preprocessing Effect per Class ---')
log(f'  {"Class":15s} {"Orig Bright":>12s} {"BG Bright":>12s} {"Change":>8s} '
    f'{"Orig Contrast":>14s} {"BG Contrast":>14s}')
log(f'  {"-"*76}')

PREPROC_SAMPLES = 100
for cls_idx in range(NUM_CLASSES):
    cls_df = meta[meta['disease_label'] == cls_idx]
    sample_df = cls_df.sample(n=min(PREPROC_SAMPLES, len(cls_df)), random_state=42)

    orig_brightness = []
    bg_brightness = []
    orig_contrast = []
    bg_contrast = []

    for _, row in sample_df.iterrows():
        try:
            orig = cv2.imread(row['image_path'])
            if orig is None:
                continue
            orig_resized = cv2.resize(orig, (IMG_SIZE, IMG_SIZE))
            preprocessed = ben_graham(row['image_path'])

            orig_gray = cv2.cvtColor(orig_resized, cv2.COLOR_BGR2GRAY)
            bg_gray   = cv2.cvtColor(preprocessed, cv2.COLOR_RGB2GRAY)

            orig_brightness.append(np.mean(orig_gray))
            bg_brightness.append(np.mean(bg_gray))
            orig_contrast.append(np.std(orig_gray))
            bg_contrast.append(np.std(bg_gray))
        except:
            continue

    if orig_brightness:
        ob = np.mean(orig_brightness)
        bb = np.mean(bg_brightness)
        oc = np.mean(orig_contrast)
        bc = np.mean(bg_contrast)
        change = bb - ob
        log(f'  {CLASS_NAMES[cls_idx]:15s} {ob:12.1f} {bb:12.1f} {change:+8.1f} '
            f'{oc:14.1f} {bc:14.1f}')

# Check if minority classes are systematically different
log('\n--- Are Minority Classes Systematically Different? ---')
majority_brightness = qdf[qdf['class'] == dist_all.idxmax()]['brightness']
for cls_idx in minority_classes:
    cls_brightness = qdf[qdf['class'] == cls_idx]['brightness']
    diff = cls_brightness.mean() - majority_brightness.mean()
    log(f'  {CLASS_NAMES[cls_idx]:15s} vs {CLASS_NAMES[dist_all.idxmax()]:15s}: '
        f'brightness diff = {diff:+.1f}, '
        f'contrast diff = {qdf[qdf["class"]==cls_idx]["contrast"].mean() - qdf[qdf["class"]==dist_all.idxmax()]["contrast"].mean():+.1f}')


# ================================================================
# 7. AUGMENTATION MINI-EXPERIMENTS (5 epochs each)
# ================================================================
log('\n' + '=' * 65)
log('  SECTION 6: AUGMENTATION MINI-EXPERIMENTS')
log('=' * 65)
log('\n  Training 4 augmentation strategies for 5 epochs each...')
log('  (Heads-only training for fast comparison)\n')

from torch.amp import GradScaler

# Define augmentation strategies to compare
AUG_STRATEGIES = {
    'baseline': transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2),
    ]),
    'strong_aug': transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(45),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
    ]),
    'light_aug': transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'geometric_only': transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}

val_tfm_norm = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

class AugDS(Dataset):
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
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        return self.tfm(img), int(r['disease_label'])

# Focal Loss for experiments
class FocalLossExp(nn.Module):
    def __init__(self, alpha=None, gamma=1.0):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            at = self.alpha.gather(0, targets)
            focal = at * focal
        return focal.mean()

MINI_EPOCHS = 5
experiment_results = {}

for strat_name, strat_tfm in AUG_STRATEGIES.items():
    log(f'  --- Experiment: {strat_name} ---')

    # Create loaders
    exp_train_ds = AugDS(train_df, strat_tfm)
    exp_val_ds   = AugDS(val_df, val_tfm_norm)
    exp_train_loader = DataLoader(exp_train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=True)
    exp_val_loader   = DataLoader(exp_val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=NUM_WORKERS, pin_memory=True)

    # Fresh model (heads only for speed)
    exp_model = MultiTaskModel().to(device)
    for p in exp_model.backbone.parameters():
        p.requires_grad = False

    cw_exp = compute_class_weight('balanced', classes=np.arange(5), y=train_df['disease_label'].values)
    alpha_exp = torch.tensor(cw_exp, dtype=torch.float32).to(device)
    alpha_exp = alpha_exp / alpha_exp.sum() * NUM_CLASSES
    criterion_exp = FocalLossExp(alpha=alpha_exp, gamma=1.0)

    opt_exp = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, exp_model.parameters()),
        lr=3e-4, weight_decay=1e-3)
    scaler_exp = GradScaler()

    epoch_metrics = []
    for ep in range(MINI_EPOCHS):
        # Train
        exp_model.train()
        for imgs, lbls in exp_train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            opt_exp.zero_grad(set_to_none=True)
            with autocast('cuda'):
                d_out, _ = exp_model(imgs)
                loss = criterion_exp(d_out, lbls)
            if not (torch.isnan(loss) or torch.isinf(loss)):
                scaler_exp.scale(loss).backward()
                scaler_exp.step(opt_exp)
                scaler_exp.update()

        # Eval
        exp_model.eval()
        ep_preds, ep_labels = [], []
        with torch.no_grad():
            for imgs, lbls in exp_val_loader:
                imgs = imgs.to(device)
                with autocast('cuda'):
                    d_out, _ = exp_model(imgs)
                ep_preds.extend(d_out.argmax(1).cpu().numpy())
                ep_labels.extend(lbls.numpy())

        ep_preds = np.array(ep_preds)
        ep_labels = np.array(ep_labels)
        ep_mf1 = f1_score(ep_labels, ep_preds, average='macro')
        ep_wf1 = f1_score(ep_labels, ep_preds, average='weighted')
        ep_acc = (ep_preds == ep_labels).mean() * 100
        ep_per_f1 = f1_score(ep_labels, ep_preds, average=None, labels=range(NUM_CLASSES), zero_division=0)
        epoch_metrics.append({
            'epoch': ep + 1, 'macro_f1': ep_mf1, 'weighted_f1': ep_wf1,
            'accuracy': ep_acc, 'per_class_f1': ep_per_f1.tolist()
        })

    # Record final epoch results
    final = epoch_metrics[-1]
    experiment_results[strat_name] = final
    log(f'    Final (E{MINI_EPOCHS}): mF1={final["macro_f1"]:.4f}  wF1={final["weighted_f1"]:.4f}  '
        f'Acc={final["accuracy"]:.1f}%')
    cls_str = ' | '.join(f'{CLASS_NAMES[i][:3]}:{final["per_class_f1"][i]:.2f}'
                         for i in range(NUM_CLASSES))
    log(f'    Per-class: {cls_str}')

    # Cleanup
    del exp_model, opt_exp, scaler_exp, exp_train_ds, exp_val_ds
    torch.cuda.empty_cache()

# Summary comparison table
log('\n--- Augmentation Strategy Comparison (after 5 epochs, heads-only) ---')
log(f'  {"Strategy":20s} {"mF1":>6s} {"wF1":>6s} {"Acc%":>6s}  '
    + '  '.join(f'{cn[:5]:>5s}' for cn in CLASS_NAMES))
log(f'  {"-"*78}')
best_strat = None
best_mf1 = 0
for sn, sr in experiment_results.items():
    cls_str = '  '.join(f'{sr["per_class_f1"][i]:.3f}' for i in range(NUM_CLASSES))
    log(f'  {sn:20s} {sr["macro_f1"]:6.4f} {sr["weighted_f1"]:6.4f} {sr["accuracy"]:5.1f}%  {cls_str}')
    if sr['macro_f1'] > best_mf1:
        best_mf1 = sr['macro_f1']
        best_strat = sn

log(f'\n  Best strategy: {best_strat} (macro F1 = {best_mf1:.4f})')

# Plot experiment results
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

strat_names = list(experiment_results.keys())
mf1_vals  = [experiment_results[s]['macro_f1'] for s in strat_names]
wf1_vals  = [experiment_results[s]['weighted_f1'] for s in strat_names]
acc_vals  = [experiment_results[s]['accuracy'] for s in strat_names]

x = np.arange(len(strat_names))
w = 0.3
axes[0].bar(x - w, mf1_vals, w, label='Macro F1', color='#2ecc71')
axes[0].bar(x, wf1_vals, w, label='Weighted F1', color='#3498db')
axes[0].bar(x + w, [a/100 for a in acc_vals], w, label='Accuracy', color='#e74c3c')
axes[0].set_xticks(x)
axes[0].set_xticklabels(strat_names, rotation=20, ha='right')
axes[0].set_title('Augmentation Strategy Comparison', fontweight='bold')
axes[0].set_ylabel('Score')
axes[0].legend()
axes[0].grid(alpha=0.3, axis='y')

# Per-class F1 grouped
x2 = np.arange(NUM_CLASSES)
bar_w = 0.8 / len(strat_names)
for si, sn in enumerate(strat_names):
    f1s = experiment_results[sn]['per_class_f1']
    axes[1].bar(x2 + si * bar_w - 0.4 + bar_w/2, f1s, bar_w, label=sn)
axes[1].set_xticks(x2)
axes[1].set_xticklabels(CLASS_NAMES, rotation=20, ha='right')
axes[1].set_title('Per-Class F1 by Augmentation Strategy', fontweight='bold')
axes[1].set_ylabel('F1 Score')
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/07_augmentation_experiments.png', dpi=150, bbox_inches='tight')
plt.close()
log(f'  Saved: {OUT_DIR}/07_augmentation_experiments.png')

# Save experiment results JSON
with open(f'{OUT_DIR}/augmentation_experiment_results.json', 'w') as f:
    json.dump(experiment_results, f, indent=2)
log(f'  Saved: {OUT_DIR}/augmentation_experiment_results.json')


# ================================================================
# 8. GENERATE COMPREHENSIVE REPORT
# ================================================================
log('\n' + '=' * 65)
log('  SECTION 7: SUMMARY AND RECOMMENDATIONS')
log('=' * 65)

log('\n--- Key Findings ---')
log(f'  1. Dataset has {len(meta)} total samples across {NUM_CLASSES} classes')
log(f'  2. Severe class imbalance: {max_cls/min_cls:.1f}x ratio between majority and minority')
log(f'  3. Diabetes/DR dominates ({dist_all.get(1,0)} samples, '
    f'{100*dist_all.get(1,0)/len(meta):.0f}% of data) - augmented by APTOS dataset')
log(f'  4. Minority classes (Glaucoma, AMD) struggle most: '
    f'F1 = {ckpt.get("history",{}).get("f1_Glaucoma",["?"])[-1] if os.path.exists(MODEL_PATH) else "?"}, '
    f'{ckpt.get("history",{}).get("f1_AMD",["?"])[-1] if os.path.exists(MODEL_PATH) else "?"}')

log('\n--- Recommendations ---')
log('')
log('  DATA AUGMENTATION:')
log('  - Implement class-specific augmentation policies')
log('  - Minority classes (Glaucoma, AMD) need stronger augmentation:')
log('    * Rotation up to 45 degrees')
log('    * Elastic deformation')
log('    * Mixup/CutMix with alpha=0.4')
log('  - Consider RandAugment or AutoAugment for automated policy search')
log('')
log('  SAMPLING STRATEGY:')
log('  - Current approach: no oversampling + Focal Loss')
log('  - Suggest: Square-root sampling (balance between uniform and natural)')
log('  - Or: Progressive rebalancing (start natural, increase balance over epochs)')
log('')
log('  PREPROCESSING:')
log('  - Ben Graham preprocessing works well for enhancing vessel structures')
log('  - Consider CLAHE (Contrast Limited Adaptive Histogram Equalization) as alternative')
log('  - Minority classes may benefit from different sigma in Gaussian blur')
log('')
log('  MODEL IMPROVEMENTS:')
log('  - Use label smoothing (0.1) to reduce overconfidence')
log('  - Implement Mixup training for better calibration')
log('  - Try larger backbone (EfficientNet-B5 or ViT)')
log('  - Ensemble multiple models for better minority class performance')
log('')
log('  TRAINING STRATEGY:')
log('  - Increase epochs to 40-50 with patience=12')
log('  - Use cosine annealing with warm restarts')
log('  - Implement knowledge distillation from larger model')
log('  - Add test-time augmentation (TTA) for evaluation')

# Save quality data CSV
qdf.to_csv(f'{OUT_DIR}/image_quality_metrics.csv', index=False)
log(f'\n  Saved: {OUT_DIR}/image_quality_metrics.csv')

# Save full report
report_path = f'{OUT_DIR}/analysis_report.txt'
with open(report_path, 'w') as f:
    f.write('\n'.join(report_lines))
log(f'  Saved: {report_path}')

# Save summary JSON
summary = {
    'total_samples': int(len(meta)),
    'train_samples': int(len(train_df)),
    'val_samples': int(len(val_df)),
    'class_distribution': {CLASS_NAMES[i]: int(dist_all.get(i, 0)) for i in range(NUM_CLASSES)},
    'imbalance_ratio': float(max_cls / min_cls),
    'majority_class': CLASS_NAMES[dist_all.idxmax()],
    'minority_class': CLASS_NAMES[dist_all.idxmin()],
    'image_quality_summary': {},
    'model_metrics': {},
}

for cls_idx in range(NUM_CLASSES):
    cq = qdf[qdf['class'] == cls_idx]
    summary['image_quality_summary'][CLASS_NAMES[cls_idx]] = {
        'mean_brightness': float(cq['brightness'].mean()),
        'mean_contrast': float(cq['contrast'].mean()),
        'mean_sharpness': float(cq['sharpness'].mean()),
        'n_analyzed': int(len(cq)),
    }

if os.path.exists(MODEL_PATH):
    summary['model_metrics'] = {
        'macro_f1': float(mf1),
        'weighted_f1': float(wf1),
        'macro_auc': float(mauc),
        'per_class_f1': {CLASS_NAMES[i]: float(f1_score(all_labels, all_preds,
                         average=None, labels=range(NUM_CLASSES))[i])
                         for i in range(NUM_CLASSES)},
        'per_class_accuracy': {CLASS_NAMES[i]: float(per_class_acc[i])
                               for i in range(NUM_CLASSES)},
    }

summary['augmentation_experiments'] = experiment_results
summary['best_augmentation_strategy'] = best_strat

with open(f'{OUT_DIR}/analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
log(f'  Saved: {OUT_DIR}/analysis_summary.json')

log(f'\n{"="*65}')
log(f'  Analysis Complete!')
log(f'  All outputs saved to: {OUT_DIR}/')
log(f'{"="*65}')
log(f'\n  Files generated:')
log(f'    01_class_distribution.png')
log(f'    02_image_quality.png')
log(f'    03_augmentation_samples.png')
log(f'    04_error_analysis.png')
log(f'    05_roc_curves.png')
log(f'    06_preprocessing_comparison.png')
log(f'    07_augmentation_experiments.png')
log(f'    image_quality_metrics.csv')
log(f'    augmentation_experiment_results.json')
log(f'    analysis_report.txt')
log(f'    analysis_summary.json')
