#!/usr/bin/env python3
"""
RetinaSense v3.0 — Production Training Script
==============================================
Vision Transformer (ViT-Base-Patch16-224) with multi-task heads for
retinal disease classification and diabetic retinopathy severity grading.

v3 Enhancements over ViT baseline:
  1. Layer-wise Learning Rate Decay (LLRD, decay=0.75)
  2. WeightedRandomSampler for class imbalance
  3. MixUp augmentation (alpha=0.4) with Focal Loss mixing
  4. CosineAnnealingWarmRestarts (T_0=25, T_mult=2)
  5. Extended training: 100 epochs, patience=20 on macro-F1
  6. Fundus-specific normalisation (loads from data/fundus_norm_stats.json)
  7. 3-way train/calib/test split (CSV-based or auto 70/15/15)
  8. Temperature scaling (post-training calibration on calib set)
  9. Per-class threshold optimisation on calib set, final eval on test set

Usage:
  python retinasense_v3.py
"""

import os
import sys
import time
import warnings
import json
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
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

import timm

from scipy.optimize import minimize_scalar

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize


# ================================================================
# CONFIG
# ================================================================
class Config:
    DATA_DIR   = './data'
    CACHE_DIR  = './preprocessed_cache_v3'
    OUTPUT_DIR = './outputs_v3'

    MODEL_NAME = 'vit_base_patch16_224'
    IMG_SIZE   = 224

    NUM_DISEASE_CLASSES  = 5
    NUM_SEVERITY_CLASSES = 5

    DROPOUT = 0.3  # reduced from 0.4 in v2

    BATCH_SIZE  = 32
    NUM_EPOCHS  = 3
    NUM_WORKERS = 8

    BASE_LR    = 3e-4
    LLRD_DECAY = 0.75
    WEIGHT_DECAY = 1e-4

    GRADIENT_ACCUMULATION = 2   # effective batch = 64

    FOCAL_GAMMA  = 1.0
    MIXUP_ALPHA  = 0.4

    PATIENCE  = 3
    MIN_DELTA = 0.001

    CLASS_NAMES = ['Normal', 'Diabetes/DR', 'Glaucoma', 'Cataract', 'AMD']

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Paths for 3-way splits
    TRAIN_CSV = './data/train_split.csv'
    CALIB_CSV = './data/calib_split.csv'
    TEST_CSV  = './data/test_split.csv'

    # ImageNet fallback normalisation
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]


cfg = Config()

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
os.makedirs(cfg.CACHE_DIR,  exist_ok=True)
os.makedirs(cfg.DATA_DIR,   exist_ok=True)

print('=' * 65)
print('      RetinaSense v3.0 — Production Training Pipeline')
print('=' * 65)
if torch.cuda.is_available():
    print(f'  GPU         : {torch.cuda.get_device_name(0)}')
    print(f'  VRAM        : {round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)} GB')
print(f'  Backbone    : {cfg.MODEL_NAME} (timm)')
print(f'  Epochs      : {cfg.NUM_EPOCHS}  (patience={cfg.PATIENCE})')
print(f'  Batch       : {cfg.BATCH_SIZE} (eff. {cfg.BATCH_SIZE * cfg.GRADIENT_ACCUMULATION} via grad accum)')
print(f'  LLRD decay  : {cfg.LLRD_DECAY}')
print(f'  MixUp alpha : {cfg.MIXUP_ALPHA}')
print(f'  Focal gamma : {cfg.FOCAL_GAMMA}')
print('=' * 65)


# ================================================================
# STEP 1 — NORMALISATION STATS
# ================================================================
print('\n[1/9] Loading normalisation stats...')

norm_stats_path = os.path.join(cfg.DATA_DIR, 'fundus_norm_stats.json')
if os.path.exists(norm_stats_path):
    with open(norm_stats_path) as f:
        norm_stats = json.load(f)
    NORM_MEAN = norm_stats['mean_rgb']
    NORM_STD  = norm_stats['std_rgb']
    print(f'  Fundus-specific stats loaded: mean={NORM_MEAN}, std={NORM_STD}')
else:
    NORM_MEAN = cfg.IMAGENET_MEAN
    NORM_STD  = cfg.IMAGENET_STD
    print(f'  fundus_norm_stats.json not found — using ImageNet defaults')
    print(f'  mean={NORM_MEAN}, std={NORM_STD}')


# ================================================================
# STEP 2 — METADATA
# ================================================================
print('\n[2/9] Building metadata...')

BASE = './'
disease_cols = ['N', 'D', 'G', 'C', 'A']
label_map    = {'N': 0, 'D': 1, 'G': 2, 'C': 3, 'A': 4}


def _load_odir(base):
    """Load and filter ODIR metadata to single-label samples."""
    odir_csv = os.path.join(base, 'odir', 'full_df.csv')
    if not os.path.exists(odir_csv):
        print('  WARNING: ODIR CSV not found, skipping ODIR samples')
        return pd.DataFrame()
    df = pd.read_csv(odir_csv)
    df['disease_count'] = df[disease_cols].sum(axis=1)
    df = df[df['disease_count'] == 1].copy()

    def get_label(row):
        for d in disease_cols:
            if row[d] == 1:
                return label_map[d]
    df['disease_label'] = df.apply(get_label, axis=1)

    img_col = next(
        c for c in df.columns
        if any(k in c.lower() for k in ['filename', 'fundus', 'image'])
    )
    out = pd.DataFrame({
        'image_path':    os.path.join(base, 'odir', 'preprocessed_images') + '/' + df[img_col].astype(str),
        'source':        'ODIR',
        'disease_label': df['disease_label'],
        'severity_label': -1,
    })
    return out


def _load_aptos(base):
    """Load APTOS metadata."""
    aptos_csv = os.path.join(base, 'aptos', 'train.csv')
    if not os.path.exists(aptos_csv):
        print('  WARNING: APTOS CSV not found, skipping APTOS samples')
        return pd.DataFrame()
    df = pd.read_csv(aptos_csv)
    out = pd.DataFrame({
        'image_path':    os.path.join(base, 'aptos', 'train_images') + '/' + df['id_code'] + '.png',
        'source':        'APTOS',
        'disease_label': 1,
        'severity_label': df['diagnosis'],
    })
    return out


def _load_refuge2(base):
    """Load REFUGE2 Glaucoma-only subset (~400 images).
    Only the Glaucoma class is used — targeted fix for the weakest class (308 samples).
    Images are Zeiss Visucam 500 quality — no Ben Graham needed."""
    glaucoma_dir = os.path.join(base, 'refuge2', 'Training400', 'Glaucoma')
    if not os.path.exists(glaucoma_dir):
        print('  WARNING: REFUGE2 not found, skipping (expected: refuge2/Training400/Glaucoma/)')
        return pd.DataFrame()
    imgs = [os.path.join(glaucoma_dir, f)
            for f in os.listdir(glaucoma_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not imgs:
        return pd.DataFrame()
    out = pd.DataFrame({
        'image_path':    imgs,
        'source':        'REFUGE2',
        'disease_label': 2,   # Glaucoma = class 2
        'severity_label': -1,
    })
    print(f'  REFUGE2 Glaucoma: {len(out)} images loaded')
    return out


odir_meta   = _load_odir(BASE)
aptos_meta  = _load_aptos(BASE)
refuge2_meta = _load_refuge2(BASE)

parts = [df for df in [odir_meta, aptos_meta, refuge2_meta] if len(df) > 0]
if len(parts) == 0:
    raise RuntimeError('No dataset found. Place ODIR/APTOS data under ./odir and ./aptos.')

meta = pd.concat(parts, ignore_index=True)
meta = meta[meta['image_path'].apply(os.path.exists)].reset_index(drop=True)

# severity -1 (unknown) → 0
meta['severity_label'] = meta['severity_label'].clip(lower=0).fillna(0).astype(int)

print(f'  Total valid samples: {len(meta)}')
dist = meta['disease_label'].value_counts().sort_index()
for i, cnt in dist.items():
    print(f'    {cfg.CLASS_NAMES[i]:15s}: {cnt:4d}  ({100 * cnt / len(meta):.1f}%)')


# ================================================================
# STEP 3 — PRE-CACHE
# ================================================================
print(f'\n[3/9] Pre-caching images @ {cfg.IMG_SIZE}x{cfg.IMG_SIZE}...')


def _read_rgb(path):
    """Read image from disk as RGB numpy array."""
    img = cv2.imread(path)
    if img is None:
        img = np.array(Image.open(path).convert('RGB'))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _circular_mask(img, sz):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (sz // 2, sz // 2), int(sz * 0.48), 255, -1)
    return cv2.bitwise_and(img, img, mask=mask)


def ben_graham(path, sz=cfg.IMG_SIZE, sigma=10):
    """Ben Graham enhancement for APTOS field-camera images.
    Removes low-frequency illumination gradients, amplifies vessel/lesion detail."""
    img = cv2.resize(_read_rgb(path), (sz, sz))
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigma), -4, 128)
    return _circular_mask(img, sz)


def clahe_preprocess(path, sz=cfg.IMG_SIZE):
    """CLAHE preprocessing for ODIR multi-source clinical images.
    Normalises local contrast without destroying fine vessel/drusen detail."""
    img = cv2.resize(_read_rgb(path), (sz, sz))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return _circular_mask(img, sz)


def resize_only(path, sz=cfg.IMG_SIZE):
    """Minimal preprocessing for already-clinical-grade images (REFUGE2).
    Zeiss Visucam 500 images are standardised high quality — no enhancement needed."""
    img = cv2.resize(_read_rgb(path), (sz, sz))
    return _circular_mask(img, sz)


def preprocess_image(path, source, sz=cfg.IMG_SIZE):
    """Source-conditional preprocessing dispatcher.
      APTOS   -> Ben Graham  (field camera, vignetting correction)
      ODIR    -> CLAHE       (multi-source clinical, contrast normalisation)
      REFUGE2 -> Resize only (Zeiss Visucam 500, already high quality)
    """
    src = str(source).upper()
    if src == 'APTOS':
        return ben_graham(path, sz)
    if src == 'REFUGE2':
        return resize_only(path, sz)
    return clahe_preprocess(path, sz)


def _cache_key(image_path):
    """Filename-based cache key (basename without extension)."""
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(cfg.CACHE_DIR, f'{stem}_{cfg.IMG_SIZE}.npy')


cache_paths = []
cached = 0
for _, row in tqdm(meta.iterrows(), total=len(meta), desc='Caching'):
    fp = _cache_key(row['image_path'])
    if not os.path.exists(fp):
        try:
            np.save(fp, preprocess_image(row['image_path'], row['source']))
        except Exception:
            np.save(fp, np.zeros((cfg.IMG_SIZE, cfg.IMG_SIZE, 3), dtype=np.uint8))
        cached += 1
    cache_paths.append(fp)

meta['cache_path'] = cache_paths
print(f'  Newly cached: {cached} | Already cached: {len(meta) - cached}')


# ================================================================
# STEP 4 — 3-WAY SPLIT
# ================================================================
print('\n[4/9] Preparing train / calib / test splits...')


def _load_or_create_splits(meta_df):
    """
    Load splits from CSV files if they exist (train/calib/test).
    Otherwise perform a stratified 70/15/15 auto-split and persist
    the CSVs so future runs are reproducible.

    Returns (train_df, calib_df, test_df).
    """
    splits_exist = (os.path.exists(cfg.TRAIN_CSV) and
                    os.path.exists(cfg.CALIB_CSV) and
                    os.path.exists(cfg.TEST_CSV))
    if splits_exist:
        train_df = pd.read_csv(cfg.TRAIN_CSV)
        calib_df = pd.read_csv(cfg.CALIB_CSV)
        test_df  = pd.read_csv(cfg.TEST_CSV)
        # Regenerate if any source is in current metadata but absent from saved splits
        stale = False
        for src in ['APTOS', 'REFUGE2']:
            if (src in meta_df['source'].values and
                    ('source' not in train_df.columns or
                     src not in train_df['source'].values)):
                print(f'  Stale splits detected ({src} missing) — regenerating...')
                stale = True
                break
        if stale:
            splits_exist = False  # fall through to recreate
        else:
            print(f'  Loaded existing splits: train={len(train_df)}, '
                  f'calib={len(calib_df)}, test={len(test_df)}')
    if not splits_exist:
        print('  Split files not found — creating 70/15/15 stratified split...')
        train_df, temp_df = train_test_split(
            meta_df, test_size=0.30,
            stratify=meta_df['disease_label'], random_state=42
        )
        calib_df, test_df = train_test_split(
            temp_df, test_size=0.50,
            stratify=temp_df['disease_label'], random_state=42
        )
        train_df.to_csv(cfg.TRAIN_CSV, index=False)
        calib_df.to_csv(cfg.CALIB_CSV, index=False)
        test_df.to_csv(cfg.TEST_CSV,   index=False)
        print(f'  Auto-split saved: train={len(train_df)}, '
              f'calib={len(calib_df)}, test={len(test_df)}')
    return train_df, calib_df, test_df


train_df, calib_df, test_df = _load_or_create_splits(meta)


# ================================================================
# STEP 5 — DATASET + TRANSFORMS
# ================================================================
print('\n[5/9] Building dataset and loaders...')


def make_transforms(phase):
    """
    Return torchvision transform pipeline.
    Train: spatial augmentation + color jitter + random erasing.
    Val / calib / test: deterministic normalisation only.
    """
    normalize = transforms.Normalize(NORM_MEAN, NORM_STD)
    if phase == 'train':
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(20),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.2),
        ])
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        normalize,
    ])


class RetinalDataset(Dataset):
    """
    Retinal fundus image dataset.

    Loads from preprocessed_cache_v3/ using a filename-based key.
    Falls back to on-the-fly ben_graham preprocessing if cache is
    missing (rare; cache is built in step 3).

    severity_label -1 is mapped to 0 (unknown severity).
    """

    def __init__(self, df, transform):
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load from cache (fast path)
        cache_fp = row.get('cache_path', _cache_key(row['image_path']))
        try:
            img = np.load(cache_fp)
        except Exception:
            # Fallback: source-conditional preprocess on the fly
            try:
                img = preprocess_image(row['image_path'], row.get('source', 'ODIR'))
            except Exception:
                img = np.zeros((cfg.IMG_SIZE, cfg.IMG_SIZE, 3), dtype=np.uint8)

        img_tensor = self.transform(img)

        disease_lbl  = int(row['disease_label'])
        severity_lbl = int(row['severity_label'])
        # Map -1 (unknown) → 0
        if severity_lbl < 0:
            severity_lbl = 0

        return (
            img_tensor,
            torch.tensor(disease_lbl,  dtype=torch.long),
            torch.tensor(severity_lbl, dtype=torch.long),
        )


# --- WeightedRandomSampler for training ---
def _make_weighted_sampler(df):
    """
    Compute per-sample weights inversely proportional to class frequency.
    Every batch will see all 5 classes roughly equally.
    """
    labels    = df['disease_label'].values
    class_cnt = np.bincount(labels, minlength=cfg.NUM_DISEASE_CLASSES).astype(float)
    class_cnt = np.where(class_cnt == 0, 1.0, class_cnt)
    weights   = 1.0 / class_cnt[labels]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(weights),
        replacement=True,
    )


train_ds = RetinalDataset(train_df, make_transforms('train'))
calib_ds = RetinalDataset(calib_df, make_transforms('val'))
test_ds  = RetinalDataset(test_df,  make_transforms('val'))

sampler = _make_weighted_sampler(train_df)

train_loader = DataLoader(
    train_ds, batch_size=cfg.BATCH_SIZE,
    sampler=sampler,          # WeightedRandomSampler replaces shuffle=True
    num_workers=cfg.NUM_WORKERS, pin_memory=True,
    persistent_workers=True,  prefetch_factor=2,
)
calib_loader = DataLoader(
    calib_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
    num_workers=cfg.NUM_WORKERS, pin_memory=True,
    persistent_workers=True,
)
test_loader = DataLoader(
    test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
    num_workers=cfg.NUM_WORKERS, pin_memory=True,
    persistent_workers=True,
)

print(f'  Train : {len(train_ds):5d}  ({len(train_loader):3d} batches) — WeightedRandomSampler')
print(f'  Calib : {len(calib_ds):5d}  ({len(calib_loader):3d} batches)')
print(f'  Test  : {len(test_ds):5d}  ({len(test_loader):3d} batches)  [SEALED until final eval]')


# ================================================================
# STEP 6 — MODEL, LOSS, LLRD OPTIMIZER
# ================================================================
print('\n[6/9] Building model and optimizer...')


# --- Focal Loss ---
class FocalLoss(nn.Module):
    """
    Focal Loss — down-weights easy examples, focuses on hard ones.
    alpha: per-class weight tensor; gamma: focusing parameter.
    """

    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None

    def forward(self, logits, targets):
        ce    = F.cross_entropy(logits, targets, reduction='none')
        pt    = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            at    = self.alpha.gather(0, targets)
            focal = at * focal
        return focal.mean()


# --- Multi-task ViT ---
class MultiTaskViT(nn.Module):
    """
    ViT-Base-Patch16-224 backbone with two classification heads:
      - disease_head  : 5-class fundus disease classification
      - severity_head : 5-class DR severity grading (APTOS only)

    Dropout reduced to 0.3 (vs 0.4 in v2) since LLRD + MixUp
    already provide strong regularisation.
    """

    def __init__(self,
                 n_disease=cfg.NUM_DISEASE_CLASSES,
                 n_severity=cfg.NUM_SEVERITY_CLASSES,
                 drop=cfg.DROPOUT):
        super().__init__()
        self.backbone = timm.create_model(
            cfg.MODEL_NAME, pretrained=True, num_classes=0
        )
        feat = 768  # ViT-Base CLS token dimension

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
        f = self.backbone(x)   # (B, 768)
        f = self.drop(f)
        return self.disease_head(f), self.severity_head(f)


# --- Layer-wise Learning Rate Decay (LLRD) ---
def get_optimizer_with_llrd(model, base_lr=cfg.BASE_LR, decay_factor=cfg.LLRD_DECAY):
    """
    Build AdamW with per-parameter-group learning rates following LLRD.

    Strategy (head → patch_embed, each step multiplies by decay_factor):
      - disease_head / severity_head / drop : base_lr (full rate = 3e-4)
      - blocks[11] : base_lr * decay^1
      - blocks[10] : base_lr * decay^2
      ...
      - blocks[0]  : base_lr * decay^12
      - patch_embed + cls_token + pos_embed : base_lr * decay^13  (~1e-6)
      - norm       : same as last block

    Returns: AdamW optimizer with separate param groups.
    """
    param_groups = []

    # 1. Classification heads (full LR)
    head_params = (
        list(model.disease_head.parameters()) +
        list(model.severity_head.parameters()) +
        list(model.drop.parameters())
    )
    param_groups.append({'params': head_params, 'lr': base_lr})

    # 2. Transformer blocks (12 blocks, indexed 11 → 0)
    blocks = model.backbone.blocks  # nn.Sequential of 12 blocks
    num_blocks = len(blocks)
    for block_idx in range(num_blocks - 1, -1, -1):
        distance_from_head = num_blocks - block_idx  # 1 for block[11], 12 for block[0]
        lr_i = base_lr * (decay_factor ** distance_from_head)
        param_groups.append({
            'params': list(blocks[block_idx].parameters()),
            'lr': lr_i,
        })

    # 3. Patch embedding + positional embedding + CLS token + norm
    embed_lr = base_lr * (decay_factor ** (num_blocks + 1))
    embed_params = (
        list(model.backbone.patch_embed.parameters()) +
        [model.backbone.cls_token,
         model.backbone.pos_embed] +
        list(model.backbone.norm.parameters())
    )
    param_groups.append({'params': embed_params, 'lr': embed_lr})

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    # Log LR distribution
    lrs = [g['lr'] for g in param_groups]
    print(f'  LLRD optimizer: {len(param_groups)} param groups')
    print(f'    Head LR    : {lrs[0]:.2e}')
    print(f'    Block[11]  : {lrs[1]:.2e}')
    print(f'    Block[0]   : {lrs[-2]:.2e}')
    print(f'    Embed LR   : {lrs[-1]:.2e}')

    return optimizer


# --- Instantiate model ---
model = MultiTaskViT().to(cfg.DEVICE)

# --- Focal loss class weights (computed on train set) ---
cw    = compute_class_weight('balanced',
                             classes=np.arange(cfg.NUM_DISEASE_CLASSES),
                             y=train_df['disease_label'].values)
alpha = torch.tensor(cw, dtype=torch.float32).to(cfg.DEVICE)
alpha = alpha / alpha.sum() * cfg.NUM_DISEASE_CLASSES  # normalise
print(f'  Focal alpha: {[f"{a:.2f}" for a in alpha.tolist()]}')

criterion_d = FocalLoss(alpha=alpha, gamma=cfg.FOCAL_GAMMA)
criterion_s = nn.CrossEntropyLoss(ignore_index=-1)

total_params = sum(p.numel() for p in model.parameters())
print(f'  Total params: {total_params:,}')

# --- Optimizer (LLRD) ---
optimizer = get_optimizer_with_llrd(model)

# --- Scheduler: OneCycleLR ---
# 10% warmup then cosine decay — avoids the epoch-3 LR collapse from
# CosineAnnealingWarmRestarts. Stepped once per optimizer update (per batch).
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=[pg['lr'] for pg in optimizer.param_groups],
    steps_per_epoch=len(train_loader),
    epochs=cfg.NUM_EPOCHS,
    pct_start=0.1,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=100.0,
)

scaler = GradScaler()


# ================================================================
# STEP 7 — MIXUP + TRAINING LOOP
# ================================================================

def mixup_data(x, y, alpha=cfg.MIXUP_ALPHA):
    """
    MixUp augmentation.

    Returns mixed inputs, the two label tensors, and the mixing coefficient.
    Loss is mixed externally: lam * L(pred, y_a) + (1-lam) * L(pred, y_b).
    """
    lam        = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index      = torch.randperm(batch_size, device=x.device)
    mixed_x    = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def evaluate(loader, model, criterion_d, criterion_s, device, desc='Eval'):
    """
    Run inference on a DataLoader.

    Returns:
        loss     : average total loss
        preds    : numpy array of argmax predictions
        targets  : numpy array of ground-truth labels
        probs    : numpy array of softmax probabilities (N, C)
    """
    model.eval()
    total_loss = 0.0
    all_preds, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for imgs, d_lbl, s_lbl in tqdm(loader, desc=desc, leave=False):
            imgs  = imgs.to(device, non_blocking=True)
            d_lbl = d_lbl.to(device, non_blocking=True)
            s_lbl = s_lbl.to(device, non_blocking=True)

            with autocast('cuda'):
                d_out, s_out = model(imgs)
                ld   = criterion_d(d_out, d_lbl)
                ls   = criterion_s(s_out, s_lbl)
                loss = ld + 0.2 * ls

            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()

            probs = torch.softmax(d_out.float(), dim=1)
            all_preds.extend(d_out.argmax(1).cpu().numpy())
            all_targets.extend(d_lbl.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(loader)
    return (avg_loss,
            np.array(all_preds),
            np.array(all_targets),
            np.array(all_probs))


print('\n[7/9] Training...')

CHECKPOINT = os.path.join(cfg.OUTPUT_DIR, 'best_model.pth')

history = {k: [] for k in [
    'train_loss', 'val_loss', 'train_acc', 'val_acc',
    'macro_f1', 'weighted_f1', 'lr',
    *(f'f1_{c}' for c in cfg.CLASS_NAMES)
]}

best_f1      = 0.0
patience_ctr = 0

t_start = time.time()
print('=' * 65)

for epoch in range(cfg.NUM_EPOCHS):
    t0 = time.time()

    # ---- TRAIN ----
    model.train()
    run_loss = 0.0
    correct  = 0
    total    = 0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(train_loader,
                desc=f'E{epoch+1:03d}/{cfg.NUM_EPOCHS} train',
                leave=False)

    for step, (imgs, d_lbl, s_lbl) in enumerate(pbar):
        imgs  = imgs.to(cfg.DEVICE, non_blocking=True)
        d_lbl = d_lbl.to(cfg.DEVICE, non_blocking=True)
        s_lbl = s_lbl.to(cfg.DEVICE, non_blocking=True)

        # MixUp augmentation (train only)
        mixed_imgs, y_a, y_b, lam = mixup_data(imgs, d_lbl, alpha=cfg.MIXUP_ALPHA)

        with autocast('cuda'):
            d_out, s_out = model(mixed_imgs)

            # Mixed Focal Loss: lam * L(y_a) + (1-lam) * L(y_b)
            loss_d = lam * criterion_d(d_out, y_a) + (1 - lam) * criterion_d(d_out, y_b)
            loss_s = criterion_s(s_out, s_lbl)
            loss   = (loss_d + 0.2 * loss_s) / cfg.GRADIENT_ACCUMULATION

        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()

        if (step + 1) % cfg.GRADIENT_ACCUMULATION == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        run_loss += loss.item() * cfg.GRADIENT_ACCUMULATION
        # Use un-mixed predictions for accuracy tracking
        with torch.no_grad():
            preds    = d_out.argmax(1)
            correct += (preds == y_a).sum().item()
            total   += d_lbl.size(0)

        pbar.set_postfix(
            loss=f'{loss.item() * cfg.GRADIENT_ACCUMULATION:.3f}',
            acc=f'{100 * correct / total:.1f}%'
        )

    # Flush remaining gradients for incomplete accumulation window
    if (len(train_loader)) % cfg.GRADIENT_ACCUMULATION != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    train_loss = run_loss / len(train_loader)
    train_acc  = 100 * correct / total

    # ---- VALIDATE on calibration set ----
    val_loss, val_preds, val_targets, val_probs = evaluate(
        calib_loader, model, criterion_d, criterion_s, cfg.DEVICE,
        desc=f'E{epoch+1:03d}/{cfg.NUM_EPOCHS} calib'
    )

    val_acc = 100 * (val_preds == val_targets).mean()
    mf1     = f1_score(val_targets, val_preds, average='macro')
    wf1     = f1_score(val_targets, val_preds, average='weighted')
    per_f1  = f1_score(val_targets, val_preds,
                       average=None, labels=range(cfg.NUM_DISEASE_CLASSES),
                       zero_division=0)

    lr_now = optimizer.param_groups[0]['lr']

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    history['macro_f1'].append(mf1)
    history['weighted_f1'].append(wf1)
    history['lr'].append(lr_now)
    for ci, cn in enumerate(cfg.CLASS_NAMES):
        history[f'f1_{cn}'].append(float(per_f1[ci]))

    elapsed = time.time() - t0

    # ---- Early stopping on macro-F1 (with min_delta) ----
    tag = ''
    if mf1 > best_f1 + cfg.MIN_DELTA:
        best_f1      = mf1
        patience_ctr = 0
        torch.save({
            'epoch':           epoch,
            'model_state_dict': model.state_dict(),
            'val_acc':         val_acc,
            'macro_f1':        mf1,
            'history':         history,
        }, CHECKPOINT)
        tag = f'  * NEW BEST (macro-F1={mf1:.4f})'
    else:
        patience_ctr += 1

    cls_str = ' | '.join(
        f'{cn[:3]}:{per_f1[ci]:.2f}'
        for ci, cn in enumerate(cfg.CLASS_NAMES)
    )
    print(
        f'E{epoch+1:03d} | {elapsed:.0f}s | LR {lr_now:.2e} | '
        f'TrL {train_loss:.3f} TrA {train_acc:.1f}% | '
        f'VL {val_loss:.3f} VA {val_acc:.1f}% | '
        f'mF1 {mf1:.4f} wF1 {wf1:.4f}{tag}'
    )
    print(f'       {cls_str}')

    if patience_ctr >= cfg.PATIENCE:
        print(f'\n  Early stopping — no improvement for {cfg.PATIENCE} epochs')
        break

total_train_time = time.time() - t_start
print(f'\nTraining complete. Best macro-F1: {best_f1:.4f}')
print(f'Total training time: {total_train_time / 60:.1f} minutes')

# Save training history
with open(os.path.join(cfg.OUTPUT_DIR, 'history.json'), 'w') as f:
    json.dump({k: [float(v) for v in vs] for k, vs in history.items()}, f, indent=2)


# ================================================================
# STEP 8 — TEMPERATURE SCALING (post-training calibration)
# ================================================================
print('\n[8/9] Temperature scaling on calibration set...')

# Reload best model
ckpt = torch.load(CHECKPOINT, map_location=cfg.DEVICE, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f'  Loaded best checkpoint (epoch {ckpt["epoch"]+1}, '
      f'macro-F1={ckpt["macro_f1"]:.4f})')


def _collect_logits_labels(loader, model, device):
    """Collect raw logits and labels (no softmax) from a DataLoader."""
    all_logits, all_labels = [], []
    with torch.no_grad():
        for imgs, d_lbl, _ in tqdm(loader, desc='Collecting logits', leave=False):
            imgs  = imgs.to(device, non_blocking=True)
            d_out, _ = model(imgs)
            all_logits.append(d_out.float().cpu())
            all_labels.append(d_lbl.cpu())
    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


def _ece(probs, labels, n_bins=15):
    """
    Expected Calibration Error.
    probs  : numpy (N, C) softmax probabilities
    labels : numpy (N,) ground truth class indices
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies  = predictions == labels

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece_val   = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        acc_bin  = accuracies[mask].mean()
        conf_bin = confidences[mask].mean()
        ece_val += mask.sum() * abs(acc_bin - conf_bin)
    return float(ece_val / len(labels))


calib_logits, calib_labels = _collect_logits_labels(calib_loader, model, cfg.DEVICE)

# ECE before calibration
probs_before = torch.softmax(calib_logits, dim=1).numpy()
ece_before   = _ece(probs_before, calib_labels.numpy())
print(f'  ECE before temperature scaling: {ece_before:.4f}')


def _nll_with_temperature(T, logits, labels):
    """Negative log-likelihood at temperature T (for scipy minimiser)."""
    scaled_logits = logits / T
    log_probs     = F.log_softmax(scaled_logits, dim=1)
    nll           = F.nll_loss(log_probs, labels).item()
    return nll


result = minimize_scalar(
    fun=_nll_with_temperature,
    args=(calib_logits, calib_labels),
    bounds=(0.01, 10.0),
    method='bounded',
)
T_opt = float(result.x)
print(f'  Optimal temperature T = {T_opt:.4f}')

probs_after = torch.softmax(calib_logits / T_opt, dim=1).numpy()
ece_after   = _ece(probs_after, calib_labels.numpy())
print(f'  ECE after  temperature scaling: {ece_after:.4f}')

# Save temperature
temp_path = os.path.join(cfg.OUTPUT_DIR, 'temperature.json')
with open(temp_path, 'w') as f:
    json.dump({'temperature': T_opt, 'ece_before': ece_before, 'ece_after': ece_after}, f, indent=2)
print(f'  Saved -> {temp_path}')


# ================================================================
# STEP 9 — THRESHOLD OPTIMISATION ON CALIB SET
# ================================================================
print('\n[9/9] Per-class threshold optimisation on calibration set...')


def optimise_thresholds(probs, labels, n_classes, n_grid=50):
    """
    Grid-search per-class decision thresholds on the calibration set.

    For each class c, sweep threshold in [0.05, 0.95] and pick
    the value maximising F1 for class c (one-vs-rest).

    Returns: list of per-class thresholds (length n_classes).
    """
    thresholds = []
    for c in range(n_classes):
        binary_labels = (labels == c).astype(int)
        best_t  = 0.5
        best_f1 = 0.0
        for t in np.linspace(0.05, 0.95, n_grid):
            preds_c = (probs[:, c] >= t).astype(int)
            f       = f1_score(binary_labels, preds_c, zero_division=0)
            if f > best_f1:
                best_f1 = f
                best_t  = t
        thresholds.append(float(best_t))
        print(f'    {cfg.CLASS_NAMES[c]:15s}: threshold={best_t:.3f}  (calib F1={best_f1:.3f})')
    return thresholds


calib_thresholds = optimise_thresholds(
    probs_after,
    calib_labels.numpy(),
    cfg.NUM_DISEASE_CLASSES,
)

thresh_path = os.path.join(cfg.OUTPUT_DIR, 'thresholds.json')
with open(thresh_path, 'w') as f:
    json.dump({'thresholds': calib_thresholds, 'class_names': cfg.CLASS_NAMES}, f, indent=2)
print(f'  Saved -> {thresh_path}')


def apply_thresholds(probs, thresholds):
    """
    Apply per-class thresholds to probability matrix.
    Assigns each sample to the class with highest prob-above-threshold.
    Falls back to argmax if no class exceeds its threshold.
    """
    preds = []
    for prob_row in probs:
        above  = [i for i, (p, t) in enumerate(zip(prob_row, thresholds)) if p >= t]
        preds.append(int(above[np.argmax([prob_row[i] for i in above])]
                         if above else np.argmax(prob_row)))
    return np.array(preds)


# ================================================================
# FINAL EVALUATION ON TEST SET (first and only time test is touched)
# ================================================================
print('\n' + '=' * 65)
print('         FINAL EVALUATION — TEST SET')
print('=' * 65)
print('  (Test set was never seen during training or threshold tuning)')

test_logits, test_labels = _collect_logits_labels(test_loader, model, cfg.DEVICE)
test_probs_calibrated    = torch.softmax(test_logits / T_opt, dim=1).numpy()
test_labels_np           = test_labels.numpy()

# Raw argmax predictions
test_preds_raw  = test_probs_calibrated.argmax(axis=1)

# Threshold-adjusted predictions
test_preds_thr  = apply_thresholds(test_probs_calibrated, calib_thresholds)

def _print_metrics(preds, targets, probs, label):
    acc  = 100 * (preds == targets).mean()
    mf1  = f1_score(targets, preds, average='macro')
    wf1  = f1_score(targets, preds, average='weighted')
    try:
        mauc = roc_auc_score(targets, probs, multi_class='ovr', average='macro')
    except Exception:
        mauc = 0.0
    per  = f1_score(targets, preds, average=None,
                    labels=range(cfg.NUM_DISEASE_CLASSES), zero_division=0)
    ece  = _ece(probs, targets)

    print(f'\n  [{label}]')
    print(f'  Accuracy   : {acc:.2f}%')
    print(f'  Macro F1   : {mf1:.4f}')
    print(f'  Weighted F1: {wf1:.4f}')
    print(f'  Macro AUC  : {mauc:.4f}')
    print(f'  ECE        : {ece:.4f}')
    print()
    print(classification_report(targets, preds,
                                target_names=cfg.CLASS_NAMES, digits=4))
    return {'accuracy': acc, 'macro_f1': mf1, 'weighted_f1': wf1,
            'macro_auc': mauc, 'ece': ece,
            **{f'f1_{cfg.CLASS_NAMES[i]}': float(per[i])
               for i in range(cfg.NUM_DISEASE_CLASSES)}}


metrics_raw = _print_metrics(test_preds_raw, test_labels_np,
                             test_probs_calibrated, 'Raw argmax (T-scaled)')
metrics_thr = _print_metrics(test_preds_thr, test_labels_np,
                             test_probs_calibrated, 'With per-class thresholds')

# Save final metrics
final_metrics = {
    'raw':        metrics_raw,
    'thresholded': metrics_thr,
    'temperature': T_opt,
    'thresholds':  calib_thresholds,
}
metrics_path = os.path.join(cfg.OUTPUT_DIR, 'final_metrics.json')
with open(metrics_path, 'w') as f:
    json.dump(final_metrics, f, indent=2)


# ================================================================
# PLOTS
# ================================================================
print('\nGenerating plots...')

ep     = range(1, len(history['train_loss']) + 1)
colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# 1. Loss
axes[0, 0].plot(ep, history['train_loss'], 'b-o', ms=3, label='Train')
axes[0, 0].plot(ep, history['val_loss'],   'r-o', ms=3, label='Calib')
axes[0, 0].set_title('Loss', fontweight='bold')
axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

# 2. Accuracy
axes[0, 1].plot(ep, history['train_acc'], 'b-o', ms=3, label='Train')
axes[0, 1].plot(ep, history['val_acc'],   'r-o', ms=3, label='Calib')
axes[0, 1].set_title('Accuracy (%)', fontweight='bold')
axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

# 3. F1
axes[0, 2].plot(ep, history['macro_f1'],    'g-o', ms=3, label='Macro F1')
axes[0, 2].plot(ep, history['weighted_f1'], 'm-o', ms=3, label='Weighted F1')
axes[0, 2].set_title('F1 Scores (calib)', fontweight='bold')
axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

# 4. Per-class F1
for ci, cn in enumerate(cfg.CLASS_NAMES):
    axes[1, 0].plot(ep, history[f'f1_{cn}'], '-o', ms=2,
                    color=colors[ci], label=cn)
axes[1, 0].set_title('Per-Class F1 (calib)', fontweight='bold')
axes[1, 0].legend(fontsize=8); axes[1, 0].grid(alpha=0.3)

# 5. Confusion matrix (test set, thresholded)
cm   = confusion_matrix(test_labels_np, test_preds_thr)
cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)
sns.heatmap(cm_n, annot=True, fmt='.2f', cmap='Blues', ax=axes[1, 1],
            xticklabels=cfg.CLASS_NAMES, yticklabels=cfg.CLASS_NAMES)
axes[1, 1].set_title('Confusion Matrix — Test Set (norm)', fontweight='bold')
axes[1, 1].set_ylabel('True'); axes[1, 1].set_xlabel('Pred')

# 6. ROC curves (test set)
y_bin = label_binarize(test_labels_np, classes=list(range(cfg.NUM_DISEASE_CLASSES)))
for ci, (cn, col) in enumerate(zip(cfg.CLASS_NAMES, colors)):
    fpr, tpr, _ = roc_curve(y_bin[:, ci], test_probs_calibrated[:, ci])
    axes[1, 2].plot(fpr, tpr, color=col, lw=2,
                    label=f'{cn} ({auc(fpr, tpr):.3f})')
axes[1, 2].plot([0, 1], [0, 1], 'k--', lw=1)
axes[1, 2].set_title('ROC Curves — Test Set', fontweight='bold')
axes[1, 2].legend(loc='lower right', fontsize=8)
axes[1, 2].grid(alpha=0.3)

plt.suptitle(
    f'RetinaSense v3.0 — Macro F1={metrics_thr["macro_f1"]:.3f} | '
    f'AUC={metrics_thr["macro_auc"]:.3f} | '
    f'Test Acc={metrics_thr["accuracy"]:.1f}%',
    fontsize=14, fontweight='bold', y=1.01
)
plt.tight_layout()
plt.savefig(os.path.join(cfg.OUTPUT_DIR, 'dashboard.png'), dpi=150, bbox_inches='tight')
plt.close()

# LR schedule plot
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(ep, history['lr'], 'b-o', ms=2)
ax.set_title('Learning Rate (head param group) — OneCycleLR',
             fontweight='bold')
ax.set_xlabel('Epoch'); ax.set_ylabel('LR')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(cfg.OUTPUT_DIR, 'lr_schedule.png'), dpi=150)
plt.close()

# Calibration reliability diagram
fig, ax = plt.subplots(figsize=(6, 6))
n_bins  = 15
confs   = test_probs_calibrated.max(axis=1)
acc_arr = (test_preds_thr == test_labels_np).astype(float)
bin_edges = np.linspace(0, 1, n_bins + 1)
bin_accs, bin_confs = [], []
for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
    mask = (confs >= lo) & (confs < hi)
    if mask.sum() > 0:
        bin_accs.append(acc_arr[mask].mean())
        bin_confs.append(confs[mask].mean())
ax.bar(bin_confs, bin_accs, width=1.0 / n_bins, alpha=0.7,
       edgecolor='black', label='Model')
ax.plot([0, 1], [0, 1], 'r--', lw=2, label='Perfect calibration')
ax.set_xlabel('Confidence'); ax.set_ylabel('Accuracy')
ax.set_title(f'Reliability Diagram (T={T_opt:.2f}, ECE={ece_after:.3f})',
             fontweight='bold')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(cfg.OUTPUT_DIR, 'calibration.png'), dpi=150)
plt.close()


# ================================================================
# SUMMARY
# ================================================================
print('\n' + '=' * 65)
print('          RETINASENSE v3.0 — FINAL SUMMARY')
print('=' * 65)
print(f'  Training epochs     : {len(history["train_loss"])}')
print(f'  Best calib macro-F1 : {best_f1:.4f}')
print(f'  Temperature T       : {T_opt:.4f}')
print(f'  ECE before / after  : {ece_before:.4f} / {ece_after:.4f}')
print()
print('  TEST SET RESULTS (with thresholds)')
print(f'    Accuracy   : {metrics_thr["accuracy"]:.2f}%')
print(f'    Macro F1   : {metrics_thr["macro_f1"]:.4f}')
print(f'    Weighted F1: {metrics_thr["weighted_f1"]:.4f}')
print(f'    Macro AUC  : {metrics_thr["macro_auc"]:.4f}')
print(f'    ECE        : {metrics_thr["ece"]:.4f}')
print()
print('  Per-class F1 (test, thresholded):')
for i, cn in enumerate(cfg.CLASS_NAMES):
    thr = calib_thresholds[i]
    fi  = metrics_thr[f'f1_{cn}']
    print(f'    {cn:15s}: F1={fi:.3f}  (threshold={thr:.3f})')
print()
print(f'  Training time       : {total_train_time / 60:.1f} minutes')
print()
print(f'  Outputs saved to {cfg.OUTPUT_DIR}/')
for fname in ['best_model.pth', 'history.json', 'temperature.json',
              'thresholds.json', 'final_metrics.json',
              'dashboard.png', 'lr_schedule.png', 'calibration.png']:
    print(f'    -- {fname}')
print('=' * 65)
