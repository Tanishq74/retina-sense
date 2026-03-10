#!/usr/bin/env python3
"""
RetinaSense — Domain-Adversarial Neural Network (DANN) Training
================================================================
Extends the MultiTaskViT (retinasense_v3.py) with a domain adversarial head
to learn domain-invariant representations across APTOS and ODIR datasets.

Architecture:
  - Backbone: ViT-Base/16 (timm, pretrained)
  - Disease head:  768 -> 512 -> 256 -> 5  (same as v3)
  - Severity head: 768 -> 256 -> 5          (same as v3)
  - Domain head:   768 -> GRL -> 256 -> 128 -> 2  (NEW: APTOS=0, ODIR=1)

The Gradient Reversal Layer (GRL) reverses gradients during backpropagation,
which trains the backbone to produce features that are discriminative for
disease classification but invariant to the source domain.

Lambda scheduling: domain loss weight ramps from 0 -> 1 over training using
  lambda_p = 2 / (1 + exp(-10 * p)) - 1     (p = training progress in [0,1])

Training recipe matches retinasense_v3.py:
  - AdamW with Layer-wise Learning Rate Decay (LLRD, decay=0.75)
  - OneCycleLR scheduler (10% warmup, cosine anneal)
  - Focal Loss for disease classification
  - MixUp augmentation (alpha=0.4)
  - Gradient accumulation (effective batch = 64)
  - WeightedRandomSampler for class balance
  - Mixed-precision training (AMP)

Usage:
  python train_dann.py
"""

import os
import sys
import time
import json
import warnings
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from collections import Counter
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.autograd import Function
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

import timm

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize


# ================================================================
# CONFIG
# ================================================================
class Config:
    DATA_DIR   = './data'
    CACHE_DIR  = './preprocessed_cache_v3'
    OUTPUT_DIR = './outputs_v3/dann'

    MODEL_NAME = 'vit_base_patch16_224'
    IMG_SIZE   = 224

    NUM_DISEASE_CLASSES  = 5
    NUM_SEVERITY_CLASSES = 5
    NUM_DOMAIN_CLASSES   = 2   # APTOS=0, ODIR=1

    DROPOUT = 0.3

    BATCH_SIZE  = 32
    NUM_EPOCHS  = 100
    NUM_WORKERS = 8

    BASE_LR    = 3e-4
    LLRD_DECAY = 0.75
    WEIGHT_DECAY = 1e-4

    GRADIENT_ACCUMULATION = 2   # effective batch = 64

    FOCAL_GAMMA  = 1.0
    MIXUP_ALPHA  = 0.4

    PATIENCE  = 20
    MIN_DELTA = 0.001

    # Domain adversarial loss weight (multiplied by scheduled lambda)
    DOMAIN_LOSS_WEIGHT = 1.0

    CLASS_NAMES  = ['Normal', 'Diabetes/DR', 'Glaucoma', 'Cataract', 'AMD']
    DOMAIN_NAMES = ['APTOS', 'ODIR']
    DOMAIN_MAP   = {'APTOS': 0, 'ODIR': 1, 'REFUGE2': 1}  # REFUGE2 grouped with ODIR

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Paths for 3-way splits
    TRAIN_CSV = './data/train_split.csv'
    CALIB_CSV = './data/calib_split.csv'
    TEST_CSV  = './data/test_split.csv'

    # Pretrained v3 checkpoint (for warm-starting)
    V3_CHECKPOINT = './outputs_v3/best_model.pth'

    # ImageNet fallback normalisation
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]


cfg = Config()

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
os.makedirs(cfg.CACHE_DIR,  exist_ok=True)
os.makedirs(cfg.DATA_DIR,   exist_ok=True)

print('=' * 65)
print('   RetinaSense DANN — Domain-Adversarial Training Pipeline')
print('=' * 65)
if torch.cuda.is_available():
    print(f'  GPU         : {torch.cuda.get_device_name(0)}')
    print(f'  VRAM        : {round(torch.cuda.get_device_properties(0).total_mem / 1e9, 1) if hasattr(torch.cuda.get_device_properties(0), "total_mem") else round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)} GB')
print(f'  Device      : {cfg.DEVICE}')
print(f'  Backbone    : {cfg.MODEL_NAME} (timm)')
print(f'  Epochs      : {cfg.NUM_EPOCHS}  (patience={cfg.PATIENCE})')
print(f'  Batch       : {cfg.BATCH_SIZE} (eff. {cfg.BATCH_SIZE * cfg.GRADIENT_ACCUMULATION} via grad accum)')
print(f'  LLRD decay  : {cfg.LLRD_DECAY}')
print(f'  MixUp alpha : {cfg.MIXUP_ALPHA}')
print(f'  Focal gamma : {cfg.FOCAL_GAMMA}')
print(f'  Domain wt   : {cfg.DOMAIN_LOSS_WEIGHT} (ramped by lambda schedule)')
print('=' * 65)


# ================================================================
# STEP 1 -- NORMALISATION STATS
# ================================================================
print('\n[1/7] Loading normalisation stats...')

norm_stats_path = os.path.join(cfg.DATA_DIR, 'fundus_norm_stats.json')
configs_norm_path = os.path.join('.', 'configs', 'fundus_norm_stats.json')

if os.path.exists(norm_stats_path):
    with open(norm_stats_path) as f:
        norm_stats = json.load(f)
    NORM_MEAN = norm_stats['mean_rgb']
    NORM_STD  = norm_stats['std_rgb']
    print(f'  Fundus-specific stats loaded: mean={NORM_MEAN}, std={NORM_STD}')
elif os.path.exists(configs_norm_path):
    with open(configs_norm_path) as f:
        norm_stats = json.load(f)
    NORM_MEAN = norm_stats['mean_rgb']
    NORM_STD  = norm_stats['std_rgb']
    print(f'  Fundus-specific stats loaded from configs/: mean={NORM_MEAN}, std={NORM_STD}')
else:
    NORM_MEAN = cfg.IMAGENET_MEAN
    NORM_STD  = cfg.IMAGENET_STD
    print(f'  fundus_norm_stats.json not found -- using ImageNet defaults')
    print(f'  mean={NORM_MEAN}, std={NORM_STD}')


# ================================================================
# STEP 2 -- METADATA + DOMAIN LABELS
# ================================================================
print('\n[2/7] Building metadata...')

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
        'image_path':     os.path.join(base, 'odir', 'preprocessed_images') + '/' + df[img_col].astype(str),
        'source':         'ODIR',
        'disease_label':  df['disease_label'],
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
        'image_path':     os.path.join(base, 'aptos', 'train_images') + '/' + df['id_code'] + '.png',
        'source':         'APTOS',
        'disease_label':  1,
        'severity_label': df['diagnosis'],
    })
    return out


def _load_refuge2(base):
    """Load REFUGE2 Glaucoma-only subset."""
    glaucoma_dir = os.path.join(base, 'refuge2', 'Training400', 'Glaucoma')
    if not os.path.exists(glaucoma_dir):
        print('  WARNING: REFUGE2 not found, skipping')
        return pd.DataFrame()
    imgs = [os.path.join(glaucoma_dir, f)
            for f in os.listdir(glaucoma_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not imgs:
        return pd.DataFrame()
    out = pd.DataFrame({
        'image_path':     imgs,
        'source':         'REFUGE2',
        'disease_label':  2,
        'severity_label': -1,
    })
    print(f'  REFUGE2 Glaucoma: {len(out)} images loaded')
    return out


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
    """Ben Graham enhancement for APTOS field-camera images."""
    img = cv2.resize(_read_rgb(path), (sz, sz))
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigma), -4, 128)
    return _circular_mask(img, sz)


def clahe_preprocess(path, sz=cfg.IMG_SIZE):
    """CLAHE preprocessing for ODIR multi-source clinical images."""
    img = cv2.resize(_read_rgb(path), (sz, sz))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return _circular_mask(img, sz)


def resize_only(path, sz=cfg.IMG_SIZE):
    """Minimal preprocessing for REFUGE2 images."""
    img = cv2.resize(_read_rgb(path), (sz, sz))
    return _circular_mask(img, sz)


def preprocess_image(path, source, sz=cfg.IMG_SIZE):
    """Source-conditional preprocessing dispatcher."""
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


# Build metadata from raw sources
odir_meta    = _load_odir(BASE)
aptos_meta   = _load_aptos(BASE)
refuge2_meta = _load_refuge2(BASE)

parts = [df for df in [odir_meta, aptos_meta, refuge2_meta] if len(df) > 0]
if len(parts) == 0:
    raise RuntimeError('No dataset found. Place ODIR/APTOS data under ./odir and ./aptos.')

meta = pd.concat(parts, ignore_index=True)
meta = meta[meta['image_path'].apply(os.path.exists)].reset_index(drop=True)
meta['severity_label'] = meta['severity_label'].clip(lower=0).fillna(0).astype(int)

# Assign domain labels
meta['domain_label'] = meta['source'].map(cfg.DOMAIN_MAP).fillna(1).astype(int)

print(f'  Total valid samples: {len(meta)}')
dist = meta['disease_label'].value_counts().sort_index()
for i, cnt in dist.items():
    print(f'    {cfg.CLASS_NAMES[i]:15s}: {cnt:4d}  ({100 * cnt / len(meta):.1f}%)')
dom_dist = meta['domain_label'].value_counts().sort_index()
for d, cnt in dom_dist.items():
    print(f'    Domain {cfg.DOMAIN_NAMES[d] if d < len(cfg.DOMAIN_NAMES) else d:6s}: {cnt:4d}  ({100 * cnt / len(meta):.1f}%)')


# ================================================================
# STEP 3 -- PRE-CACHE
# ================================================================
print(f'\n[3/7] Pre-caching images @ {cfg.IMG_SIZE}x{cfg.IMG_SIZE}...')

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
# STEP 4 -- 3-WAY SPLIT
# ================================================================
print('\n[4/7] Preparing train / calib / test splits...')


def _load_or_create_splits(meta_df):
    """
    Load splits from CSV files if they exist (train/calib/test).
    Otherwise perform a stratified 70/15/15 auto-split and persist the CSVs.
    Ensures domain_label column is present in all splits.
    """
    splits_exist = (os.path.exists(cfg.TRAIN_CSV) and
                    os.path.exists(cfg.CALIB_CSV) and
                    os.path.exists(cfg.TEST_CSV))
    if splits_exist:
        train_df = pd.read_csv(cfg.TRAIN_CSV)
        calib_df = pd.read_csv(cfg.CALIB_CSV)
        test_df  = pd.read_csv(cfg.TEST_CSV)

        # Regenerate if any expected source is missing from splits
        stale = False
        for src in ['APTOS', 'REFUGE2']:
            if (src in meta_df['source'].values and
                    ('source' not in train_df.columns or
                     src not in train_df['source'].values)):
                print(f'  Stale splits detected ({src} missing) -- regenerating...')
                stale = True
                break
        if stale:
            splits_exist = False
        else:
            print(f'  Loaded existing splits: train={len(train_df)}, '
                  f'calib={len(calib_df)}, test={len(test_df)}')
    if not splits_exist:
        print('  Split files not found -- creating 70/15/15 stratified split...')
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

    # Ensure domain_label exists on loaded splits
    for df in [train_df, calib_df, test_df]:
        if 'domain_label' not in df.columns:
            if 'source' in df.columns:
                df['domain_label'] = df['source'].map(cfg.DOMAIN_MAP).fillna(1).astype(int)
            else:
                # Fallback: merge from master metadata
                df['domain_label'] = 1  # default to ODIR if source unknown

    return train_df, calib_df, test_df


train_df, calib_df, test_df = _load_or_create_splits(meta)


# ================================================================
# STEP 5 -- DATASET + TRANSFORMS
# ================================================================
print('\n[5/7] Building dataset and loaders...')


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


class DANNRetinalDataset(Dataset):
    """
    Retinal fundus image dataset with domain labels.

    Returns (image_tensor, disease_label, severity_label, domain_label).
    Loads from preprocessed_cache_v3/ .npy files.
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
        severity_lbl = int(row.get('severity_label', 0))
        if severity_lbl < 0:
            severity_lbl = 0

        domain_lbl = int(row.get('domain_label', 1))

        return (
            img_tensor,
            torch.tensor(disease_lbl,  dtype=torch.long),
            torch.tensor(severity_lbl, dtype=torch.long),
            torch.tensor(domain_lbl,   dtype=torch.long),
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


train_ds = DANNRetinalDataset(train_df, make_transforms('train'))
calib_ds = DANNRetinalDataset(calib_df, make_transforms('val'))
test_ds  = DANNRetinalDataset(test_df,  make_transforms('val'))

sampler = _make_weighted_sampler(train_df)

train_loader = DataLoader(
    train_ds, batch_size=cfg.BATCH_SIZE,
    sampler=sampler,
    num_workers=cfg.NUM_WORKERS, pin_memory=True,
    persistent_workers=True, prefetch_factor=2,
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

print(f'  Train : {len(train_ds):5d}  ({len(train_loader):3d} batches) -- WeightedRandomSampler')
print(f'  Calib : {len(calib_ds):5d}  ({len(calib_loader):3d} batches)')
print(f'  Test  : {len(test_ds):5d}  ({len(test_loader):3d} batches)  [SEALED until final eval]')

# Domain distribution in train set
if 'domain_label' in train_df.columns:
    dom_tr = train_df['domain_label'].value_counts().sort_index()
    for d, cnt in dom_tr.items():
        name = cfg.DOMAIN_NAMES[d] if d < len(cfg.DOMAIN_NAMES) else f'Domain-{d}'
        print(f'    Train domain {name}: {cnt}')


# ================================================================
# STEP 6 -- MODEL, LOSS, LLRD OPTIMIZER
# ================================================================
print('\n[6/7] Building DANN model and optimizer...')


# --- Gradient Reversal Layer ---
class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer (GRL) autograd function.

    Forward pass: identity (no change to activations).
    Backward pass: negate and scale gradients by -lambda.

    This forces the backbone to learn features that confuse the domain
    discriminator (i.e., domain-invariant features).
    """

    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.save_for_backward(lambda_val)
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        lambda_val, = ctx.saved_tensors
        return -lambda_val * grad_output, None


class GradientReversalLayer(nn.Module):
    """
    Wraps GradientReversalFunction as a module.
    lambda_val controls the scale of gradient reversal.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer('lambda_val', torch.tensor(1.0))

    def set_lambda(self, val):
        self.lambda_val.fill_(val)

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_val)


def compute_lambda_p(progress):
    """
    DANN lambda schedule from Ganin et al. (2016).
    Ramps from 0 to 1 following:  lambda = 2 / (1 + exp(-10 * p)) - 1
    where p is training progress in [0, 1].
    """
    return float(2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0)


# --- Focal Loss ---
class FocalLoss(nn.Module):
    """
    Focal Loss -- down-weights easy examples, focuses on hard ones.
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


# --- DANN Multi-task ViT ---
class DANNMultiTaskViT(nn.Module):
    """
    ViT-Base-Patch16-224 with three heads:
      - disease_head  : 5-class fundus disease classification
      - severity_head : 5-class DR severity grading (APTOS only)
      - domain_head   : 2-class domain discriminator (APTOS vs ODIR)

    The domain head is connected through a Gradient Reversal Layer (GRL)
    which negates gradients during backpropagation, forcing the backbone
    to learn domain-invariant feature representations.
    """

    def __init__(self,
                 n_disease=cfg.NUM_DISEASE_CLASSES,
                 n_severity=cfg.NUM_SEVERITY_CLASSES,
                 n_domain=cfg.NUM_DOMAIN_CLASSES,
                 drop=cfg.DROPOUT):
        super().__init__()
        self.backbone = timm.create_model(
            cfg.MODEL_NAME, pretrained=True, num_classes=0
        )
        feat = 768  # ViT-Base CLS token dimension

        self.drop = nn.Dropout(drop)

        # Disease classification head (same as v3)
        self.disease_head = nn.Sequential(
            nn.Linear(feat, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, n_disease),
        )

        # Severity grading head (same as v3)
        self.severity_head = nn.Sequential(
            nn.Linear(feat, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, n_severity),
        )

        # Domain discriminator head (NEW) with GRL
        self.grl = GradientReversalLayer()
        self.domain_head = nn.Sequential(
            nn.Linear(feat, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_domain),
        )

    def forward(self, x, return_features=False):
        f = self.backbone(x)   # (B, 768)
        f = self.drop(f)

        disease_out  = self.disease_head(f)
        severity_out = self.severity_head(f)

        # Domain head: features pass through GRL before entering discriminator
        f_reversed = self.grl(f)
        domain_out = self.domain_head(f_reversed)

        if return_features:
            return disease_out, severity_out, domain_out, f
        return disease_out, severity_out, domain_out


# --- Layer-wise Learning Rate Decay (LLRD) ---
def get_optimizer_with_llrd(model, base_lr=cfg.BASE_LR, decay_factor=cfg.LLRD_DECAY):
    """
    Build AdamW with per-parameter-group learning rates following LLRD.

    Strategy (head -> patch_embed, each step multiplies by decay_factor):
      - disease_head / severity_head / domain_head / drop / grl : base_lr
      - blocks[11] : base_lr * decay^1
      - blocks[10] : base_lr * decay^2
      ...
      - blocks[0]  : base_lr * decay^12
      - patch_embed + cls_token + pos_embed + norm : base_lr * decay^13
    """
    param_groups = []

    # 1. All head parameters (full LR) -- includes domain_head and grl
    head_params = (
        list(model.disease_head.parameters()) +
        list(model.severity_head.parameters()) +
        list(model.domain_head.parameters()) +
        list(model.drop.parameters()) +
        list(model.grl.parameters())
    )
    param_groups.append({'params': head_params, 'lr': base_lr})

    # 2. Transformer blocks (12 blocks, indexed 11 -> 0)
    blocks = model.backbone.blocks
    num_blocks = len(blocks)
    for block_idx in range(num_blocks - 1, -1, -1):
        distance_from_head = num_blocks - block_idx
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
model = DANNMultiTaskViT().to(cfg.DEVICE)

# --- Optionally warm-start from v3 checkpoint ---
if os.path.exists(cfg.V3_CHECKPOINT):
    print(f'  Loading v3 checkpoint for warm-start: {cfg.V3_CHECKPOINT}')
    ckpt = torch.load(cfg.V3_CHECKPOINT, map_location=cfg.DEVICE, weights_only=False)
    v3_state = ckpt.get('model_state_dict', ckpt)
    # Load matching keys (backbone + disease/severity heads), skip domain head
    model_state = model.state_dict()
    loaded_keys = []
    for k, v in v3_state.items():
        if k in model_state and model_state[k].shape == v.shape:
            model_state[k] = v
            loaded_keys.append(k)
    model.load_state_dict(model_state)
    print(f'  Warm-started {len(loaded_keys)}/{len(model_state)} parameters from v3')
else:
    print(f'  No v3 checkpoint found at {cfg.V3_CHECKPOINT} -- training from ImageNet init')

# --- Focal loss class weights (computed on train set) ---
cw    = compute_class_weight('balanced',
                             classes=np.arange(cfg.NUM_DISEASE_CLASSES),
                             y=train_df['disease_label'].values)
alpha_weights = torch.tensor(cw, dtype=torch.float32).to(cfg.DEVICE)
alpha_weights = alpha_weights / alpha_weights.sum() * cfg.NUM_DISEASE_CLASSES
print(f'  Focal alpha: {[f"{a:.2f}" for a in alpha_weights.tolist()]}')

criterion_disease  = FocalLoss(alpha=alpha_weights, gamma=cfg.FOCAL_GAMMA)
criterion_severity = nn.CrossEntropyLoss(ignore_index=-1)
criterion_domain   = nn.CrossEntropyLoss()

total_params = sum(p.numel() for p in model.parameters())
print(f'  Total params: {total_params:,}')

# --- Optimizer (LLRD) ---
optimizer = get_optimizer_with_llrd(model)

# --- Scheduler: OneCycleLR ---
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
# STEP 7 -- MIXUP + DANN TRAINING LOOP
# ================================================================

def mixup_data(x, y_disease, y_domain, alpha=cfg.MIXUP_ALPHA):
    """
    MixUp augmentation for DANN.
    Mixes inputs and returns permuted labels for both disease and domain.
    Domain labels are NOT mixed (we use the original domain labels, since
    mixing two images from different domains does not have a clear domain label).
    """
    lam        = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index      = torch.randperm(batch_size, device=x.device)
    mixed_x    = lam * x + (1 - lam) * x[index]
    return mixed_x, y_disease, y_disease[index], y_domain, y_domain[index], lam, index


def evaluate(loader, model, criterion_d, criterion_s, criterion_dom, device,
             lambda_val=1.0, desc='Eval'):
    """
    Run inference on a DataLoader.

    Returns dict with loss, preds, targets, probs, domain metrics.
    """
    model.eval()
    model.grl.set_lambda(lambda_val)

    total_loss = 0.0
    total_disease_loss = 0.0
    total_domain_loss = 0.0
    all_preds, all_targets, all_probs = [], [], []
    all_dom_preds, all_dom_targets = [], []

    with torch.no_grad():
        for imgs, d_lbl, s_lbl, dom_lbl in tqdm(loader, desc=desc, leave=False):
            imgs    = imgs.to(device, non_blocking=True)
            d_lbl   = d_lbl.to(device, non_blocking=True)
            s_lbl   = s_lbl.to(device, non_blocking=True)
            dom_lbl = dom_lbl.to(device, non_blocking=True)

            with autocast('cuda' if device.type == 'cuda' else 'cpu'):
                d_out, s_out, dom_out = model(imgs)
                ld   = criterion_d(d_out, d_lbl)
                ls   = criterion_s(s_out, s_lbl)
                ldom = criterion_dom(dom_out, dom_lbl)
                loss = ld + 0.2 * ls + cfg.DOMAIN_LOSS_WEIGHT * lambda_val * ldom

            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()
                total_disease_loss += ld.item()
                total_domain_loss += ldom.item()

            probs = torch.softmax(d_out.float(), dim=1)
            all_preds.extend(d_out.argmax(1).cpu().numpy())
            all_targets.extend(d_lbl.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_dom_preds.extend(dom_out.argmax(1).cpu().numpy())
            all_dom_targets.extend(dom_lbl.cpu().numpy())

    n_batches = max(len(loader), 1)
    return {
        'loss':         total_loss / n_batches,
        'disease_loss': total_disease_loss / n_batches,
        'domain_loss':  total_domain_loss / n_batches,
        'preds':        np.array(all_preds),
        'targets':      np.array(all_targets),
        'probs':        np.array(all_probs),
        'dom_preds':    np.array(all_dom_preds),
        'dom_targets':  np.array(all_dom_targets),
    }


print('\n[7/7] Training with domain adversarial objective...')

CHECKPOINT = os.path.join(cfg.OUTPUT_DIR, 'best_model.pth')

history = {k: [] for k in [
    'train_loss', 'val_loss',
    'train_disease_loss', 'val_disease_loss',
    'train_domain_loss', 'val_domain_loss',
    'train_acc', 'val_acc',
    'train_domain_acc', 'val_domain_acc',
    'macro_f1', 'weighted_f1', 'lr', 'lambda_p',
    *(f'f1_{c}' for c in cfg.CLASS_NAMES)
]}

best_f1      = 0.0
patience_ctr = 0
total_steps  = cfg.NUM_EPOCHS * len(train_loader)

t_start = time.time()
print('=' * 75)

for epoch in range(cfg.NUM_EPOCHS):
    t0 = time.time()

    # ---- TRAIN ----
    model.train()
    run_loss = 0.0
    run_disease_loss = 0.0
    run_domain_loss = 0.0
    correct  = 0
    total    = 0
    dom_correct = 0
    dom_total   = 0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(train_loader,
                desc=f'E{epoch+1:03d}/{cfg.NUM_EPOCHS} train',
                leave=False)

    for step, (imgs, d_lbl, s_lbl, dom_lbl) in enumerate(pbar):
        # Compute training progress and lambda schedule
        global_step = epoch * len(train_loader) + step
        progress = global_step / max(total_steps, 1)
        lambda_p = compute_lambda_p(progress)
        model.grl.set_lambda(lambda_p)

        imgs    = imgs.to(cfg.DEVICE, non_blocking=True)
        d_lbl   = d_lbl.to(cfg.DEVICE, non_blocking=True)
        s_lbl   = s_lbl.to(cfg.DEVICE, non_blocking=True)
        dom_lbl = dom_lbl.to(cfg.DEVICE, non_blocking=True)

        # MixUp augmentation (train only)
        mixed_imgs, y_a, y_b, dom_a, dom_b, lam, perm_idx = mixup_data(
            imgs, d_lbl, dom_lbl, alpha=cfg.MIXUP_ALPHA
        )

        with autocast('cuda' if cfg.DEVICE.type == 'cuda' else 'cpu'):
            d_out, s_out, dom_out = model(mixed_imgs)

            # Mixed Focal Loss for disease: lam * L(y_a) + (1-lam) * L(y_b)
            loss_d = lam * criterion_disease(d_out, y_a) + (1 - lam) * criterion_disease(d_out, y_b)
            loss_s = criterion_severity(s_out, s_lbl)

            # Domain loss: use un-mixed domain labels (original batch order)
            # Since MixUp blends images, the domain label is ambiguous for
            # cross-domain mixes. We use the primary (lam-weighted) domain label.
            loss_dom = criterion_domain(dom_out, dom_lbl)

            # Combined DANN loss
            loss = (loss_d + 0.2 * loss_s + cfg.DOMAIN_LOSS_WEIGHT * lambda_p * loss_dom) / cfg.GRADIENT_ACCUMULATION

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

        batch_loss = loss.item() * cfg.GRADIENT_ACCUMULATION
        run_loss += batch_loss
        run_disease_loss += loss_d.item()
        run_domain_loss += loss_dom.item()

        # Use un-mixed predictions for accuracy tracking
        with torch.no_grad():
            preds    = d_out.argmax(1)
            correct += (preds == y_a).sum().item()
            total   += d_lbl.size(0)
            dom_preds = dom_out.argmax(1)
            dom_correct += (dom_preds == dom_lbl).sum().item()
            dom_total   += dom_lbl.size(0)

        pbar.set_postfix(
            loss=f'{batch_loss:.3f}',
            acc=f'{100 * correct / total:.1f}%',
            dom=f'{100 * dom_correct / dom_total:.1f}%',
            lam=f'{lambda_p:.3f}'
        )

    # Flush remaining gradients for incomplete accumulation window
    if len(train_loader) % cfg.GRADIENT_ACCUMULATION != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    n_train = max(len(train_loader), 1)
    train_loss        = run_loss / n_train
    train_disease_loss = run_disease_loss / n_train
    train_domain_loss = run_domain_loss / n_train
    train_acc         = 100 * correct / max(total, 1)
    train_domain_acc  = 100 * dom_correct / max(dom_total, 1)

    # ---- VALIDATE on calibration set ----
    val_results = evaluate(
        calib_loader, model, criterion_disease, criterion_severity, criterion_domain,
        cfg.DEVICE, lambda_val=lambda_p,
        desc=f'E{epoch+1:03d}/{cfg.NUM_EPOCHS} calib'
    )

    val_loss       = val_results['loss']
    val_preds      = val_results['preds']
    val_targets    = val_results['targets']
    val_dom_preds  = val_results['dom_preds']
    val_dom_targets = val_results['dom_targets']

    val_acc = 100 * (val_preds == val_targets).mean()
    val_domain_acc = 100 * (val_dom_preds == val_dom_targets).mean()
    mf1     = f1_score(val_targets, val_preds, average='macro')
    wf1     = f1_score(val_targets, val_preds, average='weighted')
    per_f1  = f1_score(val_targets, val_preds,
                       average=None, labels=range(cfg.NUM_DISEASE_CLASSES),
                       zero_division=0)

    lr_now = optimizer.param_groups[0]['lr']

    # Record history
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_disease_loss'].append(train_disease_loss)
    history['val_disease_loss'].append(val_results['disease_loss'])
    history['train_domain_loss'].append(train_domain_loss)
    history['val_domain_loss'].append(val_results['domain_loss'])
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    history['train_domain_acc'].append(train_domain_acc)
    history['val_domain_acc'].append(val_domain_acc)
    history['macro_f1'].append(mf1)
    history['weighted_f1'].append(wf1)
    history['lr'].append(lr_now)
    history['lambda_p'].append(lambda_p)
    for ci, cn in enumerate(cfg.CLASS_NAMES):
        history[f'f1_{cn}'].append(float(per_f1[ci]))

    elapsed = time.time() - t0

    # ---- Early stopping on macro-F1 (with min_delta) ----
    tag = ''
    if mf1 > best_f1 + cfg.MIN_DELTA:
        best_f1      = mf1
        patience_ctr = 0
        torch.save({
            'epoch':            epoch,
            'model_state_dict': model.state_dict(),
            'val_acc':          val_acc,
            'macro_f1':         mf1,
            'lambda_p':         lambda_p,
            'history':          history,
        }, CHECKPOINT)
        tag = f'  * NEW BEST (macro-F1={mf1:.4f})'
    else:
        patience_ctr += 1

    cls_str = ' | '.join(
        f'{cn[:3]}:{per_f1[ci]:.2f}'
        for ci, cn in enumerate(cfg.CLASS_NAMES)
    )
    print(
        f'E{epoch+1:03d} | {elapsed:.0f}s | LR {lr_now:.2e} | lam {lambda_p:.3f} | '
        f'TrL {train_loss:.3f} DL {train_disease_loss:.3f} DomL {train_domain_loss:.3f} | '
        f'TrA {train_acc:.1f}% DomA {train_domain_acc:.1f}% | '
        f'VL {val_loss:.3f} VA {val_acc:.1f}% VDomA {val_domain_acc:.1f}% | '
        f'mF1 {mf1:.4f} wF1 {wf1:.4f}{tag}'
    )
    print(f'       {cls_str}')

    if patience_ctr >= cfg.PATIENCE:
        print(f'\n  Early stopping -- no improvement for {cfg.PATIENCE} epochs')
        break

total_train_time = time.time() - t_start
print(f'\nTraining complete. Best macro-F1: {best_f1:.4f}')
print(f'Total training time: {total_train_time / 60:.1f} minutes')

# Save training history
with open(os.path.join(cfg.OUTPUT_DIR, 'history.json'), 'w') as f:
    json.dump({k: [float(v) for v in vs] for k, vs in history.items()}, f, indent=2)


# ================================================================
# FINAL EVALUATION ON TEST SET
# ================================================================
print('\n' + '=' * 65)
print('         FINAL EVALUATION -- TEST SET')
print('=' * 65)
print('  (Test set was never seen during training or threshold tuning)')

# Reload best model
if os.path.exists(CHECKPOINT):
    ckpt = torch.load(CHECKPOINT, map_location=cfg.DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f'  Loaded best checkpoint (epoch {ckpt["epoch"]+1}, '
          f'macro-F1={ckpt["macro_f1"]:.4f})')

test_results = evaluate(
    test_loader, model, criterion_disease, criterion_severity, criterion_domain,
    cfg.DEVICE, lambda_val=1.0, desc='Final test'
)

test_preds   = test_results['preds']
test_targets = test_results['targets']
test_probs   = test_results['probs']

test_acc = 100 * (test_preds == test_targets).mean()
test_mf1 = f1_score(test_targets, test_preds, average='macro')
test_wf1 = f1_score(test_targets, test_preds, average='weighted')
test_per_f1 = f1_score(test_targets, test_preds, average=None,
                        labels=range(cfg.NUM_DISEASE_CLASSES), zero_division=0)

try:
    test_mauc = roc_auc_score(test_targets, test_probs, multi_class='ovr', average='macro')
except Exception:
    test_mauc = 0.0

# Domain accuracy on test set (lower = better domain invariance)
dom_preds   = test_results['dom_preds']
dom_targets = test_results['dom_targets']
test_dom_acc = 100 * (dom_preds == dom_targets).mean()

print(f'\n  Test Accuracy      : {test_acc:.2f}%')
print(f'  Test Macro F1      : {test_mf1:.4f}')
print(f'  Test Weighted F1   : {test_wf1:.4f}')
print(f'  Test Macro AUC     : {test_mauc:.4f}')
print(f'  Test Domain Acc    : {test_dom_acc:.1f}%  (closer to 50% = better invariance)')
print()
print(classification_report(test_targets, test_preds,
                            target_names=cfg.CLASS_NAMES, digits=4))

# Per-domain disease performance
print('  Per-domain disease performance:')
for d_idx, d_name in enumerate(cfg.DOMAIN_NAMES):
    mask = dom_targets == d_idx
    if mask.sum() == 0:
        continue
    d_f1 = f1_score(test_targets[mask], test_preds[mask], average='macro', zero_division=0)
    d_acc = 100 * (test_preds[mask] == test_targets[mask]).mean()
    print(f'    {d_name:6s}: acc={d_acc:.1f}%, macro-F1={d_f1:.4f}  (n={mask.sum()})')

# Save final metrics
final_metrics = {
    'accuracy':       float(test_acc),
    'macro_f1':       float(test_mf1),
    'weighted_f1':    float(test_wf1),
    'macro_auc':      float(test_mauc),
    'domain_acc':     float(test_dom_acc),
    'per_class_f1':   {cfg.CLASS_NAMES[i]: float(test_per_f1[i])
                       for i in range(cfg.NUM_DISEASE_CLASSES)},
    'training_time_min': float(total_train_time / 60),
    'best_epoch':     int(ckpt.get('epoch', -1) + 1) if os.path.exists(CHECKPOINT) else -1,
}
metrics_path = os.path.join(cfg.OUTPUT_DIR, 'final_metrics.json')
with open(metrics_path, 'w') as f:
    json.dump(final_metrics, f, indent=2)
print(f'\n  Metrics saved -> {metrics_path}')


# ================================================================
# PLOTS
# ================================================================
print('\nGenerating plots...')

ep     = range(1, len(history['train_loss']) + 1)
colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']

fig, axes = plt.subplots(3, 3, figsize=(20, 16))

# 1. Total Loss
axes[0, 0].plot(ep, history['train_loss'], 'b-o', ms=3, label='Train')
axes[0, 0].plot(ep, history['val_loss'],   'r-o', ms=3, label='Calib')
axes[0, 0].set_title('Total Loss', fontweight='bold')
axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

# 2. Disease Loss
axes[0, 1].plot(ep, history['train_disease_loss'], 'b-o', ms=3, label='Train')
axes[0, 1].plot(ep, history['val_disease_loss'],   'r-o', ms=3, label='Calib')
axes[0, 1].set_title('Disease Loss', fontweight='bold')
axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

# 3. Domain Loss
axes[0, 2].plot(ep, history['train_domain_loss'], 'b-o', ms=3, label='Train')
axes[0, 2].plot(ep, history['val_domain_loss'],   'r-o', ms=3, label='Calib')
axes[0, 2].set_title('Domain Loss', fontweight='bold')
axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

# 4. Disease Accuracy
axes[1, 0].plot(ep, history['train_acc'], 'b-o', ms=3, label='Train')
axes[1, 0].plot(ep, history['val_acc'],   'r-o', ms=3, label='Calib')
axes[1, 0].set_title('Disease Accuracy (%)', fontweight='bold')
axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

# 5. Domain Accuracy (closer to 50% = better invariance)
axes[1, 1].plot(ep, history['train_domain_acc'], 'b-o', ms=3, label='Train')
axes[1, 1].plot(ep, history['val_domain_acc'],   'r-o', ms=3, label='Calib')
axes[1, 1].axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Chance level')
axes[1, 1].set_title('Domain Accuracy (% -- 50%=invariant)', fontweight='bold')
axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

# 6. F1 Scores
axes[1, 2].plot(ep, history['macro_f1'],    'g-o', ms=3, label='Macro F1')
axes[1, 2].plot(ep, history['weighted_f1'], 'm-o', ms=3, label='Weighted F1')
axes[1, 2].set_title('F1 Scores (calib)', fontweight='bold')
axes[1, 2].legend(); axes[1, 2].grid(alpha=0.3)

# 7. Per-class F1
for ci, cn in enumerate(cfg.CLASS_NAMES):
    axes[2, 0].plot(ep, history[f'f1_{cn}'], '-o', ms=2,
                    color=colors[ci], label=cn)
axes[2, 0].set_title('Per-Class F1 (calib)', fontweight='bold')
axes[2, 0].legend(fontsize=8); axes[2, 0].grid(alpha=0.3)

# 8. Lambda schedule
axes[2, 1].plot(ep, history['lambda_p'], 'k-o', ms=3)
axes[2, 1].set_title('DANN Lambda Schedule', fontweight='bold')
axes[2, 1].set_xlabel('Epoch'); axes[2, 1].set_ylabel('Lambda')
axes[2, 1].grid(alpha=0.3)

# 9. Learning Rate
axes[2, 2].plot(ep, history['lr'], 'b-o', ms=2)
axes[2, 2].set_title('Learning Rate (head param group)', fontweight='bold')
axes[2, 2].set_xlabel('Epoch'); axes[2, 2].set_ylabel('LR')
axes[2, 2].grid(alpha=0.3)

plt.suptitle(
    f'RetinaSense DANN -- Macro F1={test_mf1:.3f} | '
    f'AUC={test_mauc:.3f} | '
    f'Test Acc={test_acc:.1f}% | DomAcc={test_dom_acc:.1f}%',
    fontsize=14, fontweight='bold', y=1.01
)
plt.tight_layout()
plt.savefig(os.path.join(cfg.OUTPUT_DIR, 'dann_dashboard.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f'  Dashboard saved -> {os.path.join(cfg.OUTPUT_DIR, "dann_dashboard.png")}')


# ================================================================
# SUMMARY
# ================================================================
print('\n' + '=' * 65)
print('       RETINASENSE DANN -- FINAL SUMMARY')
print('=' * 65)
print(f'  Training epochs       : {len(history["train_loss"])}')
print(f'  Best calib macro-F1   : {best_f1:.4f}')
print(f'  Test macro-F1         : {test_mf1:.4f}')
print(f'  Test accuracy         : {test_acc:.2f}%')
print(f'  Test domain accuracy  : {test_dom_acc:.1f}%  (50% = perfect invariance)')
print(f'  Final lambda          : {history["lambda_p"][-1]:.4f}')
print(f'  Training time         : {total_train_time / 60:.1f} minutes')
print(f'  Checkpoint            : {CHECKPOINT}')
print(f'  Metrics               : {metrics_path}')
print()
print('  Key insight: lower domain accuracy indicates the backbone learned')
print('  domain-invariant features, reducing the APTOS<->ODIR domain shift.')
print('=' * 65)
