#!/usr/bin/env python3
"""
RetinaSense DANN v3 — Pushing Toward 90% Accuracy
====================================================
Builds on train_dann.py (DANN-v2 achieved 86.1%) with seven key improvements:

1. **Expanded dataset**: MESSIDOR-2 as 4th domain (8,241 train samples)
2. **Hard-example mining**: Oversample top-K hardest examples by 2x each epoch
3. **Class-balanced batch sampling**: WeightedRandomSampler with inverse-frequency
4. **Progressive DR alpha boost**: 1.5x -> 3.0x across training
5. **Cosine annealing with warm restarts**: T_0=10, T_mult=2
6. **Label smoothing**: 0.1 on disease loss
7. **Mixup augmentation**: alpha=0.2 on 50% of batches
8. **Test-Time Augmentation (TTA)**: 8-way (flips + rotations) for final eval

Architecture is identical to DANNMultiTaskViT from train_dann.py.
Warm-starts from DANN-v2 checkpoint (outputs_v3/dann_v2/best_model.pth).

Usage:
  python train_dann_v3.py
  python train_dann_v3.py --epochs 50 --lr 2e-5 --tta
  python train_dann_v3.py --no-warmstart --epochs 60
"""

import os
import sys
import time
import json
import copy
import math
import argparse
import warnings
import random
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from collections import Counter, defaultdict

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Sampler
from torchvision import transforms

import timm

from scipy.optimize import minimize_scalar
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, roc_curve, auc,
)
from sklearn.preprocessing import label_binarize


# ================================================================
# CLI ARGUMENTS
# ================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description='RetinaSense DANN v3 — Improved Training for 90% Accuracy',
    )
    parser.add_argument(
        '--warmstart', type=str, default='outputs_v3/dann_v2/best_model.pth',
        help='Path to warm-start checkpoint (default: outputs_v3/dann_v2/best_model.pth)',
    )
    parser.add_argument('--no-warmstart', action='store_true', help='Skip warm-start')
    parser.add_argument('--epochs', type=int, default=40, help='Training epochs (default: 40)')
    parser.add_argument('--lr', type=float, default=3e-5, help='Base learning rate (default: 3e-5)')
    parser.add_argument(
        '--domain-weight', type=float, default=0.05,
        help='Weight multiplier for domain loss (default: 0.05)',
    )
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--workers', type=int, default=8, help='DataLoader workers')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Override output directory (default: outputs_v3/dann_v3)')

    # -- v3-specific improvements --
    parser.add_argument('--dr-alpha-start', type=float, default=1.5,
                        help='DR focal alpha boost at epoch 0 (default: 1.5)')
    parser.add_argument('--dr-alpha-end', type=float, default=3.0,
                        help='DR focal alpha boost at final epoch (default: 3.0)')
    parser.add_argument('--hard-mining-k', type=int, default=500,
                        help='Number of hard examples to oversample each epoch (default: 500)')
    parser.add_argument('--hard-mining-factor', type=int, default=2,
                        help='Oversampling factor for hard examples (default: 2)')
    parser.add_argument('--mixup-alpha', type=float, default=0.2,
                        help='Mixup alpha parameter (default: 0.2)')
    parser.add_argument('--mixup-prob', type=float, default=0.5,
                        help='Probability of applying mixup per batch (default: 0.5)')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                        help='Label smoothing factor (default: 0.1)')
    parser.add_argument('--cosine-t0', type=int, default=10,
                        help='CosineAnnealingWarmRestarts T_0 (default: 10)')
    parser.add_argument('--cosine-tmult', type=int, default=2,
                        help='CosineAnnealingWarmRestarts T_mult (default: 2)')
    parser.add_argument('--tta', action='store_true',
                        help='Enable Test-Time Augmentation for final evaluation (8-way)')
    parser.add_argument('--tta-n', type=int, default=8,
                        help='Number of TTA augmentations (default: 8)')
    parser.add_argument('--max-lambda', type=float, default=0.3,
                        help='Cap for Ganin lambda schedule (default: 0.3)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    return parser.parse_args()


# ================================================================
# CONFIG
# ================================================================
class Config:
    DATA_DIR   = './data'
    OUTPUT_DIR = './outputs_v3/dann_v3'

    MODEL_NAME = 'vit_base_patch16_224'
    IMG_SIZE   = 224
    FEAT_DIM   = 768

    NUM_DISEASE_CLASSES  = 5
    NUM_SEVERITY_CLASSES = 5

    DROPOUT = 0.3

    BATCH_SIZE  = 32
    NUM_EPOCHS  = 40
    NUM_WORKERS = 8

    BASE_LR      = 3e-5
    LLRD_DECAY   = 0.85
    WEIGHT_DECAY = 1e-4

    GRADIENT_ACCUMULATION = 2  # effective batch = 64

    FOCAL_GAMMA    = 2.0
    DOMAIN_WEIGHT  = 0.05

    PATIENCE  = 15
    MIN_DELTA = 0.001

    CLASS_NAMES = ['Normal', 'Diabetes/DR', 'Glaucoma', 'Cataract', 'AMD']

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Paths for expanded splits
    TRAIN_CSV = './data/train_split_expanded.csv'
    CALIB_CSV = './data/calib_split_expanded.csv'
    TEST_CSV  = './data/test_split.csv'  # sealed — never expanded

    # Fallback splits if expanded not available
    TRAIN_CSV_FALLBACK = './data/train_split.csv'
    CALIB_CSV_FALLBACK = './data/calib_split.csv'

    # ImageNet fallback normalisation
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]


# ================================================================
# REPRODUCIBILITY
# ================================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ================================================================
# GRADIENT REVERSAL LAYER (Ganin et al. 2016)
# ================================================================
class GradientReversalFunction(torch.autograd.Function):
    """Reverses gradients during backward pass, passes through during forward."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GRL(nn.Module):
    """Gradient Reversal Layer wrapper module."""

    def __init__(self):
        super().__init__()

    def forward(self, x, alpha=1.0):
        return GradientReversalFunction.apply(x, alpha)


# ================================================================
# GANIN LAMBDA SCHEDULE
# ================================================================
def ganin_lambda(epoch, total_epochs, max_lambda=0.3):
    """
    Progressive lambda schedule from Ganin et al. (2016), capped.
    Starts near 0, ramps with sigmoid curve, capped at max_lambda
    to prevent backbone destabilisation at high adversarial strength.
    """
    p = epoch / total_epochs
    raw = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0
    return min(raw, max_lambda)


# ================================================================
# DANN MULTI-TASK VIT MODEL
# ================================================================
class DANNMultiTaskViT(nn.Module):
    """
    ViT backbone with three heads:
      - disease_head  : 5-class fundus disease classification
      - severity_head : 5-class DR severity grading
      - domain_head   : N-class domain discriminator (receives GRL features)

    Feature dim is detected dynamically from the backbone (768 for ViT-Base).
    """

    def __init__(self, n_disease=5, n_severity=5, num_domains=4,
                 drop=0.3, backbone_name='vit_base_patch16_224'):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, num_classes=0,
        )
        feat = self.backbone.num_features  # 768 for ViT-Base

        self.drop = nn.Dropout(drop)

        # Disease classification head (same architecture as MultiTaskViT)
        self.disease_head = nn.Sequential(
            nn.Linear(feat, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, n_disease),
        )

        # Severity grading head (same architecture as MultiTaskViT)
        self.severity_head = nn.Sequential(
            nn.Linear(feat, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, n_severity),
        )

        # Domain discriminator head (receives gradient-reversed features)
        self.grl = GRL()
        self.domain_head = nn.Sequential(
            nn.Linear(feat, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, num_domains),
        )

    def forward(self, x, alpha=1.0):
        f = self.backbone(x)   # (B, feat_dim)
        f = self.drop(f)

        disease_out  = self.disease_head(f)
        severity_out = self.severity_head(f)

        # Domain head gets gradient-reversed features
        domain_out = self.domain_head(self.grl(f, alpha))

        return disease_out, severity_out, domain_out

    def forward_no_domain(self, x):
        """Forward without domain head -- for inference."""
        f = self.backbone(x)
        f = self.drop(f)
        return self.disease_head(f), self.severity_head(f)

    def forward_features(self, x):
        """Return backbone features only (for hard-example mining)."""
        f = self.backbone(x)
        return f


# ================================================================
# FOCAL LOSS WITH LABEL SMOOTHING
# ================================================================
class FocalLoss(nn.Module):
    """
    Focal Loss with optional label smoothing.
    alpha: per-class weight tensor; gamma: focusing parameter.
    label_smoothing: smoothing factor (0 = no smoothing).
    """

    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits, targets, reduction='none',
            label_smoothing=self.label_smoothing,
        )
        pt    = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            at    = self.alpha.gather(0, targets)
            focal = at * focal
        return focal.mean()


# ================================================================
# NORMALISATION STATS
# ================================================================
def load_norm_stats():
    """Load fundus normalisation stats with fallback chain."""
    candidates = [
        'configs/fundus_norm_stats_unified.json',
        'configs/fundus_norm_stats.json',
        'data/fundus_norm_stats.json',
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path) as f:
                stats = json.load(f)
            mean = stats['mean_rgb']
            std  = stats['std_rgb']
            print(f'  Loaded norm stats from {path}')
            print(f'    mean={mean}, std={std}')
            return mean, std

    # ImageNet fallback
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    print('  No fundus norm stats found -- using ImageNet defaults')
    return mean, std


# ================================================================
# CACHE PATH RESOLUTION
# ================================================================
def resolve_cache_path(cache_path_from_csv):
    """
    Resolve cache path: try unified cache first, then v3 cache.
    The CSVs store paths like ./preprocessed_cache_v3/xxx.npy
    """
    # Try unified cache first
    unified_path = cache_path_from_csv.replace(
        'preprocessed_cache_v3', 'preprocessed_cache_unified'
    )
    if os.path.exists(unified_path):
        return unified_path
    if os.path.exists(cache_path_from_csv):
        return cache_path_from_csv
    return cache_path_from_csv  # will fail at load time, handled by fallback


# ================================================================
# IMAGE PREPROCESSING (fallback for missing cache)
# ================================================================
def _read_rgb(path):
    img = cv2.imread(path)
    if img is None:
        img = np.array(Image.open(path).convert('RGB'))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _circular_mask(img, sz):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (sz // 2, sz // 2), int(sz * 0.48), 255, -1)
    return cv2.bitwise_and(img, img, mask=mask)


def clahe_preprocess(path, sz=224):
    img = cv2.resize(_read_rgb(path), (sz, sz))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return _circular_mask(img, sz)


def preprocess_image(path, source='ODIR', sz=224):
    """Unified CLAHE preprocessing for all sources."""
    return clahe_preprocess(path, sz)


# ================================================================
# DATASET WITH INDEX TRACKING (for hard-example mining)
# ================================================================
class RetinalDANNv3Dataset(Dataset):
    """
    Retinal fundus dataset with domain labels for DANN training.
    Returns: (image_tensor, disease_label, severity_label, domain_label, index)

    The index is returned so the training loop can track per-sample losses
    for hard-example mining.
    """

    def __init__(self, df, transform, domain_map):
        self.df        = df.reset_index(drop=True)
        self.transform = transform
        self.domain_map = domain_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load from cache
        cache_fp = resolve_cache_path(str(row.get('cache_path', '')))
        try:
            img = np.load(cache_fp)
        except Exception:
            try:
                img = preprocess_image(
                    row['image_path'], row.get('source', 'ODIR'),
                )
            except Exception:
                img = np.zeros((224, 224, 3), dtype=np.uint8)

        img_tensor = self.transform(img)

        disease_lbl  = int(row['disease_label'])
        severity_lbl = int(row['severity_label'])
        if severity_lbl < 0:
            severity_lbl = 0

        # Domain label from source column
        source = str(row.get('source', 'UNKNOWN')).upper()
        domain_lbl = self.domain_map.get(source, 0)

        return (
            img_tensor,
            torch.tensor(disease_lbl,  dtype=torch.long),
            torch.tensor(severity_lbl, dtype=torch.long),
            torch.tensor(domain_lbl,   dtype=torch.long),
            torch.tensor(idx,          dtype=torch.long),
        )


# ================================================================
# TRANSFORMS
# ================================================================
def make_transforms(phase, norm_mean, norm_std):
    normalize = transforms.Normalize(norm_mean, norm_std)
    if phase == 'train':
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(20),
            transforms.RandomAffine(
                degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05),
            ),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02,
            ),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.2),
        ])
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        normalize,
    ])


def make_tta_transforms(norm_mean, norm_std, n_augs=8):
    """
    Build a list of deterministic TTA transforms.
    Covers: identity, H-flip, V-flip, HV-flip, 90/180/270 rotations, slight zoom.
    """
    normalize = transforms.Normalize(norm_mean, norm_std)

    base = [transforms.ToPILImage()]

    tta_list = []

    # 1. Identity
    tta_list.append(transforms.Compose([
        *base, transforms.ToTensor(), normalize,
    ]))

    # 2. Horizontal flip
    tta_list.append(transforms.Compose([
        *base, transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(), normalize,
    ]))

    # 3. Vertical flip
    tta_list.append(transforms.Compose([
        *base, transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(), normalize,
    ]))

    # 4. H + V flip
    tta_list.append(transforms.Compose([
        *base,
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(), normalize,
    ]))

    # 5. 90-degree rotation
    tta_list.append(transforms.Compose([
        *base,
        transforms.Lambda(lambda img: img.rotate(90, expand=False)),
        transforms.ToTensor(), normalize,
    ]))

    # 6. 180-degree rotation
    tta_list.append(transforms.Compose([
        *base,
        transforms.Lambda(lambda img: img.rotate(180, expand=False)),
        transforms.ToTensor(), normalize,
    ]))

    # 7. 270-degree rotation
    tta_list.append(transforms.Compose([
        *base,
        transforms.Lambda(lambda img: img.rotate(270, expand=False)),
        transforms.ToTensor(), normalize,
    ]))

    # 8. Center crop + resize (slight zoom)
    tta_list.append(transforms.Compose([
        *base,
        transforms.CenterCrop(200),
        transforms.Resize((224, 224)),
        transforms.ToTensor(), normalize,
    ]))

    return tta_list[:n_augs]


# ================================================================
# HARD-EXAMPLE MINING SAMPLER
# ================================================================
class HardExampleMiningWeightedSampler(Sampler):
    """
    A weighted sampler that combines class-balancing weights with
    hard-example boosting. After each epoch, the training loop
    updates per-sample losses; the top-K hardest examples get their
    sample weight boosted by `hard_factor`.

    Also specifically boosts DR->Normal misclassifications (class 1
    predicted as class 0) which were the dominant error mode.
    """

    def __init__(self, labels, n_classes=5, hard_k=500, hard_factor=2):
        self.labels      = np.array(labels)
        self.n_classes   = n_classes
        self.n_samples   = len(labels)
        self.hard_k      = hard_k
        self.hard_factor = hard_factor

        # Base class-balancing weights (inverse frequency)
        class_cnt = np.bincount(self.labels, minlength=n_classes).astype(float)
        class_cnt = np.where(class_cnt == 0, 1.0, class_cnt)
        self.base_weights = 1.0 / class_cnt[self.labels]

        # Per-sample loss tracking (updated each epoch)
        self.sample_losses = np.zeros(self.n_samples, dtype=np.float64)

        # Misclassification tracking
        self.sample_wrong  = np.zeros(self.n_samples, dtype=bool)
        self.sample_pred   = np.full(self.n_samples, -1, dtype=np.int64)

        # Current effective weights
        self._update_weights()

    def _update_weights(self):
        """Recompute effective sampling weights from base + hard mining."""
        weights = self.base_weights.copy()

        if self.sample_losses.sum() > 0:
            # Boost top-K highest-loss samples
            top_k_idx = np.argsort(self.sample_losses)[-self.hard_k:]
            weights[top_k_idx] *= self.hard_factor

            # Extra boost for DR->Normal misclassifications
            dr_as_normal = (
                (self.labels == 1) &           # True label is DR
                self.sample_wrong &            # Was misclassified
                (self.sample_pred == 0)        # Predicted as Normal
            )
            n_dr_normal = dr_as_normal.sum()
            if n_dr_normal > 0:
                weights[dr_as_normal] *= self.hard_factor
                # Print once per update for monitoring
                pass

        self.effective_weights = torch.DoubleTensor(weights)

    def update_losses(self, indices, losses, preds=None, targets=None):
        """
        Update per-sample losses and misclassification info.
        Called after each epoch by the training loop.

        Args:
            indices: sample indices (from dataset)
            losses:  per-sample loss values
            preds:   predicted class per sample (optional)
            targets: true class per sample (optional)
        """
        idx_np = indices.cpu().numpy() if torch.is_tensor(indices) else np.array(indices)
        loss_np = losses.cpu().numpy() if torch.is_tensor(losses) else np.array(losses)

        for i, idx in enumerate(idx_np):
            if 0 <= idx < self.n_samples:
                self.sample_losses[idx] = loss_np[i]

        if preds is not None and targets is not None:
            pred_np = preds.cpu().numpy() if torch.is_tensor(preds) else np.array(preds)
            tgt_np  = targets.cpu().numpy() if torch.is_tensor(targets) else np.array(targets)
            for i, idx in enumerate(idx_np):
                if 0 <= idx < self.n_samples:
                    self.sample_wrong[idx] = (pred_np[i] != tgt_np[i])
                    self.sample_pred[idx]  = pred_np[i]

        self._update_weights()

    def __iter__(self):
        return iter(torch.multinomial(
            self.effective_weights,
            num_samples=self.n_samples,
            replacement=True,
        ).tolist())

    def __len__(self):
        return self.n_samples

    def get_stats(self):
        """Return mining stats for logging."""
        n_hard = min(self.hard_k, (self.sample_losses > 0).sum())
        n_wrong = self.sample_wrong.sum()
        dr_as_normal = (
            (self.labels == 1) &
            self.sample_wrong &
            (self.sample_pred == 0)
        ).sum()
        return {
            'hard_samples_tracked': int(n_hard),
            'wrong_total': int(n_wrong),
            'dr_as_normal': int(dr_as_normal),
        }


# ================================================================
# LLRD OPTIMIZER
# ================================================================
def get_optimizer_with_llrd(model, base_lr, decay_factor, weight_decay=1e-4):
    """
    AdamW with layer-wise learning rate decay (LLRD).
    """
    param_groups = []

    # 1. All heads at full LR
    head_params = (
        list(model.disease_head.parameters()) +
        list(model.severity_head.parameters()) +
        list(model.domain_head.parameters()) +
        list(model.drop.parameters())
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
        [model.backbone.cls_token, model.backbone.pos_embed] +
        list(model.backbone.norm.parameters())
    )
    param_groups.append({'params': embed_params, 'lr': embed_lr})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    lrs = [g['lr'] for g in param_groups]
    print(f'  LLRD optimizer: {len(param_groups)} param groups '
          f'({num_blocks} transformer blocks)')
    print(f'    Head LR    : {lrs[0]:.2e}')
    print(f'    Block[{num_blocks-1}]  : {lrs[1]:.2e}')
    print(f'    Block[0]   : {lrs[-2]:.2e}')
    print(f'    Embed LR   : {lrs[-1]:.2e}')

    return optimizer


# ================================================================
# MIXUP (disease labels only, NOT domain labels)
# ================================================================
def mixup_data(x, y_disease, alpha=0.2):
    """
    MixUp augmentation for disease labels.
    Domain labels are NOT mixed -- the discriminator needs clean domain targets.
    """
    lam        = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index      = torch.randperm(batch_size, device=x.device)
    mixed_x    = lam * x + (1 - lam) * x[index]
    return mixed_x, y_disease, y_disease[index], lam, index


# ================================================================
# WARM-START WEIGHT LOADING
# ================================================================
def load_warmstart_weights(model, checkpoint_path, device, num_domains_new=4):
    """
    Load weights from a DANN checkpoint into DANNMultiTaskViT.
    Handles domain head dimension mismatch (3 domains -> 4 domains).
    """
    if not os.path.exists(checkpoint_path):
        print(f'  WARNING: Warm-start checkpoint not found: {checkpoint_path}')
        print('  Proceeding with random initialisation for all heads.')
        return False

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)

    model_state = model.state_dict()
    loaded_keys = []
    skipped_keys = []

    for key, value in state_dict.items():
        if key in model_state and model_state[key].shape == value.shape:
            model_state[key] = value
            loaded_keys.append(key)
        else:
            skipped_keys.append(key)

    model.load_state_dict(model_state)

    # Report
    n_loaded  = len(loaded_keys)
    n_skipped = len(skipped_keys)
    n_new     = len(model_state) - n_loaded

    epoch_info = ckpt.get('epoch', '?')
    f1_info    = ckpt.get('macro_f1', '?')
    print(f'  Warm-start loaded from {checkpoint_path}')
    print(f'    Source: epoch {epoch_info}, macro-F1={f1_info}')
    print(f'    Loaded: {n_loaded} params | Skipped: {n_skipped} | '
          f'New (random): {n_new}')

    if skipped_keys:
        print(f'    Skipped keys (shape mismatch or missing):')
        for k in skipped_keys[:10]:
            print(f'      - {k}')
        if len(skipped_keys) > 10:
            print(f'      ... and {len(skipped_keys) - 10} more')

    return True


# ================================================================
# EVALUATION
# ================================================================
def evaluate(loader, model, criterion_d, criterion_s, device, desc='Eval'):
    """Run inference. Returns loss, preds, targets, probs."""
    model.eval()
    total_loss = 0.0
    all_preds, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            imgs, d_lbl, s_lbl = batch[0], batch[1], batch[2]
            imgs  = imgs.to(device, non_blocking=True)
            d_lbl = d_lbl.to(device, non_blocking=True)
            s_lbl = s_lbl.to(device, non_blocking=True)

            with autocast('cuda'):
                d_out, s_out = model.forward_no_domain(imgs)
                ld   = criterion_d(d_out, d_lbl)
                ls   = criterion_s(s_out, s_lbl)
                loss = ld + 0.2 * ls

            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()

            probs = torch.softmax(d_out.float(), dim=1)
            all_preds.extend(d_out.argmax(1).cpu().numpy())
            all_targets.extend(d_lbl.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / max(len(loader), 1)
    return (avg_loss,
            np.array(all_preds),
            np.array(all_targets),
            np.array(all_probs))


def evaluate_domain(loader, model, device, alpha=1.0):
    """Evaluate domain classification accuracy."""
    model.eval()
    correct = 0
    total   = 0

    with torch.no_grad():
        for batch in loader:
            imgs    = batch[0].to(device, non_blocking=True)
            dom_lbl = batch[3].to(device, non_blocking=True)

            with autocast('cuda'):
                _, _, dom_out = model(imgs, alpha=alpha)

            preds    = dom_out.argmax(1)
            correct += (preds == dom_lbl).sum().item()
            total   += dom_lbl.size(0)

    return 100.0 * correct / max(total, 1)


# ================================================================
# TEST-TIME AUGMENTATION (TTA)
# ================================================================
def evaluate_with_tta(dataset_df, domain_map, model, device,
                      norm_mean, norm_std, batch_size=32, num_workers=4,
                      n_augs=8, desc='TTA Eval'):
    """
    Run TTA evaluation: for each sample, run n_augs augmented versions
    and average the softmax probabilities.

    Returns: (probs, targets) -- averaged probs and true labels.
    """
    tta_transforms = make_tta_transforms(norm_mean, norm_std, n_augs=n_augs)

    model.eval()
    n_samples = len(dataset_df)
    n_classes = 5
    all_probs = np.zeros((n_samples, n_classes), dtype=np.float64)
    all_targets = np.zeros(n_samples, dtype=np.int64)

    print(f'  Running TTA with {len(tta_transforms)} augmentations...')

    for aug_idx, tta_tfm in enumerate(tta_transforms):
        # Build dataset with this specific transform
        ds = RetinalDANNv3Dataset(dataset_df, tta_tfm, domain_map)
        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

        batch_start = 0
        with torch.no_grad():
            for batch in tqdm(loader, desc=f'  TTA {aug_idx+1}/{len(tta_transforms)}',
                              leave=False):
                imgs  = batch[0].to(device, non_blocking=True)
                d_lbl = batch[1]

                with autocast('cuda'):
                    d_out, _ = model.forward_no_domain(imgs)

                probs = torch.softmax(d_out.float(), dim=1).cpu().numpy()
                bs = probs.shape[0]

                all_probs[batch_start:batch_start+bs] += probs
                if aug_idx == 0:
                    all_targets[batch_start:batch_start+bs] = d_lbl.numpy()
                batch_start += bs

        print(f'    Aug {aug_idx+1}/{len(tta_transforms)} done')

    # Average across augmentations
    all_probs /= len(tta_transforms)

    return all_probs, all_targets


# ================================================================
# ECE (Expected Calibration Error)
# ================================================================
def compute_ece(probs, labels, n_bins=15):
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


# ================================================================
# TEMPERATURE SCALING
# ================================================================
def temperature_scale(logits, labels):
    """Find optimal temperature T via NLL minimisation."""
    def nll_at_T(T):
        scaled = logits / T
        log_p  = F.log_softmax(scaled, dim=1)
        return F.nll_loss(log_p, labels).item()

    result = minimize_scalar(nll_at_T, bounds=(0.01, 10.0), method='bounded')
    return float(result.x)


# ================================================================
# THRESHOLD OPTIMISATION
# ================================================================
def optimise_thresholds(probs, labels, n_classes, class_names, n_grid=50):
    """Grid-search per-class decision thresholds on the calibration set."""
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
        print(f'    {class_names[c]:15s}: threshold={best_t:.3f}  '
              f'(calib F1={best_f1:.3f})')
    return thresholds


def apply_thresholds(probs, thresholds):
    """Apply per-class thresholds; fallback to argmax if none exceed."""
    preds = []
    for prob_row in probs:
        above = [
            i for i, (p, t) in enumerate(zip(prob_row, thresholds))
            if p >= t
        ]
        if above:
            preds.append(int(above[np.argmax([prob_row[i] for i in above])]))
        else:
            preds.append(int(np.argmax(prob_row)))
    return np.array(preds)


# ================================================================
# DASHBOARD PLOT
# ================================================================
def plot_dashboard(history, cfg, output_dir):
    """Generate training dashboard with loss, accuracy, domain_acc, F1."""
    ep = range(1, len(history['train_loss']) + 1)
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # 1. Loss (all components)
    axes[0, 0].plot(ep, history['train_loss'], 'b-o', ms=3, label='Train Total')
    axes[0, 0].plot(ep, history['val_loss'],   'r-o', ms=3, label='Val Disease+Sev')
    axes[0, 0].plot(ep, history['domain_loss'], 'g-o', ms=3, label='Domain Loss')
    axes[0, 0].set_title('Loss', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # 2. Accuracy
    axes[0, 1].plot(ep, history['train_acc'], 'b-o', ms=3, label='Train Acc')
    axes[0, 1].plot(ep, history['val_acc'],   'r-o', ms=3, label='Val Acc')
    axes[0, 1].set_title('Disease Accuracy (%)', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # 3. Domain accuracy (should converge toward chance = domain-invariant)
    axes[0, 2].plot(ep, history['train_domain_acc'], 'b-o', ms=3, label='Train')
    axes[0, 2].plot(ep, history['val_domain_acc'],   'r-o', ms=3, label='Val')
    num_domains = history.get('num_domains', 4)
    chance = 100.0 / num_domains
    axes[0, 2].axhline(y=chance, color='gray', linestyle='--', alpha=0.5,
                        label=f'Chance ({chance:.0f}%)')
    axes[0, 2].set_title('Domain Accuracy (lower = better alignment)',
                          fontweight='bold')
    axes[0, 2].legend()
    axes[0, 2].grid(alpha=0.3)

    # 4. F1 scores
    axes[1, 0].plot(ep, history['macro_f1'],    'g-o', ms=3, label='Macro F1')
    axes[1, 0].plot(ep, history['weighted_f1'], 'm-o', ms=3, label='Weighted F1')
    axes[1, 0].set_title('F1 Scores (val)', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # 5. Per-class F1
    for ci, cn in enumerate(cfg.CLASS_NAMES):
        key = f'f1_{cn}'
        if key in history:
            axes[1, 1].plot(ep, history[key], '-o', ms=2,
                            color=colors[ci], label=cn)
    axes[1, 1].set_title('Per-Class F1 (val)', fontweight='bold')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.3)

    # 6. Lambda + DR alpha schedule
    ax_lam = axes[1, 2]
    ax_lam.plot(ep, history['lambda_p'], 'k-o', ms=3, label='Lambda')
    ax_lam.set_title('Schedules', fontweight='bold')
    ax_lam.set_xlabel('Epoch')
    ax_lam.legend(loc='upper left')
    ax_lam.grid(alpha=0.3)
    if 'dr_alpha' in history:
        ax2 = ax_lam.twinx()
        ax2.plot(ep, history['dr_alpha'], 'r-s', ms=3, alpha=0.7, label='DR Alpha')
        ax2.set_ylabel('DR Alpha Boost', color='r')
        ax2.legend(loc='upper right')

    plt.suptitle(
        'RetinaSense DANN v3 Training Dashboard',
        fontsize=14, fontweight='bold', y=1.01,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dashboard.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Dashboard saved to {output_dir}/dashboard.png')


# ================================================================
# COLLECT LOGITS
# ================================================================
def collect_logits_labels(loader, model, device):
    """Collect raw logits and labels from a DataLoader."""
    all_logits, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc='Collecting logits', leave=False):
            imgs  = batch[0].to(device, non_blocking=True)
            d_lbl = batch[1]
            d_out, _ = model.forward_no_domain(imgs)
            all_logits.append(d_out.float().cpu())
            all_labels.append(d_lbl.cpu())
    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


# ================================================================
# PROGRESSIVE DR ALPHA
# ================================================================
def compute_dr_alpha_boost(epoch, total_epochs, start=1.5, end=3.0):
    """Linearly interpolate DR alpha boost from start to end across epochs."""
    if total_epochs <= 1:
        return end
    progress = epoch / (total_epochs - 1)
    return start + (end - start) * progress


# ================================================================
# MAIN
# ================================================================
def main():
    args = parse_args()
    cfg  = Config()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Override config from CLI
    cfg.NUM_EPOCHS    = args.epochs
    cfg.BASE_LR       = args.lr
    cfg.DOMAIN_WEIGHT = args.domain_weight
    cfg.BATCH_SIZE    = args.batch_size
    cfg.NUM_WORKERS   = args.workers

    # Override output dir if specified
    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs('configs', exist_ok=True)

    print('=' * 75)
    print('   RetinaSense DANN v3 -- Improved Training Pipeline (Target: 90%)')
    print('=' * 75)
    if torch.cuda.is_available():
        print(f'  GPU           : {torch.cuda.get_device_name(0)}')
        print(f'  VRAM          : '
              f'{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    print(f'  Epochs        : {cfg.NUM_EPOCHS}  (patience={cfg.PATIENCE})')
    print(f'  Batch         : {cfg.BATCH_SIZE} '
          f'(eff. {cfg.BATCH_SIZE * cfg.GRADIENT_ACCUMULATION} via grad accum)')
    print(f'  Base LR       : {cfg.BASE_LR:.1e}')
    print(f'  LLRD decay    : {cfg.LLRD_DECAY}')
    print(f'  Domain wt     : {cfg.DOMAIN_WEIGHT}')
    print(f'  Focal gamma   : {cfg.FOCAL_GAMMA}')
    print(f'  Label smooth  : {args.label_smoothing}')
    print(f'  Mixup alpha   : {args.mixup_alpha} (prob={args.mixup_prob})')
    print(f'  DR alpha      : {args.dr_alpha_start} -> {args.dr_alpha_end} (progressive)')
    print(f'  Hard mining   : top-{args.hard_mining_k} x{args.hard_mining_factor}')
    print(f'  Scheduler     : CosineWarmRestarts T0={args.cosine_t0} Tmult={args.cosine_tmult}')
    print(f'  Max lambda    : {args.max_lambda}')
    print(f'  TTA           : {"Enabled" if args.tta else "Disabled"} ({args.tta_n}-way)')
    print(f'  Warm-start    : {args.warmstart}')
    print(f'  Seed          : {args.seed}')
    print('=' * 75)

    # ----------------------------------------------------------
    # 1. Normalisation stats
    # ----------------------------------------------------------
    print('\n[1/10] Loading normalisation stats...')
    NORM_MEAN, NORM_STD = load_norm_stats()

    # ----------------------------------------------------------
    # 2. Load splits (prefer expanded, fall back to original)
    # ----------------------------------------------------------
    print('\n[2/10] Loading data splits...')

    # Train
    if os.path.exists(cfg.TRAIN_CSV):
        train_df = pd.read_csv(cfg.TRAIN_CSV)
        print(f'  Train: loaded {cfg.TRAIN_CSV} ({len(train_df)} samples)')
    elif os.path.exists(cfg.TRAIN_CSV_FALLBACK):
        train_df = pd.read_csv(cfg.TRAIN_CSV_FALLBACK)
        print(f'  Train: loaded FALLBACK {cfg.TRAIN_CSV_FALLBACK} ({len(train_df)} samples)')
    else:
        raise FileNotFoundError(
            f'No train CSV found at {cfg.TRAIN_CSV} or {cfg.TRAIN_CSV_FALLBACK}'
        )

    # Calib
    if os.path.exists(cfg.CALIB_CSV):
        calib_df = pd.read_csv(cfg.CALIB_CSV)
        print(f'  Calib: loaded {cfg.CALIB_CSV} ({len(calib_df)} samples)')
    elif os.path.exists(cfg.CALIB_CSV_FALLBACK):
        calib_df = pd.read_csv(cfg.CALIB_CSV_FALLBACK)
        print(f'  Calib: loaded FALLBACK {cfg.CALIB_CSV_FALLBACK} ({len(calib_df)} samples)')
    else:
        raise FileNotFoundError(
            f'No calib CSV found at {cfg.CALIB_CSV} or {cfg.CALIB_CSV_FALLBACK}'
        )

    # Test (always the sealed set)
    if not os.path.exists(cfg.TEST_CSV):
        raise FileNotFoundError(f'Test CSV not found: {cfg.TEST_CSV}')
    test_df = pd.read_csv(cfg.TEST_CSV)
    print(f'  Test : loaded {cfg.TEST_CSV} ({len(test_df)} samples) [SEALED]')

    # ----------------------------------------------------------
    # 3. Build domain mapping (4 domains for expanded dataset)
    # ----------------------------------------------------------
    print('\n[3/10] Building domain mapping...')

    # Fixed domain mapping: APTOS=0, ODIR=1, REFUGE2=2, MESSIDOR2=3
    domain_map = {
        'APTOS':     0,
        'ODIR':      1,
        'REFUGE2':   2,
        'MESSIDOR2': 3,
    }

    # Include any unexpected sources at the end
    all_sources = sorted(set(
        train_df['source'].str.upper().unique().tolist() +
        calib_df['source'].str.upper().unique().tolist() +
        test_df['source'].str.upper().unique().tolist()
    ))
    for src in all_sources:
        if src not in domain_map:
            domain_map[src] = len(domain_map)
            print(f'  WARNING: unexpected source "{src}" assigned domain {domain_map[src]}')

    num_domains = len(domain_map)

    print(f'  Domains ({num_domains}):')
    for src, idx in sorted(domain_map.items(), key=lambda x: x[1]):
        train_cnt = (train_df['source'].str.upper() == src).sum()
        calib_cnt = (calib_df['source'].str.upper() == src).sum()
        test_cnt  = (test_df['source'].str.upper() == src).sum()
        print(f'    {idx}: {src:12s}  train={train_cnt:5d}  '
              f'calib={calib_cnt:4d}  test={test_cnt:4d}')

    # ----------------------------------------------------------
    # 4. Class distribution analysis
    # ----------------------------------------------------------
    print('\n[4/10] Class distribution analysis...')
    train_labels = train_df['disease_label'].values
    class_counts = np.bincount(train_labels, minlength=5)
    total_train  = len(train_labels)

    for ci, cn in enumerate(cfg.CLASS_NAMES):
        pct = 100.0 * class_counts[ci] / total_train
        print(f'  {cn:15s}: {class_counts[ci]:5d} ({pct:5.1f}%)')

    # ----------------------------------------------------------
    # 5. Datasets and loaders
    # ----------------------------------------------------------
    print('\n[5/10] Building datasets and loaders...')

    train_transform = make_transforms('train', NORM_MEAN, NORM_STD)
    val_transform   = make_transforms('val',   NORM_MEAN, NORM_STD)

    train_ds = RetinalDANNv3Dataset(train_df, train_transform, domain_map)
    calib_ds = RetinalDANNv3Dataset(calib_df, val_transform,   domain_map)
    test_ds  = RetinalDANNv3Dataset(test_df,  val_transform,   domain_map)

    # Hard-example mining sampler with class balancing
    sampler = HardExampleMiningWeightedSampler(
        labels=train_df['disease_label'].values,
        n_classes=cfg.NUM_DISEASE_CLASSES,
        hard_k=args.hard_mining_k,
        hard_factor=args.hard_mining_factor,
    )

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

    print(f'  Train : {len(train_ds):5d}  ({len(train_loader):3d} batches)')
    print(f'  Calib : {len(calib_ds):5d}  ({len(calib_loader):3d} batches)')
    print(f'  Test  : {len(test_ds):5d}  ({len(test_loader):3d} batches)')

    # ----------------------------------------------------------
    # 6. Model
    # ----------------------------------------------------------
    print('\n[6/10] Building DANN v3 model...')

    model = DANNMultiTaskViT(
        n_disease=cfg.NUM_DISEASE_CLASSES,
        n_severity=cfg.NUM_SEVERITY_CLASSES,
        num_domains=num_domains,
        drop=cfg.DROPOUT,
        backbone_name=cfg.MODEL_NAME,
    ).to(cfg.DEVICE)

    # Warm-start from existing checkpoint
    if not args.no_warmstart:
        load_warmstart_weights(model, args.warmstart, cfg.DEVICE,
                               num_domains_new=num_domains)

    total_params = sum(p.numel() for p in model.parameters())
    domain_params = sum(p.numel() for p in model.domain_head.parameters())
    print(f'  Total params  : {total_params:,}')
    print(f'  Domain head   : {domain_params:,}')

    # ----------------------------------------------------------
    # 7. Loss functions and optimizer
    # ----------------------------------------------------------
    print('\n[7/10] Setting up losses and optimizer...')

    # Focal loss with class weights (DR alpha will be updated each epoch)
    cw = compute_class_weight(
        'balanced',
        classes=np.arange(cfg.NUM_DISEASE_CLASSES),
        y=train_df['disease_label'].values,
    )
    base_alpha = torch.tensor(cw, dtype=torch.float32).to(cfg.DEVICE)
    base_alpha = base_alpha / base_alpha.sum() * cfg.NUM_DISEASE_CLASSES
    print(f'  Base focal alpha: {[f"{a:.2f}" for a in base_alpha.tolist()]}')

    # Initial DR boost (will be updated progressively)
    initial_dr_boost = args.dr_alpha_start
    alpha_epoch0 = base_alpha.clone()
    alpha_epoch0[1] *= initial_dr_boost
    print(f'  Initial DR alpha boost: {initial_dr_boost:.1f}x -> '
          f'alpha[DR]={alpha_epoch0[1]:.2f}')

    criterion_d = FocalLoss(
        alpha=alpha_epoch0, gamma=cfg.FOCAL_GAMMA,
        label_smoothing=args.label_smoothing,
    )
    criterion_s = nn.CrossEntropyLoss(
        ignore_index=-1,
        label_smoothing=args.label_smoothing,
    )
    criterion_dom = nn.CrossEntropyLoss()

    # Per-sample loss (no reduction) for hard-example mining
    criterion_per_sample = nn.CrossEntropyLoss(reduction='none')

    optimizer = get_optimizer_with_llrd(
        model, base_lr=cfg.BASE_LR, decay_factor=cfg.LLRD_DECAY,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    # CosineAnnealingWarmRestarts scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.cosine_t0,
        T_mult=args.cosine_tmult,
        eta_min=cfg.BASE_LR * 0.01,  # min LR = 1% of base
    )

    scaler = GradScaler()

    # ----------------------------------------------------------
    # 8. Training loop
    # ----------------------------------------------------------
    print('\n[8/10] Training with DANN v3 improvements...')

    CHECKPOINT = os.path.join(cfg.OUTPUT_DIR, 'best_model.pth')

    history_keys = [
        'train_loss', 'val_loss', 'domain_loss',
        'train_acc', 'val_acc',
        'train_domain_acc', 'val_domain_acc',
        'macro_f1', 'weighted_f1', 'lr', 'lambda_p',
        'dr_alpha', 'hard_samples', 'dr_as_normal_errors',
    ] + [f'f1_{cn}' for cn in cfg.CLASS_NAMES]

    history = {k: [] for k in history_keys}
    history['num_domains'] = num_domains

    best_f1      = 0.0
    patience_ctr = 0
    t_start      = time.time()

    print('=' * 85)
    print(f'{"Ep":>3} | {"Time":>4} | {"LR":>8} | {"Lam":>5} {"DRa":>4} | '
          f'{"TrL":>6} {"TrA":>5} | {"VL":>6} {"VA":>5} | '
          f'{"mF1":>6} {"wF1":>6} | {"DomA":>5} | {"Hard":>5} |')
    print('-' * 85)

    for epoch in range(cfg.NUM_EPOCHS):
        t0 = time.time()

        # --- Progressive schedules ---
        lam_p    = ganin_lambda(epoch, cfg.NUM_EPOCHS, max_lambda=args.max_lambda)
        dr_boost = compute_dr_alpha_boost(
            epoch, cfg.NUM_EPOCHS,
            start=args.dr_alpha_start, end=args.dr_alpha_end,
        )

        # Update focal loss alpha with progressive DR boost
        alpha_this_epoch = base_alpha.clone()
        alpha_this_epoch[1] *= dr_boost
        criterion_d.alpha.copy_(alpha_this_epoch)

        # ---- TRAIN ----
        model.train()
        run_loss     = 0.0
        run_dom_loss = 0.0
        correct      = 0
        dom_correct  = 0
        total        = 0

        # Accumulators for hard-example mining
        epoch_indices = []
        epoch_losses  = []
        epoch_preds   = []
        epoch_targets = []

        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(
            train_loader,
            desc=f'E{epoch+1:03d}/{cfg.NUM_EPOCHS} train',
            leave=False,
        )

        for step, (imgs, d_lbl, s_lbl, dom_lbl, sample_idx) in enumerate(pbar):
            imgs       = imgs.to(cfg.DEVICE, non_blocking=True)
            d_lbl      = d_lbl.to(cfg.DEVICE, non_blocking=True)
            s_lbl      = s_lbl.to(cfg.DEVICE, non_blocking=True)
            dom_lbl    = dom_lbl.to(cfg.DEVICE, non_blocking=True)
            sample_idx = sample_idx.to(cfg.DEVICE, non_blocking=True)

            # Decide whether to apply mixup this batch
            apply_mixup = (random.random() < args.mixup_prob) and (args.mixup_alpha > 0)

            if apply_mixup:
                mixed_imgs, y_a, y_b, lam, mix_idx = mixup_data(
                    imgs, d_lbl, alpha=args.mixup_alpha,
                )
            else:
                mixed_imgs = imgs
                y_a = d_lbl
                y_b = d_lbl
                lam = 1.0

            with autocast('cuda'):
                # Forward pass (mixed images through backbone)
                d_out, s_out, dom_out_mixed = model(mixed_imgs, alpha=lam_p)

                # Disease loss (mixed if mixup applied)
                if apply_mixup:
                    loss_d = (lam * criterion_d(d_out, y_a) +
                              (1 - lam) * criterion_d(d_out, y_b))
                else:
                    loss_d = criterion_d(d_out, y_a)

                # Severity loss (unmixed, on original labels)
                loss_s = criterion_s(s_out, s_lbl)

                # Domain loss on ORIGINAL (unmixed) images for clean domain targets
                f_orig = model.backbone(imgs)
                f_orig = model.drop(f_orig)
                dom_out_orig = model.domain_head(model.grl(f_orig, lam_p))
                loss_dom = criterion_dom(dom_out_orig, dom_lbl)

                # Total loss
                loss = (loss_d +
                        0.2 * loss_s +
                        cfg.DOMAIN_WEIGHT * lam_p * loss_dom
                        ) / cfg.GRADIENT_ACCUMULATION

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()

            if (step + 1) % cfg.GRADIENT_ACCUMULATION == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            run_loss     += loss.item() * cfg.GRADIENT_ACCUMULATION
            run_dom_loss += loss_dom.item()

            with torch.no_grad():
                preds        = d_out.argmax(1)
                correct     += (preds == y_a).sum().item()
                dom_preds    = dom_out_orig.argmax(1)
                dom_correct += (dom_preds == dom_lbl).sum().item()
                total       += d_lbl.size(0)

                # Per-sample losses for hard-example mining (use original labels)
                per_sample_loss = criterion_per_sample(d_out.float(), d_lbl)
                epoch_indices.append(sample_idx.cpu())
                epoch_losses.append(per_sample_loss.cpu())
                epoch_preds.append(preds.cpu())
                epoch_targets.append(d_lbl.cpu())

            pbar.set_postfix(
                loss=f'{loss.item() * cfg.GRADIENT_ACCUMULATION:.3f}',
                acc=f'{100 * correct / total:.1f}%',
                dom=f'{100 * dom_correct / total:.1f}%',
                dr_a=f'{dr_boost:.1f}x',
            )

        # Flush remaining gradients
        if len(train_loader) % cfg.GRADIENT_ACCUMULATION != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # Step scheduler (per-epoch for CosineWarmRestarts)
        scheduler.step(epoch)

        # --- Update hard-example mining sampler ---
        all_epoch_indices = torch.cat(epoch_indices)
        all_epoch_losses  = torch.cat(epoch_losses)
        all_epoch_preds   = torch.cat(epoch_preds)
        all_epoch_targets = torch.cat(epoch_targets)

        sampler.update_losses(
            all_epoch_indices, all_epoch_losses,
            preds=all_epoch_preds, targets=all_epoch_targets,
        )
        mining_stats = sampler.get_stats()

        train_loss     = run_loss / max(len(train_loader), 1)
        avg_dom_loss   = run_dom_loss / max(len(train_loader), 1)
        train_acc      = 100.0 * correct / max(total, 1)
        train_dom_acc  = 100.0 * dom_correct / max(total, 1)

        # ---- VALIDATE ----
        val_loss, val_preds, val_targets, val_probs = evaluate(
            calib_loader, model, criterion_d, criterion_s, cfg.DEVICE,
            desc=f'E{epoch+1:03d}/{cfg.NUM_EPOCHS} val',
        )
        val_dom_acc = evaluate_domain(
            calib_loader, model, cfg.DEVICE, alpha=lam_p,
        )

        val_acc = 100.0 * (val_preds == val_targets).mean()
        mf1     = f1_score(val_targets, val_preds, average='macro')
        wf1     = f1_score(val_targets, val_preds, average='weighted')
        per_f1  = f1_score(
            val_targets, val_preds, average=None,
            labels=range(cfg.NUM_DISEASE_CLASSES), zero_division=0,
        )

        lr_now = optimizer.param_groups[0]['lr']

        # Record history
        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        history['domain_loss'].append(float(avg_dom_loss))
        history['train_acc'].append(float(train_acc))
        history['val_acc'].append(float(val_acc))
        history['train_domain_acc'].append(float(train_dom_acc))
        history['val_domain_acc'].append(float(val_dom_acc))
        history['macro_f1'].append(float(mf1))
        history['weighted_f1'].append(float(wf1))
        history['lr'].append(float(lr_now))
        history['lambda_p'].append(float(lam_p))
        history['dr_alpha'].append(float(dr_boost))
        history['hard_samples'].append(mining_stats['hard_samples_tracked'])
        history['dr_as_normal_errors'].append(mining_stats['dr_as_normal'])
        for ci, cn in enumerate(cfg.CLASS_NAMES):
            history[f'f1_{cn}'].append(float(per_f1[ci]))

        elapsed = time.time() - t0

        # ---- Early stopping on macro-F1 ----
        tag = ''
        if mf1 > best_f1 + cfg.MIN_DELTA:
            best_f1      = mf1
            patience_ctr = 0
            torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'val_acc':          val_acc,
                'macro_f1':         mf1,
                'domain_map':       domain_map,
                'num_domains':      num_domains,
                'history':          history,
                'args':             vars(args),
                'dr_alpha_boost':   dr_boost,
            }, CHECKPOINT)
            tag = ' *BEST*'
        else:
            patience_ctr += 1

        # Print epoch summary
        print(
            f'{epoch+1:3d} | {elapsed:4.0f}s | {lr_now:.2e} | '
            f'{lam_p:.3f} {dr_boost:.1f} | '
            f'{train_loss:6.3f} {train_acc:5.1f} | '
            f'{val_loss:6.3f} {val_acc:5.1f} | '
            f'{mf1:.4f} {wf1:.4f} | {val_dom_acc:5.1f} | '
            f'{mining_stats["dr_as_normal"]:5d} |{tag}'
        )
        cls_str = ' | '.join(
            f'{cn[:3]}:{per_f1[ci]:.2f}' for ci, cn in enumerate(cfg.CLASS_NAMES)
        )
        print(f'       {cls_str}  (train_dom={train_dom_acc:.1f}%)')

        if patience_ctr >= cfg.PATIENCE:
            print(f'\n  Early stopping after {cfg.PATIENCE} epochs without improvement')
            break

    total_time = time.time() - t_start
    print(f'\nTraining complete. Best macro-F1: {best_f1:.4f}')
    print(f'Total training time: {total_time / 60:.1f} minutes')

    # Save history
    history_path = os.path.join(cfg.OUTPUT_DIR, 'history.json')
    with open(history_path, 'w') as f:
        serializable = {}
        for k, v in history.items():
            if isinstance(v, list):
                serializable[k] = [
                    float(x) if isinstance(x, (float, np.floating)) else x
                    for x in v
                ]
            else:
                serializable[k] = v
        json.dump(serializable, f, indent=2)
    print(f'  History saved to {history_path}')

    # ----------------------------------------------------------
    # 9. Temperature scaling on calibration set
    # ----------------------------------------------------------
    print('\n[9/10] Temperature scaling on calibration set...')

    # Reload best checkpoint
    ckpt = torch.load(CHECKPOINT, map_location=cfg.DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'  Loaded best checkpoint (epoch {ckpt["epoch"]+1}, '
          f'macro-F1={ckpt["macro_f1"]:.4f})')

    calib_logits, calib_labels = collect_logits_labels(
        calib_loader, model, cfg.DEVICE,
    )

    probs_before = torch.softmax(calib_logits, dim=1).numpy()
    ece_before   = compute_ece(probs_before, calib_labels.numpy())
    print(f'  ECE before temperature scaling: {ece_before:.4f}')

    T_opt = temperature_scale(calib_logits, calib_labels)
    print(f'  Optimal temperature T = {T_opt:.4f}')

    probs_after = torch.softmax(calib_logits / T_opt, dim=1).numpy()
    ece_after   = compute_ece(probs_after, calib_labels.numpy())
    print(f'  ECE after  temperature scaling: {ece_after:.4f}')

    temp_data = {
        'temperature': T_opt,
        'ece_before': ece_before,
        'ece_after': ece_after,
        'model_version': 'dann_v3',
    }
    temp_path = 'configs/temperature.json'
    with open(temp_path, 'w') as f:
        json.dump(temp_data, f, indent=2)
    print(f'  Saved -> {temp_path}')

    # ----------------------------------------------------------
    # 10. Threshold optimisation + final evaluation
    # ----------------------------------------------------------
    print('\n[10/10] Per-class threshold optimisation and final evaluation...')

    calib_thresholds = optimise_thresholds(
        probs_after, calib_labels.numpy(),
        cfg.NUM_DISEASE_CLASSES, cfg.CLASS_NAMES,
    )

    thresh_data = {
        'thresholds': calib_thresholds,
        'class_names': cfg.CLASS_NAMES,
        'model_version': 'dann_v3',
    }
    thresh_path = 'configs/thresholds.json'
    with open(thresh_path, 'w') as f:
        json.dump(thresh_data, f, indent=2)
    print(f'  Saved -> {thresh_path}')

    # ----------------------------------------------------------
    # Final evaluation on test set
    # ----------------------------------------------------------
    print('\n' + '=' * 75)
    print('         FINAL EVALUATION -- TEST SET')
    print('=' * 75)
    print('  (Test set was never seen during training or threshold tuning)')

    # Standard evaluation
    test_logits, test_labels = collect_logits_labels(
        test_loader, model, cfg.DEVICE,
    )
    test_probs     = torch.softmax(test_logits / T_opt, dim=1).numpy()
    test_labels_np = test_labels.numpy()

    test_preds_raw = test_probs.argmax(axis=1)
    test_preds_thr = apply_thresholds(test_probs, calib_thresholds)

    def print_metrics(preds, targets, probs, label):
        acc = 100.0 * (preds == targets).mean()
        mf1 = f1_score(targets, preds, average='macro')
        wf1 = f1_score(targets, preds, average='weighted')
        try:
            mauc = roc_auc_score(targets, probs, multi_class='ovr', average='macro')
        except Exception:
            mauc = 0.0
        per = f1_score(
            targets, preds, average=None,
            labels=range(cfg.NUM_DISEASE_CLASSES), zero_division=0,
        )
        ece = compute_ece(probs, targets)

        print(f'\n  [{label}]')
        print(f'  Accuracy   : {acc:.2f}%')
        print(f'  Macro F1   : {mf1:.4f}')
        print(f'  Weighted F1: {wf1:.4f}')
        print(f'  Macro AUC  : {mauc:.4f}')
        print(f'  ECE        : {ece:.4f}')
        print()
        print(classification_report(
            targets, preds, target_names=cfg.CLASS_NAMES, digits=4,
        ))
        return {
            'accuracy': float(acc), 'macro_f1': float(mf1),
            'weighted_f1': float(wf1), 'macro_auc': float(mauc),
            'ece': float(ece),
            **{f'f1_{cfg.CLASS_NAMES[i]}': float(per[i])
               for i in range(cfg.NUM_DISEASE_CLASSES)},
        }

    metrics_raw = print_metrics(
        test_preds_raw, test_labels_np, test_probs,
        'Raw argmax (T-scaled)',
    )
    metrics_thr = print_metrics(
        test_preds_thr, test_labels_np, test_probs,
        'With per-class thresholds',
    )

    # TTA evaluation (if enabled)
    metrics_tta = None
    if args.tta:
        print('\n' + '-' * 65)
        print('  Test-Time Augmentation (TTA)')
        print('-' * 65)
        tta_probs, tta_targets = evaluate_with_tta(
            test_df, domain_map, model, cfg.DEVICE,
            NORM_MEAN, NORM_STD,
            batch_size=cfg.BATCH_SIZE,
            num_workers=cfg.NUM_WORKERS,
            n_augs=args.tta_n,
            desc='TTA Test',
        )

        # Apply temperature scaling to TTA probs
        # (TTA already produces averaged softmax probs, so T-scaling is
        # approximate; but we apply it for consistency)
        tta_preds_raw = tta_probs.argmax(axis=1)
        tta_preds_thr = apply_thresholds(tta_probs, calib_thresholds)

        print('\n  TTA Results:')
        metrics_tta_raw = print_metrics(
            tta_preds_raw, tta_targets, tta_probs,
            f'TTA ({args.tta_n}-way) raw argmax',
        )
        metrics_tta = print_metrics(
            tta_preds_thr, tta_targets, tta_probs,
            f'TTA ({args.tta_n}-way) + thresholds',
        )

    # Domain accuracy on test set
    test_dom_acc = evaluate_domain(test_loader, model, cfg.DEVICE, alpha=1.0)
    print(f'  Test domain accuracy: {test_dom_acc:.1f}% '
          f'(chance={100.0/num_domains:.1f}%)')

    # Save final metrics
    final_metrics = {
        'raw': metrics_raw,
        'thresholded': metrics_thr,
        'tta': metrics_tta,
        'temperature': T_opt,
        'thresholds': calib_thresholds,
        'domain_accuracy_test': test_dom_acc,
        'num_domains': num_domains,
        'domain_map': domain_map,
        'improvements': {
            'expanded_dataset': True,
            'hard_example_mining': True,
            'progressive_dr_alpha': f'{args.dr_alpha_start} -> {args.dr_alpha_end}',
            'label_smoothing': args.label_smoothing,
            'mixup_alpha': args.mixup_alpha,
            'cosine_warm_restarts': f'T0={args.cosine_t0}, Tmult={args.cosine_tmult}',
            'tta_enabled': args.tta,
        },
    }
    metrics_path = os.path.join(cfg.OUTPUT_DIR, 'final_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=2)

    # Save confusion matrix
    cm = confusion_matrix(test_labels_np, test_preds_thr)
    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=cfg.CLASS_NAMES, yticklabels=cfg.CLASS_NAMES,
                ax=ax_cm)
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('True')
    ax_cm.set_title('DANN v3 - Test Set Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, 'confusion_matrix.png'), dpi=150)
    plt.close()

    # ----------------------------------------------------------
    # Dashboard plot
    # ----------------------------------------------------------
    print('\nGenerating dashboard...')
    plot_dashboard(history, cfg, cfg.OUTPUT_DIR)

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print('\n' + '=' * 75)
    print('         RETINASENSE DANN v3 -- FINAL SUMMARY')
    print('=' * 75)
    print(f'  Training epochs     : {len(history["train_loss"])}')
    print(f'  Best calib macro-F1 : {best_f1:.4f}')
    print(f'  Temperature T       : {T_opt:.4f}')
    print(f'  ECE before / after  : {ece_before:.4f} / {ece_after:.4f}')
    print(f'  Test domain acc     : {test_dom_acc:.1f}% '
          f'(chance={100.0/num_domains:.1f}%)')
    print()
    print('  v3 IMPROVEMENTS:')
    print(f'    Expanded dataset  : {len(train_df)} train samples '
          f'(+MESSIDOR2, 4 domains)')
    print(f'    Hard mining       : top-{args.hard_mining_k} x{args.hard_mining_factor}')
    print(f'    Progressive DR    : alpha {args.dr_alpha_start} -> {args.dr_alpha_end}')
    print(f'    Label smoothing   : {args.label_smoothing}')
    print(f'    Mixup             : alpha={args.mixup_alpha}, prob={args.mixup_prob}')
    print(f'    Scheduler         : CosineWarmRestarts '
          f'(T0={args.cosine_t0}, Tmult={args.cosine_tmult})')
    print(f'    TTA               : {"Enabled" if args.tta else "Disabled"}')
    print()
    print('  TEST SET RESULTS (with thresholds):')
    print(f'    Accuracy   : {metrics_thr["accuracy"]:.2f}%')
    print(f'    Macro F1   : {metrics_thr["macro_f1"]:.4f}')
    print(f'    Weighted F1: {metrics_thr["weighted_f1"]:.4f}')
    print(f'    Macro AUC  : {metrics_thr["macro_auc"]:.4f}')
    print(f'    ECE        : {metrics_thr["ece"]:.4f}')

    if metrics_tta:
        print()
        print(f'  TEST SET RESULTS (TTA + thresholds):')
        print(f'    Accuracy   : {metrics_tta["accuracy"]:.2f}%')
        print(f'    Macro F1   : {metrics_tta["macro_f1"]:.4f}')
        print(f'    Weighted F1: {metrics_tta["weighted_f1"]:.4f}')
        print(f'    Macro AUC  : {metrics_tta["macro_auc"]:.4f}')
        print(f'    ECE        : {metrics_tta["ece"]:.4f}')

    print()
    print('  Per-class F1 (test, thresholded):')
    for i, cn in enumerate(cfg.CLASS_NAMES):
        thr = calib_thresholds[i]
        fi  = metrics_thr[f'f1_{cn}']
        print(f'    {cn:15s}: F1={fi:.3f}  (threshold={thr:.3f})')
    print()
    print(f'  Training time: {total_time / 60:.1f} minutes')
    print()
    print(f'  Outputs saved to {cfg.OUTPUT_DIR}/')
    for fname in ['best_model.pth', 'history.json', 'dashboard.png',
                   'confusion_matrix.png', 'final_metrics.json']:
        print(f'    -- {fname}')
    print(f'  Configs saved to configs/')
    for fname in ['temperature.json', 'thresholds.json']:
        print(f'    -- {fname}')
    print('=' * 75)


if __name__ == '__main__':
    main()
