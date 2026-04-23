#!/usr/bin/env python3
"""
RetinaSense DANN v4 — RETFound Backbone + Enhanced Training
============================================================
Target: Push from 89.3% (v3) to 92-93% accuracy.

Key improvements over DANN-v3:
1. **RETFound backbone**: ViT-Large/16 pre-trained on 1.6M retinal fundus images
   via masked autoencoding — replaces generic ImageNet ViT-Base pretraining with
   domain-specific features (vessel patterns, optic disc, microaneurysms).
   Model size increases from 86M (ViT-B) to 304M (ViT-L) params.
2. **CutMix + MixUp combo**: 40% MixUp, 40% CutMix, 20% clean per batch.
3. **Class-aware augmentation**: Stronger transforms for minority classes
   (Glaucoma, Cataract, AMD) to combat 21:1 class imbalance.
4. **Stochastic Weight Averaging (SWA)**: Averages model weights from the
   last 10 epochs for flatter optima and better generalisation.
5. **Optimised hyperparams for RETFound**: Lower LR (1e-5), more aggressive
   LLRD (0.80), less dropout (0.2), 60 epochs.

Architecture is identical to DANNMultiTaskViT from train_dann_v3.py.
Backbone is initialised from RETFound, heads are randomly initialised
(or optionally warm-started from DANN-v3 heads).

Prerequisites:
  python retfound_backbone.py --setup   # download RETFound weights (~350MB)

Usage:
  python train_dann_v4.py
  python train_dann_v4.py --epochs 60 --lr 1e-5 --tta
  python train_dann_v4.py --warmstart-heads outputs_v3/dann_v3/best_model.pth
  python train_dann_v4.py --no-retfound   # fall back to ImageNet (ablation)
"""

import os
import sys
import time
import json
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
from collections import Counter

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torchvision import transforms

import timm

from scipy.optimize import minimize_scalar
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, cohen_kappa_score, matthews_corrcoef,
)


# ================================================================
# CLI ARGUMENTS
# ================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description='RetinaSense DANN v4 — RETFound + Enhanced Training',
    )
    # -- backbone --
    p.add_argument('--retfound-weights', type=str,
                   default='weights/RETFound_cfp_weights.pth',
                   help='Path to RETFound weights (default: weights/RETFound_cfp_weights.pth)')
    p.add_argument('--no-retfound', action='store_true',
                   help='Skip RETFound; use ImageNet pretrained ViT (ablation)')

    # -- warm-start --
    p.add_argument('--warmstart-heads', type=str, default=None,
                   help='Load disease/severity/domain heads from this checkpoint '
                        '(e.g. outputs_v3/dann_v3/best_model.pth)')

    # -- training --
    p.add_argument('--epochs', type=int, default=60, help='Training epochs (default: 60)')
    p.add_argument('--lr', type=float, default=1e-5,
                   help='Base learning rate (default: 1e-5, tuned for RETFound)')
    p.add_argument('--batch-size', type=int, default=32, help='Batch size (default: 32)')
    p.add_argument('--workers', type=int, default=8, help='DataLoader workers')
    p.add_argument('--img-size', type=int, default=224,
                   help='Image size (224 or 384, default: 224)')
    p.add_argument('--output-dir', type=str, default=None,
                   help='Override output directory (default: outputs_v3/dann_v4)')

    # -- v4 improvements --
    p.add_argument('--cutmix-prob', type=float, default=0.4,
                   help='Probability of CutMix per batch (default: 0.4)')
    p.add_argument('--mixup-prob', type=float, default=0.4,
                   help='Probability of MixUp per batch (default: 0.4)')
    p.add_argument('--mixup-alpha', type=float, default=0.3,
                   help='MixUp/CutMix alpha parameter (default: 0.3)')
    p.add_argument('--label-smoothing', type=float, default=0.05,
                   help='Label smoothing (default: 0.05, lower than v3 — RETFound is more discriminative)')
    p.add_argument('--swa-start', type=int, default=None,
                   help='Epoch to start SWA (default: epochs-10)')
    p.add_argument('--swa-lr', type=float, default=5e-6,
                   help='SWA learning rate (default: 5e-6)')
    p.add_argument('--dropout', type=float, default=0.2,
                   help='Head dropout (default: 0.2, lower than v3)')
    p.add_argument('--llrd-decay', type=float, default=0.80,
                   help='LLRD decay factor (default: 0.80, more aggressive than v3)')

    # -- domain adaptation --
    p.add_argument('--domain-weight', type=float, default=0.05,
                   help='Domain loss weight multiplier (default: 0.05)')
    p.add_argument('--max-lambda', type=float, default=0.3,
                   help='Cap for Ganin lambda schedule (default: 0.3)')

    # -- hard mining + DR alpha --
    p.add_argument('--dr-alpha-start', type=float, default=1.5,
                   help='DR focal alpha boost at epoch 0')
    p.add_argument('--dr-alpha-end', type=float, default=3.0,
                   help='DR focal alpha boost at final epoch')
    p.add_argument('--hard-mining-k', type=int, default=500,
                   help='Number of hard examples to oversample each epoch')
    p.add_argument('--hard-mining-factor', type=int, default=2,
                   help='Oversampling factor for hard examples')

    # -- evaluation --
    p.add_argument('--tta', action='store_true',
                   help='Enable Test-Time Augmentation for final evaluation')
    p.add_argument('--tta-n', type=int, default=8, help='Number of TTA augmentations')

    # -- misc --
    p.add_argument('--seed', type=int, default=42, help='Random seed')
    p.add_argument('--gradient-accumulation', type=int, default=2,
                   help='Gradient accumulation steps (default: 2)')

    args = p.parse_args()
    if args.swa_start is None:
        args.swa_start = max(args.epochs - 10, args.epochs // 2)
    return args


# ================================================================
# CONFIG
# ================================================================
class Config:
    DATA_DIR   = './data'
    OUTPUT_DIR = './outputs_v3/dann_v4'

    # RETFound uses ViT-Large (dim=1024, 24 blocks); fallback is ViT-Base
    MODEL_NAME = 'vit_large_patch16_224'  # overridden to vit_base if --no-retfound
    IMG_SIZE   = 224
    FEAT_DIM   = 1024  # 1024 for ViT-Large, 768 for ViT-Base

    NUM_DISEASE_CLASSES  = 5
    NUM_SEVERITY_CLASSES = 5

    BATCH_SIZE  = 32
    NUM_EPOCHS  = 60
    NUM_WORKERS = 8

    BASE_LR      = 1e-5
    LLRD_DECAY   = 0.80
    WEIGHT_DECAY = 1e-4

    GRADIENT_ACCUMULATION = 2

    FOCAL_GAMMA    = 2.0
    DOMAIN_WEIGHT  = 0.05

    PATIENCE  = 20  # more patience for longer training
    MIN_DELTA = 0.001

    CLASS_NAMES = ['Normal', 'Diabetes/DR', 'Glaucoma', 'Cataract', 'AMD']
    MINORITY_CLASSES = {2, 3, 4}  # Glaucoma, Cataract, AMD

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    TRAIN_CSV = './data/train_split_expanded.csv'
    CALIB_CSV = './data/calib_split_expanded.csv'
    TEST_CSV  = './data/test_split.csv'

    TRAIN_CSV_FALLBACK = './data/train_split.csv'
    CALIB_CSV_FALLBACK = './data/calib_split.csv'


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
# GRADIENT REVERSAL LAYER
# ================================================================
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GRL(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, alpha=1.0):
        return GradientReversalFunction.apply(x, alpha)


def ganin_lambda(epoch, total_epochs, max_lambda=0.3):
    p = epoch / total_epochs
    raw = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0
    return min(raw, max_lambda)


# ================================================================
# DANN MULTI-TASK VIT MODEL
# ================================================================
class DANNMultiTaskViT(nn.Module):
    def __init__(self, n_disease=5, n_severity=5, num_domains=4,
                 drop=0.2, backbone_name='vit_base_patch16_224',
                 pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0,
        )
        feat = self.backbone.num_features  # 768 for ViT-Base

        self.drop = nn.Dropout(drop)

        # Disease classification head
        self.disease_head = nn.Sequential(
            nn.Linear(feat, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(drop * 0.67),
            nn.Linear(256, n_disease),
        )

        # Severity grading head
        self.severity_head = nn.Sequential(
            nn.Linear(feat, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(256, n_severity),
        )

        # Domain discriminator head
        self.grl = GRL()
        self.domain_head = nn.Sequential(
            nn.Linear(feat, 256), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, num_domains),
        )

    def forward(self, x, alpha=1.0):
        f = self.backbone(x)
        f = self.drop(f)
        disease_out  = self.disease_head(f)
        severity_out = self.severity_head(f)
        domain_out   = self.domain_head(self.grl(f, alpha))
        return disease_out, severity_out, domain_out

    def forward_no_domain(self, x):
        f = self.backbone(x)
        f = self.drop(f)
        return self.disease_head(f), self.severity_head(f)

    def forward_features(self, x):
        return self.backbone(x)


# ================================================================
# FOCAL LOSS
# ================================================================
class FocalLoss(nn.Module):
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
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            at = self.alpha.gather(0, targets)
            focal = at * focal
        return focal.mean()


# ================================================================
# NORMALISATION STATS
# ================================================================
def load_norm_stats():
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
            return mean, std
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    print('  No fundus norm stats found — using ImageNet defaults')
    return mean, std


# ================================================================
# CACHE PATH RESOLUTION
# ================================================================
def resolve_cache_path(cache_path_from_csv, img_size=224):
    """Resolve cache path, supporting both 224 and 384 caches."""
    unified_path = cache_path_from_csv.replace(
        'preprocessed_cache_v3', 'preprocessed_cache_unified'
    )
    # If requesting non-224, try size-specific cache
    if img_size != 224:
        size_path = unified_path.replace('_224.npy', f'_{img_size}.npy')
        if os.path.exists(size_path):
            return size_path
    if os.path.exists(unified_path):
        return unified_path
    if os.path.exists(cache_path_from_csv):
        return cache_path_from_csv
    return cache_path_from_csv


# ================================================================
# IMAGE PREPROCESSING (fallback)
# ================================================================
def clahe_preprocess(path, sz=224):
    img = cv2.imread(path)
    if img is None:
        img = np.array(Image.open(path).convert('RGB'))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (sz, sz))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (sz // 2, sz // 2), int(sz * 0.48), 255, -1)
    return cv2.bitwise_and(img, img, mask=mask)


# ================================================================
# CLASS-AWARE TRANSFORMS
# ================================================================
def make_majority_transform(norm_mean, norm_std, img_size=224):
    """Standard augmentation for Normal and DR (majority classes)."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
        transforms.RandomErasing(p=0.2),
    ])


def make_minority_transform(norm_mean, norm_std, img_size=224):
    """Strong augmentation for Glaucoma, Cataract, AMD (minority classes)."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.4),
        transforms.RandomRotation(30),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08),
                                scale=(0.90, 1.10), shear=5),
        transforms.RandomPerspective(distortion_scale=0.15, p=0.3),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.04),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
    ])


def make_val_transform(norm_mean, norm_std, img_size=224):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])


def make_tta_transforms(norm_mean, norm_std, n_augs=8):
    normalize = transforms.Normalize(norm_mean, norm_std)
    base = [transforms.ToPILImage()]
    tta_list = [
        # 1. Identity
        transforms.Compose([*base, transforms.ToTensor(), normalize]),
        # 2. H-flip
        transforms.Compose([*base, transforms.RandomHorizontalFlip(p=1.0),
                            transforms.ToTensor(), normalize]),
        # 3. V-flip
        transforms.Compose([*base, transforms.RandomVerticalFlip(p=1.0),
                            transforms.ToTensor(), normalize]),
        # 4. HV-flip
        transforms.Compose([*base, transforms.RandomHorizontalFlip(p=1.0),
                            transforms.RandomVerticalFlip(p=1.0),
                            transforms.ToTensor(), normalize]),
        # 5-7. Rotations
        transforms.Compose([*base, transforms.Lambda(lambda img: img.rotate(90)),
                            transforms.ToTensor(), normalize]),
        transforms.Compose([*base, transforms.Lambda(lambda img: img.rotate(180)),
                            transforms.ToTensor(), normalize]),
        transforms.Compose([*base, transforms.Lambda(lambda img: img.rotate(270)),
                            transforms.ToTensor(), normalize]),
        # 8. Center crop
        transforms.Compose([*base, transforms.CenterCrop(200),
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(), normalize]),
    ]
    return tta_list[:n_augs]


# ================================================================
# DATASET WITH CLASS-AWARE AUGMENTATION
# ================================================================
class RetinalDANNv4Dataset(Dataset):
    """
    Retinal dataset with class-aware augmentation:
    - Majority classes (Normal=0, DR=1): standard augmentation
    - Minority classes (Glaucoma=2, Cataract=3, AMD=4): strong augmentation
    """

    def __init__(self, df, transform_majority, transform_minority, domain_map,
                 img_size=224):
        self.df = df.reset_index(drop=True)
        self.transform_majority = transform_majority
        self.transform_minority = transform_minority
        self.domain_map = domain_map
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image
        cache_fp = resolve_cache_path(str(row.get('cache_path', '')), self.img_size)
        try:
            img = np.load(cache_fp)
            # Resize if cache is at different resolution
            if img.shape[0] != self.img_size or img.shape[1] != self.img_size:
                img = cv2.resize(img, (self.img_size, self.img_size))
        except Exception:
            try:
                img = clahe_preprocess(
                    row['image_path'], self.img_size,
                )
            except Exception:
                img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        disease_lbl = int(row['disease_label'])

        # Class-aware transform selection
        if disease_lbl in {2, 3, 4}:  # minority
            img_tensor = self.transform_minority(img)
        else:  # majority
            img_tensor = self.transform_majority(img)

        severity_lbl = int(row['severity_label'])
        if severity_lbl < 0:
            severity_lbl = 0

        source = str(row.get('source', 'UNKNOWN')).upper()
        domain_lbl = self.domain_map.get(source, 0)

        return (
            img_tensor,
            torch.tensor(disease_lbl,  dtype=torch.long),
            torch.tensor(severity_lbl, dtype=torch.long),
            torch.tensor(domain_lbl,   dtype=torch.long),
            torch.tensor(idx,          dtype=torch.long),
        )


class RetinalValDataset(Dataset):
    """Simple dataset for val/test (no class-aware transforms needed)."""

    def __init__(self, df, transform, domain_map, img_size=224):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.domain_map = domain_map
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cache_fp = resolve_cache_path(str(row.get('cache_path', '')), self.img_size)
        try:
            img = np.load(cache_fp)
            if img.shape[0] != self.img_size or img.shape[1] != self.img_size:
                img = cv2.resize(img, (self.img_size, self.img_size))
        except Exception:
            try:
                img = clahe_preprocess(row['image_path'], self.img_size)
            except Exception:
                img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        img_tensor = self.transform(img)
        disease_lbl = int(row['disease_label'])
        severity_lbl = max(0, int(row['severity_label']))
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
# HARD-EXAMPLE MINING SAMPLER
# ================================================================
class HardExampleMiningWeightedSampler(Sampler):
    def __init__(self, labels, n_classes=5, hard_k=500, hard_factor=2):
        self.labels      = np.array(labels)
        self.n_classes   = n_classes
        self.n_samples   = len(labels)
        self.hard_k      = hard_k
        self.hard_factor = hard_factor

        class_cnt = np.bincount(self.labels, minlength=n_classes).astype(float)
        class_cnt = np.where(class_cnt == 0, 1.0, class_cnt)
        self.base_weights = 1.0 / class_cnt[self.labels]

        self.sample_losses = np.zeros(self.n_samples, dtype=np.float64)
        self.sample_wrong  = np.zeros(self.n_samples, dtype=bool)
        self.sample_pred   = np.full(self.n_samples, -1, dtype=np.int64)
        self._update_weights()

    def _update_weights(self):
        weights = self.base_weights.copy()
        if self.sample_losses.sum() > 0:
            top_k_idx = np.argsort(self.sample_losses)[-self.hard_k:]
            weights[top_k_idx] *= self.hard_factor
            # Extra boost for DR→Normal misclassifications
            dr_as_normal = (
                (self.labels == 1) & self.sample_wrong & (self.sample_pred == 0)
            )
            if dr_as_normal.sum() > 0:
                weights[dr_as_normal] *= self.hard_factor
        self.effective_weights = torch.DoubleTensor(weights)

    def update_losses(self, indices, losses, preds=None, targets=None):
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
            self.effective_weights, num_samples=self.n_samples, replacement=True,
        ).tolist())

    def __len__(self):
        return self.n_samples

    def get_stats(self):
        n_hard = min(self.hard_k, (self.sample_losses > 0).sum())
        n_wrong = self.sample_wrong.sum()
        dr_as_normal = (
            (self.labels == 1) & self.sample_wrong & (self.sample_pred == 0)
        ).sum()
        return {'hard_samples_tracked': int(n_hard), 'wrong_total': int(n_wrong),
                'dr_as_normal': int(dr_as_normal)}


# ================================================================
# CUTMIX + MIXUP COMBO
# ================================================================
def mixup_data(x, y, alpha=0.3):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def cutmix_data(x, y, alpha=0.3):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    _, _, h, w = x.shape

    cut_ratio = math.sqrt(1.0 - lam)
    cut_w = int(w * cut_ratio)
    cut_h = int(h * cut_ratio)
    cx = random.randint(0, w)
    cy = random.randint(0, h)
    x1 = max(0, cx - cut_w // 2)
    y1 = max(0, cy - cut_h // 2)
    x2 = min(w, cx + cut_w // 2)
    y2 = min(h, cy + cut_h // 2)

    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1.0 - (x2 - x1) * (y2 - y1) / (w * h)
    return mixed_x, y, y[index], lam


def apply_batch_augmentation(x, y, mixup_prob=0.4, cutmix_prob=0.4, alpha=0.3):
    """Randomly apply MixUp, CutMix, or neither to a batch."""
    r = random.random()
    if r < mixup_prob:
        return mixup_data(x, y, alpha)
    elif r < mixup_prob + cutmix_prob:
        return cutmix_data(x, y, alpha)
    else:
        return x, y, y, 1.0


# ================================================================
# LLRD OPTIMIZER
# ================================================================
def get_optimizer_with_llrd(model, base_lr, decay_factor, weight_decay=1e-4):
    param_groups = []

    # Heads at full LR
    head_params = (
        list(model.disease_head.parameters()) +
        list(model.severity_head.parameters()) +
        list(model.domain_head.parameters()) +
        list(model.drop.parameters())
    )
    param_groups.append({'params': head_params, 'lr': base_lr})

    # Transformer blocks
    blocks = model.backbone.blocks
    num_blocks = len(blocks)
    for block_idx in range(num_blocks - 1, -1, -1):
        distance_from_head = num_blocks - block_idx
        lr_i = base_lr * (decay_factor ** distance_from_head)
        param_groups.append({
            'params': list(blocks[block_idx].parameters()),
            'lr': lr_i,
        })

    # Patch/position embeddings
    embed_lr = base_lr * (decay_factor ** (num_blocks + 1))
    embed_params = (
        list(model.backbone.patch_embed.parameters()) +
        [model.backbone.cls_token, model.backbone.pos_embed] +
        list(model.backbone.norm.parameters())
    )
    param_groups.append({'params': embed_params, 'lr': embed_lr})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    lrs = [g['lr'] for g in param_groups]
    print(f'  LLRD optimizer: {len(param_groups)} param groups')
    print(f'    Head LR    : {lrs[0]:.2e}')
    print(f'    Block[{num_blocks-1}]  : {lrs[1]:.2e}')
    print(f'    Block[0]   : {lrs[-2]:.2e}')
    print(f'    Embed LR   : {lrs[-1]:.2e}')
    return optimizer


# ================================================================
# RETFOUND WEIGHT LOADING
# ================================================================
def load_retfound_backbone(model, weights_path, device):
    """Load RETFound weights into model backbone using retfound_backbone.py."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from retfound_backbone import load_retfound_into_vit, download_retfound
    from pathlib import Path

    wp = Path(weights_path)
    if not wp.exists():
        print(f'  RETFound weights not found at {wp}, downloading...')
        download_retfound(dest=wp)

    loaded, missing, unexpected = load_retfound_into_vit(model, wp, strict=False)
    return len(loaded)


def load_head_weights(model, checkpoint_path, device):
    """Load only head weights from a DANN checkpoint (not backbone).

    Note: When going from ViT-Base (768) to ViT-Large (1024), the first
    linear layer of each head will have a shape mismatch and be skipped.
    Only BN, bias, and later linear layers that match will load.
    """
    if not os.path.exists(checkpoint_path):
        print(f'  WARNING: Head warm-start checkpoint not found: {checkpoint_path}')
        return False

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model_state = model.state_dict()

    loaded = 0
    skipped = 0
    for key, value in state_dict.items():
        # Only load head weights, not backbone
        if any(key.startswith(prefix) for prefix in
               ['disease_head.', 'severity_head.', 'domain_head.', 'drop.']):
            if key in model_state and model_state[key].shape == value.shape:
                model_state[key] = value
                loaded += 1
            elif key in model_state:
                skipped += 1

    model.load_state_dict(model_state)
    print(f'  Loaded {loaded} head params from {checkpoint_path}')
    if skipped:
        print(f'  Skipped {skipped} head params (shape mismatch — ViT-Base vs ViT-Large)')
    return True


# ================================================================
# EVALUATION
# ================================================================
def evaluate(loader, model, criterion_d, criterion_s, device, desc='Eval'):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            imgs  = batch[0].to(device, non_blocking=True)
            d_lbl = batch[1].to(device, non_blocking=True)
            s_lbl = batch[2].to(device, non_blocking=True)

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
    return avg_loss, np.array(all_preds), np.array(all_targets), np.array(all_probs)


def evaluate_domain(loader, model, device, alpha=1.0):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            imgs    = batch[0].to(device, non_blocking=True)
            dom_lbl = batch[3].to(device, non_blocking=True)
            with autocast('cuda'):
                _, _, dom_out = model(imgs, alpha=alpha)
            correct += (dom_out.argmax(1) == dom_lbl).sum().item()
            total   += dom_lbl.size(0)
    return 100.0 * correct / max(total, 1)


def evaluate_with_tta(dataset_df, domain_map, model, device,
                      norm_mean, norm_std, batch_size=32, num_workers=4,
                      n_augs=8, img_size=224):
    tta_transforms = make_tta_transforms(norm_mean, norm_std, n_augs=n_augs)
    model.eval()
    n_samples = len(dataset_df)
    all_probs   = np.zeros((n_samples, 5), dtype=np.float64)
    all_targets = np.zeros(n_samples, dtype=np.int64)

    print(f'  Running TTA with {len(tta_transforms)} augmentations...')
    for aug_idx, tta_tfm in enumerate(tta_transforms):
        ds = RetinalValDataset(dataset_df, tta_tfm, domain_map, img_size)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
        batch_start = 0
        with torch.no_grad():
            for batch in tqdm(loader, desc=f'  TTA {aug_idx+1}/{len(tta_transforms)}',
                              leave=False):
                imgs = batch[0].to(device, non_blocking=True)
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

    all_probs /= len(tta_transforms)
    return all_probs, all_targets


# ================================================================
# CALIBRATION UTILITIES
# ================================================================
def compute_ece(probs, labels, n_bins=15):
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies  = predictions == labels
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        ece_val += mask.sum() * abs(accuracies[mask].mean() - confidences[mask].mean())
    return float(ece_val / len(labels))


def temperature_scale(logits, labels):
    def nll_at_T(T):
        scaled = logits / T
        return F.nll_loss(F.log_softmax(scaled, dim=1), labels).item()
    result = minimize_scalar(nll_at_T, bounds=(0.01, 10.0), method='bounded')
    return float(result.x)


def optimise_thresholds(probs, labels, n_classes, class_names, n_grid=50):
    thresholds = []
    for c in range(n_classes):
        binary_labels = (labels == c).astype(int)
        best_t, best_f1 = 0.5, 0.0
        for t in np.linspace(0.05, 0.95, n_grid):
            f = f1_score(binary_labels, (probs[:, c] >= t).astype(int), zero_division=0)
            if f > best_f1:
                best_f1, best_t = f, t
        thresholds.append(float(best_t))
        print(f'    {class_names[c]:15s}: threshold={best_t:.3f}  (calib F1={best_f1:.3f})')
    return thresholds


def apply_thresholds(probs, thresholds):
    preds = []
    for prob_row in probs:
        above = [i for i, (p, t) in enumerate(zip(prob_row, thresholds)) if p >= t]
        if above:
            preds.append(int(above[np.argmax([prob_row[i] for i in above])]))
        else:
            preds.append(int(np.argmax(prob_row)))
    return np.array(preds)


# ================================================================
# DR ALPHA SCHEDULE
# ================================================================
def compute_dr_alpha_boost(epoch, total_epochs, start=1.5, end=3.0):
    if total_epochs <= 1:
        return end
    return start + (end - start) * epoch / (total_epochs - 1)


# ================================================================
# COLLECT LOGITS
# ================================================================
def collect_logits_labels(loader, model, device):
    all_logits, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc='Collecting logits', leave=False):
            imgs = batch[0].to(device, non_blocking=True)
            d_lbl = batch[1]
            d_out, _ = model.forward_no_domain(imgs)
            all_logits.append(d_out.float().cpu())
            all_labels.append(d_lbl.cpu())
    return torch.cat(all_logits), torch.cat(all_labels)


# ================================================================
# DASHBOARD
# ================================================================
def plot_dashboard(history, cfg, output_dir):
    ep = range(1, len(history['train_loss']) + 1)
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    axes[0, 0].plot(ep, history['train_loss'], 'b-o', ms=3, label='Train')
    axes[0, 0].plot(ep, history['val_loss'], 'r-o', ms=3, label='Val')
    axes[0, 0].set_title('Loss', fontweight='bold')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(ep, history['train_acc'], 'b-o', ms=3, label='Train')
    axes[0, 1].plot(ep, history['val_acc'], 'r-o', ms=3, label='Val')
    axes[0, 1].set_title('Accuracy (%)', fontweight='bold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    axes[0, 2].plot(ep, history['train_domain_acc'], 'b-o', ms=3, label='Train')
    axes[0, 2].plot(ep, history['val_domain_acc'], 'r-o', ms=3, label='Val')
    axes[0, 2].axhline(y=25, color='gray', linestyle='--', alpha=0.5, label='Chance')
    axes[0, 2].set_title('Domain Acc (lower=better)', fontweight='bold')
    axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    axes[1, 0].plot(ep, history['macro_f1'], 'g-o', ms=3, label='Macro F1')
    axes[1, 0].plot(ep, history['weighted_f1'], 'm-o', ms=3, label='Weighted F1')
    axes[1, 0].set_title('F1 Scores', fontweight='bold')
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    for ci, cn in enumerate(cfg.CLASS_NAMES):
        key = f'f1_{cn}'
        if key in history:
            axes[1, 1].plot(ep, history[key], '-o', ms=2, color=colors[ci], label=cn)
    axes[1, 1].set_title('Per-Class F1', fontweight='bold')
    axes[1, 1].legend(fontsize=8); axes[1, 1].grid(alpha=0.3)

    axes[1, 2].plot(ep, history['lr'], 'k-o', ms=3, label='LR')
    axes[1, 2].set_title('Learning Rate', fontweight='bold')
    axes[1, 2].set_yscale('log'); axes[1, 2].grid(alpha=0.3)

    plt.suptitle('RetinaSense DANN v4 (RETFound) Training Dashboard',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dashboard.png'), dpi=150, bbox_inches='tight')
    plt.close()


# ================================================================
# MAIN TRAINING LOOP
# ================================================================
def main():
    args = parse_args()
    cfg  = Config()

    set_seed(args.seed)

    # Override config from CLI
    cfg.NUM_EPOCHS    = args.epochs
    cfg.BASE_LR       = args.lr
    cfg.DOMAIN_WEIGHT = args.domain_weight
    cfg.BATCH_SIZE    = args.batch_size
    cfg.NUM_WORKERS   = args.workers
    cfg.IMG_SIZE      = args.img_size
    cfg.LLRD_DECAY    = args.llrd_decay
    cfg.GRADIENT_ACCUMULATION = args.gradient_accumulation

    # Select backbone: RETFound uses ViT-Large, fallback uses ViT-Base
    if args.no_retfound:
        if args.img_size == 384:
            cfg.MODEL_NAME = 'vit_base_patch16_384'
        else:
            cfg.MODEL_NAME = 'vit_base_patch16_224'
        cfg.FEAT_DIM = 768
    else:
        # RETFound is ViT-Large/16 (dim=1024, 24 blocks)
        cfg.MODEL_NAME = 'vit_large_patch16_224'
        cfg.FEAT_DIM = 1024

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs('configs', exist_ok=True)

    print('=' * 80)
    print('   RetinaSense DANN v4 — RETFound + Enhanced Training (Target: 92%+)')
    print('=' * 80)
    if torch.cuda.is_available():
        print(f'  GPU           : {torch.cuda.get_device_name(0)}')
        print(f'  VRAM          : '
              f'{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    else:
        print('  GPU           : NONE (CPU mode — training will be slow)')
    print(f'  Backbone      : {"RETFound (1.6M retinal)" if not args.no_retfound else "ImageNet (ablation)"}')
    print(f'  Image size    : {cfg.IMG_SIZE}')
    print(f'  Epochs        : {cfg.NUM_EPOCHS}  (patience={cfg.PATIENCE})')
    print(f'  Batch         : {cfg.BATCH_SIZE} '
          f'(eff. {cfg.BATCH_SIZE * cfg.GRADIENT_ACCUMULATION})')
    print(f'  Base LR       : {cfg.BASE_LR:.1e}')
    print(f'  LLRD decay    : {cfg.LLRD_DECAY}')
    print(f'  Dropout       : {args.dropout}')
    print(f'  Label smooth  : {args.label_smoothing}')
    print(f'  CutMix prob   : {args.cutmix_prob}')
    print(f'  MixUp prob    : {args.mixup_prob}')
    print(f'  MixUp alpha   : {args.mixup_alpha}')
    print(f'  SWA start     : epoch {args.swa_start} (LR={args.swa_lr})')
    print(f'  Hard mining   : top-{args.hard_mining_k} x{args.hard_mining_factor}')
    print(f'  DR alpha      : {args.dr_alpha_start} -> {args.dr_alpha_end}')
    print(f'  TTA           : {"Enabled" if args.tta else "Disabled"}')
    print('=' * 80)

    # ----------------------------------------------------------
    # 1. Normalisation stats
    # ----------------------------------------------------------
    print('\n[1/11] Loading normalisation stats...')
    NORM_MEAN, NORM_STD = load_norm_stats()

    # ----------------------------------------------------------
    # 2. Load splits
    # ----------------------------------------------------------
    print('\n[2/11] Loading data splits...')
    for csv_attr, fallback_attr, name in [
        ('TRAIN_CSV', 'TRAIN_CSV_FALLBACK', 'Train'),
        ('CALIB_CSV', 'CALIB_CSV_FALLBACK', 'Calib'),
    ]:
        primary = getattr(cfg, csv_attr)
        fallback = getattr(cfg, fallback_attr)
        if os.path.exists(primary):
            df = pd.read_csv(primary)
            print(f'  {name}: {primary} ({len(df)} samples)')
        elif os.path.exists(fallback):
            df = pd.read_csv(fallback)
            print(f'  {name}: FALLBACK {fallback} ({len(df)} samples)')
        else:
            raise FileNotFoundError(f'No {name} CSV found')
        if name == 'Train':
            train_df = df
        else:
            calib_df = df

    test_df = pd.read_csv(cfg.TEST_CSV)
    print(f'  Test : {cfg.TEST_CSV} ({len(test_df)} samples) [SEALED]')

    # ----------------------------------------------------------
    # 3. Domain mapping
    # ----------------------------------------------------------
    print('\n[3/11] Building domain mapping...')
    domain_map = {'APTOS': 0, 'ODIR': 1, 'REFUGE2': 2, 'MESSIDOR2': 3}
    all_sources = sorted(set(
        train_df['source'].str.upper().unique().tolist() +
        calib_df['source'].str.upper().unique().tolist() +
        test_df['source'].str.upper().unique().tolist()
    ))
    for src in all_sources:
        if src not in domain_map:
            domain_map[src] = len(domain_map)
    num_domains = len(domain_map)

    for src, idx in sorted(domain_map.items(), key=lambda x: x[1]):
        n = (train_df['source'].str.upper() == src).sum()
        print(f'    {idx}: {src:12s}  train={n:5d}')

    # ----------------------------------------------------------
    # 4. Class distribution
    # ----------------------------------------------------------
    print('\n[4/11] Class distribution...')
    train_labels = train_df['disease_label'].values
    class_counts = np.bincount(train_labels, minlength=5)
    for ci, cn in enumerate(cfg.CLASS_NAMES):
        pct = 100.0 * class_counts[ci] / len(train_labels)
        tag = ' [minority]' if ci in cfg.MINORITY_CLASSES else ''
        print(f'  {cn:15s}: {class_counts[ci]:5d} ({pct:5.1f}%){tag}')

    # ----------------------------------------------------------
    # 5. Datasets and loaders
    # ----------------------------------------------------------
    print('\n[5/11] Building datasets with class-aware augmentation...')

    train_tfm_majority = make_majority_transform(NORM_MEAN, NORM_STD, cfg.IMG_SIZE)
    train_tfm_minority = make_minority_transform(NORM_MEAN, NORM_STD, cfg.IMG_SIZE)
    val_tfm = make_val_transform(NORM_MEAN, NORM_STD, cfg.IMG_SIZE)

    train_ds = RetinalDANNv4Dataset(
        train_df, train_tfm_majority, train_tfm_minority, domain_map, cfg.IMG_SIZE)
    calib_ds = RetinalValDataset(calib_df, val_tfm, domain_map, cfg.IMG_SIZE)
    test_ds  = RetinalValDataset(test_df,  val_tfm, domain_map, cfg.IMG_SIZE)

    sampler = HardExampleMiningWeightedSampler(
        labels=train_labels, n_classes=5,
        hard_k=args.hard_mining_k, hard_factor=args.hard_mining_factor,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE, sampler=sampler,
        num_workers=cfg.NUM_WORKERS, pin_memory=True,
        persistent_workers=True, prefetch_factor=2,
    )
    calib_loader = DataLoader(
        calib_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=True, persistent_workers=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=True, persistent_workers=True,
    )

    print(f'  Train : {len(train_ds):5d}  ({len(train_loader):3d} batches)')
    print(f'  Calib : {len(calib_ds):5d}  ({len(calib_loader):3d} batches)')
    print(f'  Test  : {len(test_ds):5d}  ({len(test_loader):3d} batches)')

    # ----------------------------------------------------------
    # 6. Model with RETFound backbone
    # ----------------------------------------------------------
    print('\n[6/11] Building DANN v4 model...')

    # Create model WITHOUT ImageNet pretraining if using RETFound
    use_imagenet = args.no_retfound
    model = DANNMultiTaskViT(
        n_disease=5, n_severity=5, num_domains=num_domains,
        drop=args.dropout, backbone_name=cfg.MODEL_NAME,
        pretrained=use_imagenet,
    ).to(cfg.DEVICE)

    # Load RETFound weights into backbone
    if not args.no_retfound:
        print('\n  Loading RETFound backbone (1.6M retinal images)...')
        n_loaded = load_retfound_backbone(model, args.retfound_weights, cfg.DEVICE)
        print(f'  RETFound: {n_loaded} backbone params loaded')

    # Optionally warm-start heads from DANN-v3
    if args.warmstart_heads:
        print(f'\n  Warm-starting heads from {args.warmstart_heads}...')
        load_head_weights(model, args.warmstart_heads, cfg.DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'  Total params: {total_params:,}')

    # ----------------------------------------------------------
    # 7. Loss, optimizer, scheduler
    # ----------------------------------------------------------
    print('\n[7/11] Setting up losses and optimizer...')

    cw = compute_class_weight('balanced', classes=np.arange(5), y=train_labels)
    base_alpha = torch.tensor(cw, dtype=torch.float32).to(cfg.DEVICE)
    base_alpha = base_alpha / base_alpha.sum() * 5
    print(f'  Base focal alpha: {[f"{a:.2f}" for a in base_alpha.tolist()]}')

    alpha_epoch0 = base_alpha.clone()
    alpha_epoch0[1] *= args.dr_alpha_start

    criterion_d = FocalLoss(alpha=alpha_epoch0, gamma=cfg.FOCAL_GAMMA,
                            label_smoothing=args.label_smoothing)
    criterion_s = nn.CrossEntropyLoss(ignore_index=-1,
                                      label_smoothing=args.label_smoothing)
    criterion_dom = nn.CrossEntropyLoss()
    criterion_per_sample = nn.CrossEntropyLoss(reduction='none')

    optimizer = get_optimizer_with_llrd(
        model, base_lr=cfg.BASE_LR, decay_factor=cfg.LLRD_DECAY,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=cfg.BASE_LR * 0.01,
    )

    scaler = GradScaler()

    # SWA model
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=args.swa_lr)

    # ----------------------------------------------------------
    # 8. Training loop
    # ----------------------------------------------------------
    print('\n[8/11] Training DANN v4...')
    CHECKPOINT = os.path.join(cfg.OUTPUT_DIR, 'best_model.pth')

    history_keys = [
        'train_loss', 'val_loss', 'domain_loss',
        'train_acc', 'val_acc',
        'train_domain_acc', 'val_domain_acc',
        'macro_f1', 'weighted_f1', 'lr', 'lambda_p',
    ] + [f'f1_{cn}' for cn in cfg.CLASS_NAMES]
    history = {k: [] for k in history_keys}

    best_f1      = 0.0
    patience_ctr = 0
    t_start      = time.time()
    swa_active   = False

    print('=' * 90)
    print(f'{"Ep":>3} | {"Time":>4} | {"LR":>8} | {"Lam":>5} | '
          f'{"TrL":>6} {"TrA":>5} | {"VL":>6} {"VA":>5} | '
          f'{"mF1":>6} {"wF1":>6} | {"DomA":>5} | {"SWA":>3} |')
    print('-' * 90)

    for epoch in range(cfg.NUM_EPOCHS):
        t0 = time.time()
        lam_p = ganin_lambda(epoch, cfg.NUM_EPOCHS, max_lambda=args.max_lambda)
        dr_boost = compute_dr_alpha_boost(
            epoch, cfg.NUM_EPOCHS, start=args.dr_alpha_start, end=args.dr_alpha_end)

        alpha_ep = base_alpha.clone()
        alpha_ep[1] *= dr_boost
        criterion_d.alpha.copy_(alpha_ep)

        # Check if SWA should start
        if epoch >= args.swa_start and not swa_active:
            swa_active = True
            print(f'\n  >>> SWA activated at epoch {epoch+1} <<<\n')

        # ---- TRAIN ----
        model.train()
        run_loss = run_dom_loss = 0.0
        correct = dom_correct = total = 0
        epoch_indices, epoch_losses, epoch_preds, epoch_targets = [], [], [], []

        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader,
                    desc=f'E{epoch+1:03d}/{cfg.NUM_EPOCHS} train', leave=False)

        for step, (imgs, d_lbl, s_lbl, dom_lbl, sample_idx) in enumerate(pbar):
            imgs       = imgs.to(cfg.DEVICE, non_blocking=True)
            d_lbl      = d_lbl.to(cfg.DEVICE, non_blocking=True)
            s_lbl      = s_lbl.to(cfg.DEVICE, non_blocking=True)
            dom_lbl    = dom_lbl.to(cfg.DEVICE, non_blocking=True)
            sample_idx = sample_idx.to(cfg.DEVICE, non_blocking=True)

            # CutMix / MixUp / Clean
            mixed_imgs, y_a, y_b, lam = apply_batch_augmentation(
                imgs, d_lbl,
                mixup_prob=args.mixup_prob,
                cutmix_prob=args.cutmix_prob,
                alpha=args.mixup_alpha,
            )

            with autocast('cuda'):
                d_out, s_out, dom_out_mixed = model(mixed_imgs, alpha=lam_p)

                if lam < 1.0:
                    loss_d = lam * criterion_d(d_out, y_a) + (1-lam) * criterion_d(d_out, y_b)
                else:
                    loss_d = criterion_d(d_out, y_a)

                loss_s = criterion_s(s_out, s_lbl)

                # Domain loss on original images
                f_orig = model.backbone(imgs)
                f_orig = model.drop(f_orig)
                dom_out_orig = model.domain_head(model.grl(f_orig, lam_p))
                loss_dom = criterion_dom(dom_out_orig, dom_lbl)

                loss = (loss_d + 0.2 * loss_s +
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
                preds = d_out.argmax(1)
                correct     += (preds == y_a).sum().item()
                dom_correct += (dom_out_orig.argmax(1) == dom_lbl).sum().item()
                total       += d_lbl.size(0)

                per_sample_loss = criterion_per_sample(d_out.float(), d_lbl)
                epoch_indices.append(sample_idx.cpu())
                epoch_losses.append(per_sample_loss.cpu())
                epoch_preds.append(preds.cpu())
                epoch_targets.append(d_lbl.cpu())

            pbar.set_postfix(loss=f'{loss.item()*cfg.GRADIENT_ACCUMULATION:.3f}',
                             acc=f'{100*correct/total:.1f}%')

        # Flush remaining gradients
        if len(train_loader) % cfg.GRADIENT_ACCUMULATION != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # Scheduler step
        if swa_active:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step(epoch)

        # Update hard-example mining
        all_idx = torch.cat(epoch_indices)
        all_loss = torch.cat(epoch_losses)
        all_pred = torch.cat(epoch_preds)
        all_tgt  = torch.cat(epoch_targets)
        sampler.update_losses(all_idx, all_loss, preds=all_pred, targets=all_tgt)
        mining_stats = sampler.get_stats()

        train_loss    = run_loss / max(len(train_loader), 1)
        train_acc     = 100.0 * correct / max(total, 1)
        train_dom_acc = 100.0 * dom_correct / max(total, 1)

        # ---- VALIDATE ----
        val_loss, val_preds, val_targets, val_probs = evaluate(
            calib_loader, model, criterion_d, criterion_s, cfg.DEVICE,
            desc=f'E{epoch+1:03d} val')
        val_dom_acc = evaluate_domain(calib_loader, model, cfg.DEVICE, alpha=lam_p)

        val_acc = 100.0 * (val_preds == val_targets).mean()
        mf1 = f1_score(val_targets, val_preds, average='macro')
        wf1 = f1_score(val_targets, val_preds, average='weighted')
        per_f1 = f1_score(val_targets, val_preds, average=None,
                          labels=range(5), zero_division=0)

        lr_now = optimizer.param_groups[0]['lr']

        # Record history
        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        history['domain_loss'].append(float(run_dom_loss / max(len(train_loader), 1)))
        history['train_acc'].append(float(train_acc))
        history['val_acc'].append(float(val_acc))
        history['train_domain_acc'].append(float(train_dom_acc))
        history['val_domain_acc'].append(float(val_dom_acc))
        history['macro_f1'].append(float(mf1))
        history['weighted_f1'].append(float(wf1))
        history['lr'].append(float(lr_now))
        history['lambda_p'].append(float(lam_p))
        for ci, cn in enumerate(cfg.CLASS_NAMES):
            history[f'f1_{cn}'].append(float(per_f1[ci]))

        elapsed = time.time() - t0

        # Early stopping on macro-F1
        tag = ''
        if mf1 > best_f1 + cfg.MIN_DELTA:
            best_f1      = mf1
            patience_ctr = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'macro_f1': mf1,
                'domain_map': domain_map,
                'num_domains': num_domains,
                'backbone': 'retfound' if not args.no_retfound else 'imagenet',
                'args': vars(args),
            }, CHECKPOINT)
            tag = ' *BEST*'
        else:
            patience_ctr += 1

        swa_tag = 'YES' if swa_active else ' - '
        print(
            f'{epoch+1:3d} | {elapsed:4.0f}s | {lr_now:.2e} | {lam_p:.3f} | '
            f'{train_loss:6.3f} {train_acc:5.1f} | '
            f'{val_loss:6.3f} {val_acc:5.1f} | '
            f'{mf1:.4f} {wf1:.4f} | {val_dom_acc:5.1f} | {swa_tag:>3} |{tag}'
        )
        cls_str = ' | '.join(f'{cn[:3]}:{per_f1[ci]:.2f}'
                             for ci, cn in enumerate(cfg.CLASS_NAMES))
        print(f'       {cls_str}')

        if patience_ctr >= cfg.PATIENCE:
            print(f'\n  Early stopping after {cfg.PATIENCE} epochs without improvement')
            break

    total_time = time.time() - t_start
    print(f'\nTraining complete. Best macro-F1: {best_f1:.4f}')
    print(f'Total time: {total_time / 60:.1f} minutes')

    # Save history
    with open(os.path.join(cfg.OUTPUT_DIR, 'history.json'), 'w') as f:
        json.dump({k: [float(x) if isinstance(x, (float, np.floating)) else x
                       for x in v] if isinstance(v, list) else v
                   for k, v in history.items()}, f, indent=2)

    # ----------------------------------------------------------
    # 9. SWA batch norm update
    # ----------------------------------------------------------
    if swa_active:
        print('\n[9/11] Updating SWA batch norm statistics...')
        # Build a simple loader without the custom sampler for BN update
        bn_loader = DataLoader(
            train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
            num_workers=cfg.NUM_WORKERS, pin_memory=True,
        )
        update_bn(bn_loader, swa_model, device=cfg.DEVICE)
        print('  SWA BN update complete.')

        # Save SWA model
        swa_checkpoint = os.path.join(cfg.OUTPUT_DIR, 'swa_model.pth')
        torch.save({
            'model_state_dict': swa_model.module.state_dict(),
            'backbone': 'retfound' if not args.no_retfound else 'imagenet',
        }, swa_checkpoint)
        print(f'  SWA model saved to {swa_checkpoint}')
    else:
        print('\n[9/11] SWA not activated (training ended before swa_start)')

    # ----------------------------------------------------------
    # 10. Temperature scaling + thresholds
    # ----------------------------------------------------------
    print('\n[10/11] Temperature scaling and threshold optimisation...')

    # Load best checkpoint (compare regular vs SWA)
    ckpt = torch.load(CHECKPOINT, map_location=cfg.DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Evaluate both regular and SWA models to pick the better one
    eval_model = model
    eval_label = 'regular'

    if swa_active:
        # Compare SWA vs regular on calib set
        _, reg_preds, reg_tgts, _ = evaluate(
            calib_loader, model, criterion_d, criterion_s, cfg.DEVICE, 'Regular eval')
        reg_f1 = f1_score(reg_tgts, reg_preds, average='macro')

        swa_model.eval()
        _, swa_preds, swa_tgts, _ = evaluate(
            calib_loader, swa_model.module, criterion_d, criterion_s, cfg.DEVICE, 'SWA eval')
        swa_f1 = f1_score(swa_tgts, swa_preds, average='macro')

        print(f'  Regular best F1: {reg_f1:.4f}')
        print(f'  SWA model F1   : {swa_f1:.4f}')

        if swa_f1 > reg_f1:
            eval_model = swa_model.module
            eval_label = 'SWA'
            print(f'  >>> Using SWA model (better by {swa_f1-reg_f1:.4f})')
        else:
            print(f'  >>> Using regular model (better by {reg_f1-swa_f1:.4f})')

    # Calibration
    calib_logits, calib_labels = collect_logits_labels(
        calib_loader, eval_model, cfg.DEVICE)

    probs_before = torch.softmax(calib_logits, dim=1).numpy()
    ece_before = compute_ece(probs_before, calib_labels.numpy())
    print(f'  ECE before temperature scaling: {ece_before:.4f}')

    T_opt = temperature_scale(calib_logits, calib_labels)
    print(f'  Optimal temperature T = {T_opt:.4f}')

    probs_after = torch.softmax(calib_logits / T_opt, dim=1).numpy()
    ece_after = compute_ece(probs_after, calib_labels.numpy())
    print(f'  ECE after  temperature scaling: {ece_after:.4f}')

    with open('configs/temperature.json', 'w') as f:
        json.dump({'temperature': T_opt, 'ece_before': ece_before,
                   'ece_after': ece_after, 'model_version': 'dann_v4'}, f, indent=2)

    # Thresholds
    calib_thresholds = optimise_thresholds(
        probs_after, calib_labels.numpy(), 5, cfg.CLASS_NAMES)

    with open('configs/thresholds.json', 'w') as f:
        json.dump({'thresholds': calib_thresholds, 'class_names': cfg.CLASS_NAMES,
                   'model_version': 'dann_v4'}, f, indent=2)

    # ----------------------------------------------------------
    # 11. Final test evaluation
    # ----------------------------------------------------------
    print('\n[11/11] Final test set evaluation...')
    print('=' * 80)
    print('         FINAL EVALUATION — TEST SET')
    print('=' * 80)

    test_logits, test_labels = collect_logits_labels(test_loader, eval_model, cfg.DEVICE)
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
        per = f1_score(targets, preds, average=None, labels=range(5), zero_division=0)
        ece = compute_ece(probs, targets)
        kappa = cohen_kappa_score(targets, preds)
        mcc = matthews_corrcoef(targets, preds)

        print(f'\n  [{label}]')
        print(f'  Accuracy    : {acc:.2f}%')
        print(f'  Macro F1    : {mf1:.4f}')
        print(f'  Weighted F1 : {wf1:.4f}')
        print(f'  Macro AUC   : {mauc:.4f}')
        print(f'  ECE         : {ece:.4f}')
        print(f'  Kappa       : {kappa:.4f}')
        print(f'  MCC         : {mcc:.4f}')
        print()
        print(classification_report(targets, preds, target_names=cfg.CLASS_NAMES, digits=4))
        return {
            'accuracy': float(acc), 'macro_f1': float(mf1),
            'weighted_f1': float(wf1), 'macro_auc': float(mauc),
            'ece': float(ece), 'kappa': float(kappa), 'mcc': float(mcc),
            **{f'f1_{cfg.CLASS_NAMES[i]}': float(per[i]) for i in range(5)},
        }

    metrics_raw = print_metrics(test_preds_raw, test_labels_np, test_probs,
                                f'{eval_label} raw argmax (T-scaled)')
    metrics_thr = print_metrics(test_preds_thr, test_labels_np, test_probs,
                                f'{eval_label} + per-class thresholds')

    # TTA evaluation
    metrics_tta = None
    if args.tta:
        print('\n  Running TTA...')
        tta_probs, tta_targets = evaluate_with_tta(
            test_df, domain_map, eval_model, cfg.DEVICE,
            NORM_MEAN, NORM_STD, batch_size=cfg.BATCH_SIZE,
            num_workers=cfg.NUM_WORKERS, n_augs=args.tta_n, img_size=cfg.IMG_SIZE)
        tta_preds = apply_thresholds(tta_probs, calib_thresholds)
        metrics_tta = print_metrics(tta_preds, tta_targets, tta_probs,
                                    f'TTA ({args.tta_n}-way) + thresholds')

    # Domain accuracy
    test_dom_acc = evaluate_domain(test_loader, eval_model, cfg.DEVICE, alpha=1.0)
    print(f'  Test domain accuracy: {test_dom_acc:.1f}% (chance={100.0/num_domains:.1f}%)')

    # Confusion matrix
    cm = confusion_matrix(test_labels_np, test_preds_thr)
    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=cfg.CLASS_NAMES, yticklabels=cfg.CLASS_NAMES, ax=ax_cm)
    ax_cm.set_xlabel('Predicted'); ax_cm.set_ylabel('True')
    ax_cm.set_title(f'DANN v4 ({eval_label}) — Test Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, 'confusion_matrix.png'), dpi=150)
    plt.close()

    # Save final metrics
    final_metrics = {
        'raw': metrics_raw, 'thresholded': metrics_thr, 'tta': metrics_tta,
        'temperature': T_opt, 'thresholds': calib_thresholds,
        'domain_accuracy_test': test_dom_acc,
        'eval_model': eval_label,
        'backbone': 'retfound' if not args.no_retfound else 'imagenet',
        'v4_improvements': {
            'retfound_backbone': not args.no_retfound,
            'cutmix_prob': args.cutmix_prob,
            'mixup_prob': args.mixup_prob,
            'class_aware_augmentation': True,
            'swa': swa_active,
            'swa_start_epoch': args.swa_start,
        },
    }
    with open(os.path.join(cfg.OUTPUT_DIR, 'final_metrics.json'), 'w') as f:
        json.dump(final_metrics, f, indent=2)

    # Dashboard
    plot_dashboard(history, cfg, cfg.OUTPUT_DIR)

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print('\n' + '=' * 80)
    print('         RETINASENSE DANN v4 — FINAL SUMMARY')
    print('=' * 80)
    print(f'  Backbone        : {"RETFound (1.6M retinal)" if not args.no_retfound else "ImageNet"}')
    print(f'  Training epochs : {len(history["train_loss"])}')
    print(f'  Best calib F1   : {best_f1:.4f}')
    print(f'  Eval model      : {eval_label}')
    print(f'  Temperature T   : {T_opt:.4f}')
    print(f'  ECE             : {ece_before:.4f} -> {ece_after:.4f}')
    print()
    print(f'  TEST RESULTS ({eval_label} + thresholds):')
    print(f'    Accuracy   : {metrics_thr["accuracy"]:.2f}%')
    print(f'    Macro F1   : {metrics_thr["macro_f1"]:.4f}')
    print(f'    Macro AUC  : {metrics_thr["macro_auc"]:.4f}')
    print(f'    Kappa      : {metrics_thr["kappa"]:.4f}')
    print(f'    MCC        : {metrics_thr["mcc"]:.4f}')

    if metrics_tta:
        print(f'\n  TEST RESULTS (TTA + thresholds):')
        print(f'    Accuracy   : {metrics_tta["accuracy"]:.2f}%')
        print(f'    Macro F1   : {metrics_tta["macro_f1"]:.4f}')

    print(f'\n  Per-class F1 (thresholded):')
    for i, cn in enumerate(cfg.CLASS_NAMES):
        print(f'    {cn:15s}: {metrics_thr[f"f1_{cn}"]:.3f}')

    print(f'\n  Training time : {total_time/60:.1f} minutes')
    print(f'  Outputs       : {cfg.OUTPUT_DIR}/')
    print('=' * 80)


if __name__ == '__main__':
    main()
