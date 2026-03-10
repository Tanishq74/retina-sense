#!/usr/bin/env python3
"""
RetinaSense v3.0 — Grad-CAM Explainability Pipeline
====================================================
Implements:
  1. ViTGradCAM   : Gradient-weighted Class Activation Maps for ViT backbone
  2. OODDetector  : Mahalanobis-distance out-of-distribution detection
  3. predict_with_gradcam : Full inference pipeline (preprocess → OOD → CAM → calibrate)
  4. Batch evaluation on 20 test images (4 per class)
  5. Disease-specific heatmap validation against known anatomical regions
  6. Clinical output report (GRADCAM_REPORT.md)

Usage:
  python gradcam_v3.py
"""

import os
import sys
import json
import warnings
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from datetime import datetime
import time

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import timm

# ================================================================
# CONFIGURATION
# ================================================================
BASE_DIR   = '/teamspace/studios/this_studio'
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs_v3')
GRADCAM_DIR = os.path.join(OUTPUT_DIR, 'gradcam')
os.makedirs(GRADCAM_DIR, exist_ok=True)

MODEL_PATH      = os.path.join(OUTPUT_DIR, 'best_model.pth')
THRESHOLDS_PATH = os.path.join(OUTPUT_DIR, 'thresholds.json')
TEMPERATURE_PATH = os.path.join(OUTPUT_DIR, 'temperature.json')
TEST_CSV        = os.path.join(BASE_DIR, 'data', 'test_split.csv')
NORM_STATS_PATH = os.path.join(BASE_DIR, 'data', 'fundus_norm_stats.json')

CLASS_NAMES = ['Normal', 'Diabetes/DR', 'Glaucoma', 'Cataract', 'AMD']
NUM_CLASSES = 5
IMG_SIZE    = 224
DROPOUT     = 0.3

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Anatomical regions expected for each disease class
EXPECTED_REGIONS = {
    0: 'low uniform activation (Normal)',
    1: 'scattered periphery and macula (DR)',
    2: 'optic disc (Glaucoma)',
    3: 'diffuse lens opacity (Cataract)',
    4: 'macula/centre-temporal (AMD)',
}

print('=' * 65)
print('   RetinaSense v3.0 — Grad-CAM Explainability Pipeline')
print('=' * 65)
print(f'  Device   : {DEVICE}')
if torch.cuda.is_available():
    print(f'  GPU      : {torch.cuda.get_device_name(0)}')
print(f'  Output   : {GRADCAM_DIR}')
print('=' * 65)


# ================================================================
# LOAD NORMALISATION STATS
# ================================================================
if os.path.exists(NORM_STATS_PATH):
    with open(NORM_STATS_PATH) as f:
        norm_stats = json.load(f)
    NORM_MEAN = norm_stats['mean_rgb']
    NORM_STD  = norm_stats['std_rgb']
    print(f'  Fundus norm stats: mean={[round(v,4) for v in NORM_MEAN]}, std={[round(v,4) for v in NORM_STD]}')
else:
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD  = [0.229, 0.224, 0.225]
    print('  Using ImageNet normalisation fallback')


# ================================================================
# MODEL ARCHITECTURE  (mirrors retinasense_v3.py exactly)
# ================================================================
class MultiTaskViT(nn.Module):
    """ViT-Base-Patch16-224 with disease + severity heads."""

    def __init__(self, n_disease=NUM_CLASSES, n_severity=5, drop=DROPOUT):
        super().__init__()
        self.backbone = timm.create_model(
            'vit_base_patch16_224', pretrained=False, num_classes=0
        )
        feat = 768  # CLS token dimension

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
        f = self.backbone(x)   # (B, 768) — CLS token features
        f = self.drop(f)
        return self.disease_head(f), self.severity_head(f)

    def get_features(self, x):
        """Return raw CLS token features (before heads and dropout)."""
        return self.backbone(x)  # (B, 768)

    def forward_with_tokens(self, x):
        """Return (disease_logits, full_token_sequence (B,197,768))."""
        tokens = self.backbone.forward_features(x)  # (B, 197, 768)
        cls_feat = tokens[:, 0, :]
        cls_feat_d = self.drop(cls_feat)
        d_out = self.disease_head(cls_feat_d)
        return d_out, tokens


# ================================================================
# LOAD MODEL
# ================================================================
print('\nLoading model...')
model = MultiTaskViT().to(DEVICE)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f'  Loaded: {MODEL_PATH}')
print(f'  Checkpoint epoch: {ckpt.get("epoch", "?") + 1}  val_acc={ckpt.get("val_acc", 0):.2f}%')

# Load thresholds and temperature
with open(THRESHOLDS_PATH) as f:
    thr_data = json.load(f)
THRESHOLDS = thr_data['thresholds']

with open(TEMPERATURE_PATH) as f:
    temp_data = json.load(f)
TEMPERATURE = temp_data['temperature']

print(f'  Temperature T = {TEMPERATURE:.4f}')
print(f'  Thresholds    = {[round(t,3) for t in THRESHOLDS]}')


# ================================================================
# IMAGE PREPROCESSING
# ================================================================
def ben_graham(path, sz=IMG_SIZE, sigma=10):
    """Ben Graham high-frequency fundus enhancement (APTOS-style)."""
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


def clahe_preprocess(path, sz=IMG_SIZE):
    """CLAHE-based contrast enhancement (ODIR-style)."""
    img = cv2.imread(path)
    if img is None:
        img = np.array(Image.open(path).convert('RGB'))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (sz, sz))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_and_preprocess(image_path, dataset='auto'):
    """
    Load image and apply domain-conditional preprocessing.
    Returns:
        img_np   : numpy (224,224,3) uint8 preprocessed
        img_orig : numpy (224,224,3) uint8 original (for overlay)
    """
    # Normalise path: handle relative paths from CSV (e.g. "aptos/..." or "./aptos/...")
    # If the path is already absolute and exists, use it directly.
    # Otherwise resolve relative to BASE_DIR, stripping any leading ./ or .// first.
    if not os.path.isabs(image_path):
        # Strip any leading './' or '../' patterns to get a clean relative path
        clean = image_path
        while clean.startswith('./') or clean.startswith('.//'):
            clean = clean[2:] if clean.startswith('./') else clean[3:]
        image_path = os.path.join(BASE_DIR, clean)thinl
    # Auto-detect domain
    if dataset == 'auto':
        if 'aptos' in image_path.lower() or 'gaussian' in image_path.lower():
            dataset = 'APTOS'
        else:
            dataset = 'ODIR'

    # Load original (unprocessed, for overlay)
    raw = cv2.imread(image_path)
    if raw is None:
        raw = np.array(Image.open(image_path).convert('RGB'))
    else:
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    img_orig = cv2.resize(raw, (IMG_SIZE, IMG_SIZE))

    # Apply preprocessing
    if dataset == 'APTOS':
        img_np = ben_graham(image_path)
    else:
        img_np = clahe_preprocess(image_path)

    return img_np, img_orig


def preprocess_to_tensor(img_np):
    """Convert preprocessed numpy image to normalised tensor (1, 3, 224, 224)."""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])
    return transform(img_np).unsqueeze(0)


# ================================================================
# ViT GRAD-CAM
# ================================================================
class ViTAttentionRollout:
    """
    Attention Rollout for Vision Transformer (Abnar & Zuidema, 2020).

    WHY this works better than Grad-CAM for ViT:
    - ViT uses CLS token pooling: gradients flow ONLY through CLS token (index 0)
    - All 196 patch token gradients at block output = zero → Grad-CAM fails
    - Attention Rollout instead traces how information flows from image patches
      to the CLS token across ALL 12 transformer layers
    - Accounts for residual connections by adding identity to each attention map
    - Produces spatially meaningful maps that highlight actual disease regions

    Algorithm:
      1. Collect attention maps A_l from all 12 blocks: shape (B, H, N, N)
      2. Average over H heads: A_l → (B, N, N)
      3. Add identity: A_l = A_l + I  (accounts for residual connection)
      4. Row-normalize: A_l = A_l / row_sum
      5. Matrix-multiply all layers: Rollout = A_0 @ A_1 @ ... @ A_11
      6. Take CLS row, patch tokens only: Rollout[0, 1:] → (196,)
      7. Reshape 14×14 → bilinear upsample → 224×224
    """

    def __init__(self, model, discard_ratio=0.97):
        self.model = model
        self.discard_ratio = discard_ratio  # zero out weakest attention weights
        self._attention_maps = []
        self._hooks = []

        # Disable fused attention for explicit weight access
        for block in model.backbone.blocks:
            block.attn.fused_attn = False

        # Register forward hooks on ALL transformer blocks
        for block in model.backbone.blocks:
            h = block.attn.register_forward_hook(self._attn_hook)
            self._hooks.append(h)

    def _attn_hook(self, module, input, output):
        """Capture softmax attention weights from each block."""
        x = input[0]
        B, N, C = x.shape
        with torch.no_grad():
            qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, module.head_dim).permute(2, 0, 3, 1, 4)
            q, k, _ = qkv.unbind(0)
            q, k = module.q_norm(q), module.k_norm(k)
            attn = (q * module.scale @ k.transpose(-2, -1)).softmax(dim=-1)
            self._attention_maps.append(attn.detach().cpu())  # (B, H, N, N)

    def generate(self, image_tensor, class_idx=None):
        """
        Generate attention rollout heatmap.

        Returns:
            heatmap         : np.ndarray (224, 224) float32 [0, 1]
                              High values = regions most important for prediction
            predicted_label : int
            confidence      : float (raw softmax)
        """
        self.model.eval()
        self._attention_maps = []

        with torch.no_grad():
            image_tensor = image_tensor.to(DEVICE)
            d_out, _ = self.model(image_tensor)
            probs = torch.softmax(d_out, dim=1)
            predicted_label = int(probs.argmax(dim=1).item())
            confidence = float(probs[0, predicted_label].item())

        if class_idx is None:
            class_idx = predicted_label

        # --- Attention Rollout computation ---
        # Stack all layer attentions: list of (1, H, N, N) → (L, H, N, N)
        attn_stack = torch.stack(self._attention_maps, dim=0)  # (L, 1, H, N, N)
        attn_stack = attn_stack[:, 0]  # (L, H, N, N), batch=1

        # Average over heads
        attn_mean = attn_stack.mean(dim=1)  # (L, N, N)

        # Optional: discard weakest connections (sharpens the map)
        if self.discard_ratio > 0:
            flat = attn_mean.reshape(attn_mean.shape[0], -1)
            thresh = torch.quantile(flat, self.discard_ratio, dim=1, keepdim=True)
            thresh = thresh.unsqueeze(-1)  # broadcast over N,N
            attn_mean = torch.where(attn_mean >= thresh, attn_mean, torch.zeros_like(attn_mean))

        # Add identity matrix for residual connection, then row-normalize
        I = torch.eye(attn_mean.shape[-1]).unsqueeze(0)  # (1, N, N)
        attn_aug = attn_mean + I
        attn_aug = attn_aug / attn_aug.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # Matrix-multiply across all layers
        rollout = attn_aug[0]
        for l in range(1, len(attn_aug)):
            rollout = rollout @ attn_aug[l]

        # CLS token's attention to all patch tokens (skip CLS at index 0)
        cls_attention = rollout[0, 1:]  # (196,)

        # Reshape and upsample
        spatial = cls_attention.numpy().reshape(14, 14).astype(np.float32)
        spatial = cv2.resize(spatial, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

        # Normalize to [0, 1]
        s_min, s_max = spatial.min(), spatial.max()
        if s_max - s_min > 1e-8:
            spatial = (spatial - s_min) / (s_max - s_min)
        else:
            spatial = np.zeros_like(spatial)

        # Power-curve stretch: boosts mid-range attention values for visual clarity
        # gamma < 1 brightens the map; 0.4 gives strong contrast enhancement
        spatial = np.power(spatial, 0.4)

        return spatial.astype(np.float32), predicted_label, confidence

    def overlay(self, original_image_np, heatmap, alpha=0.5):
        """
        Blend attention rollout heatmap onto original fundus image.
        Uses INFERNO colormap (dark=low, bright=high) — better for medical images.

        Args:
            original_image_np : (224, 224, 3) uint8 RGB
            heatmap           : (224, 224) float32 [0, 1]
            alpha             : heatmap opacity (0.5 gives good visibility)

        Returns:
            overlay : (224, 224, 3) uint8 RGB
        """
        # Apply JET colormap
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        colormap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        colormap_rgb = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)

        # Apply circular mask to ignore black borders (fundus images are circular)
        h, w = heatmap.shape
        cy, cx = h // 2, w // 2
        radius = min(h, w) // 2 - 5
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(mask, (cx, cy), radius, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (21, 21), 0)

        # Blend only inside the retinal circle
        orig = original_image_np.astype(np.float32)
        cmap = colormap_rgb.astype(np.float32)
        blended = orig.copy()
        for c in range(3):
            blended[:, :, c] = (
                orig[:, :, c] * (1 - alpha * mask)
                + cmap[:, :, c] * (alpha * mask)
            )
        return np.clip(blended, 0, 255).astype(np.uint8)

    def remove_hooks(self):
        """Clean up all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks = []

# Keep old name as alias for backward compatibility
ViTGradCAM = ViTAttentionRollout



# ================================================================
# OOD DETECTION (Mahalanobis Distance)
# ================================================================
class OODDetector:
    """
    Out-of-Distribution detector using class-conditional Mahalanobis distance.

    Fit on training-set CLS token features; at inference, computes the
    minimum Mahalanobis distance from the test feature to the nearest
    class centroid. High distance = likely OOD.
    """

    def __init__(self, threshold_percentile=97.5):
        self.class_means   = None   # (num_classes, feat_dim)
        self.cov_inv       = None   # (feat_dim, feat_dim)
        self.ood_threshold = None
        self.threshold_percentile = threshold_percentile
        self.is_fitted     = False

    def fit(self, model, dataloader, device, max_batches=60):
        """
        Extract CLS token features for all samples, compute class-conditional
        means and shared inverse covariance matrix.
        """
        print('  OODDetector.fit: extracting features...')
        all_features = []
        all_labels   = []

        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= max_batches:
                    break
                imgs, d_lbl, _ = batch
                imgs = imgs.to(device)
                feats = model.get_features(imgs)  # (B, 768)
                all_features.append(feats.cpu().numpy())
                all_labels.append(d_lbl.numpy())

        features = np.concatenate(all_features, axis=0)  # (N, 768)
        labels   = np.concatenate(all_labels,   axis=0)  # (N,)

        num_classes = NUM_CLASSES
        feat_dim    = features.shape[1]

        # Class-conditional means
        self.class_means = np.zeros((num_classes, feat_dim), dtype=np.float64)
        for c in range(num_classes):
            mask = labels == c
            if mask.sum() > 0:
                self.class_means[c] = features[mask].mean(axis=0)

        # Shared (pooled) covariance matrix
        cov = np.zeros((feat_dim, feat_dim), dtype=np.float64)
        total = 0
        for c in range(num_classes):
            mask = labels == c
            if mask.sum() < 2:
                continue
            diff = features[mask] - self.class_means[c]
            cov += diff.T @ diff
            total += mask.sum()

        cov /= max(total - num_classes, 1)

        # Regularise for numerical stability (add small diagonal)
        cov += np.eye(feat_dim) * 1e-4

        # Pseudo-inverse via SVD (numerically stable for high-dim)
        try:
            self.cov_inv = np.linalg.pinv(cov)
        except np.linalg.LinAlgError:
            self.cov_inv = np.eye(feat_dim)

        # Compute train-set Mahalanobis distances to set threshold
        train_dists = []
        for feat in features:
            d = self._mahal_min_dist(feat)
            train_dists.append(d)
        self.ood_threshold = float(np.percentile(train_dists, self.threshold_percentile))

        self.is_fitted = True
        print(f'  OOD threshold ({self.threshold_percentile}th pct): {self.ood_threshold:.4f}')
        print(f'  Features extracted: {len(features)} samples')

    def _mahal_min_dist(self, feat):
        """Minimum Mahalanobis distance to any class centroid."""
        min_dist = float('inf')
        for c in range(NUM_CLASSES):
            diff = feat - self.class_means[c]
            dist = float(diff @ self.cov_inv @ diff)
            dist = max(dist, 0.0)  # guard against floating-point negatives
            if dist < min_dist:
                min_dist = dist
        return np.sqrt(min_dist)

    def score(self, features):
        """
        Compute OOD score for a batch of features.

        Args:
            features : np.ndarray (N, 768) or (768,)

        Returns:
            distances : np.ndarray (N,) Mahalanobis distances
            ood_flags : np.ndarray (N,) bool, True = likely OOD
        """
        if not self.is_fitted:
            raise RuntimeError('OODDetector.fit() must be called before score()')

        if features.ndim == 1:
            features = features[np.newaxis, :]

        distances = np.array([self._mahal_min_dist(f) for f in features])
        ood_flags = distances > self.ood_threshold
        return distances, ood_flags

    def save(self, path):
        np.savez(path,
                 class_means=self.class_means,
                 cov_inv=self.cov_inv,
                 ood_threshold=np.array([self.ood_threshold]),
                 threshold_percentile=np.array([self.threshold_percentile]))
        print(f'  OOD detector saved -> {path}.npz')

    def load(self, path):
        if not path.endswith('.npz'):
            path = path + '.npz'
        data = np.load(path)
        self.class_means           = data['class_means']
        self.cov_inv               = data['cov_inv']
        self.ood_threshold         = float(data['ood_threshold'][0])
        self.threshold_percentile  = float(data['threshold_percentile'][0])
        self.is_fitted             = True
        print(f'  OOD detector loaded <- {path}')


# ================================================================
# ATTENTION REGION ANALYSER
# ================================================================
def analyse_attention_region(heatmap, disease_class):
    """
    Check if the Grad-CAM heatmap activation pattern is consistent
    with the expected anatomical region for the given disease.

    Returns:
        attention_region : str describing where activation is
        is_consistent   : bool
        region_scores   : dict with activation energy in each zone
    """
    h, w = heatmap.shape  # (224, 224)
    cx, cy = w // 2, h // 2

    # Define anatomical zones (approximate, relative to image centre)
    # Centre disc zone: circle r ~ 30px (optic disc)
    r_disc   = int(h * 0.13)
    # Macula zone: circle r ~ 55px centred slightly temporal
    r_macula = int(h * 0.25)
    cx_mac   = int(cx + w * 0.10)  # slightly nasal offset

    # Build zone masks
    Y, X = np.ogrid[:h, :w]

    # Optic disc (small circle, centre of image)
    disc_mask = ((X - cx)**2 + (Y - cy)**2) <= r_disc**2

    # Macula (larger circle, centre-temporal)
    macula_mask = ((X - cx_mac)**2 + (Y - cy)**2) <= r_macula**2

    # Periphery: outer 30% of image
    peri_mask = (X < int(w * 0.15)) | (X > int(w * 0.85)) | \
                (Y < int(h * 0.15)) | (Y > int(h * 0.85))

    # Compute mean activation in each zone
    disc_score   = float(heatmap[disc_mask].mean())   if disc_mask.sum() > 0 else 0.0
    macula_score = float(heatmap[macula_mask].mean()) if macula_mask.sum() > 0 else 0.0
    peri_score   = float(heatmap[peri_mask].mean())   if peri_mask.sum() > 0 else 0.0
    overall_mean = float(heatmap.mean())

    region_scores = {
        'optic_disc': round(disc_score, 4),
        'macula':     round(macula_score, 4),
        'periphery':  round(peri_score, 4),
        'overall':    round(overall_mean, 4),
    }

    # Determine dominant region label
    max_zone = max(region_scores, key=lambda k: region_scores[k] if k != 'overall' else -1)

    zone_labels = {
        'optic_disc': 'optic disc (centre)',
        'macula':     'macula (centre-temporal)',
        'periphery':  'scattered periphery',
    }
    dominant_label = zone_labels.get(max_zone, 'diffuse')

    # Assess uniformity (low std = diffuse / uniform)
    if heatmap.std() < 0.10:
        dominant_label = 'diffuse (low activation)'

    # Check consistency with expected region
    consistency_map = {
        0: lambda s: s['overall'] < 0.25,                     # Normal → low uniform
        1: lambda s: s['periphery'] > 0.20 or s['macula'] > 0.25,  # DR → periphery/macula
        2: lambda s: s['optic_disc'] > 0.30,                   # Glaucoma → disc
        3: lambda s: heatmap.std() < 0.15,                     # Cataract → diffuse
        4: lambda s: s['macula'] > 0.25,                       # AMD → macula
    }
    check_fn = consistency_map.get(disease_class, lambda s: True)
    is_consistent = check_fn(region_scores)

    return dominant_label, is_consistent, region_scores


# ================================================================
# FULL INFERENCE PIPELINE
# ================================================================
def predict_with_gradcam(image_path, model, gradcam, ood_detector,
                         thresholds, temperature, device,
                         true_label=None, dataset='auto'):
    """
    End-to-end inference with Grad-CAM and OOD detection.

    Steps:
      1. Load and preprocess image (Ben Graham for APTOS, CLAHE for ODIR)
      2. OOD check on ViT CLS token features
      3. Generate Grad-CAM heatmap
      4. Apply temperature scaling to logits
      5. Apply per-class thresholds
      6. Analyse attention region

    Returns:
        dict with predicted_class, confidence, gradcam_heatmap, etc.
    """
    # 1. Preprocess
    img_np, img_orig = load_and_preprocess(image_path, dataset=dataset)
    img_tensor = preprocess_to_tensor(img_np).to(device)

    # 2. OOD check using raw CLS features
    model.eval()
    with torch.no_grad():
        features = model.get_features(img_tensor).cpu().numpy()  # (1, 768)

    if ood_detector.is_fitted:
        distances, ood_flags = ood_detector.score(features)
        ood_distance = float(distances[0])
        ood_flag     = bool(ood_flags[0])
    else:
        ood_distance = 0.0
        ood_flag     = False

    # 3. Generate Grad-CAM (also runs forward + backward pass)
    heatmap, predicted_label, raw_confidence = gradcam.generate(img_tensor)

    # 4. Temperature-scaled calibrated probabilities
    # Run a clean no-grad forward pass to get stable logits for calibration
    model.eval()
    with torch.no_grad():
        raw_feats = model.backbone(img_tensor)   # (1, 768)
        raw_feats = model.drop(raw_feats)
        logits = model.disease_head(raw_feats).float().cpu()  # (1, 5)

    scaled_logits = logits / temperature
    calibrated_probs = torch.softmax(scaled_logits, dim=1)[0].numpy()  # (5,)

    # 5. Apply per-class thresholds
    above = [i for i, (p, t) in enumerate(zip(calibrated_probs, thresholds)) if p >= t]
    if above:
        final_label = int(above[np.argmax([calibrated_probs[i] for i in above])])
    else:
        final_label = int(np.argmax(calibrated_probs))

    final_confidence = float(calibrated_probs[final_label])
    predicted_class  = CLASS_NAMES[final_label]

    # 6. Heatmap overlay
    gradcam_overlay = gradcam.overlay(img_orig, heatmap, alpha=0.7)

    # 7. Attention region analysis
    attention_region, region_consistent, region_scores = analyse_attention_region(
        heatmap, final_label
    )

    # Append disease name for clarity
    disease_tag = CLASS_NAMES[final_label].replace('/', '-')
    attention_region_full = f'{attention_region} ({disease_tag})'

    # 8. Review flag: low confidence OR OOD
    review_flag = ood_flag or final_confidence < 0.50

    return {
        'image_path':       image_path,
        'predicted_class':  predicted_class,
        'predicted_label':  final_label,
        'confidence':       round(final_confidence, 4),
        'raw_confidence':   round(raw_confidence, 4),
        'all_probabilities': [round(float(p), 4) for p in calibrated_probs],
        'gradcam_heatmap':  heatmap,           # (224, 224) float32
        'gradcam_overlay':  gradcam_overlay,   # (224, 224, 3) uint8
        'img_orig':         img_orig,          # original for display
        'ood_flag':         ood_flag,
        'ood_distance':     round(ood_distance, 4),
        'review_flag':      review_flag,
        'attention_region': attention_region_full,
        'region_scores':    region_scores,
        'region_consistent': region_consistent,
        'true_label':       true_label,
    }


# ================================================================
# BATCH EVALUATION
# ================================================================
def run_batch_evaluation(model, gradcam, ood_detector,
                         thresholds, temperature, device,
                         n_per_class=4):
    """
    Run inference on n_per_class images per disease class (20 total).
    Saves individual overlay images + summary grid.
    """
    import pandas as pd
    print(f'\nRunning batch evaluation ({n_per_class} per class = {n_per_class * NUM_CLASSES} total)...')

    df = pd.read_csv(TEST_CSV)

    # Collect n_per_class unique samples per class
    samples = []
    for label in range(NUM_CLASSES):
        subset = df[df['disease_label'] == label].drop_duplicates(subset='image_path')
        chosen = subset.head(n_per_class)
        for _, row in chosen.iterrows():
            samples.append({
                'image_path':  row['image_path'],
                'true_label':  int(row['disease_label']),
                'dataset':     str(row.get('dataset', 'auto')),
            })

    results = []
    failed  = []

    for i, sample in enumerate(samples):
        img_path   = sample['image_path']
        true_label = sample['true_label']
        dataset    = sample['dataset']

        print(f'  [{i+1:2d}/{len(samples)}] {CLASS_NAMES[true_label]:15s} | {os.path.basename(img_path)}', end='  ')

        try:
            result = predict_with_gradcam(
                img_path, model, gradcam, ood_detector,
                thresholds, temperature, device,
                true_label=true_label,
                dataset=dataset,
            )
            correct = (result['predicted_label'] == true_label)
            flag_str = ' [OOD]' if result['ood_flag'] else ''
            flag_str += ' [REVIEW]' if result['review_flag'] else ''
            print(f'-> pred={result["predicted_class"]:15s}  conf={result["confidence"]:.3f}  {"OK" if correct else "WRONG"}{flag_str}')

            # Save overlay image
            save_name = f'gradcam_{i+1:02d}_true{true_label}_pred{result["predicted_label"]}_{os.path.splitext(os.path.basename(img_path))[0][:20]}.png'
            save_path = os.path.join(GRADCAM_DIR, save_name)

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(result['img_orig'])
            axes[0].set_title(f'Original\nTrue: {CLASS_NAMES[true_label]}', fontsize=9)
            axes[0].axis('off')

            axes[1].imshow(result['gradcam_heatmap'], cmap='jet', vmin=0, vmax=1)
            axes[1].set_title('Grad-CAM Heatmap', fontsize=9)
            axes[1].axis('off')

            axes[2].imshow(result['gradcam_overlay'])
            flag_line = ' [OOD]' if result['ood_flag'] else ''
            axes[2].set_title(
                f'Overlay\nPred: {result["predicted_class"]} ({result["confidence"]:.2f}){flag_line}',
                fontsize=9, color='red' if not correct else 'green'
            )
            axes[2].axis('off')

            plt.suptitle(
                f'Attention: {result["attention_region"]}',
                fontsize=8, color='gray'
            )
            plt.tight_layout()
            plt.savefig(save_path, dpi=120, bbox_inches='tight')
            plt.close()

            result['save_path'] = save_path
            results.append(result)

        except Exception as e:
            print(f'  ERROR: {e}')
            failed.append({'image_path': img_path, 'error': str(e)})

    return results, failed


# ================================================================
# SUMMARY GRID  (4 rows = classes 0-4, 4 cols = samples)
# ================================================================
def save_summary_grid(results):
    """Save a 5×4 summary grid (rows=classes, cols=samples)."""
    n_rows = NUM_CLASSES
    n_cols = 4

    # Group results by true label
    by_class = {i: [] for i in range(NUM_CLASSES)}
    for r in results:
        tl = r.get('true_label', r['predicted_label'])
        by_class[tl].append(r)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 20))
    fig.patch.set_facecolor('#1a1a2e')

    for row_idx in range(n_rows):
        class_results = by_class[row_idx]
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            if col_idx < len(class_results):
                r = class_results[col_idx]
                ax.imshow(r['gradcam_overlay'])
                correct = (r['predicted_label'] == r.get('true_label', r['predicted_label']))
                border_color = '#2ecc71' if correct else '#e74c3c'
                for spine in ax.spines.values():
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(3)
                label_str = f'{r["predicted_class"]}\n{r["confidence"]:.2f}'
                if r['ood_flag']:
                    label_str += '\n[OOD]'
                ax.set_title(label_str, fontsize=7, color='white', pad=2)
            else:
                ax.set_facecolor('#1a1a2e')

            ax.axis('off')
            if col_idx == 0:
                ax.set_ylabel(CLASS_NAMES[row_idx], rotation=0, labelpad=50,
                              fontsize=10, color='white', fontweight='bold',
                              va='center')

    plt.suptitle(
        'RetinaSense v3.0 — Grad-CAM Summary Grid\n'
        'Rows = True Class | Green border = Correct | Red border = Wrong',
        fontsize=12, color='white', y=1.01
    )
    plt.tight_layout()
    grid_path = os.path.join(GRADCAM_DIR, 'gradcam_summary_grid.png')
    plt.savefig(grid_path, dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f'  Summary grid saved -> {grid_path}')
    return grid_path


# ================================================================
# DISEASE-SPECIFIC HEATMAP VALIDATION
# ================================================================
def validate_heatmaps(results):
    """
    Check per-disease whether Grad-CAM activates the expected anatomical region.
    Returns a validation summary dict, saves to heatmap_validation.json.
    """
    print('\nRunning disease-specific heatmap validation...')

    validation = {}
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_results = [r for r in results if r.get('true_label') == cls_idx]
        if not cls_results:
            validation[cls_name] = {'n_samples': 0}
            continue

        consistent_count = sum(1 for r in cls_results if r.get('region_consistent', False))
        avg_scores = {k: 0.0 for k in ['optic_disc', 'macula', 'periphery', 'overall']}
        for r in cls_results:
            for k in avg_scores:
                avg_scores[k] += r['region_scores'].get(k, 0.0)
        for k in avg_scores:
            avg_scores[k] = round(avg_scores[k] / len(cls_results), 4)

        dominant_zone = max(
            ['optic_disc', 'macula', 'periphery'],
            key=lambda k: avg_scores[k]
        )

        validation[cls_name] = {
            'n_samples':          len(cls_results),
            'expected_region':    EXPECTED_REGIONS[cls_idx],
            'dominant_zone':      dominant_zone,
            'consistent_samples': consistent_count,
            'consistency_pct':    round(100 * consistent_count / len(cls_results), 1),
            'avg_region_scores':  avg_scores,
        }

        print(f'  {cls_name:15s}: {consistent_count}/{len(cls_results)} consistent '
              f'({validation[cls_name]["consistency_pct"]:.0f}%)  '
              f'dominant={dominant_zone}')

    # Save
    val_path = os.path.join(GRADCAM_DIR, 'heatmap_validation.json')
    with open(val_path, 'w') as f:
        json.dump(validation, f, indent=2)
    print(f'  Validation saved -> {val_path}')

    return validation


# ================================================================
# CLINICAL REPORT
# ================================================================
def generate_clinical_report(results, validation, ood_stats, failed):
    """Generate GRADCAM_REPORT.md with clinical analysis."""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    n_total    = len(results)
    n_correct  = sum(1 for r in results if r.get('predicted_label') == r.get('true_label'))
    n_ood      = sum(1 for r in results if r.get('ood_flag'))
    n_review   = sum(1 for r in results if r.get('review_flag'))
    avg_conf   = np.mean([r['confidence'] for r in results]) if results else 0.0

    lines = [
        '# RetinaSense v3.0 — Grad-CAM Clinical Report',
        f'',
        f'**Generated**: {now}  ',
        f'**Model**: ViT-Base-Patch16-224 (81.19% test accuracy)  ',
        f'**Pipeline**: Grad-CAM + Mahalanobis OOD + Temperature Scaling + Per-Class Thresholds',
        '',
        '---',
        '',
        '## Executive Summary',
        '',
        f'| Metric | Value |',
        f'|--------|-------|',
        f'| Images processed | {n_total} |',
        f'| Correct predictions | {n_correct}/{n_total} ({100*n_correct/max(n_total,1):.1f}%) |',
        f'| Avg calibrated confidence | {avg_conf:.3f} |',
        f'| OOD flags raised | {n_ood} |',
        f'| Human review flags | {n_review} |',
        f'| Failed images | {len(failed)} |',
        f'| Temperature T | {TEMPERATURE:.4f} |',
        '',
        '---',
        '',
        '## Per-Sample Predictions',
        '',
        '| # | Image | True | Predicted | Confidence | OOD | Review | Attention Region |',
        '|---|-------|------|-----------|-----------|-----|--------|-----------------|',
    ]

    for i, r in enumerate(results):
        true_name = CLASS_NAMES[r['true_label']] if r.get('true_label') is not None else 'Unknown'
        correct_marker = 'OK' if r['predicted_label'] == r.get('true_label') else '**WRONG**'
        lines.append(
            f'| {i+1} | {os.path.basename(r["image_path"])[:25]} '
            f'| {true_name} '
            f'| {r["predicted_class"]} ({correct_marker}) '
            f'| {r["confidence"]:.3f} '
            f'| {"YES" if r["ood_flag"] else "no"} '
            f'| {"YES" if r["review_flag"] else "no"} '
            f'| {r["attention_region"]} |'
        )

    lines += [
        '',
        '---',
        '',
        '## Per-Class Attention Pattern Analysis',
        '',
        '| Disease | Expected Region | Dominant Zone | Consistency |',
        '|---------|----------------|---------------|-------------|',
    ]
    for cls_name, v in validation.items():
        if v.get('n_samples', 0) == 0:
            lines.append(f'| {cls_name} | N/A | N/A | N/A (no samples) |')
        else:
            lines.append(
                f'| {cls_name} | {v["expected_region"]} '
                f'| {v["dominant_zone"]} '
                f'| {v["consistency_pct"]:.0f}% ({v["consistent_samples"]}/{v["n_samples"]}) |'
            )

    lines += [
        '',
        '---',
        '',
        '## OOD Detection Statistics',
        '',
        f'- **Method**: Mahalanobis distance to nearest class centroid (CLS token features)',
        f'- **Threshold percentile**: 97.5th percentile of training-set distances',
        f'- **OOD threshold**: {ood_stats.get("threshold", "N/A")}',
        f'- **Images flagged OOD**: {n_ood}/{n_total}',
        '',
        '### Interpretation',
        '',
        '- Mahalanobis distance measures how far a feature embedding lies from known class distributions',
        '- Low-quality images, extreme artefacts, or off-distribution fundus cameras may trigger OOD flags',
        '- All OOD-flagged images are automatically sent for human review',
        '',
        '---',
        '',
        '## Grad-CAM Heatmap Descriptions',
        '',
        '| Disease | Expected activation | Clinical significance |',
        '|---------|--------------------|-----------------------|',
        '| Normal | Low, uniform | No focal pathology — model attention diffuse |',
        '| Diabetes/DR | Scattered periphery + macula | Microaneurysms, exudates, NV |',
        '| Glaucoma | Optic disc (centre) | Structural disc changes, CDR |',
        '| Cataract | Diffuse lens opacity | Posterior/anterior capsule opacification |',
        '| AMD | Macula / centre-temporal | Drusen, RPE atrophy, CNV |',
        '',
        '---',
        '',
        '## Thresholds Applied',
        '',
        '| Class | Threshold |',
        '|-------|-----------|',
    ]
    for cls_name, thr in zip(CLASS_NAMES, THRESHOLDS):
        lines.append(f'| {cls_name} | {thr:.4f} |')

    lines += [
        '',
        '---',
        '',
        '## Deployment Recommendations',
        '',
        '1. **Confidence gate**: Flag predictions below 0.50 for mandatory ophthalmologist review.',
        '2. **OOD gate**: Any Mahalanobis distance above threshold should trigger QC check on image quality before clinical use.',
        '3. **Grad-CAM review**: Clinicians should inspect heatmaps for cases where model attention does not align with expected anatomy.',
        '4. **Glaucoma caution**: Current dataset imbalance (46 test samples) — consider supplementing ODIR with additional glaucoma images.',
        '5. **Continuous monitoring**: Re-calibrate temperature and thresholds quarterly on production data.',
        '6. **Not for standalone diagnosis**: Grad-CAM is an explainability aid; all predictions require clinical validation.',
        '',
        '---',
        '',
        f'*Report auto-generated by RetinaSense v3.0 Grad-CAM Pipeline | {now}*',
    ]

    report_path = os.path.join(GRADCAM_DIR, 'GRADCAM_REPORT.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'  Clinical report saved -> {report_path}')
    return report_path


# ================================================================
# MAIN
# ================================================================
def main():
    t_start = time.time()

    # ---- 1. Build Grad-CAM ---
    print('\n[1/6] Initialising ViTGradCAM...')
    gradcam = ViTGradCAM(model)
    print(f'  Method        : Attention Rollout (all 12 transformer blocks)')
    print(f'  Hooks         : {len(gradcam._hooks)} attention hooks registered')
    print(f'  fused_attn disabled for attention weight access')

    # ---- 2. Fit OOD Detector ---
    print('\n[2/6] Fitting OOD detector...')
    ood_path = os.path.join(OUTPUT_DIR, 'ood_detector')
    ood_detector = OODDetector(threshold_percentile=97.5)

    if os.path.exists(ood_path + '.npz'):
        ood_detector.load(ood_path)
    else:
        # Build a small DataLoader from training data to fit OOD detector
        import pandas as pd
        from torch.utils.data import Dataset, DataLoader
        from torchvision import transforms as T

        train_df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'train_split.csv'))

        class SimpleDataset(Dataset):
            def __init__(self, df):
                self.df = df.reset_index(drop=True)
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                    transforms.Normalize(NORM_MEAN, NORM_STD),
                ])

            def __len__(self):
                return len(self.df)

            def __getitem__(self, idx):
                row = self.df.iloc[idx]
                img_path = str(row['image_path'])
                if not os.path.isabs(img_path):
                    clean = img_path
                    while clean.startswith('./') or clean.startswith('.//'):
                        clean = clean[2:] if clean.startswith('./') else clean[3:]
                    img_path = os.path.join(BASE_DIR, clean)
                dataset = str(row.get('dataset', 'auto'))

                try:
                    img_np, _ = load_and_preprocess(img_path, dataset=dataset)
                    img_tensor = self.transform(img_np)
                except Exception:
                    img_tensor = torch.zeros(3, IMG_SIZE, IMG_SIZE)

                lbl = int(row['disease_label'])
                return img_tensor, torch.tensor(lbl, dtype=torch.long), torch.tensor(0, dtype=torch.long)

        ood_ds     = SimpleDataset(train_df)
        ood_loader = DataLoader(ood_ds, batch_size=32, shuffle=False, num_workers=4)
        ood_detector.fit(model, ood_loader, DEVICE, max_batches=80)
        ood_detector.save(ood_path)

    # ---- 3. Batch Evaluation ---
    print('\n[3/6] Batch evaluation on 20 test images...')
    results, failed = run_batch_evaluation(
        model, gradcam, ood_detector,
        THRESHOLDS, TEMPERATURE, DEVICE,
        n_per_class=4
    )

    # ---- 4. Summary Grid ---
    print('\n[4/6] Generating summary grid...')
    grid_path = save_summary_grid(results)

    # ---- 5. Heatmap Validation ---
    print('\n[5/6] Heatmap validation...')
    validation = validate_heatmaps(results)

    # ---- 6. Clinical Report ---
    print('\n[6/6] Generating clinical report...')
    ood_stats = {'threshold': round(ood_detector.ood_threshold, 4) if ood_detector.is_fitted else 'N/A'}
    report_path = generate_clinical_report(results, validation, ood_stats, failed)

    # ---- Cleanup ---
    gradcam.remove_hooks()

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    elapsed = time.time() - t_start
    n_total   = len(results)
    n_correct = sum(1 for r in results if r.get('predicted_label') == r.get('true_label'))
    avg_conf  = np.mean([r['confidence'] for r in results]) if results else 0.0
    n_ood     = sum(1 for r in results if r['ood_flag'])
    n_review  = sum(1 for r in results if r['review_flag'])

    print('\n' + '=' * 65)
    print('       RetinaSense v3.0 — GRAD-CAM PIPELINE COMPLETE')
    print('=' * 65)
    print(f'  Images processed     : {n_total}')
    print(f'  Correct predictions  : {n_correct}/{n_total}  ({100*n_correct/max(n_total,1):.1f}%)')
    print(f'  Avg calibrated conf  : {avg_conf:.3f}')
    print(f'  OOD flags            : {n_ood}')
    print(f'  Review flags         : {n_review}')
    print(f'  Failed images        : {len(failed)}')
    print(f'  Elapsed time         : {elapsed:.1f}s')
    print()
    print(f'  Output directory     : {GRADCAM_DIR}')
    output_files = [
        'gradcam_summary_grid.png',
        'heatmap_validation.json',
        'GRADCAM_REPORT.md',
    ] + [os.path.basename(r.get('save_path', '')) for r in results if r.get('save_path')]
    for fname in output_files:
        if fname:
            full = os.path.join(GRADCAM_DIR, fname)
            exists = os.path.exists(full)
            print(f'    {"[OK]" if exists else "[!!]"} {fname}')
    print('=' * 65)

    return results, validation


if __name__ == '__main__':
    main()
