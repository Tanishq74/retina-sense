#!/usr/bin/env python3
"""
RetinaSense v3.0 -- MC Dropout Uncertainty Quantification (Phase 1B)
====================================================================
Performs Monte Carlo Dropout inference on the test set to decompose
predictive uncertainty into aleatoric and epistemic components.

Strategy for efficiency:
  - Run the ViT backbone ONCE per image (deterministic, no dropout in backbone)
  - Cache the 768-dim CLS features
  - Run T=30 stochastic forward passes through the classification heads only
    (where the dropout layers live: self.drop + head dropouts)
  This is 30x faster than running the full model T times.

For each test image, computes:
  - Predictive entropy (total uncertainty)
  - Expected entropy (aleatoric uncertainty)
  - Mutual information (epistemic uncertainty)
  - Per-class prediction variance

Generates:
  - uncertainty_vs_accuracy.png
  - rejection_curve.png
  - epistemic_vs_aleatoric.png
  - uncertainty_by_class.png
  - confidence_vs_uncertainty.png
  - mc_dropout_results.json

Usage:
  python mc_dropout_uncertainty.py
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import timm

# Maximize CPU throughput
torch.set_num_threads(4)

# ================================================================
# CONFIGURATION
# ================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(BASE_DIR, 'outputs_v3')
UNCERT_DIR  = os.path.join(OUTPUT_DIR, 'uncertainty')
os.makedirs(UNCERT_DIR, exist_ok=True)

MODEL_PATH = next(
    (p for p in [
        os.path.join(OUTPUT_DIR, 'dann_v3', 'best_model.pth'),
        os.path.join(OUTPUT_DIR, 'dann_v2', 'best_model.pth'),
        os.path.join(OUTPUT_DIR, 'dann', 'best_model.pth'),
        os.path.join(OUTPUT_DIR, 'best_model.pth'),
    ] if os.path.exists(p)),
    os.path.join(OUTPUT_DIR, 'best_model.pth')
)
TEMPERATURE_PATH = os.path.join(BASE_DIR, 'configs', 'temperature.json')
NORM_STATS_PATH  = os.path.join(BASE_DIR, 'configs', 'fundus_norm_stats_unified.json')
if not os.path.exists(NORM_STATS_PATH):
    NORM_STATS_PATH = os.path.join(BASE_DIR, 'data', 'fundus_norm_stats.json')
TEST_CSV         = os.path.join(BASE_DIR, 'data', 'test_split.csv')

CLASS_NAMES = ['Normal', 'Diabetes/DR', 'Glaucoma', 'Cataract', 'AMD']
NUM_CLASSES = 5
IMG_SIZE    = 224
DROPOUT     = 0.3

T_FORWARD_PASSES = 30   # number of MC stochastic forward passes
BATCH_SIZE       = 32    # batch size for feature extraction
HEAD_BATCH       = 512   # batch size for head-only MC passes (very lightweight)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('=' * 65)
print('  RetinaSense v3.0 -- MC Dropout Uncertainty Quantification')
print('=' * 65)
print(f'  Device          : {DEVICE}')
if torch.cuda.is_available():
    print(f'  GPU             : {torch.cuda.get_device_name(0)}')
print(f'  MC passes (T)   : {T_FORWARD_PASSES}')
print(f'  Output dir      : {UNCERT_DIR}')

# ================================================================
# LOAD NORMALISATION STATS
# ================================================================
if os.path.exists(NORM_STATS_PATH):
    with open(NORM_STATS_PATH) as f:
        norm_stats = json.load(f)
    NORM_MEAN = norm_stats['mean_rgb']
    NORM_STD  = norm_stats['std_rgb']
    print(f'  Fundus norm     : mean={[round(v,4) for v in NORM_MEAN]}, '
          f'std={[round(v,4) for v in NORM_STD]}')
else:
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD  = [0.229, 0.224, 0.225]
    print('  Using ImageNet normalisation fallback')

# Load temperature
with open(TEMPERATURE_PATH) as f:
    temp_data = json.load(f)
TEMPERATURE = temp_data['temperature']
print(f'  Temperature     : {TEMPERATURE:.4f}')

# ================================================================
# MODEL ARCHITECTURE (mirrors retinasense_v3.py / gradcam_v3.py)
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
        f = self.backbone(x)   # (B, 768)
        f = self.drop(f)
        return self.disease_head(f), self.severity_head(f)

    def extract_features(self, x):
        """Run backbone only (deterministic) to get CLS features."""
        return self.backbone(x)   # (B, 768)

    def forward_heads(self, features):
        """Run dropout + disease head on pre-extracted features."""
        f = self.drop(features)
        return self.disease_head(f)


# ================================================================
# LOAD MODEL
# ================================================================
print('\nLoading model...')
model = MultiTaskViT().to(DEVICE)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
state_dict = ckpt['model_state_dict']
filtered = {k: v for k, v in state_dict.items()
            if not k.startswith('domain_head') and not k.startswith('grl')}
model.load_state_dict(filtered, strict=False)
print(f'  Loaded: {MODEL_PATH}')
print(f'  Checkpoint epoch: {ckpt.get("epoch", "?") + 1}  '
      f'val_acc={ckpt.get("val_acc", 0):.2f}%')


# ================================================================
# MC DROPOUT SETUP
# ================================================================
def enable_head_dropout(model):
    """
    Set model to eval mode, then enable dropout ONLY in the classification
    heads (self.drop, disease_head dropouts). The backbone stays fully
    deterministic (eval mode) so we only need one backbone pass per image.
    BatchNorm layers remain in eval mode (use running stats).
    """
    model.eval()  # everything to eval (including backbone)

    # Enable dropout in the drop layer and disease_head
    model.drop.train()
    for m in model.disease_head.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()


enable_head_dropout(model)

# Count active dropout layers
n_dropout_active = 0
for name, m in model.named_modules():
    if isinstance(m, (nn.Dropout, nn.Dropout2d)) and m.training:
        n_dropout_active += 1
n_dropout_total = sum(1 for m in model.modules() if isinstance(m, (nn.Dropout, nn.Dropout2d)))
print(f'\n  MC Dropout enabled in heads: {n_dropout_active} active / {n_dropout_total} total dropout layers')
print(f'  Backbone: deterministic (eval mode) -- single pass per image')
print(f'  Heads: stochastic (train mode dropout) -- {T_FORWARD_PASSES} passes per image')


# ================================================================
# PREPROCESSING (matches gradcam_v3.py pipeline)
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


def resolve_path(image_path):
    """Resolve image path relative to BASE_DIR."""
    if os.path.isabs(image_path) and os.path.exists(image_path):
        return image_path
    clean = image_path
    while clean.startswith('./'):
        clean = clean[2:]
    return os.path.join(BASE_DIR, clean)


# ================================================================
# DATASET
# ================================================================
class TestDataset(Dataset):
    """Test dataset loading preprocessed images from cache or live."""

    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
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
        dataset  = str(row.get('source', 'auto'))
        label    = int(row['disease_label'])

        # Try loading from cache first (prefer unified cache)
        cache_path = str(row.get('cache_path', ''))
        if cache_path and cache_path != 'nan':
            cache_abs = resolve_path(cache_path)
            unified_abs = cache_abs.replace('preprocessed_cache_v3', 'preprocessed_cache_unified')
            if os.path.exists(unified_abs):
                cache_abs = unified_abs
            if os.path.exists(cache_abs):
                try:
                    img_np = np.load(cache_abs)
                    img_tensor = self.transform(img_np)
                    return img_tensor, label, img_path
                except Exception:
                    pass

        # Live preprocessing
        abs_path = resolve_path(img_path)
        try:
            if dataset == 'APTOS':
                img_np = ben_graham(abs_path)
            else:
                img_np = clahe_preprocess(abs_path)
            img_tensor = self.transform(img_np)
        except Exception:
            img_tensor = torch.zeros(3, IMG_SIZE, IMG_SIZE)

        return img_tensor, label, img_path


# ================================================================
# TWO-STAGE MC DROPOUT INFERENCE
# ================================================================
def extract_all_features(model, dataloader):
    """
    Stage 1: Run backbone once per image to get CLS features (deterministic).
    Returns features (N, 768), labels (N,), paths list.
    """
    all_features = []
    all_labels   = []
    all_paths    = []

    print(f'\n  Stage 1: Extracting backbone features (deterministic)...')
    with torch.no_grad():
        for images, labels, paths in tqdm(dataloader, desc='  Features', ncols=80):
            images = images.to(DEVICE)
            feats = model.extract_features(images)  # (B, 768)
            all_features.append(feats.cpu())
            all_labels.extend(labels.numpy().tolist())
            all_paths.extend(paths)

    all_features = torch.cat(all_features, dim=0)  # (N, 768)
    all_labels   = np.array(all_labels)
    return all_features, all_labels, all_paths


def mc_dropout_on_heads(model, features, T=T_FORWARD_PASSES, temperature=TEMPERATURE):
    """
    Stage 2: Run T stochastic forward passes through heads only.
    features: (N, 768) tensor
    Returns: (N, T, C) numpy array of probability vectors.
    """
    N = features.size(0)
    all_probs = np.zeros((N, T, NUM_CLASSES), dtype=np.float32)

    print(f'\n  Stage 2: MC Dropout through heads ({T} passes, {N} samples)...')

    with torch.no_grad():
        for t in tqdm(range(T), desc='  MC Passes', ncols=80):
            # Process in chunks to avoid memory issues
            for start in range(0, N, HEAD_BATCH):
                end = min(start + HEAD_BATCH, N)
                feat_batch = features[start:end].to(DEVICE)
                logits = model.forward_heads(feat_batch)
                scaled = logits / temperature
                probs = F.softmax(scaled, dim=1)
                all_probs[start:end, t, :] = probs.cpu().numpy()

    return all_probs


# ================================================================
# UNCERTAINTY METRICS
# ================================================================
def compute_uncertainty_metrics(mc_probs):
    """
    Compute uncertainty metrics from MC dropout probability samples.

    Args:
        mc_probs: (N, T, C) array of MC sampled probability vectors

    Returns dict with:
      - p_mean, predicted_class, max_confidence
      - predictive_entropy (total), expected_entropy (aleatoric),
        mutual_info (epistemic), class_variance
    """
    N, T, C = mc_probs.shape
    eps = 1e-10

    # Predictive mean: average over T passes
    p_mean = mc_probs.mean(axis=1)                     # (N, C)
    predicted_class = p_mean.argmax(axis=1)             # (N,)
    max_confidence  = p_mean.max(axis=1)                # (N,)

    # Predictive entropy: H[p_bar] = -sum(p_bar * log(p_bar))  -- TOTAL uncertainty
    predictive_entropy = -np.sum(p_mean * np.log(p_mean + eps), axis=1)  # (N,)

    # Per-pass entropies
    per_pass_entropy = -np.sum(mc_probs * np.log(mc_probs + eps), axis=2)  # (N, T)

    # Expected entropy: E_t[H[p_t]]  -- ALEATORIC uncertainty
    expected_entropy = per_pass_entropy.mean(axis=1)    # (N,)

    # Mutual information: H - E[H]  -- EPISTEMIC uncertainty
    mutual_info = predictive_entropy - expected_entropy
    mutual_info = np.maximum(mutual_info, 0.0)

    # Prediction variance per class
    class_variance = mc_probs.var(axis=1)               # (N, C)

    return {
        'p_mean':             p_mean,
        'predicted_class':    predicted_class,
        'max_confidence':     max_confidence,
        'predictive_entropy': predictive_entropy,
        'expected_entropy':   expected_entropy,
        'mutual_info':        mutual_info,
        'class_variance':     class_variance,
    }


# ================================================================
# PLOTTING FUNCTIONS
# ================================================================
def plot_uncertainty_vs_accuracy(metrics, labels, save_path):
    """Scatter: total uncertainty vs correctness, colored by class."""
    correct = (metrics['predicted_class'] == labels).astype(int)
    entropy = metrics['predictive_entropy']

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = plt.cm.Set2(np.linspace(0, 1, NUM_CLASSES))
    for cls_idx in range(NUM_CLASSES):
        mask = labels == cls_idx
        ax.scatter(
            entropy[mask], correct[mask] + np.random.uniform(-0.08, 0.08, mask.sum()),
            c=[colors[cls_idx]], alpha=0.5, s=20, label=CLASS_NAMES[cls_idx],
            edgecolors='none'
        )

    ax.set_xlabel('Predictive Entropy (Total Uncertainty)', fontsize=12)
    ax.set_ylabel('Correctness (1=correct, 0=wrong)', fontsize=12)
    ax.set_title('MC Dropout: Uncertainty vs Prediction Correctness', fontsize=14)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Incorrect', 'Correct'])
    ax.legend(title='True Class', fontsize=9, title_fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add vertical line at median uncertainty
    med = np.median(entropy)
    ax.axvline(med, color='red', linestyle='--', alpha=0.5, label=f'Median H={med:.3f}')

    # Summary stats
    correct_ent = entropy[correct == 1]
    wrong_ent   = entropy[correct == 0]
    textstr = (f'Correct: mean H={correct_ent.mean():.3f}\n'
               f'Wrong:   mean H={wrong_ent.mean():.3f}' if len(wrong_ent) > 0
               else f'Correct: mean H={correct_ent.mean():.3f}')
    ax.text(0.98, 0.5, textstr, transform=ax.transAxes,
            fontsize=9, verticalalignment='center', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {save_path}')


def plot_rejection_curve(metrics, labels, save_path):
    """Accuracy as a function of rejection threshold on uncertainty."""
    entropy = metrics['predictive_entropy']
    correct = (metrics['predicted_class'] == labels).astype(int)

    # Sort by decreasing uncertainty
    sorted_idx = np.argsort(entropy)[::-1]
    sorted_correct = correct[sorted_idx]

    N = len(labels)
    rejection_fracs = np.linspace(0.0, 0.95, 200)
    accuracies  = []
    n_remaining = []

    for frac in rejection_fracs:
        n_reject = int(frac * N)
        kept = sorted_correct[n_reject:]
        if len(kept) == 0:
            accuracies.append(np.nan)
            n_remaining.append(0)
        else:
            accuracies.append(kept.mean() * 100)
            n_remaining.append(len(kept))

    accuracies  = np.array(accuracies)
    n_remaining = np.array(n_remaining)

    fig, ax1 = plt.subplots(figsize=(10, 7))

    color1 = '#2196F3'
    ax1.plot(rejection_fracs * 100, accuracies, color=color1, linewidth=2.0,
             label='Accuracy')
    ax1.set_xlabel('Rejection Rate (%)', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim([max(50, np.nanmin(accuracies) - 5), 101])

    # Secondary axis: number of remaining samples
    ax2 = ax1.twinx()
    color2 = '#FF9800'
    ax2.plot(rejection_fracs * 100, n_remaining, color=color2, linewidth=1.5,
             linestyle='--', alpha=0.7, label='Remaining')
    ax2.set_ylabel('Samples Remaining', fontsize=12, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    # Baseline accuracy (no rejection)
    base_acc = correct.mean() * 100
    ax1.axhline(base_acc, color='gray', linestyle=':', alpha=0.5)
    ax1.text(2, base_acc + 0.5, f'Baseline: {base_acc:.1f}%', fontsize=9, color='gray')

    ax1.set_title('Rejection Curve: Accuracy vs Uncertainty-Based Rejection', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=10)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {save_path}')


def plot_epistemic_vs_aleatoric(metrics, labels, save_path):
    """Scatter separating epistemic and aleatoric uncertainty."""
    aleatoric = metrics['expected_entropy']
    epistemic = metrics['mutual_info']
    correct   = (metrics['predicted_class'] == labels).astype(int)

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = plt.cm.Set2(np.linspace(0, 1, NUM_CLASSES))
    for cls_idx in range(NUM_CLASSES):
        mask = labels == cls_idx
        ax.scatter(
            aleatoric[mask], epistemic[mask],
            c=[colors[cls_idx]], alpha=0.45, s=20, label=CLASS_NAMES[cls_idx],
            edgecolors='none'
        )

    # Mark misclassified samples
    wrong_mask = correct == 0
    if wrong_mask.sum() > 0:
        ax.scatter(
            aleatoric[wrong_mask], epistemic[wrong_mask],
            facecolors='none', edgecolors='red', s=60, linewidths=1.2,
            label='Misclassified', zorder=5
        )

    ax.set_xlabel('Aleatoric Uncertainty (Expected Entropy)', fontsize=12)
    ax.set_ylabel('Epistemic Uncertainty (Mutual Information)', fontsize=12)
    ax.set_title('Decomposition of Uncertainty: Epistemic vs Aleatoric', fontsize=14)
    ax.legend(fontsize=9, title='Class', title_fontsize=10)
    ax.grid(True, alpha=0.3)

    # Annotate quadrants
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.text(xlim[0] + 0.02 * (xlim[1] - xlim[0]),
            ylim[1] - 0.05 * (ylim[1] - ylim[0]),
            'Low aleatoric\nHigh epistemic\n(need more data)',
            fontsize=8, alpha=0.6, va='top')
    ax.text(xlim[1] - 0.02 * (xlim[1] - xlim[0]),
            ylim[1] - 0.05 * (ylim[1] - ylim[0]),
            'High aleatoric\nHigh epistemic\n(hard + unseen)',
            fontsize=8, alpha=0.6, va='top', ha='right')
    ax.text(xlim[1] - 0.02 * (xlim[1] - xlim[0]),
            ylim[0] + 0.05 * (ylim[1] - ylim[0]),
            'High aleatoric\nLow epistemic\n(inherently noisy)',
            fontsize=8, alpha=0.6, va='bottom', ha='right')

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {save_path}')


def plot_uncertainty_by_class(metrics, labels, save_path):
    """Box plots of uncertainty per class."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    data_types = [
        ('predictive_entropy', 'Total Uncertainty (Predictive Entropy)'),
        ('expected_entropy',   'Aleatoric Uncertainty (Expected Entropy)'),
        ('mutual_info',        'Epistemic Uncertainty (Mutual Information)'),
    ]

    for ax, (key, title) in zip(axes, data_types):
        data = metrics[key]
        box_data = [data[labels == c] for c in range(NUM_CLASSES)]

        bp = ax.boxplot(box_data, labels=CLASS_NAMES, patch_artist=True,
                        widths=0.6, showfliers=True,
                        flierprops=dict(marker='o', markersize=3, alpha=0.3))

        colors = plt.cm.Set2(np.linspace(0, 1, NUM_CLASSES))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(title, fontsize=11)
        ax.set_ylabel('Uncertainty', fontsize=10)
        ax.grid(True, axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=15)

        # Add sample counts
        for i, cls_data in enumerate(box_data):
            ax.text(i + 1, ax.get_ylim()[1] * 0.95,
                    f'n={len(cls_data)}', ha='center', fontsize=8, alpha=0.6)

    plt.suptitle('Uncertainty Distribution by Disease Class', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {save_path}')


def plot_confidence_vs_uncertainty(metrics, labels, save_path):
    """Scatter showing confidence vs uncertainty (should be anti-correlated)."""
    confidence = metrics['max_confidence']
    entropy    = metrics['predictive_entropy']
    correct    = (metrics['predicted_class'] == labels).astype(int)

    fig, ax = plt.subplots(figsize=(10, 7))

    scatter_correct = ax.scatter(
        confidence[correct == 1], entropy[correct == 1],
        c='#4CAF50', alpha=0.4, s=15, label='Correct', edgecolors='none'
    )
    scatter_wrong = ax.scatter(
        confidence[correct == 0], entropy[correct == 0],
        c='#F44336', alpha=0.6, s=25, label='Incorrect', edgecolors='none',
        marker='x', linewidths=1.0
    )

    # Compute correlation
    from scipy import stats
    r, p_val = stats.pearsonr(confidence, entropy)

    ax.set_xlabel('Maximum Confidence (max p_bar)', fontsize=12)
    ax.set_ylabel('Predictive Entropy (Total Uncertainty)', fontsize=12)
    ax.set_title(f'Confidence vs Uncertainty (Pearson r={r:.3f}, p={p_val:.2e})', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(confidence, entropy, 1)
    x_line = np.linspace(confidence.min(), confidence.max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), 'k--', alpha=0.4, linewidth=1.5)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {save_path}')


# ================================================================
# MAIN
# ================================================================
def main():
    t_start = time.time()

    # ---- 1. Build DataLoader ----
    print('\nLoading test set...')
    dataset = TestDataset(TEST_CSV)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=False
    )
    print(f'  Test samples: {len(dataset)}')

    # ---- 2. Stage 1: Extract backbone features (single deterministic pass) ----
    features, true_labels, image_paths = extract_all_features(model, dataloader)
    print(f'  Features shape: {features.shape}')

    t_feat = time.time() - t_start
    print(f'  Feature extraction: {t_feat:.1f}s')

    # ---- 3. Stage 2: MC Dropout on heads only ----
    mc_probs = mc_dropout_on_heads(
        model, features, T=T_FORWARD_PASSES, temperature=TEMPERATURE
    )
    print(f'  MC probs shape: {mc_probs.shape}  (N, T, C)')

    t_mc = time.time() - t_start - t_feat
    print(f'  MC head passes: {t_mc:.1f}s')

    # ---- 4. Compute Uncertainty Metrics ----
    print('\nComputing uncertainty metrics...')
    metrics = compute_uncertainty_metrics(mc_probs)

    # Print summary statistics
    correct = (metrics['predicted_class'] == true_labels).astype(int)
    accuracy = correct.mean() * 100
    print(f'\n  --- Summary ---')
    print(f'  Accuracy (MC mean):     {accuracy:.2f}%')
    print(f'  Predictive entropy:     mean={metrics["predictive_entropy"].mean():.4f}, '
          f'std={metrics["predictive_entropy"].std():.4f}')
    print(f'  Aleatoric (exp. ent.):  mean={metrics["expected_entropy"].mean():.4f}, '
          f'std={metrics["expected_entropy"].std():.4f}')
    print(f'  Epistemic (MI):         mean={metrics["mutual_info"].mean():.4f}, '
          f'std={metrics["mutual_info"].std():.4f}')
    print(f'  Max confidence:         mean={metrics["max_confidence"].mean():.4f}, '
          f'std={metrics["max_confidence"].std():.4f}')

    # Per-class stats
    print(f'\n  Per-class uncertainty (predictive entropy):')
    for cls_idx in range(NUM_CLASSES):
        mask = true_labels == cls_idx
        n_cls = mask.sum()
        cls_acc = correct[mask].mean() * 100 if n_cls > 0 else 0
        cls_ent = metrics['predictive_entropy'][mask].mean() if n_cls > 0 else 0
        cls_mi  = metrics['mutual_info'][mask].mean() if n_cls > 0 else 0
        print(f'    {CLASS_NAMES[cls_idx]:15s}: n={n_cls:4d}, '
              f'acc={cls_acc:5.1f}%, H={cls_ent:.4f}, MI={cls_mi:.4f}')

    # ---- 5. Generate Plots ----
    print('\nGenerating plots...')

    plot_uncertainty_vs_accuracy(
        metrics, true_labels,
        os.path.join(UNCERT_DIR, 'uncertainty_vs_accuracy.png')
    )
    plot_rejection_curve(
        metrics, true_labels,
        os.path.join(UNCERT_DIR, 'rejection_curve.png')
    )
    plot_epistemic_vs_aleatoric(
        metrics, true_labels,
        os.path.join(UNCERT_DIR, 'epistemic_vs_aleatoric.png')
    )
    plot_uncertainty_by_class(
        metrics, true_labels,
        os.path.join(UNCERT_DIR, 'uncertainty_by_class.png')
    )
    plot_confidence_vs_uncertainty(
        metrics, true_labels,
        os.path.join(UNCERT_DIR, 'confidence_vs_uncertainty.png')
    )

    # ---- 6. Save JSON Results ----
    print('\nSaving results JSON...')

    per_image = []
    for i in range(len(true_labels)):
        per_image.append({
            'image_path':         image_paths[i],
            'true_label':         int(true_labels[i]),
            'true_class':         CLASS_NAMES[int(true_labels[i])],
            'predicted_label':    int(metrics['predicted_class'][i]),
            'predicted_class':    CLASS_NAMES[int(metrics['predicted_class'][i])],
            'correct':            bool(correct[i]),
            'max_confidence':     round(float(metrics['max_confidence'][i]), 6),
            'predictive_entropy': round(float(metrics['predictive_entropy'][i]), 6),
            'expected_entropy':   round(float(metrics['expected_entropy'][i]), 6),
            'mutual_information': round(float(metrics['mutual_info'][i]), 6),
            'class_variance':     [round(float(v), 8) for v in metrics['class_variance'][i]],
            'mean_probs':         [round(float(v), 6) for v in metrics['p_mean'][i]],
        })

    aggregate = {
        'n_samples':     int(len(true_labels)),
        'n_classes':     NUM_CLASSES,
        'mc_passes':     T_FORWARD_PASSES,
        'temperature':   TEMPERATURE,
        'accuracy_pct':  round(float(accuracy), 4),
        'overall': {
            'predictive_entropy': {
                'mean': round(float(metrics['predictive_entropy'].mean()), 6),
                'std':  round(float(metrics['predictive_entropy'].std()), 6),
                'min':  round(float(metrics['predictive_entropy'].min()), 6),
                'max':  round(float(metrics['predictive_entropy'].max()), 6),
            },
            'expected_entropy': {
                'mean': round(float(metrics['expected_entropy'].mean()), 6),
                'std':  round(float(metrics['expected_entropy'].std()), 6),
                'min':  round(float(metrics['expected_entropy'].min()), 6),
                'max':  round(float(metrics['expected_entropy'].max()), 6),
            },
            'mutual_information': {
                'mean': round(float(metrics['mutual_info'].mean()), 6),
                'std':  round(float(metrics['mutual_info'].std()), 6),
                'min':  round(float(metrics['mutual_info'].min()), 6),
                'max':  round(float(metrics['mutual_info'].max()), 6),
            },
            'max_confidence': {
                'mean': round(float(metrics['max_confidence'].mean()), 6),
                'std':  round(float(metrics['max_confidence'].std()), 6),
            },
        },
        'per_class': {},
    }

    for cls_idx in range(NUM_CLASSES):
        mask = true_labels == cls_idx
        n_cls = int(mask.sum())
        if n_cls == 0:
            continue
        aggregate['per_class'][CLASS_NAMES[cls_idx]] = {
            'n_samples':  n_cls,
            'accuracy':   round(float(correct[mask].mean() * 100), 4),
            'pred_entropy_mean': round(float(metrics['predictive_entropy'][mask].mean()), 6),
            'pred_entropy_std':  round(float(metrics['predictive_entropy'][mask].std()), 6),
            'aleatoric_mean':    round(float(metrics['expected_entropy'][mask].mean()), 6),
            'epistemic_mean':    round(float(metrics['mutual_info'][mask].mean()), 6),
            'confidence_mean':   round(float(metrics['max_confidence'][mask].mean()), 6),
        }

    # Rejection curve data at key thresholds
    entropy = metrics['predictive_entropy']
    sorted_idx = np.argsort(entropy)[::-1]
    sorted_correct = correct[sorted_idx]
    rejection_checkpoints = {}
    for frac in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50]:
        n_reject = int(frac * len(true_labels))
        kept = sorted_correct[n_reject:]
        if len(kept) > 0:
            rejection_checkpoints[f'reject_{int(frac*100)}pct'] = {
                'accuracy': round(float(kept.mean() * 100), 4),
                'n_remaining': int(len(kept)),
            }
    aggregate['rejection_curve'] = rejection_checkpoints

    results = {
        'aggregate':  aggregate,
        'per_image':  per_image,
    }

    json_path = os.path.join(UNCERT_DIR, 'mc_dropout_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'  Saved: {json_path}')

    elapsed = time.time() - t_start
    print(f'\nDone in {elapsed:.1f}s')
    print('=' * 65)


if __name__ == '__main__':
    main()
