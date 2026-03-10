#!/usr/bin/env python3
"""
RetinaSense v3.0 -- Phase 1C: Advanced XAI with Integrated Gradients
=====================================================================
Compares Attention Rollout (existing) vs Integrated Gradients (captum)
on 20 test images (4 per class).

Outputs (all saved to outputs_v3/xai/):
  - comparison_grid.png       : 20-row x 3-column grid [Original | Rollout | IG]
  - ig_individual_01..20.png  : Individual IG heatmaps
  - agreement_heatmap.png     : Spatial correlation matrix between methods
  - agreement_score.json      : Numerical agreement scores per image

Usage:
  python integrated_gradients_xai.py
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
from matplotlib.colors import Normalize
from PIL import Image
import pandas as pd
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm

from captum.attr import IntegratedGradients

# Maximize CPU parallelism
torch.set_num_threads(os.cpu_count() or 4)

# ================================================================
# CONFIGURATION
# ================================================================
BASE_DIR   = '/teamspace/studios/this_studio'
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs_v3')
XAI_DIR    = os.path.join(OUTPUT_DIR, 'xai')
os.makedirs(XAI_DIR, exist_ok=True)

MODEL_PATH       = os.path.join(OUTPUT_DIR, 'best_model.pth')
TEMPERATURE_PATH = os.path.join(OUTPUT_DIR, 'temperature.json')
TEST_CSV         = os.path.join(BASE_DIR, 'data', 'test_split.csv')
NORM_STATS_PATH  = os.path.join(BASE_DIR, 'data', 'fundus_norm_stats.json')

CLASS_NAMES = ['Normal', 'Diabetes/DR', 'Glaucoma', 'Cataract', 'AMD']
NUM_CLASSES = 5
IMG_SIZE    = 224
DROPOUT     = 0.3
N_PER_CLASS = 4

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('=' * 65)
print('   RetinaSense v3.0 -- Phase 1C: Integrated Gradients XAI')
print('=' * 65)
print(f'  Device   : {DEVICE}')
if torch.cuda.is_available():
    print(f'  GPU      : {torch.cuda.get_device_name(0)}')
print(f'  Output   : {XAI_DIR}')
print('=' * 65)

# ================================================================
# LOAD NORMALISATION STATS
# ================================================================
if os.path.exists(NORM_STATS_PATH):
    with open(NORM_STATS_PATH) as f:
        norm_stats = json.load(f)
    NORM_MEAN = norm_stats['mean_rgb']
    NORM_STD  = norm_stats['std_rgb']
    print(f'  Fundus norm stats: mean={[round(v,4) for v in NORM_MEAN]}, '
          f'std={[round(v,4) for v in NORM_STD]}')
else:
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD  = [0.229, 0.224, 0.225]
    print('  Using ImageNet normalisation fallback')

with open(TEMPERATURE_PATH) as f:
    TEMPERATURE = json.load(f)['temperature']
print(f'  Temperature T = {TEMPERATURE:.4f}')


# ================================================================
# MODEL ARCHITECTURE (mirrors gradcam_v3.py / retinasense_v3.py)
# ================================================================
class MultiTaskViT(nn.Module):
    """ViT-Base-Patch16-224 with disease + severity heads."""

    def __init__(self, n_disease=NUM_CLASSES, n_severity=5, drop=DROPOUT):
        super().__init__()
        self.backbone = timm.create_model(
            'vit_base_patch16_224', pretrained=False, num_classes=0
        )
        feat = 768

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
        f = self.backbone(x)
        f = self.drop(f)
        return self.disease_head(f), self.severity_head(f)


# ================================================================
# DISEASE-LOGITS WRAPPER FOR CAPTUM
# ================================================================
class DiseaseLogitModel(nn.Module):
    """
    Wraps MultiTaskViT so that forward(x) returns only the disease logits.
    Captum's IntegratedGradients requires a model whose forward output
    is either a scalar or a 1-D tensor. We select the target class
    logit inside the IG call via the `target` parameter, so here we
    return the full (B, 5) disease logits.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        disease_logits, _ = self.model(x)
        return disease_logits


# ================================================================
# ATTENTION ROLLOUT (copied from gradcam_v3.py for self-containment)
# ================================================================
class ViTAttentionRollout:
    """
    Attention Rollout for Vision Transformer.
    Traces information flow from patches to CLS token across all layers.
    """

    def __init__(self, model, discard_ratio=0.97):
        self.model = model
        self.discard_ratio = discard_ratio
        self._attention_maps = []
        self._hooks = []

        # Disable fused attention for explicit weight access
        for block in model.backbone.blocks:
            block.attn.fused_attn = False

        # Register forward hooks on all transformer blocks
        for block in model.backbone.blocks:
            h = block.attn.register_forward_hook(self._attn_hook)
            self._hooks.append(h)

    def _attn_hook(self, module, input, output):
        """Capture softmax attention weights from each block."""
        x = input[0]
        B, N, C = x.shape
        with torch.no_grad():
            qkv = module.qkv(x).reshape(
                B, N, 3, module.num_heads, module.head_dim
            ).permute(2, 0, 3, 1, 4)
            q, k, _ = qkv.unbind(0)
            q, k = module.q_norm(q), module.k_norm(k)
            attn = (q * module.scale @ k.transpose(-2, -1)).softmax(dim=-1)
            self._attention_maps.append(attn.detach().cpu())

    def generate(self, image_tensor, class_idx=None):
        """
        Generate attention rollout heatmap.
        Returns:
            heatmap (224, 224) float32 [0, 1], predicted_label, confidence
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

        attn_stack = torch.stack(self._attention_maps, dim=0)[:, 0]
        attn_mean = attn_stack.mean(dim=1)

        if self.discard_ratio > 0:
            flat = attn_mean.reshape(attn_mean.shape[0], -1)
            thresh = torch.quantile(flat, self.discard_ratio, dim=1, keepdim=True)
            thresh = thresh.unsqueeze(-1)
            attn_mean = torch.where(
                attn_mean >= thresh, attn_mean, torch.zeros_like(attn_mean)
            )

        I = torch.eye(attn_mean.shape[-1]).unsqueeze(0)
        attn_aug = attn_mean + I
        attn_aug = attn_aug / attn_aug.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        rollout = attn_aug[0]
        for l in range(1, len(attn_aug)):
            rollout = rollout @ attn_aug[l]

        cls_attention = rollout[0, 1:]
        spatial = cls_attention.numpy().reshape(14, 14).astype(np.float32)
        spatial = cv2.resize(spatial, (IMG_SIZE, IMG_SIZE),
                             interpolation=cv2.INTER_LINEAR)

        s_min, s_max = spatial.min(), spatial.max()
        if s_max - s_min > 1e-8:
            spatial = (spatial - s_min) / (s_max - s_min)
        else:
            spatial = np.zeros_like(spatial)

        spatial = np.power(spatial, 0.4)  # gamma stretch

        return spatial.astype(np.float32), predicted_label, confidence

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []


# ================================================================
# IMAGE PREPROCESSING (mirrors gradcam_v3.py)
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
    """CLAHE contrast enhancement (ODIR-style)."""
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
    Load image with domain-conditional preprocessing.
    Returns:
        img_np   : (224, 224, 3) uint8 preprocessed
        img_orig : (224, 224, 3) uint8 original
    """
    if not os.path.isabs(image_path):
        clean = image_path
        while clean.startswith('./'):
            clean = clean[2:]
        image_path = os.path.join(BASE_DIR, clean)

    if dataset == 'auto':
        if 'aptos' in image_path.lower() or 'gaussian' in image_path.lower():
            dataset = 'APTOS'
        else:
            dataset = 'ODIR'

    raw = cv2.imread(image_path)
    if raw is None:
        raw = np.array(Image.open(image_path).convert('RGB'))
    else:
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    img_orig = cv2.resize(raw, (IMG_SIZE, IMG_SIZE))

    if dataset == 'APTOS':
        img_np = ben_graham(image_path, sz=IMG_SIZE)
    else:
        img_np = clahe_preprocess(image_path, sz=IMG_SIZE)

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
# CIRCULAR FUNDUS MASK
# ================================================================
def create_fundus_mask(h=IMG_SIZE, w=IMG_SIZE):
    """
    Create a soft circular mask matching the fundus region.
    Uses a smooth Gaussian-blurred edge to avoid hard boundaries.
    Returns float32 mask [0, 1] of shape (h, w).
    """
    cy, cx = h // 2, w // 2
    radius = min(h, w) // 2 - 5
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.circle(mask, (cx, cy), radius, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    return mask


# ================================================================
# INTEGRATED GRADIENTS COMPUTATION
# ================================================================
def compute_ig_attribution(ig_model, ig_method, img_tensor, target_class,
                           n_steps=50, internal_batch_size=10, sigma=10):
    """
    Compute Integrated Gradients attribution for a single image.

    Uses a Gaussian-blurred baseline (sigma=10) which is more appropriate
    for fundus images than a black baseline (since the background is already dark).

    Args:
        ig_model          : DiseaseLogitModel wrapper
        ig_method         : captum IntegratedGradients instance
        img_tensor        : (1, 3, 224, 224) normalised tensor on DEVICE
        target_class      : int, disease class to explain
        n_steps           : number of interpolation steps
        internal_batch_size : batch size for internal IG computation
        sigma             : Gaussian blur sigma for baseline

    Returns:
        attribution : (224, 224) float32 numpy array, normalised [0, 1]
    """
    # Create blurred baseline in pixel space, then normalise
    # First undo normalisation to get pixel-space tensor
    mean_t = torch.tensor(NORM_MEAN, device=DEVICE).view(1, 3, 1, 1)
    std_t  = torch.tensor(NORM_STD, device=DEVICE).view(1, 3, 1, 1)

    # Build the blurred baseline from the input tensor
    # Denormalise -> blur -> renormalise
    img_denorm = img_tensor * std_t + mean_t  # approx [0, 1] range
    img_np_for_blur = (img_denorm[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    blurred_np = cv2.GaussianBlur(img_np_for_blur, (0, 0), sigma)
    # Convert blurred back to tensor with normalisation
    blurred_tensor = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])(blurred_np).unsqueeze(0).to(DEVICE)

    # Compute Integrated Gradients
    img_tensor.requires_grad_(True)
    attributions = ig_method.attribute(
        img_tensor,
        baselines=blurred_tensor,
        target=target_class,
        n_steps=n_steps,
        internal_batch_size=internal_batch_size,
    )

    # Aggregate across channels: take L2 norm across RGB for spatial map
    # shape: (1, 3, 224, 224) -> (224, 224)
    attr_np = attributions[0].detach().cpu().numpy()  # (3, 224, 224)
    # Use absolute values and sum over channels for a positive attribution map
    attr_spatial = np.sqrt(np.sum(attr_np ** 2, axis=0))  # (224, 224)

    # Normalise to [0, 1]
    a_min, a_max = attr_spatial.min(), attr_spatial.max()
    if a_max - a_min > 1e-8:
        attr_spatial = (attr_spatial - a_min) / (a_max - a_min)
    else:
        attr_spatial = np.zeros_like(attr_spatial)

    return attr_spatial.astype(np.float32)


# ================================================================
# OVERLAY FUNCTION
# ================================================================
def overlay_heatmap(original_np, heatmap, alpha=0.6, cmap_name='inferno'):
    """
    Blend heatmap onto original image with circular fundus mask.

    Args:
        original_np : (224, 224, 3) uint8 RGB
        heatmap     : (224, 224) float32 [0, 1]
        alpha       : heatmap blending opacity
        cmap_name   : matplotlib colormap name

    Returns:
        blended : (224, 224, 3) uint8 RGB
    """
    # Apply colormap
    cmap = plt.get_cmap(cmap_name)
    colored = cmap(heatmap)[:, :, :3]  # (224, 224, 3) float [0, 1]
    colored_uint8 = (colored * 255).astype(np.uint8)

    # Get fundus mask
    mask = create_fundus_mask(heatmap.shape[0], heatmap.shape[1])

    # Blend inside the fundus region only
    orig = original_np.astype(np.float32)
    cmap_f = colored_uint8.astype(np.float32)
    blended = orig.copy()
    for c in range(3):
        blended[:, :, c] = (
            orig[:, :, c] * (1 - alpha * mask)
            + cmap_f[:, :, c] * (alpha * mask)
        )
    return np.clip(blended, 0, 255).astype(np.uint8)


# ================================================================
# SELECT TEST IMAGES (same logic as gradcam_v3.py)
# ================================================================
def select_test_images(n_per_class=N_PER_CLASS):
    """Select n_per_class images per disease class from test split."""
    df = pd.read_csv(TEST_CSV)
    samples = []
    for label in range(NUM_CLASSES):
        subset = df[df['disease_label'] == label].drop_duplicates(subset='image_path')
        chosen = subset.head(n_per_class)
        for _, row in chosen.iterrows():
            samples.append({
                'image_path':  row['image_path'],
                'true_label':  int(row['disease_label']),
                'dataset':     str(row.get('source', 'auto')),
            })
    print(f'  Selected {len(samples)} test images '
          f'({n_per_class} per class x {NUM_CLASSES} classes)')
    return samples


# ================================================================
# COMPUTE AGREEMENT METRICS
# ================================================================
def compute_agreement(rollout_map, ig_map, fundus_mask):
    """
    Compute spatial agreement between Attention Rollout and IG heatmaps.

    Metrics:
        - Pearson correlation (within fundus mask)
        - Intersection over Union (IoU) of top-20% activated regions
        - Cosine similarity of flattened masked vectors

    Returns dict of scores.
    """
    # Flatten inside mask
    mask_bool = fundus_mask > 0.5
    r_flat = rollout_map[mask_bool]
    i_flat = ig_map[mask_bool]

    # Pearson correlation
    if len(r_flat) > 2 and r_flat.std() > 1e-8 and i_flat.std() > 1e-8:
        pearson_r, pearson_p = pearsonr(r_flat, i_flat)
    else:
        pearson_r, pearson_p = 0.0, 1.0

    # IoU of top-20% regions
    r_thresh = np.percentile(r_flat, 80)
    i_thresh = np.percentile(i_flat, 80)
    r_top = rollout_map > r_thresh
    i_top = ig_map > i_thresh
    intersection = np.logical_and(r_top, i_top).sum()
    union = np.logical_or(r_top, i_top).sum()
    iou = float(intersection / max(union, 1))

    # Cosine similarity
    r_norm = np.linalg.norm(r_flat)
    i_norm = np.linalg.norm(i_flat)
    if r_norm > 1e-8 and i_norm > 1e-8:
        cosine = float(np.dot(r_flat, i_flat) / (r_norm * i_norm))
    else:
        cosine = 0.0

    return {
        'pearson_r':     round(float(pearson_r), 4),
        'pearson_p':     round(float(pearson_p), 6),
        'iou_top20':     round(iou, 4),
        'cosine_sim':    round(cosine, 4),
    }


# ================================================================
# MAIN PIPELINE
# ================================================================
def main():
    import time
    t_start = time.time()

    # ---- 1. Load model ----
    print('\n[1/6] Loading model...')
    model = MultiTaskViT().to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'  Loaded: {MODEL_PATH}')
    print(f'  Checkpoint epoch: {ckpt.get("epoch", "?") + 1}')

    # ---- 2. Select images ----
    print('\n[2/6] Selecting test images...')
    samples = select_test_images()

    # ---- 3. Preprocess all images + run Attention Rollout ----
    # IMPORTANT: Run rollout FIRST then remove hooks BEFORE IG.
    # This prevents rollout hooks from firing during IG's many
    # forward passes (50 steps), which would be very slow on CPU.
    print('\n[3/6] Running Attention Rollout on all images...')
    rollout = ViTAttentionRollout(model, discard_ratio=0.97)
    print(f'  Attention Rollout: {len(rollout._hooks)} hooks registered')

    preprocessed = []  # store (img_np, img_orig, img_tensor) per sample
    rollout_results = []  # store (heatmap, pred_label, confidence) per sample

    for idx, sample in enumerate(samples):
        img_path   = sample['image_path']
        true_label = sample['true_label']
        dataset    = sample['dataset']
        basename   = os.path.basename(img_path)

        print(f'  [{idx+1:2d}/{len(samples)}] '
              f'{CLASS_NAMES[true_label]:15s} | {basename}', end='  ')

        try:
            img_np, img_orig = load_and_preprocess(img_path, dataset=dataset)
            img_tensor = preprocess_to_tensor(img_np).to(DEVICE)
            preprocessed.append((img_np, img_orig, img_tensor))

            heatmap, pred_label, pred_conf = rollout.generate(img_tensor)
            rollout_results.append((heatmap, pred_label, pred_conf))
            print(f'-> pred={CLASS_NAMES[pred_label]:15s}  conf={pred_conf:.3f}')

        except Exception as e:
            print(f'FAILED: {e}')
            preprocessed.append(None)
            rollout_results.append(None)

    # Remove rollout hooks BEFORE running IG to avoid extra computation
    rollout.remove_hooks()
    # Re-enable fused attention for faster forward passes during IG
    for block in model.backbone.blocks:
        block.attn.fused_attn = True
    print('  Rollout hooks removed. fused_attn re-enabled for IG speed.')

    # ---- 4. Run Integrated Gradients (no rollout hooks active) ----
    print('\n[4/6] Computing Integrated Gradients attributions...')
    disease_model = DiseaseLogitModel(model)
    disease_model.eval()
    ig_method = IntegratedGradients(disease_model)
    print(f'  Baseline: Gaussian blur (sigma=10), n_steps=50, '
          f'internal_batch_size=25')

    all_results = []
    fundus_mask = create_fundus_mask()

    for idx, sample in enumerate(samples):
        if preprocessed[idx] is None or rollout_results[idx] is None:
            continue

        img_path   = sample['image_path']
        true_label = sample['true_label']
        basename   = os.path.basename(img_path)
        img_np, img_orig, img_tensor = preprocessed[idx]
        rollout_heatmap, pred_label, pred_conf = rollout_results[idx]

        print(f'  [{idx+1:2d}/{len(samples)}] '
              f'{CLASS_NAMES[true_label]:15s} | {basename}', end='  ')

        try:
            # Fresh tensor (IG needs requires_grad)
            ig_input = img_tensor.clone().detach().to(DEVICE)

            ig_heatmap = compute_ig_attribution(
                disease_model, ig_method, ig_input,
                target_class=pred_label,
                n_steps=50,
                internal_batch_size=25,
                sigma=10,
            )

            # Agreement
            agreement = compute_agreement(rollout_heatmap, ig_heatmap,
                                          fundus_mask)

            print(f'-> pearson={agreement["pearson_r"]:.3f}  '
                  f'IoU={agreement["iou_top20"]:.3f}')

            all_results.append({
                'idx':              idx,
                'image_path':       img_path,
                'basename':         basename,
                'true_label':       true_label,
                'pred_label':       pred_label,
                'pred_class':       CLASS_NAMES[pred_label],
                'confidence':       round(pred_conf, 4),
                'img_orig':         img_orig,
                'rollout_heatmap':  rollout_heatmap,
                'ig_heatmap':       ig_heatmap,
                'agreement':        agreement,
            })

        except Exception as e:
            print(f'FAILED: {e}')
            import traceback
            traceback.print_exc()
            continue

    n_success = len(all_results)
    print(f'\n  Completed: {n_success}/{len(samples)} images')

    if n_success == 0:
        print('ERROR: No images processed successfully. Exiting.')
        sys.exit(1)

    # ---- 5. Generate visualisations ----
    print('\n[5/6] Generating visualisations...')

    # 5a. Individual IG heatmaps
    print('  Saving individual IG heatmaps...')
    for r in all_results:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

        # Original
        axes[0].imshow(r['img_orig'])
        axes[0].set_title(f'Original\nTrue: {CLASS_NAMES[r["true_label"]]}',
                          fontsize=10, fontweight='bold')
        axes[0].axis('off')

        # IG heatmap (raw)
        im = axes[1].imshow(r['ig_heatmap'], cmap='inferno', vmin=0, vmax=1)
        axes[1].set_title('Integrated Gradients\n(attribution magnitude)',
                          fontsize=10)
        axes[1].axis('off')

        # IG overlay on original
        ig_overlay = overlay_heatmap(r['img_orig'], r['ig_heatmap'],
                                     alpha=0.6, cmap_name='inferno')
        axes[2].imshow(ig_overlay)
        correct = r['pred_label'] == r['true_label']
        status = 'OK' if correct else 'WRONG'
        axes[2].set_title(
            f'IG Overlay\nPred: {r["pred_class"]} ({r["confidence"]:.2f}) [{status}]',
            fontsize=10,
            color='green' if correct else 'red',
            fontweight='bold')
        axes[2].axis('off')

        plt.tight_layout()
        save_path = os.path.join(XAI_DIR,
                                 f'ig_individual_{r["idx"]+1:02d}.png')
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='white')
        plt.close(fig)

    print(f'    Saved {n_success} individual IG heatmaps')

    # 5b. Comparison grid: 20 rows x 3 columns
    print('  Generating comparison grid...')
    n_rows = n_success
    fig, axes = plt.subplots(n_rows, 3, figsize=(14, 4.2 * n_rows))

    # Handle single-row case
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    # Column headers
    col_titles = ['Original Image', 'Attention Rollout', 'Integrated Gradients']

    for i, r in enumerate(all_results):
        true_name = CLASS_NAMES[r['true_label']]
        pred_name = r['pred_class']
        correct = r['pred_label'] == r['true_label']
        status = 'OK' if correct else 'WRONG'

        # Column 0: Original
        axes[i, 0].imshow(r['img_orig'])
        axes[i, 0].set_ylabel(f'#{r["idx"]+1}\nTrue: {true_name}',
                               fontsize=9, fontweight='bold', rotation=0,
                               labelpad=70, va='center')
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        if i == 0:
            axes[i, 0].set_title(col_titles[0], fontsize=12, fontweight='bold',
                                 pad=10)

        # Column 1: Attention Rollout overlay
        rollout_overlay = overlay_heatmap(r['img_orig'], r['rollout_heatmap'],
                                          alpha=0.6, cmap_name='inferno')
        axes[i, 1].imshow(rollout_overlay)
        axes[i, 1].axis('off')
        if i == 0:
            axes[i, 1].set_title(col_titles[1], fontsize=12, fontweight='bold',
                                 pad=10)

        # Column 2: IG overlay
        ig_overlay = overlay_heatmap(r['img_orig'], r['ig_heatmap'],
                                     alpha=0.6, cmap_name='inferno')
        axes[i, 2].imshow(ig_overlay)
        color = 'green' if correct else 'red'
        axes[i, 2].set_xlabel(
            f'Pred: {pred_name} ({r["confidence"]:.2f}) [{status}] | '
            f'Pearson r={r["agreement"]["pearson_r"]:.2f}',
            fontsize=8, color=color, fontweight='bold')
        axes[i, 2].set_xticks([])
        axes[i, 2].set_yticks([])
        if i == 0:
            axes[i, 2].set_title(col_titles[2], fontsize=12, fontweight='bold',
                                 pad=10)

    plt.suptitle('RetinaSense v3.0 -- Attention Rollout vs Integrated Gradients',
                 fontsize=14, fontweight='bold', y=1.001)
    plt.tight_layout()
    grid_path = os.path.join(XAI_DIR, 'comparison_grid.png')
    fig.savefig(grid_path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'    Saved: {grid_path}')

    # 5c. Agreement heatmap (matrix showing per-image spatial correlation)
    print('  Generating agreement heatmap...')

    # Build per-image metrics matrix
    image_labels = [
        f'#{r["idx"]+1} {CLASS_NAMES[r["true_label"]][:6]}'
        for r in all_results
    ]
    metric_names = ['Pearson r', 'IoU (top 20%)', 'Cosine Sim']
    agreement_matrix = np.zeros((n_success, 3))
    for i, r in enumerate(all_results):
        agreement_matrix[i, 0] = r['agreement']['pearson_r']
        agreement_matrix[i, 1] = r['agreement']['iou_top20']
        agreement_matrix[i, 2] = r['agreement']['cosine_sim']

    fig, ax = plt.subplots(figsize=(7, max(8, n_success * 0.45)))
    im = ax.imshow(agreement_matrix, cmap='RdYlGn', aspect='auto',
                   vmin=-0.2, vmax=1.0)

    ax.set_xticks(range(3))
    ax.set_xticklabels(metric_names, fontsize=10, fontweight='bold')
    ax.set_yticks(range(n_success))
    ax.set_yticklabels(image_labels, fontsize=8)

    # Annotate cells
    for i in range(n_success):
        for j in range(3):
            val = agreement_matrix[i, j]
            color = 'white' if val < 0.3 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=8, color=color, fontweight='bold')

    ax.set_title('Rollout vs IG Agreement Scores per Image',
                 fontsize=12, fontweight='bold', pad=12)
    plt.colorbar(im, ax=ax, shrink=0.6, label='Score')
    plt.tight_layout()

    heatmap_path = os.path.join(XAI_DIR, 'agreement_heatmap.png')
    fig.savefig(heatmap_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'    Saved: {heatmap_path}')

    # ---- 6. Save agreement scores JSON ----
    print('\n[6/6] Saving agreement scores...')

    scores_output = {
        'summary': {
            'n_images':          n_success,
            'mean_pearson_r':    round(float(agreement_matrix[:, 0].mean()), 4),
            'mean_iou_top20':    round(float(agreement_matrix[:, 1].mean()), 4),
            'mean_cosine_sim':   round(float(agreement_matrix[:, 2].mean()), 4),
            'std_pearson_r':     round(float(agreement_matrix[:, 0].std()), 4),
            'std_iou_top20':     round(float(agreement_matrix[:, 1].std()), 4),
            'std_cosine_sim':    round(float(agreement_matrix[:, 2].std()), 4),
        },
        'per_image': [],
    }
    for r in all_results:
        scores_output['per_image'].append({
            'image':       r['basename'],
            'true_label':  r['true_label'],
            'true_class':  CLASS_NAMES[r['true_label']],
            'pred_label':  r['pred_label'],
            'pred_class':  r['pred_class'],
            'confidence':  r['confidence'],
            'agreement':   r['agreement'],
        })

    # Per-class summary
    per_class = {}
    for cls_idx in range(NUM_CLASSES):
        cls_results = [r for r in all_results if r['true_label'] == cls_idx]
        if cls_results:
            pearson_vals = [r['agreement']['pearson_r'] for r in cls_results]
            iou_vals     = [r['agreement']['iou_top20'] for r in cls_results]
            cosine_vals  = [r['agreement']['cosine_sim'] for r in cls_results]
            per_class[CLASS_NAMES[cls_idx]] = {
                'n_images':        len(cls_results),
                'mean_pearson_r':  round(float(np.mean(pearson_vals)), 4),
                'mean_iou_top20':  round(float(np.mean(iou_vals)), 4),
                'mean_cosine_sim': round(float(np.mean(cosine_vals)), 4),
            }
    scores_output['per_class'] = per_class

    json_path = os.path.join(XAI_DIR, 'agreement_score.json')
    with open(json_path, 'w') as f:
        json.dump(scores_output, f, indent=2)
    print(f'  Saved: {json_path}')

    # ---- Summary ----
    elapsed = time.time() - t_start
    n_correct = sum(1 for r in all_results
                    if r['pred_label'] == r['true_label'])

    print('\n' + '=' * 65)
    print('   PHASE 1C COMPLETE: Integrated Gradients XAI')
    print('=' * 65)
    print(f'  Images processed  : {n_success}/{len(samples)}')
    print(f'  Correct preds     : {n_correct}/{n_success} '
          f'({100*n_correct/max(n_success,1):.1f}%)')
    print(f'  Mean Pearson r    : {scores_output["summary"]["mean_pearson_r"]:.4f}')
    print(f'  Mean IoU (top 20%): {scores_output["summary"]["mean_iou_top20"]:.4f}')
    print(f'  Mean Cosine Sim   : {scores_output["summary"]["mean_cosine_sim"]:.4f}')
    print(f'  Time elapsed      : {elapsed:.1f}s')
    print(f'  Outputs in        : {XAI_DIR}')
    print('=' * 65)

    # List outputs
    print('\n  Output files:')
    for fname in sorted(os.listdir(XAI_DIR)):
        fpath = os.path.join(XAI_DIR, fname)
        size_kb = os.path.getsize(fpath) / 1024
        print(f'    {fname:40s}  {size_kb:8.1f} KB')


if __name__ == '__main__':
    main()
