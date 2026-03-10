#!/usr/bin/env python3
"""
RetinaSense-ViT — Interactive Clinical Screening Demo (Gradio)
===============================================================
Upload a fundus image → get prediction, attention heatmap, confidence,
uncertainty, OOD check, and downloadable clinical report.
"""

import os, json, sys, time, tempfile, warnings
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gradio as gr
warnings.filterwarnings('ignore')

# ================================================================
# CONFIG
# ================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 224
NUM_CLASSES = 5
CLASS_NAMES = ['Normal', 'Diabetes/DR', 'Glaucoma', 'Cataract', 'AMD']
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'outputs_v3/best_model.pth')
# Config files: look in configs/ first (committed to git), fall back to outputs_v3/
def _cfg(name, subdir):
    committed = os.path.join(BASE_DIR, 'configs', name)
    original  = os.path.join(BASE_DIR, subdir, name)
    return committed if os.path.exists(committed) else original
TEMP_PATH   = _cfg('temperature.json',        'outputs_v3')
THRESH_PATH = _cfg('thresholds.json',         'outputs_v3')
NORM_PATH   = _cfg('fundus_norm_stats.json',  'data')
OOD_PATH    = os.path.join(BASE_DIR, 'outputs_v3/ood_detector')

# Load config files
with open(NORM_PATH) as f:
    ns = json.load(f)
NORM_MEAN, NORM_STD = ns['mean_rgb'], ns['std_rgb']

with open(TEMP_PATH) as f:
    T_OPT = json.load(f)['temperature']

with open(THRESH_PATH) as f:
    td = json.load(f)
THRESHOLDS = td['thresholds']


# ================================================================
# MODEL
# ================================================================
class MultiTaskViT(nn.Module):
    def __init__(self, n_disease=5, n_severity=5, drop=0.3):
        super().__init__()
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
        self.drop = nn.Dropout(drop)
        self.disease_head = nn.Sequential(
            nn.Linear(768, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, n_disease),
        )
        self.severity_head = nn.Sequential(
            nn.Linear(768, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, n_severity),
        )

    def forward(self, x):
        f = self.backbone(x)
        f = self.drop(f)
        return self.disease_head(f), self.severity_head(f)


# Load model
print('Loading model...')
model = MultiTaskViT().to(DEVICE)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f'  Loaded ViT checkpoint epoch {ckpt["epoch"]+1}, val_acc={ckpt["val_acc"]:.2f}%')


# ================================================================
# EFFICIENTNET-B3 (ENSEMBLE)
# ================================================================
class EfficientNetB3(nn.Module):
    def __init__(self, n_classes=NUM_CLASSES, drop=0.3):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b3', pretrained=False, num_classes=0)
        feat_dim = self.backbone.num_features  # 1536
        self.drop = nn.Dropout(drop)
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        f = self.backbone(x)
        f = self.drop(f)
        return self.head(f)


ENSEMBLE_PATH = os.path.join(BASE_DIR, 'outputs_v3/ensemble/efficientnet_b3.pth')
ENSEMBLE_WEIGHT_VIT = 0.35
ENSEMBLE_WEIGHT_EFF = 0.65
USE_ENSEMBLE = False

if os.path.exists(ENSEMBLE_PATH):
    print('Loading EfficientNet-B3 for ensemble...')
    eff_model = EfficientNetB3().to(DEVICE)
    eff_ckpt = torch.load(ENSEMBLE_PATH, map_location=DEVICE, weights_only=False)
    eff_model.load_state_dict(eff_ckpt['model_state_dict'])
    eff_model.eval()
    USE_ENSEMBLE = True
    print(f'  Loaded EfficientNet-B3 (epoch {eff_ckpt["epoch"]+1}, F1={eff_ckpt["macro_f1"]:.4f})')
    print(f'  Ensemble weights: ViT={ENSEMBLE_WEIGHT_VIT}, EfficientNet={ENSEMBLE_WEIGHT_EFF}')
else:
    print('  EfficientNet-B3 not found — running ViT-only mode')


# ================================================================
# ATTENTION ROLLOUT
# ================================================================
class ViTAttentionRollout:
    def __init__(self, mdl, discard_ratio=0.97):
        self.model = mdl
        self.discard_ratio = discard_ratio
        self._attention_maps = []
        self._hooks = []
        for block in mdl.backbone.blocks:
            block.attn.fused_attn = False
        for block in mdl.backbone.blocks:
            h = block.attn.register_forward_hook(self._attn_hook)
            self._hooks.append(h)

    def _attn_hook(self, module, input, output):
        x = input[0]
        B, N, C = x.shape
        with torch.no_grad():
            qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, module.head_dim).permute(2, 0, 3, 1, 4)
            q, k, _ = qkv.unbind(0)
            q, k = module.q_norm(q), module.k_norm(k)
            attn = (q * module.scale @ k.transpose(-2, -1)).softmax(dim=-1)
            self._attention_maps.append(attn.detach().cpu())

    def generate(self, image_tensor):
        self.model.eval()
        self._attention_maps = []
        with torch.no_grad():
            image_tensor = image_tensor.to(DEVICE)
            d_out, s_out = self.model(image_tensor)
            probs = torch.softmax(d_out / T_OPT, dim=1)
            pred_label = int(probs.argmax(dim=1).item())
            confidence = float(probs[0, pred_label].item())
            sev_probs = torch.softmax(s_out, dim=1)

        attn_stack = torch.stack(self._attention_maps, dim=0)[:, 0]
        attn_mean = attn_stack.mean(dim=1)
        if self.discard_ratio > 0:
            flat = attn_mean.reshape(attn_mean.shape[0], -1)
            thresh = torch.quantile(flat, self.discard_ratio, dim=1, keepdim=True).unsqueeze(-1)
            attn_mean = torch.where(attn_mean >= thresh, attn_mean, torch.zeros_like(attn_mean))
        I = torch.eye(attn_mean.shape[-1]).unsqueeze(0)
        attn_aug = attn_mean + I
        attn_aug = attn_aug / attn_aug.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        rollout = attn_aug[0]
        for l in range(1, len(attn_aug)):
            rollout = rollout @ attn_aug[l]
        cls_attn = rollout[0, 1:].numpy().reshape(14, 14).astype(np.float32)
        spatial = cv2.resize(cls_attn, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        s_min, s_max = spatial.min(), spatial.max()
        if s_max - s_min > 1e-8:
            spatial = (spatial - s_min) / (s_max - s_min)
        else:
            spatial = np.zeros_like(spatial)
        spatial = np.power(spatial, 0.4)
        return spatial, pred_label, confidence, probs[0].cpu().numpy(), sev_probs[0].cpu().numpy()

    def overlay(self, orig_img, heatmap, alpha=0.7):
        hm_uint8 = (heatmap * 255).astype(np.uint8)
        cmap = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
        cmap_rgb = cv2.cvtColor(cmap, cv2.COLOR_BGR2RGB)
        h, w = heatmap.shape
        cy, cx = h // 2, w // 2
        radius = min(h, w) // 2 - 5
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(mask, (cx, cy), radius, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        orig = orig_img.astype(np.float32)
        cm = cmap_rgb.astype(np.float32)
        blended = orig.copy()
        for c in range(3):
            blended[:, :, c] = orig[:, :, c] * (1 - alpha * mask) + cm[:, :, c] * (alpha * mask)
        return np.clip(blended, 0, 255).astype(np.uint8)


rollout = ViTAttentionRollout(model)


# ================================================================
# MC DROPOUT UNCERTAINTY
# ================================================================
def mc_dropout_predict(image_tensor, T=15):
    """Run T stochastic forward passes for uncertainty estimation."""
    # Enable dropout layers only
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    all_probs = []
    with torch.no_grad():
        image_tensor = image_tensor.to(DEVICE)
        for _ in range(T):
            d_out, _ = model(image_tensor)
            probs = torch.softmax(d_out / T_OPT, dim=1)
            all_probs.append(probs.cpu().numpy())

    model.eval()
    all_probs = np.array(all_probs)  # (T, 1, C)
    p_mean = all_probs.mean(axis=0)[0]  # (C,)
    p_var = all_probs.var(axis=0)[0]

    # Predictive entropy (total uncertainty)
    pred_entropy = -np.sum(p_mean * np.log(p_mean + 1e-10))
    # Expected entropy (aleatoric)
    exp_entropy = -np.mean(np.sum(all_probs[:, 0] * np.log(all_probs[:, 0] + 1e-10), axis=1))
    # Mutual information (epistemic)
    mutual_info = pred_entropy - exp_entropy

    return {
        'predictive_entropy': float(pred_entropy),
        'aleatoric': float(exp_entropy),
        'epistemic': float(mutual_info),
        'variance': float(p_var.sum()),
    }


# ================================================================
# OOD DETECTION
# ================================================================
class OODDetector:
    def __init__(self):
        self.class_means = None
        self.cov_inv = None
        self.ood_threshold = None
        self.is_fitted = False

    def load(self, path):
        data = np.load(path + '.npz')
        self.class_means = data['class_means']
        self.cov_inv = data['cov_inv']
        self.ood_threshold = float(data['ood_threshold'])
        self.is_fitted = True

    def score(self, feature):
        if not self.is_fitted:
            return 0.0, False
        dists = []
        for cm in self.class_means:
            diff = feature - cm
            d = np.sqrt(diff @ self.cov_inv @ diff)
            dists.append(d)
        min_dist = min(dists)
        return float(min_dist), min_dist > self.ood_threshold


ood = OODDetector()
if os.path.exists(OOD_PATH + '.npz'):
    ood.load(OOD_PATH)
    print(f'  OOD detector loaded (threshold={ood.ood_threshold:.2f})')


# ================================================================
# PREPROCESSING (matches training pipeline in retinasense_v3.py)
# ================================================================
def _crop_black_borders(img, tol=7):
    """Remove dark border padding common in fundus images."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return img
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return img[rmin:rmax+1, cmin:cmax+1]


def _apply_circular_mask(img):
    """Zero out pixels outside the circular fundus field of view."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    r = int(min(h, w) * 0.48)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    return cv2.bitwise_and(img, img, mask=mask)


def preprocess_image(img_pil):
    """Preprocess PIL image for model input.
    Matches the training pipeline: crop borders → resize → CLAHE → circular mask → normalize.
    """
    img_np = np.array(img_pil.convert('RGB'))

    # Step 1: Crop black borders (same as training)
    img_cropped = _crop_black_borders(img_np)

    # Step 2: Resize to model input size
    img_resized = cv2.resize(img_cropped, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    # Step 3: CLAHE on L-channel (default preprocessing — matches ODIR/majority of training data)
    lab = cv2.cvtColor(img_resized, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    processed = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Step 4: Circular mask (critical — model trained with black outside fundus circle)
    processed = _apply_circular_mask(processed)
    processed = np.clip(processed, 0, 255).astype(np.uint8)

    normalize = transforms.Normalize(NORM_MEAN, NORM_STD)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        normalize,
    ])
    tensor = transform(processed).unsqueeze(0)
    return tensor, img_resized, processed


# ================================================================
# CLINICAL RECOMMENDATIONS
# ================================================================
SEVERITY_NAMES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']

RECOMMENDATIONS = {
    'Normal': 'Routine re-screening in 12 months. No signs of retinal disease detected.',
    'Diabetes/DR': {
        0: 'No DR signs detected on this image. Continue annual diabetic eye screening.',
        1: 'Mild NPDR detected. Re-screen in 6-9 months. Optimize glycemic control (HbA1c < 7%).',
        2: 'Moderate NPDR detected. Refer to ophthalmologist within 3 months. Monitor for progression.',
        3: 'Severe NPDR detected. URGENT: Refer to retina specialist within 2-4 weeks. High risk of progression to PDR.',
        4: 'Proliferative DR detected. URGENT: Immediate referral for anti-VEGF/PRP treatment within 1 week.',
    },
    'Glaucoma': 'Suspected glaucoma. Refer for comprehensive evaluation: IOP measurement, visual field test, OCT RNFL analysis.',
    'Cataract': 'Cataract detected. Refer for visual acuity assessment. Consider surgical referral if vision significantly impaired.',
    'AMD': 'Age-related macular degeneration detected. Refer for OCT imaging and anti-VEGF evaluation if wet AMD suspected.',
}


def get_recommendation(pred_class, severity_idx=0):
    rec = RECOMMENDATIONS.get(CLASS_NAMES[pred_class], '')
    if isinstance(rec, dict):
        return rec.get(severity_idx, rec[0])
    return rec


# ================================================================
# REPORT GENERATION
# ================================================================
def generate_report_text(pred_class, confidence, probs, severity_idx, uncertainty, ood_score, ood_flag, triage_key='REVIEW', models_agree=True, vit_pred=0, eff_pred=0):
    lines = []
    lines.append('=' * 60)
    lines.append('  RETINASENSE-ViT CLINICAL SCREENING REPORT')
    lines.append('=' * 60)
    lines.append(f'  Date: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    model_name = 'ViT-Base/16 + EfficientNet-B3 Ensemble' if USE_ENSEMBLE else 'ViT-Base/16'
    lines.append(f'  Model: {model_name} (RetinaSense v3.0)')
    lines.append('')
    lines.append('  PRIMARY FINDING')
    lines.append(f'  Prediction: {CLASS_NAMES[pred_class]}')
    lines.append(f'  Confidence: {confidence*100:.1f}%')
    if pred_class == 1:
        lines.append(f'  DR Severity: {SEVERITY_NAMES[severity_idx]}')
    lines.append('')
    lines.append('  CLASS PROBABILITIES')
    for i, cn in enumerate(CLASS_NAMES):
        bar = '#' * int(probs[i] * 40)
        lines.append(f'    {cn:15s}: {probs[i]*100:5.1f}% |{bar}')
    lines.append('')
    lines.append('  UNCERTAINTY ASSESSMENT')
    ent = uncertainty['predictive_entropy']
    level = 'LOW' if ent < 0.5 else ('MODERATE' if ent < 1.0 else 'HIGH')
    lines.append(f'  Total uncertainty:    {ent:.4f} ({level})')
    lines.append(f'  Epistemic (model):    {uncertainty["epistemic"]:.4f}')
    lines.append(f'  Aleatoric (data):     {uncertainty["aleatoric"]:.4f}')
    lines.append('')
    lines.append('  OUT-OF-DISTRIBUTION CHECK')
    if ood.ood_threshold is not None:
        lines.append(f'  Mahalanobis score: {ood_score:.2f} (threshold: {ood.ood_threshold:.2f})')
        lines.append(f'  Status: {"WARNING - Image may be outside training distribution" if ood_flag else "PASS - Within distribution"}')
    else:
        lines.append('  OOD detector not available (ood_detector.npz missing)')
        lines.append('  Status: SKIPPED')
    lines.append('')
    triage = TRIAGE_LEVELS[triage_key]
    lines.append('  CLINICAL TRIAGE')
    lines.append(f'  Triage Level: {triage["label"]}')
    lines.append(f'  {triage["desc"]}')
    if USE_ENSEMBLE:
        agree_str = "AGREE" if models_agree else f"DISAGREE (ViT: {CLASS_NAMES[vit_pred]}, EfficientNet: {CLASS_NAMES[eff_pred]})"
        lines.append(f'  Model Agreement: {agree_str}')
    lines.append('')
    lines.append('  CLINICAL RECOMMENDATION')
    lines.append(f'  {get_recommendation(pred_class, severity_idx)}')
    lines.append('')
    if level == 'HIGH' or ood_flag:
        lines.append('  *** ADVISORY: High uncertainty or OOD detected.')
        lines.append('      This result should be verified by a specialist. ***')
        lines.append('')
    lines.append('  DISCLAIMER')
    lines.append('  This is an AI-assisted screening tool and does NOT constitute')
    lines.append('  a clinical diagnosis. All findings must be reviewed and confirmed')
    lines.append('  by a qualified ophthalmologist.')
    lines.append('=' * 60)
    return '\n'.join(lines)


# ================================================================
# TEST-TIME AUGMENTATION (TTA)
# ================================================================
def _tta_augment(tensor):
    """Generate augmented versions of input tensor for TTA."""
    augmented = [tensor]  # original
    t = tensor.squeeze(0)  # (3, H, W)
    augmented.append(torch.flip(t, dims=[2]).unsqueeze(0))    # horizontal flip
    augmented.append(torch.flip(t, dims=[1]).unsqueeze(0))    # vertical flip
    # +10 degree rotation
    angle = 10.0 * (3.14159 / 180.0)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    theta1 = torch.tensor([[cos_a, -sin_a, 0], [sin_a, cos_a, 0]], dtype=torch.float32).unsqueeze(0)
    grid1 = F.affine_grid(theta1, tensor.size(), align_corners=False)
    augmented.append(F.grid_sample(tensor, grid1, align_corners=False))
    # -10 degree rotation
    theta2 = torch.tensor([[cos_a, sin_a, 0], [-sin_a, cos_a, 0]], dtype=torch.float32).unsqueeze(0)
    grid2 = F.affine_grid(theta2, tensor.size(), align_corners=False)
    augmented.append(F.grid_sample(tensor, grid2, align_corners=False))
    return augmented


def tta_predict(tensor_list):
    """Run both models on all TTA augmentations and average probabilities."""
    all_vit_probs = []
    all_eff_probs = []
    all_sev_probs = []

    with torch.no_grad():
        for t in tensor_list:
            t = t.to(DEVICE)
            d_out, s_out = model(t)
            vp = torch.softmax(d_out / T_OPT, dim=1).cpu().numpy()[0]
            sp = torch.softmax(s_out, dim=1).cpu().numpy()[0]
            all_vit_probs.append(vp)
            all_sev_probs.append(sp)

            if USE_ENSEMBLE:
                eff_logits = eff_model(t)
                ep = torch.softmax(eff_logits / T_OPT, dim=1).cpu().numpy()[0]
                all_eff_probs.append(ep)

    vit_mean = np.mean(all_vit_probs, axis=0)
    sev_mean = np.mean(all_sev_probs, axis=0)

    if USE_ENSEMBLE:
        eff_mean = np.mean(all_eff_probs, axis=0)
        ensemble_probs = ENSEMBLE_WEIGHT_VIT * vit_mean + ENSEMBLE_WEIGHT_EFF * eff_mean
        return vit_mean, eff_mean, ensemble_probs, sev_mean
    return vit_mean, None, vit_mean, sev_mean


# ================================================================
# CLINICAL TRIAGE SYSTEM
# ================================================================
TRIAGE_LEVELS = {
    'AUTO_SCREEN': {
        'label': 'AUTO-SCREEN',
        'color': '🟢',
        'desc': 'Low risk — routine re-screening recommended.',
    },
    'REVIEW': {
        'label': 'PRIORITY REVIEW',
        'color': '🟡',
        'desc': 'Moderate risk — schedule specialist review within 2 weeks.',
    },
    'URGENT': {
        'label': 'URGENT SPECIALIST',
        'color': '🔴',
        'desc': 'High risk — refer to specialist within 48 hours.',
    },
    'REJECT': {
        'label': 'RESCAN NEEDED',
        'color': '⚪',
        'desc': 'Image quality insufficient or out-of-distribution — rescan required.',
    },
}

def compute_triage(pred_class, confidence, uncertainty, ood_flag, models_agree):
    """Uncertainty-guided clinical triage decision."""
    ent = uncertainty['predictive_entropy']
    epist = uncertainty['epistemic']

    # REJECT: OOD detected
    if ood_flag:
        return 'REJECT'

    # URGENT: low confidence OR high uncertainty OR models disagree on disease
    if confidence < 0.4 or ent > 1.2 or (not models_agree and confidence < 0.6):
        return 'URGENT'

    # PRIORITY REVIEW: moderate confidence, or elevated epistemic uncertainty
    if confidence < 0.7 or ent > 0.7 or epist > 0.05 or not models_agree:
        return 'REVIEW'

    # AUTO-SCREEN: high confidence, low uncertainty, models agree
    return 'AUTO_SCREEN'


# ================================================================
# MAIN PREDICTION FUNCTION
# ================================================================
def predict(image):
    if image is None:
        return None, None, "Please upload a fundus image.", "", None

    img_pil = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    tensor, img_orig, img_processed = preprocess_image(img_pil)

    # 1. Attention Rollout + heatmap (on original tensor)
    heatmap, vit_pred_class, vit_confidence, vit_probs_single, sev_probs_single = rollout.generate(tensor)
    overlay_img = rollout.overlay(img_orig, heatmap)

    # 2. TTA: run predictions on augmented versions
    tta_tensors = _tta_augment(tensor)
    vit_probs, eff_probs_arr, probs, sev_probs = tta_predict(tta_tensors)

    pred_class = int(probs.argmax())
    confidence = float(probs[pred_class])

    # 3. Model agreement check
    vit_pred = int(vit_probs.argmax())
    if USE_ENSEMBLE and eff_probs_arr is not None:
        eff_pred = int(eff_probs_arr.argmax())
        models_agree = (vit_pred == eff_pred)
    else:
        eff_pred = vit_pred
        models_agree = True

    # 4. MC Dropout uncertainty
    uncertainty = mc_dropout_predict(tensor, T=15)

    # 5. OOD detection
    with torch.no_grad():
        feat = model.backbone(tensor.to(DEVICE)).cpu().numpy()[0]
    ood_score, ood_flag = ood.score(feat)

    # 6. Clinical triage
    triage_key = compute_triage(pred_class, confidence, uncertainty, ood_flag, models_agree)
    triage = TRIAGE_LEVELS[triage_key]

    # 7. Severity (if DR)
    severity_idx = int(sev_probs.argmax()) if pred_class == 1 else 0

    # 8. Build probability display
    prob_dict = {cn: float(probs[i]) for i, cn in enumerate(CLASS_NAMES)}

    # 9. Build status text
    ent = uncertainty['predictive_entropy']
    unc_level = 'Low' if ent < 0.5 else ('Moderate' if ent < 1.0 else 'HIGH')
    mode_str = "Ensemble + TTA" if USE_ENSEMBLE else "ViT-Base/16 + TTA"

    status_parts = []
    status_parts.append(f"**Mode: {mode_str}**")
    status_parts.append(f"### {triage['color']} Triage: {triage['label']}")
    status_parts.append(f"*{triage['desc']}*")
    status_parts.append(f"\nPrediction: **{CLASS_NAMES[pred_class]}** ({confidence*100:.1f}%)")
    if pred_class == 1:
        status_parts.append(f"DR Severity: **{SEVERITY_NAMES[severity_idx]}**")
    if USE_ENSEMBLE:
        agree_str = "AGREE" if models_agree else f"DISAGREE (ViT: {CLASS_NAMES[vit_pred]}, EfficientNet: {CLASS_NAMES[eff_pred]})"
        status_parts.append(f"Model Agreement: **{agree_str}**")
    status_parts.append(f"Uncertainty: **{unc_level}** (entropy={ent:.3f})")
    status_parts.append(f"\n**Recommendation:** {get_recommendation(pred_class, severity_idx)}")
    status_text = '\n'.join(status_parts)

    # 10. Generate report
    report = generate_report_text(pred_class, confidence, probs, severity_idx, uncertainty, ood_score, ood_flag,
                                  triage_key=triage_key, models_agree=models_agree, vit_pred=vit_pred, eff_pred=eff_pred)
    report_path = os.path.join(tempfile.gettempdir(), 'retinasense_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)

    return overlay_img, prob_dict, status_text, report, report_path


# ================================================================
# GRADIO INTERFACE
# ================================================================
with gr.Blocks(title="RetinaSense-ViT", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # RetinaSense-ViT Clinical Screening System
    **AI-Powered Retinal Disease Detection** | ViT + EfficientNet Ensemble | 5 Disease Classes | Attention Rollout XAI

    Upload a fundus image to get instant disease screening with explainability, uncertainty quantification, and clinical recommendations.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Upload Fundus Image", type="numpy")
            submit_btn = gr.Button("Analyze", variant="primary", size="lg")

        with gr.Column(scale=1):
            attention_map = gr.Image(label="Attention Rollout Heatmap")
            confidence_bars = gr.Label(label="Class Probabilities", num_top_classes=5)

    with gr.Row():
        with gr.Column():
            status_output = gr.Markdown(label="Clinical Assessment")
        with gr.Column():
            report_output = gr.Textbox(label="Clinical Report", lines=15, interactive=False)

    report_file = gr.File(label="Download Report", visible=True)

    submit_btn.click(
        fn=predict,
        inputs=[input_image],
        outputs=[attention_map, confidence_bars, status_output, report_output, report_file],
    )

    gr.Markdown("""
    ---
    **Disclaimer:** This is a research prototype for AI-assisted retinal screening.
    It is NOT a medical device and should NOT be used for clinical decision-making
    without verification by a qualified ophthalmologist.

    **Model:** ViT-Base/16 + EfficientNet-B3 Ensemble with TTA | **Classes:** Normal, DR, Glaucoma, Cataract, AMD | **Features:** Attention Rollout XAI, MC Dropout Uncertainty, Clinical Triage
    """)


if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port=7860, share=True)
