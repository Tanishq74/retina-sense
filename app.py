#!/usr/bin/env python3
"""
RetinaSense-ViT — Interactive Clinical Screening Demo (Gradio)
===============================================================
Upload a fundus image → get prediction, attention heatmap, confidence,
uncertainty, OOD check, and downloadable clinical report.
"""

import os, json, sys, time, tempfile, warnings, argparse
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
# Primary: DANN-v3 model; fallbacks: dann_v3 -> dann_v2 -> dann -> original model
DANN_V3_MODEL_PATH = os.path.join(BASE_DIR, 'outputs_v3/dann_v3/best_model.pth')
DANN_MODEL_PATH = os.path.join(BASE_DIR, 'outputs_v3/dann/best_model.pth')
DANN_V2_MODEL_PATH = os.path.join(BASE_DIR, 'outputs_v3/dann_v2/best_model.pth')
ORIG_MODEL_PATH = os.path.join(BASE_DIR, 'outputs_v3/best_model.pth')
MODEL_PATH = next(
    (p for p in [DANN_V3_MODEL_PATH, DANN_MODEL_PATH, DANN_V2_MODEL_PATH, ORIG_MODEL_PATH] if os.path.exists(p)),
    ORIG_MODEL_PATH  # final fallback even if missing (will error clearly)
)

# Config files: always load from configs/ directory
TEMP_PATH   = os.path.join(BASE_DIR, 'configs', 'temperature.json')
THRESH_PATH = os.path.join(BASE_DIR, 'configs', 'thresholds.json')
NORM_PATH   = os.path.join(BASE_DIR, 'configs', 'fundus_norm_stats_unified.json')
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
# Filter out DANN-specific keys (domain_head, grl) before loading into MultiTaskViT
state_dict = ckpt['model_state_dict']
filtered = {k: v for k, v in state_dict.items()
            if not k.startswith('domain_head') and not k.startswith('grl')}
if len(filtered) < len(state_dict):
    print(f'  Filtered out {len(state_dict) - len(filtered)} DANN keys (domain_head/grl)')
load_result = model.load_state_dict(filtered, strict=False)
if load_result.unexpected_keys:
    print(f'  Ignored {len(load_result.unexpected_keys)} unexpected keys: '
          f'{load_result.unexpected_keys[:5]}')
if load_result.missing_keys:
    print(f'  WARNING: {len(load_result.missing_keys)} missing keys: {load_result.missing_keys[:5]}')
model.eval()
epoch_info = ckpt.get("epoch", "?")
val_acc = ckpt.get("val_acc", 0.0)
print(f'  Loaded checkpoint from {MODEL_PATH}')
print(f'  Epoch {epoch_info+1 if isinstance(epoch_info, int) else epoch_info}, val_acc={val_acc:.2f}%')


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
# TEST-TIME AUGMENTATION (TTA)
# ================================================================
USE_TTA = True  # toggled by --no-tta CLI flag


def _create_tta_batch(tensor):
    """Create 8 augmented versions of a (1,3,H,W) tensor.

    Augmentations (deterministic, no randomness):
      0 - original
      1 - horizontal flip
      2 - vertical flip
      3 - 90-degree rotation
      4 - 180-degree rotation
      5 - 270-degree rotation
      6 - horizontal flip + 90-degree rotation
      7 - vertical flip + 90-degree rotation

    Returns a (8,3,H,W) batch tensor.
    """
    x = tensor.squeeze(0)  # (3, H, W)
    augmented = [
        x,                                        # 0: original
        torch.flip(x, dims=[2]),                  # 1: horizontal flip
        torch.flip(x, dims=[1]),                  # 2: vertical flip
        torch.rot90(x, k=1, dims=[1, 2]),         # 3: 90-degree rotation
        torch.rot90(x, k=2, dims=[1, 2]),         # 4: 180-degree rotation
        torch.rot90(x, k=3, dims=[1, 2]),         # 5: 270-degree rotation
        torch.rot90(torch.flip(x, dims=[2]), k=1, dims=[1, 2]),  # 6: hflip + 90
        torch.rot90(torch.flip(x, dims=[1]), k=1, dims=[1, 2]),  # 7: vflip + 90
    ]
    return torch.stack(augmented, dim=0)  # (8, 3, H, W)


@torch.no_grad()
def tta_predict(tensor):
    """Run TTA: 8 augmented versions through the model, average softmax probs.

    Args:
        tensor: preprocessed image tensor of shape (1, 3, 224, 224)

    Returns:
        numpy array of shape (NUM_CLASSES,) with averaged class probabilities
    """
    model.eval()
    batch = _create_tta_batch(tensor).to(DEVICE)  # (8, 3, 224, 224)
    d_out, _ = model(batch)                        # (8, NUM_CLASSES)
    probs = torch.softmax(d_out / T_OPT, dim=1)   # temperature-scaled softmax
    avg_probs = probs.mean(dim=0)                  # (NUM_CLASSES,)
    return avg_probs.cpu().numpy()


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
try:
    if os.path.exists(OOD_PATH + '.npz'):
        ood.load(OOD_PATH)
        print(f'  OOD detector loaded (threshold={ood.ood_threshold:.2f})')
    else:
        print('  OOD detector not found (.npz missing) — OOD checks disabled')
except Exception as e:
    print(f'  OOD detector failed to load: {e} — OOD checks disabled')
    ood.is_fitted = False


# ================================================================
# FAISS SIMILAR CASE RETRIEVAL
# ================================================================
faiss_index = None
faiss_metadata = None
faiss_index_type = None  # "IP" or "L2"

def load_faiss_index():
    global faiss_index, faiss_metadata, faiss_index_type
    try:
        import faiss
        retrieval_dir = os.path.join(BASE_DIR, 'outputs_v3', 'retrieval')
        # Prefer IndexFlatIP (rebuilt with all 5 classes), fallback to L2 (legacy)
        ip_path = os.path.join(retrieval_dir, 'index_flat_ip.faiss')
        l2_path = os.path.join(retrieval_dir, 'index_flat_l2.faiss')
        meta_path = os.path.join(retrieval_dir, 'metadata.json')

        if os.path.exists(ip_path):
            index_path = ip_path
            faiss_index_type = "IP"
        elif os.path.exists(l2_path):
            index_path = l2_path
            faiss_index_type = "L2"
        else:
            print('  FAISS index not found — similar case retrieval disabled')
            return

        if os.path.exists(meta_path):
            faiss_index = faiss.read_index(index_path)
            with open(meta_path) as f:
                faiss_metadata = json.load(f)
            # Verify class coverage
            classes_in_index = set(m.get('class_name', '?') for m in faiss_metadata)
            print(f'  FAISS index loaded: {faiss_index.ntotal} vectors '
                  f'(type: {faiss_index_type}, classes: {sorted(classes_in_index)})')
        else:
            print('  FAISS metadata not found — similar case retrieval disabled')
    except ImportError:
        print('  faiss-cpu not installed — similar case retrieval disabled')

load_faiss_index()


def _resolve_cache_path(cache_path_str):
    """Resolve a cache path from metadata (relative or absolute)."""
    if not cache_path_str:
        return None
    # If absolute and exists, use directly
    if os.path.isabs(cache_path_str) and os.path.exists(cache_path_str):
        return cache_path_str
    # Try relative to BASE_DIR
    candidate = os.path.join(BASE_DIR, cache_path_str)
    if os.path.exists(candidate):
        return candidate
    # Try just the filename in known cache directories
    stem = os.path.basename(cache_path_str)
    for cache_dir_name in ['preprocessed_cache_unified', 'preprocessed_cache_v3']:
        candidate = os.path.join(BASE_DIR, cache_dir_name, stem)
        if os.path.exists(candidate):
            return candidate
    return None


@torch.no_grad()
def retrieve_similar(image_tensor, k=5):
    """Retrieve top-k similar cases from the FAISS index using ViT backbone embeddings.

    Supports both IndexFlatIP (cosine similarity) and IndexFlatL2 (L2 distance).
    Returns list of dicts with rank, class_name, label, similarity, image.
    """
    if faiss_index is None or faiss_metadata is None:
        return []
    try:
        import faiss as faiss_lib
        embedding = model.backbone(image_tensor.to(DEVICE)).cpu().numpy().astype(np.float32)
        faiss_lib.normalize_L2(embedding)
        distances, indices = faiss_index.search(embedding, k)
        results = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), 1):
            if idx < 0 or idx >= len(faiss_metadata):
                continue
            meta = faiss_metadata[idx]
            # Compute similarity score based on index type
            if faiss_index_type == "IP":
                similarity = max(0.0, float(dist))  # IP: score IS similarity
            else:
                similarity = max(0.0, 1.0 - dist / 4.0)  # L2: convert distance
            img = None
            cache_path = _resolve_cache_path(meta.get('cache_path', ''))
            if cache_path:
                try:
                    img = np.load(cache_path)
                except Exception:
                    pass
            results.append({
                'rank': rank,
                'class_name': meta.get('class_name', 'Unknown'),
                'label': meta.get('label', -1),
                'similarity': round(similarity * 100, 1),
                'source': meta.get('source', 'unknown'),
                'image': img,
            })
        return results
    except Exception as e:
        print(f'Retrieval error: {e}')
        return []


# RAD_ALPHA: weight for model probs vs kNN probs in retrieval-augmented prediction
RAD_ALPHA = 0.6  # final_probs = alpha * model_probs + (1-alpha) * knn_probs


@torch.no_grad()
def retrieve_augmented_prediction(image_tensor, model_probs, k=5, alpha=None):
    """Retrieval-Augmented Prediction: combine model prediction with kNN vote.

    1. Retrieves top-k similar cases from FAISS
    2. Computes similarity-weighted kNN class vote
    3. Combines: final_probs = alpha * model_probs + (1-alpha) * knn_probs

    Args:
        image_tensor: preprocessed image tensor (1, 3, 224, 224)
        model_probs: numpy array of shape (NUM_CLASSES,) from model
        k: number of neighbors to retrieve
        alpha: weight for model probs (default: RAD_ALPHA)

    Returns:
        dict with:
            'final_probs': combined probability array (NUM_CLASSES,)
            'final_pred': predicted class index
            'final_confidence': confidence of combined prediction
            'knn_probs': kNN vote probability array (NUM_CLASSES,)
            'knn_pred': kNN majority prediction
            'agreement': bool, whether model and kNN agree
            'retrieved_cases': list of retrieved case dicts
    """
    if alpha is None:
        alpha = RAD_ALPHA

    # Get retrieved cases
    retrieved = retrieve_similar(image_tensor, k=k)

    if not retrieved:
        return {
            'final_probs': model_probs,
            'final_pred': int(model_probs.argmax()),
            'final_confidence': float(model_probs.max()),
            'knn_probs': model_probs,
            'knn_pred': int(model_probs.argmax()),
            'agreement': True,
            'retrieved_cases': [],
        }

    # Compute similarity-weighted kNN vote
    knn_probs = np.zeros(NUM_CLASSES, dtype=np.float64)
    for r in retrieved:
        label = r.get('label', -1)
        if 0 <= label < NUM_CLASSES:
            weight = r['similarity'] / 100.0  # normalize back from percentage
            knn_probs[label] += weight

    # Normalize kNN probs
    total = knn_probs.sum()
    if total > 0:
        knn_probs /= total
    else:
        knn_probs = np.ones(NUM_CLASSES) / NUM_CLASSES

    knn_probs = knn_probs.astype(np.float32)
    knn_pred = int(knn_probs.argmax())

    # Combine model + kNN
    final_probs = alpha * model_probs + (1 - alpha) * knn_probs
    final_pred = int(final_probs.argmax())
    final_confidence = float(final_probs[final_pred])

    model_pred = int(model_probs.argmax())
    agreement = (knn_pred == model_pred)

    return {
        'final_probs': final_probs,
        'final_pred': final_pred,
        'final_confidence': final_confidence,
        'knn_probs': knn_probs,
        'knn_pred': knn_pred,
        'agreement': agreement,
        'retrieved_cases': retrieved,
    }


# ================================================================
# PREPROCESSING
# ================================================================
def _crop_black_borders(img):
    """Detect dark borders in fundus images and crop to the fundus region."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        img = img[y:y+h, x:x+w]
    return img


def _apply_circular_mask(img, sz):
    """Apply circular mask to remove corners (standard for fundus images)."""
    mask = np.zeros((sz, sz), dtype=np.uint8)
    cv2.circle(mask, (sz // 2, sz // 2), sz // 2, 255, -1)
    masked = cv2.bitwise_and(img, img, mask=mask)
    return masked


def preprocess_image(img_pil):
    """Preprocess PIL image for model input."""
    img_np = np.array(img_pil.convert('RGB'))
    img_np = _crop_black_borders(img_np)
    img_resized = cv2.resize(img_np, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img_resized = _apply_circular_mask(img_resized, IMG_SIZE)

    # Auto-detect domain: if image has dark borders, likely fundus
    # Apply CLAHE as default preprocessing
    lab = cv2.cvtColor(img_resized, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    processed = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

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
def generate_report_text(pred_class, confidence, probs, severity_idx, uncertainty, ood_score, ood_flag):
    lines = []
    lines.append('=' * 60)
    lines.append('  RETINASENSE-ViT CLINICAL SCREENING REPORT')
    lines.append('=' * 60)
    lines.append(f'  Date: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append(f'  Model: ViT-Base/16 (RetinaSense v3.0)')
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
    else:
        lines.append(f'  Mahalanobis score: {ood_score:.2f} (OOD detector not loaded)')
    lines.append(f'  Status: {"WARNING - Image may be outside training distribution" if ood_flag else "PASS - Within distribution"}')
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
# MAIN PREDICTION FUNCTION
# ================================================================
def predict(image):
    if image is None:
        return None, None, "Please upload a fundus image.", "", None, []

    img_pil = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    tensor, img_orig, img_processed = preprocess_image(img_pil)

    # 1. Attention Rollout + prediction (single pass for heatmap)
    heatmap, pred_class_single, confidence_single, probs_single, sev_probs = rollout.generate(tensor)
    overlay_img = rollout.overlay(img_orig, heatmap)

    # 1b. TTA for better prediction accuracy (overrides single-pass probs)
    if USE_TTA:
        tta_probs = tta_predict(tensor)  # averaged over 8 augmentations
        pred_class = int(tta_probs.argmax())
        confidence = float(tta_probs[pred_class])
        probs = tta_probs
    else:
        pred_class = pred_class_single
        confidence = confidence_single
        probs = probs_single

    # 2. MC Dropout uncertainty
    uncertainty = mc_dropout_predict(tensor, T=15)

    # 3. OOD detection
    with torch.no_grad():
        feat = model.backbone(tensor.to(DEVICE)).cpu().numpy()[0]
    ood_score, ood_flag = ood.score(feat)

    # 4. Severity (if DR)
    severity_idx = int(sev_probs.argmax()) if pred_class == 1 else 0

    # 5. Build probability display
    prob_dict = {cn: float(probs[i]) for i, cn in enumerate(CLASS_NAMES)}

    # 6. Build status text
    ent = uncertainty['predictive_entropy']
    unc_level = 'Low' if ent < 0.5 else ('Moderate' if ent < 1.0 else 'HIGH')

    status_parts = []
    status_parts.append(f"Prediction: **{CLASS_NAMES[pred_class]}** ({confidence*100:.1f}%)")
    if pred_class == 1:
        status_parts.append(f"DR Severity: **{SEVERITY_NAMES[severity_idx]}**")
    status_parts.append(f"Uncertainty: **{unc_level}** (entropy={ent:.3f})")
    status_parts.append(f"OOD Score: {ood_score:.1f} {'(WARNING)' if ood_flag else '(OK)'}")
    status_parts.append(f"\n**Recommendation:** {get_recommendation(pred_class, severity_idx)}")
    status_text = '\n'.join(status_parts)

    # 7. Similar case retrieval (FAISS)
    retrieval_results = retrieve_similar(tensor, k=5)
    gallery_images = []
    for r in retrieval_results:
        if r['image'] is not None:
            caption = f"#{r['rank']} {r['class_name']} ({r['similarity']}% similar)"
            gallery_images.append((r['image'], caption))

    # 8. Generate report
    report = generate_report_text(pred_class, confidence, probs, severity_idx, uncertainty, ood_score, ood_flag)
    report_path = os.path.join(tempfile.gettempdir(), 'retinasense_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)

    return overlay_img, prob_dict, status_text, report, report_path, gallery_images


# ================================================================
# GRADIO INTERFACE
# ================================================================
with gr.Blocks(title="RetinaSense-ViT", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # RetinaSense-ViT Clinical Screening System
    **AI-Powered Retinal Disease Detection** | ViT-Base/16 | 5 Disease Classes | Attention Rollout XAI

    Upload a fundus image to get instant disease screening with explainability, uncertainty quantification, and clinical recommendations.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Upload Fundus Image", type="numpy", height=300)
            submit_btn = gr.Button("Analyze", variant="primary", size="lg")
            gr.Markdown("*Upload any retinal fundus photograph (JPEG/PNG). "
                        "The model expects standard color fundus images.*")

        with gr.Column(scale=1):
            attention_map = gr.Image(label="Attention Rollout Heatmap", height=300)
            confidence_bars = gr.Label(label="Class Probabilities", num_top_classes=5)

    with gr.Row():
        with gr.Column():
            status_output = gr.Markdown(label="Clinical Assessment")
        with gr.Column():
            report_output = gr.Textbox(label="Clinical Report", lines=15, interactive=False)

    report_file = gr.File(label="Download Report", visible=True)

    with gr.Row():
        similar_gallery = gr.Gallery(
            label="Similar Cases from Training Database (FAISS Retrieval)",
            columns=5,
            height=250,
        )

    submit_btn.click(
        fn=predict,
        inputs=[input_image],
        outputs=[attention_map, confidence_bars, status_output, report_output, report_file, similar_gallery],
    )

    gr.Markdown("""
    ---
    **Disclaimer:** This is a research prototype for AI-assisted retinal screening.
    It is NOT a medical device and should NOT be used for clinical decision-making
    without verification by a qualified ophthalmologist.

    **Model:** ViT-Base/16 + DANN-v3 + TTA (8x augmentation) | **Accuracy:** 89.3% | **Classes:** Normal, DR, Glaucoma, Cataract, AMD
    """)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RetinaSense-ViT Clinical Screening Demo')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Override model checkpoint path')
    parser.add_argument('--share', action='store_true', default=True,
                        help='Create public share link (default: True)')
    parser.add_argument('--no-share', dest='share', action='store_false',
                        help='Disable public share link')
    parser.add_argument('--port', type=int, default=7860, help='Server port')
    parser.add_argument('--no-tta', dest='use_tta', action='store_false', default=True,
                        help='Disable Test-Time Augmentation (8x augmentation averaging)')
    args = parser.parse_args()

    # Apply TTA flag (module-level variable, no 'global' needed at top-level scope)
    USE_TTA = args.use_tta
    if not USE_TTA:
        print('TTA disabled via --no-tta flag')
    else:
        print('TTA enabled (8 augmentations per image)')

    # If user overrides model path, reload the model
    if args.model_path:
        override_path = args.model_path
        if not os.path.isabs(override_path):
            override_path = os.path.join(BASE_DIR, override_path)
        if os.path.exists(override_path):
            print(f'Reloading model from override: {override_path}')
            ckpt2 = torch.load(override_path, map_location=DEVICE, weights_only=False)
            sd2 = ckpt2.get('model_state_dict', ckpt2)
            filt2 = {k: v for k, v in sd2.items()
                     if not k.startswith('domain_head') and not k.startswith('grl')}
            model.load_state_dict(filt2, strict=False)
            model.eval()
            print(f'  Model reloaded successfully')
        else:
            print(f'WARNING: --model-path {override_path} not found, using default')

    demo.launch(server_name='0.0.0.0', server_port=args.port, share=args.share, show_error=True)
