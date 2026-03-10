# RetinaSense-ViT — Complete Run Guide
> Read this file at the start of any new session. Everything needed to understand,
> run, test, and extend this project is documented here.

---

## 0. Local Setup (Start Here If Cloning for the First Time)

The project was originally developed on a remote GPU server. All paths are now
**relative** — the code works on any machine. Run this once after cloning:

```bash
git clone https://github.com/Tanishq74/retina-sense
cd retina-sense
bash setup.sh
```

`setup.sh` does everything automatically:
1. Installs all Python dependencies (torch CPU build + timm, gradio, fastapi, etc.)
2. Creates `outputs_v3/` and `data/` directories
3. Downloads `best_model.pth` (331MB) from Hugging Face Hub → `outputs_v3/`
4. Downloads `efficientnet_b3.pth` (45MB) from Hugging Face Hub → `outputs_v3/ensemble/`
5. Verifies `configs/` JSON files are present (they are committed to git)

After setup completes:
```bash
python app.py    # Gradio web demo → http://localhost:7860 (also generates public URL)
```

**What is and isn't in git:**

| File | In git? | How to get it |
|------|---------|---------------|
| All `.py` scripts | Yes | `git clone` |
| `configs/*.json` | Yes | `git clone` (temperature, thresholds, norm stats) |
| `outputs_v3/best_model.pth` | No (331MB) | `bash setup.sh` → auto-downloads from HF |
| `outputs_v3/ensemble/efficientnet_b3.pth` | No (45MB) | `bash setup.sh` → auto-downloads from HF |
| `outputs_v3/ood_detector.npz` | No | Only on GPU server — app runs fine without it (OOD check skipped) |
| `data/*.csv` | No (large) | Only on GPU server — only needed for retraining |
| `preprocessed_cache_v3/` | No (multi-GB) | Only on GPU server — only needed for retraining |

**HF model repo:** https://huggingface.co/tanishq74/retinasense-vit

---

## 1. Project Overview

**RetinaSense-ViT** is a deep learning system for retinal disease classification from fundus photographs, featuring an ensemble architecture, uncertainty-guided clinical triage, and domain-adversarial training.

| Property | Value |
|----------|-------|
| Task | Multi-class classification, 5 diseases |
| Classes | Normal (0), Diabetes/DR (1), Glaucoma (2), Cataract (3), AMD (4) |
| Inference Model | ViT-Base/16 + EfficientNet-B3 Ensemble + TTA |
| Ensemble Accuracy | **74.7%** (vs 49.1% ViT-only) / Macro AUC **0.951** |
| Dataset | 8,540 images — APTOS (3,662) + ODIR (4,878) |
| Split | 70/15/15 — train/calib/test (stratified) |
| Best checkpoint | `outputs_v3/best_model.pth` (epoch 24) |
| GitHub | https://github.com/Tanishq74/retina-sense |
| App Features | Ensemble + TTA + Attention Rollout + MC Dropout + Clinical Triage |

---

## 2. Directory Structure

```
<repo-root>/
│
├── app.py                       # Gradio web demo (Ensemble + TTA + Triage)
├── api/
│   └── main.py                  # FastAPI REST server
│
├── ─── TRAINING SCRIPTS ───
├── retinasense_v3.py            # Main ViT training script (1220 lines)
├── train_ensemble.py            # EfficientNet-B3 ensemble training
├── train_dann.py                # NEW: Domain-Adversarial Neural Network (GPU)
├── kfold_cv.py                  # 5-fold cross-validation (GPU)
├── knowledge_distillation.py    # KD + ONNX export (GPU)
│
├── ─── IMPROVEMENT MODULES ───
├── unified_preprocessing.py     # NEW: Unified CLAHE pipeline (replaces domain-conditional)
├── retfound_backbone.py         # NEW: RETFound foundation model backbone
├── enhanced_augmentation.py     # NEW: CutMix, elastic deform, class-aware augmentation
├── prepare_datasets.py          # NEW: Download/prep 5 additional public datasets
│
├── ─── ANALYSIS & XAI ───
├── gradcam_v3.py                # Attention Rollout XAI
├── eval_dashboard.py            # Full evaluation suite
├── mc_dropout_uncertainty.py    # MC Dropout uncertainty
├── integrated_gradients_xai.py  # Integrated Gradients XAI
├── fairness_analysis.py         # Domain fairness analysis
│
├── ─── DATA & CONFIG ───
├── configs/
│   ├── fundus_norm_stats.json   # mean=[0.4298,0.2784,0.1559] std=[0.2857,0.2065,0.1465]
│   ├── temperature.json         # T=0.6438
│   └── thresholds.json          # Per-class thresholds
├── data/                        # CSVs (on GPU server only)
├── preprocessed_cache_v3/       # .npy image cache (on GPU server only)
│
├── ─── MODEL WEIGHTS (not in git) ───
├── outputs_v3/
│   ├── best_model.pth           # ViT-Base/16 checkpoint (331MB, from HF)
│   ├── ensemble/
│   │   └── efficientnet_b3.pth  # EfficientNet-B3 checkpoint (45MB, from HF)
│   ├── evaluation/              # Phase 1A outputs (7 files)
│   ├── uncertainty/             # Phase 1B outputs (6 files)
│   ├── xai/                     # Phase 1C outputs (23 files)
│   ├── fairness/                # Phase 1D outputs (7 files)
│   ├── gradcam/                 # Attention Rollout heatmaps (22 files)
│   └── dann/                    # DANN outputs (after training)
│
├── ─── DOCUMENTATION ───
├── RUN.md                       # This file
├── ARCHITECTURE_DOCUMENT.md     # System architecture
├── FUNCTIONAL_DOCUMENT.md       # Functional specification
├── FUNCTIONAL_TEST_CASE_DOCUMENT.md
├── IEEE_RESEARCH_PAPER.md       # Research paper draft
├── FINAL_COMPREHENSIVE_REPORT.md
├── Dockerfile                   # Docker deployment
└── requirements_deploy.txt      # Deployment dependencies
```

---

## 3. Model Architecture

Defined as `MultiTaskViT` in `retinasense_v3.py`:

```python
class MultiTaskViT(nn.Module):
    # Backbone: ViT-Base/16, pretrained ImageNet-21k, output_dim=768
    backbone = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
    drop = nn.Dropout(0.3)

    # Disease classification head
    disease_head = Sequential(
        Linear(768, 512), BatchNorm1d(512), ReLU, Dropout(0.3),
        Linear(512, 256), BatchNorm1d(256), ReLU, Dropout(0.2),
        Linear(256, 5)
    )

    # Severity sub-classification head (shared backbone)
    severity_head = Sequential(
        Linear(768, 256), BatchNorm1d(256), ReLU, Dropout(0.3),
        Linear(256, 5)
    )

    def forward(self, x):
        feat = self.drop(self.backbone(x))
        return self.disease_head(feat), self.severity_head(feat)
        # Returns: (disease_logits [B,5], severity_logits [B,5])
        # ALWAYS unpack both, even if only using disease_logits
```

### EfficientNet-B3 (Ensemble Member)

```python
class EfficientNetB3(nn.Module):
    backbone = timm.create_model('efficientnet_b3', num_classes=0)  # 1536-dim
    drop = nn.Dropout(0.3)
    head = Sequential(
        Linear(1536, 512), BatchNorm1d(512), ReLU, Dropout(0.3),
        Linear(512, 256), BatchNorm1d(256), ReLU, Dropout(0.2),
        Linear(256, 5)
    )
    def forward(self, x):
        return self.head(self.drop(self.backbone(x)))
        # Returns: logits [B,5] — single tensor, NOT tuple
```

### Ensemble Inference (how app.py works)

```python
# Both models loaded, both in eval mode
T = 0.6438
vit_probs  = softmax(vit_disease_logits / T, dim=1)    # from MultiTaskViT
eff_probs  = softmax(eff_logits / T, dim=1)             # from EfficientNetB3

# Weighted ensemble
ensemble_probs = 0.35 * vit_probs + 0.65 * eff_probs
pred = argmax(ensemble_probs)

# TTA: average over 4 augmented versions (original, h-flip, v-flip, ±10° rotation)
# MC Dropout: 15 stochastic passes for uncertainty estimation
```

### DANNMultiTaskViT (Domain-Adversarial, in train_dann.py)

```python
class DANNMultiTaskViT(nn.Module):
    backbone = ViT-Base/16 (768-dim)
    disease_head  = 768 → 512 → 256 → 5    (same as MultiTaskViT)
    severity_head = 768 → 256 → 5           (same as MultiTaskViT)
    domain_head   = 768 → GRL → 256 → 128 → 2   (NEW: APTOS vs ODIR)
    # GRL = Gradient Reversal Layer (negates gradients to force domain invariance)
```

---

## 4. Installed Packages

All packages are already installed in the environment:

```
torch==2.8.0+cu128        torchvision==0.23.0+cu128
timm==1.0.25              pytorch-lightning==2.6.0
gradio==6.9.0             captum==0.8.0
onnx==1.20.1              onnxruntime==1.24.3
fastapi==0.133.1          uvicorn==0.41.0
fpdf2==2.8.7              opencv-python==4.11.0.86
torchmetrics==1.7.4
```

If anything is missing:
```bash
pip install -r requirements_deploy.txt
pip install captum gradio fpdf2
```

---

## 5. What Has Already Been Run (Do NOT Re-Run)

These scripts completed successfully on a previous GPU session and their outputs are saved:

| Script | Status | Output Location | Key Result |
|--------|--------|-----------------|------------|
| `retinasense_v3.py` | DONE | `outputs_v3/best_model.pth` | Epoch 24, F1=0.854 |
| `gradcam_v3.py` | DONE | `outputs_v3/gradcam/` (22 files) | Attention Rollout heatmaps |
| `eval_dashboard.py` | DONE | `outputs_v3/evaluation/` (7 files) | Acc=49.1%, AUC=0.893 |
| `mc_dropout_uncertainty.py` | DONE | `outputs_v3/uncertainty/` (6 files) | T=30 MC passes |
| `integrated_gradients_xai.py` | DONE | `outputs_v3/xai/` (23 files) | Pearson r=0.196 |
| `fairness_analysis.py` | DONE | `outputs_v3/fairness/` (7 files) | APTOS ECE=0.51 |
| `train_ensemble.py` | DONE | `outputs_v3/ensemble/` (2 files) | Acc=74.7%, AUC=0.951 |

---

## 6. What Still Needs to Run (Requires GPU)

> **Recommended execution order:**
> 1. `python unified_preprocessing.py` — rebuild cache (fixes domain shift)
> 2. `python prepare_datasets.py --all` — expand dataset (optional but recommended)
> 3. `python train_dann.py` — domain-adversarial training (main accuracy improvement)
> 4. `python kfold_cv.py` — cross-validation for paper
> 5. `python knowledge_distillation.py` — model compression

### 6A-NEW. Unified Preprocessing + Cache Rebuild (~30 min)

```bash
python unified_preprocessing.py
```

**What it does:** Rebuilds the entire image cache using a single CLAHE pipeline for ALL
images (APTOS, ODIR, REFUGE2). This eliminates the domain-conditional preprocessing
(Ben Graham for APTOS, CLAHE for ODIR) that caused the domain shift.

**Outputs:**
- `preprocessed_cache_unified/` — new .npy cache with consistent preprocessing
- `configs/fundus_norm_stats_unified.json` — recomputed normalization stats
- `data/*_unified.csv` — updated CSVs with new cache paths

**After running:** Update training scripts to use the unified cache and new norm stats.

---

### 6B-NEW. Dataset Expansion (optional, ~1-2 hours download + preprocess)

```bash
python prepare_datasets.py --list              # show available datasets
python prepare_datasets.py --instructions      # download instructions
python prepare_datasets.py --dataset eyepacs --raw-dir ./data/eyepacs
python prepare_datasets.py --dataset refuge --raw-dir ./data/refuge
python prepare_datasets.py --dataset adam --raw-dir ./data/adam
python prepare_datasets.py --merge             # combine all into unified splits
```

**Available datasets:**

| Dataset | Images | Classes Added | Impact |
|---------|--------|--------------|--------|
| EyePACS | ~35,000 | DR + Normal | Massive DR/Normal boost |
| MESSIDOR-2 | 1,748 | DR grades | More DR diversity |
| REFUGE | ~1,200 | Glaucoma + Normal | 20× more Glaucoma samples |
| ADAM (iChallenge-AMD) | ~1,200 | AMD + Normal | 30× more AMD samples |
| ORIGA | ~650 | Glaucoma + Normal | More Glaucoma diversity |

---

### 6C-NEW. Domain-Adversarial Training (~2-3 hours on H100)

```bash
python train_dann.py
```

**What it does:** Trains a Domain-Adversarial Neural Network (DANN) with gradient reversal
that forces the ViT backbone to learn features that are predictive of disease but NOT
predictive of which dataset (APTOS vs ODIR) the image came from.

**Key components:**
- `GradientReversalLayer`: reverses gradients with Ganin schedule (lambda ramps 0→1)
- `DANNMultiTaskViT`: disease head + severity head + domain discriminator
- Loss: `disease_loss + 0.2*severity_loss + alpha*lambda*domain_loss`
- Warm-starts from `outputs_v3/best_model.pth`
- Same training recipe as v3: AdamW, LLRD, OneCycleLR, Focal Loss, Mixup

**Outputs → `outputs_v3/dann/`:**
- `best_model.pth` — DANN-trained checkpoint
- `history.json` — per-epoch metrics
- `dashboard.png` — training curves (disease + domain accuracy)

**Expected results:** Domain accuracy should converge toward ~50% (random = domain-invariant).
Disease accuracy should improve, especially on APTOS DR images.

---

### 6D-NEW. RETFound Backbone (alternative to 6C)

```python
# In your training script, replace:
from retinasense_v3 import MultiTaskViT
# With:
from retfound_backbone import MultiTaskRetFound, setup_retfound

# Download RETFound weights (once)
setup_retfound()

# Create model with retinal-pretrained backbone
model = MultiTaskRetFound(retfound_path='./weights/RETFound_cfp_weights.pth')
```

**What it does:** Swaps ImageNet-pretrained ViT for RETFound — a ViT-Base/16 pretrained
on 1.6 million retinal images using masked autoencoding. Same architecture, much better
features for retinal pathology.

---

### 6E. K-Fold Cross-Validation (~2 hours on H100)

```bash
python kfold_cv.py
```

**What it does:** 5-fold stratified CV on the full train+calib pool (7,265 images).
Produces mean ± std confidence intervals for the paper.

**Config (inside kfold_cv.py):**
```
N_FOLDS=5, N_EPOCHS=30, PATIENCE=8, BATCH_SIZE=32
BASE_LR=3e-4, LLRD_DECAY=0.75, MIXUP_ALPHA=0.4, FOCAL_GAMMA=1.0
```

**Outputs → `outputs_v3/kfold/`:**
- `fold_1_best.pth` ... `fold_5_best.pth`
- `kfold_results.json` — mean ± std per metric
- `fold_comparison.png` — bar chart per fold
- `perclass_f1_boxplot.png` — per-class F1 boxplot

**Expected results:**
```
Accuracy:          60–75% ± ~5%
Balanced Accuracy: 75–85% ± ~3%
Macro F1:          0.65–0.75 ± ~0.03
Macro AUC:         0.90–0.96 ± ~0.02
```

**Troubleshooting:**
| Problem | Fix |
|---------|-----|
| CUDA OOM | Reduce BATCH_SIZE from 32 to 16 in kfold_cv.py |
| Cache miss | Check preprocessed_cache_v3/ has .npy files |
| KeyError: mean_rgb | data/fundus_norm_stats.json must exist |
| Arch mismatch | MultiTaskViT in kfold_cv.py must match retinasense_v3.py |

After completion: update `SESSION_CONTEXT.md` with mean±std numbers.

---

### 6B. Knowledge Distillation + ONNX Export (~30 min on H100)

```bash
cd /teamspace/studios/this_studio
python knowledge_distillation.py
```

**What it does:** Compresses ViT-Base (86M params) → ViT-Tiny (5.7M params) using
knowledge distillation. Then exports to ONNX and quantizes to INT8 for CPU deployment.

**Config (inside knowledge_distillation.py):**
```
KD_ALPHA=0.3 (30% CE loss + 70% KL distillation loss)
KD_TEMP=4.0  (softens teacher logits for distillation)
Student: vit_tiny_patch16_224 (192-dim, ~5.7M params)
```

**Outputs → `outputs_v3/distillation/`:**
- `student_best.pth` — distilled ViT-Tiny checkpoint
- `retinasense_student.onnx` — ONNX model (opset 17, dynamic batch)
- `retinasense_student_int8.onnx` — INT8 quantized (~6MB)
- `distillation_results.json` — accuracy comparison + CPU benchmark
- `distillation_curves.png` — training curves

**Expected results:**
```
Teacher (ViT-Base):   Acc~74%, AUC~0.95, Size=331MB
Student (ViT-Tiny):   Acc~68-72%, AUC~0.91-0.94, Size~23MB
INT8 quantized:       Acc~67-71%, Size~6MB, CPU inference ~80ms
```

---

## 7. Running the Web Applications

### 7A. Gradio Web Demo

```bash
python app.py
```

- Opens on port **7860** (also generates a public shareable URL)
- Features:
  - **Ensemble prediction**: ViT-Base/16 (35%) + EfficientNet-B3 (65%)
  - **Test-Time Augmentation**: 4 augmented versions averaged (h-flip, v-flip, ±10° rotation)
  - **Attention Rollout heatmap**: ViT attention visualization
  - **MC Dropout uncertainty**: 15 stochastic passes (epistemic + aleatoric split)
  - **Clinical triage**: AUTO-SCREEN / PRIORITY REVIEW / URGENT / RESCAN
  - **Model disagreement detection**: shows when ViT and EfficientNet disagree
  - **OOD detection**: Mahalanobis distance (gracefully skipped if npz missing)
  - **Downloadable clinical report**: .txt file with full analysis

**What app.py does internally:**
1. Loads `outputs_v3/best_model.pth` (ViT-Base/16)
2. Loads `outputs_v3/ensemble/efficientnet_b3.pth` (EfficientNet-B3)
3. Loads configs: `temperature.json`, `thresholds.json`, `fundus_norm_stats.json`
4. Preprocessing: crop borders → resize 224 (INTER_AREA) → CLAHE → circular mask → normalize
5. Runs TTA (4 augmentations) through both models
6. Computes ensemble probabilities (35/65 weighted average)
7. Runs MC Dropout (15 passes) for uncertainty
8. Computes triage level based on confidence + uncertainty + model agreement
9. Generates Attention Rollout heatmap
10. Generates clinical recommendation + downloadable report

**Preprocessing pipeline (CRITICAL — must match training):**
```python
def preprocess_image(img_pil):
    img = crop_black_borders(img)        # remove dark padding
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    img = apply_clahe(img)               # CLAHE on L-channel in LAB
    img = apply_circular_mask(img)       # zero pixels outside fundus circle (r=0.48)
    tensor = ToTensor + Normalize(mean=[0.4298,0.2784,0.1559], std=[0.2857,0.2065,0.1465])
```

**Clinical Triage System:**

| Level | Criteria | Action |
|-------|----------|--------|
| AUTO-SCREEN | Confidence > 70%, low uncertainty, models agree | Routine re-screening |
| PRIORITY REVIEW | Confidence 40-70%, or elevated uncertainty | Specialist within 2 weeks |
| URGENT SPECIALIST | Confidence < 40%, or high uncertainty, or models disagree | Specialist within 48 hours |
| RESCAN NEEDED | OOD detected | Image quality issue, rescan |

---

### 7B. FastAPI REST Server

```bash
cd /teamspace/studios/this_studio
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Or with auto-reload during development:
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Returns `{"status": "healthy", "model": "loaded"}` |
| POST | `/predict` | Single image inference |
| POST | `/predict/batch` | Batch inference (multiple images) |
| GET | `/docs` | Interactive Swagger UI |

**Test the API:**
```bash
# Health check
curl http://localhost:8000/health

# Single prediction (replace with a real fundus image path)
curl -X POST http://localhost:8000/predict \
  -F "file=@/path/to/fundus_image.jpg"

# Response format:
{
  "predicted_class": "Diabetes/DR",
  "predicted_label": 1,
  "confidence": 0.819,
  "probabilities": {"Normal": 0.08, "Diabetes/DR": 0.82, "Glaucoma": 0.03, "Cataract": 0.04, "AMD": 0.03},
  "severity": "Moderate",
  "ood_score": 12.4,
  "is_ood": false,
  "inference_time_ms": 370
}
```

---

### 7C. Docker Deployment

```bash
cd /teamspace/studios/this_studio

# Build the image
docker build -t retinasense .

# Run the container
docker run -p 8000:8000 retinasense

# With GPU support
docker run --gpus all -p 8000:8000 retinasense
```

---

## 8. Running XAI / Attention Rollout on New Images

```bash
cd /teamspace/studios/this_studio
python gradcam_v3.py
```

**What it does:** Runs Attention Rollout (Abnar & Zuidema 2020) on 20 sample images
(4 per class), saves heatmap overlays to `outputs_v3/gradcam/`.

**Key settings in gradcam_v3.py:**
```python
discard_ratio = 0.97      # discard bottom 97% of attention (sharp focus)
alpha = 0.7               # heatmap overlay strength
power_stretch = 0.4       # contrast enhancement (np.power(spatial, 0.4))
```

**Note:** Standard Grad-CAM does NOT work on ViT (CLS token problem — zero patch gradients).
Always use `ViTAttentionRollout`, never `GradCAM` for this model.

---

## 9. Key Numbers (Current Performance)

### Single-Run Test Set (1,287 images)

| Metric | Value |
|--------|-------|
| Overall Accuracy | 49.1% |
| Balanced Accuracy | 76.7% |
| Macro F1 | 0.626 |
| Macro AUC | 0.893 |
| Cohen Kappa | 0.293 |

**Per-class breakdown:**

| Class | Precision | Recall | F1 | AUC | Support |
|-------|-----------|--------|----|-----|---------|
| Normal | 0.336 | 0.968 | 0.499 | 0.770 | 310 |
| Diabetes/DR | 0.995 | 0.253 | 0.404 | 0.830 | 837 |
| Glaucoma | 0.717 | 0.635 | 0.673 | 0.877 | 52 |
| Cataract | 0.681 | 0.979 | 0.803 | 0.988 | 48 |
| AMD | 0.597 | 1.000 | 0.748 | 0.998 | 40 |

**Ensemble (ViT 35% + EfficientNet-B3 65%):**

| Metric | Value |
|--------|-------|
| Accuracy | 74.7% |
| Macro F1 | 0.712 |
| Macro AUC | 0.951 |
| Disagreement rate | 44.2% (flags for human review) |

### Domain Gap

| Domain | N | Accuracy | ECE |
|--------|---|----------|-----|
| ODIR | 709 | 67.7% | 0.157 |
| APTOS | 570 | 26.5% | 0.510 |
| REFUGE2 | 8 | 12.5% | — |

**Critical finding:** DR recall = 25.3%. 573 of 837 DR images misclassified as Normal.
DR precision = 99.5% — when the model says DR, it's correct. But it rarely fires.
Root cause: APTOS Ben Graham preprocessing creates a domain shift from ODIR CLAHE images.

---

## 10. Configuration Files

### Temperature & Thresholds

`outputs_v3/temperature.json`:
```json
{"temperature": 0.6438, "ece_before": 0.1618, "ece_after": 0.1014}
```

`outputs_v3/thresholds.json`:
```json
{
  "thresholds": [0.638, 0.068, 0.840, 0.564, 0.289],
  "class_names": ["Normal", "Diabetes/DR", "Glaucoma", "Cataract", "AMD"]
}
```

Note: DR threshold = 0.068 (very low). Despite this, APTOS DR images still get
classified as Normal because their DR probability stays below even this threshold.

### Normalization Stats

`data/fundus_norm_stats.json` keys: `mean_rgb` and `std_rgb` (not `mean`/`std`).
```python
import json
stats = json.load(open('data/fundus_norm_stats.json'))
mean = stats['mean_rgb']   # [0.4298, 0.2784, 0.1559]
std  = stats['std_rgb']    # [0.2857, 0.2065, 0.1465]
```

### OOD Detector

```python
import numpy as np
data = np.load('outputs_v3/ood_detector.npz')
# Keys: class_means, precision_matrix, threshold
# threshold = 42.82 (Mahalanobis distance)
# If distance > 42.82 → flag as out-of-distribution
```

---

## 11. Data Loading Pattern

CSV columns: `image_path`, `label`, `source`, `cache_path`

```python
import pandas as pd, numpy as np, torch
from torchvision import transforms

df = pd.read_csv('data/test_split.csv')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4298, 0.2784, 0.1559],
                         std=[0.2857, 0.2065, 0.1465])
])

# Fast path: load from preprocessed cache (already resized to 224x224)
def load_image(row):
    cache = row['cache_path']   # e.g. preprocessed_cache_v3/xxxx.npy
    img = np.load(cache)        # uint8 [H,W,3]
    img = torch.tensor(img).permute(2,0,1).float() / 255.0
    return transform_normalize(img)  # apply only normalize, not resize

# Sources: 'APTOS', 'ODIR', 'REFUGE2'
# Labels: 0=Normal, 1=DR, 2=Glaucoma, 3=Cataract, 4=AMD
```

---

## 12. Complete Task Checklist

### Done (Training — GPU server)
- [x] Train ViT-Base/16 model (retinasense_v3.py) — epoch 24, F1=0.854
- [x] Temperature calibration (T=0.6438)
- [x] Per-class threshold optimization
- [x] Attention Rollout XAI (gradcam_v3.py) — heatmaps working
- [x] OOD Mahalanobis detector
- [x] Evaluation dashboard (eval_dashboard.py) — 7 output files
- [x] MC Dropout uncertainty (mc_dropout_uncertainty.py) — 6 output files
- [x] Integrated Gradients XAI (integrated_gradients_xai.py) — 23 output files
- [x] Fairness/domain analysis (fairness_analysis.py) — 7 output files
- [x] EfficientNet-B3 ensemble (train_ensemble.py) — 74.7% acc, AUC=0.951

### Done (App — session 2026-03-10)
- [x] Ensemble inference in app.py (ViT 35% + EfficientNet 65%)
- [x] Fixed preprocessing pipeline (border crop + circular mask + INTER_AREA)
- [x] Test-Time Augmentation (4 augmented versions averaged)
- [x] Clinical triage system (AUTO-SCREEN / REVIEW / URGENT / RESCAN)
- [x] Model disagreement detection (ViT vs EfficientNet)
- [x] OOD graceful handling (no crash when npz missing)
- [x] Gradio web app with all features — port 7860
- [x] FastAPI REST server (api/main.py) — port 8000
- [x] Docker deployment (Dockerfile)

### Done (New training code — ready for GPU)
- [x] `unified_preprocessing.py` — single CLAHE pipeline for all sources
- [x] `train_dann.py` — Domain-Adversarial Neural Network training
- [x] `retfound_backbone.py` — RETFound foundation model backbone support
- [x] `enhanced_augmentation.py` — CutMix, elastic deform, 5× minority oversampling
- [x] `prepare_datasets.py` — download/prep for EyePACS, MESSIDOR-2, REFUGE, ADAM, ORIGA

### Needs GPU (run in this order)
1. [ ] Rebuild cache: `python unified_preprocessing.py` (~30 min)
2. [ ] Expand dataset: `python prepare_datasets.py --all` (optional)
3. [ ] DANN training: `python train_dann.py` (~2-3 hrs on H100)
4. [ ] K-Fold CV: `python kfold_cv.py` (~2 hrs on H100)
5. [ ] Knowledge Distillation: `python knowledge_distillation.py` (~30 min on H100)
6. [ ] (Alternative) RETFound backbone: modify training to use `retfound_backbone.py`

### Known Issues
- [ ] DR recall = 25.3% — root cause: APTOS domain shift (Ben Graham vs CLAHE preprocessing).
      **Fix:** `unified_preprocessing.py` + `train_dann.py`
- [ ] AMD under-represented (~40 samples). **Fix:** `prepare_datasets.py` to add ADAM dataset (1,200 AMD images)
- [ ] Model overconfident on garbage input (random noise: 86%). Inherent model property.
- [ ] Ensemble thresholds not calibrated — current thresholds are for ViT-only.
      **Fix:** Recalibrate after DANN retraining.
- [ ] Temperature T=0.6438 sharpens wrong predictions. Consider recalibrating after retraining.

---

## 13. Model Weights — Hugging Face Hub

Model weights are NOT in git (too large). They are hosted on Hugging Face:

**Repo:** https://huggingface.co/tanishq74/retinasense-vit

| File | Size | Description |
|------|------|-------------|
| `best_model.pth` | 331MB | ViT-Base/16 trained checkpoint (epoch 24) |
| `efficientnet_b3.pth` | 45MB | EfficientNet-B3 ensemble checkpoint |

**Download automatically (run once):**
```bash
bash setup.sh    # handles everything including model download
```

**Or download manually:**
```python
from huggingface_hub import hf_hub_download
import shutil, os

os.makedirs("outputs_v3/ensemble", exist_ok=True)
for fname, dest in [
    ("best_model.pth",       "outputs_v3/best_model.pth"),
    ("efficientnet_b3.pth",  "outputs_v3/ensemble/efficientnet_b3.pth"),
]:
    path = hf_hub_download(repo_id="tanishq74/retinasense-vit", filename=fname)
    shutil.copy(path, dest)
```

---

## 14. Quick-Start Commands

```bash
# Full local setup from scratch (cloned repo):
bash setup.sh                        # installs deps + downloads models from HF

# Check what's already been generated
ls outputs_v3/evaluation/ outputs_v3/uncertainty/ outputs_v3/xai/ outputs_v3/fairness/ outputs_v3/ensemble/

# Start Gradio app
python app.py

# Start FastAPI server
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

# Run K-Fold CV (GPU required, ~2 hrs)
python kfold_cv.py

# Run Knowledge Distillation (GPU required, ~30 min)
python knowledge_distillation.py

# Check GPU
nvidia-smi

# Check disk space (outputs can be large)
df -h .

# Check running processes
ps aux | grep python | grep -v grep
```

---

## 15. Paper Writing Notes

Use these numbers in the paper:

**Table 1 — Main Results (Single Run, Test Set n=1,287):**
```
Overall Accuracy:    49.1%  (weighted by class imbalance — 837/1287 are DR)
Balanced Accuracy:   76.7%
Macro F1:            0.626
Macro AUC:           0.893
```

**Table 2 — Ensemble Results:**
```
ViT-Base alone:               Acc=49.1%, F1=0.626, AUC=0.893
EfficientNet-B3 alone:        Acc=72.2%, F1=0.658, AUC=0.923
ViT+EfficientNet Ensemble:    Acc=74.7%, F1=0.712, AUC=0.951
```

**Table 3 — K-Fold CV (fill in after running kfold_cv.py):**
```
Accuracy:          XX.X% ± X.X%
Balanced Accuracy: XX.X% ± X.X%
Macro F1:          0.XXX ± 0.XXX
Macro AUC:         0.XXX ± 0.XXX
```

**Table 4 — DANN Results (fill in after running train_dann.py):**
```
Overall Accuracy:    XX.X%  (expected: 80-85%)
DR Recall:           XX.X%  (expected: 60-75%, up from 25.3%)
Macro F1:            0.XXX  (expected: 0.75-0.85)
Domain accuracy:     ~50%   (closer to 50% = more domain-invariant)
```

**Reviewer issues addressed:**
- Missing per-class metrics → eval_dashboard.py outputs `metrics_report.json`
- Missing ROC/PR curves → `outputs_v3/evaluation/roc_curves_per_class.png`, `precision_recall_curves.png`
- No cross-validation → kfold_cv.py (run on GPU)
- No confidence intervals → kfold_results.json (mean ± std, after K-Fold)
- No statistical significance → fairness_analysis.py, chi-squared test p=0.296
- No uncertainty quantification → mc_dropout_uncertainty.py (epistemic/aleatoric split)

---

## 16. Research Novelty (Paper Differentiators)

### Novelty 1: Domain-Adversarial Retinal Screening
- **Problem:** Cross-dataset domain shift (APTOS vs ODIR) causes 25.3% DR recall
- **Solution:** DANN with gradient reversal forces domain-invariant features
- **Implementation:** `train_dann.py`
- **Paper angle:** "Domain-Invariant Retinal Disease Classification Across Heterogeneous Fundus Image Sources"

### Novelty 2: Uncertainty-Guided Clinical Triage
- **Problem:** When should the AI auto-screen vs defer to a human?
- **Solution:** Combine confidence + MC Dropout uncertainty + ensemble disagreement into triage levels
- **Implementation:** `app.py` (live, working)
- **Paper angle:** "Uncertainty-Aware Retinal Screening: When to Trust the AI and When to Defer"

### Novelty 3: RETFound Foundation Model Transfer
- **Problem:** ImageNet features are suboptimal for retinal pathology
- **Solution:** Use RETFound (1.6M retinal images MAE-pretrained) as backbone
- **Implementation:** `retfound_backbone.py`
- **Paper angle:** "Parameter-Efficient Transfer from Retinal Foundation Models for Small-Dataset Classification"

### Novelty 4: Preprocessing-Induced Domain Shift Analysis
- **Problem:** Different preprocessing (Ben Graham vs CLAHE) creates artificial domain shift
- **Finding:** CLAHE alone shifts Glaucoma probability by +43 percentage points
- **Solution:** Unified CLAHE pipeline eliminates this
- **Implementation:** `unified_preprocessing.py`
- **Paper angle:** Novel contribution — documented evidence that preprocessing choices create measurable domain shift in retinal AI

---

## 17. Critical Bug Fix Log (2026-03-10)

These bugs were found during the investigation session and are now fixed:

| Bug | Severity | Root Cause | Fix |
|-----|----------|-----------|-----|
| Wrong predictions on all images | CRITICAL | `app.py` missing circular mask + border crop in preprocessing | Added `_crop_black_borders()` and `_apply_circular_mask()` |
| Wrong predictions on all images | CRITICAL | `app.py` used `INTER_LINEAR` resize vs training's `INTER_AREA` | Changed to `cv2.INTER_AREA` |
| OOD report crash | HIGH | `ood.ood_threshold` is `None` when npz missing, format string fails | Added None check with graceful fallback |
| CLAHE +43% Glaucoma shift | MODERATE | CLAHE applied blindly vs domain-conditional during training | Documented; fix requires retraining with `unified_preprocessing.py` |

**Investigation methodology:** 4 parallel agents analyzed architecture, preprocessing, raw model outputs, and normalization. Full findings in memory files.
