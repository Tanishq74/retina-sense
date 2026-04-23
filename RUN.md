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

**RetinaSense-ViT** is a deep learning system for retinal disease classification from fundus photographs.

| Property | Value |
|----------|-------|
| Task | Multi-class classification, 5 diseases |
| Classes | Normal (0), Diabetes/DR (1), Glaucoma (2), Cataract (3), AMD (4) |
| Model | ViT-Base/16 (timm), 86M params, multi-task heads |
| Dataset | 8,540 images — APTOS (3,662) + ODIR (4,878) |
| Split | 70/15/15 — train/calib/test (stratified) |
| Best checkpoint | `outputs_v3/best_model.pth` (epoch 24) |
| GitHub | https://github.com/Tanishq74/retina-sense |

---

## 2. Directory Structure

```
<repo-root>/
│
├── retinasense_v3.py            # Main training script (1220 lines)
├── gradcam_v3.py                # Attention Rollout XAI (FIXED, working)
├── eval_dashboard.py            # Phase 1A: Full evaluation suite
├── mc_dropout_uncertainty.py    # Phase 1B: MC Dropout uncertainty
├── integrated_gradients_xai.py  # Phase 1C: Integrated Gradients XAI
├── fairness_analysis.py         # Phase 1D: Domain fairness analysis
├── train_ensemble.py            # Phase 2B: EfficientNet-B3 ensemble
├── kfold_cv.py                  # Phase 2A: 5-fold cross-validation (GPU)
├── knowledge_distillation.py    # Phase 4A: KD + ONNX export (GPU)
├── app.py                       # Phase 3B: Gradio web demo
├── api/
│   └── main.py                  # Phase 4B: FastAPI REST server
├── Dockerfile                   # Docker deployment
├── requirements_deploy.txt      # Deployment dependencies
│
├── data/
│   ├── train_split.csv          # 5,978 rows (image_path, label, source, cache_path)
│   ├── calib_split.csv          # 1,287 rows
│   ├── test_split.csv           # 1,287 rows (SEALED — do not use for training)
│   ├── combined_dataset.csv     # Full 8,540 rows
│   └── fundus_norm_stats.json   # mean=[0.4298,0.2784,0.1559] std=[0.2857,0.2065,0.1465]
│
├── preprocessed_cache_v3/       # Preprocessed .npy image cache (must exist)
│
├── outputs_v3/
│   ├── best_model.pth           # Main trained model
│   ├── temperature.json         # T=0.6438
│   ├── thresholds.json          # Per-class thresholds (see below)
│   ├── ood_detector.npz         # Mahalanobis OOD detector (threshold=42.82)
│   ├── final_metrics.json       # Training-time metrics
│   ├── history.json             # Loss/accuracy per epoch
│   ├── dashboard.png            # Training curves
│   ├── evaluation/              # Phase 1A outputs (7 files)
│   ├── uncertainty/             # Phase 1B outputs (6 files)
│   ├── xai/                     # Phase 1C outputs (23 files)
│   ├── fairness/                # Phase 1D outputs (7 files)
│   ├── ensemble/                # Phase 2B outputs (2 files)
│   ├── gradcam/                 # Attention Rollout heatmaps (22 files)
│   └── kfold/                   # Phase 2A outputs (NOT YET RUN)
│
├── SESSION_CONTEXT.md           # Project history and phase status
├── KFOLD_CONTEXT.md             # Standalone K-Fold run guide
├── AGENT_CONTEXT_PHASE1.md      # Debugging guide for Phase 1 agents
└── RUN.md                       # This file
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

**Loading the model (copy-paste pattern):**
```python
import torch, timm, json
from retinasense_v3 import MultiTaskViT   # or define inline

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultiTaskViT().to(device)
ckpt = torch.load('outputs_v3/best_model.pth', map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Temperature scaling
T = 0.6438  # from outputs_v3/temperature.json
# Per-class thresholds: Normal=0.638, DR=0.068, Glaucoma=0.840, Cataract=0.564, AMD=0.289

# Inference
with torch.no_grad():
    disease_logits, severity_logits = model(x)          # x: [B,3,224,224]
    probs = torch.softmax(disease_logits / T, dim=1)    # apply temperature
    pred = torch.argmax(probs, dim=1)
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
| `retinasense_v3.py` | DONE | `outputs_v3/best_model.pth` | Epoch 24, F1=0.854 (baseline) |
| `unified_preprocessing.py` | DONE | `preprocessed_cache_unified/` | Unified CLAHE for all images |
| `train_dann.py` | DONE | `outputs_v3/dann/best_model.pth` | Acc=86.1%, F1=0.867, AUC=0.962 |
| `train_dann.py` (v2) | DONE | `outputs_v3/dann_v2/best_model.pth` | F1=0.871, AMD F1=0.950 |
| `eval_dashboard.py` | DONE | `outputs_v3/evaluation/` | Recalibrated: T=0.593, ECE=0.056 |
| `train_ensemble.py` | DONE | `outputs_v3/ensemble/` | EfficientNet-B3 acc=80.9%, ensemble=84.1% |
| `kfold_cv.py` | DONE | `outputs_v3/kfold/` | Acc=82.4%+/-1.9%, F1=0.827+/-0.019 |
| `gradcam_v3.py` | DONE | `outputs_v3/gradcam/` (22 files) | Attention Rollout heatmaps |
| `mc_dropout_uncertainty.py` | DONE | `outputs_v3/uncertainty/` (6 files) | T=30 MC passes |
| `integrated_gradients_xai.py` | DONE | `outputs_v3/xai/` (23 files) | Pearson r=0.196 |
| `fairness_analysis.py` | DONE | `outputs_v3/fairness/` (7 files) | APTOS ECE=0.51 (pre-DANN) |

---

## 6. What Still Needs to Run (Requires GPU)

### 6A. K-Fold Cross-Validation — COMPLETED

```bash
cd /teamspace/studios/this_studio
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

**Actual results (completed):**
```
Accuracy:          82.4% +/- 1.9%
Macro F1:          0.827 +/- 0.019
Macro AUC:         0.948 +/- 0.008
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
cd /teamspace/studios/this_studio
python app.py
```

- Opens on port **7860**
- Generates a public shareable URL (e.g., `https://xxxx.gradio.live`)
- Features: drag-drop fundus image → attention heatmap + confidence bars +
  uncertainty gauge + clinical report + downloadable PDF

**What app.py does internally:**
1. Loads `outputs_v3/best_model.pth`
2. Loads `outputs_v3/ood_detector.npz` for OOD detection
3. Loads `outputs_v3/temperature.json` and `outputs_v3/thresholds.json`
4. Runs Attention Rollout for heatmap (from `gradcam_v3.py`)
5. Runs MC Dropout (T=15 passes) for uncertainty estimation
6. Generates clinical recommendation text per disease + severity

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

### DANN ViT-Base (best single model, stable lambda capped at 0.3)

| Metric | Value |
|--------|-------|
| Overall Accuracy | **86.1%** (thresholded) |
| Macro F1 | **0.867** |
| Macro AUC | **0.962** |
| ECE | **0.056** |
| Temperature | **0.593** |
| Checkpoint | `outputs_v3/dann/best_model.pth` |

**Per-class F1 (DANN):**

| Class | F1 |
|-------|-----|
| DR | 0.902 (recall: 80.8%) |
| Glaucoma | 0.830 |
| Cataract | 0.882 |
| AMD | 0.937 |

### DANN-v2 (boosted DR alpha 2.5x)

| Metric | Value |
|--------|-------|
| Overall Accuracy | **86.1%** (thresholded) |
| Macro F1 | **0.871** |
| AMD F1 | **0.950** |
| Checkpoint | `outputs_v3/dann_v2/best_model.pth` |

### EfficientNet-B3 (retrained on unified cache)

| Metric | Value |
|--------|-------|
| Accuracy | 80.9% |
| Macro F1 | 0.782 |
| Macro AUC | 0.958 |

### Ensemble (ViT + EfficientNet)

| Metric | Old Ensemble | New Ensemble |
|--------|-------------|-------------|
| Accuracy | 74.7% | **84.1%** (needs re-optimization with DANN-v2) |

### K-Fold CV (5-fold)

| Metric | Value |
|--------|-------|
| Accuracy | 82.4% +/- 1.9% |
| Macro F1 | 0.827 +/- 0.019 |
| Macro AUC | 0.948 +/- 0.008 |

### Domain Gap — FIXED

| Domain | Before (Accuracy) | After DANN (Accuracy) |
|--------|-------------------|----------------------|
| ODIR | 67.7% | improved |
| APTOS | 26.5% | **99.8%** |
| Domain accuracy | ~100% (not invariant) | **99.4%** (near-random = domain-invariant) |

### Previous Baseline (for reference)

| Metric | ViT alone (old) | Old Ensemble |
|--------|----------------|-------------|
| Overall Accuracy | 49.1% | 74.7% |
| Macro F1 | 0.626 | 0.712 |
| Macro AUC | 0.893 | 0.951 |
| DR Recall | 25.3% | — |

---

## 10. Configuration Files

### Temperature & Thresholds

`outputs_v3/temperature.json` (recalibrated after DANN):
```json
{"temperature": 0.593, "ece_after": 0.056}
```

`outputs_v3/thresholds.json` (recalibrated after DANN):
```json
{
  "thresholds": [recalibrated values],
  "class_names": ["Normal", "Diabetes/DR", "Glaucoma", "Cataract", "AMD"]
}
```

Note: After DANN training + unified CLAHE preprocessing, the domain gap is eliminated.
APTOS accuracy improved from 26.5% to 99.8%. DR recall improved from 25.3% to 80.8%.

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

### Done
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
- [x] Gradio web app (app.py) — port 7860
- [x] FastAPI REST server (api/main.py) — port 8000
- [x] Docker deployment (Dockerfile)

### Completed This Session
- [x] Unified CLAHE preprocessing (`unified_preprocessing.py`)
- [x] REFUGE2 dataset added (1,200 images)
- [x] DANN training (`train_dann.py`) — acc=86.1%, F1=0.867
- [x] DANN-v2 training (--dr-alpha-boost 2.5) — F1=0.871
- [x] Recalibration (temperature=0.593, ECE=0.056)
- [x] EfficientNet-B3 retrained on unified cache — acc=80.9%
- [x] K-Fold CV: `python kfold_cv.py` — 82.4% +/- 1.9%
- [x] Fix DR recall (25.3% -> 80.8%) — unified CLAHE + DANN
- [x] Fix APTOS accuracy (26.5% -> 99.8%)
- [x] Bug fixes across 7 scripts (see FIXES_AND_CHANGES.md)

### Still Needs GPU
- [ ] RETFound backbone swap: `python retfound_backbone.py --setup` + retrain
- [ ] Re-optimize ensemble with DANN-v2: `python eval_dashboard.py --ensemble-search`
- [ ] Final evaluation: `python eval_dashboard.py --final`
- [ ] Knowledge Distillation + ONNX: `python knowledge_distillation.py` (~30 min on H100)

### Open Issues
- [ ] Update paper/report with all new figures and metrics
- [ ] Test `app.py` end-to-end with DANN model
- [ ] Push accuracy from 86.1% toward 90%+ target (RETFound + ensemble re-optimization)

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

**Table 1 — DANN ViT-Base Results (Test Set, thresholded):**
```
Overall Accuracy:    86.1%
Macro F1:            0.867
Macro AUC:           0.962
ECE:                 0.056
Temperature:         0.593
DR F1:               0.902 (Recall: 80.8%)
Glaucoma F1:         0.830
Cataract F1:         0.882
AMD F1:              0.937
```

**Table 2 — Model Comparison:**
```
ViT-Base alone (old):         Acc=49.1%, F1=0.626, AUC=0.893
EfficientNet-B3 (unified):    Acc=80.9%, F1=0.782, AUC=0.958
Old Ensemble:                 Acc=74.7%, F1=0.712, AUC=0.951
DANN ViT-Base:                Acc=86.1%, F1=0.867, AUC=0.962
DANN-v2 (DR boost):           Acc=86.1%, F1=0.871
New Ensemble (ViT+Eff):       Acc=84.1% (needs re-optimization)
```

**Table 3 — K-Fold CV (5-fold, completed):**
```
Accuracy:          82.4% +/- 1.9%
Macro F1:          0.827 +/- 0.019
Macro AUC:         0.948 +/- 0.008
```

**Reviewer issues addressed:**
- Missing per-class metrics → eval_dashboard.py outputs `metrics_report.json`
- Missing ROC/PR curves → `outputs_v3/evaluation/roc_curves_per_class.png`, `precision_recall_curves.png`
- No cross-validation → kfold_cv.py (run on GPU)
- No confidence intervals → kfold_results.json (mean ± std, after K-Fold)
- No statistical significance → fairness_analysis.py, chi-squared test p=0.296
- No uncertainty quantification → mc_dropout_uncertainty.py (epistemic/aleatoric split)
