# RetinaSense-ViT: Functional Specification

**Project**: Multi-disease retinal fundus image classification with domain adaptation, uncertainty estimation, and retrieval-augmented diagnosis.
**Authors**: Tanishq Tamarkar, Rafae Mohammed Hussain, Dr. Revathi M — SRM Institute of Science and Technology, Chennai, India
**Repository**: https://github.com/Tanishq74/retina-sense
**HuggingFace**: https://huggingface.co/tanishq74/retinasense-vit

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Dataset Specifications](#2-dataset-specifications)
3. [Preprocessing Pipeline](#3-preprocessing-pipeline)
4. [Model Architecture](#4-model-architecture)
5. [Training Pipeline — DANN-v3 (Production)](#5-training-pipeline--dann-v3-production)
6. [Training Pipeline — DANN-v4 (RETFound, Experimental)](#6-training-pipeline--dann-v4-retfound-experimental)
7. [Calibration and Threshold Optimization](#7-calibration-and-threshold-optimization)
8. [FAISS Retrieval Index](#8-faiss-retrieval-index)
9. [RAD Evaluation Framework](#9-rad-evaluation-framework)
10. [Confidence Routing System](#10-confidence-routing-system)
11. [Explainability Modules](#11-explainability-modules)
12. [Uncertainty Quantification](#12-uncertainty-quantification)
13. [Fairness Analysis](#13-fairness-analysis)
14. [Evaluation Dashboard](#14-evaluation-dashboard)
15. [Knowledge Distillation](#15-knowledge-distillation)
16. [K-Fold Cross-Validation](#16-k-fold-cross-validation)
17. [Ensemble Inference](#17-ensemble-inference)
18. [Gradio Demo Application](#18-gradio-demo-application)
19. [Configuration Files](#19-configuration-files)
20. [File Inventory](#20-file-inventory)
21. [Performance Benchmarks](#21-performance-benchmarks)
22. [Execution Guide](#22-execution-guide)

---

## 1. System Overview

RetinaSense-ViT is an end-to-end clinical AI framework for automated retinal disease screening. It classifies fundus photographs into five diagnostic categories using a Vision Transformer backbone augmented with Domain-Adversarial Neural Network (DANN) training, post-hoc calibration, retrieval-augmented reasoning, and clinical confidence routing.

### 1.1 Diagnostic Classes

| Index | Class Name | Description |
|---|---|---|
| 0 | Normal | Healthy retina with no detected pathology |
| 1 | Diabetes/DR | Diabetic Retinopathy — microaneurysms, hemorrhages, exudates |
| 2 | Glaucoma | Elevated IOP damage — optic disc cupping, nerve fiber loss |
| 3 | Cataract | Lens opacity — reduced image clarity and contrast |
| 4 | AMD | Age-related Macular Degeneration — drusen, geographic atrophy |

### 1.2 Core Pipeline

```
Fundus Image
     |
     v
Source-conditional Preprocessing (Ben Graham / CLAHE / Resize)
     |
     v
Unified Cache (.npy, 224x224 RGB)
     |
     v
ViT-Base/16 Backbone (86M params)
     |
     +----> Disease Head (768 -> 512 -> 256 -> 5)
     |
     +----> Severity Head (768 -> 256 -> 5) [DR severity only]
     |
     +----> Domain Head via GRL (768 -> 256 -> 64 -> 4) [training only]
     |
     v
Temperature Scaling (T = 0.566)
     |
     v
Per-class Threshold Application
     |
     v
RAD Pipeline:
  - FAISS kNN retrieval (top-5)
  - MC Dropout uncertainty (15 passes)
  - Confidence routing (AUTO-REPORT / REVIEW / ESCALATE)
     |
     v
Clinical Output:
  - Primary diagnosis + confidence
  - DR severity grade
  - Uncertainty estimate
  - Top-5 similar cases
  - Attention heatmap
  - Routing tier
```

### 1.3 Production Model: DANN-v3

| Metric | Value |
|---|---|
| Accuracy | 89.30% |
| Macro F1 | 0.886 |
| Macro AUC | 0.975 |
| ECE (Expected Calibration Error) | 0.034 |
| Temperature | 0.566 |
| Cohen's Kappa | 0.809 |
| Matthews Correlation Coefficient | 0.810 |

Per-class F1: Normal 0.854, DR 0.920, Glaucoma 0.833, Cataract 0.899, AMD 0.895

---

## 2. Dataset Specifications

### 2.1 Sources

| Source | Size | Classes | Camera | Preprocessing |
|---|---|---|---|---|
| APTOS 2019 | 3,662 | DR (5 severity levels) | Aravind field camera | Ben Graham enhancement |
| ODIR | 4,878 | All 5 (single-label filtered) | Multi-source clinical | CLAHE |
| REFUGE2 | 1,240 | Normal + Glaucoma | Zeiss Visucam 500 | Resize only |
| MESSIDOR-2 | 1,744 | Normal + DR | Topcon TRC NW6 | CLAHE |

**Total**: 11,524 images after filtering multi-label ODIR samples.

### 2.2 Data Split

| Split | Size | Purpose |
|---|---|---|
| Train | 8,241 (70%) | Model training |
| Calibration | 1,816 (15%) | Temperature scaling + threshold optimization |
| Test | 1,467 (15%) | Final sealed evaluation |

Stratified split on `disease_label`. Split CSVs persisted to `data/train_split_expanded.csv`, `data/calib_split_expanded.csv`, `data/test_split.csv`.

### 2.3 Class Imbalance

The dataset has a 21:1 imbalance ratio (DR vs. AMD). This is addressed through:
- `WeightedRandomSampler` with inverse-frequency weights in `DataLoader`
- Focal Loss with per-class alpha weights (computed via `compute_class_weight('balanced')`)
- Progressive DR alpha boost: 1.5x at epoch 0 → 3.0x at epoch N
- Hard-example mining: top-500 highest-loss samples oversampled 2x per epoch

### 2.4 Domain Assignment

| Domain Index | Source | Training Samples |
|---|---|---|
| 0 | APTOS | 2,563 |
| 1 | ODIR | 3,414 |
| 2 | REFUGE2 | 868 |
| 3 | MESSIDOR-2 | 1,396 |

### 2.5 Metadata Files

- `final_unified_metadata.csv`: consolidated metadata for all 11,524 images
- `data/train_split_expanded.csv`: training split with cache paths
- `data/calib_split_expanded.csv`: calibration split
- `data/test_split.csv`: sealed test split (never used in training or threshold tuning)

---

## 3. Preprocessing Pipeline

### 3.1 Source-Conditional Dispatcher

Function `preprocess_image(path, source, sz=224)` in `retinasense_v3.py` and `unified_preprocessing.py` dispatches to the appropriate preprocessing method based on the image source.

```python
if source == 'APTOS':
    return ben_graham(path, sz)
elif source == 'REFUGE2':
    return resize_only(path, sz)
else:  # ODIR, MESSIDOR-2
    return clahe_preprocess(path, sz)
```

### 3.2 Ben Graham Enhancement (APTOS)

Designed for Aravind field cameras which suffer from vignetting and uneven illumination:

```
1. Read BGR via cv2, convert to RGB
2. Resize to 224x224
3. GaussianBlur with sigma=10 to create low-frequency estimate
4. addWeighted(img, 4, blurred, -4, 128)  -- amplifies local detail
5. Apply circular mask at radius=0.48*sz to remove black border
```

Rationale: Removes global illumination gradient while preserving vessel and microaneurysm detail critical for DR grading.

### 3.3 CLAHE Preprocessing (ODIR, MESSIDOR-2)

Contrast Limited Adaptive Histogram Equalization for multi-source clinical images:

```
1. Read BGR, convert to RGB
2. Resize to 224x224
3. Convert RGB to LAB color space
4. Apply CLAHE (clipLimit=2.0, tileGridSize=8x8) to L channel only
5. Convert LAB back to RGB
6. Apply circular mask at radius=0.48*sz
```

Rationale: Normalizes local contrast across cameras without affecting hue or oversaturating colors. Only the luminance channel is modified.

### 3.4 Resize Only (REFUGE2)

Zeiss Visucam 500 images are already standardized clinical quality:

```
1. Read BGR, convert to RGB
2. Resize to 224x224
3. Apply circular mask at radius=0.48*sz
```

### 3.5 Cache System

Preprocessed images are stored as `.npy` files (uint8, shape 224x224x3) in `preprocessed_cache_unified/`.

Cache key: `{stem}_{img_size}.npy` where stem is the filename without extension.

Resolution order when loading cached images:
1. `preprocessed_cache_unified/{stem}_224.npy`
2. `preprocessed_cache_v3/{stem}_224.npy`
3. On-the-fly preprocessing as fallback

### 3.6 Normalization Statistics

Fundus-specific per-channel statistics stored at `configs/fundus_norm_stats_unified.json`:
```json
{
    "mean_rgb": [mean_R, mean_G, mean_B],
    "std_rgb": [std_R, std_G, std_B]
}
```

Fallback chain: `configs/fundus_norm_stats_unified.json` → `configs/fundus_norm_stats.json` → `data/fundus_norm_stats.json` → ImageNet defaults `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`.

---

## 4. Model Architecture

### 4.1 ViT-Base/16 Backbone

- **Model**: `vit_base_patch16_224` from `timm`
- **Parameters**: 86M total
- **Input**: 224x224 RGB
- **Patch size**: 16x16 → 196 patches + 1 CLS token = 197 tokens
- **Transformer blocks**: 12 blocks
- **Hidden dimension**: 768
- **Attention heads**: 12
- **Output**: CLS token embedding of dimension 768

Loaded with `pretrained=True` for training. `pretrained=False` for inference (weights loaded from checkpoint).

### 4.2 Disease Classification Head

```
Linear(768, 512)
BatchNorm1d(512)
ReLU()
Dropout(0.3)
Linear(512, 256)
BatchNorm1d(256)
ReLU()
Dropout(0.2)
Linear(256, 5)   -> logits for 5 disease classes
```

### 4.3 Severity Grading Head

```
Linear(768, 256)
BatchNorm1d(256)
ReLU()
Dropout(0.3)
Linear(256, 5)   -> logits for DR severity (0-4)
```

Only supervised for APTOS samples; ODIR/REFUGE2/MESSIDOR-2 use `severity_label=-1` which is ignored by `CrossEntropyLoss(ignore_index=-1)`.

### 4.4 Domain Discriminator Head (DANN training only)

```
GRL (Gradient Reversal Layer, alpha=lambda_p)
Linear(768, 256)
ReLU()
Dropout(0.3)
Linear(256, 64)
ReLU()
Linear(64, num_domains)  -> 4 classes
```

The GRL negates gradients during backprop, forcing the backbone to learn domain-invariant features. The lambda schedule follows Ganin et al. 2016 sigmoid ramp, capped at 0.3.

### 4.5 Gradient Reversal Layer

```python
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None
```

### 4.6 Multi-Task Forward Pass (Training)

```python
def forward(self, x, alpha=1.0):
    f = self.backbone(x)     # (B, 768)
    f = self.drop(f)
    disease_out  = self.disease_head(f)
    severity_out = self.severity_head(f)
    domain_out   = self.domain_head(self.grl(f, alpha))
    return disease_out, severity_out, domain_out
```

### 4.7 Inference Forward Pass

```python
def forward_no_domain(self, x):
    f = self.backbone(x)
    f = self.drop(f)
    return self.disease_head(f), self.severity_head(f)
```

### 4.8 Feature Extraction

```python
def forward_features(self, x):
    return self.backbone(x)  # raw CLS token for FAISS embedding
```

---

## 5. Training Pipeline — DANN-v3 (Production)

**Script**: `train_dann_v3.py`
**Output directory**: `outputs_v3/dann_v3/`
**Checkpoint**: `outputs_v3/dann_v3/best_model.pth`

### 5.1 Hyperparameters

| Parameter | Value |
|---|---|
| Backbone | vit_base_patch16_224 |
| Batch size | 32 (effective 64 with gradient accumulation) |
| Epochs | 40 (patience=15 on macro-F1) |
| Base LR | 3e-5 |
| LLRD decay | 0.85 |
| Weight decay | 1e-4 |
| Focal gamma | 2.0 |
| Label smoothing | 0.1 |
| Mixup alpha | 0.2, probability 0.5 |
| DR alpha start | 1.5x |
| DR alpha end | 3.0x |
| Hard mining K | 500 |
| Hard mining factor | 2x |
| Max DANN lambda | 0.3 |
| Scheduler | CosineAnnealingWarmRestarts (T0=10, Tmult=2) |
| TTA | 8-way (flips + rotations + zoom) |

### 5.2 Loss Function

**Total loss per step**:
```
L = L_disease + 0.2 * L_severity + DOMAIN_WEIGHT * lambda_p * L_domain
```

Where:
- `L_disease` = Focal Loss with class weights + label smoothing (alpha=per-class, gamma=2.0)
- `L_severity` = CrossEntropy with ignore_index=-1 (only supervised for APTOS)
- `L_domain` = CrossEntropy for 4-class domain discrimination
- `DOMAIN_WEIGHT` = 0.05, `lambda_p` = Ganin progressive schedule (0→0.3)

**Focal Loss** with label smoothing:
```python
ce = F.cross_entropy(logits, targets, reduction='none', label_smoothing=0.1)
pt = torch.exp(-ce)
focal = ((1 - pt) ** gamma) * ce
if alpha is not None:
    focal = alpha[targets] * focal
return focal.mean()
```

### 5.3 Layer-wise Learning Rate Decay (LLRD)

AdamW optimizer with 14 separate parameter groups:
- Head parameters (disease + severity + domain + dropout): `base_lr = 3e-5`
- Transformer block[11] (closest to output): `base_lr * 0.85^1`
- Transformer block[10]: `base_lr * 0.85^2`
- ...
- Transformer block[0]: `base_lr * 0.85^12`
- Patch embedding + cls_token + pos_embed + norm: `base_lr * 0.85^13`

This gives the patch embedding an effective LR of approximately 2.7e-7, preventing catastrophic forgetting of ImageNet representations.

### 5.4 Hard-Example Mining Sampler

`HardExampleMiningWeightedSampler` maintains per-sample loss and misclassification state across epochs:

**Base weights**: Inverse class frequency (same as WeightedRandomSampler).

**After each epoch**:
1. Record per-sample loss (no reduction, forward only on original labels)
2. Find top-K highest-loss samples → multiply weight by `hard_factor=2`
3. Find DR→Normal misclassifications → additionally multiply by `hard_factor=2`
4. Rebuild sampling distribution

This ensures the hardest examples (especially DR misclassified as Normal) are seen more frequently in subsequent epochs.

### 5.5 MixUp Augmentation

Applied with probability 0.5 per batch:
```python
lam = Beta(0.2, 0.2)
index = randperm(B)
mixed_x = lam * x + (1 - lam) * x[index]
loss_d = lam * criterion_d(out, y_a) + (1-lam) * criterion_d(out, y_b)
```

Domain labels are NOT mixed. The domain discriminator always sees original image features to maintain clean domain supervision.

### 5.6 Progressive DR Alpha Schedule

```python
dr_boost = alpha_start + (alpha_end - alpha_start) * (epoch / (total_epochs - 1))
alpha_this_epoch = base_alpha.clone()
alpha_this_epoch[1] *= dr_boost  # index 1 = DR class
criterion_d.alpha.copy_(alpha_this_epoch)
```

Ramps DR focal weight from 1.5x to 3.0x across epochs, increasing attention to hard DR cases as training progresses.

### 5.7 Ganin Lambda Schedule

```python
def ganin_lambda(epoch, total_epochs, max_lambda=0.3):
    p = epoch / total_epochs
    raw = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0
    return min(raw, max_lambda)
```

Prevents backbone destabilization during early training when domain gradients would otherwise be too large.

### 5.8 Test-Time Augmentation (8-way TTA)

Applied during final evaluation:
1. Identity (no transform)
2. Horizontal flip
3. Vertical flip
4. Horizontal + vertical flip
5. 90-degree rotation
6. 180-degree rotation
7. 270-degree rotation
8. Center crop (200px) + resize to 224

Probabilities from all 8 passes are averaged before applying temperature scaling and thresholds.

### 5.9 Warm-Start Loading

`load_warmstart_weights()` loads compatible weight tensors from a previous checkpoint, skipping domain head weights if the number of domains changed. Reports:
- Number of loaded parameter tensors
- Number of skipped tensors (shape mismatch or missing key)
- Number of randomly initialized tensors

### 5.10 Domain Discriminator Evaluation

`evaluate_domain(loader, model, device, alpha)` evaluates how well the domain head distinguishes sources. As DANN training converges, domain accuracy should approach chance level (25% for 4 domains), indicating domain-invariant features.

### 5.11 Training Step Detail

For each batch:
```
1. Load (imgs, disease_label, severity_label, domain_label, sample_idx)
2. Apply MixUp with prob=0.5 to imgs and disease labels
3. Forward mixed images through full model (with GRL alpha=lambda_p)
4. Compute mixed disease loss + severity loss
5. Forward ORIGINAL images through backbone + domain head
6. Compute domain loss on original features
7. Total loss = disease + 0.2*severity + 0.05*lambda_p*domain
8. Divide by gradient_accumulation=2 before backward
9. Accumulate gradients for 2 steps, then:
   a. unscale (AMP)
   b. clip grad norm to 1.0
   c. optimizer step
   d. scheduler step (per epoch, not per batch)
10. Record per-sample losses for hard-example mining
```

### 5.12 Checkpoint Contents

Saved when macro-F1 improves by more than `min_delta=0.001`:
```python
{
    'epoch': int,
    'model_state_dict': dict,
    'val_acc': float,
    'macro_f1': float,
    'domain_map': dict,
    'num_domains': int,
    'history': dict,
    'args': dict,
    'dr_alpha_boost': float,
}
```

---

## 6. Training Pipeline — DANN-v4 (RETFound, Experimental)

**Script**: `train_dann_v4.py`
**Target**: Push from 89.3% (v3) to 92-93%
**Output directory**: `outputs_v3/dann_v4/`
**Status**: Script complete, awaiting GPU training.

### 6.1 RETFound Backbone

- **Architecture**: ViT-Large/16
- **Parameters**: 304M (3.5x larger than ViT-Base)
- **Transformer blocks**: 24 (vs. 12 in v3)
- **Hidden dimension**: 1024 (vs. 768 in v3)
- **Pretraining**: Masked Autoencoding on 1.6M retinal fundus images from EyePACS
- **Weights file**: `weights/RETFound_cfp_weights.pth` (1.2 GB)
- **Verified**: 294/294 backbone keys load correctly

Loading RETFound requires matching ViT-Large architecture exactly. The initial attempt failed (0/294 keys loaded) because ViT-Base (dim=768) was used instead of ViT-Large (dim=1024).

### 6.2 CutMix + MixUp Combination

Per-batch augmentation policy:
- 40% of batches: MixUp (alpha=0.4)
- 40% of batches: CutMix (alpha=1.0)
- 20% of batches: clean (no mixing)

CutMix replaces a rectangular region in one image with the corresponding region from another:
```python
def cutmix(x, y, alpha=1.0):
    lam = Beta(alpha, alpha)
    rx, ry, rw, rh = rand_bbox(H, W, lam)
    x[:, :, ry:ry+rh, rx:rx+rw] = x[index, :, ry:ry+rh, rx:rx+rw]
    lam = 1 - (rw * rh) / (W * H)
    return x, y, y[index], lam
```

### 6.3 Stochastic Weight Averaging (SWA)

Applied in the last 10 epochs:
```python
swa_model = AveragedModel(model)
swa_scheduler = SWALR(optimizer, swa_lr=1e-6)
# After each epoch in SWA phase:
swa_model.update_parameters(model)
# After training ends:
update_bn(train_loader, swa_model, device)  # update BatchNorm stats
```

SWA averages model weights over multiple optima to find flatter loss basin, improving generalization.

### 6.4 Class-Aware Augmentation

Stronger augmentation applied to minority classes (Glaucoma, Cataract, AMD) dynamically identified each epoch based on class counts in the current batch. Augmentation includes rotation up to 30 degrees, stronger color jitter, and random erasing probability increased to 0.4.

### 6.5 LLRD for ViT-Large

```
24 transformer blocks (vs. 12 in v3)
LLRD decay = 0.80 (vs. 0.85 in v3, more aggressive due to larger model)
Base LR = 1e-5 (vs. 3e-5, lower for domain-specific pretraining)
Dropout = 0.2 (vs. 0.3, less needed due to larger model capacity)
```

---

## 7. Calibration and Threshold Optimization

### 7.1 Temperature Scaling

Applied post-training on the calibration set.

**Objective**: Minimize Negative Log-Likelihood:
```
T* = argmin_T NLL(logits / T, labels)
```

Solved via `scipy.optimize.minimize_scalar` with bounds `(0.01, 10.0)`.

**Production value**: T = 0.566 (stored in `configs/temperature.json`)

**Effect**: ECE reduced from ~0.09 pre-scaling to 0.034 post-scaling.

### 7.2 Expected Calibration Error (ECE)

```python
def compute_ece(probs, labels, n_bins=15):
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies  = predictions == labels
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0: continue
        ece += mask.sum() * abs(accuracies[mask].mean() - confidences[mask].mean())
    return ece / len(labels)
```

### 7.3 Per-Class Threshold Optimization

Grid search over 50 threshold values in [0.05, 0.95] for each class independently:

```python
for c in range(n_classes):
    binary_labels = (labels == c).astype(int)
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, 50):
        preds_c = (probs[:, c] >= t).astype(int)
        f = f1_score(binary_labels, preds_c, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    thresholds.append(best_t)
```

Thresholds stored in `configs/thresholds.json`.

### 7.4 Threshold Application

```python
def apply_thresholds(probs, thresholds):
    for prob_row in probs:
        above = [i for i, (p, t) in enumerate(zip(prob_row, thresholds)) if p >= t]
        if above:
            preds.append(above[argmax([prob_row[i] for i in above])])
        else:
            preds.append(argmax(prob_row))  # fallback to argmax
```

Assigns the class with the highest probability among all classes that exceed their respective threshold. Falls back to simple argmax if no class exceeds its threshold.

---

## 8. FAISS Retrieval Index

### 8.1 Index Construction

**Script**: `rebuild_faiss_full.py`
**Index type**: `faiss.IndexFlatIP` (exact cosine similarity via inner product on L2-normalized vectors)
**Vectors**: 8,241 training set backbone embeddings (dim=768)
**File**: `outputs_v3/retrieval/index_flat_ip.faiss` (24.1 MB)

```python
# Build process:
1. Load model (DANN-v3 best_model.pth)
2. For each training image: extract backbone features (768-dim CLS token)
3. L2-normalize all vectors: faiss.normalize_L2(embeddings)
4. Create IndexFlatIP and add all vectors
5. Save index and metadata.json (label, class_name, source per vector)
```

### 8.2 Metadata Structure

`outputs_v3/retrieval/metadata.json`: list of 8,241 dicts:
```json
{
    "label": 0,
    "class_name": "Normal",
    "source": "ODIR",
    "image_path": "...",
    "cache_path": "..."
}
```

### 8.3 Retrieval at Inference

```python
query_embedding = model.backbone(img)  # (1, 768)
faiss.normalize_L2(query_embedding)
distances, indices = index.search(query_embedding, k=5)
# distances are cosine similarities in [-1, 1]
# indices are row indices into metadata.json
```

### 8.4 Updating the Index

When MESSIDOR-2 was added, `update_faiss_messidor2.py` was used to append MESSIDOR-2 embeddings to the existing index without rebuilding from scratch.

### 8.5 Index Class Distribution (After Rebuild)

All 5 classes represented: Normal, DR, Glaucoma, Cataract, AMD.
Pre-rebuild bug: AMD class was missing entirely, causing all AMD queries to retrieve incorrect neighbors.

---

## 9. RAD Evaluation Framework

**Script**: `rad_evaluation.py`
**Output**: `outputs_v3/retrieval/rad_evaluation_results.json`

### 9.1 Test Embedding Extraction

For each test sample, extracts:
- `features` from `model.backbone(batch_tensor)` — for FAISS search
- `disease_logits` from `model.disease_head(model.drop(features))` — for model predictions

### 9.2 Retrieval Metrics

**Recall@K**: Fraction of queries with at least one same-class retrieval in top-K.

```python
top_k_labels = retrieved_labels[:, :k]  # (N, k)
matches = (top_k_labels == test_labels[:, None])  # (N, k) bool
recall = matches.any(axis=1).mean()
```

**Precision@K**: Average fraction of correct retrievals in top-K.

**Mean Average Precision (MAP)**:
```python
for each query:
    relevant = (retrieved_labels == true_label)
    cum_relevant = cumsum(relevant)
    precisions = cum_relevant / arange(1, max_k+1)
    ap = sum(precisions * relevant) / relevant.sum()
map = mean(all_aps)
```

### 9.3 Results

| Metric | Value |
|---|---|
| MAP | 0.921 |
| Recall@1 | 94.0% |
| Recall@3 | 97.8% |
| Recall@5 | 98.5% |
| Recall@10 | 99.3% |

Per-class Average Precision: DR 0.952, Normal 0.906, Cataract 0.833, AMD 0.819, Glaucoma 0.742

### 9.4 RAD Combined Accuracy

Combines model probabilities with kNN similarity-weighted votes:
```python
# kNN vote (similarity-weighted):
for j in range(k):
    lbl = retrieved_labels[i, j]
    weight = max(0.0, float(distances[i, j]))  # cosine similarity
    knn_probs[i, lbl] += weight
knn_probs[i] /= knn_probs[i].sum()

# Combination:
combined_probs = alpha * model_probs + (1 - alpha) * knn_probs
combined_preds = combined_probs.argmax(axis=1)
```

Default alpha=0.5. RAD combined accuracy at K=1: **94.0%** (+4.9% over standalone model).

### 9.5 Agreement Analysis

At K=5, model and kNN agree on 92.3% of predictions. When they agree, accuracy is 97.3%. When they disagree, accuracy drops to 61.2%, indicating retrieval disagreement is a strong signal for uncertainty.

### 9.6 Class-Match Heatmap

For each true class, shows what fraction of top-5 retrieved cases belong to each class. Strong diagonal indicates the retrieval index well separates the disease classes. Off-diagonal concentrations indicate confusable pairs (e.g., Normal ↔ Glaucoma due to optic disc similarity).

---

## 10. Confidence Routing System

**Script**: `confidence_routing.py`
**Output**: `outputs_v3/retrieval/confidence_routing_results.json`

### 10.1 Routing Logic

The `ConfidenceRouter` class routes each prediction using three signals:

```python
def route(self, confidence, entropy, retrieval_agrees):
    # ESCALATE: low confidence OR high uncertainty
    if confidence < conf_low or entropy > entropy_high:
        return "ESCALATE"
    # AUTO-REPORT: high confidence AND low uncertainty AND retrieval agrees
    if confidence >= conf_high and entropy < entropy_low and retrieval_agrees:
        return "AUTO-REPORT"
    # Everything else: REVIEW
    return "REVIEW"
```

### 10.2 Default Thresholds

| Parameter | Value | Meaning |
|---|---|---|
| conf_high | 0.85 | Above this → auto-report candidate |
| conf_low | 0.50 | Below this → escalate |
| entropy_low | 0.5 nats | Below this → low uncertainty |
| entropy_high | 1.0 nats | Above this → high uncertainty |

All thresholds are CLI-configurable.

### 10.3 MC Dropout Uncertainty

Run backbone once (deterministic), then run disease_head with dropout enabled for `mc_passes=15` iterations:

```python
features = model.backbone(batch_tensor)  # deterministic
for m in model.modules():
    if isinstance(m, nn.Dropout):
        m.train()  # re-enable dropout
all_probs = []
for _ in range(mc_passes):
    f_dropped = model.drop(features)
    logits = model.disease_head(f_dropped)
    probs = softmax(logits / temperature)
    all_probs.append(probs)
mean_probs = stack(all_probs).mean(axis=0)
entropy = -sum(mean_probs * log(mean_probs + 1e-10), axis=1)
```

### 10.4 Production Routing Results

| Tier | Fraction | Accuracy |
|---|---|---|
| AUTO-REPORT | 76.9% | 96.8% |
| REVIEW | 21.4% | 65.6% |
| ESCALATE | 1.7% | 44.0% |

**Error catch rate**: 77.2% of all incorrect predictions are routed to REVIEW or ESCALATE rather than AUTO-REPORT. Only 22.8% of errors slip through to AUTO-REPORT.

**Safety guarantee**: The AUTO-REPORT tier has 96.8% accuracy, meaning only 3.2% of auto-generated reports are incorrect. For the REVIEW tier, 34.4% are incorrect — these are flagged for specialist attention.

### 10.5 Per-Class Routing Behavior

Classes with high inter-class confusion (Glaucoma, Normal) route proportionally more cases to REVIEW than high-confidence classes (DR, AMD). This aligns with the clinical reality that DR has distinctive lesion patterns while Glaucoma's optic disc changes can resemble Normal variation.

---

## 11. Explainability Modules

### 11.1 Grad-CAM v3

**Script**: `gradcam_v3.py`
**Method**: Gradient-weighted Class Activation Mapping adapted for Vision Transformers.

For ViT, attention rollout is used rather than spatial feature maps:
1. Extract attention weights from all 12 transformer blocks
2. Compute attention rollout: recursively multiply attention matrices
3. Average over attention heads
4. Reshape 14x14 attention map back to spatial domain
5. Resize to 224x224 via bilinear interpolation
6. Overlay on original image with colormap

Output: `outputs_v3/gradcam/` directory with per-class activation maps.

### 11.2 Integrated Gradients

**Script**: `integrated_gradients_xai.py`
**Method**: Axiomatic attribution method (Sundararajan et al. 2017).

```
1. Select baseline image (black / Gaussian noise)
2. Interpolate N=50 steps from baseline to input
3. Compute gradients at each interpolation step
4. Average gradients across steps
5. Multiply by (input - baseline) to get attributions
6. Normalize to [0, 1] for visualization
```

Produces pixel-level importance maps highlighting which retinal regions drove the prediction. Applied per-class to show why a specific disease was predicted vs. not predicted.

### 11.3 Attention Visualization in Demo

In `app.py`, the ViT's final-block attention weights are extracted and visualized:
```python
def get_attention_map(model, img_tensor):
    hooks = []
    attn_weights = []
    def hook_fn(module, input, output):
        attn_weights.append(output)
    for block in model.backbone.blocks[-1:]:
        hooks.append(block.attn.register_forward_hook(hook_fn))
    with torch.no_grad():
        model(img_tensor)
    for h in hooks:
        h.remove()
    # Process attention from CLS token to all patches
    attn = attn_weights[0].mean(1)  # mean over heads
    cls_attn = attn[0, 0, 1:]       # CLS attention to 196 patches
    attn_map = cls_attn.reshape(14, 14).numpy()
    return cv2.resize(attn_map, (224, 224))
```

---

## 12. Uncertainty Quantification

**Script**: `mc_dropout_uncertainty.py`

### 12.1 Monte Carlo Dropout

Beyond the routing application, full uncertainty analysis is available:
- **Predictive entropy**: `H[y|x] = -sum(p * log(p))`
- **Mutual information** (epistemic uncertainty): `I[y, omega|x] = H[y|x] - E_omega[H[y|x,omega]]`
- **Aleatoric uncertainty**: `E_omega[H[y|x,omega]]`

### 12.2 Uncertainty Calibration

Evaluates whether high entropy predictions are indeed more likely to be wrong:
- Samples sorted by entropy
- Accuracy vs. entropy scatter plot
- Selective prediction: accuracy at various entropy cutoffs

### 12.3 OOD Detection

Fitted on training embeddings using Mahalanobis distance. Samples with distance > threshold are flagged as out-of-distribution. Current status: OOD detector is stale (fitted on old preprocessing). Images from non-retinal sources or extremely low quality are flagged.

---

## 13. Fairness Analysis

**Script**: `fairness_analysis.py`
**Output**: `outputs_v3/fairness/`

### 13.1 Metrics Computed

- **Per-source accuracy and F1**: Compares model performance across APTOS, ODIR, REFUGE2, MESSIDOR-2
- **Demographic parity**: Whether different source distributions affect prediction rates
- **Equal opportunity**: Per-class true positive rates across sources
- **Calibration fairness**: ECE computed separately for each source to detect differential miscalibration

### 13.2 LODO Validation (Leave-One-Dataset-Out)

Measures generalization to unseen clinical sites:

| Held-Out Dataset | Accuracy | Weighted F1 | Classes |
|---|---|---|---|
| APTOS | 70.8% | 0.829 | DR only |
| MESSIDOR-2 | 61.6% | 0.633 | Normal + DR |
| ODIR | 51.8% | 0.439 | All 5 |
| REFUGE2 | 88.8% | 0.904 | Normal + Glaucoma |
| Average | 68.2% | 0.701 | — |

Results stored at `outputs_v3/lodo_results.json`. Note: lower LODO accuracy is expected because these models were trained without the held-out domain. The production model trained on all 4 domains significantly outperforms these LODO estimates on the full test set.

---

## 14. Evaluation Dashboard

**Script**: `eval_dashboard.py`
**Output**: `outputs_v3/evaluation/`

### 14.1 Dashboard Components

6-panel matplotlib figure generated after each training run:
1. **Loss curves**: Train and validation loss per epoch with domain loss overlaid
2. **Accuracy curves**: Train and calibration accuracy per epoch
3. **Domain accuracy**: How close domain discriminator accuracy is to chance (25%) — lower is better
4. **F1 scores**: Macro and weighted F1 per epoch
5. **Per-class F1**: Individual F1 trajectory for each of the 5 classes
6. **Schedules**: Lambda (DANN) and DR alpha boost trajectories

### 14.2 Test Evaluation Output

- Confusion matrix (raw counts and normalized by row)
- ROC curves for all 5 classes with per-class AUC annotations
- Calibration reliability diagram (bar chart of accuracy vs. confidence per bin)
- Classification report with precision, recall, F1, support per class

### 14.3 Error Analysis

`run_error_analysis.py` loads the test set and identifies:
- Most frequent misclassification pairs
- Highest-confidence wrong predictions
- Lowest-confidence correct predictions
- Distribution of errors across sources

---

## 15. Knowledge Distillation

**Script**: `knowledge_distillation.py`
**Purpose**: Compress DANN-v3 (86M params) to a smaller student model for deployment on edge devices.

### 15.1 Distillation Setup

- **Teacher**: `DANNMultiTaskViT` (ViT-Base, 86M)
- **Student**: Configurable — default MobileViT-S or ViT-Tiny
- **Temperature**: 4.0 for soft label distillation
- **Alpha**: 0.7 (70% distillation loss + 30% hard label loss)

### 15.2 Loss Function

```
L_distill = alpha * T^2 * KLDiv(log_softmax(s/T), softmax(t/T))
          + (1-alpha) * CrossEntropy(s, hard_labels)
```

Where `s` and `t` are student and teacher logits, and `T=4.0` softens the teacher's probability distribution.

### 15.3 Output

Smaller model checkpoint saved to `outputs_v3/distilled/`. Allows inference on devices without GPU by reducing parameter count from 86M to ~5-12M while retaining most of the accuracy.

---

## 16. K-Fold Cross-Validation

**Script**: `kfold_cv.py`

### 16.1 Configuration

- **Folds**: 5-fold stratified on disease label
- **Model**: DANNMultiTaskViT (v3 config)
- **Per-fold output**: checkpoint, history.json, fold_metrics.json

### 16.2 Results (5-fold, GPU-trained)

| Metric | Mean | Std Dev |
|---|---|---|
| Accuracy | 82.4% | ±1.9% |
| Macro F1 | 0.827 | ±0.019 |
| Macro AUC | 0.948 | ±0.008 |

Results demonstrate consistent generalization. The gap between 5-fold CV (82.4%) and final test set performance (89.3%) is due to the full model being trained on all 70% of data vs. 56% per fold.

---

## 17. Ensemble Inference

**Script**: `ensemble_inference.py`
**Training script**: `train_ensemble.py`

### 17.1 Ensemble Strategy

Combines predictions from multiple model checkpoints using soft voting:
- DANN-v3 best_model.pth (primary)
- DANN-v2 best_model.pth
- ViT baseline (retinasense_v3.py output)

Each model outputs calibrated probabilities (post-temperature-scaling). Final prediction is the argmax of the mean probability vector.

### 17.2 Weight Optimization

Optional: learn per-model weights that maximize calibration set F1:
```python
weights = softmax(learnable_logits)  # constrained to sum to 1
combined = sum(w_i * probs_i for w_i, probs_i in zip(weights, all_probs))
```

### 17.3 Ensemble Results

The 3-model ensemble achieves 84.8% macro F1. Note: this is below single DANN-v3 (88.6%) because the ensemble includes weaker models that pull down the average. A DANN-v3 + DANN-v4 ensemble would likely exceed both individually.

---

## 18. Gradio Demo Application

**Script**: `app.py`
**Interface**: Gradio web application

### 18.1 Inputs

- Fundus image upload (JPEG, PNG — any standard fundus format)
- Optional: specify source for optimal preprocessing selection

### 18.2 Inference Pipeline

```python
1. Read image via PIL → convert to RGB numpy array
2. CLAHE preprocessing (all inputs default to CLAHE for the demo)
3. Compose: ToTensor() + Normalize(NORM_MEAN, NORM_STD)
4. model.forward_no_domain(img_tensor)  -> disease_logits, severity_logits
5. Apply temperature: probs = softmax(disease_logits / T_OPT)
6. Apply per-class thresholds: pred = apply_thresholds(probs, THRESHOLDS)
7. Extract attention map from final ViT block
8. Overlay attention on original image
9. Generate clinical report text
```

### 18.3 Outputs

- **Predicted disease** with confidence percentage
- **DR severity grade** (0-4) with confidence
- **Probability bar chart** for all 5 classes
- **Attention heatmap** overlaid on fundus image
- **Uncertainty estimate** from 10 MC Dropout passes
- **OOD warning** if image is out-of-distribution
- **Downloadable PDF report** with findings

### 18.4 Model Loading

Priority order for model selection:
1. `outputs_v3/dann_v3/best_model.pth` (primary)
2. `outputs_v3/dann/best_model.pth`
3. `outputs_v3/dann_v2/best_model.pth`
4. `outputs_v3/best_model.pth`

DANN keys (`domain_head.*`, `grl.*`) are filtered before loading into `MultiTaskViT`. This allows using DANN-trained checkpoints for pure inference.

### 18.5 API Server

FastAPI server in `api/` directory exposing:
- `POST /predict` — JSON with base64-encoded image, returns disease, confidence, uncertainty
- `GET /health` — service health check
- `GET /classes` — class name list
- Swagger UI at `/docs`

---

## 19. Configuration Files

### 19.1 configs/temperature.json

```json
{
    "temperature": 0.566,
    "ece_before": 0.0934,
    "ece_after": 0.034,
    "model_version": "dann_v3"
}
```

### 19.2 configs/thresholds.json

```json
{
    "thresholds": [t_normal, t_dr, t_glaucoma, t_cataract, t_amd],
    "class_names": ["Normal", "Diabetes/DR", "Glaucoma", "Cataract", "AMD"],
    "model_version": "dann_v3"
}
```

### 19.3 configs/fundus_norm_stats_unified.json

```json
{
    "mean_rgb": [R_mean, G_mean, B_mean],
    "std_rgb": [R_std, G_std, B_std]
}
```

Computed from the full 11,524 image training corpus after source-conditional preprocessing.

---

## 20. File Inventory

### 20.1 Training Scripts

| File | Purpose |
|---|---|
| `retinasense_v3.py` | ViT-Base production training (pre-DANN baseline) |
| `train_dann.py` | DANN-v1: domain-adversarial training, 3 domains |
| `train_dann_v3.py` | DANN-v3: production training with all 8 improvements |
| `train_dann_v4.py` | DANN-v4: RETFound backbone + SWA + CutMix (experimental) |
| `train_ensemble.py` | Ensemble training and weight optimization |
| `kfold_cv.py` | 5-fold cross-validation |
| `knowledge_distillation.py` | Teacher-student distillation |

### 20.2 Evaluation Scripts

| File | Purpose |
|---|---|
| `rad_evaluation.py` | FAISS retrieval metrics (MAP, Recall@K, RAD accuracy) |
| `confidence_routing.py` | Clinical routing evaluation (3 tiers) |
| `mc_dropout_uncertainty.py` | MC Dropout uncertainty quantification |
| `fairness_analysis.py` | Cross-source and cross-demographic fairness metrics |
| `eval_dashboard.py` | Training dashboard and test evaluation visualization |
| `run_paper_experiments.py` | Ablation study + LODO experiments for paper |
| `run_error_analysis.py` | Misclassification analysis |
| `tta_evaluation.py` | Standalone TTA evaluation |

### 20.3 Preprocessing Scripts

| File | Purpose |
|---|---|
| `unified_preprocessing.py` | Unified preprocessing pipeline for all 4 sources |
| `prepare_datasets.py` | Dataset download helpers and metadata building |
| `enhanced_augmentation.py` | Advanced augmentation strategies |

### 20.4 Explainability Scripts

| File | Purpose |
|---|---|
| `gradcam_v3.py` | Grad-CAM and attention rollout visualization |
| `integrated_gradients_xai.py` | Integrated Gradients attribution |

### 20.5 Index Management

| File | Purpose |
|---|---|
| `rebuild_faiss_full.py` | Rebuild FAISS index from scratch (all 5 classes) |
| `update_faiss_messidor2.py` | Append MESSIDOR-2 embeddings to existing index |

### 20.6 Application

| File | Purpose |
|---|---|
| `app.py` | Gradio demo application |
| `api/` | FastAPI server |
| `Dockerfile` | Container for deployment |
| `requirements_deploy.txt` | Production dependencies |
| `setup.sh` | Environment setup + HuggingFace model download |

### 20.7 Model Backbone

| File | Purpose |
|---|---|
| `retfound_backbone.py` | RETFound ViT-Large adapter and weight loading |
| `weights/RETFound_cfp_weights.pth` | RETFound pretrained weights (1.2 GB) |

### 20.8 Output Directories

| Directory | Contents |
|---|---|
| `outputs_v3/dann_v3/` | Production model checkpoint, history, plots |
| `outputs_v3/dann_v4/` | Experimental RETFound model (post-training) |
| `outputs_v3/retrieval/` | FAISS index, metadata, RAD evaluation results |
| `outputs_v3/gradcam/` | Attention and Grad-CAM visualizations |
| `outputs_v3/uncertainty/` | MC Dropout analysis results |
| `outputs_v3/fairness/` | Fairness metrics and cross-source analysis |
| `outputs_v3/evaluation/` | Test evaluation plots and reports |
| `outputs_v3/kfold/` | Per-fold checkpoints and metrics |
| `preprocessed_cache_unified/` | Cached preprocessed images (224x224 .npy, ~multi-GB) |

### 20.9 Paper

| File | Purpose |
|---|---|
| `paper/retinasense_ieee.tex` | Complete IEEE paper (700 lines, 0 placeholders) |
| `paper/retinasense_ieee.pdf` | Compiled PDF |
| `paper/figures/` | All figures referenced in paper |
| `paper/generate_figures.py` | Figure generation script |

---

## 21. Performance Benchmarks

### 21.1 Model Comparison

| Model | Accuracy | Macro F1 | AUC | Params | Notes |
|---|---|---|---|---|---|
| **DANN-v3 (production)** | **89.30%** | **0.886** | **0.975** | 86M | With TTA |
| DANN-v3 (no TTA) | 89.09% | 0.879 | 0.972 | 86M | Standard inference |
| DANN-v2 | 86.1% | 0.871 | 0.962 | 86M | 3 domains |
| DANN-v1 | 86.1% | 0.867 | 0.962 | 86M | 3 domains |
| ViT baseline (v3.py) | 85.28% | 0.843 | 0.944 | 86M | No DANN |
| 3-model ensemble | 84.8% | 0.840 | — | 3x86M | Weaker models |

### 21.2 Ablation Study Results

| Variant | Accuracy | Macro F1 | AUC |
|---|---|---|---|
| Base ViT (no DANN) | 85.28% | 0.843 | 0.944 |
| DANN only | 84.73% | 0.843 | 0.937 |
| DANN + hard mining | 85.89% | 0.849 | 0.947 |
| DANN + mixup | 84.66% | 0.821 | 0.931 |
| DANN-v3 (full pipeline) | 89.09% | 0.879 | 0.972 |

The full v3 pipeline shows 3.81% absolute improvement over Base ViT and 4.36% over DANN-only.

### 21.3 Per-Class Detailed Performance

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Normal | 0.867 | 0.842 | 0.854 | 267 |
| Diabetes/DR | 0.915 | 0.925 | 0.920 | 547 |
| Glaucoma | 0.821 | 0.845 | 0.833 | 161 |
| Cataract | 0.912 | 0.887 | 0.899 | 246 |
| AMD | 0.897 | 0.893 | 0.895 | 246 |

DR achieves the highest F1 (0.920) due to the progressive alpha boost and its representation in training data. Glaucoma is hardest (0.833) — it has the smallest training set and highest visual ambiguity with Normal.

### 21.4 Calibration Results

| Stage | ECE |
|---|---|
| Pre-temperature scaling | 0.093 |
| Post-temperature scaling (T=0.566) | 0.034 |

The T<1 value (0.566) indicates the model was overconfident before calibration. Scaling sharpens the probability distribution, concentrating mass on the predicted class.

### 21.5 RAD Pipeline Performance

| Component | Metric | Value |
|---|---|---|
| FAISS Index | MAP | 0.921 |
| FAISS Index | Recall@1 | 94.0% |
| RAD Combined | Accuracy (K=1) | 94.0% |
| RAD Combined | Improvement over standalone | +4.9% |
| Routing | AUTO-REPORT fraction | 76.9% |
| Routing | AUTO-REPORT accuracy | 96.8% |
| Routing | Error catch rate | 77.2% |

---

## 22. Execution Guide

### 22.1 Prerequisites

```bash
# Install dependencies
pip install -r requirements_deploy.txt
# Key packages: torch>=2.0, timm, faiss-gpu, gradio, fastapi, cv2, sklearn

# Download model from HuggingFace
python setup.sh
# or manually:
# huggingface-cli download tanishq74/retinasense-vit --local-dir outputs_v3/
```

### 22.2 Prepare Datasets

Datasets must be placed in the following directories:
- `aptos/train.csv` + `aptos/train_images/*.png`
- `odir/full_df.csv` + `odir/preprocessed_images/*.jpg`
- `refuge2/Training400/Glaucoma/*.jpg`
- MESSIDOR-2: metadata CSV pointing to image files

### 22.3 Preprocessing

```bash
python unified_preprocessing.py
# Creates preprocessed_cache_unified/ (~7GB)
# Generates data/train_split_expanded.csv, data/calib_split_expanded.csv
```

### 22.4 Training (DANN-v3)

```bash
# With warm-start from v2 checkpoint (recommended):
python train_dann_v3.py --epochs 40 --lr 3e-5 --tta

# Without warm-start:
python train_dann_v3.py --no-warmstart --epochs 50 --lr 1e-4

# With custom output directory:
python train_dann_v3.py --output-dir outputs_v3/custom_run
```

### 22.5 Rebuild FAISS Index

```bash
python rebuild_faiss_full.py
# Extracts embeddings from all 8,241 training samples
# Saves outputs_v3/retrieval/index_flat_ip.faiss
# Saves outputs_v3/retrieval/metadata.json
```

### 22.6 RAD Evaluation

```bash
python rad_evaluation.py --k-values 1 3 5 10 --alpha 0.5
# Saves outputs_v3/retrieval/rad_evaluation_results.json
# Saves retrieval plots
```

### 22.7 Confidence Routing

```bash
python confidence_routing.py --mc-passes 15 --retrieval-k 5
# Saves outputs_v3/retrieval/confidence_routing_results.json
# Saves routing distribution plots
```

### 22.8 Launch Demo

```bash
python app.py
# Gradio interface at http://localhost:7860
```

### 22.9 Training DANN-v4 (RETFound)

```bash
# Verify weights are present:
ls weights/RETFound_cfp_weights.pth  # should be ~1.2 GB

# Train (~15-20 min on H100):
python train_dann_v4.py --tta
# Output: outputs_v3/dann_v4/best_model.pth
```

### 22.10 CLI Arguments — train_dann_v3.py

| Argument | Default | Description |
|---|---|---|
| --warmstart | outputs_v3/dann_v2/best_model.pth | Checkpoint for warm-start |
| --no-warmstart | False | Skip warm-start (random init) |
| --epochs | 40 | Training epochs |
| --lr | 3e-5 | Base learning rate |
| --domain-weight | 0.05 | Domain loss weight |
| --batch-size | 32 | Per-GPU batch size |
| --workers | 8 | DataLoader workers |
| --dr-alpha-start | 1.5 | DR focal alpha at epoch 0 |
| --dr-alpha-end | 3.0 | DR focal alpha at epoch N |
| --hard-mining-k | 500 | Hard example mining count |
| --hard-mining-factor | 2 | Hard example oversampling factor |
| --mixup-alpha | 0.2 | MixUp Beta distribution parameter |
| --mixup-prob | 0.5 | Per-batch MixUp probability |
| --label-smoothing | 0.1 | CrossEntropy label smoothing |
| --cosine-t0 | 10 | CosineWarmRestarts T0 |
| --cosine-tmult | 2 | CosineWarmRestarts T_mult |
| --tta | False | Enable 8-way TTA for final eval |
| --tta-n | 8 | Number of TTA augmentations |
| --max-lambda | 0.3 | Ganin lambda cap |
| --seed | 42 | Random seed |
| --output-dir | outputs_v3/dann_v3 | Override output directory |

---

*This functional specification documents the complete RetinaSense-ViT system as implemented through Session 6 (2026-03-27). The production model is DANN-v3 achieving 89.30% accuracy. DANN-v4 (RETFound backbone) is implemented and ready for GPU training.*
