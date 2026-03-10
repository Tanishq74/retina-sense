# RetinaSense-ViT: Functional Document — Module Description

**Version:** 1.0  
**Date:** March 10, 2026  
**Author:** Tanishq  
**Status:** Production Ready  

---

## 1. System Overview

RetinaSense-ViT is a multi-class retinal disease classification system that analyzes fundus images to detect five conditions: **Normal, Diabetic Retinopathy, Glaucoma, Cataract, and AMD**. The system is organized into the following functional modules:

---

## 2. Module Map

```
┌───────────────────────────────────────────────────────────────┐
│                     RetinaSense-ViT System                    │
│                                                               │
│  M1: Data         M2: Preprocessing   M3: Model              │
│  Ingestion        Pipeline             Architecture           │
│                                                               │
│  M4: Training     M5: Threshold       M6: Inference           │
│  Engine           Optimization         Pipeline               │
│                                                               │
│  M7: Ensemble     M8: Evaluation      M9: Data               │
│  System           & Visualization      Analysis               │
└───────────────────────────────────────────────────────────────┘
```

---

## 3. Module Descriptions

### M1: Data Ingestion Module

**Purpose:** Load, validate, and unify data from ODIR-5K and APTOS-2019 datasets.

| Attribute | Detail |
|-----------|--------|
| **Input** | ODIR images (512×512), APTOS images (~1949×1500), metadata CSVs |
| **Output** | `combined_dataset.csv` with 8,540 entries (path, disease_label, severity_label) |
| **Key Files** | `final_unified_metadata.csv`, `data/combined_dataset.csv` |
| **Functions** | Path cleaning (remove `./` prefixes), single-disease filtering, stratified train/val split (80/20) |

**Key Logic:**
- APTOS images are exclusively DR class with 5-level severity grading
- ODIR images span all 5 disease classes; multi-disease samples are filtered out
- Paths are normalized for cross-platform compatibility

---

### M2: Preprocessing Pipeline Module

**Purpose:** Apply Ben Graham contrast enhancement and caching to prepare images for model input.

| Attribute | Detail |
|-----------|--------|
| **Input** | Raw fundus image (any resolution) |
| **Output** | Normalized tensor (3×224×224) |
| **Key Files** | All training scripts (ben_graham_preprocess function) |
| **Dependencies** | OpenCV, NumPy, torchvision transforms |

**Functional Steps:**
1. **Resize** to target resolution (224×224 for ViT, 300×300 for EfficientNet)
2. **Ben Graham Enhancement:** `4×img − 4×GaussianBlur(σ=10) + 128`
3. **Circular Mask** application (radius = 0.48 × image_size)
4. **Caching:** Pre-compute and store as `.npy` files (one-time; ~60s for 8,540 images)
5. **Augmentation** (training only): flip, rotate, affine, color jitter, random erasing
6. **ImageNet Normalization:** μ=[0.485,0.456,0.406], σ=[0.229,0.224,0.225]

---

### M3: Model Architecture Module

**Purpose:** Define the neural network architectures for disease classification.

#### M3.1: ViT-Base-Patch16-224 (Production Model)

| Attribute | Detail |
|-----------|--------|
| **Backbone** | ViT-Base-Patch16-224 (timm, pre-trained on ImageNet) |
| **Parameters** | ~86M |
| **Feature Dim** | 768 |
| **Disease Head** | 768→512→256→5 (BatchNorm, ReLU, Dropout) |
| **Severity Head** | 768→256→5 (BatchNorm, ReLU, Dropout) |
| **Model Size** | 331 MB |

**Why ViT Excels:**
- Global self-attention captures vessel patterns across the entire fundus
- Position encoding preserves spatial relationships (optic disc, macula location)
- Handles APTOS/ODIR domain shift better than CNNs (less texture-dependent)
- Superior on minority classes: Glaucoma +144%, AMD +199% over CNN baseline

#### M3.2: EfficientNet-B3 (Backup Model)

| Attribute | Detail |
|-----------|--------|
| **Backbone** | EfficientNet-B3 (timm, pre-trained on ImageNet) |
| **Parameters** | ~12M |
| **Feature Dim** | 1,536 |
| **Model Size** | 47 MB |

---

### M4: Training Engine Module

**Purpose:** Train the model with class-imbalance-aware strategies and GPU optimization.

| Attribute | Detail |
|-----------|--------|
| **Key Files** | `retinasense_vit.py`, `retinasense_v2_extended.py`, `retinasense_v2.py` |
| **Loss Function** | Focal Loss (γ=1.0, α=class_weights) + 0.2×CE (severity) |
| **Optimizer** | AdamW (lr=3×10⁻⁴) |
| **Scheduler** | Cosine Annealing (T_max=30, η_min=1×10⁻⁷) |
| **Mixed Precision** | AMP with GradScaler |
| **Gradient Accumulation** | 2 steps (effective batch=64) |
| **Early Stopping** | Patience=10 on macro F1 |

**GPU Optimization Features:**
- Pre-cached preprocessing (100× faster data loading)
- Batch size scaling (32→128 for raw speed, 64 recommended for stability)
- 8 DataLoader workers with persistent_workers and prefetch_factor=2
- Non-blocking GPU transfers

**Training Duration:**
- ViT: ~6 minutes (30 epochs on H200)
- EfficientNet-B3 Extended: ~15 minutes (50 epochs)

---

### M5: Threshold Optimization Module

**Purpose:** Post-training optimization of per-class decision thresholds to maximize F1 score.

| Attribute | Detail |
|-----------|--------|
| **Key Files** | `threshold_optimization_vit.py`, `threshold_optimization_simple.py` |
| **Method** | Grid search (0.05–0.95, step 0.05) per class, one-vs-rest binary F1 |
| **Input** | Model softmax probabilities on validation set |
| **Output** | JSON file with optimal thresholds per class |

**Optimal Thresholds (ViT, Accuracy-focused):**

| Class | Threshold | Clinical Rationale |
|-------|-----------|-------------------|
| Normal | 0.540 | Balanced |
| Diabetes/DR | 0.240 | Lenient → high sensitivity (catch all DR) |
| Glaucoma | 0.810 | Strict → high specificity (require confidence) |
| Cataract | 0.930 | Very strict → minimize false positives |
| AMD | 0.850 | Strict → rare disease, need confidence |

**Impact:** +2.22% accuracy for ViT (82.26→84.48%); +9.84% for v2 baseline (63.52→73.36%).

---

### M6: Inference Pipeline Module

**Purpose:** Classify new fundus images using the trained model and optimized thresholds.

| Attribute | Detail |
|-----------|--------|
| **Key Files** | `RetinaSense_Production.ipynb` |
| **Latency** | ~15ms per image |
| **Throughput** | ~66 images/sec |
| **GPU Memory** | ~2 GB |

**Inference Flow:**
1. Load and preprocess image (Ben Graham)
2. Forward pass through ViT → disease logits + severity logits
3. Apply softmax → class probabilities
4. Apply per-class thresholds → final prediction
5. If confidence < threshold for all classes → flag for expert review
6. Return: class label, confidence score, all probabilities

---

### M7: Ensemble System Module

**Purpose:** Combine predictions from multiple models for improved minority class detection.

| Attribute | Detail |
|-----------|--------|
| **Key Files** | `ensemble_inference.py` |
| **Models** | ViT (85%), EfficientNet-Extended (10%), EfficientNet-v2 (5%) |
| **Strategy** | Weighted probability averaging |

**Performance Trade-off:**
- Ensemble: 80.44% accuracy, 0.858 macro F1, Cataract F1=0.952, AMD F1=0.920
- ViT Solo: 84.48% accuracy, 0.840 macro F1 (simpler, faster, recommended)

---

### M8: Evaluation & Visualization Module

**Purpose:** Comprehensive model evaluation with per-class metrics and visual dashboards.

| Attribute | Detail |
|-----------|--------|
| **Key Files** | Training scripts (eval sections), `RetinaSense_Production.ipynb` |
| **Primary Metrics** | Macro F1, accuracy, per-class F1/precision/recall |
| **Secondary Metrics** | Weighted F1, Macro AUC-ROC, confusion matrix |
| **Outputs** | `dashboard.png`, `threshold_comparison.png`, `training_curves.png` |

**Why Macro F1 (not accuracy):** Accuracy is misleading with 21:1 class imbalance (65% accuracy by always predicting DR). Macro F1 treats all classes equally.

---

### M9: Data Analysis Module

**Purpose:** Comprehensive dataset exploration to inform training strategy.

| Attribute | Detail |
|-----------|--------|
| **Key Files** | `data_analysis.py` |
| **Outputs** | `outputs_analysis/` (11 files: plots, reports, CSVs) |

**Analyses Performed:**
1. **Class distribution** — confirmed 21.1× imbalance
2. **Image quality metrics** — brightness, contrast, sharpness per class
3. **APTOS domain shift discovery** — 10.7× sharpness difference vs ODIR
4. **Error analysis** — most-confused class pairs (DR↔Normal, Normal↔AMD)
5. **Augmentation effectiveness** — light augmentation best during warmup
6. **Preprocessing impact** — Ben Graham boosts Glaucoma brightness most (+34.2)

---

## 4. Module Interaction Matrix

| From \ To | M1 | M2 | M3 | M4 | M5 | M6 | M7 | M8 | M9 |
|-----------|----|----|----|----|----|----|----|----|-----|
| **M1** Data | — | ✓ | | | | | | | ✓ |
| **M2** Preprocess | | — | | ✓ | | ✓ | | | |
| **M3** Model | | | — | ✓ | | ✓ | ✓ | | |
| **M4** Training | | | ✓ | — | ✓ | | | ✓ | |
| **M5** Threshold | | | | | — | ✓ | ✓ | ✓ | |
| **M6** Inference | | ✓ | ✓ | | ✓ | — | | | |
| **M7** Ensemble | | | ✓ | | ✓ | | — | ✓ | |
| **M8** Evaluation | | | | | | | | — | |
| **M9** Analysis | ✓ | | | | | | | | — |

---

## 5. Test-Time Augmentation (TTA) Sub-Module

**Purpose:** Improve predictions by averaging over augmented versions of the input.

**8 Augmentations:** Original, H-flip, V-flip, Both flips, Rot 90°, Rot 180°, Rot 270°, Brightness  
**Impact:** +0.29% accuracy (modest; optional for production)  
**Trade-off:** 8× slower inference  
**Recommendation:** Use selectively for uncertain cases (confidence < threshold)

---

## 6. Configuration Parameters

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `IMG_SIZE` | 224 | 224–300 | 224 for ViT, 300 for EfficientNet |
| `BATCH_SIZE` | 32 | 16–128 | 64 recommended for stability |
| `NUM_WORKERS` | 8 | 0–16 | Match to CPU cores |
| `USE_CACHE` | True | True/False | 4× speedup when True |
| `EPOCHS` | 30 | 10–100 | ViT converges by 30 |
| `ACCUM_STEPS` | 2 | 1–8 | Gradient accumulation factor |
| `PATIENCE` | 10 | 5–15 | Early stopping on macro F1 |
| `FOCAL_GAMMA` | 1.0 | 0.5–3.0 | Focusing parameter for class imbalance |

---

*Document Version: 1.0 | Last Updated: March 10, 2026*
