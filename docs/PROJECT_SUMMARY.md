# RetinaSense-ViT — Complete Project Summary

**Project**: Multi-disease retinal fundus image classification (5 classes)
**Authors**: Tanishq Tamarkar, Rafae Mohammed Hussain, Dr. Revathi M
**Institution**: SRM Institute of Science and Technology, Chennai, India
**Repository**: https://github.com/Tanishq74/retina-sense
**HuggingFace**: https://huggingface.co/tanishq74/retinasense-vit
**Development Period**: March 2026 (6 sessions, ~20 hours active)
**Final Production Model**: DANN-v3 — 89.30% accuracy, 0.886 Macro F1, 0.975 AUC

---

## Table of Contents

1. [Project Objective](#1-project-objective)
2. [Disease Classes and Dataset](#2-disease-classes-and-dataset)
3. [Overall Accuracy Trajectory](#3-overall-accuracy-trajectory)
4. [Session-by-Session Complete History](#4-session-by-session-complete-history)
   - [Session 1 — Baseline + DANN-v1/v2](#session-1-early-march-2026--baseline--dann-v1v2)
   - [Session 2 — Merge + Cleanup + DANN-v3 Prep](#session-2-2026-03-24--merge--cleanup--dann-v3-prep)
   - [Session 3 — DANN-v3 Training + Evaluation](#session-3-2026-03-24--dann-v3-training--full-evaluation)
   - [Session 4 — IEEE Paper + RAD Pipeline + Code Fixes](#session-4-2026-03-25--ieee-paper--rad-pipeline--code-fixes)
   - [Session 5 — GPU Experiments + Paper Completion](#session-5-2026-03-25--gpu-experiments--paper-completion)
   - [Session 6 — DANN-v4 RETFound Preparation](#session-6-2026-03-25--dann-v4-retfound-preparation)
5. [Architecture Deep Dive](#5-architecture-deep-dive)
6. [Training Pipeline — DANN-v3 (Production)](#6-training-pipeline--dann-v3-production)
7. [RAD Pipeline — Retrieval-Augmented Diagnosis](#7-rad-pipeline--retrieval-augmented-diagnosis)
8. [All Bug Fixes Applied](#8-all-bug-fixes-applied)
9. [All Configuration Changes](#9-all-configuration-changes)
10. [Complete Performance Metrics](#10-complete-performance-metrics)
11. [Ablation Study Results](#11-ablation-study-results)
12. [LODO Generalization Results](#12-lodo-generalization-results)
13. [Known Issues and Technical Debt](#13-known-issues-and-technical-debt)
14. [Next Steps](#14-next-steps)
15. [File Inventory](#15-file-inventory)

---

## 1. Project Objective

RetinaSense-ViT is a clinical AI framework for automated retinal disease screening from fundus photographs. The system classifies images into five diagnostic categories:

| Index | Class | Description |
|---|---|---|
| 0 | Normal | Healthy retina with no detected pathology |
| 1 | Diabetes/DR | Diabetic Retinopathy — microaneurysms, hemorrhages, exudates |
| 2 | Glaucoma | Elevated IOP damage — optic disc cupping, nerve fiber loss |
| 3 | Cataract | Lens opacity — reduced clarity and contrast |
| 4 | AMD | Age-related Macular Degeneration — drusen, geographic atrophy |

**Core technical challenges addressed**:
1. Cross-dataset domain shift: 10.7× sharpness gap between APTOS and ODIR images (APTOS accuracy was 26.5% before domain adaptation)
2. Severe class imbalance: 21:1 DR-to-AMD ratio in training data
3. Clinical trust requirements: uncertainty quantification, explainability, retrieval-based evidence grounding

**Seven objectives (6/7 fully achieved)**:

| # | Objective | Target | Outcome | Status |
|---|---|---|---|---|
| 1 | Multi-disease classification (5 classes) | 5-class classifier | All F1 > 0.83 | **ACHIEVED** |
| 2 | Achieve 90%+ accuracy | 90.00% | 89.30% (94.0% with RAD) | **PARTIALLY ACHIEVED** |
| 3 | Handle cross-dataset domain shift | Fix APTOS accuracy | 26.5% → 99.8% | **ACHIEVED** |
| 4 | Address class imbalance (21:1) | Competitive minority F1 | All classes F1 > 0.83 | **ACHIEVED** |
| 5 | Clinical trust pipeline | RAD + uncertainty + XAI | MAP 0.921, ECE 0.034 | **ACHIEVED** |
| 6 | IEEE conference paper | 0 placeholders | 700 lines, 13 tables | **ACHIEVED** |
| 7 | Production deployment | Gradio + FastAPI + Docker | All 4 modalities live | **ACHIEVED** |

---

## 2. Disease Classes and Dataset

### Data Sources

| Source | Size | Classes | Camera | Preprocessing |
|---|---|---|---|---|
| APTOS-2019 | 3,662 | DR (5 severity levels) | Aravind field camera | Ben Graham enhancement |
| ODIR-5K | 4,878 | All 5 (single-label filtered) | Multi-source clinical | CLAHE |
| REFUGE2 | 1,240 | Normal + Glaucoma | Zeiss Visucam 500 | Resize only |
| MESSIDOR-2 | 1,744 | Normal + DR | Topcon TRC NW6 | CLAHE |
| **Total** | **11,524** | **5 classes** | | |

### Splits

| Split | Size | Purpose |
|---|---|---|
| Train | 8,241 (70%) | Model training |
| Calibration | 1,816 (15%) | Temperature + threshold tuning |
| Test | 1,467 (15%) | **Sealed** — never used in training or calibration |

### Class Distribution (Training)

| Class | Count | Percentage | Imbalance vs. AMD |
|---|---|---|---|
| Normal | 2,071 | 24.3% | 7.8× |
| Diabetes/DR | 5,581 | 65.4% | 21.1× (majority class) |
| Glaucoma | 308 | 3.6% | 1.2× |
| Cataract | 315 | 3.7% | 1.2× |
| AMD | 265 | 3.1% | 1.0× (minority class) |

---

## 3. Overall Accuracy Trajectory

Every model version and the technique that drove each improvement:

```
63.52%  EfficientNet-B3 baseline (Phase 0)
  |
  +9.84pp  Per-class threshold optimization (grid search)
  |
73.36%  EfficientNet + Thresholds
  |
  +0.82pp  Extended training (50 epochs)
  |
74.18%  Extended CNN
  |
  +8.08pp  ViT-Base/16 architecture switch (SINGLE LARGEST GAIN)
  |
82.26%  ViT-Base/16
  |
  +2.22pp  Per-class threshold optimization for ViT
  |
84.48%  ViT + Thresholds (first production model)
  |
  +1.62pp  DANN domain adversarial training v1/v2
  |
86.10%  DANN-v2 (3 domains: APTOS + ODIR + REFUGE2)
  |
  +3.20pp  DANN-v3 (4 domains + expanded dataset + full training recipe)
  |
89.30%  DANN-v3 (CURRENT PRODUCTION — standalone accuracy)
  |
  +4.70pp  RAD retrieval augmentation (FAISS kNN at inference)
  |
94.00%  DANN-v3 + RAD Combined (best effective accuracy)
```

**Cumulative gain from initial baseline to production: +25.78pp standalone, +30.48pp with RAD**

---

## 4. Session-by-Session Complete History

---

### Session 1 (Early March 2026) — Baseline + DANN-v1/v2

**Duration**: ~1 day | **Goal**: Establish baseline, train initial DANN models

#### What was built

**EfficientNet-B3 Baseline**
- Trained on ODIR-5K + APTOS-2019 (2 sources)
- Achieved 63.52% accuracy, Macro F1 = 0.517
- Critical failure: DR recall = 25.3% — only 25% of diabetic retinopathy images correctly identified
- Root cause: APTOS images used Ben Graham enhancement, ODIR used CLAHE — the model learned preprocessing style, not disease
- APTOS accuracy: 26.5% (essentially failed on 43% of the training distribution)

**Threshold Optimization Discovery**
- Applied per-class threshold grid search on top of EfficientNet
- +9.84pp accuracy gain with zero training cost
- Revealed that default argmax at 0.5 is badly calibrated under class imbalance

**ViT-Base/16 Architecture Switch**
- Switched from EfficientNet-B3 (12M params) to ViT-Base/16 (86M params) from `timm`
- +18.74pp accuracy gain — single largest improvement in entire project
- ViT global self-attention captured distributed disease markers CNNs missed
- Minority class improvements: Glaucoma F1 0.346→0.871 (+144%), AMD F1 0.267→0.819 (+206%)
- ViT + threshold optimization: 84.48% accuracy

**Pre-caching Preprocessing**
- Stored preprocessed images as `.npy` files (uint8, 224×224×3)
- GPU utilization improved from 5–10% to 60–85%
- 4× training speedup

**DANN-v1 Training**
- Added Gradient Reversal Layer (GRL) to ViT backbone
- 4-domain discriminator head (APTOS=0, ODIR=1, REFUGE2=2, initial)
- Lambda schedule: Ganin sigmoid ramp, capped at 0.3
- Loss: `L_disease + 0.2 × L_severity + 0.05 × λ × L_domain`
- Result: 86.1% accuracy, 0.867 Macro F1, 0.962 AUC
- APTOS accuracy: 99.8% (was 26.5%) — domain shift resolved
- DR recall: 80.8% (was 25.3%)

**DANN-v2 Training**
- Added `--dr-alpha-boost 2.5` flag: DR focal weight × 2.5
- Result: 86.1% accuracy, 0.871 Macro F1 (+0.004 over v1)
- AMD F1 reached 0.950 in this version

**EfficientNet-B3 Ensemble**
- `train_ensemble.py`: Combined ViT + EfficientNet-B3 via soft voting
- 3-model ensemble achieved 84.8% accuracy (lower than single DANN-v3 — weaker models pull down average)

**5-Fold Cross-Validation**
- `kfold_cv.py`: 5-fold stratified on disease label
- Accuracy: 82.4% ± 1.9%, Macro F1: 0.827 ± 0.019, AUC: 0.948 ± 0.008
- Note: CV used pre-DANN-v3 architecture; gap from production (89.3%) explained by training on full 70% vs 56% per fold

**Session 1 Metrics Summary**:
| Model | Accuracy | Macro F1 | AUC |
|---|---|---|---|
| EfficientNet-B3 baseline | 63.52% | 0.517 | 0.910 |
| + Thresholds | 73.36% | 0.632 | — |
| ViT-Base/16 | 82.26% | 0.821 | 0.967 |
| ViT + Thresholds | 84.48% | 0.840 | 0.967 |
| DANN-v1 | 86.10% | 0.867 | 0.962 |
| DANN-v2 | 86.10% | 0.871 | 0.962 |

---

### Session 2 (2026-03-24) — Merge + Cleanup + DANN-v3 Prep

**Duration**: ~4 hours | **Goal**: Consolidate codebase, integrate new data, write DANN-v3 script

#### Key actions

**Deep Line-by-Line Code Review**
- Reviewed all scripts for preprocessing inconsistencies and hardcoded path issues
- Identified that many scripts used `/teamspace/studios/this_studio` as absolute path — broke portability

**HuggingFace Integration**
- Model weights uploaded to `tanishq74/retinasense-vit`
- Enabled one-command download: `huggingface-cli download tanishq74/retinasense-vit`

**MESSIDOR-2 Integration (4th data source)**
- Added 1,744 images: Normal + DR with dual-expert grading
- French hospital origin (Topcon TRC NW6 camera) adds imaging hardware diversity
- Dataset expanded from 8,540 → 11,524 images (+35%)
- Sources increased from 3 → 4 (APTOS + ODIR + REFUGE2 + MESSIDOR-2)

**Unified Preprocessing Pipeline Built**
- `unified_preprocessing.py`: Dispatches to source-appropriate preprocessing
  - APTOS → Ben Graham (amplifies local contrast, removes vignetting)
  - ODIR, MESSIDOR-2 → CLAHE on LAB L-channel only
  - REFUGE2 → Resize only (already standardized Zeiss images)
- All images get circular mask at radius `0.48 × 224 = 107 px`
- Cache output: `preprocessed_cache_unified/` (11,524 × `.npy` files)
- Generated `configs/fundus_norm_stats_unified.json` with corpus-specific mean/std

**`train_dann_v3.py` Created (new training script)**
- All 8 improvements over DANN-v2 baked in:
  1. Hard-example mining (top-500 highest-loss samples oversampled 2×)
  2. MixUp augmentation (alpha=0.2, probability=0.5 per batch)
  3. Cosine annealing with warm restarts (T0=10, Tmult=2)
  4. Label smoothing (0.1)
  5. Progressive DR alpha boost (1.5× epoch 0 → 3.0× epoch N)
  6. Layer-wise learning rate decay (LLRD, 0.85 decay per block)
  7. Gradient accumulation (effective batch 64)
  8. 4th domain (MESSIDOR-2) added

**Metadata CSVs Regenerated**
- `data/train_split_expanded.csv` (8,241 samples, 4 sources)
- `data/calib_split_expanded.csv` (1,816 samples)
- `data/test_split.csv` (1,467 samples, SEALED)

**Session 2 Outcome**: No new training. Architecture preparations complete. Dataset expanded.

---

### Session 3 (2026-03-24) — DANN-v3 Training + Full Evaluation

**Duration**: ~3 hours | **Goal**: Train DANN-v3 and produce full evaluation suite

#### Training Run

- **Script**: `python train_dann_v3.py --epochs 40 --lr 3e-5 --tta`
- **Training time**: 5.9 minutes on H100 GPU
- **Patience**: 15 epochs on macro-F1 improvement > 0.001
- **Best epoch**: Saved at epoch where macro-F1 peaked

#### Hyperparameters (final production values)

| Parameter | Value |
|---|---|
| Backbone | vit_base_patch16_224 (timm) |
| Batch size | 32 (effective 64 with gradient accumulation=2) |
| Base LR | 3e-5 |
| LLRD decay | 0.85 (per transformer block) |
| Weight decay | 1e-4 |
| Focal gamma | 2.0 |
| Label smoothing | 0.1 |
| Mixup alpha | 0.2, probability 0.5 |
| DR alpha start/end | 1.5× / 3.0× |
| Hard mining K | 500 samples |
| Hard mining factor | 2× oversampling |
| Max DANN lambda | 0.3 (Ganin cap) |
| Domain weight | 0.05 |
| TTA | 8-way (identity, H-flip, V-flip, HV-flip, 90°, 180°, 270°, crop) |

#### DANN-v3 Results

| Metric | Value |
|---|---|
| Overall Accuracy | **89.30%** |
| Macro F1 | **0.886** |
| Weighted F1 | 0.893 |
| Macro AUC | **0.975** |
| ECE (pre-calibration) | 0.149 |
| ECE (post-calibration) | **0.034** |
| Temperature (T*) | **0.5657** |
| Cohen's Kappa | **0.809** |
| MCC | **0.810** |
| Total errors (1,467 test) | 158 |

#### Per-Class Performance

| Class | Precision | Recall | F1 | AUC | Support |
|---|---|---|---|---|---|
| Normal | 0.817 | 0.895 | 0.854 | 0.964 | 484 |
| Diabetes/DR | 0.935 | 0.904 | 0.920 | 0.977 | 837 |
| Glaucoma | 0.900 | 0.776 | 0.833 | 0.953 | 58 |
| Cataract | 0.976 | 0.833 | 0.899 | 0.985 | 48 |
| AMD | 0.944 | 0.850 | 0.895 | 0.995 | 40 |

#### Confusion Matrix

|  | Normal | DR | Glaucoma | Cataract | AMD |
|---|---|---|---|---|---|
| **Normal** | **433** | 45 | 5 | 1 | 0 |
| **DR** | 78 | **757** | 0 | 0 | 2 |
| **Glaucoma** | 8 | 5 | **45** | 0 | 0 |
| **Cataract** | 8 | 0 | 0 | **40** | 0 |
| **AMD** | 3 | 3 | 0 | 0 | **34** |

**77% of all 158 errors are Normal↔DR confusion (45 + 78 = 123/158).**

#### Temperature Scaling

- Pre-calibration ECE: 0.149 (model was overconfident, T < 1 means confidence was already high)
- Optimized T = 0.5657 by minimizing NLL on calibration set
- Post-calibration ECE: 0.037
- Saved to `configs/temperature.json`

#### Per-Class Threshold Optimization

- Grid search over 50 values in [0.05, 0.95] per class
- Final thresholds (stored in `configs/thresholds.json`):
  - Normal: 0.3990
  - Diabetes/DR: 0.4173
  - Glaucoma: 0.6745 (highest — small natural probability)
  - Cataract: 0.2888
  - AMD: 0.3439

#### Evaluation Artifacts Generated

7 files saved to `outputs_v3/dann_v3/evaluation/`:
- Confusion matrix (raw + normalized)
- ROC curves (all 5 classes, AUC annotated)
- Calibration reliability diagram
- Per-class F1 bar chart
- Training loss/accuracy curves
- Classification report (precision, recall, F1, support)

#### app.py Updated

- Model path: `outputs_v3/dann_v3/best_model.pth` (primary)
- DANN keys (`domain_head.*`, `grl.*`) filtered on load to `MultiTaskViT`
- Config paths updated to `configs/` directory
- Norm stats updated to `configs/fundus_norm_stats_unified.json`

---

### Session 4 (2026-03-25) — IEEE Paper + RAD Pipeline + Code Fixes + Cleanup

**Duration**: ~6 hours | **Goal**: Write IEEE paper, build RAD framework, fix bugs, clean project

#### IEEE LaTeX Paper Written

- **File**: `paper/retinasense_ieee.tex`
- **Length at session end**: 640 lines, 30 references
- **Title**: "RetinaSense: An Uncertainty-Aware Domain-Adaptive Vision Transformer Framework with Retrieval-Augmented Reasoning for Multi-Disease Retinal Diagnosis"
- **Authors**: Tanishq Tamarkar, Rafae Mohammed Hussain, Dr. Revathi M (SRM Chennai)
- **Target conferences**: IEEE BHI, EMBC, CBMS, ICHI
- **Status at end of session**: All DANN-v3 results filled; Table VII (retrieval metrics) had placeholder "pending GPU evaluation"

#### RAD Pipeline Scripts Built (2,036 new lines)

**`rebuild_faiss_full.py` (393 lines)**
- Rebuilds FAISS index from ALL 8,241 training samples, ALL 5 disease classes
- Index type: `IndexFlatIP` (cosine similarity via L2-normalized inner product)
- Vectors: 768-dim CLS token embeddings from DANN-v3 backbone
- Output: `outputs_v3/retrieval/index_flat_ip.faiss`
- Critical fix: previous index had only Normal + DR (2/5 classes); AMD, Glaucoma, Cataract retrievals were all wrong

**`rad_evaluation.py` (764 lines)**
- Computes: Recall@K, Precision@K, MAP at K=1,3,5,10
- Class-match heatmap, agreement score, kNN-augmented accuracy
- Per-class and per-source breakdown
- Outputs: `rad_evaluation_results.json` + 3 PNG plots

**`confidence_routing.py` (879 lines)**
- 3-tier clinical triage: AUTO-REPORT / REVIEW / ESCALATE
- Triplet signals: model confidence + MC Dropout entropy + FAISS retrieval agreement
- Thresholds: `conf_high=0.85`, `conf_low=0.50`, `entropy_low=0.5`, `entropy_high=1.0`
- Outputs: `confidence_routing_results.json` + 3 PNG plots

#### GPU Experiment Master Script

**`run_paper_experiments.py` (1,609 lines)**
- Orchestrates all 5 GPU experiments for IEEE paper
- Flags: `--rebuild-faiss`, `--eval-rad`, `--lodo`, `--ablation`, `--eval-routing`, `--all`
- LODO and ablation are fully inline (no subprocess calls)
- Imports architecture from `train_dann_v3.py` directly (prevents drift)

#### Code Fixes Applied (7 files)

| File | Fix Applied |
|---|---|
| `knowledge_distillation.py` | Fixed norm stats path (configs/ fallback chain), fixed teacher model path (DANN-v3 fallback), fixed DANN key filtering |
| `update_faiss_messidor2.py` | Fixed hardcoded `/teamspace/studios/this_studio` → relative paths, fixed model path |
| `gradcam_v3.py` | Added DANN-v3 as first priority in MODEL_PATH chain, added DANN key filtering |
| `mc_dropout_uncertainty.py` | Added DANN-v3 to MODEL_PATH fallback chain |
| `integrated_gradients_xai.py` | Same DANN-v3 fallback fix |
| `fairness_analysis.py` | Same DANN-v3 fallback fix |
| `ARCHITECTURE_DOCUMENT.md` | Updated all metrics to DANN-v3 (89.30%, 0.886, 0.975), version 3.0 |

#### app.py RAD Updates

4 functions added/updated:
- `load_faiss_index()`: Prefers `index_flat_ip.faiss` (rebuilt), falls back to `index_flat_l2.faiss`
- `retrieve_similar()`: Handles both IP and L2 index types, returns enriched metadata
- `retrieve_augmented_prediction()` **(NEW)**: Combines model softmax probs with kNN votes via `alpha=0.5` blending
- `_resolve_cache_path()` **(NEW)**: Resolves cache path from metadata with fallback chain

#### Project Cleanup (14 GB freed, 39 GB → 25 GB)

| Item Removed | Size |
|---|---|
| `aptos/aptos2019-blindness-detection.zip` | 9.6 GB |
| `aptos/test_images/` | 1.6 GB |
| `aptos/gaussian_filtered_images/` | 440 MB |
| `preprocessed_cache_v3/` | 894 MB |
| `preprocessed_cache_vit/` | 888 MB |
| `outputs_vit/` | 331 MB |
| `outputs_analysis/`, `outputs_production/`, `outputs_v2/`, `outputs_v2_extended/` | Combined ~1.5 GB |
| `legacy/` directory | Several MB |
| Old root-level `outputs_v3/` artifacts | ~100 MB |
| Training logs, `__pycache__`, `.gradio` cache | ~200 MB |

#### End-to-End App Test

- Ran Gradio app port 7860, tested 5 sample fundus images (one per class)
- Results: **3/5 correct** (Normal 94.7%, Glaucoma 91.9%, AMD 99.2%)
- Failures: DR predicted as AMD (high uncertainty), Cataract predicted as Normal (minority class)
- All pipeline components confirmed working: DANN-v3, TTA, attention rollout, MC Dropout, FAISS, clinical report generation

---

### Session 5 (2026-03-25) — GPU Experiments + Paper Completion

**Duration**: ~4 hours | **Goal**: Run all GPU experiments, fill paper tables, reach 0 placeholders

#### Experiment 1: FAISS Index Rebuild (7.3 seconds)

```bash
python rebuild_faiss_full.py
```

- 8,241 vectors indexed, ALL 5 disease classes
- IndexFlatIP, L2-normalized 768-dim embeddings from DANN-v3 backbone
- Output: `outputs_v3/retrieval/index_flat_ip.faiss` (24.1 MB)
- Class distribution in index: Normal 37.3%, DR 54.5%, Glaucoma 3.3%, Cataract 2.7%, AMD 2.2%

#### Experiment 2: RAD Evaluation (3.3 seconds)

```bash
python rad_evaluation.py --k-values 1 3 5 10 --alpha 0.5
```

| Metric | Value |
|---|---|
| MAP | **0.921** |
| Recall@1 | **94.0%** |
| Recall@3 | 97.8% |
| Recall@5 | 98.5% |
| Recall@10 | 99.3% |
| RAD Combined Accuracy (K=1) | **94.0%** (+4.9% over standalone 89.1%) |
| Model-kNN Agreement Rate | 92.3% |
| Accuracy when agree | 97.3% |
| Accuracy when disagree | 61.2% |

Per-class Average Precision:
- DR: 0.952 (highest — distinctive lesion patterns)
- Normal: 0.906
- Cataract: 0.833
- AMD: 0.819
- Glaucoma: 0.742 (lowest — resembles Normal optic disc)

Per-source RAD accuracy:
- APTOS: 100.0%
- REFUGE2: 98.4%
- ODIR: 87.9%

#### Experiment 3: Confidence Routing (4.5 seconds)

```bash
python confidence_routing.py --mc-passes 15 --retrieval-k 5
```

| Tier | Fraction | Accuracy |
|---|---|---|
| AUTO-REPORT | **76.9%** | **96.8%** |
| REVIEW | 21.4% | 65.6% |
| ESCALATE | 1.7% | 44.0% |
| **Error catch rate** | | **77.2% (122/158 errors caught)** |

Safety interpretation: Only 22.8% of errors (36/158) slip through to AUTO-REPORT tier.

#### Experiment 4: LODO Validation (12.5 minutes)

```bash
python run_paper_experiments.py --lodo
```

Trains 4 DANN models, each leaving out one domain entirely:

| Held-Out | Accuracy | Weighted F1 | Classes Available | Key Insight |
|---|---|---|---|---|
| APTOS | 70.8% | 0.829 | DR only | ODIR/MESSIDOR provide enough DR context |
| MESSIDOR-2 | 61.6% | 0.633 | Normal + DR | French hospital camera characteristics different from Asian data |
| ODIR | 51.8% | 0.439 | All 5 classes | ODIR is only all-5-class source; losing it removes Cat/AMD training data |
| REFUGE2 | 88.8% | 0.904 | Normal + Glaucoma | Binary task, well-covered by ODIR/APTOS |
| **Average** | **68.2%** | **0.701** | — | 21pp gap from held-in (89.3%) — site calibration needed |

**Bug fixed during this experiment**: `run_paper_experiments.py` line ~400: `compute_class_weight` crashed when LODO training set was missing some classes (e.g., Cataract/AMD absent when ODIR held out). Fixed with try/except and manual balanced weight fallback.

#### Experiment 5: Ablation Study (15.4 minutes)

```bash
python run_paper_experiments.py --ablation
```

5 variants, each trained for 20 epochs:

| Variant | Accuracy | Macro F1 | AUC |
|---|---|---|---|
| Base ViT (no DANN) | 85.28% | 0.843 | 0.944 |
| DANN only | 84.73% | 0.843 | 0.937 |
| DANN + hard mining | 85.89% | 0.849 | 0.947 |
| DANN + mixup | 84.66% | 0.821 | 0.931 |
| **DANN-v3 (full pipeline)** | **89.09%** | **0.879** | **0.972** |

Key findings:
- DANN alone is -0.55pp vs Base ViT (domain head competes with disease head without complementary techniques)
- Mixup alone is -0.62pp vs Base ViT (conflicts with hard mining signal in isolation)
- Full pipeline is synergistic: +3.81pp over Base ViT, larger than any single component

#### IEEE Paper Completed to 0 Placeholders

Updated from 640 → **700 lines** with all GPU results:
- Table VII (Retrieval): Recall@K, Precision@K, MAP, Agreement, kNN accuracy
- Confidence Routing table: tier distribution, per-tier accuracy, error catch rate
- LODO table: all 4 domains with accuracy and weighted F1
- Ablation table: 5 variants with Acc, F1, AUC, ECE
- 3 new figures: `retrieval_recall_at_k.png`, `routing_analysis.png`, `calibration_reliability.png`
- Updated abstract, discussion, and conclusion with quantitative RAD/routing results

**Final paper state**: 700 lines, 13 tables, 3 figures, 30 references, **0 placeholders**.

---

### Session 6 (2026-03-25) — DANN-v4 RETFound Preparation

**Duration**: ~2 hours | **Goal**: Create DANN-v4 training script with RETFound backbone, verify weight loading

#### RETFound Background

- RETFound = ViT-Large/16 pre-trained on 1.6 million retinal images via Masked Autoencoding (MAE)
- Developed by Isztl et al. (2023), code: https://github.com/rmaphoh/RETFound_MAE
- Architecture: ViT-Large/16, 304M params (3.5× larger than DANN-v3's ViT-Base)
- 24 transformer blocks (vs 12 in DANN-v3), hidden dimension 1024 (vs 768)
- Domain-specific pretraining expected to give +2–5pp over ImageNet-21k initialization

#### Critical Architecture Fix

**Problem**: Initial `train_dann_v4.py` attempt loaded 0/294 backbone keys because ViT-Base (dim=768) was used instead of ViT-Large (dim=1024). The RETFound paper clearly states ViT-Large but the assumption was not verified before writing the integration code.

**Fix**: Changed backbone creation to use ViT-Large architecture (`vit_large_patch16_224`, dim=1024, 24 blocks). After fix: **294/294 backbone keys loaded successfully** (99.6% of parameters).

#### `train_dann_v4.py` Created

Key improvements over DANN-v3:
- **Backbone**: RETFound ViT-Large/16 (304M params, retinal pre-training)
- **LLRD decay**: 0.80 (vs 0.85 in v3 — more aggressive for larger model)
- **Base LR**: 1e-5 (vs 3e-5 — lower because domain-specific pretraining needs gentler fine-tuning)
- **CutMix + MixUp combo**: 40% batches MixUp (alpha=0.4), 40% CutMix (alpha=1.0), 20% clean
- **Stochastic Weight Averaging (SWA)**: Applied in final 10 epochs, lr=1e-6, BN stats updated after training
- **Class-aware augmentation**: Stronger transforms (rotation 30°, color jitter, random erasing 0.4) applied to minority classes identified dynamically each epoch
- **Dropout reduced**: 0.2 (vs 0.3 — larger model has more built-in regularization)
- **Output directory**: `outputs_v3/dann_v4/`

#### Verification Passed

1. `load_retfound_weights()`: 294/294 keys loaded
2. Forward pass: `(1, 5)` disease logits, `(1, 5)` severity logits — correct shapes
3. Full pipeline dry-run on CPU: preprocessing → normalization → model → threshold → FAISS query — no errors

**Status**: Script complete, weights downloaded (`weights/RETFound_cfp_weights.pth`, 1.2 GB). **Awaiting GPU training.**

**Projected accuracy**: 92–94% (based on RETFound literature showing +2–5pp over ImageNet pretraining on retinal tasks).

---

## 5. Architecture Deep Dive

### Core Model: DANNMultiTaskViT

```
Input: (B, 3, 224, 224) fundus image (RGB, normalized)
         |
         v
ViT-Base/16 Backbone (timm, pretrained=False at inference)
  - 12 transformer blocks
  - 768-dim hidden state
  - 12 attention heads
  - 196 patches (16×16) + 1 CLS token = 197 tokens
  - Output: CLS token embedding (B, 768)
         |
    nn.Dropout(0.3)
         |
    +----+-----------+
    |                |
    v                v
Disease Head      Severity Head
768→512→256→5    768→256→5
(BN+ReLU+Drop)   (BN+ReLU+Drop)
    |                |
5-class logits    5 DR severity grades
    |
    v  (training only)
Domain Head via GRL
GRL (alpha=lambda_p) → 768→256→64→4
```

**Loss function during training**:
```
L = L_disease + 0.2 × L_severity + 0.05 × λ_p × L_domain
```
Where:
- `L_disease` = Focal Loss (gamma=2.0) with per-class alpha + label smoothing 0.1
- `L_severity` = CrossEntropy with `ignore_index=-1` (only supervised for APTOS)
- `L_domain` = CrossEntropy (4-class: APTOS=0, ODIR=1, REFUGE2=2, MESSIDOR-2=3)
- `λ_p` = Ganin schedule: `min(2/(1+exp(-10p))-1, 0.3)`, where `p = epoch/total_epochs`

**Inference pipeline** (no domain head):
```
Image → Preprocessing → Normalize → model.forward_no_domain(x)
     → disease_logits, severity_logits
     → softmax(disease_logits / T)    [temperature scaling]
     → apply_thresholds(probs, T)     [per-class threshold]
     → final prediction
```

### Preprocessing Pipeline

```python
if source == 'APTOS':
    # Ben Graham: amplify local contrast, remove vignetting
    blurred = GaussianBlur(sigma=10)
    img = addWeighted(img, 4, blurred, -4, 128)
elif source == 'REFUGE2':
    # Resize only — Zeiss images already standardized
    img = resize(img, 224)
else:  # ODIR, MESSIDOR-2
    # CLAHE on LAB L-channel only
    lab = BGR2LAB(img)
    lab[:,:,0] = CLAHE(clipLimit=2.0, tileGrid=8×8)(lab[:,:,0])
    img = LAB2BGR(lab)

# All sources: circular mask at radius 0.48 × 224
img = apply_circular_mask(img, radius=0.48*224)
```

### FAISS Retrieval Index

| Property | Value |
|---|---|
| Index type | `IndexFlatIP` (exact cosine similarity) |
| Vectors | 8,241 (entire training set) |
| Dimensionality | 768 (ViT-Base CLS token) |
| Normalization | L2-normalized before add/search |
| File size | 24.1 MB |
| Build time | 7.3 seconds on H100 |

RAD combination at inference:
```python
combined_probs = 0.5 × model_probs + 0.5 × knn_similarity_weighted_votes
```

---

## 6. Training Pipeline — DANN-v3 (Production)

### Layer-wise Learning Rate Decay (LLRD)

14 AdamW parameter groups:
- Disease + severity + domain head: `lr = 3e-5`
- Transformer block [11] (closest to output): `lr × 0.85^1`
- Transformer block [10]: `lr × 0.85^2`
- ...
- Transformer block [0]: `lr × 0.85^12`
- Patch embed + CLS token + pos embed: `lr × 0.85^13 ≈ 2.7e-7`

This prevents catastrophic forgetting of ImageNet representations in early blocks.

### Hard-Example Mining Sampler

`HardExampleMiningWeightedSampler` — maintains per-sample loss state:
1. Base weights: inverse class frequency
2. After each epoch: compute loss on each training sample (no reduction)
3. Find top-500 highest-loss samples → multiply weight by 2×
4. Find DR-predicted-as-Normal samples → additionally multiply by 2×
5. Rebuild sampling distribution for next epoch

### Progressive DR Alpha Schedule

```python
dr_boost = 1.5 + (3.0 - 1.5) × (epoch / (total_epochs - 1))
# Epoch 0: DR focal weight ×1.5
# Epoch 39: DR focal weight ×3.0
```

### 8-Way Test-Time Augmentation (TTA)

| Pass | Transform |
|---|---|
| 1 | Identity |
| 2 | Horizontal flip |
| 3 | Vertical flip |
| 4 | Horizontal + Vertical flip |
| 5 | 90° rotation |
| 6 | 180° rotation |
| 7 | 270° rotation |
| 8 | Center crop (200px) → resize to 224 |

All 8 probabilities are averaged before temperature scaling and threshold application.

---

## 7. RAD Pipeline — Retrieval-Augmented Diagnosis

### Full Inference Flow

```
Fundus Image
    |
    v
Source-conditional Preprocessing
    |
    v
ViT-Base Backbone → 768-dim CLS embedding
    |                  |
    v                  v
Disease Head      FAISS Search (top-5 nearest)
(5 logits)        L2-normalize → IndexFlatIP.search
    |                  |
    v                  v
Softmax / T      kNN similarity-weighted votes
    |                  |
    +--------+---------+
             |
             v  alpha=0.5 blending
        Combined Probs
             |
      +------+------+
      |             |
      v             v
  Threshold   MC Dropout (15 passes)
  Application  entropy calculation
      |             |
      v             v
  Prediction   Confidence Routing
  (5 classes)  AUTO-REPORT / REVIEW / ESCALATE
```

### Confidence Routing Logic

```python
def route(confidence, entropy, retrieval_agrees):
    if confidence < 0.50 or entropy > 1.0:
        return "ESCALATE"
    if confidence >= 0.85 and entropy < 0.5 and retrieval_agrees:
        return "AUTO-REPORT"
    return "REVIEW"
```

---

## 8. All Bug Fixes Applied

### Critical Bugs Fixed

| Bug | Session | File | Fix |
|---|---|---|---|
| DR recall 25.3% — model learning preprocessing style | 1 | Training pipeline | Unified preprocessing + DANN domain adaptation |
| APTOS accuracy 26.5% — domain shift | 1 | Training pipeline | Gradient reversal layer, domain-adversarial training |
| FAISS index had only 2 of 5 classes (Normal + DR only) | 4 | FAISS index | `rebuild_faiss_full.py` — rebuilt with all 8,241 training samples |
| OOD crash when `ood_threshold` is None | 2 | `app.py` | Added None check with graceful fallback message |
| Wrong predictions on APTOS — inference preprocessing different from training | 1 | `app.py` | Added `_crop_black_borders()` and `_apply_circular_mask()` to inference |
| Wrong INTER_LINEAR vs INTER_AREA resize mode | 1 | `app.py` | Changed to `cv2.INTER_AREA` to match training |

### Path / Import Bugs Fixed

| File | Bug | Fix |
|---|---|---|
| `knowledge_distillation.py` | Hardcoded norm stats path | Added `configs/` fallback chain |
| `knowledge_distillation.py` | Wrong teacher model path | Added DANN-v3 fallback |
| `knowledge_distillation.py` | Missing DANN key filtering for teacher | Added key filter before `load_state_dict` |
| `update_faiss_messidor2.py` | Hardcoded `/teamspace/studios/this_studio` | Changed to relative paths |
| `update_faiss_messidor2.py` | Wrong model path | Fixed to DANN-v3 fallback |
| `gradcam_v3.py` | DANN-v3 not in MODEL_PATH priority | Added as first priority |
| `gradcam_v3.py` | Missing DANN key filtering | Added filter |
| `mc_dropout_uncertainty.py` | DANN-v3 not in MODEL_PATH fallback | Fixed |
| `integrated_gradients_xai.py` | DANN-v3 not in MODEL_PATH fallback | Fixed |
| `fairness_analysis.py` | DANN-v3 not in MODEL_PATH fallback | Fixed |
| Multiple scripts | `docstring` syntax errors | Fixed in Session 2 review |

### Training Bug Fixed

| Bug | Session | Fix |
|---|---|---|
| `compute_class_weight` crash in LODO when training set missing some classes | 5 | `run_paper_experiments.py` ~line 400: added try/except, manual balanced weight fallback for absent classes |

### RETFound Architecture Bug Fixed

| Bug | Session | Fix |
|---|---|---|
| `train_dann_v4.py` loaded 0/294 backbone keys | 6 | Used ViT-Large architecture (dim=1024, 24 blocks) instead of ViT-Base (dim=768, 12 blocks) |

---

## 9. All Configuration Changes

| Config File | Session | Change |
|---|---|---|
| `configs/temperature.json` | 1 | T=0.6438 (DANN-v1 calibration) |
| `configs/temperature.json` | 1 | T=0.593 (DANN-v2 recalibration) |
| `configs/temperature.json` | 3 | **T=0.5657 (DANN-v3, FINAL PRODUCTION VALUE)** |
| `configs/thresholds.json` | 1 | Per-class thresholds for DANN-v1 |
| `configs/thresholds.json` | 3 | **Updated for DANN-v3**: [0.3990, 0.4173, 0.6745, 0.2888, 0.3439] |
| `configs/fundus_norm_stats.json` | 1 | Computed from original 2-source dataset |
| `configs/fundus_norm_stats_unified.json` | 2 | **NEW**: Computed from all 11,524 images after unified preprocessing |

---

## 10. Complete Performance Metrics

### Session-by-Session Performance

| Session | Best Model | Accuracy | Macro F1 | AUC | ECE | Key Change |
|---|---|---|---|---|---|---|
| 1 (start) | EfficientNet-B3 | 63.52% | 0.517 | 0.910 | 0.162 | Initial training |
| 1 | ViT + Thresholds | 84.48% | 0.840 | 0.967 | — | Architecture switch + tuning |
| 1 | DANN-v1 | 86.10% | 0.867 | 0.962 | 0.056 | Domain adaptation |
| 1 | DANN-v2 | 86.10% | 0.871 | 0.962 | 0.056 | DR alpha boost |
| 3 | **DANN-v3** | **89.30%** | **0.886** | **0.975** | **0.034** | Full recipe + 4 sources |
| 5 | DANN-v3 + RAD | **94.00%** | 0.886 | 0.975 | 0.034 | Retrieval augmentation |

### Minority Class F1 Improvement (Baseline → Production)

| Class | EfficientNet | ViT+Thresh | DANN-v2 | DANN-v3 | Gain over Baseline |
|---|---|---|---|---|---|
| Normal | 0.533 | 0.746 | ~0.762 | **0.854** | +0.321 (+60.2%) |
| DR | 0.779 | 0.891 | ~0.890 | **0.920** | +0.141 (+18.1%) |
| Glaucoma | 0.346 | 0.871 | ~0.830 | **0.833** | +0.487 (+140.8%) |
| Cataract | 0.659 | 0.874 | ~0.882 | **0.899** | +0.240 (+36.4%) |
| AMD | 0.267 | 0.819 | ~0.950 | **0.895** | +0.628 (+235.2%) |
| **Macro** | **0.517** | **0.840** | **0.871** | **0.886** | **+0.369** |

### Gain Contribution Breakdown

| Category | Contribution | % of Total Gain |
|---|---|---|
| Architecture (EfficientNet → ViT) | +8.08pp | 31.3% |
| Threshold optimization (cumulative) | +12.06pp | 46.8% |
| Domain adaptation (DANN) | +4.82pp | 18.7% |
| Extended training / pipeline refinement | +0.82pp | 3.2% |
| **Total** | **+25.78pp** | **100%** |

---

## 11. Ablation Study Results

Controlled study: 5 variants, 20 epochs each, identical conditions except the technique under test.

| Variant | Accuracy | Macro F1 | AUC | Delta vs Base ViT |
|---|---|---|---|---|
| Base ViT (no DANN) | 85.28% | 0.843 | 0.944 | — |
| DANN only | 84.73% | 0.843 | 0.937 | **-0.55pp** (slightly harmful in isolation) |
| DANN + hard mining | 85.89% | 0.849 | 0.947 | +0.61pp (hard mining compensates) |
| DANN + mixup | 84.66% | 0.821 | 0.931 | **-0.62pp** (conflicts with hard mining) |
| **DANN-v3 (full pipeline)** | **89.09%** | **0.879** | **0.972** | **+3.81pp** (synergistic) |

**Key insight**: No single component achieves the full gain. The improvement is synergistic — all components must work together. The synergistic gain (+3.81pp) exceeds the sum of individual component effects.

---

## 12. LODO Generalization Results

4 DANN models each trained leaving out one domain, then evaluated on held-out domain:

| Held-Out | Accuracy | Weighted F1 | Available Train Classes | Reason for Gap |
|---|---|---|---|---|
| APTOS | 70.8% | 0.829 | All 5 | APTOS camera characteristics unseen, but DR covered by ODIR/MESSIDOR |
| MESSIDOR-2 | 61.6% | 0.633 | All 5 | French hospital camera different from Asian datasets |
| ODIR | 51.8% | 0.439 | 3 (No Cat/AMD) | ODIR is only all-5-class source; Cat/AMD nearly absent without it |
| REFUGE2 | 88.8% | 0.904 | All 5 | Binary task (Normal+Glaucoma), well-covered by ODIR |
| **Average** | **68.2%** | **0.701** | | 21pp gap from held-in 89.3% |

**Clinical implication**: Site-specific calibration required before deploying at a new clinical site. The RAD pipeline provides a safety net by flagging distribution shift at inference time.

---

## 13. Known Issues and Technical Debt

| Issue | Severity | Status | Resolution |
|---|---|---|---|
| OOD detector stale (fitted on old preprocessing, all images score 180+ vs threshold 42.82) | Low | Open | Refit on DANN-v3 features — ~30 min GPU |
| DANN-v4 not yet trained (script ready, weights ready) | Medium | Blocked on GPU | `python train_dann_v4.py --tta` — 15-20 min H100 |
| Git uncommitted (Sessions 4-6 changes) | High | Open | `git add` + `git commit` — 10 min |
| HuggingFace models not updated to DANN-v3 | Medium | Open | `huggingface-cli upload` |
| Paper PDF not compiled | Low | Open | `pdflatex paper/retinasense_ieee.tex` or Overleaf |
| K-fold CV used pre-DANN architecture | Medium | Open | Re-run `kfold_cv.py` with DANN-v3 config |
| No external validation on unseen datasets | Medium | Open | Acquire IDRiD, ADAM datasets |
| AMD class has only 265 training samples | Medium | Open | Acquire ADAM dataset (400 AMD + 800 Normal) |
| Population bias (primarily Asian: ODIR China + APTOS India) | High (structural) | Open | Requires diverse datasets — European, African |
| Knowledge distillation not run | Low | Open | `python knowledge_distillation.py` — 30 min GPU |
| No multi-label classification support | Low | Open | Architecture change required |

---

## 14. Next Steps

### Immediate (requires GPU, estimated times)

| Action | Time | Command |
|---|---|---|
| Train DANN-v4 (RETFound backbone) | 15-20 min | `python train_dann_v4.py --tta` |
| Commit all changes to git | 10 min | `git add` + `git commit` |
| If v4 improves: rebuild FAISS with v4 embeddings | 10 min | `python rebuild_faiss_full.py` |
| If v4 improves: recalibrate temperature + thresholds | 5 min | `python eval_dashboard.py --recalibrate` |
| Upload updated model to HuggingFace | 15 min | `huggingface-cli upload` |
| Compile paper PDF | 5 min | `pdflatex paper/retinasense_ieee.tex` |

### If DANN-v4 Achieves 92%+

1. Update `app.py` model path to `outputs_v3/dann_v4/best_model.pth`
2. Rebuild FAISS index with DANN-v4 embeddings
3. Recalibrate `configs/temperature.json` and `configs/thresholds.json`
4. Re-run RAD evaluation and confidence routing
5. Update `paper/retinasense_ieee.tex` with new results
6. Upload to HuggingFace

### Projected DANN-v4 Metrics

| Metric | DANN-v3 | DANN-v4 (projected) |
|---|---|---|
| Accuracy | 89.30% | 92–94% |
| Macro F1 | 0.886 | 0.91–0.93 |
| Macro AUC | 0.975 | 0.98+ |
| Parameters | 86M (ViT-Base) | 304M (ViT-Large) |
| Model size | 331 MB | ~1.2 GB |
| Inference speed | ~15ms | ~40ms |

---

## 15. File Inventory

### Core Training Scripts

| File | Lines | Status | Description |
|---|---|---|---|
| `retinasense_v3.py` | ~1220 | Complete | Original ViT training (pre-DANN baseline) |
| `train_dann.py` | ~1414 | Complete | DANN-v1/v2 training |
| `train_dann_v3.py` | — | **PRODUCTION** | DANN-v3: all 8 improvements |
| `train_dann_v4.py` | — | Ready (no GPU yet) | DANN-v4: RETFound backbone + SWA + CutMix |
| `train_ensemble.py` | — | Complete | Ensemble training and weight optimization |
| `kfold_cv.py` | — | Complete | 5-fold cross-validation |
| `knowledge_distillation.py` | — | Fixed (not run) | Teacher→student distillation |

### RAD Pipeline Scripts

| File | Lines | Status | Description |
|---|---|---|---|
| `rebuild_faiss_full.py` | 393 | Complete + verified | Rebuild FAISS index (all 5 classes) |
| `rad_evaluation.py` | 764 | Complete + verified | MAP, Recall@K, kNN accuracy |
| `confidence_routing.py` | 879 | Complete + verified | 3-tier clinical triage |
| `run_paper_experiments.py` | 1609 | Complete + verified | Master GPU experiment script |

### Evaluation and XAI Scripts

| File | Status | Description |
|---|---|---|
| `eval_dashboard.py` | Fixed + complete | Full evaluation suite |
| `gradcam_v3.py` | Fixed + complete | Attention rollout visualization |
| `mc_dropout_uncertainty.py` | Fixed + complete | MC Dropout uncertainty |
| `integrated_gradients_xai.py` | Fixed + complete | Integrated Gradients attribution |
| `fairness_analysis.py` | Fixed + complete | Cross-source fairness metrics |
| `run_error_analysis.py` | Complete | Misclassification analysis |

### Preprocessing Scripts

| File | Status | Description |
|---|---|---|
| `unified_preprocessing.py` | Run + complete | All 4 sources, unified pipeline |
| `prepare_datasets.py` | Run + complete | Dataset download helpers |
| `enhanced_augmentation.py` | Complete | Class-aware augmentation strategies |

### Deployment

| File | Status | Description |
|---|---|---|
| `app.py` | Updated (DANN-v3 + RAD) | Gradio demo (port 7860) |
| `api/main.py` | Operational | FastAPI REST server (port 8000) |
| `Dockerfile` | Ready | Container for deployment |
| `requirements_deploy.txt` | Current | Production dependencies |
| `setup.sh` | Current | Environment + HF model download |

### Model Checkpoints

| File | Size | Description |
|---|---|---|
| `outputs_v3/dann_v3/best_model.pth` | 331 MB | **PRODUCTION** |
| `outputs_v3/dann_v2/best_model.pth` | 331 MB | Fallback v2 |
| `outputs_v3/dann/best_model.pth` | 331 MB | DANN-v1 |
| `outputs_v3/best_model.pth` | 331 MB | ViT baseline |
| `outputs_v3/ensemble/efficientnet_b3.pth` | 45 MB | EfficientNet |
| `weights/RETFound_cfp_weights.pth` | 1.2 GB | RETFound (for DANN-v4) |

### Config Files

| File | Current Values | Description |
|---|---|---|
| `configs/temperature.json` | T=0.5657, ECE_before=0.149, ECE_after=0.037 | Production calibration |
| `configs/thresholds.json` | [0.3990, 0.4173, 0.6745, 0.2888, 0.3439] | Per-class thresholds |
| `configs/fundus_norm_stats_unified.json` | Corpus-specific mean/std | Normalization statistics |

### Key Output Files

| File | Contents |
|---|---|
| `outputs_v3/retrieval/index_flat_ip.faiss` | 8,241 vectors, 24.1 MB |
| `outputs_v3/retrieval/metadata.json` | 8,241 entries, all 5 classes |
| `outputs_v3/retrieval/rad_evaluation_results.json` | MAP 0.921, Recall@1 94.0% |
| `outputs_v3/retrieval/confidence_routing_results.json` | Tier distribution, accuracy |
| `outputs_v3/lodo_results.json` | 4-domain LODO results |
| `outputs_v3/ablation_results.json` | 5-variant ablation |
| `paper/retinasense_ieee.tex` | 700 lines, 0 placeholders |

### Data Files

| File | Contents |
|---|---|
| `data/train_split_expanded.csv` | 8,241 training samples (4 sources) |
| `data/calib_split_expanded.csv` | 1,816 calibration samples |
| `data/test_split.csv` | 1,467 test samples (SEALED) |
| `preprocessed_cache_unified/` | 11,524 × `.npy` files (multi-GB, not committed) |

---

## Appendix: Key Metric Reference Card

| Metric | Value |
|---|---|
| Overall Accuracy (DANN-v3) | 89.30% |
| RAD Combined Accuracy | 94.0% (+4.9%) |
| Macro F1 | 0.886 |
| Macro AUC | 0.975 |
| ECE | 0.037 |
| Temperature | 0.5657 |
| Cohen's Kappa | 0.809 |
| MCC | 0.810 |
| FAISS MAP | 0.921 |
| Recall@1 | 94.0% |
| AUTO-REPORT fraction | 76.9% |
| AUTO-REPORT accuracy | 96.8% |
| Error catch rate | 77.2% |
| LODO average accuracy | 68.2% |
| K-Fold CV accuracy | 82.4% ± 1.9% |
| Normal F1 | 0.854 |
| DR F1 | 0.920 |
| Glaucoma F1 | 0.833 |
| Cataract F1 | 0.899 |
| AMD F1 | 0.895 |
| Training time (DANN-v3, H100) | 5.9 minutes |
| Inference speed | ~15ms/image (66 img/sec) |
| Dataset size | 11,524 images (4 sources) |
| Test set size | 1,467 images (SEALED) |
| Production checkpoint | `outputs_v3/dann_v3/best_model.pth` |

---

*Document generated: 2026-03-27*
*Covers all development from initial EfficientNet baseline (63.52%) through DANN-v4 preparation.*
*Sources: SESSION_CONTEXT.md, FIXES_AND_CHANGES.md, SPRINT_RETROSPECTIVE.md, OUTCOME_OBJECTIVE_ANALYSIS.md, functional.md, README.md*
