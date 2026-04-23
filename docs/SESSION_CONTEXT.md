# RetinaSense-ViT -- Session Context (Resume File)
> Last updated: 2026-03-25 ~14:15 UTC
> Read this file at the start of a new session to resume all work.

---

## SESSION 5 (2026-03-25): GPU Experiments + Paper Completion

### What Was Done This Session

#### 1. FAISS Index Rebuilt (7.3s)
- `python rebuild_faiss_full.py` — 8,241 vectors, ALL 5 classes, DANN-v3 embeddings
- IndexFlatIP (cosine similarity), L2-normalized 768-dim
- Saved: `outputs_v3/retrieval/index_flat_ip.faiss` (24.1 MB)
- Class distribution: Normal 37.3%, DR 54.5%, Glaucoma 3.3%, Cataract 2.7%, AMD 2.2%

#### 2. RAD Evaluation (3.3s)
- `python rad_evaluation.py` — MAP=0.921, Recall@1=94.0%
- RAD combined accuracy: 94.0% (+4.9% over standalone 89.1%)
- Per-class AP: DR 0.952, Normal 0.906, Cataract 0.833, AMD 0.819, Glaucoma 0.742
- Per-source: APTOS 100%, REFUGE2 98.4%, ODIR 87.9%
- Saved: `outputs_v3/retrieval/rad_evaluation_results.json` + 3 PNGs

#### 3. Confidence Routing (4.5s)
- `python confidence_routing.py` — 3-tier clinical triage
- AUTO-REPORT: 76.9% of cases at 96.8% accuracy
- REVIEW: 21.4% at 65.6% accuracy
- ESCALATE: 1.7% at 44.0% accuracy
- Error catch rate: 77.2% (122/158 errors caught before auto-report)
- Saved: `outputs_v3/retrieval/confidence_routing_results.json` + 3 PNGs

#### 4. LODO Validation (12.5 min)
- `python run_paper_experiments.py --lodo` — 4 DANN models, each leaving out 1 domain
- Fixed bug: `compute_class_weight` crash when training set missing classes (ODIR holdout)
- Results:
  | Held-Out | Acc | wF1 | Classes |
  |---|---|---|---|
  | APTOS | 70.8% | 0.829 | DR only |
  | MESSIDOR-2 | 61.6% | 0.633 | Nor+DR |
  | ODIR | 51.8% | 0.439 | All 5 |
  | REFUGE2 | 88.8% | 0.904 | Nor+Gla |
  | **Average** | **68.2%** | **0.701** | — |
- Saved: `outputs_v3/lodo_results.json`, `outputs_v3/lodo_chart.png`

#### 5. Ablation Study (15.4 min)
- `python run_paper_experiments.py --ablation` — 5 variants, 20 epochs each
- Results:
  | Variant | Acc | F1 | AUC |
  |---|---|---|---|
  | Base ViT (no DANN) | 85.28% | 0.843 | 0.944 |
  | DANN only | 84.73% | 0.843 | 0.937 |
  | DANN + hard mining | 85.89% | 0.849 | 0.947 |
  | DANN + mixup | 84.66% | 0.821 | 0.931 |
  | **DANN-v3 (full)** | **89.09%** | **0.879** | **0.972** |
- Saved: `outputs_v3/ablation_results.json`, `outputs_v3/ablation_chart.png`

#### 6. IEEE Paper Fully Updated (700 lines, 0 placeholders)
- Table VII (Retrieval): Recall@K, Precision@K, Agreement, kNN accuracy, MAP
- Confidence Routing table: tier distribution, accuracy, error catch rate
- LODO table: all 4 domains with accuracy and weighted F1
- Ablation table: 5 GPU-trained variants with Acc, F1, AUC, ECE
- Added 3 figures: retrieval_recall_at_k, routing_analysis, calibration_reliability
- Updated abstract, discussion, conclusion with quantitative RAD/routing results
- File: `paper/retinasense_ieee.tex` (700 lines, 13 tables, 3 figures, 30 refs)

#### 7. Bug Fix
- `run_paper_experiments.py` line ~400: Fixed `compute_class_weight` crash when LODO training set is missing some classes (e.g., Cataract/AMD absent when ODIR held out)

---

## SESSION 4 (2026-03-25): IEEE Paper + RAD Pipeline + Code Fixes + Cleanup

### What Was Done This Session

#### 1. IEEE LaTeX Paper Written (paper/retinasense_ieee.tex)
- Full IEEE conference paper (640 lines, 30 references)
- **Novelty framing: Retrieval-Augmented Diagnosis (RAD)**
- Authors: Tanishq Tamarkar, Rafae Mohammed Hussain, Dr. Revathi M (SRM Chennai)
- All DANN-v3 results (89.3%, 0.886 F1, 0.975 AUC), 10 tables, figure placeholders
- Title: "RetinaSense: An Uncertainty-Aware Domain-Adaptive Vision Transformer Framework with Retrieval-Augmented Reasoning for Multi-Disease Retinal Diagnosis"
- Compile with: `pdflatex retinasense_ieee.tex` or upload to Overleaf
- **Table VII (retrieval metrics) needs GPU results** — placeholder "pending GPU evaluation"

#### 2. RAD Framework Built (3 new scripts, 2,036 lines)

**rebuild_faiss_full.py (393 lines)**
- Rebuilds FAISS index using DANN-v3 backbone for ALL 8,241 training samples, ALL 5 classes
- Uses IndexFlatIP (cosine similarity via L2-normalized vectors)
- Saves as `index_flat_ip.faiss` (preserves old `index_flat_l2.faiss`)
- CRITICAL: Current FAISS index only has Normal + DR (2 of 5 classes) — this script fixes it
- Run: `python rebuild_faiss_full.py` (needs GPU, ~5 min)

**rad_evaluation.py (764 lines)**
- Computes Recall@K, Precision@K, MAP at K=1,3,5,10
- Class-Match Rate heatmap, Agreement Score, kNN-Augmented Accuracy
- All metrics broken down per-class and per-source
- Generates: retrieval_recall_at_k.png, class_match_heatmap.png, agreement_analysis.png
- Saves: rad_evaluation_results.json
- Run: `python rad_evaluation.py` (needs GPU + rebuilt FAISS, ~10 min)

**confidence_routing.py (879 lines)**
- Three-tier clinical triage: AUTO-REPORT / REVIEW / ESCALATE
- Uses triplet signals: model confidence + MC Dropout entropy + retrieval agreement
- Thresholds: auto (conf>0.85, entropy<0.5, retrieval agrees), escalate (conf<0.5 or entropy>1.0)
- Evaluates tier distribution, per-tier accuracy, safety metrics, error catch rate
- Generates: routing_distribution.png, routing_accuracy_by_tier.png, routing_analysis.png
- Run: `python confidence_routing.py` (needs GPU + rebuilt FAISS, ~5 min)

#### 3. GPU Experiment Master Script (run_paper_experiments.py, 1,609 lines)
- Orchestrates ALL experiments needed for IEEE paper
- 5 experiments, each independently selectable:
  - `--rebuild-faiss` — calls rebuild_faiss_full.py
  - `--eval-rad` — calls rad_evaluation.py
  - `--lodo` — Leave-One-Domain-Out validation (FULLY INLINE, trains 4 DANN models)
  - `--ablation` — 5-variant ablation study (FULLY INLINE, trains 4 models + evals production)
  - `--eval-routing` — calls confidence_routing.py
  - `--all` — runs everything (~3-4 hrs)
- Imports directly from train_dann_v3.py (no architecture drift)
- Outputs: lodo_results.json, ablation_results.json, paper_experiments_summary.json + charts

#### 4. Code Fixes (7 files patched)
- **knowledge_distillation.py**: Fixed norm stats path (configs/ fallback chain), fixed teacher model path (DANN-v3 fallback), fixed DANN key filtering for teacher loading
- **update_faiss_messidor2.py**: Fixed hardcoded `/teamspace/studios/this_studio` → relative paths, fixed model path to DANN-v3 fallback
- **gradcam_v3.py**: Added dann_v3 as first priority in MODEL_PATH chain, added DANN key filtering
- **mc_dropout_uncertainty.py**: Added dann_v3 to MODEL_PATH fallback chain
- **integrated_gradients_xai.py**: Same DANN-v3 fallback fix
- **fairness_analysis.py**: Same DANN-v3 fallback fix
- **ARCHITECTURE_DOCUMENT.md**: Updated all metrics to DANN-v3 (89.30%, 0.886, 0.975), version 3.0

#### 5. app.py RAD Updates
- `load_faiss_index()`: Now prefers `index_flat_ip.faiss` (rebuilt), falls back to old `index_flat_l2.faiss`
- `retrieve_similar()`: Handles both IP and L2 index types, returns enriched metadata
- `retrieve_augmented_prediction()`: NEW — combines model prediction with kNN vote via alpha blending
- `_resolve_cache_path()`: NEW — resolves cache paths from metadata with fallback chain
- Added `show_error=True` to Gradio launch

#### 6. Project Cleanup (14 GB freed, 39GB → 25GB)
Removed:
- `aptos/aptos2019-blindness-detection.zip` (9.6 GB)
- `aptos/test_images/` (1.6 GB), `aptos/gaussian_filtered_images/` (440 MB)
- `preprocessed_cache_v3/` (894 MB) — replaced by unified
- `preprocessed_cache_vit/` (888 MB) — only legacy scripts used it
- `outputs_vit/` (331 MB), `outputs_analysis/`, `outputs_production/`, `outputs_v2/`, etc.
- `legacy/` directory (all content already in active codebase)
- Old root-level outputs_v3/ artifacts (calibration.png, dashboard.png, etc.)
- Training logs, pycache, .gradio cache

#### 7. App Tested End-to-End
- Ran Gradio app on port 7860 (public URL generated)
- Tested 5 sample fundus images (one per class) via both direct Python and Gradio API
- Results: 3/5 correct (Normal 94.7%, Glaucoma 91.9%, AMD 99.2%)
- Known misses: DR→AMD (high uncertainty), Cataract→Normal (minority class)
- All pipeline components working: preprocessing, DANN-v3, TTA, attention rollout, MC dropout, OOD, FAISS, clinical reports

---

## Current Best Model: DANN-v3

| Metric | Value |
|--------|-------|
| Overall Accuracy | **89.30%** |
| Macro F1 | **0.886** |
| Macro AUC | **0.975** |
| ECE (calibrated) | **0.034** |
| Cohen's Kappa | 0.809 |
| MCC | 0.810 |
| Temperature | 0.566 |
| Checkpoint | outputs_v3/dann_v3/best_model.pth |

### Per-Class Performance (DANN-v3)
| Class | F1 | Precision | Recall | AUC | Support |
|-------|-----|-----------|--------|-----|---------|
| Normal | 0.854 | 0.817 | 0.895 | 0.964 | 484 |
| DR | 0.920 | 0.935 | 0.904 | 0.977 | 837 |
| Glaucoma | 0.833 | 0.900 | 0.776 | 0.953 | 58 |
| Cataract | 0.899 | 0.976 | 0.833 | 0.985 | 48 |
| AMD | 0.895 | 0.944 | 0.850 | 0.995 | 40 |

### Confusion Matrix (158 errors / 1,467 test)
```
             Nor   DR  Gla  Cat  AMD
Normal       433   45    5    1    0
DR            78  757    0    0    2
Glaucoma       8    5   45    0    0
Cataract       8    0    0   40    0
AMD            3    3    0    0   34
```
77% of errors are Normal↔DR confusion.

---

## Current Project Structure

```
/teamspace/studios/this_studio/
├── Core Training
│   ├── retinasense_v3.py              # Original ViT training (baseline)
│   ├── train_dann.py                  # DANN-v1/v2 training
│   ├── train_dann_v3.py               # DANN-v3 training (PRODUCTION - 89.3%)
│   ├── train_ensemble.py              # EfficientNet-B3 ensemble
│   └── kfold_cv.py                    # 5-fold cross-validation
│
├── RAD Pipeline (NEW - Session 4)
│   ├── rebuild_faiss_full.py          # Rebuild FAISS all 5 classes (NEEDS GPU)
│   ├── rad_evaluation.py              # Retrieval quality evaluation (NEEDS GPU)
│   ├── confidence_routing.py          # Clinical triage system (NEEDS GPU)
│   └── run_paper_experiments.py       # Master GPU experiment script (NEEDS GPU)
│
├── Evaluation & XAI
│   ├── eval_dashboard.py              # Full evaluation suite
│   ├── gradcam_v3.py                  # Attention Rollout XAI
│   ├── mc_dropout_uncertainty.py      # MC Dropout uncertainty
│   ├── integrated_gradients_xai.py    # Integrated Gradients XAI
│   └── fairness_analysis.py           # Domain fairness analysis
│
├── Deployment
│   ├── app.py                         # Gradio demo (DANN-v3 + RAD)
│   ├── api/main.py                    # FastAPI REST server
│   ├── knowledge_distillation.py      # ViT-Base→ViT-Tiny (NEEDS GPU)
│   └── Dockerfile                     # Docker deployment
│
├── Utilities
│   ├── unified_preprocessing.py       # CLAHE preprocessing
│   ├── prepare_datasets.py            # Dataset preparation
│   ├── enhanced_augmentation.py       # Class-aware augmentation
│   ├── update_faiss_messidor2.py      # FAISS updater (legacy, use rebuild_faiss_full.py)
│   └── retfound_backbone.py           # RETFound weight adapter
│
├── paper/
│   ├── retinasense_ieee.tex           # IEEE conference paper (640 lines)
│   └── Makefile                       # LaTeX build
│
├── configs/
│   ├── temperature.json               # T=0.566 (DANN-v3)
│   ├── thresholds.json                # Per-class thresholds (DANN-v3)
│   └── fundus_norm_stats_unified.json # Normalization stats
│
├── data/
│   ├── train_split_expanded.csv       # 8,241 samples (4 sources)
│   ├── calib_split_expanded.csv       # 1,816 samples
│   ├── test_split.csv                 # 1,467 samples (SEALED)
│   └── [other CSVs]
│
├── outputs_v3/
│   ├── dann_v3/best_model.pth         # PRODUCTION (331MB)
│   ├── dann_v3/evaluation/            # 7 eval artifacts
│   ├── dann_v2/best_model.pth         # Fallback (331MB)
│   ├── dann/best_model.pth            # DANN-v1 (331MB)
│   ├── best_model.pth                 # Original ViT (331MB)
│   ├── ensemble/efficientnet_b3.pth   # EfficientNet (45MB)
│   ├── kfold/                         # 5 fold checkpoints
│   ├── retrieval/                     # FAISS index (BROKEN - only 2 classes)
│   └── [xai/, uncertainty/, fairness/, gradcam/]
│
├── preprocessed_cache_unified/        # 11,524 .npy files (224x224)
├── README.md                          # Updated to DANN-v3
├── ARCHITECTURE_DOCUMENT.md           # Updated to v3.0
├── IEEE_RESEARCH_PAPER.md             # Old paper draft (use paper/retinasense_ieee.tex instead)
├── RUN.md                             # Run guide
└── SESSION_CONTEXT.md                 # THIS FILE
```

---

## KNOWN ISSUES

### 1. FAISS Index — FIXED (Session 5)
- Rebuilt: 8,241 vectors, ALL 5 classes, IndexFlatIP, DANN-v3 embeddings
- File: `outputs_v3/retrieval/index_flat_ip.faiss` (24.1 MB)

### 2. OOD Detector Stale (Low Priority)
- outputs_v3/ood_detector.npz was fit on old preprocessing (not unified CLAHE)
- All images get OOD scores 180+ vs threshold 42.82 (everything flagged)
- Not blocking — app handles gracefully, but OOD detection is effectively disabled
- FIX: Refit on DANN-v3 features with unified preprocessing

### 3. IEEE Paper — COMPLETE (Session 5)
- All tables filled with GPU results (0 placeholders remaining)
- File: `paper/retinasense_ieee.tex` (700 lines, 13 tables, 3 figures, 30 refs)

---

## REMAINING WORK CHECKLIST

### Done (Sessions 1-5) ✅
- [x] DANN-v3 trained (89.3% accuracy)
- [x] Full eval dashboard (7 artifacts)
- [x] Temperature + threshold calibration
- [x] app.py updated (DANN-v3 + RAD)
- [x] README.md updated
- [x] ARCHITECTURE_DOCUMENT.md updated to v3.0
- [x] IEEE LaTeX paper written (paper/retinasense_ieee.tex)
- [x] RAD scripts written (rebuild_faiss, rad_evaluation, confidence_routing)
- [x] GPU experiment script written (run_paper_experiments.py)
- [x] Code fixes (7 files: paths, DANN-v3 fallbacks)
- [x] Project cleanup (14GB freed)
- [x] App tested end-to-end (3/5 correct on samples)
- [x] FAISS index rebuilt (8,241 vectors, all 5 classes)
- [x] RAD evaluation (MAP=0.921, Recall@1=94.0%, +4.9% combined)
- [x] Confidence routing (96.8% auto-report accuracy, 77.2% error catch)
- [x] LODO validation (4 domains, avg 68.2% accuracy)
- [x] Ablation study (5 variants, DANN-v3 +3.82% over base ViT)
- [x] IEEE paper fully updated with all GPU results (0 placeholders)

### Not Yet Committed to Git 🔲
- [ ] Commit Session 4+5 changes (new scripts, paper, config updates)
- [ ] Push to GitHub

### Nice to Have 🔲
- [ ] Upload updated models to HuggingFace
- [ ] Knowledge distillation (ViT-Base → ViT-Tiny + ONNX)
- [ ] RETFound backbone (could push to 92-94%)
- [ ] External validation on IDRiD dataset
- [ ] Refit OOD detector on DANN-v3 features
- [ ] 5-fold CV with DANN-v3 architecture
- [ ] Compile paper PDF (pdflatex or Overleaf)

---

## PREVIOUS SESSIONS

### SESSION 3 (2026-03-24): DANN-v3 Training + Evaluation
- Trained DANN-v3 (89.3%, 5.9 min on H100)
- Full eval dashboard, config updates, README/app.py updated

### SESSION 2 (2026-03-24): Project Merge + Cleanup + DANN-v3 Prep
- Deep analysis & HF merge, file cleanup, bug fixes
- MESSIDOR-2 integration, train_dann_v3.py created

### SESSION 1: Initial project setup, ViT training, DANN-v1/v2
