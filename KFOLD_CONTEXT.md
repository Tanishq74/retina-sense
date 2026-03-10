# K-Fold Cross-Validation — Standalone Run Guide
## RetinaSense-ViT Phase 2A

> Start a fresh session, read this file, and run kfold_cv.py on a GPU machine.
> Estimated time: ~2 hours on H100 (30 epochs x 5 folds, early stopping at patience=8)

---

## What This Does

5-fold stratified cross-validation on the full train+calib pool (7,265 images).
Reports mean ± std for all metrics — gives confidence intervals for the paper.

This directly addresses reviewer criticism:
- "No cross-validation"
- "No confidence intervals"
- "Single run numbers are unreliable"

---

## How to Run

```bash
cd /teamspace/studios/this_studio
python kfold_cv.py
```

That's it. The script is fully self-contained.

---

## What It Needs

| Requirement | Path | Notes |
|-------------|------|-------|
| GPU | H100 recommended | CPU would take ~2 days |
| Train CSV | `data/train_split.csv` | 5,978 rows |
| Calib CSV | `data/calib_split.csv` | 1,287 rows |
| Image cache | `preprocessed_cache_v3/` | Must exist (already built) |
| Norm stats | `data/fundus_norm_stats.json` | Already exists |
| timm, torch | pip installed | Already installed |

---

## Config (inside kfold_cv.py)

```python
N_FOLDS    = 5
N_EPOCHS   = 30        # early stopping at PATIENCE=8
BATCH_SIZE = 32
BASE_LR    = 3e-4
LLRD_DECAY = 0.75      # layer-wise LR decay (same as main training)
MIXUP_ALPHA= 0.4
FOCAL_GAMMA= 1.0
```

---

## Expected Outputs → `outputs_v3/kfold/`

| File | Description |
|------|-------------|
| `fold_1_best.pth` ... `fold_5_best.pth` | Best checkpoint per fold |
| `kfold_results.json` | All metrics: mean ± std per metric |
| `fold_comparison.png` | Bar chart: accuracy/F1/AUC per fold with mean±σ |
| `perclass_f1_boxplot.png` | Boxplot of per-class F1 across 5 folds |

---

## Expected Results (estimated from single-run baseline)

The sealed test set (1,287 images) is NOT touched during K-Fold.
Each fold uses ~80% of the pool for training, 20% for validation.

| Metric | Expected Range |
|--------|---------------|
| Accuracy | 60–75% ± ~5% |
| Balanced Accuracy | 75–85% ± ~3% |
| Macro F1 | 0.65–0.75 ± ~0.03 |
| Macro AUC | 0.90–0.96 ± ~0.02 |

Note: APTOS domain gap means DR recall will be the main variance driver.

---

## Paper-Ready Output Format

After running, the results JSON will give you numbers like:

```
Accuracy:          67.4% ± 3.2%
Balanced Accuracy: 78.9% ± 2.1%
Macro F1:          0.703 ± 0.028
Macro AUC:         0.934 ± 0.015
```

Use these in the paper's Table 1 / Results section.

---

## What's Already Done (Don't Re-Run)

These completed on the previous GPU session:

| Script | Status | Outputs |
|--------|--------|---------|
| `eval_dashboard.py` | DONE | `outputs_v3/evaluation/` (7 files) |
| `mc_dropout_uncertainty.py` | DONE | `outputs_v3/uncertainty/` (6 files) |
| `integrated_gradients_xai.py` | DONE | `outputs_v3/xai/` (23 files) |
| `fairness_analysis.py` | DONE | `outputs_v3/fairness/` (7 files) |
| `train_ensemble.py` | DONE | `outputs_v3/ensemble/` (2 files) |

The only training tasks left are:
1. **K-Fold CV** → `python kfold_cv.py` (~2 hrs)
2. **Knowledge Distillation** → `python knowledge_distillation.py` (~30 min)

---

## After K-Fold Completes

Update `SESSION_CONTEXT.md` Phase 2A status to DONE and add the mean±std numbers.
Then run knowledge distillation if not yet done:

```bash
python knowledge_distillation.py
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| CUDA OOM | Reduce `BATCH_SIZE` from 32 to 16 in `kfold_cv.py` |
| Cache miss errors | Check `preprocessed_cache_v3/` exists and has .npy files |
| `KeyError: mean_rgb` | Norm stats file intact at `data/fundus_norm_stats.json` |
| Fold checkpoint not saving | Check disk space (`df -h .`) |
| Model arch mismatch | MultiTaskViT in kfold_cv.py matches retinasense_v3.py exactly |
