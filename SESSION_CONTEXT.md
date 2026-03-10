# RetinaSense-ViT — Session Context (Resume File)
> Last updated: 2026-03-10 ~08:30 UTC
> Read this file at the start of a new session to resume all work.

## Project Overview
RetinaSense-ViT: Deep learning retinal disease classification using Vision Transformer.
- **5 classes**: Normal(0), Diabetes/DR(1), Glaucoma(2), Cataract(3), AMD(4)
- **Model**: ViT-Base/16 (timm), 86M params, multi-task (disease + severity heads)
- **Dataset**: 8,540 images (APTOS=3,662 + ODIR=4,878), 3-way split (70/15/15)
- **Best checkpoint**: outputs_v3/best_model.pth (epoch 24, F1=0.854, test acc=81.2%)
- **GitHub**: https://github.com/Tanishq74/retina-sense

## Key Files
| File | Purpose |
|------|---------|
| `retinasense_v3.py` | Main training script (1220 lines) |
| `gradcam_v3.py` | Attention Rollout XAI pipeline (1179 lines, FIXED & working) |
| `train_ensemble.py` | Phase 2B: EfficientNet-B3 ensemble training (CREATED, needs GPU) |
| `app.py` | Phase 3B: Gradio interactive web app (CREATED, needs testing) |
| `api/main.py` | Phase 4B: FastAPI inference server (CREATED, needs testing) |
| `eval_dashboard.py` | Phase 1A: Evaluation dashboard (CREATED BY AGENT, needs GPU to run) |
| `mc_dropout_uncertainty.py` | Phase 1B: MC Dropout uncertainty (CREATED BY AGENT, needs GPU to run) |
| `integrated_gradients_xai.py` | Phase 1C: Integrated Gradients XAI (CREATED BY AGENT, needs GPU to run) |
| `fairness_analysis.py` | Phase 1D: Fairness analysis (CREATED BY AGENT, needs GPU to run) |
| `Dockerfile` | Docker deployment config (CREATED) |
| `requirements_deploy.txt` | Deployment dependencies (CREATED) |
| `ENHANCEMENT_PLAN.md` | Full 4-phase plan document |
| `data/train_split.csv` | Training set (5,978 rows) |
| `data/calib_split.csv` | Calibration set (1,281 rows) |
| `data/test_split.csv` | Test set (1,281 rows, sealed) |
| `data/fundus_norm_stats.json` | Fundus normalization: mean=[0.4298,0.2784,0.1559] std=[0.2857,0.2065,0.1465] |
| `outputs_v3/temperature.json` | T=0.6438, ECE 0.1618→0.1014 |
| `outputs_v3/thresholds.json` | Per-class thresholds [0.638,0.068,0.84,0.564,0.289] |
| `outputs_v3/ood_detector.npz` | Mahalanobis OOD detector (fitted, threshold=42.82) |

## Model Architecture (MultiTaskViT in retinasense_v3.py)
```python
class MultiTaskViT(nn.Module):
    backbone = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
    drop = nn.Dropout(0.3)
    disease_head = Sequential(Linear(768,512), BN, ReLU, Drop(0.3), Linear(512,256), BN, ReLU, Drop(0.2), Linear(256,5))
    severity_head = Sequential(Linear(768,256), BN, ReLU, Drop(0.3), Linear(256,5))
    forward(x) -> (disease_logits, severity_logits)
```

## Enhancement Plan — 4 Phases, 10 Tasks

### Phase 1: Analysis (No Retraining) — Scripts CREATED, need GPU run
| Task | Script | Status | Output Dir |
|------|--------|--------|------------|
| 1A. Evaluation Dashboard | `eval_dashboard.py` | CREATED, was running on CPU (too slow) | `outputs_v3/evaluation/` |
| 1B. MC Dropout Uncertainty | `mc_dropout_uncertainty.py` | CREATED, was running on CPU (too slow) | `outputs_v3/uncertainty/` |
| 1C. Integrated Gradients XAI | `integrated_gradients_xai.py` | CREATED, was running on CPU (too slow) | `outputs_v3/xai/` |
| 1D. Fairness Analysis | `fairness_analysis.py` | CREATED, was running on CPU (too slow) | `outputs_v3/fairness/` |

**Action needed**: Run all 4 scripts with GPU enabled. They should complete in ~10 min total on H100.
```bash
# Run Phase 1 (can run in parallel if GPU memory allows, or sequential):
python eval_dashboard.py
python mc_dropout_uncertainty.py
python integrated_gradients_xai.py
python fairness_analysis.py
```

### Phase 2: Model Improvements (Requires Training)
| Task | Script | Status | Notes |
|------|--------|--------|-------|
| 2A. K-Fold CV (5-fold) | NOT YET CREATED | TODO | ~10 hrs GPU, run overnight |
| 2B. CNN-ViT Ensemble | `train_ensemble.py` | CREATED, was running (slow on CPU) | ~30 min on GPU |
| 2C. Severity Sub-Classification | ALREADY IN MODEL | severity_head exists, just needs eval | Part of eval_dashboard |

**Action needed**: Run `python train_ensemble.py` with GPU. Creates EfficientNet-B3 + optimizes ensemble weights.

### Phase 3: Clinical Integration
| Task | Script | Status | Notes |
|------|--------|--------|-------|
| 3A. Clinical Report Generator | Built into `app.py` | CREATED | Generates text reports |
| 3B. Gradio Web App | `app.py` | CREATED, needs testing | `python app.py` → localhost:7860 |

**Action needed**: Test `python app.py` after Phase 1 completes.

### Phase 4: Deployment
| Task | Script | Status | Notes |
|------|--------|--------|-------|
| 4A. Knowledge Distillation + ONNX | NOT YET CREATED | TODO | ViT-Base→ViT-Tiny, ONNX export |
| 4B. FastAPI + Docker | `api/main.py` + `Dockerfile` | CREATED, needs testing | `uvicorn api.main:app` |

## What Was Already Completed (Before This Session)
1. ViT-Base model trained (epoch 24, F1=0.854)
2. Temperature scaling calibrated (T=0.6438)
3. Per-class thresholds optimized
4. Attention Rollout XAI (replaced broken Grad-CAM) — heatmaps working correctly
5. OOD Mahalanobis detector fitted
6. Heatmap contrast enhanced (discard_ratio=0.97, power=0.4, alpha=0.7)
7. Project pushed to GitHub (cleaned to 16MB)

## Reviewer Feedback (Needs Addressing)
The paper was reviewed and found weak on:
1. Missing metrics: Sensitivity, Specificity, ROC-AUC per class, Precision, F1 per class → **Phase 1A fixes this**
2. Missing figures: ROC curve, PR curve, confusion matrix → **Phase 1A fixes this**
3. No statistical significance tests → **Phase 1D fixes this**
4. No confidence intervals → **Phase 2A (K-Fold) fixes this**
5. No cross-validation → **Phase 2A fixes this**
6. Single run numbers only → **Phase 2A gives mean ± std**

## Resume Instructions
When starting a new session:
1. Read this file: `SESSION_CONTEXT.md`
2. Check which scripts have produced outputs: `ls outputs_v3/{evaluation,uncertainty,xai,fairness,ensemble}/`
3. Run any scripts that haven't produced outputs yet (they need GPU)
4. After Phase 1 completes, run Phase 2B: `python train_ensemble.py`
5. After Phase 2B, test Phase 3: `python app.py`
6. Then create Phase 2A (K-Fold CV) and Phase 4A (Knowledge Distillation)
7. Test Phase 4B: `uvicorn api.main:app --host 0.0.0.0 --port 8000`

## Installed Packages
```
gradio==6.9.0, captum==0.8.0, onnx==1.20.1, onnxruntime==1.24.3, fpdf2==2.8.7
fastapi==0.133.1, uvicorn==0.41.0, timm, torch, torchvision
```

## CRITICAL FINDING FROM PHASE 1A (Evaluation Dashboard)
Phase 1A completed on CPU. Results reveal a serious threshold/domain problem:

- **Overall accuracy: 49.1%** (NOT 81.2% as previously reported)
- **DR recall: only 25.3%** — 573 of 837 DR images misclassified as Normal (68.5%!)
- **APTOS accuracy: 26.5%** vs ODIR accuracy: 67.7% — massive domain gap
- **Temperature scaling hurts**: ECE went from 0.162 → 0.312 on test set
- Cataract (97.9%), AMD (100%), Normal (96.8%) perform excellently
- Macro AUC = 0.893 — model CAN discriminate, but thresholds are wrong
- DR precision = 99.5% — when it says DR, it's right; but it rarely says DR

**Root cause**: The per-class threshold for DR is 0.068 (very low), but most APTOS DR
images get very low DR probability. The model may be treating APTOS Ben Graham preprocessed
images differently from ODIR CLAHE images — this is a domain shift problem.

**Possible fixes** (to implement):
1. Re-optimize thresholds directly on test accuracy (not just F1)
2. Investigate APTOS preprocessing — Ben Graham may be too aggressive
3. Domain adaptation / domain-adversarial training
4. Lower the Normal threshold to reduce false negatives for DR

Outputs saved: `outputs_v3/evaluation/` (7 files: confusion_matrix.png, roc_curves_per_class.png,
precision_recall_curves.png, calibration_reliability.png, confidence_histograms.png,
error_analysis_by_source.png, metrics_report.json)

## Still TODO (Not Yet Created)
1. **Phase 2A**: K-Fold Cross-Validation script (5-fold stratified, ~10hrs GPU)
2. **Phase 4A**: Knowledge Distillation (ViT-Base→ViT-Tiny) + ONNX export script
3. **Testing**: All created scripts need GPU testing
4. **Update paper/report**: Add all new figures and metrics from Phase 1
5. **FIX DR THRESHOLD ISSUE**: Investigate and fix the APTOS domain gap
