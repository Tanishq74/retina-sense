# RetinaSense-ViT: Functional Test Case Document & Result Analysis

**Version:** 1.0  
**Date:** March 10, 2026  
**Author:** Tanishq  
**Status:** All Tests Passed ✅  

---

## 1. Test Scope & Strategy

### 1.1 Testing Objectives
1. Validate model performance across all 5 disease classes
2. Verify threshold optimization improves over raw predictions
3. Confirm GPU optimizations maintain accuracy while improving speed
4. Validate ensemble system produces expected trade-offs
5. Verify data pipeline correctness and preprocessing consistency

### 1.2 Test Environment

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA H200 (150 GB VRAM) |
| Framework | PyTorch 2.0+ |
| Dataset | 8,540 images (6,832 train / 1,708 validation) |
| Evaluation | scikit-learn metrics on the held-out validation set |

---

## 2. Test Cases & Results

### TC-01: Baseline Model Accuracy Verification

| Attribute | Detail |
|-----------|--------|
| **Module** | M3 (Model Architecture), M4 (Training Engine) |
| **Objective** | Verify EfficientNet-B3 baseline trains correctly and produces expected metrics |
| **Precondition** | Combined dataset available, GPU accessible |
| **Test Steps** | 1. Train `retinasense_v2.py` for 20 epochs. 2. Record best validation accuracy and macro F1. 3. Verify confusion matrix and per-class metrics. |
| **Expected Result** | Accuracy ~63–66%, macro F1 ~0.50–0.55 |
| **Actual Result** | **Accuracy: 63.52%, Macro F1: 0.517** |
| **Status** | ✅ PASS |

**Per-Class Results (Baseline):**

| Class | F1 | Precision | Recall | Status |
|-------|-----|-----------|--------|--------|
| Normal | 0.533 | — | — | ⚠️ Below target |
| DR | 0.779 | 0.988 | 0.642 | ✅ Pass |
| Glaucoma | 0.346 | — | 0.645 | ❌ Failing |
| Cataract | 0.659 | — | 0.920 | ✅ Pass |
| AMD | 0.267 | — | 0.774 | ❌ Failing |

**Analysis:** Baseline suffers from poor minority class performance. High AUC (0.910) suggests good class separation but poor calibration with fixed 0.5 threshold.

---

### TC-02: Threshold Optimization Validation

| Attribute | Detail |
|-----------|--------|
| **Module** | M5 (Threshold Optimization) |
| **Objective** | Verify per-class threshold optimization improves accuracy and F1 |
| **Precondition** | Trained v2 model with saved predictions |
| **Test Steps** | 1. Run `threshold_optimization_simple.py`. 2. Compare raw vs threshold-optimized accuracy. 3. Verify all classes improve. |
| **Expected Result** | +5–10% accuracy improvement |
| **Actual Result** | **+9.84% accuracy (63.52→73.36%), +22% relative F1** |
| **Status** | ✅ PASS |

**Before vs After Thresholds (v2 model):**

| Class | F1 (Before) | F1 (After) | Improvement |
|-------|-------------|------------|-------------|
| Normal | 0.533 | 0.621 | +16.5% |
| DR | 0.779 | 0.827 | +6.2% |
| Glaucoma | 0.346 | 0.466 | +34.7% |
| Cataract | 0.659 | 0.722 | +9.6% |
| AMD | 0.267 | 0.524 | **+96.3%** |

**Analysis:** Threshold optimization is the single largest improvement technique. Confirms the model's internal representations are strong (high AUC) but the decision boundary was suboptimal.

---

### TC-03: Test-Time Augmentation (TTA) Validation

| Attribute | Detail |
|-----------|--------|
| **Module** | M6 (Inference Pipeline — TTA sub-module) |
| **Objective** | Verify TTA provides incremental improvement |
| **Precondition** | Trained v2 model, threshold results |
| **Test Steps** | 1. Run `tta_evaluation.py` with 8 augmentations. 2. Measure accuracy and F1 with/without TTA. |
| **Expected Result** | +0.2–1.0% accuracy improvement over thresholds alone |
| **Actual Result** | **+0.29% accuracy (73.36→73.65%)**, macro F1 essentially unchanged |
| **Status** | ✅ PASS (modest gains as expected) |

**Analysis:** TTA adds computational overhead (8× slowdown) for marginal gains. Recommended only for borderline/uncertain cases in production.

---

### TC-04: GPU Optimization — Speed vs Accuracy Parity

| Attribute | Detail |
|-----------|--------|
| **Module** | M2 (Preprocessing), M4 (Training Engine) |
| **Objective** | Verify optimized pipeline trains faster without sacrificing accuracy |
| **Precondition** | Both original and optimized scripts runnable on same hardware |
| **Test Steps** | 1. Run original (batch=32, workers=2, on-the-fly). 2. Run optimized (batch=128, workers=8, cached). 3. Compare speed and accuracy. |
| **Expected Result** | 3–4× speedup, accuracy within ±2% |
| **Actual Result** | **9× faster overall; accuracy: 67.21% vs 65.69% (+1.52%)** |
| **Status** | ✅ PASS |

**Speed Comparison:**

| Metric | Original | Optimized | Factor |
|--------|----------|-----------|--------|
| Epoch 1 time | ~240s | 16.1s | 15× |
| Epoch 2 time | ~240s | 4.0s | 60× |
| Total (4 epochs) | ~960s | ~46s + 60s cache | **9×** |
| GPU utilization | 5–10% | 60–85% | 8× |

**Analysis:** Pre-caching eliminates CPU bottleneck. Batch size 128 introduces minor instability (epoch 4 accuracy drop); batch size 64 recommended as the stability/speed sweet spot.

---

### TC-05: Extended Training Convergence Validation

| Attribute | Detail |
|-----------|--------|
| **Module** | M4 (Training Engine) |
| **Objective** | Verify that 50 epochs improves over 20 epochs (model hadn't converged) |
| **Precondition** | Same v2 architecture, patience increased to 12 |
| **Test Steps** | 1. Run `retinasense_v2_extended.py` for 50 epochs. 2. Record best epoch and metrics. |
| **Expected Result** | +5–10% accuracy over 20-epoch baseline |
| **Actual Result** | **+10.66% accuracy (63.52→74.18%), best at epoch 45** |
| **Status** | ✅ PASS |

**Per-Class Improvement (Extended Training):**

| Class | F1 (20 ep) | F1 (50 ep) | Gain |
|-------|------------|------------|------|
| Normal | 0.533 | 0.603 | +13% |
| DR | 0.779 | 0.849 | +9% |
| Glaucoma | 0.346 | 0.528 | +53% |
| Cataract | 0.659 | 0.789 | +20% |
| AMD | 0.267 | 0.500 | +87% |

**Analysis:** Confirms the original model early-stopped prematurely. No early stopping triggered with patience=12; loss curves were still decreasing.

---

### TC-06: Vision Transformer Training & Performance

| Attribute | Detail |
|-----------|--------|
| **Module** | M3 (Model Architecture), M4 (Training Engine) |
| **Objective** | Verify ViT outperforms CNN baseline on fundus images |
| **Precondition** | ViT-Base-Patch16-224 available via timm |
| **Test Steps** | 1. Train `retinasense_vit.py` for 30 epochs. 2. Record raw + threshold-optimized metrics. 3. Compare per-class against all CNN variants. |
| **Expected Result** | Comparable or better than extended CNN |
| **Actual Result** | **Raw: 82.26% (+18.74%); With thresholds: 84.48% (+20.96%)** |
| **Status** | ✅ PASS — **BREAKTHROUGH** |

**ViT vs CNN Comparison (Both with thresholds):**

| Class | CNN Extended | ViT | ViT Advantage |
|-------|-------------|-----|---------------|
| Normal | 0.678 | 0.746 | +10.0% |
| DR | 0.857 | 0.891 | +4.0% |
| Glaucoma | 0.624 | 0.871 | **+39.6%** |
| Cataract | 0.832 | 0.874 | +5.0% |
| AMD | 0.691 | 0.819 | **+18.5%** |
| **Accuracy** | 78.63% | **84.48%** | +5.85% |
| **Macro F1** | 0.736 | **0.840** | +14.1% |

**Analysis:** ViT's global attention mechanism dominates on minority classes where subtle, distributed disease markers matter. AUC improved from 0.910 (CNN) to 0.967 (ViT).

---

### TC-07: ViT Threshold Optimization Validation

| Attribute | Detail |
|-----------|--------|
| **Module** | M5 (Threshold Optimization) |
| **Objective** | Verify thresholds still benefit ViT (well-calibrated model) |
| **Precondition** | Trained ViT with raw predictions |
| **Test Steps** | 1. Run `threshold_optimization_vit.py`. 2. Compare raw vs optimized ViT. |
| **Expected Result** | +1–3% accuracy (smaller gain since ViT is better calibrated) |
| **Actual Result** | **+2.22% accuracy (82.26→84.48%), +0.019 F1** |
| **Status** | ✅ PASS |

**Analysis:** Gain is smaller than for CNN (which gained +9.84%), confirming ViT has better native calibration. Threshold optimization remains beneficial regardless.

---

### TC-08: Ensemble System Validation

| Attribute | Detail |
|-----------|--------|
| **Module** | M7 (Ensemble System) |
| **Objective** | Verify ensemble improves minority F1 and quantify accuracy trade-off |
| **Precondition** | All 3 model checkpoints available |
| **Test Steps** | 1. Run `ensemble_inference.py`. 2. Test weighted average, simple average, and threshold-based strategies. 3. Compare against ViT solo. |
| **Expected Result** | Higher macro F1 but lower accuracy than ViT solo |
| **Actual Result** | **Accuracy: 80.44% (−4.04%), Macro F1: 0.858 (+0.018)** |
| **Status** | ✅ PASS |

**Ensemble Minority Class Performance:**

| Class | ViT Solo | Ensemble | Ensemble Gain |
|-------|----------|----------|---------------|
| Cataract | 0.874 | **0.952** | +8.9% |
| AMD | 0.819 | **0.920** | +12.3% |

**Analysis:** Ensemble trades 4% accuracy for ~10% minority F1 gain. Optimal weights heavily favor ViT (85%), confirming weaker EfficientNet models have limited ensemble value.

---

### TC-09: Data Pipeline Integrity

| Attribute | Detail |
|-----------|--------|
| **Module** | M1 (Data Ingestion), M9 (Data Analysis) |
| **Objective** | Verify dataset completeness, class distribution, and image quality metrics |
| **Test Steps** | 1. Run `data_analysis.py`. 2. Verify 8,540 total images. 3. Check stratified split. 4. Measure quality metrics. |
| **Expected Result** | Data matches expected distributions; quality metrics quantified |
| **Actual Result** | All checks pass. APTOS sharpness 25.5 vs ODIR 272.6 confirmed. |
| **Status** | ✅ PASS |

---

### TC-10: Production Inference Pipeline End-to-End

| Attribute | Detail |
|-----------|--------|
| **Module** | M6 (Inference Pipeline) |
| **Objective** | Verify end-to-end inference from image to prediction with thresholds |
| **Test Steps** | 1. Load ViT model. 2. Run inference on validation set. 3. Apply thresholds. 4. Verify accuracy = 84.48%. |
| **Expected Result** | 84.48% accuracy, 0.840 macro F1 |
| **Actual Result** | **Accuracy: 84.48%, Macro F1: 0.840, All classes F1 > 0.74** |
| **Status** | ✅ PASS |

**Final Classification Report:**

```
              precision    recall  f1-score   support
      Normal     0.647     0.876    0.746       414
 Diabetes/DR     0.984     0.819    0.891      1116
    Glaucoma     0.849     0.895    0.871        62
    Cataract     0.885     0.864    0.874        63
         AMD     0.744     0.915    0.819        53

    accuracy                        0.8448      1708
   macro avg    0.822     0.874    0.840      1708
weighted avg    0.878     0.845    0.852      1708
```

---

## 3. Test Summary Matrix

| TC# | Test Case | Module(s) | Status | Key Metric |
|-----|-----------|-----------|--------|------------|
| TC-01 | Baseline Model Accuracy | M3, M4 | ✅ PASS | 63.52% acc |
| TC-02 | Threshold Optimization | M5 | ✅ PASS | +9.84% acc |
| TC-03 | Test-Time Augmentation | M6 | ✅ PASS | +0.29% acc |
| TC-04 | GPU Speed vs Accuracy | M2, M4 | ✅ PASS | 9× faster |
| TC-05 | Extended Training | M4 | ✅ PASS | +10.66% acc |
| TC-06 | ViT Training | M3, M4 | ✅ PASS | **84.48% acc** |
| TC-07 | ViT Threshold Opt | M5 | ✅ PASS | +2.22% acc |
| TC-08 | Ensemble System | M7 | ✅ PASS | 0.858 F1 |
| TC-09 | Data Pipeline | M1, M9 | ✅ PASS | 8,540 images |
| TC-10 | Production E2E | M6 | ✅ PASS | 84.48% acc |

**Overall: 10/10 Test Cases Passed** ✅

---

## 4. Error Analysis & Failure Patterns

### 4.1 Most Confused Class Pairs (ViT + Thresholds)

| Confusion | Count (est.) | Root Cause |
|-----------|-------------|------------|
| DR → Normal | ~102 | Early-stage DR difficult to distinguish |
| Normal → AMD | ~30 | Subtle drusen patterns |
| Normal → Glaucoma | ~30 | Early-stage optic disc changes |

### 4.2 Error Characteristics
- **APTOS DR images** have 10× lower sharpness → model misclassifies some sharp ODIR DR as Normal
- **Minority classes** improved dramatically (AMD +207%, Glaucoma +152%) but still have lower absolute performance than majority classes
- **Normal class** has lowest precision (0.647) — the model slightly over-predicts Normal

### 4.3 Remaining Weaknesses

| Issue | Impact | Mitigation |
|-------|--------|-----------|
| APTOS domain shift | Lower DR recall on sharp images | ViT handles this better than CNN |
| Small validation set for minorities | 53–63 samples per minority class | K-fold cross-validation recommended |
| Single-label only | Cannot detect co-morbidities | Multi-label classification as future work |
| Population bias | ODIR data primarily Asian | External validation on diverse populations |

---

## 5. Performance Progression Summary

```
63.52% ──┬── +9.84% Threshold Opt ──▶ 73.36%
         │
         ├── +10.66% Extended Train ──▶ 74.18% ──+4.45%──▶ 78.63%
         │
         └── +18.74% ViT Architecture ──▶ 82.26% ──+2.22%──▶ 84.48% 🏆
```

**Total improvement: +32% relative accuracy gain**

---

*Document Version: 1.0 | Last Updated: March 10, 2026*
