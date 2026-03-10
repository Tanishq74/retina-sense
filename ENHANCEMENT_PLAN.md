# RetinaSense-ViT Enhancement Plan
## From Research Prototype to Production-Grade Clinical AI

---

## Current Baseline Audit

| Asset | Status |
|-------|--------|
| Model | ViT-Base/16, 86M params, 331MB checkpoint |
| Performance | 81.2% accuracy, F1=0.813, AUC=0.949 |
| Classes | 5 (Normal, DR, Glaucoma, Cataract, AMD) |
| Dataset | 8,540 images (APTOS=3,662 + ODIR=4,878) |
| Imbalance | 21:1 (DR=5,581 vs AMD=265) |
| XAI | Attention Rollout (working) |
| Calibration | Temperature scaling (T=0.6438, ECE=0.101) |
| OOD | Mahalanobis distance detector |
| Infrastructure | H100 80GB, 378GB disk, PyTorch 2.x |

### Critical Limitations Discovered
- **No multi-label annotations** in current dataset (ODIR raw labels lost during preprocessing)
- **No paired L/R eye data** (only left-eye ODIR images available)
- **Packages needed**: gradio, captum, shap, onnx, onnxruntime, fpdf

These constraints eliminate 2 originally proposed features (Multi-Label, Patient-Level Aggregation)
and reframe others. The plan below is adjusted for reality.

---

## Phase 1: Evaluation & Analysis Foundation (No Retraining)
> Goal: Extract maximum insight from the existing trained model

### 1A. Rich Evaluation Dashboard
**What**: Interactive confusion matrix, per-class ROC curves, precision-recall curves,
calibration reliability diagram, confidence distribution histograms, error analysis by
image source (APTOS vs ODIR).

**Research value**: HIGH — every paper needs these. Reviewers reject without proper evaluation.

**Practical value**: MEDIUM — helps identify model weaknesses for targeted improvement.

**Requirements**:
- matplotlib, seaborn (already installed)
- Run inference on full test set (1,281 images)
- ~5 min compute on H100

**Feasibility**: TRIVIAL — pure analysis, no model changes.

**Deliverables**:
- `outputs_v3/evaluation/confusion_matrix.png`
- `outputs_v3/evaluation/roc_curves_per_class.png`
- `outputs_v3/evaluation/precision_recall_curves.png`
- `outputs_v3/evaluation/calibration_reliability.png`
- `outputs_v3/evaluation/confidence_histograms.png`
- `outputs_v3/evaluation/error_analysis_by_source.png`
- `outputs_v3/evaluation/metrics_report.json`

**Risks**: None.

---

### 1B. MC Dropout Uncertainty Quantification
**What**: Enable dropout at inference time, run T=30 stochastic forward passes per image.
Compute predictive mean, variance, and entropy. Classify uncertainty as:
- Low uncertainty + correct = TRUSTED prediction
- High uncertainty + any = REFER TO SPECIALIST
- Low uncertainty + wrong = DANGEROUS (model is confidently wrong)

**Research value**: VERY HIGH — uncertainty quantification is a hot topic in medical AI.
Bayesian deep learning via MC Dropout (Gal & Ghahramani, 2016) is the standard approach.
Separating epistemic (model doesn't know) vs aleatoric (data is noisy) uncertainty is publishable.

**Practical value**: VERY HIGH — clinical safety. A model that says "I don't know" is
infinitely more useful than one that's silently wrong.

**Requirements**:
- Modify model forward pass to keep dropout active during inference (`model.train()` on dropout layers only)
- T=30 forward passes per image (~0.5s per image on H100)
- Full test set: ~10 min

**Implementation details**:
```
For each test image:
    1. Run T=30 forward passes with dropout enabled
    2. Collect T probability vectors: p_1, p_2, ..., p_T
    3. Predictive mean: p_bar = (1/T) * sum(p_t)
    4. Predictive entropy: H = -sum(p_bar * log(p_bar))  [total uncertainty]
    5. Expected entropy: E[H] = -(1/T) * sum(sum(p_t * log(p_t)))  [aleatoric]
    6. Mutual information: MI = H - E[H]  [epistemic uncertainty]
    7. Prediction variance: var = (1/T) * sum((p_t - p_bar)^2)
```

**Feasibility**: HIGH — only inference-time change, no retraining.

**Deliverables**:
- Uncertainty scores for every test image
- Uncertainty vs accuracy scatter plot
- Rejection curve: accuracy as function of confidence threshold
- `outputs_v3/uncertainty/mc_dropout_analysis.png`
- `outputs_v3/uncertainty/rejection_curve.png`
- `outputs_v3/uncertainty/epistemic_vs_aleatoric.png`

**Risks**:
- If model has very low dropout rate, MC samples may not vary enough.
  Mitigation: Check variance; if too low, temporarily increase dropout to 0.3.
- 30 passes is standard but can be reduced to 15 if too slow.

---

### 1C. Advanced XAI: Integrated Gradients + Attention Comparison
**What**: Add Integrated Gradients (Sundararajan et al., 2017) as a second XAI method.
IG works by computing the gradient integral from a black baseline to the input image.
Unlike Attention Rollout (which shows WHERE the model looks), IG shows WHICH PIXELS
actually change the output — fundamentally different and complementary.

Then create side-by-side comparison: Original | Attention Rollout | Integrated Gradients
for the same images, showing agreement/disagreement between methods.

**Research value**: VERY HIGH — XAI method comparison is itself a publishable contribution.
If both methods agree on disease regions, explanations are robust. If they disagree, that
flags potential model issues.

**Practical value**: HIGH — multi-method explanations increase clinician trust.

**Requirements**:
- `captum` library (needs install, ~50MB)
- Integrated Gradients needs ~50 interpolation steps per image
- ~2 min for 20 test images on H100

**Implementation details**:
```
1. Baseline: black image (zeros)
2. Interpolate: alpha * input + (1-alpha) * baseline, alpha in [0, 1], 50 steps
3. Compute gradients at each interpolation point
4. Integrated Gradients = (input - baseline) * mean(gradients)
5. Sum across RGB channels -> spatial attribution map
6. Normalize, apply same overlay as attention maps
```

**Feasibility**: HIGH — captum handles the heavy lifting.

**Deliverables**:
- `outputs_v3/xai/comparison_grid.png` (3-column: original | rollout | IG)
- `outputs_v3/xai/agreement_score.json` (spatial correlation between methods)
- `outputs_v3/xai/method_comparison_report.md`

**Risks**:
- IG can be noisy for high-dimensional inputs. Mitigation: SmoothGrad averaging (add
  Gaussian noise, average over 20 IG computations).
- Baseline choice matters. Black image is standard for natural images but fundus images
  have dark backgrounds. Alternative: blurred-image baseline. Will test both.

---

### 1D. Fairness & Domain Robustness Analysis
**What**: Evaluate model performance separately on APTOS vs ODIR images. Check for:
- Performance gap between domains (domain bias)
- Per-class performance by source
- Confidence calibration by source
- Error patterns by source

**Research value**: HIGH — responsible AI is mandatory for medical applications.
Domain shift is a known problem in medical imaging.

**Practical value**: HIGH — if model performs 90% on APTOS but 70% on ODIR, it's
not safe to deploy. Need to know before clinical use.

**Requirements**:
- Test set already has source labels (`dataset` column)
- Pure analysis, no training

**Feasibility**: TRIVIAL.

**Deliverables**:
- `outputs_v3/fairness/performance_by_source.png`
- `outputs_v3/fairness/calibration_by_source.png`
- `outputs_v3/fairness/domain_gap_report.json`

**Risks**: None. But findings may be uncomfortable (expect APTOS DR to perform better
than ODIR due to class imbalance).

---

## Phase 2: Model Improvements (Requires Training)
> Goal: Push performance boundaries and add architectural sophistication

### 2A. K-Fold Cross-Validation (5-Fold Stratified)
**What**: Train 5 independent models on different data splits. Report mean +/- std
for all metrics. This is the gold standard for small datasets.

**Research value**: CRITICAL — single-split results are unreliable with 8.5K images.
Any reviewer will ask for this. A single train/test split could be lucky or unlucky.

**Practical value**: MEDIUM — gives confidence intervals on performance.

**Requirements**:
- 5x training runs, each ~25 epochs
- ~2 hours per fold on H100 (estimated from epoch 24 convergence)
- Total: ~10 hours compute
- Disk: 5 x 331MB = ~1.7GB for checkpoints

**Implementation details**:
```
For k in 1..5:
    1. Split data into 5 stratified folds
    2. Train on 4 folds (80%), validate on 1 fold (20%)
    3. Within training set, carve out 15% for calibration
    4. Apply same training recipe (LLRD, Focal Loss, MixUp, etc.)
    5. Save best checkpoint per fold
    6. Evaluate on held-out fold
```

**Feasibility**: HIGH — same training pipeline, just different splits. Can run overnight.

**Deliverables**:
- 5 model checkpoints
- Mean/std for: accuracy, F1, AUC, per-class metrics
- `outputs_v3/kfold/fold_comparison.png`
- `outputs_v3/kfold/aggregate_metrics.json`

**Risks**:
- 10 hours of GPU time. Mitigation: can run overnight or reduce to 3-fold if constrained.
- Some folds may have very few Glaucoma/Cataract/AMD samples.
  Mitigation: stratified split ensures proportional distribution.

---

### 2B. CNN-ViT Ensemble
**What**: Train EfficientNet-B3 as a second backbone using the SAME training recipe.
Then combine predictions: `p_ensemble = w1 * p_vit + w2 * p_cnn` where weights are
optimized on the calibration set.

Disagreement between CNN and ViT = automatic review flag
(CNNs focus on local textures, ViTs on global structure — complementary).

**Research value**: HIGH — hybrid CNN-Transformer ensembles are state-of-the-art.
Architectural diversity in ensembles is well-studied (Lakshminarayanan et al., 2017).

**Practical value**: VERY HIGH — ensembles are the standard in production medical AI.
Model disagreement is a free uncertainty signal.

**Requirements**:
- Train EfficientNet-B3 (~12M params, much lighter than ViT)
- ~1.5 hours on H100
- timm already has EfficientNet-B3

**Implementation details**:
```
1. Train EfficientNet-B3 with same augmentation/loss/scheduler
2. On calibration set, grid-search ensemble weights:
   w_vit in [0.3, 0.4, 0.5, 0.6, 0.7]
   w_cnn = 1 - w_vit
3. Select weights that maximize calibrated F1
4. Inference: run both models, weighted average of probabilities
5. Disagreement flag: if argmax(p_vit) != argmax(p_cnn) -> REVIEW
```

**Feasibility**: HIGH — timm + existing training pipeline.

**Deliverables**:
- EfficientNet-B3 checkpoint
- Ensemble weight optimization results
- `outputs_v3/ensemble/vit_vs_cnn_comparison.png`
- `outputs_v3/ensemble/ensemble_vs_individual.json`
- Disagreement analysis

**Risks**:
- EfficientNet may underperform ViT on this dataset (small data, ViTs need more data
  but we're fine-tuning from ImageNet). Mitigation: if CNN is much worse, use it only
  as uncertainty signal, not for prediction averaging.
- Two models = 2x inference time. Mitigation: EfficientNet-B3 is fast (~5ms per image).

---

### 2C. Severity Sub-Classification for DR
**What**: APTOS images have DR severity labels (0=No DR, 1=Mild, 2=Moderate,
3=Severe, 4=Proliferative). Currently we collapse all into "DR" binary.
Add a second classification head that predicts severity ONLY when DR is detected.

**Research value**: HIGH — severity grading is clinically critical. The difference
between mild and proliferative DR determines treatment urgency.

**Practical value**: VERY HIGH — "You have DR" is less useful than
"You have Moderate DR — schedule follow-up in 3 months."

**Requirements**:
- 3,662 APTOS images with severity labels (already in dataset)
- Add second head to model: disease head (5 classes) + severity head (5 levels)
- Multi-task learning with shared backbone
- ~2 hours retraining

**Implementation details**:
```
1. Model architecture:
   ViT backbone -> CLS token (768-dim)
   -> Disease head: Linear(768, 5) [all images]
   -> Severity head: Linear(768, 5) [only DR images, masked loss]

2. Loss: L = L_disease + lambda * L_severity (lambda=0.5)
   L_severity computed only on samples where disease_label == 1 (DR)

3. Severity labels: {0: No DR, 1: Mild, 2: Moderate, 3: Severe, 4: Proliferative}
   Note: Only APTOS images have severity; ODIR DR images get severity_label=-1 (ignored)
```

**Feasibility**: HIGH — multi-task learning is standard. Severity data already exists.

**Deliverables**:
- Updated model with dual heads
- Severity confusion matrix
- `outputs_v3/severity/severity_analysis.png`

**Risks**:
- Only APTOS has severity labels. ODIR DR images (1,919) won't contribute to
  severity learning. This is acceptable — severity head simply ignores them.
- Multi-task learning can degrade primary task. Mitigation: if disease accuracy drops
  >1%, reduce lambda or freeze backbone during severity head warmup.

---

## Phase 3: Clinical Integration Layer
> Goal: Make the model useful for real clinicians, not just ML researchers

### 3A. Automated Clinical Report Generation
**What**: For each image, generate a structured clinical report:
- Patient ID / Image metadata
- Primary finding + confidence
- Severity (if DR)
- Attention map with annotated regions
- Uncertainty assessment (MC Dropout)
- OOD flag with explanation
- Recommended clinical action
- Differential diagnoses with probabilities

**Research value**: MEDIUM — template-based generation is not novel, but ties the
system together as a complete clinical tool.

**Practical value**: VERY HIGH — this is what clinicians actually want. Not a number,
but a report they can read, verify, and file.

**Requirements**:
- fpdf or reportlab for PDF generation (needs install)
- All Phase 1 outputs (uncertainty, XAI, evaluation)
- Template design

**Implementation details**:
```
Report structure:
1. HEADER: Patient ID, Date, Image metadata
2. PRIMARY FINDING: "Diabetic Retinopathy detected (confidence: 87%)"
3. SEVERITY: "Moderate (Grade 2)" [if DR]
4. EXPLANATION: Attention map + IG map side by side
5. UNCERTAINTY: "Low epistemic uncertainty — model is confident in this prediction"
6. OOD CHECK: "Image is within training distribution (Mahalanobis: 23.4 < 42.8)"
7. DIFFERENTIAL: "DR: 87%, Normal: 8%, AMD: 3%, Glaucoma: 1%, Cataract: 1%"
8. RECOMMENDATION: "Refer to ophthalmologist within 3 months for DR management"
9. DISCLAIMER: "AI-assisted screening tool — not a replacement for clinical diagnosis"

Clinical action mapping:
- Normal (high conf, low uncertainty) -> "Routine re-screening in 12 months"
- Mild DR -> "Re-screen in 6 months"
- Moderate DR -> "Refer to ophthalmologist within 3 months"
- Severe/Proliferative DR -> "URGENT: refer within 2 weeks"
- Glaucoma -> "Refer for IOP measurement and visual field test"
- Cataract -> "Refer for visual acuity assessment"
- AMD -> "Refer for OCT and anti-VEGF evaluation"
- High uncertainty / OOD -> "Image quality insufficient — re-capture or refer"
```

**Feasibility**: HIGH — template-based, no ML involved in report generation itself.

**Deliverables**:
- `clinical_report_generator.py`
- Sample reports (PDF) for each disease class
- `outputs_v3/reports/sample_report_*.pdf`

**Risks**:
- Clinical recommendations must be medically accurate.
  Mitigation: Use established screening guidelines (UK NSC, AAO PPP).
- Legal disclaimer is mandatory — AI is a screening aid, not a diagnostic tool.

---

### 3B. Gradio Interactive Web Application
**What**: Full-featured web demo with:
- Drag-and-drop image upload
- Real-time prediction with confidence bars
- Attention Rollout + Integrated Gradients side by side
- Uncertainty gauge (MC Dropout)
- OOD warning banner
- Downloadable clinical report (PDF)
- Batch mode: upload multiple images, get summary table

**Research value**: MEDIUM — demos don't get published, but they get noticed.
Conference demos, thesis defenses, job interviews — this is where it shines.

**Practical value**: VERY HIGH — makes the entire project accessible to non-technical users.

**Requirements**:
- gradio (needs install)
- All previous components integrated
- Model loaded once, served for multiple requests

**Implementation details**:
```
Interface layout:
+-------------------------------------------------+
|  RetinaSense-ViT Clinical Screening System      |
+-------------------------------------------------+
| [Upload Image]            | PREDICTION          |
|                           | Diabetic Retinopathy|
| [Fundus image preview]    | Confidence: 87%     |
|                           | Severity: Moderate   |
|                           | Uncertainty: Low     |
|                           +---------------------+
|                           | CLASS PROBABILITIES  |
|                           | [====] DR      87%  |
|                           | [==  ] Normal   8%  |
|                           | [=   ] AMD      3%  |
|                           | [    ] Glaucoma 1%  |
|                           | [    ] Cataract 1%  |
+-------------------------------------------------+
| EXPLAINABILITY                                  |
| [Attention Rollout] [Integrated Gradients]      |
+-------------------------------------------------+
| [Download Clinical Report PDF]                  |
+-------------------------------------------------+
```

**Feasibility**: HIGH — Gradio makes this straightforward.

**Deliverables**:
- `app.py` — Gradio application
- Shareable public link (Gradio provides free hosting)
- Screenshot for README

**Risks**:
- MC Dropout (30 forward passes) may be slow for real-time use.
  Mitigation: Reduce to T=10 for web app (still meaningful uncertainty estimate),
  or run uncertainty async and show "Computing uncertainty..." spinner.
- Model loading takes ~5s. Mitigation: load once at startup, keep in memory.

---

## Phase 4: Deployment & Compression
> Goal: Production-ready, edge-deployable, reproducible

### 4A. Model Compression: Knowledge Distillation + ONNX Export
**What**: Distill ViT-Base (86M params, 331MB) into ViT-Tiny (5.7M params, ~23MB).
Export to ONNX for framework-independent inference. Quantize to INT8.

Size reduction target: 331MB -> ~6MB (55x smaller).
Speed target: ~5ms inference on CPU (vs ~20ms for ViT-Base on GPU).

**Research value**: HIGH — model compression for medical AI is actively published.
The performance-size tradeoff analysis is itself a contribution.

**Practical value**: VERY HIGH — enables deployment on:
- Mobile phones (community health workers in rural areas)
- Raspberry Pi / edge devices (clinic-level screening)
- Browser (WebAssembly + ONNX.js)

**Requirements**:
- onnx, onnxruntime (needs install)
- ViT-Tiny pretrained weights from timm
- Knowledge distillation training: ~2 hours
- ONNX export + quantization: ~10 min

**Implementation details**:
```
Knowledge Distillation:
1. Teacher: ViT-Base (frozen, our best model)
2. Student: ViT-Tiny (trainable, randomly initialized classification head)
3. Loss = alpha * CE(student, labels) + (1-alpha) * KL(student_logits/T, teacher_logits/T)
   alpha=0.3, T=4.0 (temperature for soft targets)
4. Train for 50 epochs with same augmentation pipeline

ONNX Export:
1. Export student to ONNX with dynamic batch size
2. ONNX Simplifier pass
3. INT8 quantization via onnxruntime quantization tools
4. Benchmark: latency, throughput, accuracy vs teacher

Accuracy targets:
- Student should retain >95% of teacher's performance
- Acceptable: Teacher F1=0.813, Student F1 > 0.770
```

**Feasibility**: HIGH — well-established pipeline.

**Deliverables**:
- `outputs_v3/compressed/student_model.pth` (~23MB)
- `outputs_v3/compressed/model.onnx` (~23MB)
- `outputs_v3/compressed/model_int8.onnx` (~6MB)
- `outputs_v3/compressed/compression_benchmark.json`
- Performance comparison: teacher vs student vs quantized

**Risks**:
- Small student may not learn rare classes well (Glaucoma/AMD with only ~265 samples).
  Mitigation: Increase weight on minority-class KD loss.
- ONNX export may fail for custom model layers.
  Mitigation: Use standard timm ViT-Tiny (fully ONNX-compatible).

---

### 4B. FastAPI Inference Server + Docker
**What**: REST API for clinical integration:
- `POST /predict` — single image prediction
- `POST /predict/batch` — multiple images
- `GET /health` — service health check
- `POST /report` — generate clinical report PDF

Containerized in Docker for one-command deployment.

**Research value**: LOW — engineering, not research.

**Practical value**: VERY HIGH — required for any real-world deployment.
Hospital IT systems need API endpoints, not Jupyter notebooks.

**Requirements**:
- fastapi + uvicorn (already installed)
- docker (already installed)
- Dockerfile + docker-compose.yml

**Implementation details**:
```
API Design:
POST /predict
  Input: multipart/form-data (image file)
  Output: {
    "prediction": "Diabetic Retinopathy",
    "confidence": 0.87,
    "severity": "Moderate",
    "probabilities": {"Normal": 0.08, "DR": 0.87, ...},
    "uncertainty": {"epistemic": 0.02, "aleatoric": 0.05, "total": 0.07},
    "ood_score": 23.4,
    "ood_flag": false,
    "recommendation": "Refer to ophthalmologist within 3 months"
  }

Docker:
  - Base: python:3.12-slim + CUDA runtime
  - Model baked into image OR mounted as volume
  - Health checks + graceful shutdown
  - ~2GB image size (mostly PyTorch)
```

**Feasibility**: HIGH — FastAPI + Docker is standard.

**Deliverables**:
- `api/main.py` — FastAPI server
- `Dockerfile` + `docker-compose.yml`
- API documentation (auto-generated by FastAPI)
- `outputs_v3/api/benchmark.json` (requests/sec, latency p50/p95/p99)

**Risks**: Minimal for local deployment.

---

## Implementation Schedule

```
Phase 1 (No retraining — ~3 hours total)
  1A. Evaluation Dashboard .............. 30 min
  1B. MC Dropout Uncertainty ............ 45 min
  1C. Advanced XAI (Captum IG) .......... 45 min
  1D. Fairness Analysis ................. 30 min

Phase 2 (Retraining — ~15 hours compute, can overlap)
  2A. K-Fold CV ......................... 10 hrs (overnight)
  2B. CNN-ViT Ensemble .................. 2 hrs
  2C. Severity Sub-Classification ....... 3 hrs

Phase 3 (Integration — ~4 hours)
  3A. Clinical Report Generator ......... 2 hrs
  3B. Gradio Web App .................... 2 hrs

Phase 4 (Deployment — ~5 hours)
  4A. Knowledge Distillation + ONNX ..... 3 hrs
  4B. FastAPI + Docker .................. 2 hrs
```

## Dependency Graph

```
Phase 1A (Eval Dashboard) ──────┐
Phase 1B (MC Dropout) ──────────┤
Phase 1C (Advanced XAI) ────────┼──> Phase 3A (Clinical Report) ──> Phase 3B (Gradio App)
Phase 1D (Fairness) ────────────┘                                        |
                                                                         v
Phase 2A (K-Fold) ─────────────────────────────────────────────> Final evaluation
Phase 2B (Ensemble) ───────────────────> Phase 3B (add to Gradio)
Phase 2C (Severity) ───────────────────> Phase 3A (add to reports)
                                                    |
                                                    v
                                         Phase 4A (Compress) ──> Phase 4B (Docker/API)
```

## What Was Cut (and Why)

| Originally Proposed | Reason Dropped |
|---------------------|---------------|
| Multi-label classification | No multi-label annotations in current dataset. ODIR raw labels were lost during preprocessing. Would require re-downloading and re-annotating. |
| Patient-level L/R aggregation | Only left-eye images available in ODIR subset. No paired data exists. |
| SHAP explanations | Too slow for medical images (hours per image). Integrated Gradients provides the same pixel-level attribution 1000x faster. |
| Adversarial robustness (FGSM/PGD) | Moved to "nice to have" — interesting for research but not clinically actionable. Can add later if time permits. |

## Package Installation Required

```bash
pip install gradio captum onnx onnxruntime fpdf2
```

---

## Success Criteria

When all phases are complete, RetinaSense-ViT will have:

1. Robust evaluation with confidence intervals (K-Fold)
2. Two independent XAI methods that validate each other
3. Calibrated uncertainty that flags "I don't know" cases
4. Domain fairness analysis showing cross-dataset performance
5. Dual-architecture ensemble with disagreement detection
6. DR severity grading (not just detection)
7. Automated clinical reports with actionable recommendations
8. Interactive web demo anyone can try
9. 55x compressed model for edge deployment
10. Production API with Docker containerization

This transforms RetinaSense from a "trained a ViT on fundus images" project into
a complete clinical AI screening system with safety guarantees, explainability,
and deployment readiness.
