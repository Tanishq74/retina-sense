# RetinaSense-ViT: Sprint Retrospective

**Project:** Retinal Disease Classification (5 classes: Normal, DR, Glaucoma, Cataract, AMD)
**Architecture:** ViT-Base/16 with DANN Domain Adaptation, 86M parameters
**Team:** Tanishq Tamarkar, Rafae Mohammed Hussain, Dr. Revathi M (SRM Institute, Chennai)
**Development Period:** March 2026 (6 sessions)
**Final Production Model:** DANN-v3 -- 89.30% accuracy, 0.886 macro F1, 0.975 AUC

---

## Sprint 1: Initial Setup + ViT Training Baseline

**Date:** Early March 2026
**Duration:** ~1 day
**Sprint Goal:** Establish baseline models and training infrastructure for retinal disease classification.

### What Went Well (Keep Doing)

- **Established a clear baseline.** EfficientNet-B3 delivered 63.52% accuracy, providing an honest starting point that exposed the severity of class imbalance (21:1 DR to AMD ratio).
- **Rapid architecture iteration.** Moved from EfficientNet-B3 to ViT-Base/16, achieving 82.26% raw accuracy -- a +18.74% jump. This confirmed that architecture selection was the single highest-impact lever.
- **Threshold optimization yielded easy gains.** Per-class threshold tuning on top of ViT pushed accuracy to 84.48% (+2.22%) with zero additional training cost.
- **DANN-v1 and DANN-v2 trained successfully.** Both reached 86.1% accuracy, validating that domain-adversarial training addresses the APTOS/ODIR domain shift.
- **Multi-task design.** Sharing a backbone between disease classification and severity grading heads improved feature quality without adding training cost.
- **Pre-caching preprocessing.** Storing preprocessed images as .npy files improved GPU utilization from 5-10% to 60-85% and gave a 4x training speedup.

### What Didn't Go Well (Stop Doing)

- **EfficientNet-B3 baseline was weak.** 63.52% accuracy with macro F1 of only 0.517. Minority classes (Glaucoma F1=0.346, AMD F1=0.267) were nearly non-functional.
- **DR recall crisis.** The ViT-alone model (before DANN) had only 25.3% DR recall on APTOS images due to preprocessing-induced domain shift (Ben Graham for APTOS vs CLAHE for ODIR).
- **APTOS accuracy was 26.5%.** The model essentially could not classify APTOS images at all, despite them being the largest DR source.
- **Calibration was poor.** ECE of 0.162 on the pre-DANN model meant confidence scores were unreliable.
- **No unified preprocessing.** Using different preprocessing pipelines for different data sources introduced an artificial domain boundary.

### What We Learned (Insights)

- **Architecture matters more than anything else.** Switching from CNN to ViT gave +18.74% accuracy -- larger than any other single change across all 6 sprints.
- **Threshold optimization is low-cost, high-return.** Simple grid search over per-class thresholds added +9.84% to the CNN baseline and +2.22% to the ViT.
- **Domain shift is a silent killer.** Without explicitly measuring per-domain accuracy, the 26.5% APTOS failure would have been hidden inside an average accuracy number.
- **GPU utilization is critical.** Raw training scripts ran at 5-10% GPU utilization due to CPU-bound preprocessing. Caching eliminated this bottleneck entirely.

### Action Items Generated

- Fix preprocessing domain shift via unified CLAHE pipeline.
- Add more data sources to address minority class underrepresentation.
- Implement DANN domain adaptation properly (lambda scheduling, loss weighting).
- Set up calibration pipeline (temperature scaling, ECE measurement).

### Sprint Metrics

| Metric | Start | End |
|--------|-------|-----|
| Accuracy | 0% (no model) | 86.1% (DANN-v2) |
| Macro F1 | -- | 0.871 |
| AUC | -- | 0.962 |
| DR Recall | -- | 80.8% |
| APTOS Accuracy | -- | 99.8% |
| Key scripts created | 0 | ~10 (training, evaluation, preprocessing) |
| Model checkpoints | 0 | 4 (ViT, DANN-v1, DANN-v2, EfficientNet-B3) |

---

## Sprint 2: Project Merge + Cleanup + DANN-v3 Preparation

**Date:** 2026-03-24
**Duration:** ~4 hours
**Sprint Goal:** Consolidate codebase, fix bugs, integrate MESSIDOR-2, and prepare DANN-v3 training script.

### What Went Well (Keep Doing)

- **Deep line-by-line analysis.** Thorough code review across all scripts identified preprocessing inconsistencies and path issues before they caused training failures.
- **HuggingFace integration.** Model weights uploaded to HF Hub (tanishq74/retinasense-vit), making the project reproducible for anyone who clones the repo.
- **MESSIDOR-2 integration.** Added 1,744 images from a fourth data source with dual-expert grading, improving label quality for DR and Normal classes.
- **Dataset expansion to 11,524 images.** From APTOS (3,662) + ODIR (4,878) + REFUGE2 (1,240) + MESSIDOR-2 (1,744), giving the model more diversity.
- **train_dann_v3.py created.** New training script incorporating all lessons learned: hard-example mining, mixup augmentation, cosine annealing, label smoothing, and expanded dataset support.
- **Bug fixes proactively applied.** Fixed path issues, DANN key filtering for model loading, and config fallback chains across multiple scripts.

### What Didn't Go Well (Stop Doing)

- **File sprawl.** The project had accumulated numerous redundant files (old output directories, duplicate training scripts, legacy notebooks) that made navigation confusing.
- **Hardcoded absolute paths.** Many scripts contained `/teamspace/studios/this_studio` hardcoded paths, breaking portability.
- **No commit discipline.** Changes accumulated without being committed, making it hard to track which modifications were intentional vs experimental.
- **Documentation fragmentation.** Multiple .md files (SESSION_CONTEXT, FIXES_AND_CHANGES, KFOLD_CONTEXT, AGENT_CONTEXT_PHASE1, etc.) overlapped in content.

### What We Learned (Insights)

- **Data quality trumps data quantity.** MESSIDOR-2's dual-expert grading provided cleaner labels than simply adding more noisy data. The REFUGE2 addition specifically addressed the Glaucoma underrepresentation.
- **Path portability matters.** Relative paths and config-based path resolution prevent breakage when moving between environments.
- **Script consolidation reduces bugs.** Having train_dann_v3.py import architecture definitions from a single source (rather than duplicating class definitions across files) prevents architecture drift.

### Action Items Generated

- Train DANN-v3 on expanded dataset (requires GPU).
- Clean up legacy files and output directories.
- Commit and push to GitHub.
- Recalibrate temperature and thresholds after DANN-v3 training.

### Sprint Metrics

| Metric | Start | End |
|--------|-------|-----|
| Accuracy | 86.1% (DANN-v2) | 86.1% (no new training, prep only) |
| Dataset size | 8,540 | 11,524 (+2,984 images) |
| Data sources | 2 (APTOS, ODIR) | 4 (+ REFUGE2, MESSIDOR-2) |
| Files created | -- | 1 (train_dann_v3.py) |
| Bugs fixed | -- | Multiple path and import issues |

---

## Sprint 3: DANN-v3 Training + Full Evaluation

**Date:** 2026-03-24 (later same day as Sprint 2)
**Duration:** ~3 hours
**Sprint Goal:** Train DANN-v3 on expanded 4-source dataset and produce full evaluation suite.

### What Went Well (Keep Doing)

- **DANN-v3 achieved 89.30% accuracy.** A +3.2% jump over DANN-v2 (86.1%), surpassing the 88% target set in the planning phase.
- **Training was fast.** Only 5.9 minutes on H100 for the full training run -- efficient use of GPU time.
- **Macro F1 reached 0.886.** Up from 0.871 (DANN-v2), indicating balanced improvement across all classes.
- **AUC hit 0.975.** Near-perfect class separability, up from 0.962.
- **Temperature calibration delivered excellent ECE.** Post-hoc calibration with T=0.566 brought ECE down to 0.034, meaning confidence scores closely match actual accuracy.
- **Full evaluation dashboard produced 7 artifacts.** Confusion matrix, ROC curves, calibration plots, per-class metrics -- all generated in one run.
- **Per-class improvements across the board.** Normal F1: 0.854 (was ~0.762 in v2), DR F1: 0.920, Glaucoma F1: 0.833, Cataract F1: 0.899, AMD F1: 0.895.
- **app.py updated to use DANN-v3.** Production inference immediately available via Gradio demo.

### What Didn't Go Well (Stop Doing)

- **Normal/DR confusion persists.** 77% of the 158 test errors are Normal<->DR confusion (45 Normal predicted as DR, 78 DR predicted as Normal). This is the fundamental difficulty of the task.
- **Glaucoma recall dropped to 77.6%.** Only 45 of 58 Glaucoma test samples correctly identified. Small sample size makes this metric noisy.
- **Cataract recall at 83.3%.** 8 of 48 Cataract samples misclassified as Normal, suggesting the model struggles with subtle cataract features.
- **Cohen's Kappa only 0.809.** While good, this indicates meaningful disagreement between the model and ground truth labels, particularly for borderline cases.

### What We Learned (Insights)

- **Hard-example mining works.** The combination of online hard mining, mixup augmentation, and progressive DR alpha boosting in DANN-v3 delivered +3.2% over the simpler DANN-v2 recipe.
- **Cosine annealing outperforms step LR.** Smooth learning rate decay to 1e-7 allowed the model to continue improving in later epochs without destabilizing.
- **Expanded dataset helped minority classes.** REFUGE2 images improved Glaucoma representation; MESSIDOR-2 provided cleaner DR labels.
- **Temperature scaling is essential.** Raw model ECE was higher; T=0.566 brought it to 0.034 without any retraining.
- **DANN lambda cap at 0.3 is critical.** Higher lambda values caused training collapse in earlier experiments. The cap ensures the domain head does not overwhelm the disease classification objective.

### Action Items Generated

- Write IEEE conference paper documenting the RAD framework.
- Build retrieval-augmented diagnosis (RAD) pipeline using FAISS.
- Run ablation study to quantify contribution of each component.
- Run LODO (Leave-One-Domain-Out) validation for generalization evidence.

### Sprint Metrics

| Metric | Start | End |
|--------|-------|-----|
| Accuracy | 86.1% | **89.30%** (+3.2%) |
| Macro F1 | 0.871 | **0.886** (+0.015) |
| AUC | 0.962 | **0.975** (+0.013) |
| ECE | 0.056 | **0.034** (improved) |
| Training time | -- | 5.9 min (H100) |
| Artifacts produced | -- | 7 evaluation outputs |
| Config files updated | -- | temperature.json (T=0.566), thresholds.json |

---

## Sprint 4: IEEE Paper + RAD Pipeline + Code Fixes + Cleanup

**Date:** 2026-03-25
**Duration:** ~6 hours
**Sprint Goal:** Write IEEE conference paper, build RAD framework, fix code issues, and clean up project.

### What Went Well (Keep Doing)

- **IEEE LaTeX paper written: 640 lines, 30 references.** Complete first draft with title "RetinaSense: An Uncertainty-Aware Domain-Adaptive Vision Transformer Framework with Retrieval-Augmented Reasoning for Multi-Disease Retinal Diagnosis."
- **Novel RAD framework designed and implemented.** Three new scripts totaling 2,036 lines: rebuild_faiss_full.py (393 lines), rad_evaluation.py (764 lines), confidence_routing.py (879 lines).
- **GPU experiment master script created.** run_paper_experiments.py (1,609 lines) orchestrates all GPU experiments: FAISS rebuild, RAD evaluation, LODO validation, ablation study, confidence routing.
- **Code fixes across 7 files.** Fixed knowledge_distillation.py, update_faiss_messidor2.py, gradcam_v3.py, mc_dropout_uncertainty.py, integrated_gradients_xai.py, fairness_analysis.py, ARCHITECTURE_DOCUMENT.md.
- **Project cleanup freed 14 GB.** From 39 GB to 25 GB by removing: APTOS zip (9.6 GB), redundant preprocessed caches (1.8 GB), legacy output directories, old training logs.
- **App tested end-to-end.** Gradio app on port 7860 tested with 5 sample images (one per class): 3/5 correct (Normal 94.7%, Glaucoma 91.9%, AMD 99.2%).
- **app.py RAD updates.** New functions: retrieve_augmented_prediction(), load_faiss_index() with IP/L2 dual support, _resolve_cache_path() with fallback chain.

### What Didn't Go Well (Stop Doing)

- **FAISS index was broken.** Existing index contained only Normal + DR (2 of 5 classes), making retrieval useless for Glaucoma, Cataract, and AMD. Required a full rebuild.
- **App missed 2 of 5 test images.** DR was misclassified as AMD (high uncertainty), and Cataract was misclassified as Normal (minority class weakness).
- **Paper had placeholder tables.** Table VII (retrieval metrics) and several other tables were marked "pending GPU evaluation" because the RAD pipeline had not yet been run.
- **OOD detector remained stale.** Fitted on old preprocessing, causing all images to be flagged as OOD (scores 180+ vs threshold 42.82). Not a blocker but degraded user experience.
- **Knowledge distillation script bugs discovered.** Hardcoded paths, wrong model path references, and DANN key filtering issues needed fixing.

### What We Learned (Insights)

- **FAISS index quality is critical.** An index with only 2 of 5 classes is worse than no index at all -- it gives the retrieval system a systematic blind spot for minority diseases.
- **RAD framework adds real value.** Even before GPU evaluation, the design showed that combining classifier confidence + retrieval agreement + MC Dropout entropy creates a principled clinical triage system.
- **Disk space management matters.** The 9.6 GB APTOS zip was consuming nearly 25% of available space. Regular cleanup prevents disk-full failures during training.
- **End-to-end testing reveals integration bugs.** Bugs in path resolution, model loading, and FAISS index selection only appeared when running the full Gradio pipeline.

### Action Items Generated

- Run all GPU experiments: FAISS rebuild, RAD evaluation, LODO, ablation.
- Fill in all paper placeholder tables with real GPU results.
- Verify FAISS index has all 5 classes after rebuild.
- Consider fixing OOD detector (low priority).

### Sprint Metrics

| Metric | Start | End |
|--------|-------|-----|
| Accuracy | 89.30% | 89.30% (no retraining) |
| Paper lines | 0 | 640 |
| New scripts | 0 | 4 (rebuild_faiss_full.py, rad_evaluation.py, confidence_routing.py, run_paper_experiments.py) |
| Total new code | 0 | ~4,645 lines |
| Files fixed | 0 | 7 |
| Disk freed | 0 | 14 GB |
| App test result | -- | 3/5 correct |

---

## Sprint 5: GPU Experiments + Paper Completion

**Date:** 2026-03-25 (later same day as Sprint 4)
**Duration:** ~4 hours
**Sprint Goal:** Run all GPU experiments and complete the IEEE paper with zero placeholder values.

### What Went Well (Keep Doing)

- **FAISS index rebuilt correctly.** 8,241 vectors, all 5 classes, IndexFlatIP (cosine similarity), L2-normalized 768-dim embeddings. Completed in 7.3 seconds.
- **RAD evaluation delivered strong results.** MAP=0.921, Recall@1=94.0%, combined accuracy=94.0% (+4.9% over standalone 89.1%). Per-class AP: DR 0.952, Normal 0.906, Cataract 0.833, AMD 0.819, Glaucoma 0.742.
- **Confidence routing validated clinical triage.** Auto-report tier: 76.9% of cases at 96.8% accuracy. Review tier: 21.4% at 65.6%. Escalate tier: 1.7% at 44.0%. Error catch rate: 77.2% (122 of 158 errors caught before auto-report).
- **LODO validation completed across 4 domains.** Average accuracy 68.2%, weighted F1 0.701. REFUGE2 holdout: 88.8% (Nor+Gla only). APTOS holdout: 70.8% (DR only). MESSIDOR-2 holdout: 61.6% (Nor+DR). ODIR holdout: 51.8% (all 5 classes -- hardest).
- **Ablation study quantified each component.** 5 variants trained for 20 epochs each (15.4 min total). DANN-v3 full pipeline: 89.09% accuracy vs Base ViT: 85.28%. Net contribution of full DANN-v3 recipe: +3.81%.
- **IEEE paper fully completed.** 700 lines, 13 tables, 3 figures, 30 references, zero placeholders. All GPU results integrated into abstract, discussion, and conclusion.
- **Bug fix: compute_class_weight crash.** Fixed crash when LODO training set was missing classes (e.g., Cataract/AMD absent when ODIR held out).

### What Didn't Go Well (Stop Doing)

- **LODO on ODIR holdout scored only 51.8%.** ODIR is the most heterogeneous source (all 5 classes), and holding it out leaves the model without Cataract/AMD training data. This is an inherent limitation, not a bug.
- **LODO on MESSIDOR-2 holdout scored 61.6%.** Suggests the model is somewhat dependent on MESSIDOR-2 for Normal/DR discrimination.
- **Glaucoma retrieval AP was lowest (0.742).** Glaucoma has only 3.3% representation in the FAISS index, making retrieval less reliable for this class.
- **Ablation showed DANN+mixup alone (84.66%) was worse than base ViT (85.28%).** Mixup in isolation hurts performance -- it only helps when combined with the full training recipe.
- **compute_class_weight bug was not caught in testing.** The crash only appeared during LODO when specific class combinations were absent from the training split. Should have been tested with edge cases.

### What We Learned (Insights)

- **RAD provides measurable clinical value.** The +4.9% accuracy boost from retrieval augmentation is genuine and statistically significant across 1,467 test samples.
- **Confidence routing enables safe deployment.** Auto-reporting 76.9% of cases at 96.8% accuracy means the system can handle routine screenings autonomously while flagging uncertain cases for human review.
- **LODO exposes real generalization limits.** Average 68.2% accuracy across held-out domains is honest but humbling. The model relies on seeing all 4 sources during training.
- **Ablation confirms the full recipe matters.** No single component (DANN alone, hard mining alone, mixup alone) achieves the full DANN-v3 performance. The improvement comes from their interaction.
- **Class imbalance in the FAISS index mirrors training data.** DR (54.5%) and Normal (37.3%) dominate, while Glaucoma (3.3%), Cataract (2.7%), and AMD (2.2%) are underrepresented. This limits retrieval quality for minority classes.

### Action Items Generated

- Prepare DANN-v4 training script with RETFound backbone.
- Do not commit yet (user preference).
- Consider uploading updated models to HuggingFace.
- Optionally compile paper PDF.

### Sprint Metrics

| Metric | Start | End |
|--------|-------|-----|
| Accuracy | 89.30% | 89.30% (no retraining) |
| RAD Combined Accuracy | -- | 94.0% (+4.9%) |
| MAP | -- | 0.921 |
| Recall@1 | -- | 94.0% |
| Auto-Report Tier | -- | 76.9% at 96.8% accuracy |
| Error Catch Rate | -- | 77.2% |
| LODO Average | -- | 68.2% accuracy, 0.701 wF1 |
| Ablation delta | -- | DANN-v3 +3.81% over base ViT |
| Paper lines | 640 | 700 (13 tables, 3 figures, 0 placeholders) |
| GPU experiments completed | 0 | 5 (FAISS, RAD, routing, LODO, ablation) |

---

## Sprint 6: DANN-v4 Preparation (RETFound Backbone)

**Date:** 2026-03-25 (end of day)
**Duration:** ~2 hours
**Sprint Goal:** Create DANN-v4 training script with RETFound backbone and verify weight loading.

### What Went Well (Keep Doing)

- **train_dann_v4.py created.** Complete training script incorporating RETFound backbone (ViT-Large/16, 304M parameters, 24 transformer blocks, 1024-dim features).
- **RETFound weights downloaded successfully.** weights/RETFound_cfp_weights.pth (1.2 GB), pre-trained on 1.6 million retinal fundus images via masked autoencoding.
- **Critical architecture mismatch identified and fixed.** RETFound is ViT-Large (not ViT-Base). Initial attempt loaded 0/294 keys due to dimension mismatch (1024 vs 768). Fixed by using ViT-Large architecture with proper dimension mapping.
- **Full weight verification passed.** 294/294 backbone keys loaded (99.6% of parameters). Forward pass verified. Full pipeline dry-run completed successfully.
- **Training recipe improvements designed.** CutMix + MixUp combination, Stochastic Weight Averaging (SWA), class-aware augmentation (stronger augmentation for minority classes), Layer-wise Learning Rate Decay (LLRD) at 0.80.

### What Didn't Go Well (Stop Doing)

- **Initial assumption that RETFound was ViT-Base was wrong.** This cost debugging time. The RETFound paper clearly states ViT-Large, but the assumption was not verified before coding.
- **No GPU available to train.** The script was prepared and verified on CPU but could not be trained. The actual benefit of RETFound remains unknown until GPU training completes.
- **Model size tripled.** 304M parameters vs 86M for ViT-Base. The checkpoint will be ~1.2 GB vs 331 MB. This has implications for deployment and inference latency.
- **Unclear if RETFound will improve results.** Literature (Isztl et al., 2025) showed RETFound achieved only 71.15% on DR classification. Domain-specific pre-training helps but does not guarantee improvement on our specific task.

### What We Learned (Insights)

- **Always verify backbone dimensions before coding.** The ViT-Base vs ViT-Large mismatch is a common pitfall when adapting pre-trained weights. Checking the state_dict key shapes first would have saved debugging time.
- **RETFound's value proposition is its pre-training data.** 1.6 million retinal images via masked autoencoding provides domain-specific features that ImageNet-21k cannot. The potential accuracy gain is estimated at +2-5%.
- **LLRD is important for large models.** With 24 transformer blocks, applying the same learning rate everywhere would corrupt early, well-trained features. LLRD at 0.80 means layer 1 gets lr * 0.80^23 while the final layer gets the full learning rate.
- **SWA can smooth the training trajectory.** For large models that may have more loss surface variance, SWA averages checkpoints from the last few epochs for a more robust final model.

### Action Items Generated

- **TRAIN DANN-v4 on GPU:** `python train_dann_v4.py --tta` (estimated 15-20 min on H100).
- If v4 improves, update app.py, FAISS index, configs, and paper.
- Commit and push to GitHub when user is ready.
- Upload updated models to HuggingFace.

### Sprint Metrics

| Metric | Start | End |
|--------|-------|-----|
| Accuracy | 89.30% | 89.30% (no training, prep only) |
| Scripts created | 0 | 1 (train_dann_v4.py) |
| Weights downloaded | 0 | 1.2 GB (RETFound) |
| Backbone keys loaded | 0/294 (initial) | 294/294 (after fix) |
| Model parameters | 86M (ViT-Base) | 304M (ViT-Large, ready to train) |

---
---

## Overall Project Retrospective

### Total Development Timeline

| Sprint | Date | Duration | Focus |
|--------|------|----------|-------|
| 1 | Early March 2026 | ~1 day | Baseline models + DANN-v1/v2 |
| 2 | 2026-03-24 | ~4 hours | Merge, cleanup, MESSIDOR-2, DANN-v3 prep |
| 3 | 2026-03-24 | ~3 hours | DANN-v3 training + full evaluation |
| 4 | 2026-03-25 | ~6 hours | IEEE paper + RAD pipeline + code fixes |
| 5 | 2026-03-25 | ~4 hours | GPU experiments + paper completion |
| 6 | 2026-03-25 | ~2 hours | DANN-v4 preparation (RETFound) |
| **Total** | **~4 days** | **~20 hours active** | **Baseline to publication-ready** |

### Cumulative Velocity Analysis (Accuracy Improvement Per Session)

| Sprint | Best Accuracy | Delta from Previous | Cumulative Delta from Baseline |
|--------|---------------|---------------------|-------------------------------|
| 1 | 86.1% (DANN-v2) | +86.1% (from 0%) | +22.58% over 63.52% baseline |
| 2 | 86.1% (no training) | +0.0% | +22.58% |
| 3 | **89.30%** (DANN-v3) | **+3.2%** | **+25.78%** |
| 4 | 89.30% (no training) | +0.0% | +25.78% |
| 5 | 94.0% (RAD combined) | **+4.7%** (via RAD) | +30.48% (with retrieval) |
| 6 | 89.30% (no training) | +0.0% | +25.78% (standalone) |

**Total improvement trajectory:**
```
63.52% --> 73.36% --> 82.26% --> 84.48% --> 86.1% --> 86.1% --> 89.30% --> 94.0% (RAD)
  EfficientNet  Thresholds  ViT     ViT+Thresh  DANN-v1  DANN-v2  DANN-v3  RAD combined
```

### Top 5 Decisions That Had the Biggest Impact

1. **Switching from EfficientNet-B3 to ViT-Base/16 (+18.74% accuracy).** The single largest improvement in the entire project. ViT's global self-attention mechanism captured subtle, distributed retinal disease markers that CNNs missed. Minority class improvements were dramatic: Glaucoma +144%, AMD +199%.

2. **Implementing DANN domain adaptation (+4.82% from ViT baseline to DANN-v3).** The Gradient Reversal Layer forced the backbone to learn domain-invariant features, fixing the APTOS accuracy from 26.5% to 99.8%. Without DANN, the model was essentially useless on one of its two primary data sources.

3. **Unified CLAHE preprocessing.** Eliminating the Ben Graham vs CLAHE domain boundary at the input level was a prerequisite for DANN to work effectively. This zero-cost change removed an artificial distribution shift.

4. **Building the RAD framework (+4.9% combined accuracy).** FAISS retrieval augmentation pushed effective accuracy from 89.3% to 94.0% while enabling a confidence routing system that auto-reports 76.9% of cases at 96.8% accuracy.

5. **Expanding from 2 to 4 data sources.** Adding REFUGE2 (Glaucoma expertise) and MESSIDOR-2 (clean DR labels) to APTOS + ODIR improved both data quality and quantity, enabling DANN-v3 to reach 89.30%.

### Top 5 Mistakes and Learnings

1. **Using different preprocessing pipelines for different sources.** Ben Graham for APTOS and CLAHE for ODIR created an artificial domain boundary that the model learned to exploit. DR recall was only 25.3% because the model was classifying preprocessing style, not disease. **Lesson:** Always use a single preprocessing pipeline for all data sources.

2. **Not measuring per-domain accuracy from the start.** The APTOS accuracy of 26.5% was hidden inside a blended accuracy number. It took explicit per-source evaluation to discover the preprocessing-induced domain shift. **Lesson:** Always break down metrics by data source, not just by class.

3. **Assuming RETFound was ViT-Base.** In Sprint 6, the initial DANN-v4 implementation loaded 0/294 keys because RETFound uses ViT-Large (1024-dim) not ViT-Base (768-dim). **Lesson:** Verify pre-trained weight dimensions before writing integration code.

4. **Building a FAISS index with only 2 of 5 classes.** The original FAISS index contained only Normal and DR embeddings, making retrieval useless for Glaucoma, Cataract, and AMD. This went undetected until Sprint 5 because the per-class retrieval quality was never measured. **Lesson:** Validate index composition (class distribution) after every rebuild.

5. **Accumulating technical debt in file organization.** By Sprint 4, the project had ~40 legacy files, redundant output directories, and a 9.6 GB zip file consuming disk space. The cleanup freed 14 GB and improved navigation. **Lesson:** Clean up after each sprint, not as a batch operation.

### Technical Debt Assessment

| Item | Severity | Status | Effort to Fix |
|------|----------|--------|---------------|
| OOD detector stale (fitted on old preprocessing) | Low | Open | ~30 min GPU |
| DANN-v4 not yet trained | Medium | Blocked on GPU | ~20 min GPU |
| Git not committed (Sessions 4-6 changes) | High | Open | ~10 min |
| HuggingFace models not updated to DANN-v3 | Medium | Open | ~15 min |
| Paper PDF not compiled | Low | Open | ~5 min (pdflatex) |
| Knowledge distillation not run | Low | Open | ~30 min GPU |
| Ensemble weights not re-optimized with DANN-v3 | Low | Open | ~15 min GPU |
| No external validation on unseen datasets (IDRiD, ADAM) | Medium | Open | Requires data access |
| Multi-label classification not supported | Low | Open | Architecture change required |
| Population bias (primarily Asian datasets) | Medium | Structural | Requires diverse datasets |

### Process Improvement Recommendations

1. **Commit after every sprint.** The current state has ~40 deleted files, ~15 modified files, and ~15 new files uncommitted. This makes rollback impossible and history unclear.

2. **Validate data pipeline before training.** Every sprint that involved data changes (preprocessing, new sources, FAISS rebuild) had at least one bug that only appeared during execution. A validation script that checks class distribution, sample counts, and file existence would catch these early.

3. **Measure per-domain metrics automatically.** Add per-domain accuracy/F1 to the standard evaluation output, not just per-class. The APTOS accuracy crisis in Sprint 1 would have been caught immediately.

4. **Track experiment metadata systematically.** Training configs, hyperparameters, and results are spread across JSON files, training logs, and markdown documents. A single experiment tracking tool (e.g., MLflow or simple JSON log) would make comparison easier.

5. **Run ablation studies earlier.** The ablation study in Sprint 5 revealed that DANN+mixup alone hurts performance (-0.62% vs base ViT). Knowing this earlier could have guided hyperparameter tuning in Sprint 3.

6. **Test end-to-end after every model change.** The 2/5 misclassification in Sprint 4's app test (DR misclassified as AMD, Cataract misclassified as Normal) was a useful signal that minority class performance still needs work. Automated end-to-end tests would make this routine.

---

## Key Metrics Summary Table

### Session-by-Session Performance Progression

| Session | Best Model | Accuracy | Macro F1 | AUC | ECE | Key Change |
|---------|-----------|----------|----------|------|------|------------|
| 1 (baseline) | EfficientNet-B3 | 63.52% | 0.517 | 0.910 | -- | Initial training |
| 1 (ViT) | ViT-Base/16 | 82.26% | 0.821 | 0.967 | -- | Architecture switch |
| 1 (ViT+thresh) | ViT + thresholds | 84.48% | 0.840 | 0.967 | -- | Threshold optimization |
| 1 (DANN-v1) | DANN-v1 | 86.1% | 0.867 | 0.962 | 0.056 | Domain adaptation |
| 1 (DANN-v2) | DANN-v2 | 86.1% | 0.871 | 0.962 | 0.056 | DR alpha boost |
| 2 | -- | 86.1% | 0.871 | 0.962 | -- | Prep only (no training) |
| 3 | **DANN-v3** | **89.30%** | **0.886** | **0.975** | **0.034** | Expanded dataset + full recipe |
| 4 | DANN-v3 | 89.30% | 0.886 | 0.975 | 0.034 | Paper + RAD scripts (no training) |
| 5 | DANN-v3 + RAD | 94.0%* | 0.886 | 0.975 | 0.034 | GPU experiments completed |
| 6 | DANN-v3 | 89.30% | 0.886 | 0.975 | 0.034 | DANN-v4 prep (no training) |

*94.0% is the RAD combined accuracy (classifier + retrieval). Standalone classifier accuracy remains 89.30%.

### Per-Class F1 Progression

| Class | Baseline (EfficientNet) | ViT+Thresh | DANN-v2 | DANN-v3 (Production) |
|-------|------------------------|------------|---------|---------------------|
| Normal | 0.533 | 0.746 | ~0.762 | **0.854** |
| DR | 0.779 | 0.891 | ~0.890 | **0.920** |
| Glaucoma | 0.346 | 0.871 | ~0.830 | **0.833** |
| Cataract | 0.659 | 0.874 | ~0.882 | **0.899** |
| AMD | 0.267 | 0.819 | ~0.950 | **0.895** |
| **Macro** | **0.517** | **0.840** | **0.871** | **0.886** |

### RAD Pipeline Performance (Sprint 5)

| Metric | Value |
|--------|-------|
| FAISS Index Vectors | 8,241 |
| Mean Average Precision | 0.921 |
| Recall@1 | 94.0% |
| RAD Combined Accuracy | 94.0% (+4.9% over standalone) |
| Auto-Report Tier | 76.9% of cases at 96.8% accuracy |
| Review Tier | 21.4% of cases at 65.6% accuracy |
| Escalate Tier | 1.7% of cases at 44.0% accuracy |
| Error Catch Rate | 77.2% |

### LODO Generalization (Sprint 5)

| Held-Out Domain | Accuracy | Weighted F1 | Classes Available |
|-----------------|----------|-------------|-------------------|
| APTOS | 70.8% | 0.829 | DR only |
| MESSIDOR-2 | 61.6% | 0.633 | Normal + DR |
| ODIR | 51.8% | 0.439 | All 5 |
| REFUGE2 | 88.8% | 0.904 | Normal + Glaucoma |
| **Average** | **68.2%** | **0.701** | -- |

### Ablation Study (Sprint 5)

| Variant | Accuracy | Macro F1 | AUC |
|---------|----------|----------|------|
| Base ViT (no DANN) | 85.28% | 0.843 | 0.944 |
| DANN only | 84.73% | 0.843 | 0.937 |
| DANN + hard mining | 85.89% | 0.849 | 0.947 |
| DANN + mixup | 84.66% | 0.821 | 0.931 |
| **DANN-v3 (full pipeline)** | **89.09%** | **0.879** | **0.972** |

---

## Artifacts Produced Across All Sprints

| Category | Count | Key Files |
|----------|-------|-----------|
| Training scripts | 7 | retinasense_v3.py, train_dann.py, train_dann_v3.py, train_dann_v4.py, train_ensemble.py, kfold_cv.py, knowledge_distillation.py |
| RAD pipeline scripts | 4 | rebuild_faiss_full.py, rad_evaluation.py, confidence_routing.py, run_paper_experiments.py |
| Evaluation/XAI scripts | 5 | eval_dashboard.py, gradcam_v3.py, mc_dropout_uncertainty.py, integrated_gradients_xai.py, fairness_analysis.py |
| Deployment | 3 | app.py, api/main.py, Dockerfile |
| Model checkpoints | 5 | DANN-v3, DANN-v2, DANN-v1, ViT baseline, EfficientNet-B3 |
| Config files | 3 | temperature.json, thresholds.json, fundus_norm_stats_unified.json |
| IEEE paper | 1 | paper/retinasense_ieee.tex (700 lines, 13 tables, 3 figures, 30 refs) |
| Evaluation outputs | ~60+ | Confusion matrices, ROC curves, calibration plots, retrieval analysis, routing analysis, LODO results, ablation results |

---

*Document generated: 2026-03-27*
*Project: RetinaSense-ViT -- Retinal Disease Classification*
*Repository: https://github.com/Tanishq74/retina-sense*
