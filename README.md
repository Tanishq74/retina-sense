# 🔬 RetinaSense-ViT: Deep Learning for Retinal Disease Classification

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A production-ready deep learning system for multi-disease retinal classification achieving **89.30% accuracy** and **0.886 macro F1** using Vision Transformers with Domain-Adversarial Neural Network (DANN) training.

## 🎯 Project Overview

RetinaSense-ViT is an AI-powered system for automated detection of five major retinal diseases from fundus images:
- **Normal** (healthy retina)
- **Diabetic Retinopathy (DR)**
- **Glaucoma**
- **Cataract**
- **Age-related Macular Degeneration (AMD)**

### Key Achievements
- **89.30% accuracy** on test set (DANN-v3 model)
- **0.886 macro F1** across all classes
- **0.975 macro AUC-ROC**
- **+40.6% relative improvement** from baseline (63.52% → 89.30%)
- **Retrieval-Augmented Diagnosis (RAD)**: 94.0% combined accuracy (+4.9%) via FAISS kNN retrieval (MAP=0.921)
- **Confidence routing**: Auto-reports 76.9% of cases at 96.8% accuracy, catches 77.2% of errors
- **Domain-adversarial training (DANN)** eliminates APTOS/ODIR domain shift
- **Production-ready** with Gradio web demo, FastAPI server, and Docker deployment
- **IEEE paper ready**: 700-line LaTeX paper with all GPU experiment results

## Performance Metrics

| Model | Accuracy | Macro F1 | AUC | Best Use Case |
|-------|----------|----------|-----|---------------|
| **DANN-v3 ViT** (Production) | **89.30%** | **0.886** | **0.975** | General screening |
| DANN-v2 ViT | 86.1% | 0.871 | 0.962 | Previous production |
| DANN-v1 ViT | 86.1% | 0.867 | 0.962 | Alternative |
| 3-Model Ensemble | 84.8% | 0.840 | -- | Ensemble diversity |
| ViT + Thresholds (pre-DANN) | 84.48% | 0.840 | 0.967 | Research baseline |

### Per-Class Performance (DANN-v3)

| Disease | F1 Score | Notes |
|---------|----------|-------|
| Diabetes/DR | 0.920 | Significant improvement over DANN-v2 (0.890) |
| Glaucoma | 0.833 | |
| Cataract | 0.899 | |
| AMD | 0.923 | |
| Normal | 0.853 | Major improvement over DANN-v2 (0.762) |

### Retrieval-Augmented Diagnosis (RAD) Pipeline

| Metric | Value |
|--------|-------|
| Mean Average Precision (MAP) | 0.921 |
| Recall@1 | 94.0% |
| RAD Combined Accuracy (K=1) | **94.0%** (+4.9% over standalone) |
| Auto-Report Tier | 76.9% of cases at 96.8% accuracy |
| Error Catch Rate | 77.2% |

### Controlled Ablation Study (GPU-trained)

| Model Variant | Accuracy | Macro F1 | AUC |
|---------------|----------|----------|-----|
| Base ViT (no DANN) | 85.28% | 0.843 | 0.944 |
| DANN only | 84.73% | 0.843 | 0.937 |
| DANN + hard mining | 85.89% | 0.849 | 0.947 |
| DANN + mixup | 84.66% | 0.821 | 0.931 |
| **DANN-v3 (full pipeline)** | **89.09%** | **0.879** | **0.972** |

### K-Fold Cross-Validation (5-fold)

| Metric | Value |
|--------|-------|
| Accuracy | 82.4% +/- 1.9% |
| Macro F1 | 0.827 +/- 0.019 |
| Macro AUC | 0.948 +/- 0.008 |

## 🏗️ Architecture

**Vision Transformer (ViT-Base-Patch16-224)** with multi-task learning and domain-adversarial training:
- Pre-trained on ImageNet-21k, 86M parameters
- 768-dimensional feature vectors
- Separate heads for disease classification and severity grading
- **Domain-Adversarial Neural Network (DANN)** with Gradient Reversal Layer for domain-invariant features

### Key Technical Features
- **DANN Domain Adaptation**: Gradient Reversal Layer eliminates APTOS/ODIR domain shift (APTOS accuracy: 26.5% -> 99.8%)
- **Unified CLAHE Preprocessing**: Consistent contrast enhancement across all data sources
- **Focal Loss with DR Alpha Boost**: Handles severe class imbalance (21:1 ratio) with 2.5x weight boost for DR
- **Temperature Scaling**: Post-hoc calibration (T=0.566, ECE=0.034)
- **Threshold Optimization**: Per-class decision thresholds
- **FAISS Similar Case Retrieval**: Find visually similar cases from the training set for clinical context
- **Mixed Precision Training**: Faster training with AMP
- **Gradient Accumulation**: Effective batch size of 64

## Project Structure

```
retinasense/
├── Core Training
│   ├── retinasense_v3.py               # Main ViT training script (1220 lines)
│   ├── train_dann.py                   # DANN domain-adversarial training
│   ├── train_dann_v3.py               # DANN-v3 training (89.30% acc)
│   ├── unified_preprocessing.py        # Unified CLAHE preprocessing pipeline
│   ├── train_ensemble.py               # EfficientNet-B3 + ensemble training
│   ├── kfold_cv.py                     # 5-fold cross-validation
│   ├── app.py                          # Gradio web demo (port 7860)
│   └── api/main.py                     # FastAPI REST server (port 8000)
│
├── RAD Pipeline
│   ├── rebuild_faiss_full.py           # Rebuild FAISS index (all 5 classes)
│   ├── rad_evaluation.py              # Recall@K, MAP, kNN evaluation
│   ├── confidence_routing.py          # 3-tier clinical triage system
│   └── run_paper_experiments.py       # LODO + ablation master script
│
├── Evaluation & XAI
│   ├── eval_dashboard.py               # Full evaluation suite
│   ├── gradcam_v3.py                   # Attention Rollout XAI
│   ├── mc_dropout_uncertainty.py       # MC Dropout uncertainty quantification
│   ├── integrated_gradients_xai.py     # Integrated Gradients XAI
│   └── fairness_analysis.py            # Domain fairness analysis
│
├── paper/
│   └── retinasense_ieee.tex           # IEEE conference paper (700 lines, complete)
│
├── configs/
│   ├── temperature.json                # T=0.566 calibration
│   ├── thresholds.json                 # Per-class decision thresholds
│   └── fundus_norm_stats_unified.json  # Dataset normalization stats
│
├── outputs_v3/
│   ├── dann_v3/best_model.pth          # Production DANN-v3 model
│   ├── retrieval/index_flat_ip.faiss   # FAISS index (8,241 vectors, 5 classes)
│   ├── retrieval/*_results.json        # RAD + routing evaluation results
│   ├── lodo_results.json               # LODO validation results
│   ├── ablation_results.json           # Ablation study results
│   └── evaluation/, xai/, fairness/    # Analysis outputs
│
├── RUN.md                              # Complete run guide
├── SESSION_CONTEXT.md                  # Session history + resume file
└── README.md                           # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/Tanishq74/retina-sense.git
cd retina-sense

# Install dependencies
pip install torch torchvision timm pandas opencv-python scikit-learn matplotlib seaborn tqdm
```

### 2. Download Pre-trained Model

**Note**: Model files are not included in the repository due to size (331MB). Train your own model or contact for pre-trained weights.

### 3. Run Production Inference

```python
# Open RetinaSense_Production.ipynb
jupyter notebook RetinaSense_Production.ipynb

# Or use Python script
from inference import predict_image

prediction = predict_image('path/to/fundus_image.jpg')
print(f"Disease: {prediction['class']}")
print(f"Confidence: {prediction['confidence']:.2%}")
```

## 🎓 Training Your Own Model

### Requirements
- GPU with 8GB+ VRAM
- ~8,500 labeled fundus images
- 2-3 hours training time

### Steps

1. **Prepare Data**: Organize images and create metadata CSV
```csv
image_path,disease_label,severity_label
images/001.jpg,1,-1
images/002.jpg,0,-1
```

2. **Train Model**:
```bash
python retinasense_vit.py
```

3. **Optimize Thresholds** (+2% accuracy boost):
```bash
python threshold_optimization_vit.py
```

4. **Evaluate**:
```python
# See RetinaSense_Production.ipynb for evaluation code
```

## 📈 Research Journey

Our research achieved a **+40.6% relative improvement** through systematic optimization:

```
Phase 0: Original Baseline
├─ 63.52% accuracy
└─ Poor minority class performance

Phase 1: Threshold Optimization (+10 min)
├─ 73.36% accuracy (+9.84%)
└─ Insight: Model poorly calibrated

Phase 2: Extended Training (+15 min)
├─ 74.18% accuracy (+10.66%)
└─ Insight: Needed more epochs

Phase 3: ViT Architecture (+6 min) ⭐
├─ 82.26% accuracy (+18.74%)
└─ Insight: Architecture matters most

Phase 4: ViT + Threshold Opt (+2 min)
├─ 84.48% accuracy (+20.96%)
└─ First production model

Phase 5: DANN-v2 Domain Adaptation
├─ 86.1% accuracy (+22.58%)
└─ Domain-adversarial training eliminates APTOS/ODIR shift

Phase 6: DANN-v3 + Expanded Dataset 🏆
├─ 89.30% accuracy (+25.78%)
└─ Hard-example mining, mixup, cosine annealing, MESSIDOR-2 data

Total Time: ~45 min active research + 4-5 hours training
```

## 🔬 Key Research Insights

1. **Architecture > Everything**: Switching to ViT provided the biggest gain (+18.74%)
2. **Threshold Optimization Works**: Simple per-class thresholds add +2.22%
3. **Focal Loss Essential**: Critical for handling 21:1 class imbalance
4. **Domain Shift Matters**: APTOS images 10x lower quality than ODIR
5. **Ensemble Trade-offs**: Sacrifices 4% accuracy for +10% minority F1

## Dataset

- **Sources**: ODIR-5K + APTOS-2019 + REFUGE2 + MESSIDOR-2
- **Total Images**: 11,524 fundus images (APTOS=3,662 + ODIR=4,878 + REFUGE2=1,200 + MESSIDOR-2=1,744)
- **Resolution**: 224x224 (preprocessed with unified CLAHE)
- **Class Distribution**:
  - Normal: 2,071 (24%)
  - Diabetes/DR: 5,581 (65%)
  - Glaucoma: 308 (4%)
  - Cataract: 315 (4%)
  - AMD: 265 (3%)

**Challenge**: Severe class imbalance (21:1 ratio) and cross-dataset domain shift (10.7x sharpness difference between APTOS and ODIR)

## 🛠️ Technical Stack

| Component | Technology |
|-----------|-----------|
| **Framework** | PyTorch 2.0+ |
| **Architecture** | ViT-Base-Patch16-224 (timm) |
| **Preprocessing** | OpenCV, Ben Graham method |
| **Training** | Mixed Precision (AMP), Focal Loss |
| **Optimization** | AdamW, Cosine Annealing LR |
| **Evaluation** | scikit-learn |
| **Visualization** | matplotlib, seaborn |

## 📖 Documentation

Comprehensive documentation available:

- **[paper/retinasense_ieee.tex](paper/retinasense_ieee.tex)**: IEEE conference paper (700 lines, 13 tables, 30 refs)
- **[RUN.md](RUN.md)**: Complete run guide for training, evaluation, and deployment
- **[SESSION_CONTEXT.md](SESSION_CONTEXT.md)**: Session history and resume file
- **[ARCHITECTURE_DOCUMENT.md](ARCHITECTURE_DOCUMENT.md)**: System architecture document

## 🎯 Use Cases

### Primary Use Case: General Screening
- **Model**: DANN-v3 ViT + Threshold Optimization
- **Accuracy**: 89.30%
- **Macro F1**: 0.886
- **Speed**: ~15ms per image (66 images/sec)
- **Best for**: High-volume clinics, community health programs

### Alternative Use Case: Rare Disease Detection
- **Model**: 3-Model Ensemble (DANN-v1 30% + DANN-v3 50% + EfficientNet 20%)
- **Accuracy**: 84.8%
- **Macro F1**: 0.840
- **Best for**: Academic medical centers, research studies

## 🔒 Clinical Considerations

⚠️ **Important**: This system is intended for research and educational purposes. Not FDA-approved for clinical use. Always consult qualified ophthalmologists for diagnosis.

### Strengths
- High sensitivity for diabetic retinopathy (92.0% F1 with DANN-v3)
- Excellent AMD detection (92.3% F1 with DANN-v3)
- Excellent glaucoma detection (83.3% F1)
- Strong Normal class detection (85.3% F1, up from 76.2%)
- Domain-invariant features via DANN (handles APTOS/ODIR shift)
- Fast inference (15ms per image)
- Well-calibrated predictions (ECE=0.034)

### Limitations
- Trained primarily on Asian populations (ODIR + APTOS datasets)
- LODO validation shows 51.8% on ODIR holdout (most heterogeneous source)
- Requires high-quality fundus images
- No prospective clinical validation yet

## Contributing

Contributions welcome! Areas for improvement:
- External validation on new datasets (ADAM, additional MESSIDOR-2 subsets)
- Support for additional diseases
- Knowledge distillation (ViT-Base -> ViT-Tiny + ONNX)
- Mobile/edge deployment
- Pushing toward 92%+ accuracy with further ensemble and TTA refinements

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **Datasets**: ODIR-5K, APTOS-2019, REFUGE2, MESSIDOR-2
- **Architecture**: Vision Transformer (ViT) by Google Research
- **Domain Adaptation**: DANN (Ganin et al., 2016) with Gradient Reversal Layer
- **Preprocessing**: Ben Graham method + unified CLAHE
- **Framework**: PyTorch, timm library

## 📧 Contact

**Project Maintainer**: Tanishq
- GitHub: [@Tanishq74](https://github.com/Tanishq74)
- Repository: [retina-sense](https://github.com/Tanishq74/retina-sense)

## 📊 Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{retinasense2026,
  title={RetinaSense: An Uncertainty-Aware Domain-Adaptive Vision Transformer
         Framework with Retrieval-Augmented Reasoning for Multi-Disease Retinal Diagnosis},
  author={Tamarkar, Tanishq and Hussain, Rafae Mohammed and Revathi, M},
  year={2026},
  institution={SRM Institute of Science and Technology, Chennai, India},
  url={https://github.com/Tanishq74/retina-sense}
}
```

---

**Last Updated**: March 2026
**Status**: Production Ready
**Best Model**: DANN-v3 -- 89.30% accuracy, 0.886 macro F1, 0.975 AUC
**Production Checkpoint**: `outputs_v3/dann_v3/best_model.pth`
**License**: MIT
