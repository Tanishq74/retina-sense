# 🔬 RetinaSense-ViT: Deep Learning for Retinal Disease Classification

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A production-ready deep learning system for multi-disease retinal classification achieving **84.48% accuracy** using Vision Transformers.

## 🎯 Project Overview

RetinaSense-ViT is an AI-powered system for automated detection of five major retinal diseases from fundus images:
- **Normal** (healthy retina)
- **Diabetic Retinopathy (DR)**
- **Glaucoma**
- **Cataract**
- **Age-related Macular Degeneration (AMD)**

### Key Achievements
- ✅ **84.48% accuracy** on validation set
- ✅ **0.840 macro F1** across all classes
- ✅ **+32% relative improvement** from baseline (63.52% → 84.48%)
- ✅ **Production-ready** with optimized inference pipeline

## 📊 Performance Metrics

| Model | Accuracy | Macro F1 | Best Use Case |
|-------|----------|----------|---------------|
| **ViT + Thresholds** (Recommended) | **84.48%** | **0.840** | General screening |
| ViT Raw | 82.26% | 0.821 | Research baseline |
| Ensemble | 80.44% | 0.858 | Maximum rare disease detection |

### Per-Class Performance

| Disease | F1 Score | Precision | Recall |
|---------|----------|-----------|--------|
| Normal | 0.746 | 0.707 | 0.789 |
| Diabetes/DR | 0.891 | 0.918 | 0.865 |
| Glaucoma | 0.871 | 0.900 | 0.844 |
| Cataract | 0.874 | 0.906 | 0.844 |
| AMD | 0.819 | 0.891 | 0.759 |

## 🏗️ Architecture

**Vision Transformer (ViT-Base-Patch16-224)** with multi-task learning:
- Pre-trained on ImageNet
- 86M parameters
- 768-dimensional feature vectors
- Separate heads for disease classification and severity grading

### Key Technical Features
- **Ben Graham Preprocessing**: Specialized fundus image preprocessing
- **Focal Loss**: Handles severe class imbalance (21:1 ratio)
- **Threshold Optimization**: Per-class decision thresholds
- **Mixed Precision Training**: Faster training with AMP
- **Gradient Accumulation**: Effective batch size of 64

## 📁 Project Structure

```
retinasense/
├── 📓 Notebooks
│   ├── RetinaSense_Production.ipynb       # Production inference (⭐ START HERE)
│   ├── RetinaSense_ViT_Training.ipynb     # Complete training process
│   └── RetinaSense_Optimized.ipynb        # Optimization experiments
│
├── 🐍 Training Scripts
│   ├── retinasense_vit.py                 # ViT training (84.48% accuracy)
│   ├── retinasense_v2_extended.py         # Extended CNN training
│   └── retinasense_fixed.py               # Original fixed version
│
├── 🔧 Optimization Scripts
│   ├── threshold_optimization_vit.py      # Per-class thresholds (+2% boost)
│   ├── ensemble_inference.py              # Model ensemble evaluation
│   ├── tta_evaluation.py                  # Test-time augmentation
│   └── data_analysis.py                   # Dataset analysis
│
├── 📊 Research Documentation
│   ├── PRODUCTION_MODEL_DECISION.md       # Final model selection
│   ├── COMPLETE_RESEARCH_REPORT.md        # Full research journey
│   ├── TRAINING_NOTEBOOK_GUIDE.md         # Training guide
│   └── FINAL_RESULTS_COMPARISON.md        # Performance comparison
│
└── 📚 Additional Docs
    ├── README.md                           # This file
    └── .gitignore                          # Git ignore rules
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

Our research achieved a **+32% relative improvement** through systematic optimization:

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
├─ 84.48% accuracy (+20.96%) 🏆
└─ PRODUCTION READY

Total Time: ~45 min active research + 2-3 hours training
```

## 🔬 Key Research Insights

1. **Architecture > Everything**: Switching to ViT provided the biggest gain (+18.74%)
2. **Threshold Optimization Works**: Simple per-class thresholds add +2.22%
3. **Focal Loss Essential**: Critical for handling 21:1 class imbalance
4. **Domain Shift Matters**: APTOS images 10x lower quality than ODIR
5. **Ensemble Trade-offs**: Sacrifices 4% accuracy for +10% minority F1

## 📊 Dataset

- **Sources**: ODIR-5K + APTOS-2019
- **Total Images**: 8,540 fundus images
- **Resolution**: 224×224 (preprocessed)
- **Class Distribution**:
  - Normal: 2,071 (24%)
  - Diabetes/DR: 5,581 (65%)
  - Glaucoma: 308 (4%)
  - Cataract: 315 (4%)
  - AMD: 265 (3%)

**Challenge**: Severe class imbalance (21:1 ratio)

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

- **[PRODUCTION_MODEL_DECISION.md](PRODUCTION_MODEL_DECISION.md)**: Final model selection rationale
- **[TRAINING_NOTEBOOK_GUIDE.md](TRAINING_NOTEBOOK_GUIDE.md)**: Complete training guide
- **[COMPLETE_RESEARCH_REPORT.md](COMPLETE_RESEARCH_REPORT.md)**: Full research journey (35+ pages)
- **[FINAL_RESULTS_COMPARISON.md](FINAL_RESULTS_COMPARISON.md)**: Model comparison

## 🎯 Use Cases

### Primary Use Case: General Screening
- **Model**: ViT + Threshold Optimization
- **Accuracy**: 84.48%
- **Speed**: ~15ms per image (66 images/sec)
- **Best for**: High-volume clinics, community health programs

### Alternative Use Case: Rare Disease Detection
- **Model**: Ensemble + ViT Thresholds
- **Accuracy**: 80.44%
- **Macro F1**: 0.858 (best minorities)
- **Best for**: Academic medical centers, research studies

## 🔒 Clinical Considerations

⚠️ **Important**: This system is intended for research and educational purposes. Not FDA-approved for clinical use. Always consult qualified ophthalmologists for diagnosis.

### Strengths
- ✅ High sensitivity for diabetic retinopathy (89% F1)
- ✅ Excellent glaucoma detection (87% F1)
- ✅ Fast inference (15ms per image)
- ✅ Handles class imbalance well

### Limitations
- ⚠️ Lower performance on rare diseases (AMD: 82% F1)
- ⚠️ Trained primarily on Asian populations (ODIR dataset)
- ⚠️ May not generalize to different imaging equipment
- ⚠️ Requires high-quality fundus images

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- External validation on new datasets
- Support for additional diseases
- Deployment optimization (TensorRT, ONNX)
- Mobile/edge deployment
- Explainability (Grad-CAM, attention maps)

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **Datasets**: ODIR-5K, APTOS-2019
- **Architecture**: Vision Transformer (ViT) by Google Research
- **Preprocessing**: Ben Graham method from Kaggle competitions
- **Framework**: PyTorch, timm library

## 📧 Contact

**Project Maintainer**: Tanishq
- GitHub: [@Tanishq74](https://github.com/Tanishq74)
- Repository: [retina-sense](https://github.com/Tanishq74/retina-sense)

## 📊 Citation

If you use this work in your research, please cite:

```bibtex
@software{retinasense2026,
  title={RetinaSense-ViT: Deep Learning for Retinal Disease Classification},
  author={Tanishq},
  year={2026},
  url={https://github.com/Tanishq74/retina-sense}
}
```

---

**Last Updated**: February 2026
**Status**: ✅ Production Ready
**Performance**: 84.48% accuracy, 0.840 macro F1
**License**: MIT
