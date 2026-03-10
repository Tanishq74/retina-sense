# RetinaSense-ViT: System Architecture Document

**Version:** 1.0  
**Date:** March 10, 2026  
**Author:** Tanishq  
**Status:** Production Ready  

---

## 1. Introduction

### 1.1 Purpose
This document describes the system architecture of **RetinaSense-ViT**, a deep learning system for automated multi-class retinal disease classification from fundus images. The system detects five retinal conditions — Normal, Diabetic Retinopathy (DR), Glaucoma, Cataract, and Age-related Macular Degeneration (AMD) — achieving **84.48% accuracy** and **0.840 macro F1 score**.

### 1.2 Scope
This architecture covers:
- Data ingestion and preprocessing pipeline
- Model architecture (Vision Transformer and EfficientNet variants)
- Training infrastructure and GPU optimization
- Inference pipeline with threshold optimization
- Evaluation and monitoring subsystems

### 1.3 Intended Audience
ML engineers, software architects, clinical researchers, and deployment teams.

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      RetinaSense-ViT System                        │
│                                                                     │
│  ┌──────────┐   ┌───────────────┐   ┌──────────┐   ┌────────────┐ │
│  │ Data     │──▶│ Preprocessing │──▶│ Model    │──▶│ Inference  │ │
│  │ Ingestion│   │ Pipeline      │   │ Training │   │ Pipeline   │ │
│  └──────────┘   └───────────────┘   └──────────┘   └────────────┘ │
│       │               │                  │               │         │
│       ▼               ▼                  ▼               ▼         │
│  ┌──────────┐   ┌───────────────┐   ┌──────────┐   ┌────────────┐ │
│  │ ODIR-5K  │   │ Ben Graham    │   │ViT-Base  │   │ Threshold  │ │
│  │ APTOS-19 │   │ Enhancement   │   │Patch16   │   │ Optimizer  │ │
│  │ Combined │   │ + Caching     │   │-224      │   │            │ │
│  └──────────┘   └───────────────┘   └──────────┘   └────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Architecture

### 3.1 Data Sources

| Source | Images | Resolution | Classes | Notes |
|--------|--------|-----------|---------|-------|
| **ODIR-5K** | 4,966 | 512×512 | All 5 | Preprocessed fundus images |
| **APTOS-2019** | 3,662 | ~1949×1500 | DR only | Raw fundus, 5-level severity |
| **Combined** | **8,540** | 224×224 (resized) | 5 classes | Single-disease filtered |

### 3.2 Class Distribution

| Class | Samples | Percentage | Imbalance Ratio |
|-------|---------|-----------|-----------------|
| Normal | 2,071 | 24.3% | 1.0x |
| Diabetes/DR | 5,581 | 65.4% | **21.1x** |
| Glaucoma | 308 | 3.6% | 0.1x |
| Cataract | 315 | 3.7% | 0.1x |
| AMD | 265 | 3.1% | 0.1x |

### 3.3 Data Split Strategy
- **Training:** 6,832 samples (80%, stratified)
- **Validation:** 1,708 samples (20%, stratified)
- Stratified split preserves class distribution in both sets

### 3.4 Domain Shift: Critical Architectural Consideration
- **APTOS images** have 10.7× lower sharpness than ODIR (25.5 vs 272.6)
- All APTOS images map exclusively to the DR class
- This creates two distinct visual subpopulations within DR
- The ViT architecture handles this domain gap better than CNNs due to its global attention mechanism

---

## 4. Preprocessing Architecture

### 4.1 Ben Graham Enhancement Pipeline

```
Input Image ──▶ Resize (224×224) ──▶ Gaussian Blur (σ=10)
                                            │
                                            ▼
                                     Weighted Subtraction
                                     4*img - 4*blur + 128
                                            │
                                            ▼
                                     Circular Mask
                                     (r = 0.48 × size)
                                            │
                                            ▼
                                     ImageNet Normalization
                                     μ=[0.485,0.456,0.406]
                                     σ=[0.229,0.224,0.225]
                                            │
                                            ▼
                                     Output Tensor (3×224×224)
```

### 4.2 Pre-Caching Architecture
To eliminate the CPU bottleneck (Ben Graham preprocessing: 100–200ms/image), a caching layer stores preprocessed images as `.npy` files:

```
One-Time Caching Phase:
  Raw Image → Ben Graham Preprocessing → np.save('cache/{id}.npy')
  Cost: ~60 seconds for 8,540 images

Training Phase:
  np.load('cache/{id}.npy') → GPU tensor        (~1ms vs 100–200ms)
```

**Impact:** GPU utilization improved from 5–10% → 60–85%; training speedup ~4×.

### 4.3 Data Augmentation (Training Only)

| Augmentation | Parameters | Purpose |
|-------------|-----------|---------|
| RandomHorizontalFlip | p=0.5 | Geometric invariance |
| RandomVerticalFlip | p=0.3 | Geometric invariance |
| RandomRotation | 20° | Orientation invariance |
| RandomAffine | translate=0.05, scale=0.95–1.05 | Position invariance |
| ColorJitter | brightness=0.3, contrast=0.3 | Lighting robustness |
| RandomErasing | p=0.2 | Occlusion robustness |

---

## 5. Model Architecture

### 5.1 Production Model: Vision Transformer (ViT-Base-Patch16-224)

```
Input Image (3×224×224)
        │
        ▼
┌──────────────────────────────────┐
│   Patch Embedding Layer          │
│   14×14 = 196 patches (16×16)    │
│   + 1 [CLS] token                │
│   + Position Embeddings          │
│   → 197 × 768                    │
└──────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────┐
│   12× Transformer Encoder Blocks │
│   ┌────────────────────────────┐ │
│   │ Multi-Head Self-Attention  │ │
│   │ (12 heads, 768 dim)       │ │
│   ├────────────────────────────┤ │
│   │ Layer Norm + Residual      │ │
│   ├────────────────────────────┤ │
│   │ MLP (768 → 3072 → 768)   │ │
│   ├────────────────────────────┤ │
│   │ Layer Norm + Residual      │ │
│   └────────────────────────────┘ │
└──────────────────────────────────┘
        │
        ▼  [CLS] token output (768-dim)
        │
   ┌────┴────┐
   ▼         ▼
┌────────┐ ┌──────────┐
│Disease │ │ Severity │
│Head    │ │ Head     │
│        │ │          │
│768→512 │ │768→256   │
│BN+ReLU │ │BN+ReLU  │
│Drop 0.3│ │Drop 0.3 │
│512→256 │ │256→5    │
│BN+ReLU │ │(severity)│
│Drop 0.2│ └──────────┘
│256→5   │
│(class) │
└────────┘
```

**Key Specifications:**

| Property | Value |
|----------|-------|
| Architecture | ViT-Base-Patch16-224 (timm) |
| Parameters | ~86M |
| Pre-trained | ImageNet-21k |
| Feature Dimension | 768 |
| Patch Size | 16×16 |
| Sequence Length | 197 (196 patches + 1 CLS) |
| Model File Size | 331 MB |

### 5.2 Backup Model: EfficientNet-B3

| Property | Value |
|----------|-------|
| Architecture | EfficientNet-B3 (timm) |
| Parameters | ~12M |
| Feature Dimension | 1,536 |
| Image Size | 300×300 |
| Model File Size | 47 MB |

### 5.3 Multi-Task Learning Design
Both models share a common backbone with two specialized heads:
1. **Disease Classification Head** → 5-class output (softmax)
2. **Severity Grading Head** → 5-level DR severity (for APTOS-sourced samples)

Loss = `Focal Loss (disease)` + `0.2 × CrossEntropy (severity)`

---

## 6. Training Architecture

### 6.1 Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Epochs | 30 | Best checkpoint at epoch 30 |
| Batch Size | 32 (effective 64) | Gradient accumulation ×2 |
| Optimizer | AdamW | Weight decay regularization |
| Learning Rate | 3×10⁻⁴ | Stable for ViT fine-tuning |
| LR Scheduler | Cosine Annealing (T_max=30, η_min=1×10⁻⁷) | Smooth decay |
| Mixed Precision | AMP (GradScaler) | 2× speed, reduced VRAM |
| Early Stopping | Patience=10 on macro F1 | Prevent overfitting |

### 6.2 Loss Function: Focal Loss

```
FL(p_t) = −α_t × (1 − p_t)^γ × log(p_t)

Parameters:
  γ = 1.0   (focusing parameter)
  α = class_weights  (inverse class frequency)
```

Focal Loss down-weights easy (well-classified) examples, forcing the model to focus on hard minority samples — critical for the 21:1 class imbalance.

### 6.3 GPU Optimization Architecture

```
Original Pipeline:                  Optimized Pipeline:
┌────────┐ ┌──────────┐ ┌──────┐   ┌─────────┐ ┌──────┐
│Disk I/O│→│Ben Graham│→│ GPU  │   │Cache I/O│→│ GPU  │
│ 10ms   │ │ 100-200ms│ │ 20ms │   │  1ms    │ │ 25ms │
└────────┘ └──────────┘ └──────┘   └─────────┘ └──────┘

GPU Util: 6%                        GPU Util: 96%
Speed: ~1 it/s                      Speed: ~4-5 it/s
```

**Optimizations applied:**
- Pre-cached preprocessing (100× faster data loading)
- Batch size: 32 → 128 (4× larger)
- DataLoader workers: 2 → 8 (4× parallel loading)
- Persistent workers, prefetch_factor=2
- Non-blocking GPU transfers
- `optimizer.zero_grad(set_to_none=True)`

---

## 7. Inference Architecture

### 7.1 Single-Image Inference Pipeline

```
Input Image ──▶ Ben Graham Preprocess ──▶ ImageNet Normalize
                                                │
                                                ▼
                                         ViT Forward Pass
                                         (disease_logits, severity_logits)
                                                │
                                                ▼
                                            Softmax
                                                │
                                                ▼
                                     ┌──────────────────────┐
                                     │ Threshold-Based      │
                                     │ Decision Logic        │
                                     │                      │
                                     │ Per-Class Thresholds: │
                                     │  Normal:    0.540    │
                                     │  DR:        0.240    │
                                     │  Glaucoma:  0.810    │
                                     │  Cataract:  0.930    │
                                     │  AMD:       0.850    │
                                     └──────────────────────┘
                                                │
                                                ▼
                                     Prediction + Confidence Score
```

### 7.2 Threshold Optimization Method
Per-class thresholds are optimized via grid search (0.05 to 0.95, step 0.05) on the validation set, converting each class to a one-vs-rest binary problem and maximizing F1 score per class.

**Two threshold strategies available:**

| Strategy | Accuracy | Macro F1 | Use Case |
|----------|----------|----------|----------|
| Accuracy-focused (Default) | **84.48%** | 0.840 | General screening |
| F1-focused | 80.44% | **0.858** | Rare disease detection |

### 7.3 Inference Performance

| Config | Latency | Throughput | GPU Memory |
|--------|---------|-----------|------------|
| ViT Solo | ~15ms | ~66 img/s | ~2 GB |
| ViT + TTA (8×) | ~120ms | ~8 img/s | ~2 GB |
| Ensemble (3 models) | ~45ms | ~22 img/s | ~4 GB |

### 7.4 Optional: Hybrid Inference Architecture

```
Image ──▶ ViT First-Pass (fast, 15ms)
              │
              ├─ Confidence ≥ 0.75 AND majority class ──▶ Return prediction
              │
              └─ Confidence < 0.75 OR rare class ──▶ Ensemble Second-Pass ──▶ Return
```

---

## 8. Ensemble Architecture (Optional)

| Model | Weight | Architecture | Size |
|-------|--------|-------------|------|
| ViT-Base-Patch16-224 | 0.85 | Vision Transformer | 331 MB |
| EfficientNet-B3 Extended | 0.10 | CNN (50 epochs) | 47 MB |
| EfficientNet-B3 v2 | 0.05 | CNN (20 epochs) | 47 MB |

**Ensemble Strategy:** Weighted probability averaging  
`final_prob = 0.85×ViT + 0.10×EffNetExt + 0.05×EffNetv2`

---

## 9. Technology Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| Framework | PyTorch | 2.0+ |
| Model Library | timm | 0.9+ |
| Vision Utils | torchvision | 0.18+ |
| Image Processing | OpenCV | 4.8+ |
| Data Handling | pandas | 2.0+ |
| ML Metrics | scikit-learn | 1.3+ |
| Visualization | matplotlib, seaborn | Latest |
| GPU | NVIDIA H200 | 150 GB VRAM |
| Training | CUDA + AMP (Mixed Precision) | — |

---

## 10. File and Directory Structure

```
retina-sense/
├── Notebooks
│   ├── RetinaSense_Production.ipynb        # Production inference ⭐
│   ├── RetinaSense_ViT_Training.ipynb      # ViT training process
│   └── RetinaSense_Optimized.ipynb         # GPU optimization experiments
│
├── Training Scripts
│   ├── retinasense_vit.py                  # ViT training (84.48%)
│   ├── retinasense_v2_extended.py          # Extended CNN (50 epochs)
│   ├── retinasense_v2.py                   # Baseline CNN (20 epochs)
│   └── retinasense_fixed.py                # Bug-fixed original
│
├── Optimization Scripts
│   ├── threshold_optimization_vit.py       # ViT threshold tuning
│   ├── threshold_optimization_simple.py    # v2 threshold tuning
│   ├── ensemble_inference.py               # Model ensemble
│   ├── tta_evaluation.py                   # Test-time augmentation
│   └── data_analysis.py                    # Dataset analysis
│
├── Model Outputs
│   ├── outputs_vit/                        # ViT checkpoints + results
│   ├── outputs_v2/                         # v2 baseline outputs
│   ├── outputs_v2_extended/                # Extended training outputs
│   ├── outputs_optimized/                  # GPU optimization outputs
│   ├── outputs_ensemble/                   # Ensemble results
│   └── outputs_analysis/                   # Data analysis outputs
│
└── Data
    ├── data/combined_dataset.csv           # Unified metadata
    └── final_unified_metadata.csv          # Full metadata file
```

---

## 11. Deployment Architecture

### 11.1 Production Deployment Specification

```yaml
Model:
  Architecture: ViT-Base-Patch16-224
  Checkpoint: outputs_vit/best_model.pth
  Size: 331 MB
  Parameters: ~86M

Input:
  Image Size: 224×224 pixels
  Format: RGB fundus image
  Preprocessing: Ben Graham + ImageNet normalization

Output:
  Class: [Normal, DR, Glaucoma, Cataract, AMD]
  Confidence: Float [0.0, 1.0]
  All Probabilities: Array of 5 floats
  Flag for Review: If confidence < threshold

Hardware Requirements:
  GPU: NVIDIA (CUDA required), 2+ GB VRAM
  Inference Speed: ~66 images/sec
```

### 11.2 Monitoring Requirements
- Track prediction class distribution for data drift
- Monitor confidence score calibration over time
- Log flagged (low-confidence) cases for expert review
- Alert on out-of-distribution inputs
- Track inference latency and throughput

---

## 12. Limitations and Constraints

1. **Population Bias:** Trained primarily on Asian populations (ODIR dataset)
2. **Equipment Sensitivity:** May not generalize across different fundus cameras
3. **Image Quality Dependence:** Requires high-quality fundus images
4. **Single-Label:** Does not handle co-morbidities (multi-label not supported)
5. **Domain Shift:** APTOS/ODIR quality gap (10× sharpness difference) is partially addressed by ViT but remains a concern
6. **Not FDA/CE Approved:** Research/educational use only

---

*Document Version: 1.0 | Last Updated: March 10, 2026*
