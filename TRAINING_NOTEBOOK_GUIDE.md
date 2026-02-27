# RetinaSense-ViT Training Notebook Guide

## ✅ Path Issues Fixed!

The training notebook has been updated to correctly handle image paths.

### What Was Fixed

1. **Path Cleaning**: Removed leading `./` or `.//` from image paths
   - Original: `.//odir/preprocessed_images/0_left.jpg`
   - Fixed: `odir/preprocessed_images/0_left.jpg`

2. **Dataset Setup**: Created `./data/combined_dataset.csv` with 8,540 images

3. **Simplified Preprocessing**: Removed caching complexity, now does on-the-fly preprocessing

### Dataset Details

- **Location**: `./data/combined_dataset.csv`
- **Total Images**: 8,540
- **Classes**: 5 (Normal, Diabetes/DR, Glaucoma, Cataract, AMD)
- **Class Distribution**:
  - Normal: 2,071 (24%)
  - Diabetes/DR: 5,581 (65%)
  - Glaucoma: 308 (4%)
  - Cataract: 315 (4%)
  - AMD: 265 (3%)

## 🚀 How to Use the Training Notebook

### Option 1: View Training Process (Recommended)

**Purpose**: Understand how the model was trained

Simply open `RetinaSense_ViT_Training.ipynb` and read through the cells to see:
- Data preprocessing steps
- Model architecture
- Training loop with metrics
- Visualization of results

**You don't need to re-run training** - the model is already trained and achieves 84.48% accuracy!

### Option 2: Re-train from Scratch

**Purpose**: Experiment with different hyperparameters or train on new data

**Requirements**:
- GPU with at least 8GB VRAM
- ~2-3 hours training time
- Dataset in `./data/combined_dataset.csv`

**Steps**:
1. Open `RetinaSense_ViT_Training.ipynb`
2. Run all cells sequentially
3. Model will be saved to `./outputs_vit_training/best_model.pth`
4. Training history and plots saved automatically

**Expected Results**:
- Raw accuracy: ~82%
- With threshold optimization: ~84%
- Training time: 2-3 hours on H200 GPU

### Option 3: Use Pre-trained Model (Best Option!)

**Purpose**: Make predictions on new images immediately

Use `RetinaSense_Production.ipynb` instead:
- Loads pre-trained model from `outputs_vit/best_model.pth`
- Includes inference function for new images
- Applies optimal thresholds automatically
- Ready for deployment

## 📊 Training Notebook Structure

### Section 1: Setup & Configuration
- Install dependencies
- Set random seeds
- Configure hyperparameters
- Detect GPU

### Section 2: Data Preprocessing
- Ben Graham preprocessing function
- Dataset verification
- Path fixing

### Section 3: Dataset Class
- Custom PyTorch Dataset
- On-the-fly preprocessing
- Data augmentation (train only)

### Section 4: Model Architecture
- Vision Transformer (ViT-Base-Patch16-224)
- Multi-task heads (disease + severity)
- 86M parameters

### Section 5: Training Setup
- Focal Loss for class imbalance
- AdamW optimizer
- Cosine learning rate schedule
- Mixed precision training (AMP)

### Section 6: Training Loop
- 30 epochs (early stopping after 10 with no improvement)
- Real-time progress bars
- Per-epoch metrics
- Automatic checkpointing

### Section 7: Visualization
- Training curves (loss, accuracy, F1)
- Learning rate schedule
- Confusion matrix
- Classification report

### Section 8: Final Evaluation
- Load best model
- Detailed metrics
- Export results

## 🎯 Key Hyperparameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| **Batch Size** | 32 (effective 64) | Balance speed/memory |
| **Learning Rate** | 1e-4 | Stable for fine-tuning |
| **Epochs** | 30 | With early stopping |
| **Focal Loss γ** | 1.0 | Handle class imbalance |
| **Dropout** | 0.4 | Prevent overfitting |
| **Image Size** | 224×224 | ViT standard size |

## 📁 Output Files

Training creates the following in `./outputs_vit_training/`:

- `best_model.pth` - Best checkpoint (based on macro F1)
- `checkpoint_epoch_N.pth` - Checkpoints every 5 epochs
- `training_curves.png` - Visualization of metrics
- `confusion_matrix.png` - Final confusion matrix
- `training_history.csv` - Complete metrics log
- `training_summary.json` - Summary statistics

## 🔍 Troubleshooting

### "FileNotFoundError: combined_dataset.csv"
**Solution**: Dataset is already set up at `./data/combined_dataset.csv`

### "Image not found" errors
**Solution**: Path fixing is already implemented in the notebook

### Low GPU utilization
**Solution**: Increase `num_workers` in DataLoader (currently 4)

### Out of memory
**Solution**: Reduce batch size from 32 to 16

## 💡 Tips

1. **Monitor GPU usage**: Use `nvidia-smi` in terminal
2. **Adjust batch size**: Based on available GPU memory
3. **Use early stopping**: Saves time and prevents overfitting
4. **Track metrics**: Check validation F1, not just accuracy
5. **Save checkpoints**: Every 5 epochs in case of interruption

## 🎉 Expected Results

After successful training:

```
Best Model Performance:
  Accuracy: ~82%
  Macro F1: ~0.82

Per-Class F1 Scores:
  Normal:       0.73
  Diabetes/DR:  0.87
  Glaucoma:     0.84
  Cataract:     0.86
  AMD:          0.80
```

After threshold optimization (+2%):
```
  Accuracy: 84.48%
  Macro F1: 0.840
```

## 📚 Next Steps

1. **Apply threshold optimization**: Run `threshold_optimization_vit.py`
2. **Use production notebook**: `RetinaSense_Production.ipynb`
3. **Deploy model**: Export to TorchScript or ONNX
4. **External validation**: Test on new datasets

---

**Note**: You already have a trained model at `outputs_vit/best_model.pth` achieving 84.48% accuracy. Unless you want to experiment with different hyperparameters, you can skip re-training and use the production notebook directly!
