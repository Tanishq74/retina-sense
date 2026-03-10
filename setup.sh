#!/usr/bin/env bash
# RetinaSense-ViT — Local Setup Script
# Run once after cloning the repo.
# Usage: bash setup.sh
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

echo "=== RetinaSense-ViT Setup ==="
echo "Repo: $REPO_DIR"

# ── 1. Python dependencies ─────────────────────────────────────────────────────
echo ""
echo "[1/4] Installing Python dependencies..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install timm gradio captum onnx onnxruntime fastapi uvicorn python-multipart \
            Pillow numpy opencv-python fpdf2

# ── 2. Directory structure ─────────────────────────────────────────────────────
echo ""
echo "[2/4] Creating required directories..."
mkdir -p outputs_v3
mkdir -p data

# ── 3. Model weights ───────────────────────────────────────────────────────────
echo ""
echo "[3/4] Checking model weights..."
MODEL_PATH="$REPO_DIR/outputs_v3/best_model.pth"

if [ -f "$MODEL_PATH" ]; then
    echo "  best_model.pth already present. Skipping download."
else
    echo "  Downloading best_model.pth from Hugging Face Hub..."
    pip install -q huggingface_hub
    python - <<'PYEOF'
from huggingface_hub import hf_hub_download
import shutil, os

path = hf_hub_download(
    repo_id="tanishq74/retinasense-vit",
    filename="best_model.pth",
    repo_type="model",
)
os.makedirs("outputs_v3", exist_ok=True)
shutil.copy(path, "outputs_v3/best_model.pth")
print("  Downloaded: outputs_v3/best_model.pth")
PYEOF
fi

# Download EfficientNet-B3 ensemble weights
EFFNET_PATH="$REPO_DIR/outputs_v3/ensemble/efficientnet_b3.pth"
if [ -f "$EFFNET_PATH" ]; then
    echo "  efficientnet_b3.pth already present. Skipping download."
else
    echo "  Downloading efficientnet_b3.pth from Hugging Face Hub..."
    python - <<'PYEOF'
from huggingface_hub import hf_hub_download
import shutil, os

path = hf_hub_download(
    repo_id="tanishq74/retinasense-vit",
    filename="efficientnet_b3.pth",
    repo_type="model",
)
os.makedirs("outputs_v3/ensemble", exist_ok=True)
shutil.copy(path, "outputs_v3/ensemble/efficientnet_b3.pth")
print("  Downloaded: outputs_v3/ensemble/efficientnet_b3.pth")
PYEOF
fi

# ── 4. Verify config files ─────────────────────────────────────────────────────
echo ""
echo "[4/4] Verifying config files..."
for f in configs/temperature.json configs/thresholds.json configs/fundus_norm_stats.json; do
    if [ -f "$f" ]; then
        echo "  OK  $f"
    else
        echo "  MISSING  $f  (this should be in git — re-clone the repo)"
    fi
done

# ── Done ───────────────────────────────────────────────────────────────────────
echo ""
echo "=== Setup complete ==="
echo ""
echo "Once best_model.pth is in place, start the apps:"
echo ""
echo "  Gradio demo:  python app.py"
echo "                → http://localhost:7860"
echo ""
echo "  FastAPI:      python -m uvicorn api.main:app --host 0.0.0.0 --port 8000"
echo "                → http://localhost:8000/docs"
echo ""
echo "  Docker:       docker build -t retinasense . && docker run -p 8000:8000 retinasense"
