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
    echo ""
    echo "  ╔══════════════════════════════════════════════════════════════╗"
    echo "  ║  MODEL WEIGHTS NOT FOUND (outputs_v3/best_model.pth)        ║"
    echo "  ║                                                              ║"
    echo "  ║  The trained model (~331MB) is not stored in git.           ║"
    echo "  ║  You have two options:                                       ║"
    echo "  ║                                                              ║"
    echo "  ║  Option A — Download from remote server (if you have SSH):  ║"
    echo "  ║    scp user@server:/teamspace/studios/this_studio/           ║"
    echo "  ║              outputs_v3/best_model.pth outputs_v3/          ║"
    echo "  ║                                                              ║"
    echo "  ║  Option B — Upload via Hugging Face Hub (recommended):      ║"
    echo "  ║    On the GPU server, run:                                   ║"
    echo "  ║      pip install huggingface_hub                             ║"
    echo "  ║      python scripts/upload_model_to_hf.py                   ║"
    echo "  ║    Then here, run:                                           ║"
    echo "  ║      python scripts/download_model_from_hf.py               ║"
    echo "  ║                                                              ║"
    echo "  ║  Option C — Place the file manually:                        ║"
    echo "  ║    Copy best_model.pth into outputs_v3/best_model.pth       ║"
    echo "  ╚══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "  Continuing setup. The app will fail to start until the model is placed."
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
