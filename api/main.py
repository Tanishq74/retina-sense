#!/usr/bin/env python3
"""
RetinaSense-ViT — FastAPI Inference Server (Phase 4B)
======================================================
REST API for clinical retinal screening integration.

Endpoints:
  POST /predict        — single image prediction
  POST /predict/batch  — multiple images
  GET  /health         — service health check
"""

import os, sys, json, io, time, tempfile, warnings
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
warnings.filterwarnings('ignore')

# ================================================================
# CONFIG
# ================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 224
NUM_CLASSES = 5
CLASS_NAMES = ['Normal', 'Diabetes/DR', 'Glaucoma', 'Cataract', 'AMD']
SEVERITY_NAMES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
BASE_DIR = '/teamspace/studios/this_studio'

# Load configs
with open(os.path.join(BASE_DIR, 'data/fundus_norm_stats.json')) as f:
    ns = json.load(f)
NORM_MEAN, NORM_STD = ns['mean_rgb'], ns['std_rgb']
with open(os.path.join(BASE_DIR, 'outputs_v3/temperature.json')) as f:
    T_OPT = json.load(f)['temperature']
with open(os.path.join(BASE_DIR, 'outputs_v3/thresholds.json')) as f:
    THRESHOLDS = json.load(f)['thresholds']


# ================================================================
# MODEL
# ================================================================
class MultiTaskViT(nn.Module):
    def __init__(self, n_disease=5, n_severity=5, drop=0.3):
        super().__init__()
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
        self.drop = nn.Dropout(drop)
        self.disease_head = nn.Sequential(
            nn.Linear(768, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, n_disease),
        )
        self.severity_head = nn.Sequential(
            nn.Linear(768, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, n_severity),
        )

    def forward(self, x):
        f = self.backbone(x)
        f = self.drop(f)
        return self.disease_head(f), self.severity_head(f)


model = MultiTaskViT().to(DEVICE)
ckpt = torch.load(os.path.join(BASE_DIR, 'outputs_v3/best_model.pth'),
                   map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

normalize = transforms.Normalize(NORM_MEAN, NORM_STD)
val_transform = transforms.Compose([
    transforms.ToPILImage(), transforms.ToTensor(), normalize,
])


# ================================================================
# HELPERS
# ================================================================
def preprocess(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img_np = cv2.resize(np.array(img), (IMG_SIZE, IMG_SIZE))
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    processed = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return val_transform(processed).unsqueeze(0)


RECOMMENDATIONS = {
    0: 'Routine re-screening in 12 months.',
    1: {0: 'Annual diabetic eye screening.', 1: 'Re-screen in 6-9 months.',
        2: 'Refer to ophthalmologist within 3 months.',
        3: 'URGENT: Retina specialist within 2-4 weeks.',
        4: 'URGENT: Immediate anti-VEGF/PRP referral.'},
    2: 'Refer for IOP measurement and visual field test.',
    3: 'Refer for visual acuity assessment.',
    4: 'Refer for OCT and anti-VEGF evaluation.',
}


def get_recommendation(pred, sev=0):
    r = RECOMMENDATIONS.get(pred, '')
    return r.get(sev, r.get(0, '')) if isinstance(r, dict) else r


@torch.no_grad()
def run_inference(tensor):
    tensor = tensor.to(DEVICE)
    d_out, s_out = model(tensor)
    probs = torch.softmax(d_out / T_OPT, dim=1).cpu().numpy()[0]
    sev_probs = torch.softmax(s_out, dim=1).cpu().numpy()[0]
    pred = int(probs.argmax())
    conf = float(probs[pred])
    sev = int(sev_probs.argmax()) if pred == 1 else -1
    return pred, conf, probs, sev


# ================================================================
# FASTAPI APP
# ================================================================
app = FastAPI(
    title="RetinaSense-ViT API",
    description="AI-powered retinal disease screening API",
    version="3.0.0",
)


class PredictionResult(BaseModel):
    prediction: str
    class_index: int
    confidence: float
    severity: Optional[str] = None
    probabilities: dict
    recommendation: str
    inference_time_ms: float


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": "ViT-Base/16 (RetinaSense v3.0)",
        "device": str(DEVICE),
        "classes": CLASS_NAMES,
        "checkpoint_epoch": ckpt['epoch'] + 1,
    }


@app.post("/predict", response_model=PredictionResult)
async def predict(image: UploadFile = File(...)):
    if not image.content_type or not image.content_type.startswith('image/'):
        raise HTTPException(400, "File must be an image")

    img_bytes = await image.read()
    t0 = time.time()
    tensor = preprocess(img_bytes)
    pred, conf, probs, sev = run_inference(tensor)
    elapsed = (time.time() - t0) * 1000

    return PredictionResult(
        prediction=CLASS_NAMES[pred],
        class_index=pred,
        confidence=conf,
        severity=SEVERITY_NAMES[sev] if sev >= 0 else None,
        probabilities={cn: float(probs[i]) for i, cn in enumerate(CLASS_NAMES)},
        recommendation=get_recommendation(pred, max(sev, 0)),
        inference_time_ms=round(elapsed, 1),
    )


@app.post("/predict/batch")
async def predict_batch(images: List[UploadFile] = File(...)):
    results = []
    for image in images:
        img_bytes = await image.read()
        t0 = time.time()
        tensor = preprocess(img_bytes)
        pred, conf, probs, sev = run_inference(tensor)
        elapsed = (time.time() - t0) * 1000
        results.append({
            "filename": image.filename,
            "prediction": CLASS_NAMES[pred],
            "confidence": conf,
            "severity": SEVERITY_NAMES[sev] if sev >= 0 else None,
            "probabilities": {cn: float(probs[i]) for i, cn in enumerate(CLASS_NAMES)},
            "recommendation": get_recommendation(pred, max(sev, 0)),
            "inference_time_ms": round(elapsed, 1),
        })
    return {"results": results, "total": len(results)}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
