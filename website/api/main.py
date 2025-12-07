from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from typing import Optional
import numpy as np
import uvicorn
import base64
import io

# Visualization
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# Optional smoothing for cleaner shapes
from scipy.ndimage import binary_opening, binary_closing

# Local imports
from models.unet_inference import BurnScarUNetInference
from models.burn_scar_inference import BurnScarInference, SentinelHubAPI


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ============================================================
#  PATHS & INITIALIZATION
# ============================================================

BASE_DIR = Path(__file__).parent.parent
WEIGHTS_PATH = BASE_DIR / "models" / "best_unet_burn_scar.pth"

app = FastAPI(
    title="Burn Scar Detection API",
    description="Bounding box + Sentinel-2 + U-Net",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Engines
file_inference_engine = BurnScarUNetInference(str(WEIGHTS_PATH))
bbox_inference_engine = BurnScarInference(str(WEIGHTS_PATH))
sentinel_api = SentinelHubAPI()


# ============================================================
#  MODELS / SCHEMAS
# ============================================================

class PredictionSummary(BaseModel):
    burned_fraction: float
    burned_pixels: int
    total_pixels: int
    height: int
    width: int


class BBoxRequest(BaseModel):
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float
    date_from: str
    date_to: str


# ============================================================
# FILE UPLOAD ENDPOINT (unchanged)
# ============================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...), return_mask_png: bool = False):
    if not file.filename.lower().endswith(".tif"):
        raise HTTPException(400, "Please upload a .tif GeoTIFF")

    tif_bytes = await file.read()
    try:
        mask, prob_map = file_inference_engine.predict_mask(tif_bytes)
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}")

    h, w = mask.shape
    total = mask.size
    burned = int(np.sum(mask == 1))

    summary = PredictionSummary(
        burned_fraction=burned / total,
        burned_pixels=burned,
        total_pixels=total,
        height=h,
        width=w,
    )

    if not return_mask_png:
        return JSONResponse(content=summary.dict())

    png_bytes = file_inference_engine.mask_to_png_bytes(mask)
    return Response(content=png_bytes, media_type="image/png")


# ============================================================
# NEW ENDPOINT: BBOX + SENTINEL + 4 IMAGES
# ============================================================

@app.post("/predict_with_bbox")
async def predict_with_bbox(req: BBoxRequest):

    bbox = [req.min_lon, req.min_lat, req.max_lon, req.max_lat]

    # 1. Download Sentinel-2 imagery
    try:
        bands = sentinel_api.download_imagery(
            bbox=bbox,
            date_from=req.date_from,
            date_to=req.date_to,
            max_cloud_cover=40
        )
    except Exception as e:
        raise HTTPException(500, f"Sentinel download failed: {e}")

    # 2. Run U-Net inference
    try:
        prediction, confidence = bbox_inference_engine.predict(bands)
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}")

    h, w = prediction.shape
    total = prediction.size

    # =======================================================
    # CONFIDENCE-THRESHOLDED BURN MASK (used everywhere)
    # =======================================================
    THRESHOLD = 0.97

    # high-confidence burned pixels
    burn_mask = (prediction == 1) & (confidence > THRESHOLD)

    # smooth the mask a bit
    burn_mask = binary_opening(burn_mask, structure=np.ones((3, 3)))
    burn_mask = binary_closing(burn_mask, structure=np.ones((3, 3)))

    burned = int(np.sum(burn_mask))
    burned_fraction = burned / total if total > 0 else 0.0

    avg_conf = float(np.mean(confidence))
    burned_conf = float(np.mean(confidence[burn_mask])) if burned > 0 else 0.0

    # =======================================================
    # --- RGB IMAGE ---
    # =======================================================
    rgb = np.stack([bands[2], bands[1], bands[0]], axis=-1)
    rgb_vis = np.zeros_like(rgb)
    for i in range(3):
        p2, p98 = np.percentile(rgb[:, :, i], (2, 98))
        rgb_vis[:, :, i] = np.clip((rgb[:, :, i] - p2) / (p98 - p2 + 1e-8), 0, 1)

    fig1 = plt.figure(figsize=(6, 6))
    plt.imshow(rgb_vis)
    plt.title("Satellite Imagery (RGB)", fontsize=14)
    plt.axis("off")
    rgb_b64 = fig_to_base64(fig1)

    # =======================================================
    # --- OVERLAY (uses burn_mask) ---
    # =======================================================
    fig2 = plt.figure(figsize=(6, 6))
    plt.imshow(rgb_vis)

    overlay = np.zeros((*burn_mask.shape, 4))
    overlay[:, :, 0] = 1.0               # red
    overlay[:, :, 3] = burn_mask * 0.55  # alpha only where confident burned

    plt.imshow(overlay)
    plt.title(f"Burn Scar Overlay (conf > {THRESHOLD})", fontsize=14)
    plt.axis("off")
    overlay_b64 = fig_to_base64(fig2)

    # =======================================================
    # --- BURN SCAR MASK (also uses burn_mask) ---
    # =======================================================
    fig3 = plt.figure(figsize=(6, 6))
    cmap_mask = ListedColormap(["lightgray", "darkred"])
    # burn_mask is boolean; imshow handles it fine as 0/1
    plt.imshow(burn_mask, cmap=cmap_mask, vmin=0, vmax=1)
    plt.title(f"Burn Scar Mask (conf > {THRESHOLD})", fontsize=14)
    plt.axis("off")
    mask_b64 = fig_to_base64(fig3)

    # =======================================================
    # --- CONFIDENCE MAP (same as before) ---
    # =======================================================
    fig4 = plt.figure(figsize=(6, 6))
    im = plt.imshow(confidence, cmap="YlOrRd", vmin=0, vmax=1)
    plt.title("Model Confidence", fontsize=14)
    plt.axis("off")
    plt.colorbar(im, fraction=0.046, pad=0.02)
    confidence_b64 = fig_to_base64(fig4)

    # =======================================================

    return {
        "burned_fraction": burned_fraction,
        "burned_pixels": burned,
        "total_pixels": total,
        "height": h,
        "width": w,
        "average_confidence": avg_conf,
        "burned_area_confidence": burned_conf,
        "rgb_base64": rgb_b64,
        "overlay_base64": overlay_b64,
        "mask_base64": mask_b64,
        "confidence_base64": confidence_b64,
    }


# ============================================================

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
