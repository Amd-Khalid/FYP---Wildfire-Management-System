import sys
import os
from pathlib import Path
import numpy as np
import uvicorn
import base64
import io
import requests
import math
from datetime import datetime, timedelta

# --------------------------------------------------------
# 1. SYSTEM PATH & CONFIG
# --------------------------------------------------------
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from scipy.ndimage import binary_closing, label
import rasterio.features
from rasterio.transform import from_bounds

# Local imports
from model.burn_scar_inference import BurnScarInference, SentinelHubAPI

# Define paths
BASE_DIR = Path(__file__).parent.parent
DET_WEIGHTS = BASE_DIR / "model" / "best_resnet34_unet.pth" # YOUR NEW MODEL
PRED_WEIGHTS = BASE_DIR / "model" / "best_unet_prediction.pth" # TEAMMATE'S PREDICTION MODEL

app = FastAPI(title="Burn Scar & Prediction API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Initialize Engine with BOTH models
bbox_inference_engine = BurnScarInference(
    detection_path=str(DET_WEIGHTS),
    prediction_path=str(PRED_WEIGHTS) if PRED_WEIGHTS.exists() else None
)
sentinel_api = SentinelHubAPI()

class BBoxRequest(BaseModel):
    min_lon: float; min_lat: float; max_lon: float; max_lat: float
    date_from: str; date_to: str

# -----------------------------------------------------------
# HELPERS
# -----------------------------------------------------------
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", transparent=True, pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def calculate_pixel_area_ha(bbox, w=512, h=512):
    min_lon, min_lat, max_lon, max_lat = bbox
    R = 6378137 
    lat_diff = math.radians(max_lat - min_lat)
    height_m = R * lat_diff
    avg_lat = math.radians((min_lat + max_lat) / 2)
    width_m = R * math.radians(max_lon - min_lon) * math.cos(avg_lat)
    return (height_m * width_m / (w * h)) / 10000.0

def filter_small_blobs(mask, min_size):
    labeled_mask, num_features = label(mask)
    if num_features == 0: return mask
    component_sizes = np.bincount(labeled_mask.ravel())
    mask[component_sizes[labeled_mask] < min_size] = 0
    return mask

def check_infrastructure(bbox, burn_mask):
    min_lon, min_lat, max_lon, max_lat = bbox
    overpass_url = "http://overpass-api.de/api/interpreter"
    query = f'[out:json][timeout:25];(way["building"]({min_lat},{min_lon},{max_lat},{max_lon});way["highway"]({min_lat},{min_lon},{max_lat},{max_lon}););(._;>;);out body;'
    try:
        data = requests.get(overpass_url, params={'data': query}, timeout=30).json()
        nodes = {el['id']: (el['lon'], el['lat']) for el in data.get("elements", []) if el.get('type') == 'node'}
        b_hits, r_hits = 0, 0
        h, w = burn_mask.shape
        transform = from_bounds(min_lon, min_lat, max_lon, max_lat, w, h)
        
        for el in [e for e in data.get("elements", []) if e.get('type') == 'way']:
            coords = [nodes[nid] for nid in el.get('nodes', []) if nid in nodes]
            if not coords: continue
            tags = el.get('tags') or {}
            geom = {"type": "Polygon", "coordinates": [coords + [coords[0]]]} if 'building' in tags else {"type": "LineString", "coordinates": coords}
            raster = rasterio.features.rasterize([(geom, 1)], out_shape=(h, w), transform=transform)
            if np.any((raster == 1) & (burn_mask == 1)):
                if 'building' in tags: b_hits += 1
                else: r_hits += 1
        return {"buildings_risk": b_hits, "roads_risk": r_hits}
    except: return {"buildings_risk": 0, "roads_risk": 0}

def robust_rgb_stretch(rgb_bands):
    str_rgb = np.zeros_like(rgb_bands)
    for i in range(3):
        channel = rgb_bands[:, :, i]
        p2, p98 = np.percentile(channel[channel > 0], (2, 98)) if np.any(channel > 0) else (0, 1)
        str_rgb[:, :, i] = np.clip((channel - p2) / (p98 - p2 + 1e-8), 0, 1)
    return str_rgb

# -----------------------------------------------------------
# MAIN ENDPOINT
# -----------------------------------------------------------
@app.post("/predict_with_bbox")
async def predict_with_bbox(req: BBoxRequest):
    bbox = [req.min_lon, req.min_lat, req.max_lon, req.max_lat]
    
    # 1. Download Current Imagery (BOTH S2 and S1 for our new model)
    try:
        # Added max_cloud_cover=100 back to prevent black mosaic clipping
        bands = sentinel_api.download_imagery(bbox, req.date_from, req.date_to, max_cloud_cover=100)
        s1_bands = sentinel_api.download_s1_imagery(bbox, req.date_from, req.date_to)
    except Exception as e:
        raise HTTPException(500, f"Satellite Fetch Error: {str(e)}")

    if bands is None or bands.size == 0 or np.max(bands) == 0:
        raise HTTPException(status_code=404, detail="No satellite data found.")

    # 2. Run Inference (Passes BOTH arrays, returns dict with detection AND risk)
    res = bbox_inference_engine.predict(bands, s1_bands, bbox=bbox, date=req.date_from)
    
    # 3. Refine Mask logic (Using YOUR tuned thresholds and water masking)
    green, nir, swir = bands[1], bands[3], bands[9]
    with np.errstate(divide='ignore', invalid='ignore'):
        ndwi = (green - nir) / (green + nir + 1e-8)
        nbr_post = (nir - swir) / (nir + swir + 1e-8)
        
    is_water = ndwi > 0.0
    
    # We use 0.30 because our Focal Loss model is stricter
    burn_mask = (res['detection'] == 1) & (res['det_confidence'] > 0.30) & (~is_water)
    burn_mask = filter_small_blobs(binary_closing(burn_mask, structure=np.ones((2, 2))), 20)
    
    # 4. Analytics
    burned_pixels = int(np.sum(burn_mask))
    pixel_ha = calculate_pixel_area_ha(bbox, w=bands.shape[2], h=bands.shape[1])
    burned_area_ha = burned_pixels * pixel_ha
    co2_tonnes = burned_area_ha * 25.0
    infra_stats = check_infrastructure(bbox, burn_mask) if burned_pixels > 0 else {"buildings_risk": 0, "roads_risk": 0}

    # 5. Pre-fire & Severity (TEAMMATE'S LOGIC)
    pre_b64, sev_b64 = None, None
    sev_counts = {"high": 0, "moderate": 0, "low": 0}
    
    post_rgb_raw = np.stack([bands[2], bands[1], bands[0]], -1)
    rgb_vis = robust_rgb_stretch(post_rgb_raw)

    try:
        if burned_pixels > 0:
            # Create a 30-day window to fetch pre-fire baseline
            pre_date_to = (datetime.strptime(req.date_from, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
            pre_date_from = (datetime.strptime(req.date_from, "%Y-%m-%d") - timedelta(days=30)).strftime("%Y-%m-%d")
            
            pre_bands = sentinel_api.download_imagery(bbox, pre_date_from, pre_date_to)
            pre_rgb_raw = np.stack([pre_bands[2], pre_bands[1], pre_bands[0]], -1)
            
            fig_pre, ax_pre = plt.subplots(figsize=(5, 5))
            ax_pre.imshow(robust_rgb_stretch(pre_rgb_raw))
            ax_pre.axis('off'); pre_b64 = fig_to_base64(fig_pre)

            # Teammate's Dynamic Severity (dNBR)
            pre_nbr = (pre_bands[3] - pre_bands[9]) / (pre_bands[3] + pre_bands[9] + 1e-8)
            dnbr = pre_nbr - nbr_post
            
            fig_sev, ax_sev = plt.subplots(figsize=(5, 5))
            ax_sev.imshow(rgb_vis) # Geographic background
            
            severity_overlay = np.where(burn_mask == 1, dnbr, np.nan)
            active_dnbr = dnbr[burn_mask == 1]
            if len(active_dnbr) > 0:
                vmin_dyn, vmax_dyn = np.percentile(active_dnbr, (5, 95))
                if vmin_dyn == vmax_dyn: vmin_dyn, vmax_dyn = 0.1, 0.7
            else:
                vmin_dyn, vmax_dyn = 0.1, 0.7
                
            ax_sev.imshow(severity_overlay, cmap='YlOrRd', vmin=vmin_dyn, vmax=vmax_dyn)
            ax_sev.axis('off'); sev_b64 = fig_to_base64(fig_sev)
            
            # Update severity stats for frontend charts
            sev_counts["high"] = float(np.sum((dnbr > 0.6) & (burn_mask == 1)) / burned_pixels)
            sev_counts["moderate"] = float(np.sum((dnbr > 0.25) & (dnbr <= 0.6) & (burn_mask == 1)) / burned_pixels)
            sev_counts["low"] = float(np.sum((dnbr <= 0.25) & (burn_mask == 1)) / burned_pixels)
    except Exception as e: print(f"! Pre-fire error: {e}")

    # 6. Post-Fire Imagery 
    fig_rgb, ax_rgb = plt.subplots(figsize=(5, 5))
    ax_rgb.imshow(rgb_vis); ax_rgb.axis('off'); rgb_b64 = fig_to_base64(fig_rgb)

    fig_det, ax_det = plt.subplots(figsize=(5, 5))
    ax_det.imshow(rgb_vis)
    overlay = np.zeros((*burn_mask.shape, 4))
    overlay[burn_mask == 1] = [1, 0, 0, 0.5]
    ax_det.imshow(overlay); ax_det.axis('off'); det_b64 = fig_to_base64(fig_det)

    # 7. Spread Risk (TEAMMATE'S LOGIC)
    pred_b64 = None
    if 'prediction_risk' in res:
        fig_p, ax_p = plt.subplots(figsize=(5, 5))
        ax_p.imshow(rgb_vis) 
        
        cmap = LinearSegmentedColormap.from_list("spread", ["#00000000", "#fbbf24", "#ef4444", "#7f1d1d"])
        raw_risk = res['prediction_risk']
        risk = np.where(raw_risk > 0.40, raw_risk, np.nan)
        
        # Crop borders to remove U-Net padding lines
        margin = 10
        risk[0:margin,:], risk[-margin:,:], risk[:,0:margin], risk[:,-margin:] = np.nan, np.nan, np.nan, np.nan
        
        ax_p.imshow(risk, cmap=cmap, vmin=0.60, vmax=1.0) 
        ax_p.axis('off'); pred_b64 = fig_to_base64(fig_p)

    return {
        "burned_area_ha": round(burned_area_ha, 2),
        "co2_tonnes": round(co2_tonnes, 2),
        "infrastructure": infra_stats,
        "severity_breakdown": sev_counts,
        "overlay_base64": det_b64, 
        "predict_risk_base64": pred_b64,
        "rgb_base64": rgb_b64,      
        "pre_fire_base64": pre_b64, 
        "severity_base64": sev_b64,
        "status": "Success"
    }

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)