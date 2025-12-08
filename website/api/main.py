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
# 1. SYSTEM PATH FIX
# --------------------------------------------------------
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Visualization
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import binary_opening, binary_closing, label

# Local imports
from models.burn_scar_inference import BurnScarInference, SentinelHubAPI

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor='#0f172a')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

BASE_DIR = Path(__file__).parent.parent
WEIGHTS_PATH = BASE_DIR / "models" / "best_unet_burn_scar.pth"

app = FastAPI(title="Burn Scar Analytics API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

bbox_inference_engine = BurnScarInference(str(WEIGHTS_PATH))
sentinel_api = SentinelHubAPI()

class BBoxRequest(BaseModel):
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float
    date_from: str
    date_to: str

# -----------------------------------------------------------
# HELPER: Haversine Formula for Distance/Area
# -----------------------------------------------------------
def calculate_pixel_area_ha(bbox, img_width=512, img_height=512):
    """Returns the area of a single pixel in Hectares"""
    min_lon, min_lat, max_lon, max_lat = bbox
    
    # Radius of Earth in meters
    R = 6378137 
    
    # Height (Lat difference)
    lat_diff = math.radians(max_lat - min_lat)
    height_m = R * lat_diff
    
    # Width (Lon difference at the average latitude)
    avg_lat = math.radians((min_lat + max_lat) / 2)
    lon_diff = math.radians(max_lon - min_lon)
    width_m = R * lon_diff * math.cos(avg_lat)
    
    total_area_m2 = height_m * width_m
    pixel_area_m2 = total_area_m2 / (img_width * img_height)
    
    return pixel_area_m2 / 10000.0  # Convert m2 to Hectares

# -----------------------------------------------------------
# HELPER: OpenStreetMap Infrastructure Check
# -----------------------------------------------------------
def check_infrastructure(bbox, burn_mask):
    """Queries OSM for buildings/roads and checks overlap with burn mask"""
    min_lon, min_lat, max_lon, max_lat = bbox
    
    # Overpass API Query (timeout 25s)
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json][timeout:25];
    (
      way["building"]({min_lat},{min_lon},{max_lat},{max_lon});
      way["highway"]({min_lat},{min_lon},{max_lat},{max_lon});
    );
    out center;
    """
    
    try:
        response = requests.get(overpass_url, params={'data': overpass_query}, timeout=30)
        data = response.json()
    except:
        return {"buildings_risk": 0, "roads_risk": 0, "status": "OSM Timeout"}

    buildings_hit = 0
    roads_hit = 0
    
    h, w = burn_mask.shape
    
    for element in data.get("elements", []):
        # Get center coordinates
        if 'center' in element:
            lat = element['center']['lat']
            lon = element['center']['lon']
        elif 'lat' in element:
             lat = element['lat']
             lon = element['lon']
        else:
            continue

        # Map Lat/Lon to Pixel Coordinates
        # X = (lon - min_lon) / (max_lon - min_lon) * w
        # Y = (max_lat - lat) / (max_lat - min_lat) * h  <-- Inverted Y
        
        px = int((lon - min_lon) / (max_lon - min_lon) * w)
        py = int((max_lat - lat) / (max_lat - min_lat) * h)
        
        # Check bounds
        if 0 <= px < w and 0 <= py < h:
            # Check if this pixel is burned
            if burn_mask[py, px] == 1:
                if "building" in element.get("tags", {}):
                    buildings_hit += 1
                elif "highway" in element.get("tags", {}):
                    roads_hit += 1

    return {"buildings_risk": buildings_hit, "roads_risk": roads_hit, "status": "ok"}

# -----------------------------------------------------------
# HELPER: Pre-Fire Image Fetcher
# -----------------------------------------------------------
def get_pre_fire_image(bbox, current_date_str):
    try:
        curr_date = datetime.strptime(current_date_str, "%Y-%m-%d")
        # Go back 30-60 days
        prev_date = curr_date - timedelta(days=45)
        prev_date_to = prev_date + timedelta(days=15)
        
        bands = sentinel_api.download_imagery(
            bbox=bbox, 
            date_from=prev_date.strftime("%Y-%m-%d"), 
            date_to=prev_date_to.strftime("%Y-%m-%d"), 
            max_cloud_cover=20 # Stricter for pre-fire to look nice
        )
        
        if bands is None or np.max(bands) == 0:
            return None
            
        rgb = np.stack([bands[2], bands[1], bands[0]], axis=-1)
        rgb_vis = np.zeros_like(rgb)
        for i in range(3):
            c = rgb[:, :, i]
            vp = c[c>0]
            if vp.size > 0:
                p2, p98 = np.percentile(vp, (2, 98))
                rgb_vis[:, :, i] = np.clip((c - p2) / (p98 - p2 + 1e-8), 0, 1)
        
        fig = plt.figure(figsize=(5, 5))
        plt.imshow(rgb_vis)
        plt.axis('off')
        return fig_to_base64(fig)
    except:
        return None

def filter_small_blobs(mask, min_size):
    labeled_mask, num_features = label(mask)
    if num_features == 0: return mask
    component_sizes = np.bincount(labeled_mask.ravel())
    too_small = component_sizes < min_size
    too_small_mask = too_small[labeled_mask]
    mask[too_small_mask] = 0
    return mask

@app.post("/predict_with_bbox")
async def predict_with_bbox(req: BBoxRequest):
    bbox = [req.min_lon, req.min_lat, req.max_lon, req.max_lat]

    # 1. Main Sentinel Download
    try:
        bands = sentinel_api.download_imagery(
            bbox=bbox, date_from=req.date_from, date_to=req.date_to, max_cloud_cover=100
        )
    except Exception as e:
        raise HTTPException(500, f"Sentinel Hub Error: {str(e)}")

    if bands is None or bands.size == 0 or np.max(bands) == 0:
        raise HTTPException(status_code=404, detail="No satellite data found.")

    # 2. Inference
    prediction, confidence = bbox_inference_engine.predict(bands)
    h, w = prediction.shape
    total_pixels = prediction.size

    # 3. Spectral Filtering (The Clean Logic)
    green, nir, swir = bands[1], bands[3], bands[4]
    with np.errstate(divide='ignore', invalid='ignore'):
        ndwi = (green - nir) / (green + nir + 1e-8)
        nbr = (nir - swir) / (nir + swir + 1e-8)

    is_water = ndwi > 0.0
    is_spectrally_burned = nbr < 0.10  # Strict charcoal check

    burn_mask = (prediction == 1) & (confidence > 0.90) & (~is_water) & (is_spectrally_burned)
    burn_mask = filter_small_blobs(burn_mask, min_size=50)
    burn_mask = binary_closing(burn_mask, structure=np.ones((2, 2)))
    
    burned_pixels = int(np.sum(burn_mask))
    burned_fraction = burned_pixels / total_pixels

    # Global Safety Switch
    if burned_fraction < 0.002:
        burn_mask[:] = 0
        burned_pixels = 0

    # 4. NEW: Carbon & Area Calc
    pixel_ha = calculate_pixel_area_ha(bbox, w, h)
    burned_area_ha = burned_pixels * pixel_ha
    # Est: 20-30 tonnes CO2 per hectare for savanna/forest fires
    co2_tonnes = burned_area_ha * 25.0 

    # 5. NEW: Infrastructure Check
    infra_stats = {"buildings_risk": 0, "roads_risk": 0}
    if burned_pixels > 0:
        infra_stats = check_infrastructure(bbox, burn_mask)

    # 6. NEW: Pre-Fire Image
    pre_fire_b64 = None
    if burned_pixels > 0: # Only fetch pre-fire if there IS a fire
        pre_fire_b64 = get_pre_fire_image(bbox, req.date_from)

    # 7. Severity (Existing)
    mask_indices = np.where(burn_mask == 1)
    sev_counts = {"high": 0, "moderate": 0, "low": 0}
    if burned_pixels > 0:
        burned_nbr = nbr[mask_indices]
        sev_class = np.zeros_like(burned_nbr)
        sev_class[burned_nbr < 0.05] = 3
        sev_class[(burned_nbr >= 0.05) & (burned_nbr < 0.15)] = 2
        sev_class[burned_nbr >= 0.15] = 1
        sev_counts["high"] = float(np.sum(sev_class == 3) / burned_pixels)
        sev_counts["moderate"] = float(np.sum(sev_class == 2) / burned_pixels)
        sev_counts["low"] = float(np.sum(sev_class == 1) / burned_pixels)

    # 8. Visuals (Existing)
    rgb = np.stack([bands[2], bands[1], bands[0]], axis=-1)
    rgb_vis = np.zeros_like(rgb)
    for i in range(3):
        c = rgb[:, :, i]
        vp = c[c>0]
        if vp.size > 0:
            p2, p98 = np.percentile(vp, (2, 98))
            rgb_vis[:, :, i] = np.clip((c - p2) / (p98 - p2 + 1e-8), 0, 1)

    fig_rgb = plt.figure(figsize=(5, 5))
    plt.imshow(rgb_vis)
    plt.axis('off')
    rgb_b64 = fig_to_base64(fig_rgb)

    fig_overlay = plt.figure(figsize=(5, 5))
    plt.imshow(rgb_vis)
    overlay = np.zeros((h, w, 4))
    overlay[burn_mask == 1] = [1, 0, 0, 0.5] 
    plt.imshow(overlay)
    plt.axis('off')
    overlay_b64 = fig_to_base64(fig_overlay)

    fig_sev = plt.figure(figsize=(5, 5))
    colors = ['#00000000', '#fde047', '#f97316', '#7f1d1d']
    cmap_sev = ListedColormap(colors)
    plt.imshow(np.zeros_like(burn_mask), cmap='gray', vmin=0, vmax=1)
    display_map = np.zeros_like(nbr)
    display_map[mask_indices] = np.nan # reset
    # We need to reconstruct the severity class map for the whole image
    # (Simplified for display)
    sev_full = np.zeros((h, w))
    sev_full[mask_indices] = sev_class
    plt.imshow(sev_full, cmap=cmap_sev, vmin=0, vmax=3, interpolation='nearest')
    plt.axis('off')
    severity_b64 = fig_to_base64(fig_sev)

    return {
        "burned_pixels": burned_pixels,
        "burned_area_ha": round(burned_area_ha, 2),
        "co2_tonnes": round(co2_tonnes, 2),
        "infrastructure": infra_stats,
        "severity_breakdown": sev_counts,
        "rgb_base64": rgb_b64,
        "pre_fire_base64": pre_fire_b64,
        "overlay_base64": overlay_b64,
        "severity_base64": severity_b64
    }

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)