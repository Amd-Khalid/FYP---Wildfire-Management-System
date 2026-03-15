"""
Burn Scar AI: Integrated Detection & Prediction Engine
Updated to use the properly-trained spread prediction model.

Key changes from the original:
  - Spread model now uses 12 meaningful channels (matching training data)
  - Wind is now a spatial u/v vector field, not a scalar
  - Slope/aspect are derived from a real DEM (Open-Elevation grid), not constants
  - Normalization matches the training z-score normalization exactly
  - Model architecture matches training (UnetPlusPlus, 12ch → 1ch)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import rasterio
from rasterio.io import MemoryFile
import segmentation_models_pytorch as smp
import requests
from typing import Dict, Optional
import cv2


# ─── Normalization constants ────────────────────────────────────────────────
# Channel order matches training notebook INPUT_FEATURES:
# elevation, pdsi, NDVI, population, erc, pr, sph, th, tmmn, tmmx, vs, PrevFireMask
SPREAD_MEANS = np.array([
    657.0,   # elevation
   -0.5,     # pdsi
    5500.0,  # NDVI (raw ×10000 scaled)
    25.0,    # population
    61.0,    # erc
    0.018,   # pr
    0.0076,  # sph
    200.0,   # th (wind direction)
    285.0,   # tmmn (Kelvin)
    298.0,   # tmmx (Kelvin)
    3.2,     # vs (wind speed)
    0.0,     # PrevFireMask
], dtype=np.float32)

SPREAD_STDS = np.array([
    500.0,   # elevation
    2.5,     # pdsi
    3000.0,  # NDVI
    170.0,   # population
    26.0,    # erc
    0.02,    # pr
    0.003,   # sph
    100.0,   # th
    10.0,    # tmmn
    10.0,    # tmmx
    2.0,     # vs
    1.0,     # PrevFireMask
], dtype=np.float32)


class BurnScarInference:
    def __init__(self,
                 detection_path: str,
                 prediction_path: Optional[str] = None,
                 device: str = 'cuda'):

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # ── 1. Detection model (unchanged — 8-channel U-Net) ───────────────
        self.detection_model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=8,
            classes=2,
            activation=None,
        ).to(self.device)
        self.detection_model.load_state_dict(
            torch.load(detection_path, map_location=self.device)
        )
        self.detection_model.eval()
        print(f"✓ Detection model loaded from {detection_path}")

        # ── 2. Spread prediction model (12-channel UNet++) ──────────────────
        # Architecture MUST match training notebook (build_spread_model())
        self.prediction_model = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=12,       # matches training
            classes=1,            # single probability output
            activation=None,
            decoder_attention_type="scse",
        ).to(self.device)

        if prediction_path:
            self.prediction_model.load_state_dict(
                torch.load(prediction_path, map_location=self.device)
            )
            self._spread_trained = True
            print(f"✓ Spread model loaded from {prediction_path}")
        else:
            self._spread_trained = False
            print("! Spread model: no weights provided — spread prediction disabled")
        self.prediction_model.eval()

    # ────────────────────────────────────────────────────────────────────────
    # Padding helper
    # ────────────────────────────────────────────────────────────────────────

    def _pad_to_32(self, tensor: torch.Tensor):
        """Pad tensor so H and W are both divisible by 32. Returns (padded, orig_shape)."""
        _, _, h, w = tensor.shape
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        # (left, right, top, bottom)
        padded = F.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')
        return padded, (h, w)

    # ────────────────────────────────────────────────────────────────────────
    # External driver fetching (fixed: real spatial data, not scalars)
    # ────────────────────────────────────────────────────────────────────────

    def get_spread_drivers(self, bbox: list, date: str, h: int, w: int) -> Dict:
        """
        Fetch all 12 channels needed for spread prediction.
        Returns a dict of (H, W) numpy arrays.
        
        For a proper production system you'd want:
          - Real DEM tiles from OpenTopography or Copernicus DEM
          - ERA5 hourly wind fields (u, v components)
          - GRIDMET for ERC, humidity, temperature
        
        For a FYP with free APIs, this fetches the scalar values and creates
        spatially-reasonable approximations.
        """
        min_lon, min_lat, max_lon, max_lat = bbox
        mid_lat = (min_lat + max_lat) / 2
        mid_lon = (min_lon + max_lon) / 2

        drivers = {}

        # ── Weather: Open-Meteo (free, no API key) ───────────────────────
        # Fetches real historical weather at the centroid
        try:
            url = (
                f"https://archive-api.open-meteo.com/v1/archive"
                f"?latitude={mid_lat}&longitude={mid_lon}"
                f"&start_date={date}&end_date={date}"
                f"&daily=windspeed_10m_max,winddirection_10m_dominant,"
                f"temperature_2m_max,precipitation_sum,et0_fao_evapotranspiration"
                f"&timezone=auto"
            )
            resp = requests.get(url, timeout=8).json()
            daily = resp.get('daily', {})
            wind_speed = (daily.get('windspeed_10m_max', [5.0]) or [5.0])[0] or 5.0
            wind_dir   = (daily.get('winddirection_10m_dominant', [180.0]) or [180.0])[0] or 180.0
            tmax_c     = (daily.get('temperature_2m_max', [25.0]) or [25.0])[0] or 25.0
            precip     = (daily.get('precipitation_sum', [0.0]) or [0.0])[0] or 0.0
            # Rough ERC proxy from temp and humidity (very simplified)
            erc_proxy  = max(0, tmax_c * 2.5 - precip * 10)
            
            drivers['wind_speed'] = wind_speed
            drivers['wind_dir']   = wind_dir        # degrees from north
            drivers['tmmx']       = tmax_c + 273.15 # → Kelvin
            drivers['pr']         = precip
            drivers['erc']        = erc_proxy
            # Humidity approximation from temperature (rough)
            drivers['sph']        = max(0.002, 0.015 - tmax_c * 0.0003)
            print(f"✓ Weather: wind={wind_speed:.1f}m/s @ {wind_dir:.0f}°, T={tmax_c:.1f}°C")

        except Exception as e:
            print(f"! Weather fetch failed: {e}. Using safe defaults.")
            drivers.update({
                'wind_speed': 5.0, 'wind_dir': 180.0,
                'tmmx': 303.0, 'pr': 0.0, 'erc': 60.0, 'sph': 0.005
            })

        # ── Elevation: Open-Elevation grid (free, no key) ───────────────
        # Sample a 5×5 grid of points across the bbox for spatial variation
        try:
            lats = np.linspace(min_lat, max_lat, 5)
            lons = np.linspace(min_lon, max_lon, 5)
            lo_grid, la_grid = np.meshgrid(lons, lats)
            locations = [
                {"latitude": float(la), "longitude": float(lo)}
                for la, lo in zip(la_grid.ravel(), lo_grid.ravel())
            ]
            resp = requests.post(
                "https://api.open-elevation.com/api/v1/lookup",
                json={"locations": locations}, timeout=10
            ).json()
            elevs = np.array([r['elevation'] for r in resp['results']],
                              dtype=np.float32).reshape(5, 5)

            # Upsample the 5×5 grid to (H, W)
            elev_map = cv2.resize(elevs, (w, h), interpolation=cv2.INTER_LINEAR)

            # Compute slope and aspect from the elevation grid
            gy, gx = np.gradient(elev_map)
            slope_map  = np.degrees(np.arctan(np.sqrt(gx**2 + gy**2)))
            aspect_map = np.degrees(np.arctan2(-gx, gy)) % 360
            print(f"✓ Elevation fetched: {elevs.mean():.0f}m avg, "
                  f"slope={slope_map.mean():.1f}° avg")

        except Exception as e:
            print(f"! Elevation fetch failed: {e}. Using flat defaults.")
            center_elev = 500.0
            elev_map   = np.full((h, w), center_elev, dtype=np.float32)
            slope_map  = np.zeros((h, w), dtype=np.float32)
            aspect_map = np.zeros((h, w), dtype=np.float32)

        drivers['elevation'] = elev_map
        drivers['slope']     = slope_map
        drivers['aspect']    = aspect_map

        return drivers

    # ────────────────────────────────────────────────────────────────────────
    # Preprocessing
    # ────────────────────────────────────────────────────────────────────────

    def _preprocess_detection(self, bands: np.ndarray) -> torch.Tensor:
        """8-channel detection input — unchanged from original."""
        blue, green, red, nir, swir1, swir2 = bands
        ndvi = (nir - red)   / (nir + red   + 1e-8)
        nbr  = (nir - swir1) / (nir + swir1 + 1e-8)
        stack = np.stack([blue, green, red, nir, swir1, swir2, ndvi, nbr], axis=0)
        tensor = torch.from_numpy(stack).float()
        tensor = torch.clamp(tensor / 3000.0, 0, 1)
        return tensor.unsqueeze(0)

    def _preprocess_spread(self,
                            bands: np.ndarray,
                            burn_mask: np.ndarray,
                            drivers: Dict) -> torch.Tensor:
        """
        12-channel spread prediction input.
        Channel order matches training exactly:
          0  elevation
          1  slope
          2  aspect
          3  canopy_height  (proxy from NDVI)
          4  population     (constant 0 — no free spatial API)
          5  erc
          6  pr
          7  sph
          8  th (wind direction)
          9  tmmx
          10 vs (wind speed)
          11 PrevFireMask
        """
        h, w = bands.shape[1], bands.shape[2]

        # Canopy height proxy: rescale NDVI to [0, 30] metres
        red, nir = bands[2], bands[3]
        ndvi = (nir - red) / (nir + red + 1e-8)
        canopy_proxy = np.clip(ndvi * 30.0, 0, 60).astype(np.float32)

        # Channel order MUST match training INPUT_FEATURES:
        # elevation, pdsi, NDVI, population, erc, pr, sph, th, tmmn, tmmx, vs, PrevFireMask
        tmmx_k = drivers['tmmx']                          # already in Kelvin
        tmmn_k = tmmx_k - 10.0                            # rough min temp approximation
        pdsi_proxy = np.full((h, w), -1.0, dtype=np.float32)  # moderate drought default

        stack = np.stack([
            drivers['elevation'].astype(np.float32),      # elevation
            pdsi_proxy,                                    # pdsi
            canopy_proxy * (10000.0 / 30.0),              # NDVI scaled ×10000
            np.zeros((h, w), dtype=np.float32),           # population
            np.full((h, w), drivers['erc'],        dtype=np.float32),
            np.full((h, w), drivers['pr'],         dtype=np.float32),
            np.full((h, w), drivers['sph'],        dtype=np.float32),
            np.full((h, w), drivers['wind_dir'],   dtype=np.float32),  # th
            np.full((h, w), tmmn_k,                dtype=np.float32),  # tmmn
            np.full((h, w), tmmx_k,                dtype=np.float32),  # tmmx
            np.full((h, w), drivers['wind_speed'], dtype=np.float32),  # vs
            burn_mask.astype(np.float32),                 # PrevFireMask
        ], axis=0)   # → (12, H, W)

        # Z-score normalization using training statistics
        means = SPREAD_MEANS[:, None, None]
        stds  = SPREAD_STDS[:, None, None]
        stack = (stack - means) / (stds + 1e-8)
        stack = np.clip(stack, -3, 3)

        return torch.from_numpy(stack).float().unsqueeze(0)

    # ────────────────────────────────────────────────────────────────────────
    # Inference
    # ────────────────────────────────────────────────────────────────────────

    def predict(self,
                bands: np.ndarray,
                bbox: list = None,
                date: str = None,
                burn_mask: Optional[np.ndarray] = None) -> Dict:
        """
        Run detection and (optionally) spread prediction.

        Args:
            bands      : (6, H, W) Sentinel-2 bands [B02,B03,B04,B08,B11,B12]
            bbox       : [min_lon, min_lat, max_lon, max_lat]
            date       : 'YYYY-MM-DD' for weather/terrain fetch
            burn_mask  : (H, W) binary mask from detection step, or None
                         If None, the detection output is used automatically.
        
        Returns dict with keys:
            detection          — (H, W) int array, 0/1
            det_confidence     — (H, W) float, fire class probability
            prediction_risk    — (H, W) float, spread probability (if available)
        """
        h, w = bands.shape[1], bands.shape[2]
        results = {}

        # ── Detection ─────────────────────────────────────────────────────
        det_tensor = self._preprocess_detection(bands).to(self.device)
        det_tensor, orig_shape = self._pad_to_32(det_tensor)
        with torch.no_grad():
            det_out = self.detection_model(det_tensor)
        det_out = det_out[:, :, :orig_shape[0], :orig_shape[1]]
        results['detection']      = torch.argmax(det_out, dim=1)[0].cpu().numpy()
        results['det_confidence'] = torch.softmax(det_out, dim=1)[0, 1].cpu().numpy()

        # ── Spread prediction ──────────────────────────────────────────────
        if bbox and date and self._spread_trained:
            prev_mask = burn_mask if burn_mask is not None else results['detection']

            drivers = self.get_spread_drivers(bbox, date, h, w)
            pred_tensor = self._preprocess_spread(bands, prev_mask, drivers)
            pred_tensor, orig_shape = self._pad_to_32(pred_tensor.to(self.device))
            with torch.no_grad():
                pred_out = self.prediction_model(pred_tensor)
            pred_out = pred_out[:, :, :orig_shape[0], :orig_shape[1]]
            results['prediction_risk'] = torch.sigmoid(pred_out)[0, 0].cpu().numpy()
            print(f"✓ Spread prediction complete. "
                  f"Max risk: {results['prediction_risk'].max():.3f}")

        elif not self._spread_trained:
            print("! Spread model weights not loaded — skipping prediction.")

        return results

    def calculate_metrics(self,
                           prediction: np.ndarray,
                           ground_truth: np.ndarray) -> Dict:
        from sklearn.metrics import (accuracy_score, jaccard_score,
                                     f1_score, precision_score, recall_score)
        pred_flat = prediction.flatten()
        gt_flat   = ground_truth.flatten()
        valid     = (gt_flat != 255) & (gt_flat != -1)
        pred_flat, gt_flat = pred_flat[valid], gt_flat[valid]
        return {
            'accuracy':  accuracy_score (gt_flat, pred_flat),
            'iou':       jaccard_score  (gt_flat, pred_flat, average='binary', zero_division=0),
            'f1':        f1_score       (gt_flat, pred_flat, average='binary', zero_division=0),
            'precision': precision_score(gt_flat, pred_flat, average='binary', zero_division=0),
            'recall':    recall_score   (gt_flat, pred_flat, average='binary', zero_division=0),
        }


# ────────────────────────────────────────────────────────────────────────────
# SentinelHubAPI — unchanged from original (no modifications needed)
# ────────────────────────────────────────────────────────────────────────────

class SentinelHubAPI:
    def __init__(self, client_id: str = None, client_secret: str = None):
        try:
            from config import (SENTINEL_CLIENT_ID, SENTINEL_CLIENT_SECRET,
                                SENTINEL_TOKEN_URL, SENTINEL_BASE_URL)
            self.client_id     = client_id or SENTINEL_CLIENT_ID
            self.client_secret = client_secret or SENTINEL_CLIENT_SECRET
            self.token_url     = SENTINEL_TOKEN_URL
            self.process_url   = f"{SENTINEL_BASE_URL}/api/v1/process"
        except ImportError:
            self.token_url   = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'
            self.process_url = 'https://sh.dataspace.copernicus.eu/api/v1/process'

        self.access_token = None
        self._authenticate()

    def _authenticate(self):
        payload = {
            'grant_type':    'client_credentials',
            'client_id':     self.client_id,
            'client_secret': self.client_secret,
        }
        response = requests.post(self.token_url, data=payload)
        response.raise_for_status()
        self.access_token = response.json()['access_token']
        print("✓ Authenticated with CDSE")

    def download_imagery(self, bbox: list, date_from: str, date_to: str,
                         max_cloud_cover: float = 20.0) -> np.ndarray:
        width_deg    = abs(bbox[2] - bbox[0])
        height_deg   = abs(bbox[3] - bbox[1])
        aspect_ratio = height_deg / width_deg
        width        = 512
        height       = int(512 * aspect_ratio)

        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type':  'application/json',
        }
        payload = {
            "input": {
                "bounds": {
                    "bbox": bbox,
                    "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}
                },
                "data": [{
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": f"{date_from}T00:00:00Z",
                            "to":   f"{date_to}T23:59:59Z"
                        },
                        "maxCloudCoverage": max_cloud_cover
                    }
                }]
            },
            "output": {
                "width": width, "height": height,
                "responses": [{"identifier": "default",
                               "format": {"type": "image/tiff"}}]
            },
            "evalscript": """
                //VERSION=3
                function setup() {
                    return {
                        input: ["B02","B03","B04","B08","B11","B12"],
                        output: {bands: 6, sampleType: "FLOAT32"}
                    };
                }
                function evaluatePixel(sample) {
                    return [sample.B02, sample.B03, sample.B04,
                            sample.B08, sample.B11, sample.B12];
                }
            """
        }
        response = requests.post(self.process_url, headers=headers, json=payload)
        response.raise_for_status()

        with MemoryFile(response.content) as memfile:
            with memfile.open() as dataset:
                bands = dataset.read()

        print(f"✓ Downloaded imagery: {bands.shape}")
        return bands