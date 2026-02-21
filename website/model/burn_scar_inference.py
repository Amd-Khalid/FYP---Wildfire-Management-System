"""
Burn Scar AI: Integrated Detection & Prediction Engine
"""
import torch
import numpy as np
import rasterio
from rasterio.io import MemoryFile
import segmentation_models_pytorch as smp
import requests
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt

class BurnScarInference:
    def __init__(self, detection_path: str, prediction_path: Optional[str] = None, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 1. DETECTION MODEL (Standard 8-channel U-Net)
        self.detection_model = smp.Unet(
            encoder_name="resnet34", encoder_weights=None,
            in_channels=8, classes=2, activation=None
        ).to(self.device)
        self.detection_model.load_state_dict(torch.load(detection_path, map_location=self.device))
        self.detection_model.eval()
        print(f"✓ Detection Model loaded from {detection_path}")

        # 2. PREDICTION MODEL (Enhanced 11-channel U-Net for PoC)
        self.prediction_model = smp.Unet(
            encoder_name="resnet34", encoder_weights=None,
            in_channels=11, classes=2, activation=None
        ).to(self.device)
        
        if prediction_path:
            self.prediction_model.load_state_dict(torch.load(prediction_path, map_location=self.device))
            print(f"✓ Prediction Model loaded from {prediction_path}")
        else:
            print("! Prediction model running with initialized weights (PoC Mode)")
        self.prediction_model.eval()

    def get_external_drivers(self, bbox: list, date: str) -> Dict:
        """Fetches topography and wind data for prediction."""
        min_lon, min_lat, max_lon, max_lat = bbox
        mid_lat, mid_lon = (min_lat + max_lat) / 2, (min_lon + max_lon) / 2

        try:
            # Weather: Open-Meteo (Wind speed/Temp)
            weather_url = f"https://archive-api.open-meteo.com/v1/archive?latitude={mid_lat}&longitude={mid_lon}&start_date={date}&end_date={date}&daily=windspeed_10m_max,temperature_2m_max&timezone=auto"
            w_resp = requests.get(weather_url, timeout=5).json()
            wind = w_resp['daily']['windspeed_10m_max'][0] or 12.0
            
            # Elevation: Open-Elevation
            elev_url = f"https://api.open-elevation.com/api/v1/lookup?locations={mid_lat},{mid_lon}"
            e_resp = requests.get(elev_url, timeout=5).json()
            elevation = e_resp['results'][0]['elevation']
        except Exception as e:
            print(f"! Driver Fetch Error: {e}. Using defaults.")
            wind, elevation = 12.0, 500.0

        return {'wind': wind, 'elevation': elevation}

    def preprocess(self, bands: np.ndarray, mode: str = 'detection', extra: Dict = None) -> torch.Tensor:
        # Base 8 channels (Spectral + Indices)
        blue, green, red, nir, swir1, swir2 = bands
        ndvi = (nir - red) / (nir + red + 1e-8)
        nbr = (nir - swir1) / (nir + swir1 + 1e-8)
        base_stack = [blue, green, red, nir, swir1, swir2, ndvi, nbr]

        if mode == 'predict' and extra:
            h, w = bands.shape[1], bands.shape[2]
            # Normalize to 0-1 range based on realistic max values
            elev_map = np.full((h, w), extra['elevation'] / 5000.0)
            wind_map = np.full((h, w), extra['wind'] / 50.0)
            
            # Simple gradient for slope
            dy, dx = np.gradient(np.full((h, w), extra['elevation']))
            slope_map = np.clip(np.sqrt(dx**2 + dy**2) / 10.0, 0, 1)
            
            base_stack.extend([elev_map, wind_map, slope_map])

        img_stack = np.stack(base_stack, axis=0)
        img_tensor = torch.from_numpy(img_stack).float()
        
        # CRITICAL: Match the exact training normalizations!
        if mode == 'detection':
            img_tensor = torch.clamp(img_tensor / 3000.0, 0, 1)
        else:
            # Predictive model was trained on pre-normalized TFRecords
            img_tensor = torch.clamp(img_tensor, 0, 1)
            
        return img_tensor.unsqueeze(0)
    
    def predict(self, bands: np.ndarray, bbox: list = None, date: str = None) -> Dict:
        """Runs both detection and prediction if drivers are provided."""
        results = {}

        # 1. Detection Mode
        det_tensor = self.preprocess(bands, mode='detection').to(self.device)
        with torch.no_grad():
            det_out = self.detection_model(det_tensor)
        results['detection'] = torch.argmax(det_out, dim=1)[0].cpu().numpy()
        results['det_confidence'] = torch.softmax(det_out, dim=1)[0, 1].cpu().numpy()

        # 2. Prediction Mode
        if bbox and date:
            extra = self.get_external_drivers(bbox, date)
            pred_tensor = self.preprocess(bands, mode='predict', extra=extra).to(self.device)
            with torch.no_grad():
                pred_out = self.prediction_model(pred_tensor)
            results['prediction_risk'] = torch.softmax(pred_out, dim=1)[0, 1].cpu().numpy()
        
        return results

    def calculate_metrics(self, prediction: np.ndarray, ground_truth: np.ndarray) -> Dict:
        from sklearn.metrics import accuracy_score, jaccard_score, f1_score, precision_score, recall_score
        pred_flat = prediction.flatten()
        gt_flat = ground_truth.flatten()
        valid_mask = (gt_flat != 255) & (gt_flat != -1)
        pred_flat, gt_flat = pred_flat[valid_mask], gt_flat[valid_mask]
        
        return {
            'accuracy': accuracy_score(gt_flat, pred_flat),
            'iou': jaccard_score(gt_flat, pred_flat, average='binary', zero_division=0),
            'f1': f1_score(gt_flat, pred_flat, average='binary', zero_division=0),
            'precision': precision_score(gt_flat, pred_flat, average='binary', zero_division=0),
            'recall': recall_score(gt_flat, pred_flat, average='binary', zero_division=0)
        }

class SentinelHubAPI:
    def __init__(self, client_id: str = None, client_secret: str = None):
        try:
            from config import (
                SENTINEL_CLIENT_ID, SENTINEL_CLIENT_SECRET, 
                SENTINEL_TOKEN_URL, SENTINEL_BASE_URL
            )
            self.client_id = client_id or SENTINEL_CLIENT_ID
            self.client_secret = client_secret or SENTINEL_CLIENT_SECRET
            self.token_url = SENTINEL_TOKEN_URL
            self.process_url = f"{SENTINEL_BASE_URL}/api/v1/process"
            
        except ImportError:
            self.token_url = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'
            self.process_url = 'https://sh.dataspace.copernicus.eu/api/v1/process'

        self.access_token = None
        self._authenticate()
    
    def _authenticate(self):
        payload = {
            'grant_type': 'client_credentials', 
            'client_id': self.client_id, 
            'client_secret': self.client_secret
        }
        response = requests.post(self.token_url, data=payload)
        response.raise_for_status()
        self.access_token = response.json()['access_token']
        print("✓ Successfully Authenticated with CDSE")
    
    def download_imagery(self, bbox: list, date_from: str, date_to: str, max_cloud_cover: float = 20.0) -> np.ndarray:
        width_deg = abs(bbox[2] - bbox[0])
        height_deg = abs(bbox[3] - bbox[1])
        aspect_ratio = height_deg / width_deg
    
        width = 512
        # Force height to be divisible by 32 to prevent internal U-Net dimension crashes
        raw_height = int(512 * aspect_ratio)
        height = max(32, round(raw_height / 32) * 32)

        headers = {'Authorization': f'Bearer {self.access_token}', 'Content-Type': 'application/json'}
    
        payload = {
        "input": {
            "bounds": {"bbox": bbox, "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
            "data": [{
                "type": "sentinel-2-l2a", 
                "dataFilter": {
                    "timeRange": {"from": f"{date_from}T00:00:00Z", "to": f"{date_to}T23:59:59Z"},
                    "maxCloudCoverage": max_cloud_cover
                }
            }]
        },
        "output": {
            "width": width, 
            "height": height, 
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]
        },
        "evalscript": """
            //VERSION=3
            function setup() { 
                return {input: ["B02", "B03", "B04", "B08", "B11", "B12"], output: {bands: 6, sampleType: "FLOAT32"}}; 
            }
            function evaluatePixel(sample) { 
                return [sample.B02, sample.B03, sample.B04, sample.B08, sample.B11, sample.B12]; 
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