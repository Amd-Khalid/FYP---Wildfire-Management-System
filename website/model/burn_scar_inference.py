"""
Burn Scar AI: Integrated Detection & Prediction Engine (14-Channel + Spread Risk)
"""
import torch
import torch.nn as nn
import numpy as np
import rasterio
from rasterio.io import MemoryFile
import requests
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from scipy import ndimage

# =====================================================================
# CUSTOM PREPROCESSING (Speckle & Clouds)
# =====================================================================
class SpeckleFilter:
    @staticmethod
    def lee_filter(image: np.ndarray, window_size: int = 7) -> np.ndarray:
        if window_size % 2 == 0:
            window_size += 1
        valid_mask = image > 0
        result = image.copy()
        kernel = np.ones((window_size, window_size)) / (window_size ** 2)
        local_mean = ndimage.convolve(image.astype(np.float64), kernel, mode='reflect')
        local_sq_mean = ndimage.convolve((image.astype(np.float64) ** 2), kernel, mode='reflect')
        local_var = np.maximum(local_sq_mean - local_mean ** 2, 0)
        valid_pixels = image[valid_mask]
        noise_var = (np.mean(valid_pixels) ** 2) * 0.25 if len(valid_pixels) > 0 else 1.0
        weight = np.clip(local_var / (local_var + noise_var + 1e-10), 0, 1)
        result = local_mean + weight * (image - local_mean)
        result[~valid_mask] = 0
        return result.astype(np.float32)

class CloudMasker:
    DEFAULT_MASK_CLASSES = {
        0: True, 1: True, 2: False, 3: True, 4: False, 5: False, 
        6: False, 7: False, 8: True, 9: True, 10: True, 11: False
    }
    def create_mask(self, scl_band: np.ndarray) -> np.ndarray:
        valid_mask = np.ones(scl_band.shape, dtype=bool)
        for class_value, should_mask in self.DEFAULT_MASK_CLASSES.items():
            if should_mask:
                valid_mask &= (scl_band != class_value)
        return valid_mask

# =====================================================================
# INFERENCE ENGINE (Detection & Prediction)
# =====================================================================
class BurnScarInference:
    def __init__(self, detection_path: str, prediction_path: Optional[str] = None, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 1. DETECTION MODEL (Our New 14-Channel Model)
        self.detection_model = smp.Unet(
            encoder_name="resnet34", encoder_weights=None,
            in_channels=14, classes=1, activation=None
        ).to(self.device)
        
        checkpoint = torch.load(detection_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint) 
        clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.detection_model.load_state_dict(clean_state_dict)
        self.detection_model.eval()
        print(f"✓ Detection Model loaded from {detection_path}")

        # 2. PREDICTION MODEL (Teammate's Enhanced 11-channel U-Net)
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
            weather_url = f"https://archive-api.open-meteo.com/v1/archive?latitude={mid_lat}&longitude={mid_lon}&start_date={date}&end_date={date}&daily=windspeed_10m_max,temperature_2m_max&timezone=auto"
            w_resp = requests.get(weather_url, timeout=5).json()
            wind = w_resp['daily']['windspeed_10m_max'][0] or 12.0
            
            elev_url = f"https://api.open-elevation.com/api/v1/lookup?locations={mid_lat},{mid_lon}"
            e_resp = requests.get(elev_url, timeout=5).json()
            elevation = e_resp['results'][0]['elevation']
        except Exception as e:
            print(f"! Driver Fetch Error: {e}. Using defaults.")
            wind, elevation = 12.0, 500.0

        return {'wind': wind, 'elevation': elevation}

    def preprocess_detection(self, s2_bands: np.ndarray, s1_bands: np.ndarray) -> torch.Tensor:
        """Our 14-Channel Logic"""
        optical = s2_bands[:10]
        scl_band = s2_bands[10]
        
        cloud_masker = CloudMasker()
        valid_mask = cloud_masker.create_mask(scl_band)
        optical[:, ~valid_mask] = 0
        
        bands_scaled = optical 
        red = bands_scaled[2]   
        nir = bands_scaled[3]   
        swir = bands_scaled[9]  
        
        epsilon = 1e-8
        ndvi = (nir - red) / (nir + red + epsilon)
        nbr = (nir - swir) / (nir + swir + epsilon)
        indices_data = np.stack([ndvi, nbr], axis=0)
        
        processed_s1 = np.zeros_like(s1_bands)
        if np.max(s1_bands) > 0:
            for i in range(2):
                filtered = SpeckleFilter.lee_filter(s1_bands[i], window_size=7)
                db_data = np.where(filtered > 0, 10 * np.log10((filtered**2) + 1e-10), -50)
                db_data = np.clip(db_data, -30, 0)
                processed_s1[i] = (db_data + 30) / 30
        
        img_14ch = np.concatenate([bands_scaled, processed_s1, indices_data], axis=0)
        return torch.from_numpy(img_14ch).float().unsqueeze(0)

    def preprocess_prediction(self, s2_bands: np.ndarray, extra: Dict) -> torch.Tensor:
        """Teammate's 11-Channel Logic (Extracts 6 bands from our 11-band fetch)"""
        # Our fetch order: B02(0), B03(1), B04(2), B08(3), B05(4), B06(5), B07(6), B8A(7), B11(8), B12(9), SCL(10)
        blue, green, red = s2_bands[0], s2_bands[1], s2_bands[2]
        nir, swir1, swir2 = s2_bands[3], s2_bands[8], s2_bands[9]
        
        ndvi = (nir - red) / (nir + red + 1e-8)
        nbr = (nir - swir1) / (nir + swir1 + 1e-8)
        base_stack = [blue, green, red, nir, swir1, swir2, ndvi, nbr]

        h, w = s2_bands.shape[1], s2_bands.shape[2]
        elev_map = np.full((h, w), extra['elevation'] / 5000.0)
        wind_map = np.full((h, w), extra['wind'] / 50.0)
        
        dy, dx = np.gradient(np.full((h, w), extra['elevation']))
        slope_map = np.clip(np.sqrt(dx**2 + dy**2) / 10.0, 0, 1)
        
        base_stack.extend([elev_map, wind_map, slope_map])
        img_stack = np.stack(base_stack, axis=0)
        img_tensor = torch.from_numpy(img_stack).float()
        img_tensor = torch.clamp(img_tensor, 0, 1) 
            
        return img_tensor.unsqueeze(0)
    
    def predict(self, s2_bands: np.ndarray, s1_bands: np.ndarray, bbox: list = None, date: str = None) -> Dict:
        """Runs both models cleanly."""
        results = {}

        # 1. Detection Mode (14-channel ResNet-34)
        det_tensor = self.preprocess_detection(s2_bands, s1_bands).to(self.device)
        with torch.no_grad():
            det_out = self.detection_model(det_tensor)
            
        probs = torch.sigmoid(det_out)[0, 0].cpu().numpy()
        results['detection'] = (probs > 0.25).astype(np.uint8) 
        results['det_confidence'] = probs 

        # 2. Prediction Mode (11-channel)
        if bbox and date:
            extra = self.get_external_drivers(bbox, date)
            pred_tensor = self.preprocess_prediction(s2_bands, extra).to(self.device)
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

# =====================================================================
# SENTINEL HUB API
# =====================================================================
class SentinelHubAPI:
    def __init__(self, client_id: str = None, client_secret: str = None):
        try:
            from config import SENTINEL_CLIENT_ID, SENTINEL_CLIENT_SECRET, SENTINEL_TOKEN_URL, SENTINEL_BASE_URL
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
        payload = {'grant_type': 'client_credentials', 'client_id': self.client_id, 'client_secret': self.client_secret}
        response = requests.post(self.token_url, data=payload)
        response.raise_for_status()
        self.access_token = response.json()['access_token']
        print("✓ Authenticated with Sentinel Hub")

    def _get_dynamic_dimensions(self, bbox: list) -> Tuple[int, int]:
        """Teammate's math: calculates dimensions divisible by 32 to prevent U-Net crashes."""
        width_deg = abs(bbox[2] - bbox[0])
        height_deg = abs(bbox[3] - bbox[1])
        aspect_ratio = height_deg / width_deg
        width = 512
        raw_height = int(512 * aspect_ratio)
        height = max(32, round(raw_height / 32) * 32)
        return width, height
    
    def download_imagery(self, bbox: list, date_from: str, date_to: str, max_cloud_cover: float = 20.0) -> np.ndarray:
        width, height = self._get_dynamic_dimensions(bbox)
        headers = {'Authorization': f'Bearer {self.access_token}', 'Content-Type': 'application/json'}
        payload = {
            "input": {
                "bounds": {"bbox": bbox, "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
                "data": [{"type": "sentinel-2-l2a", "dataFilter": {"timeRange": {"from": f"{date_from}T00:00:00Z", "to": f"{date_to}T23:59:59Z"}, "maxCloudCoverage": max_cloud_cover}}]
            },
            "output": {"width": width, "height": height, "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
            "evalscript": """
                //VERSION=3
                function setup() { return {input: ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "SCL"], output: {bands: 11, sampleType: "FLOAT32"}}; }
                function evaluatePixel(sample) { return [sample.B02, sample.B03, sample.B04, sample.B08, sample.B05, sample.B06, sample.B07, sample.B8A, sample.B11, sample.B12, sample.SCL]; }
            """
        }
        
        response = requests.post(self.process_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        with MemoryFile(response.content) as memfile:
            with memfile.open() as dataset: bands = dataset.read()
        print(f"✓ Downloaded Sentinel-2 imagery: {bands.shape}")
        return bands

    def download_s1_imagery(self, bbox: list, date_from: str, date_to: str) -> np.ndarray:
        width, height = self._get_dynamic_dimensions(bbox)
        headers = {'Authorization': f'Bearer {self.access_token}', 'Content-Type': 'application/json'}
        payload = {
            "input": {
                "bounds": {"bbox": bbox, "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
                "data": [{"type": "sentinel-1-grd", "dataFilter": {"timeRange": {"from": f"{date_from}T00:00:00Z", "to": f"{date_to}T23:59:59Z"}, "acquisitionMode": "IW", "polarization": "DV"}}]
            },
            "output": {"width": width, "height": height, "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
            "evalscript": """
                //VERSION=3
                function setup() { return {input: ["VV", "VH"], output: {bands: 2, sampleType: "FLOAT32"}}; }
                function evaluatePixel(sample) { return [sample.VV, sample.VH]; }
            """
        }
        
        response = requests.post(self.process_url, headers=headers, json=payload, timeout=60)
        try:
            response.raise_for_status()
            with MemoryFile(response.content) as memfile:
                with memfile.open() as dataset: s1_bands = dataset.read()
            print(f"✓ Downloaded Sentinel-1 imagery: {s1_bands.shape}")
            return s1_bands
        except:
            print("⚠ Sentinel-1 data unavailable. Returning zeros.")
            return np.zeros((2, height, width), dtype=np.float32)