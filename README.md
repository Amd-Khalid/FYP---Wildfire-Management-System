# FEWMS - FarmEvo Wildfire Management System

**A real-time AI-powered wildfire detection, analysis, and prediction platform leveraging satellite imagery and deep learning.**

---

## Overview

FEWMS is an advanced wildfire management system that combines satellite data from Copernicus Sentinel Hub with state-of-the-art deep learning models to provide comprehensive wildfire analytics. The system enables users to analyze historical wildfire events, assess burn severity, predict fire spread risk, and evaluate infrastructure impact—all through an intuitive web interface.

### Key Capabilities

- **Burn Scar Detection**: AI-powered identification of burned areas using multi-spectral satellite imagery
- **Severity Analysis**: Dynamic burn severity mapping using dNBR (differenced Normalized Burn Ratio)
- **Fire Spread Prediction**: Machine learning-based risk assessment for fire propagation
- **Infrastructure Impact**: Automated evaluation of buildings and roads at risk
- **Before/After Visualization**: Temporal comparison of pre-fire and post-fire imagery
- **Environmental Metrics**: CO2 emissions estimation and burned area quantification

---

## Technology Stack

### Backend
- **FastAPI** - High-performance Python web framework
- **PyTorch** - Deep learning framework for model inference
- **Segmentation Models PyTorch** - ResNet34-UNet architecture
- **NumPy & SciPy** - Numerical computation and image processing
- **Rasterio** - Geospatial raster data handling

### Frontend
- **Leaflet.js** - Interactive mapping library
- **Leaflet.Draw** - Drawing tools for region selection
- **Vanilla JavaScript** - Lightweight, responsive UI

### Data Sources
- **Copernicus Sentinel-2** - Multispectral optical imagery (13 bands)
- **Copernicus Sentinel-1** - Synthetic Aperture Radar (SAR) data
- **OpenStreetMap via Overpass API** - Infrastructure data (buildings, roads)

### AI Models
- **Detection Model**: ResNet34-UNet trained with Focal Loss (14-channel input: S2 + S1)
- **Prediction Model**: U-Net architecture for fire spread risk assessment

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (HTML/JS)                       │
│  • Interactive map with region selection                        │
│  • Before/after imagery comparison                              │
│  • Severity visualization & analytics dashboard                 │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP/REST API
┌───────────────────────────▼─────────────────────────────────────┐
│                      FastAPI Backend                             │
│  • Request handling & validation                                │
│  • Satellite data retrieval orchestration                       │
│  • Model inference coordination                                 │
│  • Infrastructure risk assessment                               │
└───────┬────────────────────┬────────────────────┬───────────────┘
        │                    │                    │
        ▼                    ▼                    ▼
┌──────────────┐   ┌──────────────────┐   ┌─────────────────────┐
│  Sentinel    │   │   AI Models      │   │  OpenStreetMap      │
│  Hub API     │   │  - Detection     │   │  Overpass API       │
│  (S2 + S1)   │   │  - Prediction    │   │  (Infrastructure)   │
└──────────────┘   └──────────────────┘   └─────────────────────┘
```

### Processing Pipeline

1. **Data Acquisition**: Fetches Sentinel-2 (optical) and Sentinel-1 (SAR) imagery for user-defined region and time range
2. **Preprocessing**: Applies cloud masking, speckle filtering, and spectral normalization
3. **Detection**: ResNet34-UNet model processes 14-channel input to identify burned areas
4. **Post-Processing**: Removes water bodies (NDWI), filters small artifacts, applies morphological operations
5. **Severity Analysis**: Calculates dNBR from pre/post-fire NBR indices
6. **Prediction**: U-Net model generates fire spread risk heatmap
7. **Infrastructure Analysis**: Queries OSM data and performs spatial intersection with burn mask
8. **Visualization**: Generates colormapped overlays and returns base64-encoded images

---

## Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional, for faster inference)

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/FYP---Wildfire-Management-System.git
   cd FYP---Wildfire-Management-System/website
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   ```

3. **Activate virtual environment**
   - Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - Linux/Mac:
     ```bash
     source .venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Configure Sentinel Hub credentials**
   
   Edit `config.py` with your Copernicus Data Space credentials:
   ```python
   SENTINEL_CLIENT_ID = "your-client-id"
   SENTINEL_CLIENT_SECRET = "your-client-secret"
   ```
   
   *Register at: https://dataspace.copernicus.eu/*

6. **Ensure model weights are present**
   
   Place trained models in `website/model/`:
   - `best_resnet34_unet.pth` (Detection model)
   - `best_unet_prediction.pth` (Prediction model)

---

## Usage

### Starting the Server

```bash
cd website
uvicorn api.main:app --reload
```

The API will be available at `http://localhost:8000`

### Using the Web Interface

1. Open `frontend/index.html` in a web browser
2. **Select a preset fire** or **draw a custom region** on the map
3. **Set date range** for analysis period
4. Click **Run Full Analysis**
5. View results across multiple tabs:
   - **Map Overlay**: Detected burn areas
   - **Severity**: Burn intensity classification
   - **Before/After**: Temporal comparison
   - **Spread Risk**: Predicted fire propagation zones

### Features

- **Interactive Map**: Before/After overlay toggle with opacity control
- **Image Popup**: Click any sidebar image to enlarge
- **Reset**: Clear analysis and start over
- **Preset Fires**: Quick access to notable wildfire events (Glass Fire, Creek Fire, Lahaina, Rhodes)

---

## API Documentation

### Endpoint: `/predict_with_bbox`

**Method**: `POST`

**Request Body**:
```json
{
  "min_lon": -122.58,
  "min_lat": 38.54,
  "max_lon": -122.50,
  "max_lat": 38.62,
  "date_from": "2020-10-10",
  "date_to": "2020-10-20"
}
```

**Response**:
```json
{
  "burned_area_ha": 1245.67,
  "co2_tonnes": 31141.75,
  "infrastructure": {
    "buildings_risk": 23,
    "roads_risk": 8
  },
  "severity_breakdown": {
    "high": 0.35,
    "moderate": 0.45,
    "low": 0.20
  },
  "overlay_base64": "iVBORw0KGgoAAAANS...",
  "rgb_base64": "iVBORw0KGgoAAAANS...",
  "pre_fire_base64": "iVBORw0KGgoAAAANS...",
  "severity_base64": "iVBORw0KGgoAAAANS...",
  "predict_risk_base64": "iVBORw0KGgoAAAANS...",
  "status": "Success"
}
```

**Error Responses**:
- `404`: No satellite data available for specified region/time
- `500`: Satellite fetch error or processing failure

---

## Model Details

### Detection Model (ResNet34-UNet)
- **Architecture**: Encoder-decoder with skip connections
- **Input**: 14 channels (S2: B01-B12 + SCL, S1: VV + VH)
- **Output**: Binary burn mask + confidence scores
- **Training**: Focal Loss for class imbalance, 512×512 patches
- **Preprocessing**: Cloud masking, Lee speckle filtering, percentile normalization

### Prediction Model (U-Net)
- **Architecture**: Standard U-Net
- **Input**: Temporal features + environmental factors
- **Output**: Fire spread risk probability map
- **Postprocessing**: 0.40 threshold, edge cropping to remove artifacts

### Performance Optimizations
- Confidence threshold: 0.30 (tuned for Focal Loss)
- Water body removal using NDWI > 0.0
- Morphological closing (2×2 kernel) + blob filtering (min 20 pixels)
- Dynamic severity scaling using 5th-95th percentile

---

## Metrics & Analysis

### Environmental Impact
- **Burned Area**: Calculated using haversine formula with WGS84 ellipsoid
- **CO2 Emissions**: Estimated at 25 tonnes/hectare (typical forest biomass)

### Severity Classification
Based on dNBR (Differenced Normalized Burn Ratio):
- **Low**: dNBR ≤ 0.25
- **Moderate**: 0.25 < dNBR ≤ 0.60
- **High**: dNBR > 0.60

### Infrastructure Risk
Spatial intersection of burn mask with:
- OSM building footprints (`building=*`)
- OSM highway segments (`highway=*`)

---

## Configuration

### Adjustable Parameters (in `api/main.py`)

```python
# Detection confidence threshold
res['det_confidence'] > 0.30

# Water masking threshold
is_water = ndwi > 0.0

# Minimum blob size (pixels)
filter_small_blobs(burn_mask, min_size=20)

# Pre-fire baseline window
pre_date_from = date_from - 30 days

# Prediction risk visibility threshold
risk = np.where(raw_risk > 0.40, raw_risk, np.nan)
```

---

## Project Structure

```
FYP---Wildfire-Management-System/
├── website/
│   ├── api/
│   │   └── main.py              # FastAPI backend
│   ├── frontend/
│   │   └── index.html           # Web interface
│   ├── model/
│   │   ├── burn_scar_inference.py   # AI inference engine
│   │   ├── unet_inference.py        # Prediction model
│   │   ├── best_resnet34_unet.pth   # Detection weights
│   │   └── best_unet_prediction.pth # Prediction weights
│   ├── config.py                # Sentinel Hub credentials
│   └── requirements.txt         # Python dependencies
└── README.md
```

---

## Contributing

Contributions are welcome! Please ensure:
- Code follows PEP 8 style guidelines
- Models are documented with architecture details
- API changes include updated documentation
- Frontend changes are cross-browser compatible

---

## License

This project is developed as part of a Final Year Project at Habib University.

---

## Acknowledgments

- **Copernicus EU** - Sentinel satellite data access
- **Segmentation Models PyTorch** - Pre-built U-Net architectures
- **OpenStreetMap Contributors** - Infrastructure data
- **Leaflet.js** - Mapping framework

---

## Contact

For questions or support, please contact the development team at Habib University.

---

**Built by the FarmEvo Team**
