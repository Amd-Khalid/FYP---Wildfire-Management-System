import io
from typing import Tuple

import numpy as np
import torch
import rasterio
import segmentation_models_pytorch as smp
from torchvision.utils import save_image


class BurnScarUNetInference:
    def __init__(self, weights_path: str, device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=8,
            classes=2,
            activation=None
        ).to(self.device)

        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @staticmethod
    def _load_scene_from_tif(tif_bytes: bytes) -> np.ndarray:
        with rasterio.io.MemoryFile(tif_bytes) as memfile:
            with memfile.open() as src:
                img = src.read().astype(np.float32)
        return img

    @staticmethod
    def _compute_indices(img: np.ndarray) -> np.ndarray:
        blue, green, red, nir, swir1, swir2 = img
        ndvi = (nir - red) / (nir + red + 1e-8)
        nbr = (nir - swir1) / (nir + swir1 + 1e-8)
        img_with_indices = np.stack(
            [blue, green, red, nir, swir1, swir2, ndvi, nbr],
            axis=0
        )
        return img_with_indices

    @staticmethod
    def _normalize(img_8ch: np.ndarray) -> np.ndarray:
        img_norm = img_8ch / 3000.0
        img_norm = np.clip(img_norm, 0.0, 1.0)
        return img_norm

    @torch.no_grad()
    def predict_mask(self, tif_bytes: bytes) -> Tuple[np.ndarray, np.ndarray]:
        img = self._load_scene_from_tif(tif_bytes)
        img = self._compute_indices(img)
        img = self._normalize(img)

        tensor = torch.from_numpy(img).unsqueeze(0).float()
        tensor = tensor.to(self.device)

        outputs = self.model(tensor)
        probs = torch.softmax(outputs, dim=1)[:, 1, :, :]

        preds = torch.argmax(outputs, dim=1)
        mask = preds.squeeze(0).cpu().numpy().astype(np.uint8)
        prob_map = probs.squeeze(0).cpu().numpy().astype(np.float32)

        return mask, prob_map

    @staticmethod
    def mask_to_png_bytes(mask: np.ndarray) -> bytes:
        tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        tensor = tensor / tensor.max().clamp(min=1.0)

        buf = io.BytesIO()
        save_image(tensor, buf, format="PNG")
        buf.seek(0)
        return buf.read()
