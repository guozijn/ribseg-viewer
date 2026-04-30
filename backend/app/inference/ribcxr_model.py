import numpy as np
import torch
from albumentations import Compose, Resize
from albumentations.pytorch import ToTensorV2

from cvcore.config import get_cfg_defaults
from cvcore.model import build_model
from .labels import RIB_LABELS
from .postprocessing import build_masks, merged_mask, make_overlay, compute_areas


def _load_state_dict(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError:
        pass
    stripped = {
        k.replace("module.", "", 1) if k.startswith("module.") else k: v
        for k, v in state_dict.items()
    }
    model.load_state_dict(stripped)


class RibCXRSegmenter:
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device_str: str = "cuda",
        default_threshold: float = 0.5,
    ) -> None:

        if device_str == "cuda" and not torch.cuda.is_available():
            device_str = "cpu"
        self.device = torch.device(device_str)
        self.default_threshold = default_threshold
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path

        cfg = get_cfg_defaults()
        cfg.merge_from_file(config_path)

        self.model = build_model(cfg).to(self.device)
        _load_state_dict(self.model, checkpoint_path, self.device)
        self.model.eval()

        self._transform = Compose([Resize(512, 512), ToTensorV2()])

    def predict(
        self,
        image_np: np.ndarray,
        model_input: np.ndarray,
        threshold: float | None = None,
    ) -> dict:
        thr = threshold if threshold is not None else self.default_threshold
        h, w = image_np.shape[:2]

        tensor = (
            self._transform(image=model_input)["image"]
            .unsqueeze(0)
            .to(device=self.device, dtype=torch.float)
        )

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits)
            raw_masks = (probs > thr).squeeze(0).cpu().numpy().astype(np.uint8)

        masks = build_masks(raw_masks, h, w)
        overlay = make_overlay(image_np, masks)
        combined_mask = merged_mask(masks)
        areas = compute_areas(masks)

        return {
            "masks": masks,
            "overlay": overlay,
            "merged_mask": combined_mask,
            "areas": areas,
            "labels": RIB_LABELS,
            "threshold": thr,
            "original_height": h,
            "original_width": w,
        }
