import cv2
import numpy as np
from .labels import RIB_LABELS

MASK_COLORS = cv2.applyColorMap(
    np.linspace(0, 255, len(RIB_LABELS), dtype=np.uint8).reshape(-1, 1),
    cv2.COLORMAP_TURBO,
)[:, 0, ::-1]


def build_masks(raw_masks: np.ndarray, original_height: int, original_width: int) -> np.ndarray:
    """Resize model output masks (20, 512, 512) to original image dimensions."""
    resized = np.stack([
        cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
        for mask in raw_masks
    ])
    return resized


def merged_mask(masks: np.ndarray) -> np.ndarray:
    """Combine all rib masks into a single binary mask (uint8, 0/255)."""
    return (masks.any(axis=0).astype(np.uint8)) * 255


def make_overlay(image_np: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Produce an RGB overlay matching infer.py behavior."""
    overlay = cv2.cvtColor(image_np.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    alpha = 0.45
    for mask, color in zip(masks, MASK_COLORS):
        area = mask > 0
        overlay[area] = ((1 - alpha) * overlay[area] + alpha * color).astype(np.uint8)

    return overlay


def make_label_overlay(mask: np.ndarray, label_index: int) -> np.ndarray:
    """Create an RGBA overlay for one rib mask with transparent background."""
    color = MASK_COLORS[label_index]
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    area = mask > 0
    rgba[area, :3] = color
    rgba[area, 3] = 255
    return rgba


def compute_areas(masks: np.ndarray) -> dict[str, int]:
    return {label: int(mask.sum()) for label, mask in zip(RIB_LABELS, masks)}
