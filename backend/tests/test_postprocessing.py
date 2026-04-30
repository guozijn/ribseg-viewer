import numpy as np
import pytest

from app.inference.labels import RIB_LABELS
from app.inference.postprocessing import build_masks, compute_areas, make_overlay, merged_mask


def _fake_raw_masks(h: int = 512, w: int = 512) -> np.ndarray:
    masks = np.zeros((20, h, w), dtype=np.uint8)
    masks[0, 10:20, 10:20] = 1
    masks[10, 30:40, 30:40] = 1
    return masks


def test_build_masks_shape():
    raw = _fake_raw_masks(512, 512)
    resized = build_masks(raw, 256, 320)
    assert resized.shape == (20, 256, 320)


def test_build_masks_preserves_content():
    raw = _fake_raw_masks(512, 512)
    resized = build_masks(raw, 512, 512)
    assert np.array_equal(resized[0], raw[0])


def test_merged_mask_shape_and_dtype():
    raw = _fake_raw_masks(512, 512)
    masks = build_masks(raw, 100, 120)
    combined = merged_mask(masks)
    assert combined.shape == (100, 120)
    assert set(np.unique(combined)).issubset({0, 255})


def test_merged_mask_is_union():
    masks = np.zeros((20, 10, 10), dtype=np.uint8)
    masks[0, 0, 0] = 1
    masks[1, 9, 9] = 1
    combined = merged_mask(masks)
    assert combined[0, 0] == 255
    assert combined[9, 9] == 255
    assert combined[5, 5] == 0


def test_make_overlay_shape():
    image_np = np.zeros((64, 80), dtype=np.float32)
    masks = np.zeros((20, 64, 80), dtype=np.uint8)
    overlay = make_overlay(image_np, masks)
    assert overlay.shape == (64, 80, 3)


def test_compute_areas_labels():
    raw = _fake_raw_masks(512, 512)
    masks = build_masks(raw, 512, 512)
    areas = compute_areas(masks)
    assert list(areas.keys()) == RIB_LABELS
    assert areas["R1"] == 100
    assert areas["L1"] == 100
    assert areas["R2"] == 0
