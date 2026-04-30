import io

import numpy as np
import pytest
from PIL import Image

from app.inference.preprocessing import load_image


def _make_png(mode: str, size: tuple[int, int] = (64, 48)) -> bytes:
    img = Image.new(mode, size, color=128 if mode == "L" else (100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_grayscale_image_shape():
    png = _make_png("L", (64, 48))
    image_np, model_input = load_image(png)
    assert image_np.shape == (48, 64)
    assert model_input.shape == (48, 64, 1)


def test_rgb_image_converted_to_gray():
    png = _make_png("RGB", (32, 32))
    image_np, model_input = load_image(png)
    assert image_np.ndim == 2
    assert model_input.shape == (32, 32, 1)


def test_model_input_normalized():
    png = _make_png("L", (16, 16))
    _, model_input = load_image(png)
    assert model_input.min() >= 0.0
    assert model_input.max() <= 1.0


def test_image_np_is_float32():
    png = _make_png("L", (8, 8))
    image_np, model_input = load_image(png)
    assert image_np.dtype == np.float32
    assert model_input.dtype == np.float32
