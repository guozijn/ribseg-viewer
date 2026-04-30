import numpy as np
from PIL import Image


def load_image(file_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Return (original_uint8_gray, model_input_float32_hwc1)."""
    image = Image.open(__import__("io").BytesIO(file_bytes)).convert("L")
    image_np = np.asarray(image, dtype=np.float32)
    model_input = np.expand_dims(image_np / 255.0, axis=2)
    return image_np, model_input
