import io

import numpy as np
from PIL import Image, UnidentifiedImageError

from backend.config import IMG_SIZE


class BadImageError(ValueError):
    pass


def load_image(file_bytes: bytes):
    """Return (batch [1, H, W, 3] float32 in 0-255, rgb_uint8 [H, W, 3] for the overlay)."""
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise BadImageError("The file is not a valid image. Upload a JPG or PNG MRI slice.") from exc
    img = img.resize((IMG_SIZE, IMG_SIZE))
    rgb = np.array(img, dtype=np.uint8)
    return rgb[None].astype("float32"), rgb
