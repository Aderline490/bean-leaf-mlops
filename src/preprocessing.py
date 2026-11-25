"""Image preprocessing shared by training, prediction and retraining.

Every image goes through the same steps so the model sees identical input
in the notebook, in the API and during retraining:

1. decode the file and apply its EXIF rotation (phone photos often carry one)
2. convert to 3 channel RGB
3. resize to 224 x 224
4. return floats in the 0 to 255 range

Pixel scaling to the -1 to 1 range happens inside the model, so a saved model
can never be served with the wrong scaling.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable, List, Tuple, Union

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (*IMAGE_SIZE, 3)
MIN_SIDE = 32
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

CLASSES = ["angular_leaf_spot", "bean_rust", "healthy"]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASSES)}

ImageSource = Union[str, Path, bytes]


class ImageError(ValueError):
    """Raised when a file cannot be used as a leaf image."""


def load_image(source: ImageSource) -> Image.Image:
    """Decode a file path or raw bytes into an upright RGB image."""
    if isinstance(source, (bytes, bytearray)) and len(source) > MAX_UPLOAD_BYTES:
        raise ImageError("Image is larger than the 10 MB limit.")

    handle = io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source
    try:
        with Image.open(handle) as image:
            image = ImageOps.exif_transpose(image)
            rgb = image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ImageError(f"Could not read image: {exc}") from exc

    if min(rgb.size) < MIN_SIDE:
        raise ImageError(f"Image is too small. Each side must be at least {MIN_SIDE} pixels.")
    return rgb


def preprocess(source: ImageSource) -> np.ndarray:
    """Run the full pipeline on one file path or raw bytes."""
    image = load_image(source).resize(IMAGE_SIZE, Image.BILINEAR)
    return np.asarray(image, dtype="float32")


def preprocess_many(paths: Iterable[Path]) -> Tuple[np.ndarray, List[Path], List[dict]]:
    """Preprocess several files, returning features, kept paths and skipped files."""
    features, kept, skipped = [], [], []
    for path in paths:
        try:
            features.append(preprocess(path))
            kept.append(Path(path))
        except ImageError as exc:
            skipped.append({"file": str(path), "reason": str(exc)})
    empty = np.empty((0, *INPUT_SHAPE), dtype="float32")
    return (np.stack(features) if features else empty), kept, skipped
