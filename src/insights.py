"""Measure visual features of the training images for the dashboard.

Soil is brown like a lesion, so each image is first reduced to its main leaf.
Features are then measured only inside that leaf:

* yellowing: share of the leaf that is yellow
* lesion coverage: share of the leaf covered by brown or dark spots
* surface specks: small bright marks per 1,000 leaf pixels

The leaf mask is approximate. Neighbouring leaves can be included and some soil
can slip through, so these numbers describe overall patterns, not single images.

Run from the project root: python -m src.insights
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PIL import Image
from scipy import ndimage

from src.config import BASE_MODEL_DIR, DATA_DIR
from src.preprocessing import ALLOWED_EXTENSIONS, CLASSES

SIZE = 256
MIN_LEAF_SHARE = 0.08
OUTPUT = BASE_MODEL_DIR / "insights.json"
FEATURES = {
    "yellowing_pct": {"title": "Leaf yellowing", "unit": "% of leaf", "max": 48},
    "lesion_pct": {"title": "Lesion coverage", "unit": "% of leaf", "max": 18},
    "speck_density": {"title": "Surface specks", "unit": "specks per 1,000 leaf pixels", "max": 15},
}
BINS = 12


def leaf_mask(hue: np.ndarray, sat: np.ndarray, val: np.ndarray) -> np.ndarray:
    """Find the plant region closest to the image centre and fill its holes."""
    plant = (hue >= 38) & (hue <= 170) & (sat >= 0.12) & (val >= 0.15)
    plant = ndimage.binary_opening(plant, iterations=2)
    closed = ndimage.binary_closing(plant, iterations=4)
    labels, count = ndimage.label(closed)
    if count == 0:
        return np.zeros_like(plant)
    sizes = ndimage.sum(closed, labels, index=np.arange(1, count + 1))
    yy, xx = np.ogrid[:SIZE, :SIZE]
    centre = np.exp(-(((yy - SIZE / 2) ** 2 + (xx - SIZE / 2) ** 2) / (2 * (SIZE / 3) ** 2)))
    weight = ndimage.sum(centre, labels, index=np.arange(1, count + 1))
    best = int(np.argmax(weight if weight.max() > 0 else sizes)) + 1
    mask = ndimage.binary_fill_holes(labels == best)
    return ndimage.binary_erosion(mask, iterations=3)


def measure(path: str) -> Optional[Dict[str, float]]:
    with Image.open(path) as image:
        hsv = np.asarray(image.convert("RGB").resize((SIZE, SIZE)).convert("HSV"), dtype="float32")
    hue, sat, val = hsv[..., 0] * 360 / 255, hsv[..., 1] / 255, hsv[..., 2] / 255
    mask = leaf_mask(hue, sat, val)
    leaf = int(mask.sum())
    if leaf < MIN_LEAF_SHARE * SIZE * SIZE:
        return None

    yellow = mask & (hue >= 38) & (hue < 68) & (sat >= 0.4) & (val >= 0.45)
    lesion = mask & (((hue < 38) | (hue > 300)) & (sat >= 0.15) & (val >= 0.12))
    lesion = ndimage.binary_opening(lesion, iterations=1)

    bright_detail = val - ndimage.grey_opening(val, size=(7, 7))
    specks = mask & (bright_detail >= 0.06) & (hue < 75)
    labels, count = ndimage.label(specks)
    sizes = ndimage.sum(specks, labels, index=np.arange(1, count + 1)) if count else np.array([])
    small_specks = int(((sizes >= 2) & (sizes <= 40)).sum())
    return {
        "yellowing_pct": 100 * float(yellow.sum()) / leaf,
        "lesion_pct": 100 * float(lesion.sum()) / leaf,
        "speck_density": 1000 * small_specks / leaf,
    }


def add_stories(features: dict) -> None:
    """Write each interpretation from the measured medians."""

    def median(key: str, name: str) -> float:
        return features[key]["classes"][name]["median"]

    y_a, y_r, y_h = (median("yellowing_pct", c) for c in CLASSES)
    l_a, l_r, l_h = (median("lesion_pct", c) for c in CLASSES)
    s_a, s_r, s_h = (median("speck_density", c) for c in CLASSES)
    features["yellowing_pct"]["story"] = (
        f"A typical angular leaf spot leaf is {y_a:.1f}% yellow, about {y_a / y_h:.1f} times a healthy leaf "
        f"({y_h:.1f}%). Bean rust stays close to healthy at {y_r:.1f}%, so yellowing points to angular leaf "
        "spot but cannot detect rust by itself."
    )
    features["lesion_pct"]["story"] = (
        f"Brown damage covers {l_a:.1f}% of a typical angular leaf spot leaf and {l_r:.1f}% of a bean rust "
        f"leaf, against {l_h:.1f}% for healthy leaves. The two diseases overlap heavily here, which helps "
        "explain why they are the hardest pair for the model to separate."
    )
    features["speck_density"]["story"] = (
        f"Healthy leaves have about {s_h:.1f} small specks per 1,000 leaf pixels, compared with {s_r:.1f} for "
        f"bean rust and {s_a:.1f} for angular leaf spot. Fine surface texture is the clearest of these three "
        "signals that a leaf is diseased, and it is the kind of detail a CNN can learn."
    )


def summarise(values_by_class: Dict[str, np.ndarray], maximum: float) -> dict:
    edges = np.linspace(0, maximum, BINS + 1)
    output = {"edges": [round(float(edge), 4) for edge in edges], "classes": {}}
    for name, values in values_by_class.items():
        counts, _ = np.histogram(np.clip(values, 0, maximum), bins=edges)
        output["classes"][name] = {
            "n": int(values.size),
            "median": round(float(np.median(values)), 3),
            "q1": round(float(np.percentile(values, 25)), 3),
            "q3": round(float(np.percentile(values, 75)), 3),
            "share_pct": [round(100 * float(c) / values.size, 2) for c in counts],
        }
    return output


def main() -> None:
    paths, labels = [], []
    for name in CLASSES:
        for path in sorted((DATA_DIR / "train" / name).iterdir()):
            if path.suffix.lower() in ALLOWED_EXTENSIONS:
                paths.append(str(path))
                labels.append(name)

    with ProcessPoolExecutor() as pool:
        measured = list(pool.map(measure, paths, chunksize=16))

    kept = [(label, row) for label, row in zip(labels, measured) if row is not None]
    output = {
        "images_total": len(paths),
        "images_measured": len(kept),
        "class_counts": {name: labels.count(name) for name in CLASSES},
        "features": {},
    }
    for key, meta in FEATURES.items():
        values = {name: np.array([row[key] for label, row in kept if label == name]) for name in CLASSES}
        output["features"][key] = {**meta, **summarise(values, meta["max"])}
    add_stories(output["features"])
    OUTPUT.write_text(json.dumps(output, indent=2))

    print("measured", len(kept), "of", len(paths))
    for key in FEATURES:
        print(key)
        for name in CLASSES:
            item = output["features"][key]["classes"][name]
            values = np.array([row[key] for label, row in kept if label == name])
            print(f"  {name:18} n={item['n']} median={item['median']} q1={item['q1']} q3={item['q3']} p95={np.percentile(values, 95):.3f} max={values.max():.3f}")


if __name__ == "__main__":
    main()
