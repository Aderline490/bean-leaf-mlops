"""Rank unseen test images by how safe they are to use in the video demo.

Keeps only correct predictions, then ranks by confidence and sharpness.
Run from the project root: python scripts/select_demo_images.py
"""
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA_DIR, ROOT  # noqa: E402
from src.prediction import Predictor  # noqa: E402
from src.preprocessing import CLASSES  # noqa: E402

OUTPUT = ROOT / "demo" / "candidates.csv"


def sharpness(path: Path) -> float:
    gray = np.asarray(Image.open(path).convert("L").resize((224, 224)), dtype="float32")
    laplacian = (
        -4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    )
    return float(laplacian.var())


def main() -> None:
    predictor = Predictor()
    predictor.load()
    rows = []
    for label in CLASSES:
        for path in sorted((DATA_DIR / "test" / label).glob("*.jpg")):
            result = predictor.predict(path)
            rows.append(
                {
                    "file": str(path.relative_to(ROOT)),
                    "true_label": label,
                    "predicted": result["label"],
                    "correct": result["label"] == label,
                    "confidence": result["confidence"],
                    "sharpness": round(sharpness(path), 1),
                }
            )

    OUTPUT.parent.mkdir(exist_ok=True)
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("correct", sum(r["correct"] for r in rows), "of", len(rows))
    for label in CLASSES:
        correct = [r for r in rows if r["true_label"] == label and r["correct"]]
        median = np.median([r["sharpness"] for r in correct])
        ranked = sorted((r for r in correct if r["sharpness"] >= median), key=lambda r: -r["confidence"])
        print(label)
        for row in ranked[:4]:
            print(" ", row["file"], row["confidence"], row["sharpness"])


if __name__ == "__main__":
    main()
