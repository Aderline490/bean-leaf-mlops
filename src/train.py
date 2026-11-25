"""Train the first model version. Run from the project root: python -m src.train"""
from __future__ import annotations

import json
import time
from pathlib import Path

import tensorflow as tf

from src.model import SEED, build_model, evaluate_model, fit_two_stage, list_images, make_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
MODEL_PATH = MODELS / "bean_leaf_model.keras"


def main() -> None:
    tf.keras.utils.set_random_seed(SEED)
    MODELS.mkdir(exist_ok=True)

    train_paths, train_labels = list_images(DATA / "train")
    val_paths, val_labels = list_images(DATA / "validation")
    test_paths, test_labels = list_images(DATA / "test")
    print(f"images: train={len(train_paths)} validation={len(val_paths)} test={len(test_paths)}", flush=True)

    started = time.time()
    model = build_model()
    history = fit_two_stage(
        model,
        make_dataset(train_paths, train_labels, training=True),
        make_dataset(val_paths, val_labels),
        MODEL_PATH,
    )
    seconds = time.time() - started

    best = tf.keras.models.load_model(MODEL_PATH)
    metrics = {
        "version": 1,
        "training_seconds": round(seconds, 1),
        "epochs_run": len(history["loss"]),
        "validation": evaluate_model(best, val_paths, val_labels),
        "test": evaluate_model(best, test_paths, test_labels),
    }
    (MODELS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (MODELS / "history.json").write_text(json.dumps(history, indent=2))

    test = metrics["test"]
    print("TEST " + " ".join(f"{k}={test[k]:.4f}" for k in (
        "accuracy", "loss", "precision_macro", "recall_macro", "f1_macro", "roc_auc_ovr", "cohen_kappa"
    )), flush=True)
    print("confusion_matrix", test["confusion_matrix"], flush=True)


if __name__ == "__main__":
    main()
