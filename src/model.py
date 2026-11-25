"""Model creation, training, fine-tuning and evaluation."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    roc_auc_score,
    recall_score,
)

from src.preprocessing import ALLOWED_EXTENSIONS, CLASSES, CLASS_TO_INDEX, INPUT_SHAPE, preprocess

BACKBONE_NAME = "backbone"
BATCH_SIZE = 32
SEED = 42


def list_images(split_dir: Path) -> Tuple[List[str], List[int]]:
    """List image paths and integer labels from class-named folders."""
    paths, labels = [], []
    for name in CLASSES:
        folder = Path(split_dir) / name
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS:
                paths.append(str(path))
                labels.append(CLASS_TO_INDEX[name])
    return paths, labels


def make_dataset(
    paths: Sequence[str], labels: Sequence[int], training: bool = False, batch_size: int = BATCH_SIZE
) -> tf.data.Dataset:
    """Build a tf.data pipeline that uses the shared preprocessing code."""

    def decode(path: bytes) -> np.ndarray:
        return preprocess(path.decode("utf-8"))

    def load(path, label):
        image = tf.numpy_function(decode, [path], tf.float32)
        image.set_shape(INPUT_SHAPE)
        return image, label

    dataset = tf.data.Dataset.from_tensor_slices((list(paths), np.asarray(labels, dtype="int32")))
    if training:
        dataset = dataset.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    return dataset.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_model(dropout: float = 0.3, l2: float = 1e-4) -> tf.keras.Model:
    """MobileNetV2 transfer-learning classifier with augmentation built in."""
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=SEED),
            tf.keras.layers.RandomRotation(0.15, seed=SEED),
            tf.keras.layers.RandomZoom(0.15, seed=SEED),
            tf.keras.layers.RandomContrast(0.15, seed=SEED),
        ],
        name="augmentation",
    )
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=INPUT_SHAPE, include_top=False, weights="imagenet", name=BACKBONE_NAME
    )
    backbone.trainable = False

    inputs = tf.keras.Input(shape=INPUT_SHAPE, name="image")
    x = augmentation(inputs)
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="scale")(x)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="pool")(x)
    x = tf.keras.layers.Dropout(dropout, name="dropout")(x)
    outputs = tf.keras.layers.Dense(
        len(CLASSES),
        activation="softmax",
        kernel_regularizer=tf.keras.regularizers.l2(l2),
        name="classifier",
    )(x)
    return tf.keras.Model(inputs, outputs, name="bean_leaf_classifier")


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def unfreeze_top(model: tf.keras.Model, layers: int = 30) -> None:
    """Unfreeze the top backbone layers, keeping BatchNorm statistics frozen."""
    backbone = model.get_layer(BACKBONE_NAME)
    backbone.trainable = True
    cutoff = len(backbone.layers) - layers
    for index, layer in enumerate(backbone.layers):
        layer.trainable = index >= cutoff and not isinstance(layer, tf.keras.layers.BatchNormalization)


def training_callbacks(checkpoint: Path, patience: int = 4, best_so_far: Optional[float] = None) -> list:
    """Callbacks for one stage. The checkpoint only saves a new overall best."""
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7),
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint), monitor="val_loss", save_best_only=True, initial_value_threshold=best_so_far
        ),
    ]


def fit_two_stage(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    checkpoint: Path,
    head_epochs: int = 12,
    fine_epochs: int = 15,
    class_weight: Optional[Dict[int, float]] = None,
) -> Dict[str, list]:
    """Train the head, then fine-tune. The best model overall is left at `checkpoint`."""
    compile_model(model, 1e-3)
    first = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=head_epochs,
        callbacks=training_callbacks(checkpoint),
        class_weight=class_weight,
        verbose=2,
    )
    unfreeze_top(model)
    compile_model(model, 1e-5)
    second = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=fine_epochs,
        callbacks=training_callbacks(checkpoint, best_so_far=min(first.history["val_loss"])),
        class_weight=class_weight,
        verbose=2,
    )
    return {key: first.history[key] + second.history[key] for key in first.history if key in second.history}


def evaluate_model(model: tf.keras.Model, paths: Sequence[str], labels: Sequence[int]) -> dict:
    """Compute the evaluation metrics on a labelled set of images."""
    y_true = np.asarray(labels)
    probabilities = model.predict(make_dataset(paths, labels), verbose=0)
    y_pred = probabilities.argmax(axis=1)
    class_ids = list(range(len(CLASSES)))
    return {
        "samples": int(y_true.size),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "loss": float(log_loss(y_true, probabilities, labels=class_ids)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc_ovr": float(roc_auc_score(y_true, probabilities, multi_class="ovr", labels=class_ids)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=class_ids).tolist(),
        "per_class": classification_report(
            y_true, y_pred, labels=class_ids, target_names=CLASSES, output_dict=True, zero_division=0
        ),
    }
