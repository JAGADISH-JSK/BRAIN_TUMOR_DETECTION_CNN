"""Two-stage training: (1) train the head with a frozen backbone, (2) fine-tune the top layers.

Expected data layout (Kaggle 4-class style):
    data/Training/<class_name>/*.jpg
    data/Testing/<class_name>/*.jpg
Run from the repo root:
    python -m training.train --data-dir data --epochs-head 10 --epochs-ft 10
"""
import argparse
import json
import os

import tensorflow as tf

from backend import config
from backend.model.build import build_model, unfreeze_top

SEED = 42


def make_datasets(data_dir: str, batch_size: int):
    common = dict(
        image_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=batch_size,
        label_mode="categorical",
        seed=SEED,
    )
    train_dir = os.path.join(data_dir, "Training")
    # Validation is carved out of the TRAINING folder. The Testing folder stays untouched.
    train = tf.keras.utils.image_dataset_from_directory(
        train_dir, validation_split=0.15, subset="training", **common)
    val = tf.keras.utils.image_dataset_from_directory(
        train_dir, validation_split=0.15, subset="validation", **common)
    class_names = train.class_names

    # Augmentation is applied to the training set ONLY (never to val/test).
    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
    ])
    train = train.map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    return train.prefetch(tf.data.AUTOTUNE), val.prefetch(tf.data.AUTOTUNE), class_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs-head", type=int, default=10)
    ap.add_argument("--epochs-ft", type=int, default=10)
    ap.add_argument("--unfreeze", type=int, default=30, help="backbone layers to unfreeze in stage 2")
    args = ap.parse_args()

    tf.keras.utils.set_random_seed(SEED)
    train, val, class_names = make_datasets(args.data_dir, args.batch_size)
    print("Classes:", class_names)

    os.makedirs(os.path.dirname(config.WEIGHTS_PATH), exist_ok=True)
    with open(config.CLASS_NAMES_PATH, "w") as f:
        json.dump(class_names, f)

    model = build_model(num_classes=len(class_names), freeze_base=True)
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        config.WEIGHTS_PATH, monitor="val_accuracy", save_best_only=True, verbose=1)
    stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=4, restore_best_weights=True)

    # Stage 1: head only
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train, validation_data=val, epochs=args.epochs_head, callbacks=[ckpt, stop])

    # Stage 2: fine-tune the top of the backbone with a small learning rate
    unfreeze_top(model, args.unfreeze)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train, validation_data=val, epochs=args.epochs_ft, callbacks=[ckpt, stop])

    print(f"Best model saved to {config.WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
