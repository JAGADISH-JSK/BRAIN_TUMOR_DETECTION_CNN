"""Shared settings. Change CLASS_NAMES to match your dataset.

Keras reads class folders in alphabetical order, so keep this list alphabetical
(training also writes weights/class_names.json, which the API prefers).
"""
import os

IMG_SIZE = 224
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = len(CLASS_NAMES)

DROPOUT_RATE = 0.4
MC_PASSES = 20                 # T forward passes with dropout ON

# Normalised entropy is in [0, 1]. This is only a placeholder:
# tune it on the VALIDATION set (see training/evaluate.py), never on the test set.
ENTROPY_THRESHOLD = float(os.environ.get("ENTROPY_THRESHOLD", 0.35))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_PATH = os.path.join(ROOT, "weights", "brain_tumor_model.keras")
CLASS_NAMES_PATH = os.path.join(ROOT, "weights", "class_names.json")
