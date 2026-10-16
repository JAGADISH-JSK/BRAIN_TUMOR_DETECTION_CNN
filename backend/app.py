"""Flask API. Run from the repo root:  python -m backend.app"""
import json
import os

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from flask_cors import CORS

from backend import config
from backend.model.gradcam import make_heatmap, overlay_png_base64
from backend.model.mc_dropout import mc_predict, normalized_entropy
from backend.utils.preprocess import BadImageError, load_image

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024   # 8 MB upload limit

# Load ONCE at startup, not per request.
MODEL = None
CLASS_NAMES = config.CLASS_NAMES
if os.path.exists(config.WEIGHTS_PATH):
    MODEL = tf.keras.models.load_model(config.WEIGHTS_PATH)
    if os.path.exists(config.CLASS_NAMES_PATH):
        with open(config.CLASS_NAMES_PATH) as f:
            CLASS_NAMES = json.load(f)
else:
    print(f"[warn] No model at {config.WEIGHTS_PATH}. Train first (see README).")


@app.get("/health")
def health():
    return jsonify(status="ok", model_loaded=MODEL is not None, classes=CLASS_NAMES)


@app.post("/predict")
def predict():
    if MODEL is None:
        return jsonify(error="Model not loaded. Train the model first and place it in weights/."), 503
    file = request.files.get("image")
    if file is None:
        return jsonify(error="Missing file. Send the image in the 'image' form field."), 400
    try:
        batch, rgb = load_image(file.read())
    except BadImageError as exc:
        return jsonify(error=str(exc)), 400

    mean_probs, _ = mc_predict(MODEL, batch)
    probs = mean_probs[0]
    class_idx = int(np.argmax(probs))
    uncertainty = float(normalized_entropy(mean_probs)[0])
    needs_review = uncertainty > config.ENTROPY_THRESHOLD

    heatmap = make_heatmap(MODEL, batch, class_idx)
    return jsonify(
        label=CLASS_NAMES[class_idx],
        confidence=float(probs[class_idx]),
        uncertainty=uncertainty,
        threshold=config.ENTROPY_THRESHOLD,
        needs_review=needs_review,
        probabilities={n: float(p) for n, p in zip(CLASS_NAMES, probs)},
        heatmap=overlay_png_base64(heatmap, rgb),
    )


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="File too large. The limit is 8 MB."), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
