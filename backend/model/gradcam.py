"""Grad-CAM on the last convolutional feature map of the ResNet50V2 backbone."""
import base64

import cv2
import numpy as np
import tensorflow as tf

from backend.model.build import BACKBONE_NAME


def make_heatmap(model: tf.keras.Model, image: np.ndarray, class_index: int) -> np.ndarray:
    """image: float32 array [1, H, W, 3] in 0-255. Returns heatmap [h, w] in 0-1."""
    names = [l.name for l in model.layers]
    idx = names.index(BACKBONE_NAME)
    pre = model.layers[1:idx]       # skip InputLayer, keep rescale
    base = model.layers[idx]
    head = model.layers[idx + 1:]

    x = tf.convert_to_tensor(image, dtype=tf.float32)
    with tf.GradientTape() as tape:
        for layer in pre:
            x = layer(x)
        conv = base(x, training=False)
        tape.watch(conv)
        y = conv
        for layer in head:
            y = layer(y, training=False)       # Dropout off: a stable, single explanation
        score = y[:, class_index]
    grads = tape.gradient(score, conv)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.reduce_sum(weights * conv, axis=-1)[0]
    cam = tf.nn.relu(cam).numpy()
    return cam / (cam.max() + 1e-8)


def overlay_png_base64(heatmap: np.ndarray, rgb_uint8: np.ndarray, alpha: float = 0.4) -> str:
    h, w = rgb_uint8.shape[:2]
    cam = cv2.resize(heatmap, (w, h))
    colored = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)   # BGR
    base_bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    blended = cv2.addWeighted(colored, alpha, base_bgr, 1 - alpha, 0)
    ok, buf = cv2.imencode(".png", blended)
    if not ok:
        raise RuntimeError("Could not encode heatmap")
    return base64.b64encode(buf.tobytes()).decode("ascii")
