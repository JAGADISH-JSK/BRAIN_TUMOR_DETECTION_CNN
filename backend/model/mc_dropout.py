"""Monte Carlo Dropout: run T stochastic passes and measure how much they disagree."""
import numpy as np
import tensorflow as tf

from backend.config import MC_PASSES


def mc_predict(model: tf.keras.Model, batch: np.ndarray, passes: int = MC_PASSES):
    """Return (mean_probs [N, C], all_probs [T, N, C]). training=True turns Dropout on."""
    x = tf.convert_to_tensor(batch, dtype=tf.float32)
    all_probs = np.stack([model(x, training=True).numpy() for _ in range(passes)])
    return all_probs.mean(axis=0), all_probs


def normalized_entropy(mean_probs: np.ndarray) -> np.ndarray:
    """Predictive entropy of the mean prediction, scaled to [0, 1] (0 = sure, 1 = clueless)."""
    p = np.clip(mean_probs, 1e-12, 1.0)
    ent = -(p * np.log(p)).sum(axis=-1)
    return ent / np.log(p.shape[-1])
