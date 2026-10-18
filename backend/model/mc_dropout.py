"""Monte Carlo Dropout: run T stochastic passes and measure how much they disagree."""
import numpy as np
import tensorflow as tf

from backend.config import MC_PASSES

CHUNK = 32   # max images per forward call (same size evaluate already used successfully on your laptop)


def mc_predict(model: tf.keras.Model, batch: np.ndarray, passes: int = MC_PASSES):
    """Return (mean_probs [N, C], all_probs [T, N, C]).

    Each image is repeated `passes` times and the copies go through the network as ONE batch
    with training=True. Dropout draws an independent random mask for every copy, so this is
    the same as T separate stochastic passes, but much faster than T separate calls.
    """
    batch = np.asarray(batch, dtype="float32")
    n = batch.shape[0]
    repeated = np.repeat(batch, passes, axis=0)            # image0 x T, image1 x T, ...
    outs = [
        model(tf.convert_to_tensor(repeated[i:i + CHUNK]), training=True).numpy()
        for i in range(0, len(repeated), CHUNK)
    ]
    probs = np.concatenate(outs).reshape(n, passes, -1)     # [N, T, C]
    all_probs = probs.transpose(1, 0, 2)                    # [T, N, C]
    return all_probs.mean(axis=0), all_probs


def normalized_entropy(mean_probs: np.ndarray) -> np.ndarray:
    """Predictive entropy of the mean prediction, scaled to [0, 1] (0 = sure, 1 = clueless)."""
    p = np.clip(mean_probs, 1e-12, 1.0)
    ent = -(p * np.log(p)).sum(axis=-1)
    return ent / np.log(p.shape[-1])
