"""Evaluate the model AND the uncertainty ("is the failure-aware idea actually working?").

Run from the repo root:
    python -m training.evaluate --data-dir data

Outputs in results/: metrics.txt, confusion_matrix.png, coverage_accuracy.csv/.png,
uncertainty_hist.png. The entropy threshold is chosen on VALIDATION data, then applied to TEST.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from backend import config
from backend.model.mc_dropout import mc_predict, normalized_entropy

RESULTS = os.path.join(config.ROOT, "results")


def load_ds(folder: str, batch_size: int, **kw):
    return tf.keras.utils.image_dataset_from_directory(
        folder, image_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=batch_size, label_mode="int", **kw)


def run_mc(model, ds, passes):
    y_true, y_pred, ent = [], [], []
    for x, y in ds:
        mean, _ = mc_predict(model, x.numpy(), passes)
        y_true += y.numpy().tolist()
        y_pred += np.argmax(mean, axis=1).tolist()
        ent += normalized_entropy(mean).tolist()
    return np.array(y_true), np.array(y_pred), np.array(ent)


def coverage_table(y_true, y_pred, ent, thresholds):
    rows = []
    for t in thresholds:
        keep = ent <= t
        acc = float((y_true[keep] == y_pred[keep]).mean()) if keep.any() else float("nan")
        rows.append((float(t), float(keep.mean()), acc))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--passes", type=int, default=config.MC_PASSES)
    ap.add_argument("--target-accuracy", type=float, default=0.98,
                    help="pick the loosest threshold whose accepted-validation accuracy reaches this")
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    model = tf.keras.models.load_model(config.WEIGHTS_PATH)
    with open(config.CLASS_NAMES_PATH) as f:
        class_names = json.load(f)

    train_dir = os.path.join(args.data_dir, "Training")
    val_ds = load_ds(train_dir, args.batch_size, validation_split=0.15, subset="validation",
                     seed=42, shuffle=True)      # same seed/split as training
    test_ds = load_ds(os.path.join(args.data_dir, "Testing"), args.batch_size, shuffle=False)

    # 1) choose the threshold on VALIDATION
    vy, vp, ve = run_mc(model, val_ds, args.passes)
    grid = np.linspace(0.02, 1.0, 50)
    chosen = float(grid[0])
    for t, cov, acc in coverage_table(vy, vp, ve, grid):
        if cov > 0 and acc >= args.target_accuracy:
            chosen = t           # grid is increasing: keeps the loosest threshold that qualifies
    print(f"Threshold chosen on validation: {chosen:.2f}  (set ENTROPY_THRESHOLD to this)")

    # 2) report on TEST
    ty, tp, te = run_mc(model, test_ds, args.passes)
    report = classification_report(ty, tp, target_names=class_names, digits=3)
    acc_all = float((ty == tp).mean())
    keep = te <= chosen
    acc_acc = float((ty[keep] == tp[keep]).mean()) if keep.any() else float("nan")
    summary = (
        f"Test accuracy (all images): {acc_all:.3f}\n"
        f"Entropy threshold (from validation): {chosen:.2f}\n"
        f"Coverage (share of test images accepted): {keep.mean():.3f}\n"
        f"Test accuracy on accepted images: {acc_acc:.3f}\n"
        f"Mean uncertainty, correct predictions: {te[ty == tp].mean():.3f}\n"
        f"Mean uncertainty, wrong predictions:   {te[ty != tp].mean() if (ty != tp).any() else float('nan'):.3f}\n\n"
        + report
    )
    print(summary)
    with open(os.path.join(RESULTS, "metrics.txt"), "w") as f:
        f.write(summary)

    cm = confusion_matrix(ty, tp)
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(cmap="Blues", xticks_rotation=30)
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "confusion_matrix.png"), dpi=150); plt.close()

    # accuracy vs coverage on TEST (reject the most uncertain images first)
    order = np.argsort(te)
    cov, accs = [], []
    for frac in np.linspace(0.3, 1.0, 15):
        k = max(1, int(len(order) * frac))
        idx = order[:k]
        cov.append(frac); accs.append(float((ty[idx] == tp[idx]).mean()))
    np.savetxt(os.path.join(RESULTS, "coverage_accuracy.csv"),
               np.c_[cov, accs], delimiter=",", header="coverage,accuracy", comments="")
    plt.plot(cov, accs, marker="o"); plt.xlabel("Coverage (share of images the model answers)")
    plt.ylabel("Accuracy on answered images"); plt.grid(alpha=.3)
    plt.title("Accuracy rises when uncertain cases are rejected")
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "coverage_accuracy.png"), dpi=150); plt.close()

    plt.hist(te[ty == tp], bins=25, alpha=.7, label="correct")
    if (ty != tp).any():
        plt.hist(te[ty != tp], bins=25, alpha=.7, label="wrong")
    plt.xlabel("Normalised entropy (uncertainty)"); plt.ylabel("Images"); plt.legend()
    plt.title("Do wrong predictions have higher uncertainty?")
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "uncertainty_hist.png"), dpi=150); plt.close()
    print(f"Saved plots and metrics to {RESULTS}")


if __name__ == "__main__":
    main()
