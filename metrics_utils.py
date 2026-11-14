"""
metrics_utils.py - one function that writes ALL evaluation outputs for a model.

For model <name> it creates in results/:
    <name>.json                 accuracy, macro-F1, per-class report, extra info
    <name>_confusion.png        row-normalised confusion matrix
    <name>_curves.png           train/val accuracy and loss per epoch
Deliberately framework-free (numpy + sklearn + matplotlib) so both the Keras
and the PyTorch (wav2vec2) scripts can use it.
"""
import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")                 # write PNGs without needing a display
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import config


def plot_confusion(cm_norm, class_names, title, path):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_curves(history, title, path):
    """history: dict with keys accuracy / val_accuracy / loss / val_loss (lists per epoch)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key in zip(axes, ["accuracy", "loss"]):
        if key in history:
            ax.plot(history[key], label=f"train {key}")
        if f"val_{key}" in history:
            ax.plot(history[f"val_{key}"], label=f"val {key}")
        ax.set_xlabel("epoch")
        ax.set_title(f"{title} - {key}")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_results(model_name, y_true, y_pred, history=None, class_names=config.EMOTIONS,
                 results_dir=config.RESULTS_DIR, extra=None):
    """Compute test metrics, save JSON + plots, print a short summary, return the metrics dict."""
    os.makedirs(results_dir, exist_ok=True)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = list(range(len(class_names)))

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    report = classification_report(y_true, y_pred, labels=labels, target_names=class_names,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    plot_confusion(cm_norm, class_names, f"{model_name} (test acc {acc:.3f})",
                   os.path.join(results_dir, f"{model_name}_confusion.png"))
    if history:
        plot_curves(history, model_name, os.path.join(results_dir, f"{model_name}_curves.png"))

    results = {
        "model": model_name,
        "test_accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "n_test": int(len(y_true)),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "class_names": list(class_names),
    }
    if history and "val_accuracy" in history:
        results["best_val_accuracy"] = float(max(history["val_accuracy"]))
        results["epochs_trained"] = len(history["val_accuracy"])
    if extra:
        results.update(extra)
    with open(os.path.join(results_dir, f"{model_name}.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n===== {model_name}: test accuracy {acc:.4f} | macro-F1 {macro_f1:.4f} =====")
    print(classification_report(y_true, y_pred, labels=labels, target_names=class_names, zero_division=0))
    return results
