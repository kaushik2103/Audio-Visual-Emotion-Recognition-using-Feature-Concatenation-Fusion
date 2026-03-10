import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix


def compute_confusion_matrix(labels, preds):

    cm = confusion_matrix(labels, preds)

    return cm


def normalize_confusion_matrix(cm):

    cm = cm.astype("float")

    row_sums = cm.sum(axis=1)[:, np.newaxis]

    normalized = np.divide(cm, row_sums, where=row_sums != 0)

    return normalized


def plot_confusion_matrix(
        labels,
        preds,
        class_names,
        save_path,
        normalize=False,
        figsize=(7,6)
):

    cm = compute_confusion_matrix(labels, preds)

    if normalize:
        cm = normalize_confusion_matrix(cm)

    plt.figure(figsize=figsize)

    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    title = "Normalized Confusion Matrix" if normalize else "Confusion Matrix"

    plt.title(title)

    save_path = Path(save_path)

    save_path.parent.mkdir(exist_ok=True)

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()


def save_confusion_matrices(
        labels,
        preds,
        class_names,
        results_dir
):

    results_dir = Path(results_dir)

    results_dir.mkdir(exist_ok=True)

    plot_confusion_matrix(
        labels,
        preds,
        class_names,
        results_dir / "confusion_matrix.png",
        normalize=False
    )

    plot_confusion_matrix(
        labels,
        preds,
        class_names,
        results_dir / "confusion_matrix_normalized.png",
        normalize=True
    )