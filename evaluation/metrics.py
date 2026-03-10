import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


def compute_metrics(labels, preds):

    accuracy = accuracy_score(labels, preds)

    precision = precision_score(
        labels,
        preds,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        labels,
        preds,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        labels,
        preds,
        average="weighted",
        zero_division=0
    )

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)
    }

    return metrics


def generate_classification_report(labels, preds, class_names):

    report = classification_report(
        labels,
        preds,
        target_names=class_names,
        zero_division=0
    )

    return report


def compute_confusion_matrix(labels, preds):

    cm = confusion_matrix(labels, preds)

    return cm


def save_metrics(metrics, save_path):

    save_path = Path(save_path)

    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=4)


def save_classification_report(report, save_path):

    save_path = Path(save_path)

    with open(save_path, "w") as f:
        f.write(report)



def save_predictions(labels, preds, save_path):

    df = pd.DataFrame({
        "true_label": labels,
        "predicted_label": preds
    })

    df.to_csv(save_path, index=False)


def evaluate_predictions(labels, preds, class_names, results_dir):

    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)

    metrics = compute_metrics(labels, preds)

    report = generate_classification_report(
        labels,
        preds,
        class_names
    )

    cm = compute_confusion_matrix(labels, preds)

    save_metrics(
        metrics,
        results_dir / "metrics.json"
    )

    save_classification_report(
        report,
        results_dir / "classification_report.txt"
    )

    save_predictions(
        labels,
        preds,
        results_dir / "predictions.csv"
    )

    return metrics, report, cm