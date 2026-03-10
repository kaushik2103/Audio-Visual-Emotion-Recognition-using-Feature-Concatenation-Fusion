import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from dataloaders.fusion_dataset import FusionDataset
from models.fusion_model import FusionModel


BATCH_SIZE = 16
NUM_CLASSES = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = PROJECT_ROOT / "experiments" / "checkpoints" / "best_fusion_model.pth"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(exist_ok=True)


emotion_labels = [
    "neutral",
    "happy",
    "sad",
    "angry",
    "fearful"
]


def load_model():

    model = FusionModel(num_classes=NUM_CLASSES)

    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))

    model = model.to(DEVICE)

    model.eval()

    return model


def evaluate(model, loader):

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for face, audio, labels in tqdm(loader):

            face = face.to(DEVICE)
            audio = audio.to(DEVICE)

            outputs = model(face, audio)

            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


def save_confusion_matrix(labels, preds):

    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=emotion_labels,
        yticklabels=emotion_labels
    )

    plt.title("Fusion Model Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.savefig(RESULTS_DIR / "fusion_confusion_matrix.png")

    plt.close()


def main():

    print("Running evaluation on test dataset")

    test_dataset = FusionDataset(split="test")

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = load_model()

    labels, preds = evaluate(model, test_loader)

    accuracy = accuracy_score(labels, preds)

    precision = precision_score(labels, preds, average="weighted")

    recall = recall_score(labels, preds, average="weighted")

    f1 = f1_score(labels, preds, average="weighted")

    print("\nTest Results")
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)
    }

    with open(RESULTS_DIR / "fusion_test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    report = classification_report(
        labels,
        preds,
        target_names=emotion_labels
    )

    with open(RESULTS_DIR / "fusion_classification_report.txt", "w") as f:
        f.write(report)


    save_confusion_matrix(labels, preds)


    df = pd.DataFrame({
        "true_label": labels,
        "predicted_label": preds
    })

    df.to_csv(RESULTS_DIR / "fusion_predictions.csv", index=False)

    print("\nEvaluation complete")
    print("Results saved in /results folder")


if __name__ == "__main__":

    main()