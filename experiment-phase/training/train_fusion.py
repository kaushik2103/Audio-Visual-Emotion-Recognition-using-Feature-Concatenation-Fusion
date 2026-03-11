import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from dataloaders.fusion_dataset import FusionDataset
from models.fusion_model import FusionModel


BATCH_SIZE = 16
EPOCHS = 50
LR = 0.001
NUM_CLASSES = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_DIR = PROJECT_ROOT / "experiments" / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


emotion_labels = [
    "neutral",
    "happy",
    "sad",
    "angry",
    "fearful"
]


def train_epoch(model, loader, criterion, optimizer):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for face, audio, labels in tqdm(loader):

        face = face.to(DEVICE)
        audio = audio.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(face, audio)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)

        correct += (preds == labels).sum().item()

        total += labels.size(0)

    accuracy = correct / total

    return total_loss / len(loader), accuracy


def validate(model, loader, criterion):

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for face, audio, labels in tqdm(loader):

            face = face.to(DEVICE)
            audio = audio.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(face, audio)

            loss = criterion(outputs, labels)

            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()

            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total

    return total_loss / len(loader), accuracy, all_preds, all_labels


def save_plots(history):

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig(RESULTS_DIR / "loss_plot.png")
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Val Accuracy")
    plt.legend()
    plt.title("Accuracy Curve")
    plt.savefig(RESULTS_DIR / "accuracy_plot.png")
    plt.close()


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

    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.title("Confusion Matrix")

    plt.savefig(RESULTS_DIR / "confusion_matrix.png")

    plt.close()


def main():

    print("Using device:", DEVICE)

    train_dataset = FusionDataset(split="train")
    val_dataset = FusionDataset(split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = FusionModel(num_classes=NUM_CLASSES).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )

    best_val_acc = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    final_preds = None
    final_labels = None

    for epoch in range(EPOCHS):

        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_acc, preds, labels = validate(
            model,
            val_loader,
            criterion
        )

        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Train Acc : {train_acc:.4f}")

        print(f"Val Loss  : {val_loss:.4f}")
        print(f"Val Acc   : {val_acc:.4f}")

        history["epoch"].append(epoch+1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        checkpoint_path = CHECKPOINT_DIR / f"fusion_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), checkpoint_path)

        if val_acc > best_val_acc:

            best_val_acc = val_acc

            best_path = CHECKPOINT_DIR / "best_fusion_model.pth"

            torch.save(model.state_dict(), best_path)

            final_preds = preds
            final_labels = labels

            print("Best models saved!")

    print("\nTraining Complete!")
    print("Best Validation Accuracy:", best_val_acc)

    df = pd.DataFrame(history)
    df.to_csv(RESULTS_DIR / "training_log.csv", index=False)

    save_plots(history)

    save_confusion_matrix(final_labels, final_preds)

    report = classification_report(
        final_labels,
        final_preds,
        target_names=emotion_labels
    )

    with open(RESULTS_DIR / "classification_report.txt", "w") as f:
        f.write(report)

    metrics = {
        "best_val_accuracy": float(best_val_acc)
    }

    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("\nAll results saved in /results folder")


if __name__ == "__main__":

    main()