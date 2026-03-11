import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import json


BATCH_SIZE = 32
EPOCHS = 50
LR = 0.01
NUM_CLASSES = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURE_DIR = PROJECT_ROOT / "features" / "mfcc_features"
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



class AudioDataset(Dataset):

    def __init__(self, split):

        self.X = np.load(FEATURE_DIR / f"X_{split}.npy")
        self.y = np.load(FEATURE_DIR / f"y_{split}.npy")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        audio = torch.tensor(self.X[idx], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.long)

        return audio, label



class AudioNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(40, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, NUM_CLASSES)
        )

    def forward(self, x):

        return self.network(x)



def train_epoch(model, loader, criterion, optimizer):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for audio, labels in tqdm(loader):

        audio = audio.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(audio)

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

        for audio, labels in tqdm(loader):

            audio = audio.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(audio)

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
    plt.plot(epochs, history["train_loss"], label="Train")
    plt.plot(epochs, history["val_loss"], label="Validation")
    plt.title("Loss Curve")
    plt.legend()
    plt.savefig(RESULTS_DIR / "audio_loss_plot.png")
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train")
    plt.plot(epochs, history["val_acc"], label="Validation")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.savefig(RESULTS_DIR / "audio_accuracy_plot.png")
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

    plt.title("Audio Model Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.savefig(RESULTS_DIR / "audio_confusion_matrix.png")

    plt.close()


def main():

    print("Using device:", DEVICE)

    train_dataset = AudioDataset("train")
    val_dataset = AudioDataset("val")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = AudioNet().to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

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

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)

        val_loss, val_acc, preds, labels = validate(model, val_loader, criterion)

        history["epoch"].append(epoch+1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:

            best_val_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best_audio_model.pth")

            final_preds = preds
            final_labels = labels

            print("Best audio models saved!")

    print("\nTraining Complete!")

    pd.DataFrame(history).to_csv(RESULTS_DIR / "audio_training_log.csv", index=False)

    save_plots(history)

    save_confusion_matrix(final_labels, final_preds)

    report = classification_report(final_labels, final_preds, target_names=emotion_labels)

    with open(RESULTS_DIR / "audio_classification_report.txt", "w") as f:
        f.write(report)

    metrics = {
        "best_val_accuracy": float(best_val_acc)
    }

    with open(RESULTS_DIR / "audio_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Results saved in results/ folder")


if __name__ == "__main__":

    main()