import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
from PIL import Image
import random
import timm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json
import torchvision.transforms as transforms


BATCH_SIZE = 32
EPOCHS = 50
LR = 0.0001
NUM_CLASSES = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FRAME_DIR = PROJECT_ROOT / "dataset" / "processed" / "frames"
DATASET_DIR = PROJECT_ROOT / "dataset"

CHECKPOINT_DIR = PROJECT_ROOT / "experiments" / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


emotion_to_label = {
    "neutral": 0,
    "happy": 1,
    "sad": 2,
    "angry": 3,
    "fearful": 4
}

emotion_labels = list(emotion_to_label.keys())


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


class FaceDataset(Dataset):

    def __init__(self, split):

        csv_path = DATASET_DIR / f"{split}.csv"
        self.data = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.data)

    def load_random_frame(self, video_name, emotion):

        frame_folder = FRAME_DIR / emotion / video_name

        frame_files = list(frame_folder.glob("*.jpg"))

        frame_path = random.choice(frame_files)

        img = Image.open(frame_path).convert("RGB")

        img = transform(img)

        return img

    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        video_path = Path(row["video_path"])
        emotion = row["emotion"]

        video_name = video_path.stem

        img = self.load_random_frame(video_name, emotion)

        label = emotion_to_label[emotion]

        return img, torch.tensor(label)


class FaceModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            num_classes=0
        )

        in_features = self.backbone.num_features

        self.classifier = nn.Sequential(

            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(512, NUM_CLASSES)
        )

    def forward(self, x):

        features = self.backbone(x)

        out = self.classifier(features)

        return out


def train_epoch(model, loader, criterion, optimizer):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(loader):

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

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

        for images, labels in tqdm(loader):

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

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
    plt.legend()
    plt.title("Face Model Loss")
    plt.savefig(RESULTS_DIR / "face_loss_plot.png")
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train")
    plt.plot(epochs, history["val_acc"], label="Validation")
    plt.legend()
    plt.title("Face Model Accuracy")
    plt.savefig(RESULTS_DIR / "face_accuracy_plot.png")
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

    plt.title("Face Model Confusion Matrix")

    plt.savefig(RESULTS_DIR / "face_confusion_matrix.png")

    plt.close()


def main():

    print("Using device:", DEVICE)

    train_dataset = FaceDataset("train")
    val_dataset = FaceDataset("val")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = FaceModel().to(DEVICE)

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

            torch.save(model.state_dict(), CHECKPOINT_DIR / "best_face_model.pth")

            final_preds = preds
            final_labels = labels

            print("Best face model saved!")

    print("\nTraining Complete!")

    pd.DataFrame(history).to_csv(RESULTS_DIR / "face_training_log.csv", index=False)

    save_plots(history)

    save_confusion_matrix(final_labels, final_preds)

    report = classification_report(final_labels, final_preds, target_names=emotion_labels)

    with open(RESULTS_DIR / "face_classification_report.txt", "w") as f:
        f.write(report)

    metrics = {
        "best_val_accuracy": float(best_val_acc)
    }

    with open(RESULTS_DIR / "face_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Results saved in results/")


if __name__ == "__main__":
    main()