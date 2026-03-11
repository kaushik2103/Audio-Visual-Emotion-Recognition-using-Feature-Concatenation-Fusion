import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


# ==========================================================
# IMAGE ENCODER (ResNet50)
# ==========================================================

class ImageEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)

        self.feature_extractor = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        self.output_dim = 2048

    def forward(self, x):

        x = self.feature_extractor(x)

        x = x.view(x.size(0), -1)

        return x


# ==========================================================
# AUDIO ENCODER (CNN + BiLSTM)
# ==========================================================

class AudioEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.cnn = nn.Sequential(

            nn.Conv1d(40, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.output_dim = 512

    def forward(self, x):

        x = self.cnn(x)

        x = x.permute(0, 2, 1)

        x, _ = self.lstm(x)

        x = x.mean(dim=1)

        return x


# ==========================================================
# FUSION MODEL
# ==========================================================

class FusionModel(nn.Module):

    def __init__(self, num_classes=5):

        super().__init__()

        self.image_encoder = ImageEncoder()

        self.audio_encoder = AudioEncoder()

        fusion_dim = (
            self.image_encoder.output_dim +
            self.audio_encoder.output_dim
        )

        self.attention = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(

            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, num_classes)
        )

    def forward(self, image, audio):

        img_feat = self.image_encoder(image)

        aud_feat = self.audio_encoder(audio)

        fusion = torch.cat([img_feat, aud_feat], dim=1)

        attention = self.attention(fusion)

        fusion = fusion * attention

        out = self.classifier(fusion)

        return out


# ==========================================================
# SAVE MODEL
# ==========================================================

def save_model(model, checkpoint_dir):

    checkpoint_dir = Path(checkpoint_dir)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        model.state_dict(),
        checkpoint_dir / "fusion_model.pth"
    )


# ==========================================================
# SAVE METRICS
# ==========================================================

def save_metrics(metrics, results_dir):

    results_dir = Path(results_dir)

    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)


# ==========================================================
# SAVE CLASSIFICATION REPORT
# ==========================================================

def save_report(labels, preds, class_names, results_dir):

    results_dir = Path(results_dir)

    report = classification_report(
        labels,
        preds,
        target_names=class_names
    )

    with open(results_dir / "report.txt", "w") as f:
        f.write(report)


# ==========================================================
# SAVE CONFUSION MATRIX
# ==========================================================

def save_confusion_matrix(labels, preds, class_names, results_dir):

    results_dir = Path(results_dir)

    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    plt.tight_layout()

    plt.savefig(results_dir / "confusion_matrix.png")

    plt.close()


# ==========================================================
# SAVE TRAINING CURVE
# ==========================================================

def save_training_curve(losses, results_dir):

    results_dir = Path(results_dir)

    plt.figure()

    plt.plot(losses)

    plt.title("Training Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.savefig(results_dir / "training_curve.png")

    plt.close()