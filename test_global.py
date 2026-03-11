import torch
from torch.utils.data import Dataset, DataLoader

import cv2
import numpy as np
import av
import librosa
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from models.multimodal_model import FusionModel


# ==========================================================
# CONFIG
# ==========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_DIR = Path("dataset")
MODEL_PATH = Path("checkpoints/fusion_model.pth")

RESULTS_DIR = Path("results/global_test")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 8

emotion_map = {
    "01":0,
    "03":1,
    "04":2,
    "05":3,
    "06":4
}

emotion_labels = [
    "neutral",
    "happy",
    "sad",
    "angry",
    "fearful"
]

MAX_AUDIO_LEN = 200


# ==========================================================
# IMAGE TRANSFORM
# ==========================================================

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


# ==========================================================
# DATASET
# ==========================================================

class GlobalTestDataset(Dataset):

    def __init__(self, videos):
        self.videos = videos

    def extract_frame(self, path):

        cap = cv2.VideoCapture(str(path))

        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            frame = np.zeros((224,224,3),dtype=np.uint8)
        else:
            frame = frames[len(frames)//2]

        frame = cv2.resize(frame,(224,224))
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        img = Image.fromarray(frame)

        return transform(img)

    def extract_audio(self, path):

        try:
            container = av.open(str(path))
        except:
            return np.zeros(16000),16000

        audio_stream = None

        for stream in container.streams:
            if stream.type == "audio":
                audio_stream = stream
                break

        if audio_stream is None:
            return np.zeros(16000),16000

        frames = []

        for frame in container.decode(audio_stream):
            audio = frame.to_ndarray()
            frames.append(audio)

        if len(frames) == 0:
            return np.zeros(16000),16000

        audio = np.concatenate(frames, axis=1)

        if audio.shape[0] > 1:
            audio = np.mean(audio, axis=0)
        else:
            audio = audio[0]

        return audio, audio_stream.rate

    def extract_mfcc(self, path):

        audio, sr = self.extract_audio(path)

        mfcc = librosa.feature.mfcc(
            y=audio.astype(np.float32),
            sr=sr,
            n_mfcc=40
        )

        if mfcc.shape[1] < MAX_AUDIO_LEN:

            pad = MAX_AUDIO_LEN - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0,0),(0,pad)))

        else:

            mfcc = mfcc[:, :MAX_AUDIO_LEN]

        return torch.tensor(mfcc, dtype=torch.float32)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):

        path = self.videos[idx]

        emotion_code = path.name.split("-")[2]

        label = emotion_map[emotion_code]

        img = self.extract_frame(path)

        audio = self.extract_mfcc(path)

        return img, audio, label, path.name


# ==========================================================
# LOAD DATASET
# ==========================================================

def load_dataset():

    videos = list(DATASET_DIR.rglob("Actor_04/*.mp4"))

    filtered = []

    for v in videos:

        emotion_code = v.name.split("-")[2]

        if emotion_code in emotion_map:
            filtered.append(v)

    print("Total test videos:", len(filtered))

    loader = DataLoader(
        GlobalTestDataset(filtered),
        batch_size=BATCH_SIZE
    )

    return loader


# ==========================================================
# SAVE CONFUSION MATRIX
# ==========================================================

def save_confusion_matrices(labels, preds):

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

    plt.tight_layout()

    plt.savefig(RESULTS_DIR/"confusion_matrix.png")

    plt.close()

    # Normalized

    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm_norm,
        annot=True,
        cmap="Blues",
        xticklabels=emotion_labels,
        yticklabels=emotion_labels
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.title("Normalized Confusion Matrix")

    plt.tight_layout()

    plt.savefig(RESULTS_DIR/"confusion_matrix_normalized.png")

    plt.close()


# ==========================================================
# EVALUATION
# ==========================================================

def evaluate():

    print("Loading model...")

    model = FusionModel(num_classes=5).to(DEVICE)

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )

    model.eval()

    loader = load_dataset()

    preds = []
    labels = []
    names = []

    with torch.no_grad():

        for img, audio, label, name in tqdm(loader):

            img = img.to(DEVICE)
            audio = audio.to(DEVICE)

            out = model(img, audio)

            p = torch.argmax(out,1).cpu().numpy()

            preds.extend(p)
            labels.extend(label.numpy())
            names.extend(name)

    accuracy = accuracy_score(labels, preds)

    print("Accuracy:", accuracy)

    # ======================================================
    # SAVE METRICS
    # ======================================================

    metrics = {
        "accuracy": float(accuracy)
    }

    with open(RESULTS_DIR/"metrics.json","w") as f:
        json.dump(metrics, f, indent=4)

    # ======================================================
    # SAVE REPORT
    # ======================================================

    report = classification_report(
        labels,
        preds,
        target_names=emotion_labels
    )

    with open(RESULTS_DIR/"classification_report.txt","w") as f:
        f.write(report)

    # ======================================================
    # SAVE CONFUSION MATRIX
    # ======================================================

    save_confusion_matrices(labels, preds)

    # ======================================================
    # SAVE PREDICTIONS
    # ======================================================

    df = pd.DataFrame({
        "video": names,
        "true_label": labels,
        "predicted_label": preds
    })

    df.to_csv(RESULTS_DIR/"predictions.csv", index=False)

    print("\nClassification Report:\n")
    print(report)


# ==========================================================

if __name__ == "__main__":

    evaluate()