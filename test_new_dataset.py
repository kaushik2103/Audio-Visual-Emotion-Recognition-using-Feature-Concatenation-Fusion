import torch
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
import json
import av

from PIL import Image
import torchvision.transforms as transforms

import librosa

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns

from models.fusion_model import FusionModel


# ==========================================
# CONFIG
# ==========================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 5
N_MFCC = 40

emotion_map = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful"
}

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


# ==========================================
# LOAD MODEL
# ==========================================

def load_model():

    checkpoint = Path("experiments/checkpoints/best_fusion_model.pth")

    model = FusionModel(num_classes=NUM_CLASSES)

    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))

    model = model.to(DEVICE)

    model.eval()

    return model


# ==========================================
# EXTRACT FRAME
# ==========================================

def extract_frame(video_path):

    cap = cv2.VideoCapture(str(video_path))

    frames = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frames.append(frame)

    cap.release()

    frame = frames[np.random.randint(len(frames))]

    frame = cv2.resize(frame, (224,224))

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    img = Image.fromarray(frame)

    img = transform(img)

    return img


# ==========================================
# EXTRACT AUDIO FROM VIDEO
# ==========================================

def extract_audio(video_path):

    container = av.open(str(video_path))

    audio_stream = next(s for s in container.streams if s.type == "audio")

    audio_frames = []

    for frame in container.decode(audio_stream):

        audio = frame.to_ndarray()

        audio_frames.append(audio)

    audio = np.concatenate(audio_frames, axis=1)

    if audio.shape[0] > 1:
        audio = np.mean(audio, axis=0)

    else:
        audio = audio[0]

    return audio, audio_stream.rate


# ==========================================
# MFCC
# ==========================================

def extract_mfcc(video_path):

    audio, sr = extract_audio(video_path)

    mfcc = librosa.feature.mfcc(
        y=audio.astype(np.float32),
        sr=sr,
        n_mfcc=N_MFCC
    )

    mfcc = np.mean(mfcc.T, axis=0)

    return torch.tensor(mfcc, dtype=torch.float32)


# ==========================================
# GET LABEL
# ==========================================

def get_label(filename):

    parts = filename.split("-")

    emotion_id = parts[2]

    if emotion_id not in emotion_map:
        return None

    emotion = emotion_map[emotion_id]

    return emotion_to_label[emotion]


# ==========================================
# CONFUSION MATRIX
# ==========================================

def save_confusion_matrix(labels, preds, results_dir):

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

    plt.savefig(results_dir / "confusion_matrix.png")

    plt.close()


# ==========================================
# MAIN
# ==========================================

def main(dataset_path):

    dataset_path = Path(dataset_path)

    results_dir = Path("results/new_dataset")

    results_dir.mkdir(parents=True, exist_ok=True)

    model = load_model()

    video_files = list(dataset_path.rglob("*.mp4"))

    print("Total videos found:", len(video_files))

    all_preds = []
    all_labels = []

    for video_path in tqdm(video_files):

        label = get_label(video_path.name)

        if label is None:
            continue

        face = extract_frame(video_path)

        audio = extract_mfcc(video_path)

        face = face.unsqueeze(0).to(DEVICE)
        audio = audio.unsqueeze(0).to(DEVICE)

        with torch.no_grad():

            output = model(face, audio)

            pred = torch.argmax(output, dim=1).item()

        all_preds.append(pred)
        all_labels.append(label)

    labels = np.array(all_labels)
    preds = np.array(all_preds)

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="weighted")
    recall = recall_score(labels, preds, average="weighted")
    f1 = f1_score(labels, preds, average="weighted")

    print("\nRESULTS")
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

    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    report = classification_report(
        labels,
        preds,
        target_names=emotion_labels
    )

    with open(results_dir / "classification_report.txt", "w") as f:
        f.write(report)

    save_confusion_matrix(labels, preds, results_dir)

    df = pd.DataFrame({
        "true_label": labels,
        "predicted_label": preds
    })

    df.to_csv(results_dir / "predictions.csv", index=False)

    print("\nResults saved in:", results_dir)


# ==========================================
# ENTRY
# ==========================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to new dataset"
    )

    args = parser.parse_args()

    main(args.dataset)