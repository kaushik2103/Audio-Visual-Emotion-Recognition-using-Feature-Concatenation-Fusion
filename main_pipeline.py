import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import cv2
import numpy as np
import random
import av
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
import librosa
import json

from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

from models.multimodal_model import (
    FusionModel,
    save_model,
    save_metrics,
    save_report,
    save_confusion_matrix
)

# ==========================================================
# CONFIG
# ==========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_DIR = Path("dataset/raw")
RESULTS_DIR = Path("results")
CHECKPOINT_DIR = Path("checkpoints")

RESULTS_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)

EPOCHS = 20
BATCH_SIZE = 8
LR = 1e-4

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

class EmotionDataset(Dataset):

    def __init__(self, video_files):
        self.video_files = video_files

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
            frame = random.choice(frames)

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

        if len(frames)==0:
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
        return len(self.video_files)

    def __getitem__(self, idx):

        path = self.video_files[idx]

        emotion_code = path.name.split("-")[2]

        label = emotion_map[emotion_code]

        image = self.extract_frame(path)

        audio = self.extract_mfcc(path)

        return image, audio, label


# ==========================================================
# LOAD DATASET
# ==========================================================

def load_dataset():

    videos = list(DATASET_DIR.rglob("*.mp4"))

    print("Total videos found:", len(videos))

    filtered_videos = []

    for v in videos:

        emotion_code = v.name.split("-")[2]

        if emotion_code in emotion_map:
            filtered_videos.append(v)

    print("Filtered usable videos:", len(filtered_videos))

    random.shuffle(filtered_videos)

    split = int(0.8 * len(filtered_videos))

    train_videos = filtered_videos[:split]
    test_videos = filtered_videos[split:]

    train_loader = DataLoader(
        EmotionDataset(train_videos),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )

    test_loader = DataLoader(
        EmotionDataset(test_videos),
        batch_size=BATCH_SIZE,
        drop_last=True
    )

    return train_loader, test_loader


# ==========================================================
# TRAIN MODEL
# ==========================================================

def train_model(train_loader):

    model = FusionModel(num_classes=5).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=LR)

    losses = []

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0

        pbar = tqdm(train_loader)

        for img, audio, label in pbar:

            img = img.to(DEVICE)
            audio = audio.to(DEVICE)
            label = label.to(DEVICE)

            optimizer.zero_grad()

            output = model(img, audio)

            loss = criterion(output, label)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            pbar.set_description(f"Epoch {epoch+1}")

        epoch_loss = running_loss / len(train_loader)

        losses.append(epoch_loss)

        print(f"Epoch {epoch+1}/{EPOCHS} Loss: {epoch_loss:.4f}")

    save_model(model, CHECKPOINT_DIR)

    return model, losses


# ==========================================================
# EVALUATE MODEL
# ==========================================================

def evaluate(model, test_loader):

    model.eval()

    preds = []
    labels = []

    with torch.no_grad():

        for img, audio, label in tqdm(test_loader):

            img = img.to(DEVICE)
            audio = audio.to(DEVICE)

            out = model(img, audio)

            p = torch.argmax(out,1).cpu().numpy()

            preds.extend(p)
            labels.extend(label.numpy())

    accuracy = np.mean(np.array(labels)==np.array(preds))

    metrics = {"accuracy": float(accuracy)}

    save_metrics(metrics, RESULTS_DIR)

    save_report(labels, preds, emotion_labels, RESULTS_DIR)

    save_confusion_matrix(labels, preds, emotion_labels, RESULTS_DIR)

    report = classification_report(labels, preds, target_names=emotion_labels)

    return report


# ==========================================================
# TRAINING CURVE
# ==========================================================

def save_training_plot(losses):

    plt.figure()

    plt.plot(losses)

    plt.title("Training Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.savefig(RESULTS_DIR/"training_curve.png")

    plt.close()


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("Loading dataset...")

    train_loader, test_loader = load_dataset()

    print("Training model...")

    model, losses = train_model(train_loader)

    print("Evaluating model...")

    report = evaluate(model, test_loader)

    save_training_plot(losses)

    print("\nClassification Report:\n")

    print(report)

    print("\nPipeline Finished")


if __name__ == "__main__":
    main()