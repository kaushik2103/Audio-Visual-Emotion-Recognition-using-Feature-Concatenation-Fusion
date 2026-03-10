import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
import random


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"
FRAME_DIR = DATASET_DIR / "processed" / "frames"
MFCC_DIR = PROJECT_ROOT / "features" / "mfcc_features"


emotion_to_label = {
    "neutral": 0,
    "happy": 1,
    "sad": 2,
    "angry": 3,
    "fearful": 4
}


image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


class FusionDataset(Dataset):

    def __init__(self, split="train"):

        self.split = split

        csv_path = DATASET_DIR / f"{split}.csv"
        self.data = pd.read_csv(csv_path)

        # load MFCC features
        self.audio_features = np.load(MFCC_DIR / f"X_{split}.npy")
        self.labels = np.load(MFCC_DIR / f"y_{split}.npy")

    def __len__(self):
        return len(self.data)

    def load_random_frame(self, video_name, emotion):

        frame_folder = FRAME_DIR / emotion / video_name

        if not frame_folder.exists():
            raise RuntimeError(f"Missing frame folder: {frame_folder}")

        frame_files = list(frame_folder.glob("*.jpg"))

        if len(frame_files) == 0:
            raise RuntimeError(f"No frames found in {frame_folder}")

        frame_path = random.choice(frame_files)

        image = Image.open(frame_path).convert("RGB")

        image = image_transform(image)

        return image

    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        video_path = Path(row["video_path"])
        emotion = row["emotion"]
        video_name = video_path.stem

        # Load face frame
        face_tensor = self.load_random_frame(video_name, emotion)

        # Load audio features
        audio_feature = self.audio_features[idx]
        audio_feature = torch.tensor(audio_feature, dtype=torch.float32)

        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.long)

        return face_tensor, audio_feature, label