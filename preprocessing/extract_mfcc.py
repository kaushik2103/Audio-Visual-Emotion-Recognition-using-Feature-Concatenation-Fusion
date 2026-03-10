import os
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"
AUDIO_DIR = DATASET_DIR / "processed" / "audio"

FEATURE_DIR = PROJECT_ROOT / "features" / "mfcc_features"
os.makedirs(FEATURE_DIR, exist_ok=True)

CSV_FILES = {
    "train": DATASET_DIR / "train.csv",
    "val": DATASET_DIR / "val.csv",
    "test": DATASET_DIR / "test.csv"
}


SAMPLE_RATE = 16000
N_MFCC = 40
MAX_AUDIO_LENGTH = 5  # seconds

emotion_to_label = {
    "neutral": 0,
    "happy": 1,
    "sad": 2,
    "angry": 3,
    "fearful": 4
}


def extract_mfcc(audio_path):

    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

    max_len = SAMPLE_RATE * MAX_AUDIO_LENGTH

    if len(audio) > max_len:
        audio = audio[:max_len]
    else:
        padding = max_len - len(audio)
        audio = np.pad(audio, (0, padding))

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC
    )

    mfcc = mfcc.T  # shape: (time_steps, n_mfcc)

    mfcc = np.mean(mfcc, axis=0)  # compress time dimension

    return mfcc



def process_split(split_name, csv_path):

    df = pd.read_csv(csv_path)

    features = []
    labels = []

    print(f"\nProcessing {split_name} ({len(df)} samples)")

    for _, row in tqdm(df.iterrows(), total=len(df)):

        video_path = Path(row["video_path"])
        emotion = row["emotion"]

        audio_path = AUDIO_DIR / emotion / f"{video_path.stem}.wav"

        if not audio_path.exists():
            continue

        mfcc = extract_mfcc(audio_path)

        features.append(mfcc)
        labels.append(emotion_to_label[emotion])

    X = np.array(features)
    y = np.array(labels)

    np.save(FEATURE_DIR / f"X_{split_name}.npy", X)
    np.save(FEATURE_DIR / f"y_{split_name}.npy", y)

    print(f"Saved {split_name} features: {X.shape}")


def main():

    print("Starting MFCC extraction\n")

    for split_name, csv_path in CSV_FILES.items():

        if not csv_path.exists():
            print("Missing:", csv_path)
            continue

        process_split(split_name, csv_path)

    print("\nMFCC feature extraction complete!")


if __name__ == "__main__":
    main()