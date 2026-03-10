import os
import av
import pandas as pd
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"
AUDIO_OUTPUT_DIR = DATASET_DIR / "processed" / "audio"

CSV_FILES = [
    DATASET_DIR / "train.csv",
    DATASET_DIR / "val.csv",
    DATASET_DIR / "test.csv"
]

TARGET_SR = 16000


def extract_audio(video_path, output_path):

    try:

        container = av.open(str(video_path))

        audio_stream = None
        for stream in container.streams:
            if stream.type == "audio":
                audio_stream = stream
                break

        if audio_stream is None:
            print(f"No audio stream found: {video_path}")
            return

        frames = []

        for frame in container.decode(audio_stream):
            frame_array = frame.to_ndarray()
            frames.append(frame_array)

        if len(frames) == 0:
            return

        audio = np.concatenate(frames, axis=1)

        # convert to mono
        if audio.shape[0] > 1:
            audio = np.mean(audio, axis=0)
        else:
            audio = audio[0]

        sr = audio_stream.rate

        # resample if needed
        if sr != TARGET_SR:
            import librosa
            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)
            sr = TARGET_SR

        sf.write(output_path, audio, sr)

    except Exception as e:
        print("Error extracting audio:", video_path)
        print(e)


def process_csv(csv_file):

    df = pd.read_csv(csv_file)

    print(f"\nProcessing {csv_file.name} ({len(df)} videos)")

    for _, row in tqdm(df.iterrows(), total=len(df)):

        video_path = Path(row["video_path"])
        emotion = row["emotion"]

        video_name = video_path.stem

        output_dir = AUDIO_OUTPUT_DIR / emotion
        os.makedirs(output_dir, exist_ok=True)

        output_wav = output_dir / f"{video_name}.wav"

        if output_wav.exists():
            continue

        extract_audio(video_path, output_wav)


def main():

    print("Starting audio extraction...\n")

    os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)

    for csv_file in CSV_FILES:

        if not csv_file.exists():
            print("Missing CSV:", csv_file)
            continue

        process_csv(csv_file)

    print("\nAudio extraction complete!")


if __name__ == "__main__":
    main()