import os
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"
FRAME_OUTPUT_DIR = DATASET_DIR / "processed" / "frames"

CSV_FILES = [
    DATASET_DIR / "train.csv",
    DATASET_DIR / "val.csv",
    DATASET_DIR / "test.csv"
]

# number of frames to extract per video
FRAMES_PER_VIDEO = 10

IMAGE_SIZE = (224, 224)


def extract_frames(video_path, output_folder):

    cap = cv2.VideoCapture(str(video_path))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        return

    step = max(total_frames // FRAMES_PER_VIDEO, 1)

    frame_count = 0
    saved_count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % step == 0 and saved_count < FRAMES_PER_VIDEO:

            frame = cv2.resize(frame, IMAGE_SIZE)

            frame_name = f"frame_{saved_count:03d}.jpg"

            save_path = output_folder / frame_name

            cv2.imwrite(str(save_path), frame)

            saved_count += 1

        frame_count += 1

    cap.release()


def process_csv(csv_file):

    df = pd.read_csv(csv_file)

    print(f"\nProcessing {csv_file.name} - {len(df)} videos")

    for _, row in tqdm(df.iterrows(), total=len(df)):

        video_path = Path(row["video_path"])
        emotion = row["emotion"]

        video_name = video_path.stem

        output_folder = FRAME_OUTPUT_DIR / emotion / video_name

        os.makedirs(output_folder, exist_ok=True)

        extract_frames(video_path, output_folder)



def main():

    print("Frame extraction started\n")

    os.makedirs(FRAME_OUTPUT_DIR, exist_ok=True)

    for csv_file in CSV_FILES:

        if not csv_file.exists():
            print(f"Missing {csv_file}")
            continue

        process_csv(csv_file)

    print("\nFrame extraction complete!")


if __name__ == "__main__":
    main()