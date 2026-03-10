import os
import csv
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = PROJECT_ROOT / "dataset" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "dataset"

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

RANDOM_SEED = 42

emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# choose fewer classes for easier training
SELECTED_EMOTIONS = ["01", "03", "04", "05", "06"]


def parse_filename(filename):

    parts = filename.split("-")

    modality = parts[0]
    vocal_channel = parts[1]
    emotion = parts[2]
    intensity = parts[3]
    statement = parts[4]
    repetition = parts[5]
    actor = parts[6].split(".")[0]

    return {
        "modality": modality,
        "vocal_channel": vocal_channel,
        "emotion": emotion,
        "emotion_label": emotion_map.get(emotion, "unknown"),
        "intensity": intensity,
        "statement": statement,
        "repetition": repetition,
        "actor": actor
    }


def collect_samples():

    samples = []

    for root, dirs, files in os.walk(DATASET_ROOT):

        for file in files:

            if not file.endswith(".mp4"):
                continue

            info = parse_filename(file)

            # keep only full AV samples
            if info["modality"] != "01":
                continue

            # keep selected emotions
            if info["emotion"] not in SELECTED_EMOTIONS:
                continue

            full_path = Path(root) / file

            sample = {
                "video_path": str(full_path),
                "emotion": info["emotion_label"],
                "emotion_id": info["emotion"],
                "actor": info["actor"]
            }

            samples.append(sample)

    return samples


def split_dataset(samples):

    random.shuffle(samples)

    total = len(samples)

    train_end = int(total * TRAIN_SPLIT)
    val_end = int(total * (TRAIN_SPLIT + VAL_SPLIT))

    train_set = samples[:train_end]
    val_set = samples[train_end:val_end]
    test_set = samples[val_end:]

    return train_set, val_set, test_set


def save_csv(data, filename):

    filepath = OUTPUT_DIR / filename

    with open(filepath, "w", newline="") as csvfile:

        fieldnames = ["video_path", "emotion", "emotion_id", "actor"]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for row in data:
            writer.writerow(row)

    print("Saved:", filepath)


def main():

    random.seed(RANDOM_SEED)

    print("Dataset root:", DATASET_ROOT)
    print("Output folder:", OUTPUT_DIR)

    print("\nScanning dataset...\n")

    samples = collect_samples()

    print("Total valid samples:", len(samples))

    if len(samples) == 0:
        print("\nERROR: No samples found.")
        print("Check dataset/raw path.")
        return

    print("\nSplitting dataset...\n")

    train_set, val_set, test_set = split_dataset(samples)

    print("Train:", len(train_set))
    print("Validation:", len(val_set))
    print("Test:", len(test_set))

    print("\nSaving CSV files...\n")

    save_csv(train_set, "train.csv")
    save_csv(val_set, "val.csv")
    save_csv(test_set, "test.csv")

    print("\nDataset build complete!")


if __name__ == "__main__":
    main()