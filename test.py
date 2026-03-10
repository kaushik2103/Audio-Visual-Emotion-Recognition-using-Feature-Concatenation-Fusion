import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from dataloaders.fusion_dataset import FusionDataset
from models.fusion_model import FusionModel

from evaluation.confusion_matrix import save_confusion_matrices
from evaluation.metrics import evaluate_predictions


BATCH_SIZE = 16
NUM_CLASSES = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = Path(__file__).resolve().parent

CHECKPOINT_DIR = PROJECT_ROOT / "experiments" / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(exist_ok=True)


emotion_labels = [
    "neutral",
    "happy",
    "sad",
    "angry",
    "fearful"
]


def load_model(model_type):

    if model_type == "fusion":

        model = FusionModel(num_classes=NUM_CLASSES)

        checkpoint = CHECKPOINT_DIR / "best_fusion_model.pth"

    elif model_type == "audio":

        from training.train_audio_only import AudioNet
        model = AudioNet()

        checkpoint = CHECKPOINT_DIR / "best_audio_model.pth"

    elif model_type == "face":

        from training.train_face_only import FaceModel
        model = FaceModel()

        checkpoint = CHECKPOINT_DIR / "best_face_model.pth"

    else:

        raise ValueError("Invalid model type")

    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))

    model = model.to(DEVICE)

    model.eval()

    return model

def evaluate(model, loader, model_type):

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for batch in tqdm(loader):

            if model_type == "fusion":

                face, audio, labels = batch

                face = face.to(DEVICE)
                audio = audio.to(DEVICE)

                outputs = model(face, audio)

            elif model_type == "audio":

                audio, labels = batch

                audio = audio.to(DEVICE)

                outputs = model(audio)

            elif model_type == "face":

                images, labels = batch

                images = images.to(DEVICE)

                outputs = model(images)

            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=["audio", "face", "fusion"],
        help="Model to test"
    )

    args = parser.parse_args()

    print("\nTesting model:", args.model)

    if args.model == "fusion":

        dataset = FusionDataset("test")

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )

    elif args.model == "audio":

        from training.train_audio_only import AudioDataset
        dataset = AudioDataset("test")

        loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    elif args.model == "face":

        from training.train_face_only import FaceDataset
        dataset = FaceDataset("test")

        loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    model = load_model(args.model)

    labels, preds = evaluate(model, loader, args.model)

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="weighted")
    recall = recall_score(labels, preds, average="weighted")
    f1 = f1_score(labels, preds, average="weighted")

    print("\nResults")
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)

    model_results_dir = RESULTS_DIR / args.model
    model_results_dir.mkdir(exist_ok=True)

    metrics, report, cm = evaluate_predictions(
        labels,
        preds,
        emotion_labels,
        model_results_dir
    )

    save_confusion_matrices(
        labels,
        preds,
        emotion_labels,
        model_results_dir
    )

    print("\nResults saved in:", model_results_dir)

if __name__ == "__main__":
    main()