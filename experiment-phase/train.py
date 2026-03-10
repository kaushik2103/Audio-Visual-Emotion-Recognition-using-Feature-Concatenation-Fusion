import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

TRAIN_DIR = PROJECT_ROOT / "training"


TRAIN_SCRIPTS = {
    "audio": "train_audio_only.py",
    "face": "train_face_only.py",
    "fusion": "train_fusion.py"
}


def run_training(model_type):

    if model_type not in TRAIN_SCRIPTS:
        print("Invalid model type.")
        print("Choose from: audio, face, fusion")
        sys.exit(1)

    script_name = TRAIN_SCRIPTS[model_type]

    script_path = TRAIN_DIR / script_name

    if not script_path.exists():
        print(f"Training script not found: {script_path}")
        sys.exit(1)

    print("\n" + "="*60)
    print(f"Starting training: {model_type.upper()} model")
    print("="*60)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT
    )

    if result.returncode != 0:
        print("\nTraining failed.")
        sys.exit(1)

    print("\nTraining completed successfully.")


def main():

    parser = argparse.ArgumentParser(
        description="Train emotion recognition models"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["audio", "face", "fusion"],
        help="Model type to train"
    )

    args = parser.parse_args()

    run_training(args.model)


if __name__ == "__main__":
    main()