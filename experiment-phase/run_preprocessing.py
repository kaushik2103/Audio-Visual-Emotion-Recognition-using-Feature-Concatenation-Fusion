import subprocess
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent

PREPROCESS_DIR = PROJECT_ROOT / "preprocessing"


SCRIPTS = [
    "dataset_builder.py",
    "extract_frames.py",
    "extract_audio.py",
    "extract_mfcc.py"
]


def run_script(script_name):

    script_path = PREPROCESS_DIR / script_name

    if not script_path.exists():
        print(f"Script not found: {script_path}")
        sys.exit(1)

    print("\n" + "="*60)
    print(f"Running: {script_name}")
    print("="*60)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT
    )

    if result.returncode != 0:
        print(f"\nError running {script_name}")
        sys.exit(1)

    print(f"\nCompleted: {script_name}")


def main():

    print("\nStarting Full Preprocessing Pipeline\n")

    for script in SCRIPTS:
        run_script(script)

    print("\nAll preprocessing steps completed successfully!")


if __name__ == "__main__":

    main()