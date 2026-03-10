import os

ROOT = os.getcwd()

folders = [

    # dataset
    "dataset/processed/frames",
    "dataset/processed/audio",

    # features
    "features/mfcc_features",
    "features/face_features",

    # models
    "models",

    # preprocessing
    "preprocessing",

    # dataloaders
    "dataloaders",

    # training
    "training",

    # evaluation
    "evaluation",

    # utils
    "utils",

    # experiments
    "experiments/logs",
    "experiments/checkpoints",
    "experiments/plots",

    # results
    "results"
]

files = [

    # models
    "models/face_model.py",
    "models/audio_model.py",
    "models/fusion_model.py",

    # preprocessing
    "preprocessing/extract_frames.py",
    "preprocessing/extract_audio.py",
    "preprocessing/extract_mfcc.py",
    "preprocessing/dataset_builder.py",

    # dataloaders
    "dataloaders/fusion_dataset.py",

    # training
    "training/train_fusion.py",
    "training/train_face_only.py",
    "training/train_audio_only.py",

    # evaluation
    "evaluation/evaluate.py",
    "evaluation/metrics.py",
    "evaluation/confusion_matrix.py",

    # utils
    "utils/emotion_labels.py",
    "utils/config.py",
    "utils/seed.py",

    # root scripts
    "run_preprocessing.py",
    "train.py",
    "test.py",

    # project files
    "requirements.txt",
    "README.md"
]

def create_structure():

    print("Creating project folders\n")

    for folder in folders:
        path = os.path.join(ROOT, folder)
        os.makedirs(path, exist_ok=True)
        print("Folder:", folder)

    print("\nCreating project files\n")

    for file in files:
        path = os.path.join(ROOT, file)

        if not os.path.exists(path):
            open(path, "w").close()
            print("✔ File:", file)
        else:
            print("Already exists:", file)

    print("\nProject structure created successfully!")

if __name__ == "__main__":
    create_structure()