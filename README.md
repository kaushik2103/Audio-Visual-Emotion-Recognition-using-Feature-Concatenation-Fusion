# Audio-Visual-Emotion-Recognition-using-Feature-Concatenation-Fusion

## Problem Statement

**Detect human emotions using two modalities:**

1. Facial expressions (video frames)
2. Speech tone (audio signal)

### The model combines both modalities to improve emotion recognition accuracy compared to single-modality systems.

**Example emotions:**

1. Happy
2. Sad
3. Angry
4. Neutral
5. Fear
6. Disgust

*Project Working FLow*:
`Video (.mp4)
     │
Frame Extraction
     │
CNN (ResNet18)
     │
Face Feature Vector (512)`

`Audio (.wav)
     │
MFCC Extraction
     │
Audio CNN / LSTM
     │
Audio Feature Vector (128)`

`Fusion
     │
Concatenate (512 + 128)
     │
Fully Connected Layers
     │
Softmax
     │
Emotion Prediction`

# Audio–Visual Emotion Recognition using Feature Concatenation Fusion

## Overview

This project implements a **multimodal emotion recognition system** that detects human emotions using **both facial expressions (video frames)** and **speech signals (audio)**.
The system combines information from **visual and audio modalities** using a **feature-level fusion architecture** to improve recognition accuracy compared to single-modality models.

The model is trained on the **RAVDESS dataset** and evaluated on an **unseen actor dataset** to test generalization.

---

# Model Components

## 1. Visual Model (Face Encoder)

* Backbone: **ResNet50**
* Pretrained on **ImageNet**
* Extracts **2048-dim facial features**

Input:

```
224 × 224 RGB face frame
```

Output:

```
2048-dim visual feature vector
```

---

## 2. Audio Model (Speech Encoder)

Audio is processed using **MFCC features** extracted from the speech signal.

Architecture:

```
MFCC (40 × T)
      │
1D CNN Layers
      │
BiLSTM (2 Layers)
      │
Temporal Average Pooling
      │
512-dim audio feature vector
```

---

## 3. Fusion Model

Features from both modalities are combined using **feature concatenation with attention gating**.

```
Visual Feature (2048)
        +
Audio Feature (512)
        │
Concatenation
        │
Attention Fusion
        │
Fully Connected Classifier
        │
Emotion Prediction
```

Output classes:

```
Neutral
Happy
Sad
Angry
Fearful
```

---

# Dataset

Dataset used: **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)**

Dataset structure used in this project:

```
dataset/raw/

Video_Speech_Actor_01/
Video_Speech_Actor_02/
Video_Song_Actor_01/
Video_Song_Actor_02/
```

For **final global testing**:

```
dataset/

Video_Speech_Actor_04/
Video_Song_Actor_04/
```

Only the following emotions are used:

| Code | Emotion |
| ---- | ------- |
| 01   | Neutral |
| 03   | Happy   |
| 04   | Sad     |
| 05   | Angry   |
| 06   | Fearful |

---

# Training Configuration

| Parameter      | Value                             |
| -------------- | --------------------------------- |
| Image Encoder  | ResNet50                          |
| Audio Features | MFCC (40)                         |
| Audio Model    | CNN + BiLSTM                      |
| Fusion Method  | Feature Concatenation + Attention |
| Epochs         | 20                                |
| Batch Size     | 8                                 |
| Learning Rate  | 1e-4                              |
| Optimizer      | Adam                              |
| Loss           | Cross Entropy                     |

---

# Training Results

Training + Validation Results:

```
Accuracy: 97%
```

Classification Report:

| Emotion | Precision | Recall | F1   |
| ------- | --------- | ------ | ---- |
| Neutral | 1.00      | 1.00   | 1.00 |
| Happy   | 1.00      | 1.00   | 1.00 |
| Sad     | 0.87      | 1.00   | 0.93 |
| Angry   | 1.00      | 0.94   | 0.97 |
| Fearful | 1.00      | 0.95   | 0.97 |

Overall Accuracy:

```
97%
```

---

# Global Testing Results (Unseen Actor)

Testing was performed on **Actor_04**, which was not used during training.

Dataset size:

```
144 samples
```

Accuracy:

```
90%
```

Classification Report:

| Emotion | Precision | Recall | F1   |
| ------- | --------- | ------ | ---- |
| Neutral | 0.89      | 1.00   | 0.94 |
| Happy   | 1.00      | 0.84   | 0.92 |
| Sad     | 1.00      | 1.00   | 1.00 |
| Angry   | 1.00      | 0.69   | 0.81 |
| Fearful | 0.71      | 1.00   | 0.83 |

Overall Accuracy:

```
90%
```

---

# Results Analysis

Observations:

* The **fusion model significantly improves accuracy** compared to single-modality systems.
* **Sad emotion is detected with perfect accuracy** in global testing.
* Some confusion occurs between:

  * **Angry ↔ Fearful**
  * **Happy ↔ Neutral**

This is expected because these emotions share similar **facial and vocal patterns**.

Despite testing on an **unseen actor**, the model maintains **90% accuracy**, showing strong generalization ability.

---

# How to Run the Project

## 1. Install Dependencies

```
pip install torch torchvision librosa av opencv-python seaborn tqdm
```

---

## 2. Train the Model

```
python main_pipeline.py
```

Outputs:

```
checkpoints/fusion_model.pth
results/
```

---

## 3. Run Global Testing

```
python test_global.py
```

Outputs:

```
results/global_test/
```

---

# Outputs Generated

Training outputs:

```
results/
    metrics.json
    report.txt
    confusion_matrix.png
    training_curve.png
```

Global test outputs:

```
results/global_test/
    metrics.json
    classification_report.txt
    predictions.csv
    confusion_matrix.png
```

---

# Future Improvements

Possible improvements for future work:

* Use **Wav2Vec2 for audio representation**
* Use **Vision Transformers for facial features**
* Implement **cross-modal transformer fusion**
* Perform **speaker-independent evaluation**
* Add **data augmentation for audio and video**

---

# Conclusion

This project demonstrates an effective **multimodal emotion recognition system** using **audio-visual fusion**.

Key achievements:

* **97% accuracy on training/validation**
* **90% accuracy on unseen actor testing**
* Robust multimodal architecture combining **visual and speech features**

The system confirms that **combining audio and visual information improves emotion recognition performance** compared to single-modality approaches.

---
