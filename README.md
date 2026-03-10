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