import torch
import torch.nn as nn
import timm


class FaceEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            num_classes=0
        )

        self.output_dim = self.backbone.num_features

    def forward(self, x):

        features = self.backbone(x)

        return features


class AudioEncoder(nn.Module):

    def __init__(self, input_dim=40):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.output_dim = 128

    def forward(self, x):

        return self.network(x)


class FusionModel(nn.Module):

    def __init__(self, num_classes=5):

        super().__init__()

        self.face_encoder = FaceEncoder()

        self.audio_encoder = AudioEncoder()

        fusion_dim = self.face_encoder.output_dim + self.audio_encoder.output_dim

        self.classifier = nn.Sequential(

            nn.Linear(fusion_dim, 512),

            nn.BatchNorm1d(512),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(512, 128),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(128, num_classes)
        )

    def forward(self, face, audio):

        face_features = self.face_encoder(face)

        audio_features = self.audio_encoder(audio)

        fused = torch.cat((face_features, audio_features), dim=1)

        output = self.classifier(fused)

        return output


if __name__ == "__main__":

    model = FusionModel()

    face = torch.randn(8, 3, 224, 224)
    audio = torch.randn(8, 40)

    out = model(face, audio)

    print("Output shape:", out.shape)