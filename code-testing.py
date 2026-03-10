from torch.utils.data import DataLoader
from dataloaders.fusion_dataset import FusionDataset

dataset = FusionDataset(split="train")

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)

for face, audio, label in loader:

    print(face.shape)
    print(audio.shape)
    print(label.shape)

    break