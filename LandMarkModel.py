import torch
import torch.nn as nn


class LandmarkCnnModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),  # 3x244x244 to 16x244x244
            nn.BatchNorm2d(16),  # 16x244x244
            nn.ReLU(),  # 16x244x244
            nn.MaxPool2d(2, 2),  # to 16x112x112

            nn.Conv2d(16, 32, 3, padding=1),  # to 32x112x112
            nn.BatchNorm2d(32),  # 32x112x112
            nn.ReLU(),  # 32x112x112
            nn.MaxPool2d(2, 2),  # to 32x56x56

            nn.Conv2d(32, 64, 3, padding=1),  # to 64x56x56
            nn.BatchNorm2d(64),  # 64x56x56
            nn.ReLU(),  # 64x56x56
            nn.MaxPool2d(2, 2),  # to 64x28x28

            nn.Conv2d(64, 128, 3, padding=1),  # to 128x28x28
            nn.BatchNorm2d(128),  # 128x28x28
            nn.ReLU(),  # 128x28x28
            nn.MaxPool2d(2, 2),  # to 128x14x14

            nn.Flatten(),  # to 25088

            nn.Linear(128 * 14 * 14, 256),  # -> 256
            nn.Dropout(p=dropout),

            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.model(x)
