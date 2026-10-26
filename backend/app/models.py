from __future__ import annotations

import torch
from torch import nn

from .model_config import CLASS_LABELS, CREMA_D_CLASS_LABELS, MODEL_ARCHITECTURES


class ConvEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded = self.layers(inputs)
        return encoded.mean(dim=2).transpose(1, 2)


class EmotionCNN(nn.Module):
    def __init__(self, class_count: int = len(CLASS_LABELS)) -> None:
        super().__init__()
        self.encoder = ConvEncoder()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(96, class_count),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence = self.encoder(inputs)
        return self.head(sequence.transpose(1, 2))


class EmotionCNNLSTM(nn.Module):
    def __init__(self, class_count: int = len(CLASS_LABELS)) -> None:
        super().__init__()
        self.encoder = ConvEncoder()
        self.recurrent = nn.LSTM(
            input_size=96,
            hidden_size=96,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.head = nn.Sequential(nn.Dropout(0.35), nn.Linear(96, class_count))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence = self.encoder(inputs)
        recurrent, _ = self.recurrent(sequence)
        return self.head(recurrent[:, -1, :])


class EmotionCNNBiLSTMAttention(nn.Module):
    def __init__(self, class_count: int = len(CLASS_LABELS)) -> None:
        super().__init__()
        self.encoder = ConvEncoder()
        self.recurrent = nn.LSTM(
            input_size=96,
            hidden_size=96,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = nn.Linear(192, 1)
        self.head = nn.Sequential(nn.Dropout(0.35), nn.Linear(192, class_count))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence = self.encoder(inputs)
        recurrent, _ = self.recurrent(sequence)
        weights = torch.softmax(self.attention(recurrent).squeeze(-1), dim=1)
        attended = torch.sum(recurrent * weights.unsqueeze(-1), dim=1)
        return self.head(attended)


def create_model(architecture: str, class_count: int = len(CLASS_LABELS)) -> nn.Module:
    if architecture == "cnn":
        return EmotionCNN(class_count)
    if architecture == "cnn_lstm":
        return EmotionCNNLSTM(class_count)
    if architecture == "cnn_bilstm_attention":
        return EmotionCNNBiLSTMAttention(class_count)
    raise ValueError(f"Unsupported model architecture: {architecture}")
