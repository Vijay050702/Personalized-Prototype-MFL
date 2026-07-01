from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from app.models.base.base_encoder import BaseEncoder


class SensorEncoder(BaseEncoder):
    def __init__(
        self,
        embedding_dim: int = 128,
        output_dim: int | None = None,
        input_channels: int = 9,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
        pooling: str = "mean",
        normalize: bool = False,
        encoder_type: str = "bilstm",
    ):
        super().__init__(embedding_dim=embedding_dim, output_dim=output_dim)
        self._pooling = pooling
        self._normalize = normalize
        self._encoder_type = encoder_type

        if encoder_type == "bilstm":
            self.encoder: nn.Module = BiLSTMEncoder(
                input_channels=input_channels,
                hidden_dim=hidden_dim,
                embedding_dim=embedding_dim,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=bidirectional,
            )
        elif encoder_type == "tcnn":
            self.encoder = TemporalCNNEncoder(
                input_channels=input_channels,
                embedding_dim=embedding_dim,
                dropout=dropout,
            )
        else:
            raise ValueError(
                f"Unknown encoder_type: {encoder_type}. Choose 'bilstm' or 'tcnn'."
            )

    def encode(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        x = self.encoder(x)
        if self._normalize:
            x = nn.functional.normalize(x, p=2, dim=1)
        return x

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.encode(x, **kwargs)


class BiLSTMEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_dim: int,
        embedding_dim: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.projection = nn.Linear(lstm_out_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        x = lstm_out.mean(dim=1)
        x = self.projection(x)
        return x


class TemporalCNNEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        embedding_dim: int,
        dropout: float,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = input_channels
        for out_ch, kernel, stride in [
            (64, 7, 2),
            (128, 5, 2),
            (256, 3, 2),
        ]:
            layers.extend(
                [
                    nn.Conv1d(
                        in_ch, out_ch, kernel, stride=stride, padding=kernel // 2
                    ),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            in_ch = out_ch

        self.conv_layers = nn.Sequential(*layers)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.projection = nn.Linear(in_ch, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.adaptive_pool(x).squeeze(-1)
        x = self.projection(x)
        return x
