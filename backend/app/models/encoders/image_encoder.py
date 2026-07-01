from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torchvision.models as models

from app.models.base.base_encoder import BaseEncoder


class ImageEncoder(BaseEncoder):
    def __init__(
        self,
        embedding_dim: int = 512,
        output_dim: int | None = None,
        pretrained: bool = True,
        dropout: float = 0.0,
        normalize: bool = False,
        freeze_backbone: bool = False,
    ):
        super().__init__(embedding_dim=embedding_dim, output_dim=output_dim)
        resnet = models.resnet18(weights="DEFAULT" if pretrained else None)
        self._input_channels = 3

        modules = list(resnet.children())[:-2]
        self.backbone = nn.Sequential(*modules)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        in_features = resnet.fc.in_features
        projection_layers: list[nn.Module] = []
        if dropout > 0:
            projection_layers.append(nn.Dropout(dropout))
        projection_layers.append(nn.Linear(in_features, self._output_dim))
        self.projection = nn.Sequential(*projection_layers)

        self._normalize = normalize
        self._dropout = dropout

    def encode(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        x = self.backbone(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.projection(x)
        if self._normalize:
            x = nn.functional.normalize(x, p=2, dim=1)
        return x

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.encode(x, **kwargs)
