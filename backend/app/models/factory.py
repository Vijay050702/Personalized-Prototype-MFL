from __future__ import annotations

from typing import Any

import torch.nn as nn

from app.core.logging import logger
from app.models.base.base_encoder import BaseEncoder
from app.models.classifier.classifier_head import ClassifierHead
from app.models.encoders.audio_encoder import AudioEncoder
from app.models.encoders.image_encoder import ImageEncoder
from app.models.encoders.sensor_encoder import SensorEncoder
from app.models.encoders.text_encoder import TextEncoder
from app.models.fusion.multimodal_fusion import MultimodalFusion
from app.models.projection.projection_head import ProjectionHead


_ENCODER_REGISTRY: dict[str, type[BaseEncoder]] = {}


def register_encoder(name: str) -> Any:
    def decorator(cls: type[BaseEncoder]) -> type[BaseEncoder]:
        _ENCODER_REGISTRY[name] = cls
        logger.debug(f"Registered encoder: {name} -> {cls.__name__}")
        return cls

    return decorator


def get_encoder_class(name: str) -> type[BaseEncoder]:
    if name not in _ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown encoder '{name}'. Registered: {list(_ENCODER_REGISTRY.keys())}"
        )
    return _ENCODER_REGISTRY[name]


register_encoder("image")(ImageEncoder)
register_encoder("text")(TextEncoder)
register_encoder("audio")(AudioEncoder)
register_encoder("sensor")(SensorEncoder)


class ModelFactory:
    @staticmethod
    def create_encoder(
        modality: str,
        embedding_dim: int = 256,
        output_dim: int | None = None,
        **kwargs: Any,
    ) -> BaseEncoder:
        encoder_cls = get_encoder_class(modality)
        encoder = encoder_cls(
            embedding_dim=embedding_dim,
            output_dim=output_dim,
            **kwargs,
        )
        logger.info(
            f"Created {modality} encoder: {type(encoder).__name__} "
            f"(embed_dim={embedding_dim}, params={encoder.num_parameters:,})"
        )
        return encoder

    @staticmethod
    def create_fusion(
        embed_dim: int = 512,
        strategy: str = "concat",
        num_heads: int = 4,
        dropout: float = 0.1,
        projection_dim: int | None = None,
    ) -> MultimodalFusion:
        fusion = MultimodalFusion(
            embed_dim=embed_dim,
            strategy=strategy,
            num_heads=num_heads,
            dropout=dropout,
            projection_dim=projection_dim,
        )
        logger.info(f"Created fusion: {strategy} (params={fusion.num_parameters:,})")
        return fusion

    @staticmethod
    def create_projection_head(
        input_dim: int,
        output_dim: int = 128,
        hidden_dim: int | None = None,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        normalize_output: bool = True,
    ) -> ProjectionHead:
        proj = ProjectionHead(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
            normalize_output=normalize_output,
        )
        logger.info(
            f"Created projection head: {input_dim}->{output_dim} "
            f"(params={proj.num_parameters:,})"
        )
        return proj

    @staticmethod
    def create_classifier(
        input_dim: int,
        num_classes: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
        use_layer_norm: bool = False,
    ) -> ClassifierHead:
        classifier = ClassifierHead(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        )
        logger.info(
            f"Created classifier: {input_dim}->{num_classes} "
            f"(params={classifier.num_parameters:,})"
        )
        return classifier
