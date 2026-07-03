from __future__ import annotations

from typing import Any, Callable

import torch.nn as nn

from app.personalization.adaptive_gate import AdaptiveGate
from app.personalization.adaptation import AdaptationEngine
from app.personalization.fusion_engine import FusionEngine
from app.personalization.gating_network import GatingNetwork
from app.personalization.losses import (
    CombinedPersonalizationLoss,
    ConsistencyLoss,
    FusionLoss,
    PersonalizationLoss,
    PrototypeRegularizationLoss,
)
from app.personalization.weighting import WeightCalculator


class PersonalizationRegistry:
    def __init__(self):
        self._fusion_strategies: dict[str, str] = {
            "weighted_sum": "weighted_sum",
            "adaptive": "adaptive",
            "confidence_weighted": "confidence_weighted",
            "learnable": "learnable",
        }
        self._gating_networks: dict[str, type[GatingNetwork]] = {
            "default": GatingNetwork,
        }
        self._adaptation_methods: dict[str, str] = {
            "ema": "ema",
            "momentum": "momentum",
            "residual": "residual",
            "adaptive_blending": "adaptive_blending",
        }
        self._weight_calculators: dict[str, str] = {
            "fixed": "fixed",
            "confidence": "confidence",
            "similarity": "similarity",
            "adaptive": "adaptive",
            "learnable": "learnable",
        }
        self._loss_functions: dict[str, type[nn.Module]] = {
            "fusion": FusionLoss,
            "consistency": ConsistencyLoss,
            "personalization": PersonalizationLoss,
            "prototype_regularization": PrototypeRegularizationLoss,
            "combined": CombinedPersonalizationLoss,
        }
        self._custom_factories: dict[str, Callable[..., Any]] = {}

    def register_fusion_strategy(self, name: str) -> None:
        if name in self._fusion_strategies:
            raise ValueError(f"Fusion strategy '{name}' is already registered")
        self._fusion_strategies[name] = name

    def get_fusion_strategy(self, name: str) -> str:
        if name not in self._fusion_strategies:
            raise ValueError(
                f"Unknown fusion strategy '{name}'. "
                f"Available: {self.list_fusion_strategies()}"
            )
        return self._fusion_strategies[name]

    def list_fusion_strategies(self) -> list[str]:
        return sorted(self._fusion_strategies.keys())

    def register_gating_network(self, name: str, cls: type[GatingNetwork]) -> None:
        if name in self._gating_networks:
            raise ValueError(f"Gating network '{name}' is already registered")
        self._gating_networks[name] = cls

    def get_gating_network(self, name: str) -> type[GatingNetwork]:
        if name not in self._gating_networks:
            raise ValueError(
                f"Unknown gating network '{name}'. "
                f"Available: {self.list_gating_networks()}"
            )
        return self._gating_networks[name]

    def list_gating_networks(self) -> list[str]:
        return sorted(self._gating_networks.keys())

    def register_adaptation_method(self, name: str) -> None:
        if name in self._adaptation_methods:
            raise ValueError(f"Adaptation method '{name}' is already registered")
        self._adaptation_methods[name] = name

    def get_adaptation_method(self, name: str) -> str:
        if name not in self._adaptation_methods:
            raise ValueError(
                f"Unknown adaptation method '{name}'. "
                f"Available: {self.list_adaptation_methods()}"
            )
        return self._adaptation_methods[name]

    def list_adaptation_methods(self) -> list[str]:
        return sorted(self._adaptation_methods.keys())

    def register_weight_calculator(self, name: str) -> None:
        if name in self._weight_calculators:
            raise ValueError(f"Weight calculator '{name}' is already registered")
        self._weight_calculators[name] = name

    def get_weight_calculator(self, name: str) -> str:
        if name not in self._weight_calculators:
            raise ValueError(
                f"Unknown weight calculator '{name}'. "
                f"Available: {self.list_weight_calculators()}"
            )
        return self._weight_calculators[name]

    def list_weight_calculators(self) -> list[str]:
        return sorted(self._weight_calculators.keys())

    def register_loss_function(self, name: str, cls: type[nn.Module]) -> None:
        if name in self._loss_functions:
            raise ValueError(f"Loss function '{name}' is already registered")
        self._loss_functions[name] = cls

    def get_loss_function(self, name: str) -> type[nn.Module]:
        if name not in self._loss_functions:
            raise ValueError(
                f"Unknown loss function '{name}'. "
                f"Available: {self.list_loss_functions()}"
            )
        return self._loss_functions[name]

    def list_loss_functions(self) -> list[str]:
        return sorted(self._loss_functions.keys())

    def register_component(self, name: str, factory: Callable[..., Any]) -> None:
        if name in self._custom_factories:
            raise ValueError(f"Component '{name}' is already registered")
        self._custom_factories[name] = factory

    def get_component(self, name: str, **kwargs: Any) -> Any:
        if name not in self._custom_factories:
            raise ValueError(
                f"Unknown component '{name}'. Available: {self.list_components()}"
            )
        return self._custom_factories[name](**kwargs)

    def list_components(self) -> list[str]:
        return sorted(self._custom_factories.keys())

    def to_config(self) -> dict[str, Any]:
        return {
            "fusion_strategies": self.list_fusion_strategies(),
            "gating_networks": self.list_gating_networks(),
            "adaptation_methods": self.list_adaptation_methods(),
            "weight_calculators": self.list_weight_calculators(),
            "loss_functions": self.list_loss_functions(),
            "custom_components": self.list_components(),
        }
