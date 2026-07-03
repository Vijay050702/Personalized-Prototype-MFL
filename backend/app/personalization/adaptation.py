from __future__ import annotations

import time
from typing import Any

import torch

from app.core.logging import logger
from app.personalization.personalized_prototype import PersonalizedPrototype
from app.personalization.profile import ClientProfile
from app.personalization.utils import PersonalizationLogger, compute_prototype_drift


class AdaptationEngine:
    def __init__(
        self,
        strategy: str = "ema",
        ema_alpha: float = 0.9,
        momentum: float = 0.9,
        logger_instance: PersonalizationLogger | None = None,
    ):
        if strategy not in {"ema", "momentum", "residual", "adaptive_blending"}:
            raise ValueError(
                f"Unknown adaptation strategy '{strategy}'. "
                f"Choose from: ema, momentum, residual, adaptive_blending"
            )
        if not (0.0 <= ema_alpha <= 1.0):
            raise ValueError(f"ema_alpha must be in [0.0, 1.0], got {ema_alpha}")
        if not (0.0 <= momentum <= 1.0):
            raise ValueError(f"momentum must be in [0.0, 1.0], got {momentum}")

        self._strategy = strategy
        self._ema_alpha = ema_alpha
        self._momentum = momentum
        self._previous: dict[str, torch.Tensor] = {}
        self._logger = logger_instance or PersonalizationLogger()

    def adapt(
        self,
        prototype: PersonalizedPrototype,
        client_profile: ClientProfile | None = None,
    ) -> PersonalizedPrototype:
        if prototype.personalized_prototype is None:
            raise ValueError("Personalized prototype vector is None, cannot adapt")

        current = torch.tensor(prototype.personalized_prototype, dtype=torch.float32)
        key = f"{prototype.client_id}_{prototype.class_id}_{prototype.modality}"

        if self._strategy == "ema":
            adapted = self._ema_adapt(current, key)
        elif self._strategy == "momentum":
            adapted = self._momentum_adapt(current, key)
        elif self._strategy == "residual":
            adapted = self._residual_adapt(current, key)
        elif self._strategy == "adaptive_blending":
            adapted = self._adaptive_blend(current, prototype)
        else:
            adapted = current

        drift = compute_prototype_drift(adapted, current).item()

        if client_profile is not None:
            client_profile.update_drift(drift)

        self._previous[key] = adapted.clone()

        prototype.personalized_prototype = adapted.detach().cpu().tolist()

        self._logger.log_adaptation(
            client_id=prototype.client_id,
            strategy=self._strategy,
            drift=drift,
            step=client_profile.training_steps if client_profile else 0,
        )

        return prototype

    def _ema_adapt(
        self,
        current: torch.Tensor,
        key: str,
    ) -> torch.Tensor:
        if key not in self._previous:
            return current.clone()
        prev = self._previous[key]
        return self._ema_alpha * prev + (1.0 - self._ema_alpha) * current

    def _momentum_adapt(
        self,
        current: torch.Tensor,
        key: str,
    ) -> torch.Tensor:
        if key not in self._previous:
            return current.clone()
        prev = self._previous[key]
        return prev + self._momentum * (current - prev)

    def _residual_adapt(
        self,
        current: torch.Tensor,
        key: str,
    ) -> torch.Tensor:
        if key not in self._previous:
            return current.clone()
        prev = self._previous[key]
        residual = current - prev
        return current + 0.1 * residual

    def _adaptive_blend(
        self,
        current: torch.Tensor,
        prototype: PersonalizedPrototype,
    ) -> torch.Tensor:
        if prototype.fusion_weights:
            local_w = prototype.fusion_weights.get("local", 0.0)
            global_w = prototype.fusion_weights.get("global", 0.0)
            blend = 0.5 + 0.5 * (local_w - global_w)
            blend = max(0.0, min(1.0, blend))
        else:
            blend = 0.5
        key = f"{prototype.client_id}_{prototype.class_id}_{prototype.modality}"
        if key not in self._previous:
            return current.clone()
        prev = self._previous[key]
        return blend * current + (1.0 - blend) * prev

    @property
    def strategy(self) -> str:
        return self._strategy

    @property
    def ema_alpha(self) -> float:
        return self._ema_alpha

    def to_config(self) -> dict[str, Any]:
        return {
            "strategy": self._strategy,
            "ema_alpha": self._ema_alpha,
            "momentum": self._momentum,
        }
