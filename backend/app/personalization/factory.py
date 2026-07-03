from __future__ import annotations

from typing import Any

from app.knowledge_transfer.cross_modal_mapper import CrossModalMapper
from app.knowledge_transfer.factory import TransferFactory as KTFactory
from app.knowledge_transfer.inference import InferenceEngine
from app.knowledge_transfer.modality_graph import ModalityGraph
from app.knowledge_transfer.prototype_generator import PrototypeGenerator
from app.personalization.adaptation import AdaptationEngine
from app.personalization.fusion_engine import FusionEngine
from app.personalization.gating_network import GatingNetwork
from app.personalization.losses import CombinedPersonalizationLoss
from app.personalization.metrics import PersonalizationMetrics
from app.personalization.personalized_memory import PersonalizedMemory
from app.personalization.prototype_selector import PrototypeSelector
from app.personalization.weighting import WeightCalculator


class PersonalizationFactory:
    @staticmethod
    def create_gating_network(
        input_dim: int,
        num_sources: int = 3,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
        normalization: str = "softmax",
        temperature: float = 1.0,
    ) -> GatingNetwork:
        return GatingNetwork(
            input_dim=input_dim,
            num_sources=num_sources,
            hidden_dims=hidden_dims,
            dropout=dropout,
            normalization=normalization,
            temperature=temperature,
        )

    @staticmethod
    def create_weight_calculator(
        strategy: str = "fixed",
        fixed_weights: dict[str, float] | None = None,
        similarity_metric: str = "cosine",
        temperature: float = 1.0,
    ) -> WeightCalculator:
        return WeightCalculator(
            strategy=strategy,
            fixed_weights=fixed_weights,
            similarity_metric=similarity_metric,
            temperature=temperature,
        )

    @staticmethod
    def create_fusion_engine(
        strategy: str = "weighted_sum",
        modalities: dict[str, int] | None = None,
        mappings: list[tuple[str, str]] | None = None,
        gating_network: GatingNetwork | None = None,
        weight_calculator: WeightCalculator | None = None,
    ) -> FusionEngine:
        adaptive_gate = None
        if strategy == "adaptive":
            gate = gating_network or PersonalizationFactory.create_gating_network(
                input_dim=_compute_gate_input_dim(modalities),
            )
            from app.personalization.adaptive_gate import AdaptiveGate

            adaptive_gate = AdaptiveGate(gating_network=gate)

        return FusionEngine(
            strategy=strategy,
            weight_calculator=weight_calculator,
            adaptive_gate=adaptive_gate,
        )

    @staticmethod
    def create_prototype_selector(
        confidence_threshold: float = 0.3,
    ) -> PrototypeSelector:
        return PrototypeSelector(
            confidence_threshold=confidence_threshold,
        )

    @staticmethod
    def create_adaptation_engine(
        strategy: str = "ema",
        ema_alpha: float = 0.9,
        momentum: float = 0.9,
    ) -> AdaptationEngine:
        return AdaptationEngine(
            strategy=strategy,
            ema_alpha=ema_alpha,
            momentum=momentum,
        )

    @staticmethod
    def create_loss(
        fusion_weight: float = 1.0,
        consistency_weight: float = 1.0,
        personalization_weight: float = 1.0,
        regularization_weight: float = 0.1,
        adaptive_weighting_weight: float = 0.1,
    ) -> CombinedPersonalizationLoss:
        return CombinedPersonalizationLoss(
            fusion_weight=fusion_weight,
            consistency_weight=consistency_weight,
            personalization_weight=personalization_weight,
            regularization_weight=regularization_weight,
            adaptive_weighting_weight=adaptive_weighting_weight,
        )

    @staticmethod
    def create_metrics(
        similarity_metric: str = "cosine",
    ) -> PersonalizationMetrics:
        return PersonalizationMetrics(similarity_metric=similarity_metric)

    @staticmethod
    def create_memory(capacity: int = 1000) -> PersonalizedMemory:
        return PersonalizedMemory(capacity=capacity)

    @staticmethod
    def create_from_config(
        config: dict[str, Any],
    ) -> dict[str, Any]:
        components: dict[str, Any] = {}

        mod_config = config.get("modalities", {"image": 8, "text": 8})
        mappings = config.get("mappings", [("image", "text")])

        kt_components = KTFactory.create_from_config(
            {
                "modalities": mod_config,
                "mappings": mappings,
                "loss": config.get("kt_loss", {"type": "info_nce"}),
            }
        )
        components["mapper"] = kt_components.get("mapper")
        components["graph"] = kt_components.get("graph")

        gate_cfg = config.get("gating_network", {})
        gate = PersonalizationFactory.create_gating_network(
            input_dim=gate_cfg.get("input_dim", _compute_gate_input_dim(mod_config)),
            num_sources=gate_cfg.get("num_sources", 3),
            hidden_dims=gate_cfg.get("hidden_dims"),
            dropout=gate_cfg.get("dropout", 0.1),
            normalization=gate_cfg.get("normalization", "softmax"),
            temperature=gate_cfg.get("temperature", 1.0),
        )
        components["gating_network"] = gate

        wc_cfg = config.get("weight_calculator", {})
        wc = PersonalizationFactory.create_weight_calculator(
            strategy=wc_cfg.get("strategy", "fixed"),
            fixed_weights=wc_cfg.get("fixed_weights"),
        )
        components["weight_calculator"] = wc

        fusion_cfg = config.get("fusion", {})
        engine = PersonalizationFactory.create_fusion_engine(
            strategy=fusion_cfg.get("strategy", "weighted_sum"),
            modalities=mod_config,
            mappings=mappings,
            gating_network=gate,
            weight_calculator=wc,
        )
        components["fusion_engine"] = engine

        selector_cfg = config.get("selector", {})
        components["prototype_selector"] = (
            PersonalizationFactory.create_prototype_selector(
                confidence_threshold=selector_cfg.get("confidence_threshold", 0.3),
            )
        )

        adapt_cfg = config.get("adaptation", {})
        components["adaptation_engine"] = (
            PersonalizationFactory.create_adaptation_engine(
                strategy=adapt_cfg.get("strategy", "ema"),
                ema_alpha=adapt_cfg.get("ema_alpha", 0.9),
                momentum=adapt_cfg.get("momentum", 0.9),
            )
        )

        loss_cfg = config.get("loss", {})
        components["loss"] = PersonalizationFactory.create_loss(
            fusion_weight=loss_cfg.get("fusion_weight", 1.0),
            consistency_weight=loss_cfg.get("consistency_weight", 1.0),
            personalization_weight=loss_cfg.get("personalization_weight", 1.0),
            regularization_weight=loss_cfg.get("regularization_weight", 0.1),
            adaptive_weighting_weight=loss_cfg.get("adaptive_weighting_weight", 0.1),
        )

        components["metrics"] = PersonalizationFactory.create_metrics(
            similarity_metric=config.get("similarity_metric", "cosine"),
        )

        components["memory"] = PersonalizationFactory.create_memory(
            capacity=config.get("memory_capacity", 1000),
        )

        return components


def _compute_gate_input_dim(
    modalities: dict[str, int] | None,
) -> int:
    if not modalities:
        return 16
    total_emb_dim = sum(modalities.values()) if modalities else 16
    num_mods = len(modalities) if modalities else 2
    num_pairs = num_mods * (num_mods - 1) // 2
    return total_emb_dim + num_pairs + 4
