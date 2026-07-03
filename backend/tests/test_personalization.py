from __future__ import annotations

import time
from typing import Any

import pytest
import torch
import torch.nn as nn

from app.federated.models import AggregatedPrototype
from app.knowledge_transfer.cross_modal_mapper import CrossModalMapper
from app.knowledge_transfer.inference import InferenceOutput
from app.knowledge_transfer.modality_graph import ModalityGraph
from app.knowledge_transfer.prototype_generator import SynthesisResult
from app.personalization.adaptive_gate import AdaptiveGate
from app.personalization.adaptation import AdaptationEngine
from app.personalization.confidence import PersonalizedConfidence
from app.personalization.factory import PersonalizationFactory
from app.personalization.fusion_engine import FusionEngine
from app.personalization.gating_network import GatingNetwork
from app.personalization.losses import (
    AdaptiveWeightingLoss,
    CombinedPersonalizationLoss,
    ConsistencyLoss,
    FusionLoss,
    PersonalizationLoss,
    PrototypeRegularizationLoss,
)
from app.personalization.metrics import PersonalizationMetrics
from app.personalization.personalized_memory import PersonalizedMemory
from app.personalization.personalized_prototype import PersonalizedPrototype
from app.personalization.profile import ClientProfile
from app.personalization.prototype_selector import PrototypeSelector
from app.personalization.registry import PersonalizationRegistry
from app.personalization.regularization import (
    FusionSmoothnessRegularization,
    PrototypeConsistencyRegularization,
    PrototypeStabilityRegularization,
    TemporalConsistencyRegularization,
)
from app.personalization.utils import PersonalizationLogger
from app.personalization.validation import (
    validate_confidence_range,
    validate_dimensions,
    validate_duplicate_prototypes,
    validate_fusion_sources,
    validate_missing_modalities,
    validate_shape_match,
    validate_weights_sum_to_one,
)
from app.personalization.weighting import WeightCalculator
from app.prototypes.prototype import Prototype

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_prototype(
    class_id: int = 0,
    modality: str = "image",
    dim: int = 8,
    confidence: float = 0.9,
) -> Prototype:
    return Prototype(
        embedding=torch.randn(dim),
        class_id=class_id,
        modality=modality,
        confidence=confidence,
    )


def _make_aggregated(
    class_id: int = 0,
    modality: str = "image",
    dim: int = 8,
    confidence: float = 0.9,
) -> AggregatedPrototype:
    return AggregatedPrototype(
        class_id=class_id,
        modality=modality,
        prototype_vector=[float(i) for i in range(dim)],
        embedding_dim=dim,
        sample_count=10,
        confidence=confidence,
    )


def _make_inference_output(
    class_id: int = 0,
    modality: str = "text",
    dim: int = 8,
    confidence: float = 0.7,
) -> InferenceOutput:
    return InferenceOutput(
        modality=modality,
        class_id=class_id,
        prototype_vector=[float(i) for i in range(dim)],
        embedding_dim=dim,
        confidence=confidence,
        source_modality="image",
        path=["image", "text"],
    )


# ===================================================================
# TestValidation
# ===================================================================


class TestValidation:
    def test_validate_dimensions_ok(self):
        validate_dimensions(torch.randn(8), 8, "test")

    def test_validate_dimensions_scalar(self):
        with pytest.raises(ValueError, match="scalar"):
            validate_dimensions(torch.tensor(1.0), 8)

    def test_validate_dimensions_mismatch(self):
        with pytest.raises(ValueError, match="dimension"):
            validate_dimensions(torch.randn(8), 16)

    def test_weights_sum_to_one_ok(self):
        validate_weights_sum_to_one({"local": 0.5, "global": 0.5}, ["local", "global"])

    def test_weights_sum_to_one_missing_source(self):
        with pytest.raises(ValueError, match="Missing weight"):
            validate_weights_sum_to_one({"local": 1.0}, ["local", "global"])

    def test_weights_sum_to_one_bad_sum(self):
        with pytest.raises(ValueError, match="expected 1.0"):
            validate_weights_sum_to_one(
                {"local": 1.0, "global": 1.0}, ["local", "global"]
            )

    def test_missing_modalities_ok(self):
        validate_missing_modalities({"image"}, {"text"}, {"image", "text"})

    def test_missing_modalities_unknown_available(self):
        with pytest.raises(ValueError, match="not in the known"):
            validate_missing_modalities({"unknown"}, {"text"}, {"image", "text"})

    def test_missing_modalities_unknown_missing(self):
        with pytest.raises(ValueError, match="not in the known"):
            validate_missing_modalities({"image"}, {"unknown"}, {"image", "text"})

    def test_missing_modalities_overlap(self):
        with pytest.raises(ValueError, match="cannot be both"):
            validate_missing_modalities({"image"}, {"image"}, {"image", "text"})

    def test_missing_modalities_empty(self):
        with pytest.raises(ValueError, match="No missing"):
            validate_missing_modalities({"image"}, set(), {"image"})

    def test_confidence_range_ok(self):
        validate_confidence_range(0.5)
        validate_confidence_range(0.0)
        validate_confidence_range(1.0)

    def test_confidence_range_low(self):
        with pytest.raises(ValueError, match="must be in"):
            validate_confidence_range(-0.1)

    def test_confidence_range_high(self):
        with pytest.raises(ValueError, match="must be in"):
            validate_confidence_range(1.1)

    def test_duplicate_prototypes_ok(self):
        p1 = PersonalizedPrototype(client_id="c1", class_id=0, modality="image")
        p2 = PersonalizedPrototype(client_id="c2", class_id=0, modality="image")
        validate_duplicate_prototypes([p1, p2])

    def test_duplicate_prototypes_raises(self):
        p1 = PersonalizedPrototype(client_id="c1", class_id=0, modality="image")
        p2 = PersonalizedPrototype(client_id="c1", class_id=0, modality="image")
        with pytest.raises(ValueError, match="Duplicate"):
            validate_duplicate_prototypes([p1, p2])

    def test_shape_match_ok(self):
        validate_shape_match(torch.randn(4, 8), torch.randn(4, 8))

    def test_shape_match_mismatch(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            validate_shape_match(torch.randn(4, 8), torch.randn(2, 8))

    def test_fusion_sources_ok(self):
        validate_fusion_sources(["local", "global", "cross_modal"])

    def test_fusion_sources_unknown(self):
        with pytest.raises(ValueError, match="Unknown fusion source"):
            validate_fusion_sources(["unknown"])


# ===================================================================
# TestPersonalizedPrototype
# ===================================================================


class TestPersonalizedPrototype:
    def test_create_minimal(self):
        pp = PersonalizedPrototype(client_id="c1", class_id=0, modality="image")
        assert pp.client_id == "c1"
        assert pp.class_id == 0
        assert pp.modality == "image"
        assert pp.fusion_weights == {}
        assert pp.confidence == 0.0

    def test_create_full(self):
        pp = PersonalizedPrototype(
            client_id="c1",
            class_id=1,
            modality="text",
            local_prototype=[1.0, 2.0],
            global_prototype=[3.0, 4.0],
            cross_modal_prototype=[5.0, 6.0],
            personalized_prototype=[2.5, 3.5],
            fusion_weights={"local": 0.5, "global": 0.5},
            confidence=0.85,
            embedding_dim=2,
            metadata={"source": "test"},
        )
        assert pp.confidence == 0.85
        assert pp.embedding_dim == 2
        assert pp.metadata["source"] == "test"

    def test_to_tensor(self):
        pp = PersonalizedPrototype(
            client_id="c1",
            class_id=0,
            modality="image",
            personalized_prototype=[1.0, 2.0, 3.0],
        )
        t = pp.to_tensor()
        assert isinstance(t, torch.Tensor)
        assert t.shape == (3,)

    def test_to_tensor_none_raises(self):
        pp = PersonalizedPrototype(client_id="c1", class_id=0, modality="image")
        with pytest.raises(ValueError, match="No personalized"):
            pp.to_tensor()

    def test_has_local(self):
        pp = PersonalizedPrototype(client_id="c1", class_id=0, modality="image")
        assert not pp.has_local()
        pp.local_prototype = [1.0]
        assert pp.has_local()

    def test_has_global(self):
        pp = PersonalizedPrototype(client_id="c1", class_id=0, modality="image")
        assert not pp.has_global()
        pp.global_prototype = [1.0]
        assert pp.has_global()

    def test_has_cross_modal(self):
        pp = PersonalizedPrototype(client_id="c1", class_id=0, modality="image")
        assert not pp.has_cross_modal()
        pp.cross_modal_prototype = [1.0]
        assert pp.has_cross_modal()

    def test_available_sources(self):
        pp = PersonalizedPrototype(client_id="c1", class_id=0, modality="image")
        assert pp.available_sources() == []
        pp.local_prototype = [1.0]
        pp.global_prototype = [1.0]
        assert sorted(pp.available_sources()) == ["global", "local"]

    def test_to_dict(self):
        pp = PersonalizedPrototype(client_id="c1", class_id=0, modality="image")
        d = pp.to_dict()
        assert d["client_id"] == "c1"
        assert d["class_id"] == 0
        assert d["modality"] == "image"


# ===================================================================
# TestClientProfile
# ===================================================================


class TestClientProfile:
    def test_create(self):
        p = ClientProfile(client_id="c1")
        assert p.client_id == "c1"
        assert p.training_steps == 0

    def test_update_modalities(self):
        p = ClientProfile(client_id="c1")
        p.update_modalities({"image", "text"}, {"image", "text", "audio"})
        assert p.available_modalities == {"image", "text"}
        assert p.missing_modalities == {"audio"}

    def test_record_training_step(self):
        p = ClientProfile(client_id="c1")
        p.record_training_step()
        assert p.training_steps == 1

    def test_record_prototype(self):
        p = ClientProfile(client_id="c1")
        p.record_prototype(0, "image", 0.9)
        assert len(p.prototype_history) == 1

    def test_record_confidence(self):
        p = ClientProfile(client_id="c1")
        p.record_confidence("image", 0, 0.85)
        assert len(p.confidence_history) == 1

    def test_update_drift(self):
        p = ClientProfile(client_id="c1")
        p.update_drift(0.5)
        assert p.prototype_drift == 0.5
        assert p.drift_history == [0.5]

    def test_average_confidence_empty(self):
        p = ClientProfile(client_id="c1")
        assert p.average_confidence == 0.0

    def test_average_confidence(self):
        p = ClientProfile(client_id="c1")
        p.record_confidence("image", 0, 0.8)
        p.record_confidence("image", 0, 0.9)
        assert p.average_confidence == pytest.approx(0.85)

    def test_confidence_trend_insufficient(self):
        p = ClientProfile(client_id="c1")
        assert p.confidence_trend == 0.0

    def test_confidence_trend(self):
        p = ClientProfile(client_id="c1")
        for c in [0.5, 0.6, 0.7, 0.8, 0.9]:
            p.record_confidence("image", 0, c)
        assert p.confidence_trend == pytest.approx(0.4, abs=1e-4)

    def test_average_drift_empty(self):
        p = ClientProfile(client_id="c1")
        assert p.average_drift == 0.0

    def test_average_drift(self):
        p = ClientProfile(client_id="c1")
        p.update_drift(0.1)
        p.update_drift(0.3)
        assert p.average_drift == 0.2

    def test_to_dict(self):
        p = ClientProfile(client_id="c1")
        d = p.to_dict()
        assert d["client_id"] == "c1"
        assert "average_confidence" in d


# ===================================================================
# TestPersonalizedConfidence
# ===================================================================


class TestPersonalizedConfidence:
    def test_estimate_base_only(self):
        pc = PersonalizedConfidence()
        conf = pc.estimate(base_confidence=0.8)
        assert conf == pytest.approx(0.8)

    def test_estimate_with_variance(self):
        pc = PersonalizedConfidence()
        conf = pc.estimate(base_confidence=0.8, prototype_variance=0.0)
        assert conf > 0.8

    def test_estimate_with_consistency(self):
        pc = PersonalizedConfidence()
        conf = pc.estimate(base_confidence=0.5, consistency=0.9)
        assert conf > 0.5

    def test_estimate_with_similarity(self):
        pc = PersonalizedConfidence()
        conf = pc.estimate(base_confidence=0.5, similarity_score=1.0)
        assert conf > 0.5

    def test_estimate_with_history(self):
        pc = PersonalizedConfidence()
        conf = pc.estimate(base_confidence=0.5, history_confidences=[0.5, 0.6, 0.55])
        assert 0.0 <= conf <= 1.0

    def test_estimate_clamps(self):
        pc = PersonalizedConfidence()
        conf = pc.estimate(base_confidence=2.0)
        assert conf == 1.0

    def test_batch_estimate(self):
        pc = PersonalizedConfidence()
        confs = pc.batch_estimate(
            base_confidences=[0.5, 0.8],
            prototype_variances=[0.1, None],
        )
        assert len(confs) == 2
        assert all(0.0 <= c <= 1.0 for c in confs)

    def test_stability_single(self):
        pc = PersonalizedConfidence()
        assert pc._stability([0.5]) == 1.0


# ===================================================================
# TestGatingNetwork
# ===================================================================


class TestGatingNetwork:
    def test_forward_softmax(self):
        net = GatingNetwork(input_dim=16, num_sources=3, normalization="softmax")
        x = torch.randn(16)
        w = net(x)
        assert w.shape == (3,)
        assert w.sum().item() == pytest.approx(1.0, abs=1e-4)

    def test_forward_sigmoid(self):
        net = GatingNetwork(input_dim=16, num_sources=3, normalization="sigmoid")
        x = torch.randn(16)
        w = net(x)
        assert w.shape == (3,)
        assert w.sum().item() == pytest.approx(1.0, abs=1e-4)

    def test_forward_batch(self):
        net = GatingNetwork(input_dim=16, num_sources=3)
        x = torch.randn(5, 16)
        w = net(x)
        assert w.shape == (5, 3)

    def test_invalid_normalization(self):
        with pytest.raises(ValueError, match="normalization"):
            GatingNetwork(input_dim=16, normalization="invalid")

    def test_invalid_temperature(self):
        with pytest.raises(ValueError, match="Temperature"):
            GatingNetwork(input_dim=16, temperature=0.0)

    def test_properties(self):
        net = GatingNetwork(
            input_dim=16, num_sources=3, normalization="sigmoid", temperature=0.5
        )
        assert net.num_sources == 3
        assert net.normalization == "sigmoid"
        assert net.temperature == 0.5

    def test_to_config(self):
        net = GatingNetwork(input_dim=16, num_sources=2)
        cfg = net.to_config()
        assert cfg["num_sources"] == 2
        assert cfg["normalization"] == "softmax"

    def test_grad_flow(self):
        net = GatingNetwork(input_dim=8, num_sources=2)
        x = torch.randn(8)
        w = net(x)
        loss = w.sum()
        loss.backward()
        for p in net.parameters():
            assert p.grad is not None


# ===================================================================
# TestAdaptiveGate
# ===================================================================


class TestAdaptiveGate:
    def test_compute_weights(self):
        net = GatingNetwork(input_dim=18, num_sources=2)
        gate = AdaptiveGate(gating_network=net)
        embs = {
            "local": torch.randn(8),
            "global": torch.randn(8),
        }
        weights = gate.compute_weights(embs, global_confidence=0.8)
        assert sorted(weights.keys()) == ["global", "local"]
        assert abs(sum(weights.values()) - 1.0) < 1e-4

    def test_compute_weights_with_profile(self):
        net = GatingNetwork(input_dim=21, num_sources=2)
        gate = AdaptiveGate(gating_network=net)
        embs = {"local": torch.randn(8), "global": torch.randn(8)}
        weights = gate.compute_weights(
            embs, global_confidence=0.8, profile_features=[0.5, 0.1, 0.2]
        )
        assert abs(sum(weights.values()) - 1.0) < 1e-4

    def test_compute_weights_single_source(self):
        net = GatingNetwork(input_dim=8, num_sources=1)
        gate = AdaptiveGate(gating_network=net)
        embs = {"local": torch.randn(8)}
        weights = gate.compute_weights(embs)
        assert weights == {"local": 1.0}

    def test_properties(self):
        net = GatingNetwork(input_dim=8, num_sources=2)
        gate = AdaptiveGate(gating_network=net)
        assert gate.gating_network is net

    def test_to_config(self):
        net = GatingNetwork(input_dim=8, num_sources=2)
        gate = AdaptiveGate(gating_network=net)
        cfg = gate.to_config()
        assert "gating_network" in cfg

    def test_compute_weights_no_features(self):
        net = GatingNetwork(input_dim=8, num_sources=1)
        gate = AdaptiveGate(gating_network=net)
        weights = gate.compute_weights({})
        assert weights == {}


# ===================================================================
# TestWeightCalculator
# ===================================================================


class TestWeightCalculator:
    def test_fixed_strategy(self):
        wc = WeightCalculator(
            strategy="fixed", fixed_weights={"local": 0.7, "global": 0.3}
        )
        weights = wc.compute(["local", "global"])
        assert weights["local"] == 0.7
        assert weights["global"] == 0.3

    def test_fixed_default_equal(self):
        wc = WeightCalculator(strategy="fixed")
        weights = wc.compute(["local", "global"])
        assert weights["local"] == 0.5
        assert weights["global"] == 0.5

    def test_confidence_strategy(self):
        wc = WeightCalculator(strategy="confidence")
        weights = wc.compute(
            ["local", "global"],
            confidences={"local": 0.9, "global": 0.1},
        )
        assert abs(sum(weights.values()) - 1.0) < 1e-4
        assert weights["local"] > weights["global"]

    def test_confidence_strategy_no_confidences(self):
        wc = WeightCalculator(strategy="confidence")
        with pytest.raises(ValueError, match="Confidences required"):
            wc.compute(["local", "global"])

    def test_similarity_strategy(self):
        wc = WeightCalculator(strategy="similarity")
        weights = wc.compute(
            ["local", "global"],
            embeddings={
                "local": torch.randn(8),
                "global": torch.randn(8),
            },
        )
        assert abs(sum(weights.values()) - 1.0) < 1e-4

    def test_similarity_strategy_no_embeddings(self):
        wc = WeightCalculator(strategy="similarity")
        with pytest.raises(ValueError, match="Embeddings required"):
            wc.compute(["local", "global"])

    def test_similarity_strategy_single_source(self):
        wc = WeightCalculator(strategy="similarity")
        weights = wc.compute(["local"], embeddings={"local": torch.randn(8)})
        assert weights["local"] == 1.0

    def test_adaptive_strategy(self):
        wc = WeightCalculator(strategy="adaptive")
        weights = wc.compute(
            ["local", "global"],
            adaptive_weights={"local": 0.6, "global": 0.4},
        )
        assert weights["local"] == 0.6

    def test_adaptive_strategy_no_weights(self):
        wc = WeightCalculator(strategy="adaptive")
        with pytest.raises(ValueError, match="Adaptive weights required"):
            wc.compute(["local", "global"])

    def test_learnable_strategy(self):
        wc = WeightCalculator(strategy="learnable")
        weights = wc.compute(["local", "global"])
        assert abs(sum(weights.values()) - 1.0) < 1e-4

    def test_invalid_strategy(self):
        with pytest.raises(ValueError, match="Unknown weighting"):
            WeightCalculator(strategy="invalid")

    def test_unknown_source(self):
        wc = WeightCalculator(strategy="fixed")
        with pytest.raises(ValueError, match="Unknown fusion source"):
            wc.compute(["unknown"])

    def test_properties(self):
        wc = WeightCalculator(strategy="confidence")
        assert wc.strategy == "confidence"

    def test_to_config(self):
        wc = WeightCalculator(
            strategy="fixed", fixed_weights={"local": 0.5, "global": 0.5}
        )
        cfg = wc.to_config()
        assert cfg["strategy"] == "fixed"

    def test_learnable_no_weights_fallback(self):
        wc = WeightCalculator(strategy="learnable")
        wc._learnable_weights = None
        weights = wc._learnable(["local"])
        assert weights["local"] == 1.0

    def test_fallback_strategy(self):
        wc = WeightCalculator(strategy="fixed")
        wc._strategy = "unknown"
        weights = wc.compute(["local"])
        assert weights["local"] == 1.0


# ===================================================================
# TestPrototypeSelector
# ===================================================================


class TestPrototypeSelector:
    def test_select_best_local(self):
        selector = PrototypeSelector()
        protos = [
            _make_prototype(class_id=0, modality="image", confidence=0.7),
            _make_prototype(class_id=0, modality="image", confidence=0.9),
        ]
        best = selector.select_best_local(protos, 0, "image")
        assert best is not None
        assert best.confidence == 0.9

    def test_select_best_local_none(self):
        selector = PrototypeSelector(confidence_threshold=0.9)
        protos = [_make_prototype(class_id=0, modality="image", confidence=0.5)]
        best = selector.select_best_local(protos, 0, "image")
        assert best is None

    def test_select_best_local_no_match(self):
        selector = PrototypeSelector()
        protos = [_make_prototype(class_id=0, modality="image")]
        best = selector.select_best_local(protos, 1, "image")
        assert best is None

    def test_select_best_global(self):
        selector = PrototypeSelector()
        protos = [
            _make_aggregated(class_id=0, modality="image", confidence=0.8),
            _make_aggregated(class_id=0, modality="image", confidence=0.95),
        ]
        best = selector.select_best_global(protos, 0, "image")
        assert best is not None
        assert best.confidence == 0.95

    def test_select_best_global_none(self):
        selector = PrototypeSelector(confidence_threshold=0.9)
        protos = [_make_aggregated(class_id=0, modality="image", confidence=0.5)]
        best = selector.select_best_global(protos, 0, "image")
        assert best is None

    def test_select_best_transferred(self):
        selector = PrototypeSelector()
        outputs = [
            _make_inference_output(class_id=0, modality="text", confidence=0.6),
            _make_inference_output(class_id=0, modality="text", confidence=0.85),
        ]
        best = selector.select_best_transferred(outputs, 0, "text")
        assert best is not None
        assert best.confidence == 0.85

    def test_select_best_transferred_none(self):
        selector = PrototypeSelector(confidence_threshold=0.9)
        outputs = [_make_inference_output(class_id=0, modality="text", confidence=0.5)]
        best = selector.select_best_transferred(outputs, 0, "text")
        assert best is None

    def test_select_sources(self):
        selector = PrototypeSelector()
        profile = ClientProfile(client_id="c1")
        local = [_make_prototype(class_id=0, modality="image", confidence=0.9)]
        global_p = [_make_aggregated(class_id=0, modality="image", confidence=0.8)]
        transferred = [
            _make_inference_output(class_id=0, modality="image", confidence=0.7)
        ]
        result = selector.select_sources(
            local, global_p, transferred, 0, "image", profile
        )
        assert result.client_id == "c1"
        assert result.class_id == 0
        assert result.modality == "image"
        assert result.local_prototype is not None
        assert result.global_prototype is not None
        assert result.cross_modal_prototype is not None

    def test_select_sources_no_profile(self):
        selector = PrototypeSelector()
        result = selector.select_sources([], [], [], 0, "image")
        assert result.client_id == "unknown"

    def test_invalid_threshold(self):
        with pytest.raises(ValueError, match="confidence_threshold"):
            PrototypeSelector(confidence_threshold=1.5)

    def test_to_config(self):
        selector = PrototypeSelector(confidence_threshold=0.5)
        assert selector.to_config()["confidence_threshold"] == 0.5


# ===================================================================
# TestFusionEngine
# ===================================================================


class TestFusionEngine:
    def test_fuse_weighted_sum(self):
        engine = FusionEngine(strategy="weighted_sum")
        pp = PersonalizedPrototype(
            client_id="c1",
            class_id=0,
            modality="image",
            local_prototype=[1.0, 2.0],
            global_prototype=[3.0, 4.0],
            embedding_dim=2,
        )
        result = engine.fuse(pp)
        assert result.personalized_prototype is not None
        assert len(result.personalized_prototype) == 2
        assert abs(sum(result.fusion_weights.values()) - 1.0) < 1e-4
        assert result.confidence > 0

    def test_fuse_single_source(self):
        engine = FusionEngine(strategy="weighted_sum")
        pp = PersonalizedPrototype(
            client_id="c1",
            class_id=0,
            modality="image",
            local_prototype=[1.0, 2.0, 3.0],
            embedding_dim=3,
        )
        result = engine.fuse(pp)
        assert result.personalized_prototype == [1.0, 2.0, 3.0]

    def test_fuse_no_sources(self):
        engine = FusionEngine()
        pp = PersonalizedPrototype(client_id="c1", class_id=0, modality="image")
        with pytest.raises(ValueError, match="No prototype sources"):
            engine.fuse(pp)

    def test_fuse_adaptive_strategy(self):
        gate_net = GatingNetwork(input_dim=6, num_sources=2)
        gate = AdaptiveGate(gating_network=gate_net)
        wc = WeightCalculator(strategy="adaptive")
        engine = FusionEngine(
            strategy="adaptive", weight_calculator=wc, adaptive_gate=gate
        )
        pp = PersonalizedPrototype(
            client_id="c1",
            class_id=0,
            modality="image",
            local_prototype=[1.0, 0.0],
            global_prototype=[0.0, 1.0],
            embedding_dim=2,
            confidence=0.5,
        )
        result = engine.fuse(pp)
        assert result.personalized_prototype is not None

    def test_fuse_confidence_weighted(self):
        engine = FusionEngine(strategy="confidence_weighted")
        pp = PersonalizedPrototype(
            client_id="c1",
            class_id=0,
            modality="image",
            local_prototype=[1.0, 2.0],
            global_prototype=[3.0, 4.0],
            embedding_dim=2,
        )
        result = engine.fuse(pp)
        assert result.personalized_prototype is not None

    def test_invalid_strategy(self):
        with pytest.raises(ValueError, match="Unknown fusion strategy"):
            FusionEngine(strategy="invalid")

    def test_to_config(self):
        engine = FusionEngine(strategy="weighted_sum")
        cfg = engine.to_config()
        assert cfg["strategy"] == "weighted_sum"

    def test_strategy_property(self):
        engine = FusionEngine(strategy="confidence_weighted")
        assert engine.strategy == "confidence_weighted"

    def test_fuse_all_three_sources(self):
        engine = FusionEngine(strategy="confidence_weighted")
        pp = PersonalizedPrototype(
            client_id="c1",
            class_id=0,
            modality="image",
            local_prototype=[1.0, 0.0, 0.0],
            global_prototype=[0.0, 1.0, 0.0],
            cross_modal_prototype=[0.0, 0.0, 1.0],
            embedding_dim=3,
        )
        result = engine.fuse(pp)
        assert result.personalized_prototype is not None
        assert sorted(result.fusion_weights.keys()) == [
            "cross_modal",
            "global",
            "local",
        ]

    def test_fuse_adaptive_with_profile(self):
        gate_net = GatingNetwork(input_dim=9, num_sources=2)
        gate = AdaptiveGate(gating_network=gate_net)
        wc = WeightCalculator(strategy="adaptive")
        engine = FusionEngine(
            strategy="adaptive", weight_calculator=wc, adaptive_gate=gate
        )
        profile = ClientProfile(client_id="c1")
        profile.record_confidence("image", 0, 0.8)
        pp = PersonalizedPrototype(
            client_id="c1",
            class_id=0,
            modality="image",
            local_prototype=[1.0, 0.0],
            global_prototype=[0.0, 1.0],
            embedding_dim=2,
            confidence=0.5,
        )
        result = engine.fuse(pp, client_profile=profile)
        assert result.personalized_prototype is not None

    def test_weight_calculator_property(self):
        wc = WeightCalculator(strategy="fixed")
        engine = FusionEngine(strategy="weighted_sum", weight_calculator=wc)
        assert engine.weight_calculator is wc

    def test_fuse_empty_embeddings_raises(self):
        engine = FusionEngine(strategy="weighted_sum")
        pp = PersonalizedPrototype(
            client_id="c1",
            class_id=0,
            modality="image",
            embedding_dim=0,
        )
        with pytest.raises(ValueError, match="No prototype sources"):
            engine.fuse(pp)


# ===================================================================
# TestPersonalizedMemory
# ===================================================================


class TestPersonalizedMemory:
    def test_store_and_retrieve(self):
        mem = PersonalizedMemory()
        pp = PersonalizedPrototype(client_id="c1", class_id=0, modality="image")
        mem.store(pp)
        retrieved = mem.retrieve("c1", 0, "image")
        assert retrieved is not None
        assert retrieved.client_id == "c1"

    def test_retrieve_nonexistent(self):
        mem = PersonalizedMemory()
        assert mem.retrieve("c1", 0, "image") is None

    def test_retrieve_all(self):
        mem = PersonalizedMemory()
        mem.store(PersonalizedPrototype(client_id="c1", class_id=0, modality="image"))
        mem.store(PersonalizedPrototype(client_id="c1", class_id=1, modality="text"))
        all_p = mem.retrieve_all(client_id="c1")
        assert len(all_p) == 2

    def test_retrieve_by_client(self):
        mem = PersonalizedMemory()
        mem.store(PersonalizedPrototype(client_id="c1", class_id=0, modality="image"))
        mem.store(PersonalizedPrototype(client_id="c2", class_id=0, modality="image"))
        assert len(mem.retrieve_by_client("c1")) == 1

    def test_retrieve_by_class(self):
        mem = PersonalizedMemory()
        mem.store(PersonalizedPrototype(client_id="c1", class_id=0, modality="image"))
        mem.store(PersonalizedPrototype(client_id="c1", class_id=1, modality="text"))
        assert len(mem.retrieve_by_class(0)) == 1

    def test_retrieve_by_modality(self):
        mem = PersonalizedMemory()
        mem.store(PersonalizedPrototype(client_id="c1", class_id=0, modality="image"))
        mem.store(PersonalizedPrototype(client_id="c1", class_id=1, modality="text"))
        assert len(mem.retrieve_by_modality("image")) == 1

    def test_update(self):
        mem = PersonalizedMemory()
        pp = PersonalizedPrototype(
            client_id="c1", class_id=0, modality="image", confidence=0.5
        )
        mem.store(pp)
        pp.confidence = 0.9
        assert mem.update(pp)
        retrieved = mem.retrieve("c1", 0, "image")
        assert retrieved is not None
        assert retrieved.confidence == 0.9

    def test_update_nonexistent(self):
        mem = PersonalizedMemory()
        pp = PersonalizedPrototype(client_id="c1", class_id=0, modality="image")
        assert not mem.update(pp)

    def test_remove(self):
        mem = PersonalizedMemory()
        pp = PersonalizedPrototype(client_id="c1", class_id=0, modality="image")
        mem.store(pp)
        assert mem.remove("c1", 0, "image")
        assert mem.retrieve("c1", 0, "image") is None

    def test_remove_nonexistent(self):
        mem = PersonalizedMemory()
        assert not mem.remove("c1", 0, "image")

    def test_clear_client(self):
        mem = PersonalizedMemory()
        mem.store(PersonalizedPrototype(client_id="c1", class_id=0, modality="image"))
        mem.store(PersonalizedPrototype(client_id="c1", class_id=1, modality="text"))
        mem.store(PersonalizedPrototype(client_id="c2", class_id=0, modality="image"))
        assert mem.clear_client("c1") == 2
        assert mem.size == 1

    def test_get_history(self):
        mem = PersonalizedMemory()
        pp = PersonalizedPrototype(client_id="c1", class_id=0, modality="image")
        mem.store(pp)
        history = mem.get_history("c1", 0, "image")
        assert len(history) == 1
        assert history[0]["confidence"] == 0.0

    def test_statistics(self):
        mem = PersonalizedMemory()
        mem.store(PersonalizedPrototype(client_id="c1", class_id=0, modality="image"))
        mem.store(PersonalizedPrototype(client_id="c2", class_id=0, modality="text"))
        stats = mem.statistics()
        assert stats["size"] == 2
        assert stats["unique_clients"] == 2

    def test_capacity(self):
        mem = PersonalizedMemory(capacity=2)
        mem.store(PersonalizedPrototype(client_id="c1", class_id=0, modality="image"))
        mem.store(PersonalizedPrototype(client_id="c1", class_id=1, modality="image"))
        mem.store(PersonalizedPrototype(client_id="c1", class_id=2, modality="image"))
        assert mem.size == 2

    def test_clear(self):
        mem = PersonalizedMemory()
        mem.store(PersonalizedPrototype(client_id="c1", class_id=0, modality="image"))
        mem.clear()
        assert mem.size == 0
        assert mem.get_history("c1", 0, "image") == []

    def test_duplicate_store_raises(self):
        mem = PersonalizedMemory()
        pp = PersonalizedPrototype(client_id="c1", class_id=0, modality="image")
        mem.store(pp)
        with pytest.raises(ValueError, match="Duplicate"):
            mem.store(pp)

    def test_properties(self):
        mem = PersonalizedMemory(capacity=500)
        assert mem.capacity == 500


# ===================================================================
# TestAdaptationEngine
# ===================================================================


class TestAdaptationEngine:
    def test_ema_adapt(self):
        engine = AdaptationEngine(strategy="ema", ema_alpha=0.9)
        pp = PersonalizedPrototype(
            client_id="c1",
            class_id=0,
            modality="image",
            personalized_prototype=[1.0, 2.0],
        )
        result = engine.adapt(pp)
        assert result.personalized_prototype == [1.0, 2.0]

    def test_ema_adapt_second_call(self):
        engine = AdaptationEngine(strategy="ema", ema_alpha=0.8)
        pp = PersonalizedPrototype(
            client_id="c1",
            class_id=0,
            modality="image",
            personalized_prototype=[1.0, 2.0],
        )
        engine.adapt(pp)
        pp.personalized_prototype = [3.0, 4.0]
        result = engine.adapt(pp)
        assert result.personalized_prototype is not None
        assert result.personalized_prototype[0] == pytest.approx(0.8 * 1.0 + 0.2 * 3.0)

    def test_momentum_adapt(self):
        engine = AdaptationEngine(strategy="momentum", momentum=0.5)
        pp = PersonalizedPrototype(
            client_id="c1",
            class_id=0,
            modality="image",
            personalized_prototype=[1.0, 2.0],
        )
        engine.adapt(pp)
        pp.personalized_prototype = [3.0, 4.0]
        result = engine.adapt(pp)
        assert result.personalized_prototype is not None

    def test_residual_adapt(self):
        engine = AdaptationEngine(strategy="residual")
        pp = PersonalizedPrototype(
            client_id="c1",
            class_id=0,
            modality="image",
            personalized_prototype=[1.0, 2.0],
        )
        engine.adapt(pp)
        pp.personalized_prototype = [3.0, 4.0]
        result = engine.adapt(pp)
        assert result.personalized_prototype is not None

    def test_adaptive_blend(self):
        engine = AdaptationEngine(strategy="adaptive_blending")
        pp = PersonalizedPrototype(
            client_id="c1",
            class_id=0,
            modality="image",
            personalized_prototype=[1.0, 2.0],
            fusion_weights={"local": 0.7, "global": 0.3},
        )
        engine.adapt(pp)
        pp.personalized_prototype = [3.0, 4.0]
        result = engine.adapt(pp)
        assert result.personalized_prototype is not None

    def test_adapt_no_personalized(self):
        engine = AdaptationEngine()
        pp = PersonalizedPrototype(client_id="c1", class_id=0, modality="image")
        with pytest.raises(ValueError, match="cannot adapt"):
            engine.adapt(pp)

    def test_with_profile(self):
        engine = AdaptationEngine(strategy="ema")
        profile = ClientProfile(client_id="c1")
        pp = PersonalizedPrototype(
            client_id="c1",
            class_id=0,
            modality="image",
            personalized_prototype=[1.0, 2.0],
        )
        engine.adapt(pp, client_profile=profile)
        assert profile.training_steps == 0
        assert profile.prototype_drift == 0.0

    def test_invalid_strategy(self):
        with pytest.raises(ValueError, match="Unknown adaptation"):
            AdaptationEngine(strategy="invalid")

    def test_invalid_ema_alpha(self):
        with pytest.raises(ValueError, match="ema_alpha"):
            AdaptationEngine(strategy="ema", ema_alpha=1.5)

    def test_invalid_momentum(self):
        with pytest.raises(ValueError, match="momentum"):
            AdaptationEngine(strategy="momentum", momentum=1.5)

    def test_properties(self):
        engine = AdaptationEngine(strategy="ema", ema_alpha=0.7)
        assert engine.strategy == "ema"
        assert engine.ema_alpha == 0.7

    def test_to_config(self):
        engine = AdaptationEngine(strategy="momentum", momentum=0.8)
        cfg = engine.to_config()
        assert cfg["strategy"] == "momentum"
        assert cfg["momentum"] == 0.8


# ===================================================================
# TestRegularization
# ===================================================================


class TestPrototypeConsistencyRegularization:
    def test_forward(self):
        reg = PrototypeConsistencyRegularization()
        pers = torch.randn(8)
        local = pers + 0.01
        global_p = pers + 0.02
        loss = reg(pers, local, global_p)
        assert loss.ndim == 0
        assert loss >= 0

    def test_forward_no_sources(self):
        reg = PrototypeConsistencyRegularization()
        loss = reg(torch.randn(8), None, None)
        assert loss.item() == 0.0


class TestFusionSmoothnessRegularization:
    def test_forward(self):
        reg = FusionSmoothnessRegularization()
        weights = torch.tensor([0.5, 0.3, 0.2])
        loss = reg(weights)
        assert loss.ndim == 0
        assert loss >= 0

    def test_forward_single(self):
        reg = FusionSmoothnessRegularization()
        loss = reg(torch.tensor([1.0]))
        assert loss.item() == 0.0


class TestPrototypeStabilityRegularization:
    def test_forward(self):
        reg = PrototypeStabilityRegularization()
        current = torch.randn(8)
        loss = reg(current, current.clone())
        assert loss.ndim == 0
        assert loss >= 0

    def test_forward_no_previous(self):
        reg = PrototypeStabilityRegularization()
        loss = reg(torch.randn(8), None)
        assert loss.item() == 0.0


class TestTemporalConsistencyRegularization:
    def test_forward_first_call(self):
        reg = TemporalConsistencyRegularization()
        loss = reg(torch.randn(8), key="test")
        assert loss.item() == 0.0

    def test_forward_second_call(self):
        reg = TemporalConsistencyRegularization()
        reg(torch.randn(8), key="test")
        loss = reg(torch.randn(8), key="test")
        assert loss.ndim == 0
        assert loss >= 0

    def test_different_keys(self):
        reg = TemporalConsistencyRegularization()
        reg(torch.randn(8), key="a")
        loss_a = reg(torch.randn(8), key="a")
        loss_b = reg(torch.randn(8), key="b")
        assert loss_b.item() == 0.0
        assert loss_a >= 0


# ===================================================================
# TestLosses
# ===================================================================


class TestFusionLoss:
    def test_forward(self):
        loss_fn = FusionLoss()
        fused = torch.randn(8)
        targets = [torch.randn(8) for _ in range(2)]
        weights = [0.5, 0.5]
        loss = loss_fn(fused, targets, weights)
        assert loss.ndim == 0
        assert loss >= 0

    def test_empty_targets(self):
        loss_fn = FusionLoss()
        with pytest.raises(ValueError, match="must not be empty"):
            loss_fn(torch.randn(8), [], [])

    def test_mismatched_lengths(self):
        loss_fn = FusionLoss()
        with pytest.raises(ValueError, match="must match"):
            loss_fn(torch.randn(8), [torch.randn(8)], [0.5, 0.5])


class TestConsistencyLoss:
    def test_forward(self):
        loss_fn = ConsistencyLoss()
        pred = torch.randn(8)
        target = pred + 0.01
        loss = loss_fn(pred, target)
        assert loss.ndim == 0
        assert loss >= 0

    def test_shape_mismatch(self):
        loss_fn = ConsistencyLoss()
        with pytest.raises(ValueError, match="Shape mismatch"):
            loss_fn(torch.randn(8), torch.randn(4))


class TestPersonalizationLoss:
    def test_forward_all(self):
        loss_fn = PersonalizationLoss()
        pers = torch.randn(8)
        local = pers + 0.01
        global_p = pers + 0.02
        cross = pers + 0.03
        loss = loss_fn(pers, local, global_p, cross)
        assert loss.ndim == 0
        assert loss >= 0

    def test_forward_no_sources(self):
        loss_fn = PersonalizationLoss()
        loss = loss_fn(torch.randn(8))
        assert loss.item() == 0.0

    def test_forward_single_source(self):
        loss_fn = PersonalizationLoss()
        pers = torch.randn(8)
        loss = loss_fn(pers, local=pers.clone())
        assert loss.ndim == 0


class TestPrototypeRegularizationLoss:
    def test_forward(self):
        loss_fn = PrototypeRegularizationLoss()
        proto = torch.randn(8)
        loss = loss_fn(proto, norm_weight=0.01)
        assert loss.ndim == 0
        assert loss >= 0


class TestAdaptiveWeightingLoss:
    def test_forward(self):
        loss_fn = AdaptiveWeightingLoss(regularization_strength=0.01)
        weights = torch.tensor([0.5, 0.5])
        source_losses = torch.tensor([0.1, 0.2])
        loss = loss_fn(weights, source_losses)
        assert loss.ndim == 0

    def test_mismatched_shapes(self):
        loss_fn = AdaptiveWeightingLoss()
        with pytest.raises(ValueError, match="must have the same"):
            loss_fn(torch.tensor([0.5, 0.5]), torch.tensor([0.1]))


class TestCombinedPersonalizationLoss:
    def test_forward_all(self):
        loss_fn = CombinedPersonalizationLoss()
        fused = torch.randn(8)
        targets = [torch.randn(8) for _ in range(2)]
        target_weights = [0.5, 0.5]
        local = fused + 0.01
        losses = loss_fn(
            fused=fused,
            targets=targets,
            target_weights=target_weights,
            local=local,
            global_p=fused + 0.02,
            cross_modal=fused + 0.03,
            consistency_target=fused.clone(),
            fusion_weights=torch.tensor([0.6, 0.4]),
            source_losses=torch.tensor([0.1, 0.2]),
        )
        assert "fusion" in losses
        assert "consistency" in losses
        assert "personalization" in losses
        assert "regularization" in losses
        assert "adaptive_weighting" in losses
        assert "total" in losses

    def test_forward_minimal(self):
        loss_fn = CombinedPersonalizationLoss()
        fused = torch.randn(8)
        targets = [torch.randn(8)]
        losses = loss_fn(fused=fused, targets=targets, target_weights=[1.0])
        assert "fusion" in losses
        assert "consistency" not in losses
        assert "total" in losses

    def test_to_config(self):
        loss_fn = CombinedPersonalizationLoss(fusion_weight=0.5)
        cfg = loss_fn.to_config()
        assert cfg["fusion_weight"] == 0.5


# ===================================================================
# TestPersonalizationMetrics
# ===================================================================


class TestPersonalizationMetrics:
    def test_personalization_gain(self):
        metrics = PersonalizationMetrics()
        pers = torch.randn(8)
        glob = torch.randn(8)
        gain = metrics.personalization_gain(pers, glob)
        assert isinstance(gain, float)
        assert 0.0 <= gain <= 2.0

    def test_prototype_drift(self):
        metrics = PersonalizationMetrics()
        drift = metrics.prototype_drift(torch.randn(8), torch.randn(8))
        assert isinstance(drift, float)
        assert drift >= 0

    def test_alignment_score(self):
        metrics = PersonalizationMetrics()
        pers = torch.randn(8)
        score = metrics.alignment_score(pers, pers.clone(), pers.clone())
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0 + 1e-6

    def test_alignment_score_no_refs(self):
        metrics = PersonalizationMetrics()
        score = metrics.alignment_score(torch.randn(8))
        assert score == 0.0

    def test_fusion_quality(self):
        metrics = PersonalizationMetrics()
        pp = PersonalizedPrototype(
            client_id="c1",
            class_id=0,
            modality="image",
            fusion_weights={"local": 0.5, "global": 0.5},
        )
        quality = metrics.fusion_quality(pp)
        assert 0.0 <= quality <= 1.0

    def test_fusion_quality_no_weights(self):
        metrics = PersonalizationMetrics()
        pp = PersonalizedPrototype(client_id="c1", class_id=0, modality="image")
        assert metrics.fusion_quality(pp) == 0.0

    def test_fusion_quality_single(self):
        metrics = PersonalizationMetrics()
        pp = PersonalizedPrototype(
            client_id="c1",
            class_id=0,
            modality="image",
            fusion_weights={"local": 1.0},
        )
        assert metrics.fusion_quality(pp) == 1.0

    def test_client_diversity(self):
        metrics = PersonalizationMetrics()
        protos = [
            PersonalizedPrototype(
                client_id="c1",
                class_id=0,
                modality="image",
                personalized_prototype=[1.0, 0.0],
            ),
            PersonalizedPrototype(
                client_id="c2",
                class_id=0,
                modality="image",
                personalized_prototype=[0.0, 1.0],
            ),
        ]
        div = metrics.client_diversity(protos)
        assert 0.0 <= div <= 1.0

    def test_client_diversity_single(self):
        metrics = PersonalizationMetrics()
        protos = [
            PersonalizedPrototype(
                client_id="c1",
                class_id=0,
                modality="image",
                personalized_prototype=[1.0, 0.0],
            ),
        ]
        assert metrics.client_diversity(protos) == 0.0

    def test_client_diversity_no_personalized(self):
        metrics = PersonalizationMetrics()
        protos = [
            PersonalizedPrototype(client_id="c1", class_id=0, modality="image"),
            PersonalizedPrototype(client_id="c2", class_id=0, modality="image"),
        ]
        assert metrics.client_diversity(protos) == 0.0

    def test_confidence_trend(self):
        metrics = PersonalizationMetrics()
        profile = ClientProfile(client_id="c1")
        for c in [0.5, 0.6, 0.7, 0.8, 0.9]:
            profile.record_confidence("image", 0, c)
        assert metrics.confidence_trend(profile) == pytest.approx(0.4, abs=1e-4)

    def test_prototype_stability(self):
        metrics = PersonalizationMetrics()
        stability = metrics.prototype_stability([0.5, 0.6, 0.55])
        assert 0.0 <= stability <= 1.0

    def test_prototype_stability_single(self):
        metrics = PersonalizationMetrics()
        assert metrics.prototype_stability([0.5]) == 1.0

    def test_compute_all_empty(self):
        metrics = PersonalizationMetrics()
        results = metrics.compute_all([])
        assert results["personalization_gain"] == 0.0
        assert results["alignment_score"] == 0.0

    def test_compute_all(self):
        metrics = PersonalizationMetrics()
        protos = [
            PersonalizedPrototype(
                client_id="c1",
                class_id=0,
                modality="image",
                local_prototype=[1.0, 0.0],
                global_prototype=[1.0, 0.0],
                personalized_prototype=[1.0, 0.0],
                fusion_weights={"local": 0.5, "global": 0.5},
            ),
        ]
        results = metrics.compute_all(protos)
        assert results["personalization_gain"] >= 0
        assert results["fusion_quality"] >= 0


# ===================================================================
# TestPersonalizationLogger
# ===================================================================


class TestPersonalizationLogger:
    def test_log_fusion(self):
        log = PersonalizationLogger()
        log.log_fusion("c1", 0, "image", "weighted_sum", {"local": 0.5}, 0.8)
        history = log.get_history("fusion")
        assert len(history) == 1

    def test_log_selection(self):
        log = PersonalizationLogger()
        log.log_selection("c1", 0, "image", "local", 0.9)
        assert len(log.get_history("selection")) == 1

    def test_log_adaptation(self):
        log = PersonalizationLogger()
        log.log_adaptation("c1", "ema", 0.1, 5)
        assert len(log.get_history("adaptation")) == 1

    def test_log_loss(self):
        log = PersonalizationLogger()
        log.log_loss("fusion", 0.5, step=1)
        assert len(log.get_history("loss")) == 1

    def test_log_confidence(self):
        log = PersonalizationLogger()
        log.log_confidence("c1", "image", 0, 0.85)
        history = log.get_history("confidence")
        assert len(history) == 1

    def test_get_history_all(self):
        log = PersonalizationLogger()
        log.log_fusion("c1", 0, "image", "weighted_sum", {}, 0.8)
        log.log_loss("test", 0.1)
        assert len(log.get_history()) == 2

    def test_summary(self):
        log = PersonalizationLogger()
        log.log_fusion("c1", 0, "image", "weighted_sum", {}, 0.8)
        log.log_loss("test", 0.1)
        summary = log.summary()
        assert summary["total_events"] == 2

    def test_reset(self):
        log = PersonalizationLogger()
        log.log_fusion("c1", 0, "image", "weighted_sum", {}, 0.8)
        log.reset()
        assert log.summary()["total_events"] == 0


# ===================================================================
# TestPersonalizationRegistry
# ===================================================================


class TestPersonalizationRegistry:
    def test_create(self):
        r = PersonalizationRegistry()
        assert "weighted_sum" in r.list_fusion_strategies()
        assert "ema" in r.list_adaptation_methods()

    def test_register_fusion_strategy(self):
        r = PersonalizationRegistry()
        r.register_fusion_strategy("test")
        assert "test" in r.list_fusion_strategies()

    def test_register_fusion_strategy_duplicate(self):
        r = PersonalizationRegistry()
        with pytest.raises(ValueError, match="already registered"):
            r.register_fusion_strategy("weighted_sum")

    def test_get_fusion_strategy(self):
        r = PersonalizationRegistry()
        assert r.get_fusion_strategy("weighted_sum") == "weighted_sum"

    def test_get_fusion_strategy_unknown(self):
        r = PersonalizationRegistry()
        with pytest.raises(ValueError, match="Unknown"):
            r.get_fusion_strategy("nonexistent")

    def test_register_gating_network(self):
        r = PersonalizationRegistry()
        r.register_gating_network("test", GatingNetwork)
        cls = r.get_gating_network("test")
        assert cls is GatingNetwork

    def test_register_gating_network_duplicate(self):
        r = PersonalizationRegistry()
        with pytest.raises(ValueError, match="already registered"):
            r.register_gating_network("default", GatingNetwork)

    def test_get_gating_network_unknown(self):
        r = PersonalizationRegistry()
        with pytest.raises(ValueError, match="Unknown"):
            r.get_gating_network("nonexistent")

    def test_register_adaptation_method(self):
        r = PersonalizationRegistry()
        r.register_adaptation_method("test")
        assert "test" in r.list_adaptation_methods()

    def test_register_adaptation_method_duplicate(self):
        r = PersonalizationRegistry()
        with pytest.raises(ValueError, match="already registered"):
            r.register_adaptation_method("ema")

    def test_get_adaptation_method_unknown(self):
        r = PersonalizationRegistry()
        with pytest.raises(ValueError, match="Unknown"):
            r.get_adaptation_method("nonexistent")

    def test_register_weight_calculator(self):
        r = PersonalizationRegistry()
        r.register_weight_calculator("test")
        assert r.get_weight_calculator("test") == "test"

    def test_register_weight_calculator_duplicate(self):
        r = PersonalizationRegistry()
        with pytest.raises(ValueError, match="already registered"):
            r.register_weight_calculator("fixed")

    def test_get_weight_calculator_unknown(self):
        r = PersonalizationRegistry()
        with pytest.raises(ValueError, match="Unknown"):
            r.get_weight_calculator("nonexistent")

    def test_register_loss_function(self):
        r = PersonalizationRegistry()
        r.register_loss_function("test", FusionLoss)
        cls = r.get_loss_function("test")
        assert cls is FusionLoss

    def test_register_loss_function_duplicate(self):
        r = PersonalizationRegistry()
        with pytest.raises(ValueError, match="already registered"):
            r.register_loss_function("fusion", FusionLoss)

    def test_get_loss_function_unknown(self):
        r = PersonalizationRegistry()
        with pytest.raises(ValueError, match="Unknown"):
            r.get_loss_function("nonexistent")

    def test_register_component(self):
        r = PersonalizationRegistry()
        r.register_component("double", lambda x: x * 2)
        assert r.get_component("double", x=5) == 10

    def test_register_component_duplicate(self):
        r = PersonalizationRegistry()
        r.register_component("test", lambda: 1)
        with pytest.raises(ValueError, match="already registered"):
            r.register_component("test", lambda: 1)

    def test_get_component_unknown(self):
        r = PersonalizationRegistry()
        with pytest.raises(ValueError, match="Unknown"):
            r.get_component("nonexistent")

    def test_list_components(self):
        r = PersonalizationRegistry()
        assert r.list_components() == []
        r.register_component("test", lambda: 1)
        assert r.list_components() == ["test"]

    def test_to_config(self):
        r = PersonalizationRegistry()
        cfg = r.to_config()
        assert "fusion_strategies" in cfg
        assert "gating_networks" in cfg


# ===================================================================
# TestPersonalizationFactory
# ===================================================================


class TestPersonalizationFactory:
    def test_create_gating_network(self):
        gate = PersonalizationFactory.create_gating_network(input_dim=16, num_sources=3)
        assert isinstance(gate, GatingNetwork)
        assert gate.num_sources == 3

    def test_create_weight_calculator(self):
        wc = PersonalizationFactory.create_weight_calculator(strategy="confidence")
        assert isinstance(wc, WeightCalculator)
        assert wc.strategy == "confidence"

    def test_create_fusion_engine_weighted_sum(self):
        engine = PersonalizationFactory.create_fusion_engine(strategy="weighted_sum")
        assert isinstance(engine, FusionEngine)
        assert engine.strategy == "weighted_sum"

    def test_create_fusion_engine_adaptive(self):
        engine = PersonalizationFactory.create_fusion_engine(
            strategy="adaptive",
            modalities={"image": 8, "text": 8},
            mappings=[("image", "text")],
        )
        assert isinstance(engine, FusionEngine)
        assert engine.strategy == "adaptive"

    def test_create_prototype_selector(self):
        selector = PersonalizationFactory.create_prototype_selector(
            confidence_threshold=0.5
        )
        assert isinstance(selector, PrototypeSelector)
        assert selector.confidence_threshold == 0.5

    def test_create_adaptation_engine(self):
        engine = PersonalizationFactory.create_adaptation_engine(
            strategy="momentum", momentum=0.7
        )
        assert isinstance(engine, AdaptationEngine)
        assert engine.strategy == "momentum"

    def test_create_loss(self):
        loss = PersonalizationFactory.create_loss()
        assert isinstance(loss, CombinedPersonalizationLoss)

    def test_create_metrics(self):
        metrics = PersonalizationFactory.create_metrics()
        assert isinstance(metrics, PersonalizationMetrics)

    def test_create_memory(self):
        mem = PersonalizationFactory.create_memory(capacity=500)
        assert isinstance(mem, PersonalizedMemory)
        assert mem.capacity == 500

    def test_create_from_config(self):
        config = {
            "modalities": {"image": 8, "text": 8},
            "mappings": [("image", "text")],
            "gating_network": {"input_dim": 28, "num_sources": 3},
            "fusion": {"strategy": "weighted_sum"},
            "loss": {"fusion_weight": 0.5},
        }
        components = PersonalizationFactory.create_from_config(config)
        assert "mapper" in components
        assert "graph" in components
        assert "gating_network" in components
        assert "weight_calculator" in components
        assert "fusion_engine" in components
        assert "prototype_selector" in components
        assert "adaptation_engine" in components
        assert "loss" in components
        assert "metrics" in components
        assert "memory" in components


# ===================================================================
# TestEdgeCases
# ===================================================================


class TestEdgeCases:
    def test_personalized_prototype_timestamp(self):
        pp = PersonalizedPrototype(client_id="c1", class_id=0, modality="image")
        assert pp.timestamp > 0

    def test_fusion_engine_grad_flow(self):
        engine = FusionEngine(strategy="weighted_sum")
        pp = PersonalizedPrototype(
            client_id="c1",
            class_id=0,
            modality="image",
            local_prototype=[1.0, 2.0],
            global_prototype=[3.0, 4.0],
            embedding_dim=2,
        )
        result = engine.fuse(pp)
        assert result.personalized_prototype is not None
        assert len(result.personalized_prototype) == 2

    def test_confidence_all_factors(self):
        pc = PersonalizedConfidence()
        conf = pc.estimate(
            base_confidence=0.5,
            prototype_variance=0.1,
            consistency=0.8,
            similarity_score=0.9,
            history_confidences=[0.5, 0.6, 0.55],
        )
        assert 0.0 <= conf <= 1.0

    def test_memory_statistics_empty(self):
        mem = PersonalizedMemory()
        stats = mem.statistics()
        assert stats["size"] == 0
        assert stats["unique_clients"] == 0

    def test_selector_no_candidates(self):
        selector = PrototypeSelector()
        result = selector.select_sources([], [], [], 0, "image")
        assert result.local_prototype is None
        assert result.global_prototype is None
        assert result.cross_modal_prototype is None
        assert result.embedding_dim == 0

    def test_regularization_grad_flow(self):
        reg = PrototypeConsistencyRegularization()
        pers = torch.randn(8, requires_grad=True)
        local = pers.clone().detach().requires_grad_(True)
        loss = reg(pers, local, None)
        loss.backward()
        assert pers.grad is not None

    def test_gating_network_weight_sum(self):
        net = GatingNetwork(input_dim=8, num_sources=4, normalization="softmax")
        x = torch.randn(8)
        w = net(x)
        assert w.sum().item() == pytest.approx(1.0, abs=1e-4)
        assert all(w >= 0)

    def test_adaptation_engine_drift_tracking(self):
        engine = AdaptationEngine(strategy="ema")
        profile = ClientProfile(client_id="c1")
        pp = PersonalizedPrototype(
            client_id="c1",
            class_id=0,
            modality="image",
            personalized_prototype=[1.0, 0.0],
        )
        engine.adapt(pp, client_profile=profile)
        pp.personalized_prototype = [0.0, 1.0]
        engine.adapt(pp, client_profile=profile)
        assert profile.prototype_drift > 0

    def test_metrics_no_profiles(self):
        metrics = PersonalizationMetrics()
        protos = [
            PersonalizedPrototype(
                client_id="c1",
                class_id=0,
                modality="image",
                personalized_prototype=[1.0, 0.0],
                global_prototype=[1.0, 0.0],
            ),
        ]
        results = metrics.compute_all(protos)
        assert results["confidence_trend"] == 0.0

    def test_metrics_with_profiles(self):
        metrics = PersonalizationMetrics()
        profile = ClientProfile(client_id="c1")
        for c in [0.7, 0.8, 0.9]:
            profile.record_confidence("image", 0, c)
        protos = [
            PersonalizedPrototype(
                client_id="c1",
                class_id=0,
                modality="image",
                personalized_prototype=[1.0, 0.0],
                global_prototype=[1.0, 0.0],
            ),
        ]
        results = metrics.compute_all(protos, profiles={"c1": profile})
        assert results["confidence_trend"] != 0.0
        assert results["prototype_stability"] > 0

    def test_weight_calculator_learnable_grad(self):
        wc = WeightCalculator(strategy="learnable")
        weights = wc.compute(["local", "global", "cross_modal"])
        assert abs(sum(weights.values()) - 1.0) < 1e-4

    def test_combined_loss_no_optional(self):
        loss_fn = CombinedPersonalizationLoss()
        fused = torch.randn(8)
        targets = [torch.randn(8)]
        losses = loss_fn(fused=fused, targets=targets, target_weights=[1.0])
        assert "adaptive_weighting" not in losses
        assert "consistency" not in losses
