from __future__ import annotations

import time
from typing import Any

import pytest
import torch
import torch.nn as nn

from app.federated.models import AggregatedPrototype
from app.knowledge_transfer.alignment_network import AlignmentNetwork
from app.knowledge_transfer.contrastive_alignment import (
    ContrastiveAlignmentLoss,
    InfoNCELoss,
    TripletLoss,
)
from app.knowledge_transfer.cross_modal_mapper import CrossModalMapper
from app.knowledge_transfer.factory import TransferFactory
from app.knowledge_transfer.inference import InferenceEngine, InferenceOutput
from app.knowledge_transfer.modality_graph import ModalityGraph
from app.knowledge_transfer.prototype_generator import (
    PrototypeGenerator,
    SynthesisResult,
)
from app.knowledge_transfer.registry import TransferRegistry
from app.knowledge_transfer.similarity import Similarity
from app.knowledge_transfer.transfer_loss import TransferLoss
from app.knowledge_transfer.utils import TransferLogger
from app.knowledge_transfer.validation import (
    validate_mapping_dimensions,
    validate_missing_modalities,
    validate_no_nan,
    validate_prototype_size,
    validate_shape_match,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_aggregated(
    class_id: int = 0,
    modality: str = "image",
    dim: int = 8,
    sample_count: int = 10,
    confidence: float = 0.9,
) -> AggregatedPrototype:
    return AggregatedPrototype(
        class_id=class_id,
        modality=modality,
        prototype_vector=[float(i) for i in range(dim)],
        embedding_dim=dim,
        sample_count=sample_count,
        confidence=confidence,
    )


def _default_modalities() -> dict[str, int]:
    return {"image": 8, "text": 8, "audio": 8, "sensor": 8}


def _default_mappings() -> list[tuple[str, str]]:
    return [
        ("image", "text"),
        ("text", "image"),
        ("image", "audio"),
        ("audio", "image"),
        ("sensor", "image"),
        ("image", "sensor"),
    ]


# ===================================================================
# TestValidation
# ===================================================================


class TestValidation:
    def test_validate_mapping_dimensions_ok(self):
        validate_mapping_dimensions(8, 16)

    def test_validate_mapping_dimensions_invalid_source(self):
        with pytest.raises(ValueError, match="source.*dimension"):
            validate_mapping_dimensions(0, 16)

    def test_validate_mapping_dimensions_invalid_target(self):
        with pytest.raises(ValueError, match="target.*dimension"):
            validate_mapping_dimensions(8, 0)

    def test_validate_prototype_size_1d(self):
        validate_prototype_size(torch.randn(8))

    def test_validate_prototype_size_2d(self):
        validate_prototype_size(torch.randn(4, 8))

    def test_validate_prototype_size_3d_raises(self):
        with pytest.raises(ValueError, match="3-D"):
            validate_prototype_size(torch.randn(2, 4, 8))

    def test_validate_prototype_size_empty(self):
        with pytest.raises(ValueError, match="empty"):
            validate_prototype_size(torch.randn(0))

    def test_validate_missing_modalities_ok(self):
        validate_missing_modalities({"image"}, {"text"}, {"image", "text", "audio"})

    def test_validate_missing_modalities_unknown_available(self):
        with pytest.raises(ValueError, match="not in known"):
            validate_missing_modalities({"unknown"}, {"text"}, {"image", "text"})

    def test_validate_missing_modalities_unknown_missing(self):
        with pytest.raises(ValueError, match="not in known"):
            validate_missing_modalities({"image"}, {"unknown"}, {"image", "text"})

    def test_validate_missing_modalities_overlap(self):
        with pytest.raises(ValueError, match="cannot be both"):
            validate_missing_modalities({"image"}, {"image"}, {"image", "text"})

    def test_validate_missing_modalities_empty_missing(self):
        with pytest.raises(ValueError, match="No missing"):
            validate_missing_modalities({"image"}, set(), {"image"})

    def test_validate_no_nan_ok(self):
        validate_no_nan(torch.randn(8))

    def test_validate_no_nan_raises(self):
        with pytest.raises(ValueError, match="NaN"):
            validate_no_nan(torch.tensor([1.0, float("nan")]))

    def test_validate_no_nan_inf_raises(self):
        with pytest.raises(ValueError, match="Inf"):
            validate_no_nan(torch.tensor([1.0, float("inf")]))

    def test_validate_shape_match_ok(self):
        validate_shape_match(torch.randn(4, 8), torch.randn(4, 8))

    def test_validate_shape_match_mismatch(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            validate_shape_match(torch.randn(4, 8), torch.randn(2, 8))


# ===================================================================
# TestSimilarity
# ===================================================================


class TestSimilarity:
    def test_create_cosine(self):
        s = Similarity(metric="cosine")
        assert s.metric == "cosine"

    def test_create_euclidean(self):
        s = Similarity(metric="euclidean")
        assert s.metric == "euclidean"

    def test_create_dot(self):
        s = Similarity(metric="dot")
        assert s.metric == "dot"

    def test_invalid_metric(self):
        with pytest.raises(ValueError, match="Unsupported"):
            Similarity(metric="manhattan")

    def test_compute_cosine(self):
        s = Similarity("cosine")
        a = torch.tensor([1.0, 0.0])
        b = torch.tensor([1.0, 0.0])
        assert s.compute(a, b).item() == pytest.approx(1.0, abs=1e-4)

    def test_compute_euclidean(self):
        s = Similarity("euclidean")
        a = torch.zeros(4)
        b = torch.zeros(4)
        assert s.compute(a, b).item() == pytest.approx(1.0, abs=1e-4)

    def test_compute_dot(self):
        s = Similarity("dot")
        a = torch.tensor([2.0, 3.0])
        b = torch.tensor([4.0, 5.0])
        assert s.compute(a, b).item() == pytest.approx(23.0, abs=1e-4)

    def test_pairwise_cosine(self):
        s = Similarity("cosine")
        a = torch.randn(4, 8)
        b = torch.randn(3, 8)
        mat = s.pairwise(a, b)
        assert mat.shape == (4, 3)

    def test_pairwise_euclidean(self):
        s = Similarity("euclidean")
        a = torch.randn(4, 8)
        b = torch.randn(3, 8)
        mat = s.pairwise(a, b)
        assert mat.shape == (4, 3)

    def test_pairwise_dot(self):
        s = Similarity("dot")
        a = torch.randn(4, 8)
        b = torch.randn(3, 8)
        mat = s.pairwise(a, b)
        assert mat.shape == (4, 3)

    def test_to_config(self):
        s = Similarity("euclidean")
        assert s.to_config() == {"metric": "euclidean"}

    def test_compute_shape_mismatch(self):
        s = Similarity("cosine")
        a = torch.randn(8)
        b = torch.randn(4)
        with pytest.raises(ValueError, match="Shape mismatch"):
            s.compute(a, b)

    def test_pairwise_unsupported_metric(self):
        s = Similarity("cosine")
        s._metric = "manhattan"
        a = torch.randn(4, 8)
        b = torch.randn(3, 8)
        with pytest.raises(ValueError, match="Unsupported metric"):
            s.pairwise(a, b)

    def test_compute_unknown_metric_fallback(self):
        s = Similarity("cosine")
        s._metric = "unknown"
        a = torch.randn(4)
        b = torch.randn(4)
        result = s.compute(a, b)
        assert result.item() == 0.0


class TestAlignmentNetwork:
    def test_linear_forward(self):
        net = AlignmentNetwork(source_dim=8, target_dim=16, mapper_type="linear")
        x = torch.randn(8)
        out = net(x)
        assert out.shape == (16,)

    def test_mlp_forward(self):
        net = AlignmentNetwork(
            source_dim=8,
            target_dim=16,
            hidden_dims=[32, 32],
            mapper_type="mlp",
        )
        x = torch.randn(8)
        out = net(x)
        assert out.shape == (16,)

    def test_mlp_no_hidden(self):
        net = AlignmentNetwork(source_dim=8, target_dim=16, mapper_type="mlp")
        x = torch.randn(8)
        out = net(x)
        assert out.shape == (16,)

    def test_mlp_batch(self):
        net = AlignmentNetwork(
            source_dim=8, target_dim=16, hidden_dims=[12], mapper_type="mlp"
        )
        x = torch.randn(5, 8)
        out = net(x)
        assert out.shape == (5, 16)

    def test_invalid_mapper_type(self):
        with pytest.raises(ValueError, match="mapper_type"):
            AlignmentNetwork(source_dim=8, target_dim=16, mapper_type="cnn")

    def test_invalid_activation(self):
        with pytest.raises(ValueError, match="activation"):
            AlignmentNetwork(source_dim=8, target_dim=16, activation="sigmoid")

    def test_tanh_activation(self):
        net = AlignmentNetwork(
            source_dim=8, target_dim=8, activation="tanh", mapper_type="mlp"
        )
        x = torch.randn(8)
        out = net(x)
        assert out.shape == (8,)

    def test_gelu_activation(self):
        net = AlignmentNetwork(
            source_dim=8, target_dim=8, activation="gelu", mapper_type="mlp"
        )
        x = torch.randn(8)
        out = net(x)
        assert out.shape == (8,)

    def test_source_dim_property(self):
        net = AlignmentNetwork(source_dim=4, target_dim=8)
        assert net.source_dim == 4

    def test_target_dim_property(self):
        net = AlignmentNetwork(source_dim=4, target_dim=8)
        assert net.target_dim == 8

    def test_mapper_type_property(self):
        net = AlignmentNetwork(source_dim=4, target_dim=8, mapper_type="mlp")
        assert net.mapper_type == "mlp"

    def test_to_config(self):
        net = AlignmentNetwork(source_dim=4, target_dim=8)
        cfg = net.to_config()
        assert cfg["source_dim"] == 4
        assert cfg["target_dim"] == 8


# ===================================================================
# TestModalityGraph
# ===================================================================


class TestModalityGraph:
    def test_empty_graph(self):
        g = ModalityGraph()
        assert g.modalities() == []
        assert g.count_edges() == 0

    def test_add_modality(self):
        g = ModalityGraph()
        g.add_modality("image", 8)
        assert g.has_modality("image")
        assert g.get_embedding_dim("image") == 8

    def test_add_modality_no_dim(self):
        g = ModalityGraph()
        g.add_modality("image")
        assert g.get_embedding_dim("image") is None

    def test_add_mapping(self):
        g = ModalityGraph()
        g.add_mapping("image", "text")
        assert g.is_directly_connected("image", "text")
        assert g.is_directly_connected("text", "image")

    def test_direct_neighbors(self):
        g = ModalityGraph()
        g.add_mapping("image", "text")
        g.add_mapping("image", "audio")
        assert g.direct_neighbors("image") == ["audio", "text"]
        assert g.direct_neighbors("text") == ["image"]

    def test_set_embedding_dim(self):
        g = ModalityGraph()
        g.add_modality("image")
        g.set_embedding_dim("image", 16)
        assert g.get_embedding_dim("image") == 16

    def test_find_path_direct(self):
        g = ModalityGraph()
        g.add_mapping("a", "b")
        path = g.find_path("a", "b")
        assert path == ["a", "b"]

    def test_find_path_indirect(self):
        g = ModalityGraph()
        g.add_mapping("a", "b")
        g.add_mapping("b", "c")
        path = g.find_path("a", "c")
        assert path == ["a", "b", "c"]

    def test_find_path_none(self):
        g = ModalityGraph()
        g.add_mapping("a", "b")
        assert g.find_path("a", "c") is None

    def test_find_path_no_source(self):
        g = ModalityGraph()
        assert g.find_path("x", "y") is None

    def test_reachable_modalities(self):
        g = ModalityGraph()
        g.add_mapping("a", "b")
        g.add_mapping("b", "c")
        g.add_mapping("d", "e")
        reachable = g.reachable_modalities("a")
        assert reachable == {"b", "c"}

    def test_reachable_modalities_no_source(self):
        g = ModalityGraph()
        assert g.reachable_modalities("x") == set()

    def test_paths_to_missing(self):
        g = ModalityGraph()
        g.add_mapping("a", "b")
        paths = g.paths_to_missing({"a"}, {"b"})
        assert paths["b"] == ["a", "b"]

    def test_paths_to_missing_no_path(self):
        g = ModalityGraph()
        g.add_mapping("a", "b")
        g.add_modality("c")
        paths = g.paths_to_missing({"a"}, {"c"})
        assert paths["c"] is None

    def test_paths_to_missing_shortest(self):
        g = ModalityGraph()
        g.add_mapping("a", "c")
        g.add_mapping("a", "b")
        g.add_mapping("b", "c")
        paths = g.paths_to_missing({"a"}, {"c"})
        assert paths["c"] == ["a", "c"]

    def test_count_edges(self):
        g = ModalityGraph()
        g.add_mapping("a", "b")
        g.add_mapping("b", "c")
        assert g.count_edges() == 2

    def test_to_config(self):
        g = ModalityGraph()
        g.add_mapping("a", "b")
        g.set_embedding_dim("a", 4)
        cfg = g.to_config()
        assert "a" in cfg["modalities"]
        assert cfg["embedding_dims"]["a"] == 4


# ===================================================================
# TestContrastiveAlignment
# ===================================================================


class TestInfoNCELoss:
    def test_forward(self):
        loss_fn = InfoNCELoss(temperature=0.07)
        anchor = torch.randn(4, 8)
        positive = anchor + 0.01
        negatives = torch.randn(4, 8)
        loss = loss_fn(anchor, positive, negatives)
        assert loss.ndim == 0
        assert loss > 0

    def test_invalid_temperature(self):
        with pytest.raises(ValueError, match="Temperature"):
            InfoNCELoss(temperature=0.0)
        with pytest.raises(ValueError, match="Temperature"):
            InfoNCELoss(temperature=-1.0)

    def test_temperature_property(self):
        loss_fn = InfoNCELoss(temperature=0.5)
        assert loss_fn.temperature == 0.5

    def test_to_config(self):
        loss_fn = InfoNCELoss(temperature=0.1)
        assert loss_fn.to_config() == {"temperature": 0.1}


class TestTripletLoss:
    def test_forward(self):
        loss_fn = TripletLoss(margin=1.0)
        anchor = torch.randn(4, 8)
        positive = anchor + 0.01
        negative = torch.randn(4, 8)
        loss = loss_fn(anchor, positive, negative)
        assert loss.ndim == 0
        assert loss >= 0

    def test_zero_loss(self):
        loss_fn = TripletLoss(margin=0.0)
        anchor = torch.randn(4, 8)
        positive = anchor.clone()
        negative = anchor.clone()
        loss = loss_fn(anchor, positive, negative)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_invalid_margin(self):
        with pytest.raises(ValueError, match="Margin"):
            TripletLoss(margin=-0.5)

    def test_margin_property(self):
        loss_fn = TripletLoss(margin=2.0)
        assert loss_fn.margin == 2.0

    def test_to_config(self):
        loss_fn = TripletLoss(margin=0.5)
        assert loss_fn.to_config() == {"margin": 0.5}


class TestContrastiveAlignmentLoss:
    def test_forward(self):
        loss_fn = ContrastiveAlignmentLoss(margin=1.0)
        a = torch.randn(4, 8)
        b = torch.randn(4, 8)
        labels = torch.tensor([0, 0, 1, 1])
        loss = loss_fn(a, b, labels)
        assert loss.ndim == 0
        assert loss >= 0

    def test_invalid_margin(self):
        with pytest.raises(ValueError, match="Margin"):
            ContrastiveAlignmentLoss(margin=-0.5)

    def test_margin_property(self):
        loss_fn = ContrastiveAlignmentLoss(margin=0.5)
        assert loss_fn.margin == 0.5

    def test_to_config(self):
        loss_fn = ContrastiveAlignmentLoss(margin=0.3)
        assert loss_fn.to_config() == {"margin": 0.3}


# ===================================================================
# TestCrossModalMapper
# ===================================================================


class TestCrossModalMapper:
    def test_create(self):
        graph = ModalityGraph()
        mapper = CrossModalMapper(graph)
        assert mapper.mapping_count() == 0

    def test_add_mapping_network(self):
        graph = ModalityGraph()
        graph.add_modality("image", 8)
        graph.add_modality("text", 8)
        mapper = CrossModalMapper(graph)
        net = AlignmentNetwork(source_dim=8, target_dim=8)
        mapper.add_mapping_network("image", "text", net)
        assert mapper.has_mapping("image", "text")
        assert mapper.mapping_count() == 1

    def test_add_mapping_network_duplicate(self):
        graph = ModalityGraph()
        graph.add_modality("image", 8)
        graph.add_modality("text", 8)
        mapper = CrossModalMapper(graph)
        net = AlignmentNetwork(source_dim=8, target_dim=8)
        mapper.add_mapping_network("image", "text", net)
        with pytest.raises(ValueError, match="already exists"):
            mapper.add_mapping_network("image", "text", net)

    def test_add_mapping_network_dim_mismatch(self):
        graph = ModalityGraph()
        graph.add_modality("image", 8)
        graph.add_modality("text", 16)
        mapper = CrossModalMapper(graph)
        net = AlignmentNetwork(source_dim=8, target_dim=8)
        with pytest.raises(ValueError, match="target_dim"):
            mapper.add_mapping_network("image", "text", net)

    def test_add_mapping_network_source_dim_mismatch(self):
        graph = ModalityGraph()
        graph.add_modality("image", 16)
        graph.add_modality("text", 8)
        mapper = CrossModalMapper(graph)
        net = AlignmentNetwork(source_dim=8, target_dim=8)
        with pytest.raises(ValueError, match="source_dim"):
            mapper.add_mapping_network("image", "text", net)

    def test_get_mapping_network(self):
        graph = ModalityGraph()
        graph.add_modality("image", 8)
        graph.add_modality("text", 8)
        mapper = CrossModalMapper(graph)
        net = AlignmentNetwork(source_dim=8, target_dim=8)
        mapper.add_mapping_network("image", "text", net)
        assert mapper.get_mapping_network("image", "text") is net
        assert mapper.get_mapping_network("text", "image") is None

    def test_get_or_create_mapping_network_creates(self):
        graph = ModalityGraph()
        graph.add_modality("image", 8)
        graph.add_modality("text", 8)
        mapper = CrossModalMapper(graph)
        net = mapper.get_or_create_mapping_network("image", "text")
        assert isinstance(net, AlignmentNetwork)
        assert mapper.has_mapping("image", "text")

    def test_get_or_create_mapping_network_returns_existing(self):
        graph = ModalityGraph()
        graph.add_modality("image", 8)
        graph.add_modality("text", 8)
        mapper = CrossModalMapper(graph)
        net1 = mapper.get_or_create_mapping_network("image", "text")
        net2 = mapper.get_or_create_mapping_network("image", "text")
        assert net1 is net2

    def test_get_or_create_network_no_dims(self):
        graph = ModalityGraph()
        graph.add_modality("image")
        graph.add_modality("text")
        mapper = CrossModalMapper(graph)
        with pytest.raises(ValueError, match="dimensions"):
            mapper.get_or_create_mapping_network("image", "text")

    def test_translate_direct(self):
        graph = ModalityGraph()
        graph.add_modality("image", 8)
        graph.add_modality("text", 8)
        mapper = CrossModalMapper(graph)
        net = AlignmentNetwork(source_dim=8, target_dim=8)
        mapper.add_mapping_network("image", "text", net)
        emb = torch.randn(8)
        result = mapper.translate("image", "text", emb)
        assert result.shape == (8,)

    def test_translate_same_modality(self):
        graph = ModalityGraph()
        graph.add_modality("image", 8)
        mapper = CrossModalMapper(graph)
        emb = torch.randn(8)
        result = mapper.translate("image", "image", emb)
        assert torch.equal(result, emb)

    def test_translate_no_path(self):
        graph = ModalityGraph()
        graph.add_modality("image", 8)
        graph.add_modality("text", 8)
        mapper = CrossModalMapper(graph)
        emb = torch.randn(8)
        with pytest.raises(ValueError, match="No mapping or path"):
            mapper.translate("image", "text", emb)

    def test_translate_along_path(self):
        graph = ModalityGraph()
        graph.add_modality("a", 8)
        graph.add_modality("b", 8)
        graph.add_modality("c", 8)
        mapper = CrossModalMapper(graph)
        mapper.add_mapping_network(
            "a", "b", AlignmentNetwork(source_dim=8, target_dim=8)
        )
        mapper.add_mapping_network(
            "b", "c", AlignmentNetwork(source_dim=8, target_dim=8)
        )
        emb = torch.randn(8)
        result = mapper.translate("a", "c", emb)
        assert result.shape == (8,)

    def test_translate_along_path_missing_middle(self):
        graph = ModalityGraph()
        graph.add_modality("a", 8)
        graph.add_modality("b", 8)
        graph.add_modality("c", 8)
        graph.add_mapping("a", "b")
        graph.add_mapping("b", "c")
        mapper = CrossModalMapper(graph)
        mapper.add_mapping_network(
            "a", "b", AlignmentNetwork(source_dim=8, target_dim=8)
        )
        emb = torch.randn(8)
        with pytest.raises(ValueError, match="Missing mapping network"):
            mapper.translate("a", "c", emb)

    def test_batch_translate(self):
        graph = ModalityGraph()
        graph.add_modality("image", 8)
        graph.add_modality("text", 8)
        mapper = CrossModalMapper(graph)
        net = AlignmentNetwork(source_dim=8, target_dim=8)
        mapper.add_mapping_network("image", "text", net)
        embs = torch.randn(4, 8)
        result = mapper.batch_translate("image", "text", embs)
        assert result.shape == (4, 8)

    def test_batch_translate_single(self):
        graph = ModalityGraph()
        graph.add_modality("image", 8)
        graph.add_modality("text", 8)
        mapper = CrossModalMapper(graph)
        net = AlignmentNetwork(source_dim=8, target_dim=8)
        mapper.add_mapping_network("image", "text", net)
        emb = torch.randn(8)
        result = mapper.batch_translate("image", "text", emb)
        assert result.shape == (1, 8)

    def test_batch_translate_no_path(self):
        graph = ModalityGraph()
        graph.add_modality("image", 8)
        graph.add_modality("text", 8)
        mapper = CrossModalMapper(graph)
        with pytest.raises(ValueError, match="No mapping or path"):
            mapper.batch_translate("image", "text", torch.randn(2, 8))

    def test_batch_translate_along_path(self):
        graph = ModalityGraph()
        graph.add_modality("a", 8)
        graph.add_modality("b", 8)
        graph.add_modality("c", 8)
        mapper = CrossModalMapper(graph)
        mapper.add_mapping_network(
            "a", "b", AlignmentNetwork(source_dim=8, target_dim=8)
        )
        mapper.add_mapping_network(
            "b", "c", AlignmentNetwork(source_dim=8, target_dim=8)
        )
        embs = torch.randn(3, 8)
        result = mapper.batch_translate("a", "c", embs)
        assert result.shape == (3, 8)

    def test_batch_translate_along_path_missing(self):
        graph = ModalityGraph()
        graph.add_modality("a", 8)
        graph.add_modality("b", 8)
        graph.add_modality("c", 8)
        graph.add_mapping("a", "b")
        graph.add_mapping("b", "c")
        mapper = CrossModalMapper(graph)
        mapper.add_mapping_network(
            "a", "b", AlignmentNetwork(source_dim=8, target_dim=8)
        )
        embs = torch.randn(3, 8)
        with pytest.raises(ValueError, match="Missing mapping network"):
            mapper.batch_translate("a", "c", embs)

    def test_available_mappings(self):
        graph = ModalityGraph()
        graph.add_modality("image", 8)
        graph.add_modality("text", 8)
        mapper = CrossModalMapper(graph)
        assert mapper.available_mappings() == []
        mapper.add_mapping_network(
            "image", "text", AlignmentNetwork(source_dim=8, target_dim=8)
        )
        assert mapper.available_mappings() == [("image", "text")]

    def test_to_config(self):
        graph = ModalityGraph()
        graph.add_modality("image", 8)
        graph.add_modality("text", 8)
        mapper = CrossModalMapper(graph)
        mapper.add_mapping_network(
            "image", "text", AlignmentNetwork(source_dim=8, target_dim=8)
        )
        cfg = mapper.to_config()
        assert len(cfg["mappings"]) == 1


# ===================================================================
# TestTransferLoss
# ===================================================================


class TestTransferLoss:
    def test_alignment_loss(self):
        loss_fn = TransferLoss()
        source = torch.randn(4, 8)
        target = torch.randn(4, 8)
        labels = torch.ones(4)
        losses = loss_fn(source, target, labels)
        assert "alignment" in losses
        assert "total" in losses
        assert losses["alignment"] >= 0

    def test_with_reconstruction(self):
        loss_fn = TransferLoss()
        source = torch.randn(4, 8)
        target = source.clone()
        labels = torch.ones(4)
        reconstructed = source + 0.01
        losses = loss_fn(source, target, labels, reconstructed=reconstructed)
        assert "reconstruction" in losses

    def test_with_similarity(self):
        loss_fn = TransferLoss()
        source = torch.randn(4, 8)
        target = source.clone()
        labels = torch.ones(4)
        pairs = [(torch.randn(8), torch.randn(8)) for _ in range(3)]
        losses = loss_fn(
            source, target, labels, original_pairs=pairs, translated_pairs=pairs
        )
        assert "similarity" in losses

    def test_with_consistency(self):
        loss_fn = TransferLoss()
        source = torch.randn(4, 8)
        target = source.clone()
        labels = torch.ones(4)
        fwd = torch.randn(4, 8)
        bwd = torch.randn(4, 8)
        losses = loss_fn(source, target, labels, forward_cycle=fwd, backward_cycle=bwd)
        assert "consistency" in losses

    def test_to_config(self):
        loss_fn = TransferLoss(alignment_weight=0.5)
        cfg = loss_fn.to_config()
        assert cfg["alignment_weight"] == 0.5

    def test_similarity_preservation_empty(self):
        loss_fn = TransferLoss()
        result = loss_fn.similarity_preservation_loss([], [])
        assert result.item() == 0.0

    def test_properties(self):
        loss_fn = TransferLoss(
            alignment_weight=0.3,
            reconstruction_weight=0.4,
            similarity_weight=0.2,
            consistency_weight=0.1,
        )
        assert loss_fn.alignment_weight == 0.3
        assert loss_fn.reconstruction_weight == 0.4
        assert loss_fn.similarity_weight == 0.2
        assert loss_fn.consistency_weight == 0.1


# ===================================================================
# TestPrototypeGenerator
# ===================================================================


class TestPrototypeGenerator:
    def test_synthesize(self):
        graph = ModalityGraph()
        graph.add_modality("image", 8)
        graph.add_modality("text", 8)
        graph.add_mapping("image", "text")
        mapper = CrossModalMapper(graph)
        mapper.add_mapping_network(
            "image", "text", AlignmentNetwork(source_dim=8, target_dim=8)
        )
        gen = PrototypeGenerator(mapper, graph)
        proto = _make_aggregated(modality="image", class_id=0, dim=8)
        result = gen.synthesize(proto, "text")
        assert isinstance(result, SynthesisResult)
        assert result.modality == "text"
        assert result.class_id == 0
        assert result.source_modality == "image"
        assert len(result.prototype_vector) == 8

    def test_synthesize_same_modality_raises(self):
        graph = ModalityGraph()
        graph.add_modality("image", 8)
        mapper = CrossModalMapper(graph)
        gen = PrototypeGenerator(mapper, graph)
        proto = _make_aggregated(modality="image")
        with pytest.raises(ValueError, match="same"):
            gen.synthesize(proto, "image")

    def test_synthesize_no_path(self):
        graph = ModalityGraph()
        graph.add_modality("image", 8)
        graph.add_modality("text", 8)
        mapper = CrossModalMapper(graph)
        gen = PrototypeGenerator(mapper, graph)
        proto = _make_aggregated(modality="image")
        with pytest.raises(ValueError, match="No path"):
            gen.synthesize(proto, "text")

    def test_batch_synthesize(self):
        graph = ModalityGraph()
        graph.add_modality("image", 8)
        graph.add_modality("text", 8)
        graph.add_mapping("image", "text")
        mapper = CrossModalMapper(graph)
        mapper.add_mapping_network(
            "image", "text", AlignmentNetwork(source_dim=8, target_dim=8)
        )
        gen = PrototypeGenerator(mapper, graph)
        protos = [
            _make_aggregated(modality="image", class_id=0, dim=8),
            _make_aggregated(modality="image", class_id=1, dim=8),
        ]
        results = gen.batch_synthesize(protos, "text")
        assert len(results) == 2

    def test_batch_synthesize_empty(self):
        graph = ModalityGraph()
        mapper = CrossModalMapper(graph)
        gen = PrototypeGenerator(mapper, graph)
        assert gen.batch_synthesize([], "text") == []

    def test_batch_synthesize_skips_target_modality(self):
        graph = ModalityGraph()
        graph.add_modality("image", 8)
        graph.add_modality("text", 8)
        graph.add_mapping("image", "text")
        mapper = CrossModalMapper(graph)
        mapper.add_mapping_network(
            "image", "text", AlignmentNetwork(source_dim=8, target_dim=8)
        )
        gen = PrototypeGenerator(mapper, graph)
        protos = [
            _make_aggregated(modality="image", class_id=0, dim=8),
            _make_aggregated(modality="text", class_id=1, dim=8),
        ]
        results = gen.batch_synthesize(protos, "text")
        assert len(results) == 1
        assert results[0].source_modality == "image"

    def test_synthesize_missing_modalities(self):
        graph = ModalityGraph()
        graph.add_modality("image", 8)
        graph.add_modality("text", 8)
        graph.add_mapping("image", "text")
        mapper = CrossModalMapper(graph)
        mapper.add_mapping_network(
            "image", "text", AlignmentNetwork(source_dim=8, target_dim=8)
        )
        gen = PrototypeGenerator(mapper, graph)
        protos = [_make_aggregated(modality="image", class_id=0, dim=8)]
        results = gen.synthesize_missing_modalities(protos, {"image", "text"})
        assert len(results) == 1
        assert results[0].modality == "text"

    def test_synthesize_missing_no_candidates(self):
        graph = ModalityGraph()
        graph.add_modality("image", 8)
        graph.add_modality("text", 8)
        mapper = CrossModalMapper(graph)
        gen = PrototypeGenerator(mapper, graph)
        protos = [_make_aggregated(modality="image")]
        results = gen.synthesize_missing_modalities(protos, {"image", "text"})
        assert results == []

    def test_estimate_confidence(self):
        c = PrototypeGenerator._estimate_confidence(1.0, ["a", "b"])
        assert c == pytest.approx(0.9, abs=1e-6)
        c2 = PrototypeGenerator._estimate_confidence(1.0, ["a", "b", "c"])
        assert c2 == pytest.approx(0.81, abs=1e-6)

    def test_properties(self):
        graph = ModalityGraph()
        mapper = CrossModalMapper(graph)
        gen = PrototypeGenerator(mapper, graph)
        assert gen.mapper is mapper
        assert gen.graph is graph
        assert isinstance(gen.logger, TransferLogger)


# ===================================================================
# TestInferenceEngine
# ===================================================================


class TestInferenceEngine:
    def _setup(self) -> tuple[InferenceEngine, CrossModalMapper, ModalityGraph]:
        graph = ModalityGraph()
        graph.add_modality("image", 8)
        graph.add_modality("text", 8)
        graph.add_modality("audio", 8)
        graph.add_mapping("image", "text")
        graph.add_mapping("text", "audio")
        mapper = CrossModalMapper(graph)
        mapper.add_mapping_network(
            "image", "text", AlignmentNetwork(source_dim=8, target_dim=8)
        )
        mapper.add_mapping_network(
            "text", "audio", AlignmentNetwork(source_dim=8, target_dim=8)
        )
        gen = PrototypeGenerator(mapper, graph)
        engine = InferenceEngine(mapper, graph, gen)
        return engine, mapper, graph

    def test_infer_missing_modalities(self):
        engine, _, _ = self._setup()
        protos = [_make_aggregated(modality="image", class_id=0, dim=8)]
        results = engine.infer_missing_modalities(
            protos, {"text", "audio"}, {"image", "text", "audio"}
        )
        assert len(results) == 2
        for r in results:
            assert isinstance(r, InferenceOutput)
            assert r.source_modality == "image"

    def test_infer_missing_no_target_present(self):
        engine, _, _ = self._setup()
        protos = [_make_aggregated(modality="image")]
        results = engine.infer_missing_modalities(protos, set(), {"image"})
        assert results == []

    def test_infer_missing_no_path(self):
        graph = ModalityGraph()
        graph.add_modality("image", 8)
        graph.add_modality("text", 8)
        mapper = CrossModalMapper(graph)
        gen = PrototypeGenerator(mapper, graph)
        engine = InferenceEngine(mapper, graph, gen)
        protos = [_make_aggregated(modality="image")]
        results = engine.infer_missing_modalities(protos, {"text"}, {"image", "text"})
        assert results == []

    def test_infer_missing_translation_fails(self):
        graph = ModalityGraph()
        graph.add_modality("image", 8)
        graph.add_modality("text", 8)
        mapper = CrossModalMapper(graph)
        net = AlignmentNetwork(source_dim=8, target_dim=8)
        mapper.add_mapping_network("image", "text", net)
        gen = PrototypeGenerator(mapper, graph)
        engine = InferenceEngine(mapper, graph, gen)
        protos = [_make_aggregated(modality="image")]
        mapper._networks.clear()
        results = engine.infer_missing_modalities(protos, {"text"}, {"image", "text"})
        assert results == []

    def test_infer_single(self):
        engine, _, _ = self._setup()
        proto = _make_aggregated(modality="image", class_id=0, dim=8)
        result = engine.infer_single(proto, "text")
        assert isinstance(result, InferenceOutput)
        assert result.modality == "text"
        assert result.class_id == 0

    def test_batch_infer(self):
        engine, _, _ = self._setup()
        protos = [
            _make_aggregated(modality="image", class_id=0, dim=8),
            _make_aggregated(modality="image", class_id=1, dim=8),
        ]
        results = engine.batch_infer(protos, "text")
        assert len(results) == 2

    def test_batch_infer_skips_same_modality(self):
        engine, _, _ = self._setup()
        protos = [_make_aggregated(modality="text")]
        results = engine.batch_infer(protos, "text")
        assert results == []

    def test_batch_infer_no_path(self):
        graph = ModalityGraph()
        graph.add_modality("image", 8)
        graph.add_modality("text", 8)
        mapper = CrossModalMapper(graph)
        gen = PrototypeGenerator(mapper, graph)
        engine = InferenceEngine(mapper, graph, gen)
        protos = [_make_aggregated(modality="image")]
        results = engine.batch_infer(protos, "text")
        assert results == []

    def test_properties(self):
        engine, mapper, graph = self._setup()
        assert engine.mapper is mapper
        assert engine.graph is graph
        assert isinstance(engine.generator, PrototypeGenerator)


# ===================================================================
# TestTransferLogger
# ===================================================================


class TestTransferLogger:
    def test_log_translation(self):
        log = TransferLogger()
        log.log_translation("image", "text", 0, 0.9, 0.01)
        history = log.get_history("translation")
        assert len(history) == 1
        assert history[0]["source_modality"] == "image"

    def test_log_loss(self):
        log = TransferLogger()
        log.log_loss("info_nce", 0.5, step=1)
        history = log.get_history("loss")
        assert len(history) == 1

    def test_log_confidence(self):
        log = TransferLogger()
        log.log_confidence("image", 0, 0.8)
        history = log.get_history("confidence")
        assert len(history) == 1

    def test_log_alignment(self):
        log = TransferLogger()
        log.log_alignment("image", "text", 0.95)
        history = log.get_history("alignment")
        assert len(history) == 1

    def test_get_history_all(self):
        log = TransferLogger()
        log.log_translation("a", "b", 0, 0.9, 0.0)
        log.log_loss("test", 0.1)
        assert len(log.get_history()) == 2

    def test_summary(self):
        log = TransferLogger()
        log.log_translation("a", "b", 0, 0.9, 0.0)
        log.log_loss("test", 0.1)
        summary = log.summary()
        assert summary["total_events"] == 2
        assert summary["translations"] == 1
        assert summary["losses"] == 1

    def test_reset(self):
        log = TransferLogger()
        log.log_translation("a", "b", 0, 0.9, 0.0)
        log.reset()
        assert log.summary()["total_events"] == 0


# ===================================================================
# TestTransferRegistry
# ===================================================================


class TestTransferRegistry:
    def test_create(self):
        r = TransferRegistry()
        assert "info_nce" in r.list_loss_functions()
        assert "cosine" in r.list_similarity_metrics()

    def test_register_alignment_network(self):
        r = TransferRegistry()
        r.register_alignment_network("test", AlignmentNetwork)
        assert "test" in r.list_alignment_networks()

    def test_register_alignment_network_duplicate(self):
        r = TransferRegistry()
        with pytest.raises(ValueError, match="already registered"):
            r.register_alignment_network("default", AlignmentNetwork)

    def test_get_alignment_network_ok(self):
        r = TransferRegistry()
        cls = r.get_alignment_network("default")
        assert cls is AlignmentNetwork

    def test_get_alignment_network_unknown(self):
        r = TransferRegistry()
        with pytest.raises(ValueError, match="Unknown"):
            r.get_alignment_network("nonexistent")

    def test_register_loss_function(self):
        r = TransferRegistry()
        r.register_loss_function("test", InfoNCELoss)
        assert "test" in r.list_loss_functions()

    def test_register_loss_function_duplicate(self):
        r = TransferRegistry()
        with pytest.raises(ValueError, match="already registered"):
            r.register_loss_function("info_nce", InfoNCELoss)

    def test_get_loss_function_ok(self):
        r = TransferRegistry()
        cls = r.get_loss_function("info_nce")
        assert cls is InfoNCELoss

    def test_get_loss_function_unknown(self):
        r = TransferRegistry()
        with pytest.raises(ValueError, match="Unknown"):
            r.get_loss_function("nonexistent")

    def test_register_similarity_metric(self):
        r = TransferRegistry()
        r.register_similarity_metric("manhattan")
        assert "manhattan" in r.list_similarity_metrics()

    def test_register_similarity_metric_duplicate(self):
        r = TransferRegistry()
        with pytest.raises(ValueError, match="already registered"):
            r.register_similarity_metric("cosine")

    def test_get_similarity_metric_ok(self):
        r = TransferRegistry()
        metric = r.get_similarity_metric("cosine")
        assert metric == "cosine"

    def test_get_similarity_metric_unknown(self):
        r = TransferRegistry()
        with pytest.raises(ValueError, match="Unknown"):
            r.get_similarity_metric("nonexistent")

    def test_list_mapper_types(self):
        r = TransferRegistry()
        assert "linear" in r.list_mapper_types()
        assert "mlp" in r.list_mapper_types()

    def test_list_activations(self):
        r = TransferRegistry()
        assert "relu" in r.list_activations()
        assert "tanh" in r.list_activations()
        assert "gelu" in r.list_activations()

    def test_register_component(self):
        r = TransferRegistry()
        r.register_component("double", lambda x: x * 2)
        assert r.get_component("double", x=5) == 10

    def test_register_component_duplicate(self):
        r = TransferRegistry()
        r.register_component("test", lambda: 1)
        with pytest.raises(ValueError, match="already registered"):
            r.register_component("test", lambda: 1)

    def test_get_component_unknown(self):
        r = TransferRegistry()
        with pytest.raises(ValueError, match="Unknown"):
            r.get_component("nonexistent")

    def test_list_components(self):
        r = TransferRegistry()
        assert r.list_components() == []
        r.register_component("test", lambda: 1)
        assert r.list_components() == ["test"]

    def test_to_config(self):
        r = TransferRegistry()
        cfg = r.to_config()
        assert "alignment_networks" in cfg
        assert "loss_functions" in cfg
        assert "similarity_metrics" in cfg
        assert "mapper_types" in cfg
        assert "activations" in cfg


# ===================================================================
# TestTransferFactory
# ===================================================================


class TestTransferFactory:
    def test_create_graph_with_modalities(self):
        modalities = {"image": 8, "text": 16}
        graph = TransferFactory.create_graph_with_modalities(
            modalities, [("image", "text")]
        )
        assert graph.has_modality("image")
        assert graph.get_embedding_dim("image") == 8
        assert graph.is_directly_connected("image", "text")

    def test_create_graph_with_modalities_no_mappings(self):
        modalities = {"image": 8}
        graph = TransferFactory.create_graph_with_modalities(modalities)
        assert graph.has_modality("image")

    def test_create_default_mapper(self):
        modalities = _default_modalities()
        mapper = TransferFactory.create_default_mapper(modalities, _default_mappings())
        assert mapper.mapping_count() == 6

    def test_create_default_mapper_mlp(self):
        modalities = _default_modalities()
        mapper = TransferFactory.create_default_mapper(
            modalities,
            [("image", "text")],
            mapper_type="mlp",
            hidden_dims=[16],
        )
        assert mapper.mapping_count() == 1

    def test_create_default_generator(self):
        modalities = _default_modalities()
        gen = TransferFactory.create_default_generator(modalities, _default_mappings())
        assert isinstance(gen, PrototypeGenerator)

    def test_create_default_inference(self):
        modalities = _default_modalities()
        inf = TransferFactory.create_default_inference(modalities, _default_mappings())
        assert isinstance(inf, InferenceEngine)

    def test_create_loss_info_nce(self):
        loss = TransferFactory.create_loss("info_nce", temperature=0.1)
        assert isinstance(loss, InfoNCELoss)
        assert loss.temperature == 0.1

    def test_create_loss_triplet(self):
        loss = TransferFactory.create_loss("triplet", margin=2.0)
        assert isinstance(loss, TripletLoss)
        assert loss.margin == 2.0

    def test_create_loss_contrastive(self):
        loss = TransferFactory.create_loss("contrastive", margin=0.5)
        assert isinstance(loss, ContrastiveAlignmentLoss)
        assert loss.margin == 0.5

    def test_create_loss_unknown(self):
        with pytest.raises(ValueError, match="Unknown loss type"):
            TransferFactory.create_loss("unknown")

    def test_create_transfer_loss(self):
        loss = TransferFactory.create_transfer_loss(
            alignment_weight=0.5, similarity_metric="dot"
        )
        assert isinstance(loss, TransferLoss)
        assert loss.alignment_weight == 0.5

    def test_create_from_config(self):
        config = {
            "modalities": {"image": 8, "text": 8},
            "mappings": [("image", "text")],
            "loss": {"type": "info_nce", "temperature": 0.5},
        }
        components = TransferFactory.create_from_config(config)
        assert "mapper" in components
        assert "generator" in components
        assert "inference" in components
        assert "loss" in components
        assert "transfer_loss" in components


# ===================================================================
# TestUtils
# ===================================================================


class TestUtils:
    def test_compute_embedding_norm(self):
        from app.knowledge_transfer.utils import compute_embedding_norm

        emb = torch.tensor([3.0, 4.0])
        norm = compute_embedding_norm(emb)
        assert norm.item() == pytest.approx(5.0, abs=1e-6)

    def test_l2_normalize(self):
        from app.knowledge_transfer.utils import l2_normalize

        emb = torch.tensor([3.0, 4.0])
        normalized = l2_normalize(emb)
        assert normalized.norm(p=2).item() == pytest.approx(1.0, abs=1e-6)


# ===================================================================
# TestEdgeCases
# ===================================================================


class TestEdgeCases:
    def test_alignment_network_grad_flow(self):
        net = AlignmentNetwork(source_dim=8, target_dim=16)
        x = torch.randn(8)
        out = net(x)
        loss = out.sum()
        loss.backward()
        for p in net.parameters():
            assert p.grad is not None

    def test_similarity_orthogonal_vectors(self):
        s = Similarity("cosine")
        a = torch.tensor([1.0, 0.0])
        b = torch.tensor([0.0, 1.0])
        assert s.compute(a, b).item() == pytest.approx(0.0, abs=1e-4)

    def test_transfer_loss_all_components(self):
        loss_fn = TransferLoss(
            alignment_weight=0.5,
            reconstruction_weight=0.3,
            similarity_weight=0.1,
            consistency_weight=0.1,
        )
        source = torch.randn(4, 8)
        target = source.clone()
        labels = torch.ones(4)
        reconstructed = source + 0.01
        pairs = [(torch.randn(8), torch.randn(8)) for _ in range(3)]
        fwd = torch.randn(4, 8)
        bwd = fwd.clone()
        losses = loss_fn(
            source,
            target,
            labels,
            reconstructed=reconstructed,
            original_pairs=pairs,
            translated_pairs=pairs,
            forward_cycle=fwd,
            backward_cycle=bwd,
        )
        assert "alignment" in losses
        assert "reconstruction" in losses
        assert "similarity" in losses
        assert "consistency" in losses
        assert "total" in losses

    def test_generator_synthesis_result_dataclass(self):
        result = SynthesisResult(
            modality="text",
            class_id=0,
            prototype_vector=[1.0, 2.0],
            embedding_dim=2,
            confidence=0.8,
            source_modality="image",
            path=["image", "text"],
        )
        assert result.modality == "text"
        assert result.embedding_dim == 2

    def test_inference_output_dataclass(self):
        output = InferenceOutput(
            modality="text",
            class_id=0,
            prototype_vector=[1.0],
            embedding_dim=1,
            confidence=0.7,
            source_modality="image",
            path=["image", "text"],
        )
        assert output.modality == "text"
        assert output.confidence == 0.7

    def test_modality_graph_add_modality_twice(self):
        g = ModalityGraph()
        g.add_modality("image", 8)
        g.add_modality("image", 16)
        assert g.get_embedding_dim("image") == 16

    def test_similarity_pairwise_square(self):
        s = Similarity("cosine")
        a = torch.randn(4, 8)
        mat = s.pairwise(a, a)
        assert mat.shape == (4, 4)
        assert torch.allclose(mat, mat.T, atol=1e-6)

    def test_factory_mlp_with_hidden(self):
        modalities = {"a": 4, "b": 8}
        mapper = TransferFactory.create_default_mapper(
            modalities, [("a", "b")], mapper_type="mlp", hidden_dims=[16, 16]
        )
        assert mapper.mapping_count() == 1

    def test_inference_logger_passthrough(self):
        log = TransferLogger()
        engine, _, _ = TestInferenceEngine()._setup()
        engine2 = InferenceEngine(
            mapper=engine.mapper,
            graph=engine.graph,
            generator=engine.generator,
            logger_instance=log,
        )
        assert engine2._logger is log

    def test_generator_scalar_result(self):
        class ScalarOutputNet(AlignmentNetwork):
            def forward(self, x):
                return torch.tensor(0.5)

        graph = ModalityGraph()
        graph.add_modality("image", 8)
        graph.add_modality("text", 1)
        graph.add_mapping("image", "text")
        mapper = CrossModalMapper(graph)
        mapper.add_mapping_network(
            "image", "text", ScalarOutputNet(source_dim=8, target_dim=1)
        )
        gen = PrototypeGenerator(mapper, graph)
        proto = _make_aggregated(modality="image", class_id=0, dim=8)
        result = gen.synthesize(proto, "text")
        assert result.prototype_vector == [0.5]

    def test_inference_scalar_result(self):
        class ScalarOutputNet(AlignmentNetwork):
            def forward(self, x):
                return torch.tensor(0.5)

        graph = ModalityGraph()
        graph.add_modality("image", 8)
        graph.add_modality("text", 1)
        graph.add_mapping("image", "text")
        mapper = CrossModalMapper(graph)
        mapper.add_mapping_network(
            "image", "text", ScalarOutputNet(source_dim=8, target_dim=1)
        )
        gen = PrototypeGenerator(mapper, graph)
        engine = InferenceEngine(mapper, graph, gen)
        protos = [_make_aggregated(modality="image", class_id=0, dim=8)]
        results = engine.infer_missing_modalities(protos, {"text"}, {"image", "text"})
        assert len(results) == 1
        assert results[0].prototype_vector == [0.5]
