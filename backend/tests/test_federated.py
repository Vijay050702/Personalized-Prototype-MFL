from __future__ import annotations

import json
import time
from typing import Any

import pytest
import torch

from app.federated.adaptive_weighting import AdaptiveWeightCalculator
from app.federated.aggregation import PrototypeAggregator
from app.federated.aggregator import FederatedAggregator
from app.federated.communication import CommunicationHandler
from app.federated.completeness import CompletenessScorer
from app.federated.divergence import DivergenceCalculator
from app.federated.factory import FederatedFactory
from app.federated.models import (
    AggregatedPrototype,
    AggregationRound,
    ClientPrototypePackage,
    DivergenceReport,
    FederatedState,
    ModalityCompletenessReport,
    WeightedPrototype,
)
from app.federated.registry import FederatedRegistry
from app.federated.repository import FederatedRepository
from app.federated.scheduler import RoundScheduler
from app.federated.serialization import PrototypeSerializer
from app.federated.statistics import AggregationStatistics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_package(
    client_id: str = "client_a",
    round_id: int = 1,
    modality: str = "image",
    class_id: int = 0,
    dim: int = 8,
    sample_count: int = 10,
    confidence: float = 1.0,
) -> ClientPrototypePackage:
    return ClientPrototypePackage(
        client_id=client_id,
        round_id=round_id,
        modality=modality,
        class_id=class_id,
        prototype_vector=[float(i) for i in range(dim)],
        sample_count=sample_count,
        embedding_dim=dim,
        confidence=confidence,
    )


def _make_aggregated(
    class_id: int = 0,
    modality: str = "image",
    dim: int = 8,
    sample_count: int = 10,
    round_id: int = 1,
) -> AggregatedPrototype:
    return AggregatedPrototype(
        class_id=class_id,
        modality=modality,
        prototype_vector=[float(i) for i in range(dim)],
        embedding_dim=dim,
        sample_count=sample_count,
        confidence=0.9,
        round_id=round_id,
    )


def _default_aggregator_instance() -> FederatedAggregator:
    return FederatedFactory.create_default()


# ===================================================================
# TestModels
# ===================================================================


class TestClientPrototypePackage:
    def test_create(self):
        pkg = _make_package()
        assert pkg.client_id == "client_a"
        assert pkg.round_id == 1
        assert pkg.modality == "image"
        assert pkg.class_id == 0
        assert len(pkg.prototype_vector) == 8
        assert pkg.sample_count == 10
        assert pkg.embedding_dim == 8
        assert pkg.timestamp > 0
        assert pkg.confidence == 1.0

    def test_to_tensor(self):
        pkg = _make_package(dim=4)
        t = pkg.to_tensor()
        assert isinstance(t, torch.Tensor)
        assert t.shape == (4,)

    def test_package_id(self):
        pkg = _make_package(client_id="c1", round_id=2, modality="audio", class_id=3)
        assert pkg.package_id() == "c1_r2_audio_c3"

    def test_empty_vector_raises(self):
        with pytest.raises(ValueError):
            ClientPrototypePackage(
                client_id="c",
                round_id=1,
                modality="m",
                class_id=0,
                prototype_vector=[],
                sample_count=1,
                embedding_dim=0,
            )

    def test_negative_sample_count(self):
        with pytest.raises(ValueError):
            _make_package(sample_count=0)

    def test_confidence_range(self):
        with pytest.raises(ValueError):
            _make_package(confidence=1.5)
        with pytest.raises(ValueError):
            _make_package(confidence=-0.1)

    def test_metadata(self):
        pkg = _make_package()
        assert pkg.metadata == {}
        pkg2 = ClientPrototypePackage(
            client_id="c",
            round_id=1,
            modality="m",
            class_id=0,
            prototype_vector=[1.0],
            sample_count=1,
            embedding_dim=1,
            metadata={"key": "val"},
        )
        assert pkg2.metadata["key"] == "val"


class TestAggregatedPrototype:
    def test_create(self):
        ap = _make_aggregated()
        assert ap.class_id == 0
        assert ap.modality == "image"
        assert len(ap.prototype_vector) == 8
        assert ap.embedding_dim == 8
        assert ap.sample_count == 10
        assert ap.confidence == 0.9
        assert ap.variance == 0.0
        assert ap.num_contributors == 0

    def test_to_tensor(self):
        ap = _make_aggregated(dim=6)
        t = ap.to_tensor()
        assert isinstance(t, torch.Tensor)
        assert t.shape == (6,)

    def test_aggregated_at(self):
        ap = _make_aggregated()
        assert ap.aggregated_at > 0


class TestAggregationRound:
    def test_create(self):
        r = AggregationRound(round_id=1)
        assert r.round_id == 1
        assert r.participating_clients == []
        assert r.status == "pending"
        assert r.started_at > 0

    def test_complete(self):
        r = AggregationRound(round_id=1)
        r.complete()
        assert r.status == "completed"
        assert r.completed_at is not None

    def test_duration_pending(self):
        r = AggregationRound(round_id=1)
        assert r.duration == 0.0

    def test_duration_completed(self):
        r = AggregationRound(round_id=1)
        time.sleep(0.01)
        r.complete()
        assert r.duration > 0


class TestModalityCompletenessReport:
    def test_create(self):
        r = ModalityCompletenessReport(
            available_modalities=["image"],
            missing_modalities=["audio"],
            total_possible=2,
            completeness_ratio=0.5,
        )
        assert r.completeness_ratio == 0.5
        assert r.available_modalities == ["image"]


class TestDivergenceReport:
    def test_create(self):
        r = DivergenceReport(
            client_id="c1",
            modality="image",
            class_id=0,
            divergence_score=0.5,
            divergence_metric="cosine",
        )
        assert r.client_id == "c1"
        assert r.divergence_score == 0.5


class TestFederatedState:
    def test_create(self):
        s = FederatedState()
        assert s.current_round == 0
        assert s.total_clients_ever == set()
        assert s.packages_received == 0

    def test_to_dict(self):
        s = FederatedState(current_round=5)
        d = s.to_dict()
        assert d["current_round"] == 5
        assert d["packages_received"] == 0


class TestWeightedPrototype:
    def test_create(self):
        wp = WeightedPrototype(
            prototype_vector=torch.tensor([1.0, 2.0]),
            weight=0.5,
            sample_count=10,
            client_id="c1",
            class_id=0,
            modality="image",
            confidence=0.8,
        )
        assert wp.weight == 0.5
        assert wp.client_id == "c1"


# ===================================================================
# TestDivergenceCalculator
# ===================================================================


class TestDivergenceCalculator:
    def test_create_default(self):
        dc = DivergenceCalculator()
        assert dc.metric == "cosine"

    def test_create_custom(self):
        dc = DivergenceCalculator(metric="euclidean")
        assert dc.metric == "euclidean"

    def test_invalid_metric(self):
        with pytest.raises(ValueError):
            DivergenceCalculator(metric="unknown")

    def test_compute_cosine(self):
        dc = DivergenceCalculator(metric="cosine")
        a = _make_package(client_id="a", dim=4)
        b = _make_package(client_id="b", dim=4)
        d = dc.compute(a, b)
        assert isinstance(d, float)
        assert 0.0 <= d <= 2.0

    def test_compute_euclidean(self):
        dc = DivergenceCalculator(metric="euclidean")
        a = _make_package(client_id="a", dim=4)
        b = _make_package(client_id="b", dim=4)
        d = dc.compute(a, b)
        assert isinstance(d, float)
        assert d >= 0

    def test_compute_manhattan(self):
        dc = DivergenceCalculator(metric="manhattan")
        a = _make_package(client_id="a", dim=4)
        b = _make_package(client_id="b", dim=4)
        d = dc.compute(a, b)
        assert isinstance(d, float)
        assert d >= 0

    def test_compute_dim_mismatch(self):
        dc = DivergenceCalculator()
        a = _make_package(dim=4)
        b = _make_package(dim=8)
        with pytest.raises(ValueError, match="dimension mismatch"):
            dc.compute(a, b)

    def test_compute_class_mismatch(self):
        dc = DivergenceCalculator()
        a = _make_package(class_id=0)
        b = _make_package(class_id=1)
        with pytest.raises(ValueError, match="Class ID mismatch"):
            dc.compute(a, b)

    def test_compute_modality_mismatch(self):
        dc = DivergenceCalculator()
        a = _make_package(modality="image")
        b = _make_package(modality="audio")
        with pytest.raises(ValueError, match="Modality mismatch"):
            dc.compute(a, b)

    def test_compute_from_tensors(self):
        dc = DivergenceCalculator()
        a = torch.tensor([1.0, 0.0])
        b = torch.tensor([0.0, 1.0])
        d = dc.compute_from_tensors(a, b)
        assert 0.0 <= d <= 2.0

    def test_compute_from_tensors_shape_mismatch(self):
        dc = DivergenceCalculator()
        with pytest.raises(ValueError, match="Shape mismatch"):
            dc.compute_from_tensors(torch.tensor([1.0]), torch.tensor([1.0, 2.0]))

    def test_compute_pairwise(self):
        dc = DivergenceCalculator()
        pkgs = [
            _make_package(client_id="a"),
            _make_package(client_id="b"),
            _make_package(client_id="c"),
        ]
        pairs = dc.compute_pairwise(pkgs)
        assert len(pairs) == 3
        for c1, c2, d in pairs:
            assert isinstance(c1, str)
            assert isinstance(c2, str)
            assert isinstance(d, float)

    def test_compute_pairwise_different_class_modality(self):
        dc = DivergenceCalculator()
        pkgs = [
            _make_package(client_id="a", class_id=0, modality="image"),
            _make_package(client_id="b", class_id=1, modality="image"),
        ]
        pairs = dc.compute_pairwise(pkgs)
        assert len(pairs) == 0

    def test_batch_divergence(self):
        dc = DivergenceCalculator()
        ref = _make_package(client_id="ref", class_id=0, modality="image")
        pkgs = [_make_package(client_id="a"), _make_package(client_id="b")]
        scores = dc.batch_divergence(pkgs, ref)
        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)

    def test_batch_divergence_filters_mismatch(self):
        dc = DivergenceCalculator()
        ref = _make_package(client_id="ref", class_id=0, modality="image")
        pkgs = [
            _make_package(client_id="a", class_id=0, modality="image"),
            _make_package(client_id="b", class_id=1, modality="image"),
        ]
        scores = dc.batch_divergence(pkgs, ref)
        assert len(scores) == 1


# ===================================================================
# TestCompletenessScorer
# ===================================================================


class TestCompletenessScorer:
    def test_create_default(self):
        cs = CompletenessScorer()
        assert cs.expected_modalities == []

    def test_create_with_expected(self):
        cs = CompletenessScorer(expected_modalities=["image", "audio"])
        assert cs.expected_modalities == ["audio", "image"]

    def test_expected_modalities_setter(self):
        cs = CompletenessScorer()
        cs.expected_modalities = ["audio", "image"]
        assert cs.expected_modalities == ["audio", "image"]

    def test_score_package(self):
        cs = CompletenessScorer()
        pkg = _make_package()
        assert cs.score_package(pkg) == 1.0

    def test_score_client_packages_sets_expected(self):
        cs = CompletenessScorer()
        pkgs = [
            _make_package(modality="image"),
            _make_package(modality="audio"),
        ]
        report = cs.score_client_packages(pkgs)
        assert report.completeness_ratio == 1.0
        assert cs.expected_modalities == ["audio", "image"]

    def test_score_client_packages_with_expected(self):
        cs = CompletenessScorer(expected_modalities=["image", "audio", "text"])
        pkgs = [_make_package(modality="image"), _make_package(modality="audio")]
        report = cs.score_client_packages(pkgs)
        assert report.completeness_ratio == pytest.approx(2.0 / 3.0)
        assert report.missing_modalities == ["text"]

    def test_score_client_packages_no_packages(self):
        cs = CompletenessScorer(expected_modalities=["image"])
        report = cs.score_client_packages([])
        assert report.completeness_ratio == 0.0
        assert report.missing_modalities == ["image"]

    def test_score_modality_set_no_expected(self):
        cs = CompletenessScorer()
        report = cs.score_modality_set({"image", "audio"})
        assert report.completeness_ratio == 1.0
        assert report.missing_modalities == []

    def test_score_modality_set_with_expected(self):
        cs = CompletenessScorer(expected_modalities=["image", "audio", "text"])
        report = cs.score_modality_set({"image"})
        assert report.completeness_ratio == pytest.approx(1.0 / 3.0)

    def test_running_statistics(self):
        cs = CompletenessScorer()
        cs.score_client_packages([_make_package(modality="image")])
        stats = cs.running_statistics()
        assert stats["total_packages_seen"] == 1
        assert "image" in stats["modality_counts"]

    def test_reset(self):
        cs = CompletenessScorer()
        cs.score_client_packages([_make_package(modality="image")])
        cs.reset()
        stats = cs.running_statistics()
        assert stats["total_packages_seen"] == 0

    def test_client_completeness_scores(self):
        cs = CompletenessScorer(expected_modalities=["image", "audio"])
        client_pkgs = {
            "c1": [
                _make_package(client_id="c1", modality="image"),
                _make_package(client_id="c1", modality="audio"),
            ],
            "c2": [_make_package(client_id="c2", modality="image")],
        }
        scores = cs.client_completeness_scores(client_pkgs)
        assert scores["c1"] == 1.0
        assert scores["c2"] == 0.5


# ===================================================================
# TestAdaptiveWeightCalculator
# ===================================================================


class TestAdaptiveWeightCalculator:
    def test_create_default(self):
        awc = AdaptiveWeightCalculator()
        assert awc.temperature == 1.0

    def test_create_custom(self):
        awc = AdaptiveWeightCalculator(temperature=2.0)
        assert awc.temperature == 2.0

    def test_invalid_temperature(self):
        with pytest.raises(ValueError, match="Temperature must be > 0"):
            AdaptiveWeightCalculator(temperature=0.0)
        with pytest.raises(ValueError, match="Temperature must be > 0"):
            AdaptiveWeightCalculator(temperature=-1.0)

    def test_compute_weight_with_divergence(self):
        awc = AdaptiveWeightCalculator()
        pkg = _make_package(sample_count=100)
        w = awc.compute_weight(pkg, completeness_ratio=1.0, divergence_score=0.5)
        assert isinstance(w, float)
        assert 0 < w < 1

    def test_compute_weight_without_divergence(self):
        awc = AdaptiveWeightCalculator()
        pkg = _make_package(sample_count=100)
        w = awc.compute_weight(pkg, completeness_ratio=1.0)
        assert isinstance(w, float)

    def test_compute_weight_low_sample_count(self):
        awc = AdaptiveWeightCalculator()
        pkg = _make_package(sample_count=1)
        w = awc.compute_weight(pkg, completeness_ratio=0.0, divergence_score=2.0)
        assert isinstance(w, float)

    def test_compute_normalized_weights(self):
        awc = AdaptiveWeightCalculator()
        pkgs = [
            _make_package(client_id="a", sample_count=100),
            _make_package(client_id="b", sample_count=50),
        ]
        cr = {"a": 1.0, "b": 0.5}
        dv = {"a": 0.2, "b": 0.8}
        weights = awc.compute_normalized_weights(pkgs, cr, dv)
        assert len(weights) == 2
        assert abs(sum(weights) - 1.0) < 1e-4
        assert all(w > 0 for w in weights)

    def test_compute_normalized_weights_no_divergence(self):
        awc = AdaptiveWeightCalculator()
        pkgs = [_make_package(client_id="a"), _make_package(client_id="b")]
        cr = {"a": 1.0, "b": 0.5}
        weights = awc.compute_normalized_weights(pkgs, cr)
        assert len(weights) == 2

    def test_compute_normalized_weights_empty(self):
        awc = AdaptiveWeightCalculator()
        assert awc.compute_normalized_weights([], {}) == []

    def test_to_config(self):
        awc = AdaptiveWeightCalculator(temperature=1.5, divergence_weight=0.4)
        cfg = awc.to_config()
        assert cfg["temperature"] == 1.5
        assert cfg["divergence_weight"] == 0.4


# ===================================================================
# TestPrototypeAggregator
# ===================================================================


class TestPrototypeAggregator:
    def test_create(self):
        pa = PrototypeAggregator()
        assert pa is not None

    def test_invalid_epsilon(self):
        with pytest.raises(ValueError):
            PrototypeAggregator(epsilon=0.0)
        with pytest.raises(ValueError):
            PrototypeAggregator(epsilon=-0.1)

    def test_aggregate_weighted(self):
        pa = PrototypeAggregator()
        pkgs = [
            _make_package(client_id="a", class_id=0, modality="image", dim=4),
            _make_package(client_id="b", class_id=0, modality="image", dim=4),
        ]
        weights = [0.6, 0.4]
        result = pa.aggregate_weighted(pkgs, weights)
        assert isinstance(result, AggregatedPrototype)
        assert result.class_id == 0
        assert result.modality == "image"
        assert result.embedding_dim == 4
        assert result.sample_count == 20
        assert result.num_contributors == 2
        assert result.confidence > 0

    def test_aggregate_weighted_empty(self):
        pa = PrototypeAggregator()
        with pytest.raises(ValueError, match="No packages"):
            pa.aggregate_weighted([], [])

    def test_aggregate_weighted_length_mismatch(self):
        pa = PrototypeAggregator()
        pkgs = [_make_package()]
        with pytest.raises(ValueError, match="must match"):
            pa.aggregate_weighted(pkgs, [0.5, 0.5])

    def test_aggregate_weighted_zero_weights(self):
        pa = PrototypeAggregator()
        pkgs = [
            _make_package(client_id="a", class_id=0, modality="image", dim=4),
            _make_package(client_id="b", class_id=0, modality="image", dim=4),
        ]
        result = pa.aggregate_weighted(pkgs, [0.0, 0.0])
        assert result.num_contributors == 2

    def test_aggregate_simple(self):
        pa = PrototypeAggregator()
        pkgs = [
            _make_package(client_id="a", class_id=0, modality="image", dim=4),
            _make_package(client_id="b", class_id=0, modality="image", dim=4),
        ]
        result = pa.aggregate_simple(pkgs)
        assert result.num_contributors == 2

    def test_aggregate_simple_empty(self):
        pa = PrototypeAggregator()
        with pytest.raises(ValueError):
            pa.aggregate_simple([])

    def test_aggregate_by_client_weights(self):
        pa = PrototypeAggregator()
        pkgs_by_client = {
            "a": [_make_package(client_id="a", class_id=0, modality="image", dim=4)],
            "b": [_make_package(client_id="b", class_id=0, modality="image", dim=4)],
        }
        result = pa.aggregate_by_client_weights(pkgs_by_client, {"a": 0.7, "b": 0.3})
        assert result.num_contributors == 2

    def test_aggregate_by_client_weights_empty(self):
        pa = PrototypeAggregator()
        with pytest.raises(ValueError):
            pa.aggregate_by_client_weights({}, {})

    def test_per_class_aggregation(self):
        pa = PrototypeAggregator()
        pkgs = [
            _make_package(client_id="a", class_id=0, modality="image", dim=4),
            _make_package(client_id="b", class_id=0, modality="image", dim=4),
            _make_package(client_id="a", class_id=1, modality="image", dim=4),
            _make_package(client_id="b", class_id=1, modality="image", dim=4),
        ]
        weights = [0.25, 0.25, 0.25, 0.25]
        result = pa.per_class_aggregation(pkgs, weights)
        assert len(result) == 2
        assert (0, "image") in result
        assert (1, "image") in result

    def test_validate_incompatible_class(self):
        pa = PrototypeAggregator()
        pkgs = [
            _make_package(class_id=0),
            _make_package(class_id=1),
        ]
        with pytest.raises(ValueError, match="Mixed class_ids"):
            pa.aggregate_weighted(pkgs, [0.5, 0.5])

    def test_validate_incompatible_modality(self):
        pa = PrototypeAggregator()
        pkgs = [
            _make_package(modality="image"),
            _make_package(modality="audio"),
        ]
        with pytest.raises(ValueError, match="Mixed modalities"):
            pa.aggregate_weighted(pkgs, [0.5, 0.5])

    def test_validate_incompatible_dim(self):
        pa = PrototypeAggregator()
        pkgs = [
            _make_package(dim=4),
            _make_package(dim=8),
        ]
        with pytest.raises(ValueError, match="Mixed embedding dimensions"):
            pa.aggregate_weighted(pkgs, [0.5, 0.5])


# ===================================================================
# TestFederatedRepository
# ===================================================================


class TestFederatedRepository:
    def test_create(self):
        repo = FederatedRepository()
        assert repo.state is not None
        assert repo.current_round == 0
        assert repo.global_count() == 0

    def test_store_and_get_global_prototype(self):
        repo = FederatedRepository()
        proto = _make_aggregated(class_id=0, modality="image")
        key = repo.store_global_prototype(proto)
        assert key == "image_c0"
        retrieved = repo.get_global_prototype("image", 0)
        assert retrieved is not None
        assert retrieved.class_id == 0

    def test_get_global_prototype_not_found(self):
        repo = FederatedRepository()
        assert repo.get_global_prototype("image", 99) is None

    def test_list_global_prototypes(self):
        repo = FederatedRepository()
        repo.store_global_prototype(_make_aggregated(class_id=0, modality="image"))
        repo.store_global_prototype(_make_aggregated(class_id=1, modality="audio"))
        assert repo.global_count() == 2
        assert len(repo.list_global_prototypes()) == 2

    def test_has_prototype(self):
        repo = FederatedRepository()
        assert not repo.has_prototype("image", 0)
        repo.store_global_prototype(_make_aggregated(class_id=0, modality="image"))
        assert repo.has_prototype("image", 0)

    def test_store_round_and_get(self):
        repo = FederatedRepository()
        r = AggregationRound(round_id=1)
        repo.store_round(r)
        assert repo.current_round == 1
        retrieved = repo.get_round(1)
        assert retrieved is not None
        assert retrieved.round_id == 1

    def test_get_round_not_found(self):
        repo = FederatedRepository()
        assert repo.get_round(99) is None

    def test_list_rounds(self):
        repo = FederatedRepository()
        repo.store_round(AggregationRound(round_id=2))
        repo.store_round(AggregationRound(round_id=1))
        rounds = repo.list_rounds()
        assert len(rounds) == 2
        assert rounds[0].round_id == 1
        assert rounds[1].round_id == 2

    def test_latest_round(self):
        repo = FederatedRepository()
        assert repo.latest_round() is None
        repo.store_round(AggregationRound(round_id=1))
        repo.store_round(AggregationRound(round_id=2))
        assert repo.latest_round().round_id == 2

    def test_store_client_packages(self):
        repo = FederatedRepository()
        pkgs = [_make_package(client_id="a"), _make_package(client_id="b")]
        repo.store_client_packages(1, pkgs)
        assert len(repo.get_client_packages(1)) == 2
        assert repo.state.packages_received == 2
        assert repo.state.total_clients_ever == {"a", "b"}

    def test_get_client_ids_for_round(self):
        repo = FederatedRepository()
        pkgs = [_make_package(client_id="a"), _make_package(client_id="b")]
        repo.store_client_packages(1, pkgs)
        ids = repo.get_client_ids_for_round(1)
        assert ids == {"a", "b"}

    def test_get_previous_prototype(self):
        repo = FederatedRepository()
        assert repo.get_previous_prototype("image", 0) is None
        proto1 = _make_aggregated(class_id=0, modality="image", round_id=1)
        repo.store_global_prototype(proto1)
        proto2 = _make_aggregated(class_id=0, modality="image", round_id=2)
        repo.store_global_prototype(proto2)
        prev = repo.get_previous_prototype("image", 0)
        assert prev is not None
        assert prev.round_id == 1

    def test_snapshot_and_restore(self):
        repo = FederatedRepository()
        repo.store_global_prototype(_make_aggregated(class_id=0, modality="image"))
        v = repo.create_snapshot()
        assert v == 1
        repo.store_global_prototype(_make_aggregated(class_id=1, modality="image"))
        assert repo.global_count() == 2
        repo.restore_snapshot(v)
        assert repo.global_count() == 1

    def test_restore_snapshot_not_found(self):
        repo = FederatedRepository()
        with pytest.raises(ValueError, match="Snapshot version"):
            repo.restore_snapshot(99)

    def test_list_snapshots(self):
        repo = FederatedRepository()
        assert repo.list_snapshots() == []
        repo.create_snapshot()
        repo.create_snapshot()
        assert repo.list_snapshots() == [1, 2]

    def test_round_count(self):
        repo = FederatedRepository()
        assert repo.round_count() == 0
        repo.store_round(AggregationRound(round_id=1))
        assert repo.round_count() == 1

    def test_export_import_state(self):
        repo = FederatedRepository()
        repo.store_global_prototype(_make_aggregated(class_id=0, modality="image"))
        repo.store_round(AggregationRound(round_id=1))
        state = repo.export_state()
        assert "global_prototypes" in state
        assert "round_history" in state

        repo2 = FederatedRepository()
        repo2.import_state(state)
        assert repo2.global_count() == 1
        assert repo2.round_count() == 1

    def test_clear(self):
        repo = FederatedRepository()
        repo.store_global_prototype(_make_aggregated())
        repo.store_round(AggregationRound(round_id=1))
        repo.clear()
        assert repo.global_count() == 0
        assert repo.round_count() == 0
        assert repo.state.current_round == 0


# ===================================================================
# TestPrototypeSerializer
# ===================================================================


class TestPrototypeSerializer:
    def test_package_to_from_dict(self):
        pkg = _make_package()
        d = PrototypeSerializer.package_to_dict(pkg)
        assert d["client_id"] == "client_a"
        restored = PrototypeSerializer.package_from_dict(d)
        assert restored.client_id == pkg.client_id
        assert restored.prototype_vector == pkg.prototype_vector

    def test_package_to_from_json(self):
        pkg = _make_package()
        j = PrototypeSerializer.package_to_json(pkg)
        restored = PrototypeSerializer.package_from_json(j)
        assert restored.client_id == pkg.client_id

    def test_packages_to_from_json(self):
        pkgs = [_make_package(client_id="a"), _make_package(client_id="b")]
        j = PrototypeSerializer.packages_to_json(pkgs)
        restored = PrototypeSerializer.packages_from_json(j)
        assert len(restored) == 2

    def test_aggregated_to_from_dict(self):
        ap = _make_aggregated()
        d = PrototypeSerializer.aggregated_to_dict(ap)
        restored = PrototypeSerializer.aggregated_from_dict(d)
        assert restored.class_id == ap.class_id

    def test_checksum(self):
        pkg = _make_package()
        cs = PrototypeSerializer.compute_checksum(pkg)
        assert isinstance(cs, str)
        assert len(cs) == 64
        assert PrototypeSerializer.verify_checksum(pkg, cs)
        assert not PrototypeSerializer.verify_checksum(pkg, "x" * 64)

    def test_compress_tensor(self):
        t = torch.tensor([1.123456789, 2.987654321])
        compressed = PrototypeSerializer.compress_tensor(t, decimals=3)
        assert compressed == [1.123, 2.988]

    def test_serialize_with_metadata(self):
        pkgs = [_make_package()]
        j = PrototypeSerializer.serialize_with_metadata(pkgs, {"source": "test"})
        restored_pkgs, meta = PrototypeSerializer.deserialize_with_metadata(j)
        assert len(restored_pkgs) == 1
        assert meta["source"] == "test"


# ===================================================================
# TestCommunicationHandler
# ===================================================================


class TestCommunicationHandler:
    def test_receive_package(self):
        ch = CommunicationHandler()
        pkg = _make_package()
        d = PrototypeSerializer.package_to_dict(pkg)
        received = ch.receive_package(d)
        assert received.client_id == pkg.client_id
        assert ch.received_count == 1

    def test_receive_package_missing_client_id(self):
        ch = CommunicationHandler()
        with pytest.raises(ValueError, match="missing client_id"):
            ch.receive_package(
                {
                    "client_id": "",
                    "round_id": 1,
                    "modality": "image",
                    "class_id": 0,
                    "prototype_vector": [1.0],
                    "sample_count": 1,
                    "embedding_dim": 1,
                }
            )

    def test_receive_package_missing_modality(self):
        ch = CommunicationHandler()
        with pytest.raises(ValueError, match="missing modality"):
            ch.receive_package(
                {
                    "client_id": "c",
                    "round_id": 1,
                    "modality": "",
                    "class_id": 0,
                    "prototype_vector": [1.0],
                    "sample_count": 1,
                    "embedding_dim": 1,
                }
            )

    def test_receive_package_dim_mismatch(self):
        ch = CommunicationHandler()
        data = PrototypeSerializer.package_to_dict(_make_package(dim=4))
        data["embedding_dim"] = 8
        with pytest.raises(ValueError, match="does not match"):
            ch.receive_package(data)

    def test_receive_package_json(self):
        ch = CommunicationHandler()
        pkg = _make_package()
        j = PrototypeSerializer.package_to_json(pkg)
        received = ch.receive_package_json(j)
        assert received.client_id == pkg.client_id

    def test_receive_batch(self):
        ch = CommunicationHandler()
        pkgs = [_make_package(client_id="a"), _make_package(client_id="b")]
        data = [PrototypeSerializer.package_to_dict(p) for p in pkgs]
        received = ch.receive_batch(data)
        assert len(received) == 2
        assert ch.received_count == 2

    def test_send_package(self):
        ch = CommunicationHandler()
        pkg = _make_package()
        d = ch.send_package(pkg)
        assert d["client_id"] == pkg.client_id
        assert ch.sent_count == 1

    def test_send_package_json(self):
        ch = CommunicationHandler()
        pkg = _make_package()
        j = ch.send_package_json(pkg)
        assert isinstance(j, str)

    def test_send_aggregated(self):
        ch = CommunicationHandler()
        pkgs = [_make_package(client_id="a"), _make_package(client_id="b")]
        result = ch.send_aggregated(pkgs)
        assert len(result) == 2
        assert ch.sent_count == 2

    def test_clear_history(self):
        ch = CommunicationHandler()
        ch.receive_package(PrototypeSerializer.package_to_dict(_make_package()))
        ch.send_package(_make_package())
        assert ch.received_count == 1
        assert ch.sent_count == 1
        ch.clear_history()
        assert ch.received_count == 0
        assert ch.sent_count == 0

    def test_validate_sample_count(self):
        ch = CommunicationHandler()
        pkg_dict = PrototypeSerializer.package_to_dict(_make_package(sample_count=1))
        pkg_dict["sample_count"] = 0
        with pytest.raises(ValueError, match="sample_count"):
            ch.receive_package(pkg_dict)


# ===================================================================
# TestRoundScheduler
# ===================================================================


class TestRoundScheduler:
    def test_create_default(self):
        rs = RoundScheduler()
        assert rs.current_round_id == 0
        assert rs.current_round is None

    def test_invalid_timeout(self):
        with pytest.raises(ValueError):
            RoundScheduler(timeout_seconds=0)
        with pytest.raises(ValueError):
            RoundScheduler(timeout_seconds=-1)

    def test_invalid_min_clients(self):
        with pytest.raises(ValueError):
            RoundScheduler(min_clients=0)

    def test_invalid_max_late(self):
        with pytest.raises(ValueError):
            RoundScheduler(max_late_clients=-1)

    def test_new_round(self):
        rs = RoundScheduler()
        r = rs.new_round(["a", "b", "c"])
        assert r.round_id == 1
        assert r.status == "active"
        assert rs.current_round_id == 1

    def test_client_arrived(self):
        rs = RoundScheduler()
        rs.new_round(["a", "b"])
        assert rs.client_arrived("a")
        assert rs.client_arrived("b")

    def test_client_arrived_no_active_round(self):
        rs = RoundScheduler()
        assert not rs.client_arrived("a")

    def test_late_client_accepted(self):
        rs = RoundScheduler(max_late_clients=2)
        rs.new_round(["a"])
        assert rs.client_arrived("b")  # late but accepted
        assert rs.client_arrived("c")  # late but accepted

    def test_late_client_rejected(self):
        rs = RoundScheduler(max_late_clients=0)
        rs.new_round(["a"])
        assert not rs.client_arrived("b")

    def test_has_timed_out_no_round(self):
        rs = RoundScheduler()
        assert not rs.has_timed_out()

    def test_has_timed_out_false(self):
        rs = RoundScheduler(timeout_seconds=300)
        rs.new_round(["a"])
        assert not rs.has_timed_out()

    def test_time_remaining(self):
        rs = RoundScheduler(timeout_seconds=300)
        assert rs.time_remaining() == 0.0  # no active round
        rs.new_round(["a"])
        remaining = rs.time_remaining()
        assert 0 < remaining <= 300

    def test_can_finalize_no_round(self):
        rs = RoundScheduler()
        assert not rs.can_finalize()

    def test_can_finalize_all_arrived(self):
        rs = RoundScheduler(min_clients=2)
        rs.new_round(["a", "b"])
        rs.client_arrived("a")
        rs.client_arrived("b")
        assert rs.can_finalize()

    def test_can_finalize_partial(self):
        rs = RoundScheduler(min_clients=1, allow_partial=True)
        rs.new_round(["a", "b"])
        rs.client_arrived("a")
        assert rs.can_finalize()

    def test_can_finalize_not_ready(self):
        rs = RoundScheduler(min_clients=2, allow_partial=False, timeout_seconds=300)
        rs.new_round(["a", "b"])
        assert rs.can_finalize()

    def test_finalize_round(self):
        rs = RoundScheduler()
        rs.new_round(["a"])
        r = rs.finalize_round()
        assert r.status == "completed"
        assert rs.current_round is None

    def test_finalize_no_active_round(self):
        rs = RoundScheduler()
        with pytest.raises(ValueError, match="No active round"):
            rs.finalize_round()

    def test_abort_round(self):
        rs = RoundScheduler()
        rs.new_round(["a"])
        rs.abort_round()
        assert rs.current_round is None

    def test_abort_no_active_round(self):
        rs = RoundScheduler()
        rs.abort_round()  # no error

    def test_to_config(self):
        rs = RoundScheduler(timeout_seconds=60, min_clients=2)
        cfg = rs.to_config()
        assert cfg["timeout_seconds"] == 60
        assert cfg["min_clients"] == 2


# ===================================================================
# TestAggregationStatistics
# ===================================================================


class TestAggregationStatistics:
    def test_create(self):
        stats = AggregationStatistics()
        assert stats.total_packages_received == 0
        assert stats.unique_clients == 0

    def test_record_package(self):
        stats = AggregationStatistics()
        stats.record_package(
            _make_package(client_id="c1", modality="image", class_id=0)
        )
        assert stats.total_packages_received == 1
        assert stats.unique_clients == 1

    def test_record_packages(self):
        stats = AggregationStatistics()
        stats.record_packages(
            [
                _make_package(client_id="c1"),
                _make_package(client_id="c2"),
            ]
        )
        assert stats.total_packages_received == 2

    def test_record_round_completion(self):
        stats = AggregationStatistics()
        stats.record_round_completion(round_id=1, duration=1.5, num_participants=3)
        assert stats.total_rounds == 1
        assert stats.avg_round_duration == 1.5

    def test_record_weight_distribution(self):
        stats = AggregationStatistics()
        stats.record_weight_distribution(1, [0.2, 0.3, 0.5])
        ws = stats.weight_statistics(1)
        assert ws is not None
        assert ws["mean"] == pytest.approx(0.3333, abs=1e-3)

    def test_weight_statistics_missing(self):
        stats = AggregationStatistics()
        assert stats.weight_statistics(99) is None

    def test_record_drift(self):
        stats = AggregationStatistics()
        old = [_make_aggregated(class_id=0, modality="image")]
        new = [_make_aggregated(class_id=0, modality="image")]
        drifts = stats.record_drift(old, new)
        assert isinstance(drifts, dict)

    def test_record_missing_modalities(self):
        stats = AggregationStatistics()
        stats.record_missing_modalities("c1", {"image", "audio"}, {"image"})
        stats.record_missing_modalities("c2", {"image"}, {"image"})
        d = stats.to_dict()
        assert d["missing_modality_events"] == 1

    def test_unique_modalities_and_classes(self):
        stats = AggregationStatistics()
        stats.record_package(_make_package(modality="image", class_id=0))
        stats.record_package(_make_package(modality="audio", class_id=1))
        assert stats.unique_modalities == ["audio", "image"]
        assert stats.unique_classes == [0, 1]

    def test_average_drift_empty(self):
        stats = AggregationStatistics()
        assert stats.average_drift() == 0.0

    def test_to_dict(self):
        stats = AggregationStatistics()
        d = stats.to_dict()
        assert "total_packages_received" in d
        assert "uptime_seconds" in d

    def test_reset(self):
        stats = AggregationStatistics()
        stats.record_package(_make_package())
        stats.reset()
        assert stats.total_packages_received == 0

    def test_weight_statistics_single(self):
        stats = AggregationStatistics()
        stats.record_weight_distribution(1, [0.5])
        ws = stats.weight_statistics(1)
        assert ws["std"] == 0.0


# ===================================================================
# TestFederatedRegistry
# ===================================================================


class TestFederatedRegistry:
    def test_create(self):
        r = FederatedRegistry()
        assert "cosine" in r.list_divergence_metrics()
        assert "adaptive" in r.list_weight_strategies()
        assert "weighted" in r.list_aggregation_methods()

    def test_register_divergence_metric(self):
        r = FederatedRegistry()
        r.register_divergence_metric("test_metric", "cosine")
        assert r.get_divergence_metric("test_metric") == "cosine"

    def test_register_divergence_metric_duplicate(self):
        r = FederatedRegistry()
        with pytest.raises(ValueError, match="already registered"):
            r.register_divergence_metric("cosine", "cosine")

    def test_get_divergence_metric_unknown(self):
        r = FederatedRegistry()
        with pytest.raises(ValueError, match="Unknown"):
            r.get_divergence_metric("nonexistent")

    def test_unregister_divergence_metric(self):
        r = FederatedRegistry()
        r.unregister_divergence_metric("cosine")
        assert "cosine" not in r.list_divergence_metrics()

    def test_register_weight_strategy(self):
        r = FederatedRegistry()
        r.register_weight_strategy("test", AdaptiveWeightCalculator)
        assert r.get_weight_strategy("test") == AdaptiveWeightCalculator

    def test_register_weight_strategy_duplicate(self):
        r = FederatedRegistry()
        with pytest.raises(ValueError, match="already registered"):
            r.register_weight_strategy("adaptive", AdaptiveWeightCalculator)

    def test_get_weight_strategy_unknown(self):
        r = FederatedRegistry()
        with pytest.raises(ValueError, match="Unknown"):
            r.get_weight_strategy("nonexistent")

    def test_unregister_weight_strategy(self):
        r = FederatedRegistry()
        r.unregister_weight_strategy("adaptive")
        assert "adaptive" not in r.list_weight_strategies()

    def test_register_aggregation_method(self):
        r = FederatedRegistry()
        r.register_aggregation_method("test", PrototypeAggregator)
        assert r.get_aggregation_method("test") == PrototypeAggregator

    def test_register_aggregation_method_duplicate(self):
        r = FederatedRegistry()
        with pytest.raises(ValueError, match="already registered"):
            r.register_aggregation_method("weighted", PrototypeAggregator)

    def test_get_aggregation_method_unknown(self):
        r = FederatedRegistry()
        with pytest.raises(ValueError, match="Unknown"):
            r.get_aggregation_method("nonexistent")

    def test_unregister_aggregation_method(self):
        r = FederatedRegistry()
        r.unregister_aggregation_method("weighted")
        assert "weighted" not in r.list_aggregation_methods()

    def test_register_component(self):
        r = FederatedRegistry()
        r.register_component("test_comp", lambda x: x)
        assert "test_comp" in r.list_components()

    def test_register_component_duplicate(self):
        r = FederatedRegistry()
        r.register_component("test_comp", lambda x: x)
        with pytest.raises(ValueError, match="already registered"):
            r.register_component("test_comp", lambda x: x)

    def test_get_component(self):
        r = FederatedRegistry()
        r.register_component("double", lambda x: x * 2)
        assert r.get_component("double", x=5) == 10

    def test_get_component_unknown(self):
        r = FederatedRegistry()
        with pytest.raises(ValueError, match="Unknown"):
            r.get_component("nonexistent")

    def test_unregister_component(self):
        r = FederatedRegistry()
        r.register_component("test", lambda: 1)
        r.unregister_component("test")
        assert "test" not in r.list_components()

    def test_create_divergence_calculator(self):
        r = FederatedRegistry()
        dc = r.create_divergence_calculator(metric="euclidean")
        assert isinstance(dc, DivergenceCalculator)
        assert dc.metric == "euclidean"

    def test_create_repository(self):
        r = FederatedRegistry()
        repo = r.create_repository()
        assert isinstance(repo, FederatedRepository)

    def test_create_serializer(self):
        r = FederatedRegistry()
        s = r.create_serializer()
        assert isinstance(s, PrototypeSerializer)

    def test_create_statistics(self):
        r = FederatedRegistry()
        s = r.create_statistics()
        assert isinstance(s, AggregationStatistics)

    def test_create_scheduler(self):
        r = FederatedRegistry()
        s = r.create_scheduler(timeout_seconds=60, min_clients=2)
        assert isinstance(s, RoundScheduler)
        assert s.current_round_id == 0

    def test_create_completeness_scorer(self):
        r = FederatedRegistry()
        cs = r.create_completeness_scorer(expected_modalities=["image"])
        assert isinstance(cs, CompletenessScorer)

    def test_create_weight_calculator(self):
        r = FederatedRegistry()
        awc = r.create_weight_calculator(temperature=2.0, divergence_metric="euclidean")
        assert isinstance(awc, AdaptiveWeightCalculator)
        assert awc.temperature == 2.0

    def test_create_aggregator(self):
        r = FederatedRegistry()
        pa = r.create_aggregator(epsilon=1e-6)
        assert isinstance(pa, PrototypeAggregator)

    def test_create_communication_handler(self):
        r = FederatedRegistry()
        ch = r.create_communication_handler()
        assert isinstance(ch, CommunicationHandler)

    def test_to_config(self):
        r = FederatedRegistry()
        cfg = r.to_config()
        assert "divergence_metrics" in cfg
        assert "weight_strategies" in cfg
        assert "aggregation_methods" in cfg
        assert "custom_components" in cfg


# ===================================================================
# TestFederatedFactory
# ===================================================================


class TestFederatedFactory:
    def test_create_default(self):
        fa = FederatedFactory.create_default()
        assert isinstance(fa, FederatedAggregator)
        assert isinstance(fa.repository, FederatedRepository)
        assert isinstance(fa.scheduler, RoundScheduler)
        assert isinstance(fa.statistics, AggregationStatistics)
        assert isinstance(fa.completeness_scorer, CompletenessScorer)
        assert isinstance(fa.divergence_calculator, DivergenceCalculator)
        assert isinstance(fa.weight_calculator, AdaptiveWeightCalculator)

    def test_create_with_config(self):
        config = {
            "scheduler": {"timeout_seconds": 60, "min_clients": 2},
            "weighting": {"temperature": 0.5, "divergence_metric": "euclidean"},
            "expected_modalities": ["image", "audio"],
        }
        fa = FederatedFactory.create_with_config(config)
        assert isinstance(fa, FederatedAggregator)

    def test_create_custom_all_default(self):
        fa = FederatedFactory.create_custom()
        assert isinstance(fa, FederatedAggregator)

    def test_create_custom_with_overrides(self):
        repo = FederatedRepository()
        fa = FederatedFactory.create_custom(repository=repo)
        assert fa.repository is repo

    def test_create_custom_all_overrides(self):
        fa = FederatedFactory.create_custom(
            repository=FederatedRepository(),
            scheduler=RoundScheduler(timeout_seconds=10),
            divergence_calculator=DivergenceCalculator(metric="manhattan"),
            completeness_scorer=CompletenessScorer(expected_modalities=["image"]),
            weight_calculator=AdaptiveWeightCalculator(temperature=3.0),
            statistics=AggregationStatistics(),
            serializer=PrototypeSerializer(),
            communication_handler=CommunicationHandler(),
            aggregator=PrototypeAggregator(epsilon=1e-7),
        )
        assert isinstance(fa, FederatedAggregator)


# ===================================================================
# TestFederatedAggregator
# ===================================================================


class TestFederatedAggregator:
    def test_create(self):
        fa = _default_aggregator_instance()
        assert isinstance(fa, FederatedAggregator)

    def test_start_round(self):
        fa = _default_aggregator_instance()
        r = fa.start_round(["a", "b", "c"])
        assert r.round_id == 1
        assert len(r.participating_clients) == 3

    def test_receive_packages(self):
        fa = _default_aggregator_instance()
        pkgs_data = [
            PrototypeSerializer.package_to_dict(
                _make_package(client_id="a", modality="image", class_id=0)
            ),
            PrototypeSerializer.package_to_dict(
                _make_package(client_id="b", modality="image", class_id=0)
            ),
        ]
        packages = fa.receive_packages(pkgs_data)
        assert len(packages) == 2

    def test_receive_packages_from_client(self):
        fa = _default_aggregator_instance()
        fa.start_round(["a"])
        pkgs_data = [PrototypeSerializer.package_to_dict(_make_package(client_id="a"))]
        packages = fa.receive_packages_from_client("a", pkgs_data)
        assert len(packages) == 1

    def test_compute_completeness(self):
        fa = _default_aggregator_instance()
        pkgs = [
            _make_package(client_id="a", modality="image"),
            _make_package(client_id="a", modality="audio"),
            _make_package(client_id="b", modality="image"),
        ]
        scores = fa.compute_completeness(pkgs)
        assert "a" in scores
        assert "b" in scores

    def test_compute_divergences_no_global(self):
        fa = _default_aggregator_instance()
        pkgs = [
            _make_package(client_id="a", modality="image", class_id=0, dim=4),
            _make_package(client_id="b", modality="image", class_id=0, dim=4),
        ]
        divs = fa.compute_divergences(pkgs)
        assert "a" in divs
        assert "b" in divs
        assert all(isinstance(v, float) for v in divs.values())

    def test_compute_divergences_with_global(self):
        fa = _default_aggregator_instance()
        fa.repository.store_global_prototype(
            _make_aggregated(class_id=0, modality="image", dim=4)
        )
        pkgs = [
            _make_package(client_id="a", modality="image", class_id=0, dim=4),
            _make_package(client_id="b", modality="image", class_id=0, dim=4),
        ]
        divs = fa.compute_divergences(pkgs)
        assert "a" in divs
        assert "b" in divs

    def test_compute_divergences_single_package(self):
        fa = _default_aggregator_instance()
        pkgs = [
            _make_package(client_id="a", modality="image", class_id=0, dim=4),
        ]
        divs = fa.compute_divergences(pkgs)
        assert divs.get("a") == 0.0

    def test_compute_weights(self):
        fa = _default_aggregator_instance()
        pkgs = [
            _make_package(
                client_id="a", modality="image", class_id=0, sample_count=100
            ),
            _make_package(client_id="b", modality="image", class_id=0, sample_count=50),
        ]
        scores = {"a": 1.0, "b": 0.5}
        divs = {"a": 0.2, "b": 0.8}
        weights = fa.compute_weights(pkgs, scores, divs)
        assert len(weights) == 2
        assert abs(sum(weights) - 1.0) < 1e-4

    def test_aggregate(self):
        fa = _default_aggregator_instance()
        fa.start_round(["a", "b"])
        pkgs = [
            _make_package(client_id="a", modality="image", class_id=0, dim=4),
            _make_package(client_id="b", modality="image", class_id=0, dim=4),
        ]
        weights = [0.6, 0.4]
        results = fa.aggregate(pkgs, weights)
        assert len(results) == 1
        assert (0, "image") in results

    def test_store_aggregated(self):
        fa = _default_aggregator_instance()
        proto = _make_aggregated(class_id=0, modality="image")
        results = {(0, "image"): proto}
        fa.store_aggregated(results, round_id=1)
        stored = fa.repository.get_global_prototype("image", 0)
        assert stored is not None

    def test_finalize_round(self):
        fa = _default_aggregator_instance()
        fa.start_round(["a"])
        r = fa.finalize_round()
        assert r is not None
        assert r.status == "completed"

    def test_finalize_round_not_ready(self):
        fa = _default_aggregator_instance()
        fa.start_round(["a", "b"])
        r = fa.finalize_round()
        assert r is not None

    def test_finalize_round_no_active(self):
        fa = _default_aggregator_instance()
        assert fa.finalize_round() is None

    def test_run_round(self):
        fa = _default_aggregator_instance()
        client_packages = {
            "a": [
                PrototypeSerializer.package_to_dict(
                    _make_package(
                        client_id="a",
                        modality="image",
                        class_id=0,
                        dim=4,
                        sample_count=100,
                    )
                ),
                PrototypeSerializer.package_to_dict(
                    _make_package(
                        client_id="a",
                        modality="audio",
                        class_id=0,
                        dim=4,
                        sample_count=100,
                    )
                ),
            ],
            "b": [
                PrototypeSerializer.package_to_dict(
                    _make_package(
                        client_id="b",
                        modality="image",
                        class_id=0,
                        dim=4,
                        sample_count=50,
                    )
                ),
                PrototypeSerializer.package_to_dict(
                    _make_package(
                        client_id="b",
                        modality="audio",
                        class_id=0,
                        dim=4,
                        sample_count=50,
                    )
                ),
            ],
        }
        results = fa.run_round(client_packages)
        assert len(results) == 2
        assert (0, "image") in results
        assert (0, "audio") in results
        assert fa.global_prototypes is not None

    def test_run_round_single_client(self):
        fa = _default_aggregator_instance()
        client_packages = {
            "a": [
                PrototypeSerializer.package_to_dict(
                    _make_package(client_id="a", modality="image", class_id=0, dim=4)
                ),
            ],
        }
        results = fa.run_round(client_packages)
        assert len(results) == 1
        assert fa.statistics.total_rounds == 1

    def test_get_divergence_reports(self):
        fa = _default_aggregator_instance()
        fa.repository.store_global_prototype(
            _make_aggregated(class_id=0, modality="image", dim=4)
        )
        pkgs = [
            _make_package(client_id="a", modality="image", class_id=0, dim=4),
        ]
        reports = fa.get_divergence_reports(pkgs)
        assert len(reports) == 1
        assert reports[0].client_id == "a"

    def test_get_divergence_reports_no_global(self):
        fa = _default_aggregator_instance()
        pkgs = [_make_package()]
        assert fa.get_divergence_reports(pkgs) == []

    def test_get_statistics_summary(self):
        fa = _default_aggregator_instance()
        d = fa.get_statistics_summary()
        assert "total_packages_received" in d

    def test_snapshot_restore(self):
        fa = _default_aggregator_instance()
        fa.repository.store_global_prototype(_make_aggregated(class_id=0))
        v = fa.create_snapshot()
        fa.repository.store_global_prototype(_make_aggregated(class_id=1))
        assert fa.repository.global_count() == 2
        fa.restore_snapshot(v)
        assert fa.repository.global_count() == 1

    def test_export_import_state(self):
        fa = _default_aggregator_instance()
        fa.repository.store_global_prototype(_make_aggregated(class_id=0))
        state = fa.export_state()
        fa2 = _default_aggregator_instance()
        fa2.import_state(state)
        assert fa2.repository.global_count() == 1

    def test_clear(self):
        fa = _default_aggregator_instance()
        fa.repository.store_global_prototype(_make_aggregated())
        fa.clear()
        assert fa.repository.global_count() == 0
        d = fa.get_statistics_summary()
        assert d["total_packages_received"] == 0


# ===================================================================
# TestEdgeCases
# ===================================================================


class TestEdgeCases:
    def test_scheduler_can_finalize_timed_out(self):
        rs = RoundScheduler(timeout_seconds=0.001, min_clients=1)
        rs.new_round(["a", "b"])
        rs.client_arrived("a")
        time.sleep(0.002)
        assert rs.has_timed_out()
        assert rs.can_finalize()

    def test_scheduler_time_remaining_active(self):
        rs = RoundScheduler(timeout_seconds=100)
        rs.new_round(["a"])
        remaining = rs.time_remaining()
        assert 0 < remaining <= 100

    def test_repository_previous_prototype_metadata(self):
        repo = FederatedRepository()
        p1 = _make_aggregated(class_id=0, modality="image", round_id=1)
        repo.store_global_prototype(p1)
        p2 = _make_aggregated(class_id=0, modality="image", round_id=2)
        repo.store_global_prototype(p2)
        stored = repo.get_global_prototype("image", 0)
        assert stored.metadata.get("previous_round") == 1

    def test_aggregation_statistics_weight_single(self):
        stats = AggregationStatistics()
        stats.record_weight_distribution(1, [1.0])
        ws = stats.weight_statistics(1)
        assert ws["std"] == 0.0

    def test_serializer_empty_packages(self):
        j = PrototypeSerializer.packages_to_json([])
        restored = PrototypeSerializer.packages_from_json(j)
        assert restored == []

    def test_communication_handler_custom_serializer(self):
        serializer = PrototypeSerializer()
        ch = CommunicationHandler(serializer=serializer)
        assert ch is not None

    def test_completeness_scorer_running_statistics_empty(self):
        cs = CompletenessScorer()
        stats = cs.running_statistics()
        assert stats["total_packages_seen"] == 0

    def test_federated_aggregator_receive_empty(self):
        fa = _default_aggregator_instance()
        pkgs = fa.receive_packages([])
        assert pkgs == []

    def test_aggregator_per_class_partial(self):
        pa = PrototypeAggregator()
        pkgs = [
            _make_package(client_id="a", class_id=0, modality="image", dim=4),
            _make_package(client_id="b", class_id=0, modality="image", dim=4),
            _make_package(client_id="a", class_id=1, modality="audio", dim=4),
        ]
        weights = [0.4, 0.3, 0.3]
        result = pa.per_class_aggregation(pkgs, weights)
        assert len(result) == 2

    def test_divergence_identical_packages(self):
        dc = DivergenceCalculator(metric="cosine")
        a = _make_package(client_id="a", dim=4)
        b = _make_package(client_id="b", dim=4)
        d = dc.compute(a, b)
        assert d >= 0

    def test_repository_multiple_snapshots(self):
        repo = FederatedRepository()
        v1 = repo.create_snapshot()
        v2 = repo.create_snapshot()
        v3 = repo.create_snapshot()
        assert repo.list_snapshots() == [1, 2, 3]
        repo.restore_snapshot(v2)
        assert repo.list_snapshots() == [1, 2, 3]

    def test_statistics_drift_no_common_keys(self):
        stats = AggregationStatistics()
        old = [_make_aggregated(class_id=0, modality="image")]
        new = [_make_aggregated(class_id=1, modality="audio")]
        drifts = stats.record_drift(old, new)
        assert drifts == {}

    def test_factory_create_with_empty_config(self):
        fa = FederatedFactory.create_with_config({})
        assert isinstance(fa, FederatedAggregator)

    def test_aggregator_run_round_with_global(self):
        fa = _default_aggregator_instance()
        fa.repository.store_global_prototype(
            _make_aggregated(class_id=0, modality="image", dim=4)
        )
        client_packages = {
            "a": [
                PrototypeSerializer.package_to_dict(
                    _make_package(
                        client_id="a",
                        modality="image",
                        class_id=0,
                        dim=4,
                        sample_count=100,
                    )
                ),
            ],
            "b": [
                PrototypeSerializer.package_to_dict(
                    _make_package(
                        client_id="b",
                        modality="image",
                        class_id=0,
                        dim=4,
                        sample_count=50,
                    )
                ),
            ],
        }
        results = fa.run_round(client_packages)
        assert len(results) == 1
