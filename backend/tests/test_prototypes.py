from __future__ import annotations

import time
import uuid
from typing import Any

import pytest
import torch

from app.prototypes.prototype import Prototype
from app.prototypes.repository import PrototypeRepository
from app.prototypes.generator import PrototypeGenerator
from app.prototypes.memory import PrototypeMemory
from app.prototypes.updater import PrototypeUpdater
from app.prototypes.matcher import PrototypeMatcher
from app.prototypes.similarity import SimilarityEngine
from app.prototypes.confidence import ConfidenceEstimator
from app.prototypes.clustering import PrototypeClustering
from app.prototypes.visualization import VisualizationSupport
from app.prototypes.losses import (
    CenterLoss,
    PrototypeCompactnessLoss,
    PrototypeConsistencyLoss,
    PrototypeDiversityLoss,
    PrototypeSeparationLoss,
)
from app.prototypes.metrics import PrototypeMetrics
from app.prototypes.factory import PrototypeFactory
from app.prototypes.utils import (
    Timer,
    check_nan,
    cosine_similarity_matrix,
    euclidean_distance_matrix,
    validate_class_id,
    validate_embedding,
    validate_prototype_list,
    validate_similarity_metric,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_emb(dim: int = 16) -> torch.Tensor:
    return torch.randn(dim)


def _make_proto(
    class_id: int = 0, dim: int = 16, modality: str = "shared", sample_count: int = 1
) -> Prototype:
    return Prototype(
        embedding=_make_emb(dim),
        class_id=class_id,
        modality=modality,
        sample_count=sample_count,
    )


def _make_protos(count: int, dim: int = 16) -> list[Prototype]:
    return [_make_proto(i % 3, dim) for i in range(count)]


# ---------------------------------------------------------------------------
# TestPrototype
# ---------------------------------------------------------------------------


class TestPrototype:
    def test_create(self):
        emb = _make_emb()
        p = Prototype(embedding=emb, class_id=1)
        assert p.class_id == 1
        assert p.sample_count == 1
        assert p.confidence == 1.0
        assert p.modality == "shared"
        assert isinstance(p.prototype_id, str)
        assert len(p.prototype_id) > 0
        assert torch.equal(p.embedding, emb)

    def test_create_with_all_params(self):
        pid = str(uuid.uuid4())
        emb = _make_emb()
        p = Prototype(
            embedding=emb,
            class_id=2,
            modality="image",
            prototype_id=pid,
            sample_count=10,
            confidence=0.8,
            metadata={"source": "test"},
        )
        assert p.prototype_id == pid
        assert p.class_id == 2
        assert p.modality == "image"
        assert p.sample_count == 10
        assert p.confidence == 0.8
        assert p.metadata["source"] == "test"

    def test_invalid_embedding_type(self):
        with pytest.raises(TypeError):
            Prototype(embedding="not_a_tensor", class_id=0)

    def test_invalid_embedding_dim(self):
        with pytest.raises(ValueError):
            Prototype(embedding=torch.randn(2, 16), class_id=0)

    def test_empty_embedding(self):
        with pytest.raises(ValueError):
            Prototype(embedding=torch.tensor([]), class_id=0)

    def test_nan_embedding(self):
        with pytest.raises(ValueError):
            Prototype(embedding=torch.tensor([float("nan"), 1.0]), class_id=0)

    def test_inf_embedding(self):
        with pytest.raises(ValueError):
            Prototype(embedding=torch.tensor([float("inf"), 1.0]), class_id=0)

    def test_property_prototype_id(self):
        p = _make_proto()
        assert isinstance(p.prototype_id, str)

    def test_property_class_id_getter(self):
        p = _make_proto(class_id=5)
        assert p.class_id == 5

    def test_class_id_setter(self):
        p = _make_proto(class_id=0)
        p.class_id = 3
        assert p.class_id == 3

    def test_property_modality(self):
        p = _make_proto(modality="audio")
        assert p.modality == "audio"

    def test_property_embedding_getter(self):
        emb = _make_emb()
        p = Prototype(embedding=emb, class_id=0)
        assert torch.equal(p.embedding, emb)

    def test_embedding_setter(self):
        p = _make_proto()
        new_emb = _make_emb()
        p.embedding = new_emb
        assert torch.equal(p.embedding, new_emb)

    def test_embedding_setter_validates(self):
        p = _make_proto()
        with pytest.raises(ValueError):
            p.embedding = torch.randn(2, 16)

    def test_sample_count_getter(self):
        p = _make_proto()
        assert p.sample_count >= 1

    def test_sample_count_setter_clamps_below_zero(self):
        p = _make_proto()
        p.sample_count = -5
        assert p.sample_count == 0

    def test_sample_count_setter(self):
        p = _make_proto(sample_count=1)
        p.sample_count = 42
        assert p.sample_count == 42

    def test_confidence_getter(self):
        p = _make_proto()
        assert p.confidence == 1.0

    def test_confidence_setter_clamps(self):
        p = _make_proto()
        p.confidence = 1.5
        assert p.confidence == 1.0
        p.confidence = -0.5
        assert p.confidence == 0.0

    def test_confidence_setter(self):
        p = _make_proto()
        p.confidence = 0.75
        assert p.confidence == 0.75

    def test_timestamp(self):
        p = _make_proto()
        assert isinstance(p.timestamp, float)
        assert p.timestamp > 0

    def test_metadata(self):
        p = _make_proto()
        assert isinstance(p.metadata, dict)
        p2 = Prototype(embedding=_make_emb(), class_id=0, metadata={"key": "val"})
        assert p2.metadata["key"] == "val"

    def test_update_embedding(self):
        p = _make_proto()
        old_emb = p.embedding.clone()
        new_emb = _make_emb()
        p.update(embedding=new_emb)
        assert torch.equal(p.embedding, new_emb)
        assert not torch.equal(p.embedding, old_emb)

    def test_update_sample_count(self):
        p = _make_proto(sample_count=5)
        p.update(embedding=_make_emb(), sample_count=3)
        assert p.sample_count == 8

    def test_update_confidence(self):
        p = _make_proto()
        p.update(embedding=_make_emb(), confidence=0.5)
        assert p.confidence == 0.5

    def test_update_metadata(self):
        p = _make_proto()
        p.update(embedding=_make_emb(), metadata={"new": "data"})
        assert p.metadata["new"] == "data"

    def test_update_timestamp_changes(self):
        p = _make_proto()
        old_ts = p.timestamp
        time.sleep(0.01)
        p.update(embedding=_make_emb())
        assert p.timestamp > old_ts

    def test_clone_different_id(self):
        p = _make_proto()
        clone = p.clone()
        assert clone.prototype_id != p.prototype_id
        assert torch.equal(clone.embedding, p.embedding)
        assert clone.class_id == p.class_id
        assert clone.modality == p.modality
        assert clone.sample_count == p.sample_count
        assert clone.confidence == p.confidence

    def test_distance_euclidean(self):
        a = Prototype(embedding=torch.zeros(4), class_id=0)
        b = Prototype(embedding=torch.ones(4) * 3, class_id=1)
        d = a.distance(b, metric="euclidean")
        assert d.item() == pytest.approx(6.0, abs=1e-4)

    def test_distance_cosine(self):
        a = Prototype(embedding=torch.tensor([1.0, 0.0]), class_id=0)
        b = Prototype(embedding=torch.tensor([0.0, 1.0]), class_id=1)
        d = a.distance(b, metric="cosine")
        assert d.item() == pytest.approx(1.0, abs=1e-4)

    def test_distance_manhattan(self):
        a = Prototype(embedding=torch.tensor([1.0, 2.0]), class_id=0)
        b = Prototype(embedding=torch.tensor([4.0, 6.0]), class_id=1)
        d = a.distance(b, metric="manhattan")
        assert d.item() == pytest.approx(7.0, abs=1e-4)

    def test_distance_invalid_metric(self):
        a = _make_proto()
        b = _make_proto()
        with pytest.raises(ValueError):
            a.distance(b, metric="unknown")

    def test_distance_with_tensor(self):
        a = _make_proto()
        t = _make_emb()
        d = a.distance(t)
        assert isinstance(d, torch.Tensor)

    def test_similarity_cosine(self):
        a = Prototype(embedding=torch.tensor([1.0, 0.0]), class_id=0)
        b = Prototype(embedding=torch.tensor([1.0, 0.0]), class_id=1)
        s = a.similarity(b, metric="cosine")
        assert s.item() == pytest.approx(1.0, abs=1e-4)

    def test_similarity_dot(self):
        a = Prototype(embedding=torch.tensor([2.0, 3.0]), class_id=0)
        b = Prototype(embedding=torch.tensor([4.0, 5.0]), class_id=1)
        s = a.similarity(b, metric="dot")
        assert s.item() == pytest.approx(23.0, abs=1e-4)

    def test_similarity_euclidean(self):
        a = Prototype(embedding=torch.zeros(4), class_id=0)
        b = Prototype(embedding=torch.zeros(4), class_id=1)
        s = a.similarity(b, metric="euclidean")
        assert s.item() == pytest.approx(1.0, abs=1e-4)

    def test_similarity_invalid_metric(self):
        a = _make_proto()
        b = _make_proto()
        with pytest.raises(ValueError):
            a.similarity(b, metric="unknown")

    def test_similarity_with_tensor(self):
        a = _make_proto()
        t = _make_emb()
        s = a.similarity(t)
        assert isinstance(s, torch.Tensor)

    def test_normalize(self):
        p = _make_proto()
        p.normalize()
        assert p.embedding.norm(p=2).item() == pytest.approx(1.0, abs=1e-4)

    def test_normalize_zero_vector(self):
        p = Prototype(embedding=torch.zeros(16), class_id=0)
        p.normalize()
        assert torch.allclose(p.embedding, torch.zeros(16))

    def test_to_dict(self):
        emb = _make_emb()
        p = Prototype(
            embedding=emb,
            class_id=3,
            modality="sensor",
            sample_count=7,
            confidence=0.9,
            metadata={"env": "test"},
        )
        d = p.to_dict()
        assert d["class_id"] == 3
        assert d["modality"] == "sensor"
        assert d["sample_count"] == 7
        assert d["confidence"] == 0.9
        assert d["metadata"]["env"] == "test"
        assert d["embedding_dim"] == emb.size(0)
        assert isinstance(d["embedding"], list)
        assert len(d["embedding"]) == emb.size(0)

    def test_repr(self):
        p = _make_proto(class_id=2)
        r = repr(p)
        assert "Prototype" in r
        assert "class=2" in r

    def test_immutable_embedding_on_creation(self):
        emb = _make_emb()
        p = Prototype(embedding=emb, class_id=0)
        emb_mod = emb.clone()
        emb_mod[0] = 999
        assert not torch.equal(p.embedding, emb_mod)

    def test_immutable_embedding_on_setter(self):
        p = _make_proto()
        new_emb = _make_emb()
        p.embedding = new_emb
        new_emb[0] = 999
        assert not torch.equal(p.embedding, new_emb)


# ---------------------------------------------------------------------------
# TestPrototypeRepository
# ---------------------------------------------------------------------------


class TestPrototypeRepository:
    def test_store_and_retrieve(self):
        repo = PrototypeRepository()
        p = _make_proto()
        repo.store(p)
        retrieved = repo.retrieve(p.prototype_id)
        assert retrieved is p

    def test_store_overwrite_warning(self, caplog):
        repo = PrototypeRepository()
        p = _make_proto()
        repo.store(p)
        repo.store(p)
        assert any("Overwriting" in r.message for r in caplog.records)

    def test_retrieve_not_found(self):
        repo = PrototypeRepository()
        with pytest.raises(KeyError):
            repo.retrieve("nonexistent")

    def test_replace(self):
        repo = PrototypeRepository()
        p = _make_proto()
        repo.store(p)
        new_p = Prototype(
            embedding=_make_emb(), class_id=2, prototype_id=p.prototype_id
        )
        repo.replace(new_p)
        assert repo.retrieve(p.prototype_id).class_id == 2

    def test_replace_not_found(self):
        repo = PrototypeRepository()
        p = _make_proto()
        with pytest.raises(KeyError):
            repo.replace(p)

    def test_update(self):
        repo = PrototypeRepository()
        p = _make_proto(class_id=0)
        repo.store(p)
        new_emb = _make_emb()
        repo.update(p.prototype_id, embedding=new_emb, confidence=0.5, sample_count=10)
        updated = repo.retrieve(p.prototype_id)
        assert torch.equal(updated.embedding, new_emb)
        assert updated.confidence == 0.5
        assert updated.sample_count == 10

    def test_update_class_id(self):
        repo = PrototypeRepository()
        p = _make_proto(class_id=0)
        repo.store(p)
        repo.update(p.prototype_id, class_id=99)
        assert repo.retrieve(p.prototype_id).class_id == 99

    def test_update_metadata(self):
        repo = PrototypeRepository()
        p = _make_proto()
        repo.store(p)
        repo.update(p.prototype_id, metadata={"key": "val"})
        assert repo.retrieve(p.prototype_id).metadata["key"] == "val"

    def test_remove(self):
        repo = PrototypeRepository()
        p = _make_proto()
        repo.store(p)
        repo.remove(p.prototype_id)
        assert repo.is_empty

    def test_remove_not_found(self):
        repo = PrototypeRepository()
        with pytest.raises(KeyError):
            repo.remove("nonexistent")

    def test_list(self):
        repo = PrototypeRepository()
        protos = _make_protos(5)
        for p in protos:
            repo.store(p)
        assert len(repo.list()) == 5

    def test_clear(self):
        repo = PrototypeRepository()
        for p in _make_protos(3):
            repo.store(p)
        repo.clear()
        assert repo.is_empty
        assert repo.size == 0

    def test_filter(self):
        repo = PrototypeRepository()
        for p in _make_protos(6):
            repo.store(p)
        result = repo.filter(lambda p: p.class_id == 0)
        assert all(r.class_id == 0 for r in result)

    def test_size(self):
        repo = PrototypeRepository()
        assert repo.size == 0
        repo.store(_make_proto())
        assert repo.size == 1

    def test_is_empty(self):
        repo = PrototypeRepository()
        assert repo.is_empty
        repo.store(_make_proto())
        assert not repo.is_empty

    def test_by_class(self):
        repo = PrototypeRepository()
        for p in _make_protos(9):
            repo.store(p)
        result = repo.by_class(1)
        assert all(r.class_id == 1 for r in result)
        assert len(result) == 3

    def test_by_modality(self):
        repo = PrototypeRepository()
        repo.store(_make_proto(modality="image"))
        repo.store(_make_proto(modality="audio"))
        result = repo.by_modality("image")
        assert len(result) == 1
        assert result[0].modality == "image"

    def test_class_ids(self):
        repo = PrototypeRepository()
        for p in _make_protos(6):
            repo.store(p)
        assert repo.class_ids() == {0, 1, 2}

    def test_modalities(self):
        repo = PrototypeRepository()
        repo.store(_make_proto(modality="a"))
        repo.store(_make_proto(modality="b"))
        assert repo.modalities() == {"a", "b"}

    def test_get_embeddings_matrix(self):
        repo = PrototypeRepository()
        for p in _make_protos(3):
            repo.store(p)
        mat = repo.get_embeddings_matrix()
        assert mat.shape == (3, 16)

    def test_get_embeddings_matrix_empty(self):
        repo = PrototypeRepository()
        mat = repo.get_embeddings_matrix()
        assert mat.shape == (0, 0)

    def test_get_labels(self):
        repo = PrototypeRepository()
        repo.store(_make_proto(class_id=5))
        repo.store(_make_proto(class_id=3))
        assert sorted(repo.get_labels()) == [3, 5]

    def test_statistics(self):
        repo = PrototypeRepository()
        for p in _make_protos(4):
            repo.store(p)
        stats = repo.statistics()
        assert stats["count"] == 4
        assert stats["classes"] == 3
        assert stats["total_samples"] == 4

    def test_statistics_empty(self):
        repo = PrototypeRepository()
        stats = repo.statistics()
        assert stats["count"] == 0

    def test_export_state(self):
        repo = PrototypeRepository()
        for p in _make_protos(2):
            repo.store(p)
        state = repo.export_state()
        assert len(state["prototypes"]) == 2
        assert "embedding" in state["prototypes"][0]

    def test_import_state(self):
        repo = PrototypeRepository()
        source = PrototypeRepository()
        source.store(_make_proto(class_id=10))
        source.store(_make_proto(class_id=20))
        state = source.export_state()
        repo.import_state(state)
        assert repo.size == 2
        assert repo.by_class(10)

    def test_import_state_clears_first(self):
        repo = PrototypeRepository()
        repo.store(_make_proto())
        state = {"prototypes": []}
        repo.import_state(state)
        assert repo.is_empty

    def test_repr(self):
        repo = PrototypeRepository()
        assert "PrototypeRepository" in repr(repo)


# ---------------------------------------------------------------------------
# TestPrototypeGenerator
# ---------------------------------------------------------------------------


class TestPrototypeGenerator:
    def test_centroid_strategy(self):
        gen = PrototypeGenerator(strategy="centroid")
        embs = torch.randn(10, 8)
        labels = torch.tensor([0] * 5 + [1] * 5)
        proto = gen.generate_from_embeddings(embs, labels, class_id=0)
        assert proto.class_id == 0
        assert proto.embedding.shape == (8,)
        assert proto.sample_count == 5

    def test_weighted_centroid(self):
        gen = PrototypeGenerator(strategy="weighted_centroid")
        embs = torch.randn(10, 8)
        labels = torch.tensor([0] * 5 + [1] * 5)
        weights = torch.randn(10).abs()
        proto = gen.generate_from_embeddings(embs, labels, class_id=0, weights=weights)
        assert proto.class_id == 0
        assert proto.embedding.shape == (8,)

    def test_weighted_centroid_no_weights_fallsback(self):
        gen = PrototypeGenerator(strategy="weighted_centroid")
        embs = torch.randn(10, 8)
        labels = torch.tensor([0] * 10)
        proto = gen.generate_from_embeddings(embs, labels, class_id=0)
        assert proto.class_id == 0

    def test_median_strategy(self):
        gen = PrototypeGenerator(strategy="median")
        embs = torch.randn(10, 8)
        labels = torch.tensor([0] * 5 + [1] * 5)
        proto = gen.generate_from_embeddings(embs, labels, class_id=1)
        assert proto.class_id == 1
        assert proto.sample_count == 5

    def test_generate_from_embeddings_no_samples(self):
        gen = PrototypeGenerator()
        embs = torch.randn(5, 8)
        labels = torch.tensor([0, 0, 0, 0, 0])
        with pytest.raises(ValueError):
            gen.generate_from_embeddings(embs, labels, class_id=1)

    def test_generate_all(self):
        gen = PrototypeGenerator()
        embs = torch.randn(12, 8)
        labels = torch.tensor([0] * 4 + [1] * 4 + [2] * 4)
        protos = gen.generate_all(embs, labels)
        assert len(protos) == 3

    def test_generate_from_repository(self):
        gen = PrototypeGenerator()
        repo = PrototypeRepository()
        embs = torch.randn(10, 8)
        labels = torch.tensor([0] * 5 + [1] * 5)
        n = gen.generate_from_repository(repo, embs, labels)
        assert n == 2
        assert repo.size == 2

    def test_generate_from_repository_replaces_existing(self):
        gen = PrototypeGenerator()
        repo = PrototypeRepository()
        p = _make_proto(class_id=0, dim=8)
        repo.store(p)
        embs = torch.randn(10, 8)
        labels = torch.zeros(10, dtype=torch.long)
        n = gen.generate_from_repository(repo, embs, labels)
        assert n == 1
        assert repo.size == 1

    def test_incremental_update(self):
        gen = PrototypeGenerator()
        p = _make_proto(dim=8)
        old = p.embedding.clone()
        new_embs = torch.randn(3, 8)
        gen.incremental_update(p, new_embs, alpha=0.5)
        assert p.sample_count == 1 + 3
        assert not torch.equal(p.embedding, old)

    def test_incremental_update_1d(self):
        gen = PrototypeGenerator()
        p = _make_proto(dim=8)
        old = p.sample_count
        gen.incremental_update(p, _make_emb(8), alpha=0.5)
        assert p.sample_count == old + 1

    def test_batch_update(self):
        gen = PrototypeGenerator()
        protos = [_make_proto(class_id=0, dim=8), _make_proto(class_id=1, dim=8)]
        embs = torch.randn(6, 8)
        labels = torch.tensor([0, 0, 0, 1, 1, 1])
        updated = gen.batch_update(protos, embs, labels)
        assert len(updated) == 2

    def test_invalid_strategy(self):
        gen = PrototypeGenerator(strategy="invalid")
        with pytest.raises(ValueError):
            gen.generate_from_embeddings(torch.randn(5, 8), torch.zeros(5), class_id=0)

    def test_strategy_property(self):
        gen = PrototypeGenerator(strategy="median")
        assert gen.strategy == "median"


# ---------------------------------------------------------------------------
# TestPrototypeMemory
# ---------------------------------------------------------------------------


class TestPrototypeMemory:
    def test_store_global(self):
        mem = PrototypeMemory()
        p = _make_proto()
        mem.store_global(p)
        assert mem.global_size == 1

    def test_store_local(self):
        mem = PrototypeMemory()
        p = _make_proto()
        mem.store_local(p)
        assert mem.local_size == 1

    def test_promote_to_global(self):
        mem = PrototypeMemory()
        p = _make_proto()
        mem.store_local(p)
        result = mem.promote_to_global(p.prototype_id)
        assert result is not None
        assert mem.local_size == 0
        assert mem.global_size == 1

    def test_promote_nonexistent(self):
        mem = PrototypeMemory()
        result = mem.promote_to_global("nonexistent")
        assert result is None

    def test_snapshot_and_restore(self):
        mem = PrototypeMemory()
        p = _make_proto()
        mem.store_global(p)
        mem.snapshot()
        mem.global_repo.clear()
        mem.local_repo.clear()
        mem.restore_snapshot(-1)
        assert mem.global_size == 1

    def test_restore_no_snapshots(self):
        mem = PrototypeMemory()
        with pytest.raises(ValueError):
            mem.restore_snapshot()

    def test_clear_snapshots(self):
        mem = PrototypeMemory()
        mem.snapshot()
        mem.snapshot()
        assert mem.num_snapshots == 2
        mem.clear_snapshots()
        assert mem.num_snapshots == 0

    def test_max_snapshots(self):
        mem = PrototypeMemory()
        for _ in range(60):
            mem.snapshot()
        assert mem.num_snapshots == 50

    def test_eviction_global(self):
        mem = PrototypeMemory(max_global=2, max_local=10)
        for _ in range(3):
            mem.store_global(_make_proto())
        assert mem.global_size == 2

    def test_eviction_local(self):
        mem = PrototypeMemory(max_global=10, max_local=2)
        for _ in range(3):
            mem.store_local(_make_proto())
        assert mem.local_size == 2

    def test_age_prototypes(self):
        mem = PrototypeMemory()
        p = _make_proto()
        mem.store_global(p)
        aged = mem.age_prototypes(max_age=-1.0)
        assert aged == 1
        assert mem.global_size == 0

    def test_age_no_prototypes(self):
        mem = PrototypeMemory()
        mem.store_global(_make_proto())
        aged = mem.age_prototypes(max_age=86400.0)
        assert aged == 0

    def test_statistics(self):
        mem = PrototypeMemory()
        mem.store_global(_make_proto())
        mem.store_local(_make_proto())
        stats = mem.statistics()
        assert stats["global_count"] == 1
        assert stats["local_count"] == 1
        assert stats["total"] == 2
        assert stats["snapshots"] == 0

    def test_clear(self):
        mem = PrototypeMemory()
        mem.store_global(_make_proto())
        mem.store_local(_make_proto())
        mem.snapshot()
        mem.clear()
        assert mem.total_size == 0
        assert mem.num_snapshots == 0

    def test_global_repo_property(self):
        mem = PrototypeMemory()
        assert isinstance(mem.global_repo, PrototypeRepository)

    def test_local_repo_property(self):
        mem = PrototypeMemory()
        assert isinstance(mem.local_repo, PrototypeRepository)

    def test_total_size(self):
        mem = PrototypeMemory()
        mem.store_global(_make_proto())
        mem.store_local(_make_proto())
        assert mem.total_size == 2

    def test_get_history(self):
        mem = PrototypeMemory()
        p = _make_proto()
        mem.store_global(p)
        history = mem.get_history(p.prototype_id)
        assert len(history) == 1
        assert history[0]["event"] == "stored"

    def test_get_history_empty(self):
        mem = PrototypeMemory()
        assert mem.get_history("nonexistent") == []

    def test_repr(self):
        mem = PrototypeMemory()
        assert "PrototypeMemory" in repr(mem)


# ---------------------------------------------------------------------------
# TestPrototypeUpdater
# ---------------------------------------------------------------------------


class TestPrototypeUpdater:
    def test_ema_update(self):
        updater = PrototypeUpdater(strategy="ema")
        p = _make_proto(dim=8)
        old = p.embedding.clone()
        new_emb = torch.randn(8)
        updater.update(p, new_emb, alpha=0.9)
        expected = 0.9 * old + 0.1 * new_emb
        assert torch.allclose(p.embedding, expected, atol=1e-6)
        assert p.sample_count == 2

    def test_moving_average_update(self):
        updater = PrototypeUpdater(strategy="moving_average")
        p = Prototype(embedding=torch.zeros(8), class_id=0, sample_count=4)
        new_emb = torch.ones(8) * 10
        updater.update(p, new_emb)
        expected = (torch.zeros(8) * 4 + torch.ones(8) * 10) / 5
        assert torch.allclose(p.embedding, expected, atol=1e-6)

    def test_replacement_update(self):
        updater = PrototypeUpdater(strategy="replacement")
        p = _make_proto(dim=8)
        new_emb = torch.randn(8)
        updater.update(p, new_emb)
        assert torch.equal(p.embedding, new_emb)
        assert p.sample_count == 1

    def test_weighted_update(self):
        updater = PrototypeUpdater(strategy="weighted")
        p = _make_proto(dim=8)
        old = p.embedding.clone()
        new_emb = torch.randn(8)
        updater.update(p, new_emb, weight=0.3)
        expected = 0.7 * old + 0.3 * new_emb
        assert torch.allclose(p.embedding, expected, atol=1e-6)

    def test_adaptive_update(self):
        updater = PrototypeUpdater(strategy="adaptive")
        p = _make_proto(dim=8)
        old_count = p.sample_count
        new_emb = p.embedding.clone() + 0.01
        updater.update(p, new_emb)
        assert p.sample_count == old_count + 1

    def test_batch_update(self):
        updater = PrototypeUpdater(strategy="ema")
        protos = [_make_proto(class_id=0, dim=8), _make_proto(class_id=1, dim=8)]
        embs = torch.randn(6, 8)
        labels = torch.tensor([0, 0, 0, 1, 1, 1])
        updated = updater.batch_update(protos, embs, labels)
        assert len(updated) == 2

    def test_invalid_strategy(self):
        updater = PrototypeUpdater(strategy="unknown")
        with pytest.raises(ValueError):
            updater.update(_make_proto(), _make_emb())

    def test_strategy_property(self):
        u = PrototypeUpdater(strategy="adaptive")
        assert u.strategy == "adaptive"


# ---------------------------------------------------------------------------
# TestPrototypeMatcher
# ---------------------------------------------------------------------------


class TestPrototypeMatcher:
    def test_match_top_k(self):
        repo = PrototypeRepository()
        for p in _make_protos(5):
            repo.store(p)
        matcher = PrototypeMatcher(repo)
        emb = _make_emb()
        results = matcher.match(emb, top_k=3)
        assert len(results) == 3
        for proto, score in results:
            assert isinstance(proto, Prototype)
            assert isinstance(score, float)

    def test_match_empty_repo(self):
        repo = PrototypeRepository()
        matcher = PrototypeMatcher(repo)
        results = matcher.match(_make_emb())
        assert results == []

    def test_match_class_filter(self):
        repo = PrototypeRepository()
        repo.store(_make_proto(class_id=0))
        repo.store(_make_proto(class_id=1))
        matcher = PrototypeMatcher(repo)
        results = matcher.match(_make_emb(), class_filter=[0])
        assert len(results) == 1
        assert results[0][0].class_id == 0

    def test_match_modality_filter(self):
        repo = PrototypeRepository()
        repo.store(_make_proto(modality="image"))
        repo.store(_make_proto(modality="audio"))
        matcher = PrototypeMatcher(repo)
        results = matcher.match(_make_emb(), modality_filter="audio")
        assert len(results) == 1
        assert results[0][0].modality == "audio"

    def test_batch_match(self):
        repo = PrototypeRepository()
        for p in _make_protos(5):
            repo.store(p)
        matcher = PrototypeMatcher(repo)
        embs = torch.randn(3, 16)
        results = matcher.batch_match(embs, top_k=2)
        assert len(results) == 3
        assert len(results[0]) == 2

    def test_nearest_prototype(self):
        repo = PrototypeRepository()
        repo.store(_make_proto())
        matcher = PrototypeMatcher(repo)
        result = matcher.nearest_prototype(_make_emb())
        assert result is not None
        proto, score = result
        assert isinstance(proto, Prototype)

    def test_nearest_prototype_empty(self):
        repo = PrototypeRepository()
        matcher = PrototypeMatcher(repo)
        assert matcher.nearest_prototype(_make_emb()) is None

    def test_rank(self):
        repo = PrototypeRepository()
        for p in _make_protos(4):
            repo.store(p)
        matcher = PrototypeMatcher(repo)
        ranked = matcher.rank(_make_emb())
        assert len(ranked) == 4
        scores = [s for _, s in ranked]
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    def test_match_to_class(self):
        repo = PrototypeRepository()
        repo.store(_make_proto(class_id=0))
        repo.store(_make_proto(class_id=1))
        matcher = PrototypeMatcher(repo)
        results = matcher.match_to_class(_make_emb(), top_k=2)
        assert len(results) == 2
        for cid, score in results:
            assert isinstance(cid, int)
            assert isinstance(score, float)

    def test_default_similarity_engine(self):
        repo = PrototypeRepository()
        matcher = PrototypeMatcher(repo, metric="euclidean")
        assert matcher.metric == "euclidean"

    def test_repository_property(self):
        repo = PrototypeRepository()
        matcher = PrototypeMatcher(repo)
        assert matcher.repository is repo


# ---------------------------------------------------------------------------
# TestSimilarityEngine
# ---------------------------------------------------------------------------


class TestSimilarityEngine:
    def test_cosine_similarity(self):
        eng = SimilarityEngine(metric="cosine")
        a = torch.tensor([1.0, 0.0])
        b = torch.tensor([1.0, 0.0])
        assert eng.similarity(a, b).item() == pytest.approx(1.0, abs=1e-4)

    def test_euclidean_similarity(self):
        eng = SimilarityEngine(metric="euclidean")
        a = torch.zeros(4)
        b = torch.zeros(4)
        assert eng.similarity(a, b).item() == pytest.approx(1.0, abs=1e-4)

    def test_manhattan_similarity(self):
        eng = SimilarityEngine(metric="manhattan")
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([1.0, 2.0])
        s = eng.similarity(a, b)
        assert s.item() == pytest.approx(1.0, abs=1e-4)

    def test_dot_similarity(self):
        eng = SimilarityEngine(metric="dot")
        a = torch.tensor([2.0, 3.0])
        b = torch.tensor([4.0, 5.0])
        s = eng.similarity(a, b)
        assert s.item() == pytest.approx(23.0, abs=1e-4)

    def test_cosine_distance(self):
        eng = SimilarityEngine(metric="cosine")
        a = torch.tensor([1.0, 0.0])
        b = torch.tensor([0.0, 1.0])
        d = eng.distance(a, b)
        assert d.item() == pytest.approx(1.0, abs=1e-4)

    def test_euclidean_distance(self):
        eng = SimilarityEngine(metric="euclidean")
        a = torch.zeros(4)
        b = torch.ones(4) * 3
        d = eng.distance(a, b)
        assert d.item() == pytest.approx(6.0, abs=1e-4)

    def test_manhattan_distance(self):
        eng = SimilarityEngine(metric="manhattan")
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([4.0, 6.0])
        d = eng.distance(a, b)
        assert d.item() == pytest.approx(7.0, abs=1e-4)

    def test_dot_distance(self):
        eng = SimilarityEngine(metric="dot")
        a = torch.tensor([1.0, 0.0])
        b = torch.tensor([1.0, 0.0])
        d = eng.distance(a, b)
        assert d.item() > 0

    def test_batch_similarity(self):
        eng = SimilarityEngine(metric="cosine")
        emb = _make_emb()
        protos = _make_protos(4)
        sims = eng.batch_similarity(emb, protos)
        assert sims.shape == (4,)

    def test_batch_similarity_empty(self):
        eng = SimilarityEngine()
        sims = eng.batch_similarity(_make_emb(), [])
        assert sims.numel() == 0

    def test_batch_similarity_euclidean(self):
        eng = SimilarityEngine(metric="euclidean")
        emb = _make_emb()
        protos = _make_protos(3)
        sims = eng.batch_similarity(emb, protos)
        assert sims.shape == (3,)

    def test_batch_similarity_dot(self):
        eng = SimilarityEngine(metric="dot")
        emb = _make_emb()
        protos = _make_protos(3)
        sims = eng.batch_similarity(emb, protos)
        assert sims.shape == (3,)

    def test_batch_similarity_invalid_metric(self):
        eng = SimilarityEngine(metric="manhattan")
        emb = _make_emb()
        protos = _make_protos(3)
        with pytest.raises(ValueError):
            eng.batch_similarity(emb, protos)

    def test_pairwise_similarity_matrix_cosine(self):
        eng = SimilarityEngine(metric="cosine")
        embs = torch.randn(5, 16)
        mat = eng.pairwise_similarity_matrix(embs)
        assert mat.shape == (5, 5)
        assert torch.allclose(mat, mat.T, atol=1e-6)

    def test_pairwise_similarity_matrix_euclidean(self):
        eng = SimilarityEngine(metric="euclidean")
        embs = torch.randn(5, 16)
        mat = eng.pairwise_similarity_matrix(embs)
        assert mat.shape == (5, 5)

    def test_pairwise_similarity_matrix_dot(self):
        eng = SimilarityEngine(metric="dot")
        embs = torch.randn(5, 16)
        mat = eng.pairwise_similarity_matrix(embs)
        assert mat.shape == (5, 5)

    def test_prototype_similarity_matrix(self):
        eng = SimilarityEngine()
        protos = _make_protos(4)
        mat = eng.prototype_similarity_matrix(protos)
        assert mat.shape == (4, 4)

    def test_prototype_distance_matrix(self):
        eng = SimilarityEngine()
        protos = _make_protos(4)
        mat = eng.prototype_distance_matrix(protos)
        assert mat.shape == (4, 4)

    def test_invalid_metric_similarity(self):
        eng = SimilarityEngine(metric="unknown")
        with pytest.raises(ValueError):
            eng.similarity(_make_emb(), _make_emb())

    def test_invalid_metric_distance(self):
        eng = SimilarityEngine(metric="unknown")
        with pytest.raises(ValueError):
            eng.distance(_make_emb(), _make_emb())

    def test_invalid_metric_pairwise(self):
        eng = SimilarityEngine(metric="manhattan")
        embs = torch.randn(3, 8)
        with pytest.raises(ValueError):
            eng.pairwise_similarity_matrix(embs)

    def test_metric(self):
        eng = SimilarityEngine(metric="dot")
        assert eng.metric == "dot"


# ---------------------------------------------------------------------------
# TestConfidenceEstimator
# ---------------------------------------------------------------------------


class TestConfidenceEstimator:
    def test_estimate_no_embedding(self):
        est = ConfidenceEstimator()
        p = _make_proto()
        c = est.estimate(p)
        assert 0.0 <= c <= 1.0

    def test_estimate_with_embedding(self):
        est = ConfidenceEstimator()
        p = _make_proto()
        emb = p.embedding.clone()
        c = est.estimate(p, embedding=emb)
        assert 0.0 <= c <= 1.0

    def test_estimate_sample_count_factor(self):
        est = ConfidenceEstimator()
        p = _make_proto()
        p.sample_count = 50
        assert est._sample_count_factor(p) == 0.5
        p.sample_count = 200
        assert est._sample_count_factor(p) == 1.0

    def test_estimate_base_confidence_factor(self):
        est = ConfidenceEstimator()
        p = Prototype(embedding=_make_emb(), class_id=0, confidence=0.7)
        assert est._base_confidence_factor(p) == 0.7

    def test_estimate_distance_factor(self):
        est = ConfidenceEstimator()
        p = _make_proto()
        emb = p.embedding.clone()
        d = est._distance_factor(p, emb)
        assert 0.0 <= d <= 1.0

    def test_batch_estimate(self):
        est = ConfidenceEstimator()
        protos = _make_protos(4)
        embs = torch.randn(4, 16)
        confs = est.batch_estimate(protos, embs)
        assert len(confs) == 4
        assert all(0.0 <= c <= 1.0 for c in confs)

    def test_batch_estimate_no_embeddings(self):
        est = ConfidenceEstimator()
        protos = _make_protos(3)
        confs = est.batch_estimate(protos)
        assert len(confs) == 3

    def test_stability_score_high(self):
        est = ConfidenceEstimator()
        history = [{"confidence": 0.9}, {"confidence": 0.9}]
        score = est.stability_score(_make_proto(), history)
        assert score == pytest.approx(1.0, abs=1e-4)

    def test_stability_score_low(self):
        est = ConfidenceEstimator()
        history = [{"confidence": 0.1}, {"confidence": 0.9}]
        score = est.stability_score(_make_proto(), history)
        assert score < 1.0

    def test_stability_score_short_history(self):
        est = ConfidenceEstimator()
        assert est.stability_score(_make_proto(), []) == 1.0
        assert est.stability_score(_make_proto(), [{"confidence": 0.5}]) == 1.0

    def test_normalized_confidence(self):
        est = ConfidenceEstimator()
        assert est.normalized_confidence(0.5, 0.0, 1.0) == 0.5
        assert est.normalized_confidence(0.0, 0.0, 1.0) == 0.0
        assert est.normalized_confidence(1.5, 0.0, 1.0) == 1.0
        assert est.normalized_confidence(0.5, 1.0, 0.0) == 0.0


# ---------------------------------------------------------------------------
# TestPrototypeClustering
# ---------------------------------------------------------------------------


class TestPrototypeClustering:
    def test_kmeans(self):
        clusterer = PrototypeClustering(strategy="kmeans")
        embs = torch.randn(20, 8)
        result = clusterer.cluster(embs, n_clusters=3)
        assert result["strategy"] == "kmeans"
        assert result["n_clusters"] == 3
        assert result["centroids"].shape == (3, 8)
        assert len(result["assignments"]) == 20
        assert isinstance(result["inertia"], float)

    def test_hierarchical(self):
        clusterer = PrototypeClustering(strategy="hierarchical")
        embs = torch.randn(10, 4)
        result = clusterer.cluster(embs, n_clusters=2)
        assert result["strategy"] == "hierarchical"
        assert result["n_clusters"] == 2
        assert result["centroids"].shape[1] == 4

    def test_hierarchical_n_equal_n_clusters(self):
        clusterer = PrototypeClustering(strategy="hierarchical")
        embs = torch.randn(2, 4)
        result = clusterer.cluster(embs, n_clusters=5)
        assert result["n_clusters"] == 2

    def test_dbscan(self):
        clusterer = PrototypeClustering(strategy="dbscan")
        embs = torch.randn(20, 8)
        result = clusterer.cluster(embs, eps=2.0, min_samples=2)
        assert result["strategy"] == "dbscan"

    def test_cluster_prototypes(self):
        clusterer = PrototypeClustering(strategy="kmeans")
        protos = _make_protos(10, dim=8)
        result = clusterer.cluster_prototypes(protos, n_clusters=3)
        assert "prototype_ids" in result
        assert len(result["prototype_ids"]) == 10

    def test_invalid_strategy(self):
        clusterer = PrototypeClustering(strategy="unknown")
        with pytest.raises(ValueError):
            clusterer.cluster(torch.randn(5, 4), n_clusters=2)


# ---------------------------------------------------------------------------
# TestVisualizationSupport
# ---------------------------------------------------------------------------


class TestVisualizationSupport:
    def test_embedding_data(self):
        vis = VisualizationSupport()
        embs = torch.randn(5, 8)
        data = vis.embedding_data(embs, labels=[0, 1, 0, 1, 0])
        assert data["dim"] == 8
        assert data["count"] == 5
        assert data["labels"] == [0, 1, 0, 1, 0]

    def test_embedding_data_no_labels(self):
        vis = VisualizationSupport()
        embs = torch.randn(3, 4)
        data = vis.embedding_data(embs)
        assert "labels" not in data

    def test_prototype_trajectories(self):
        vis = VisualizationSupport()
        history = [
            {"embedding": [0.1, 0.2], "timestamp": 1.0},
            {"embedding": [0.3, 0.4], "timestamp": 2.0},
        ]
        data = vis.prototype_trajectories(history, embedding_dim=2)
        assert data["steps"] == 2
        assert data["dim"] == 2
        assert len(data["trajectory"]) == 2

    def test_similarity_heatmap(self):
        vis = VisualizationSupport()
        protos = _make_protos(4, dim=8)
        data = vis.similarity_heatmap(protos)
        assert data["size"] == 4
        assert len(data["prototype_ids"]) == 4
        assert len(data["class_ids"]) == 4
        assert len(data["matrix"]) == 4

    def test_cluster_plot_data(self):
        vis = VisualizationSupport()
        embs = torch.randn(10, 2)
        assignments = [0, 0, 1, 1, 2, 2, 0, 0, 1, 1]
        data = vis.cluster_plot_data(embs, assignments, centroids=torch.randn(3, 2))
        assert "embeddings" in data
        assert data["assignments"] == assignments
        assert data["n_clusters"] == 3
        assert "centroids" in data

    def test_cluster_plot_data_no_centroids(self):
        vis = VisualizationSupport()
        embs = torch.randn(5, 2)
        data = vis.cluster_plot_data(embs, [0, 0, 1, 1, -1])
        assert "centroids" not in data

    def test_pairwise_distances(self):
        vis = VisualizationSupport()
        protos = _make_protos(3, dim=8)
        data = vis.pairwise_distances(protos)
        assert data["size"] == 3
        assert len(data["prototype_ids"]) == 3
        assert len(data["distance_matrix"]) == 3

    def test_prototype_summary(self):
        vis = VisualizationSupport()
        p = _make_proto(class_id=7)
        summary = vis.prototype_summary(p)
        assert summary["class_id"] == 7


# ---------------------------------------------------------------------------
# TestLosses
# ---------------------------------------------------------------------------


class TestPrototypeCompactnessLoss:
    def test_forward(self):
        loss_fn = PrototypeCompactnessLoss()
        embs = torch.randn(4, 16)
        protos = _make_protos(2, dim=16)
        labels = torch.tensor([0, 0, 1, 1])
        loss = loss_fn(embs, protos, labels)
        assert loss.ndim == 0
        assert loss >= 0

    def test_forward_no_matching_prototypes(self):
        loss_fn = PrototypeCompactnessLoss()
        embs = torch.randn(4, 16)
        protos = [_make_proto(class_id=99, dim=16)]
        labels = torch.zeros(4, dtype=torch.long)
        loss = loss_fn(embs, protos, labels)
        assert loss.ndim == 0


class TestPrototypeSeparationLoss:
    def test_forward(self):
        loss_fn = PrototypeSeparationLoss(margin=1.0)
        embs = torch.randn(4, 16)
        protos = _make_protos(2, dim=16)
        labels = torch.tensor([0, 0, 1, 1])
        loss = loss_fn(embs, protos, labels)
        assert loss.ndim == 0
        assert loss >= 0

    def test_forward_no_other_prototypes(self):
        loss_fn = PrototypeSeparationLoss()
        embs = torch.randn(2, 16)
        protos = [_make_proto(class_id=0, dim=16)]
        labels = torch.zeros(2, dtype=torch.long)
        loss = loss_fn(embs, protos, labels)
        assert loss.ndim == 0


class TestCenterLoss:
    def test_forward(self):
        loss_fn = CenterLoss(num_classes=3, embedding_dim=16)
        embs = torch.randn(6, 16)
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        loss = loss_fn(embs, labels)
        assert loss.ndim == 0
        assert loss >= 0

    def test_centers_parameter(self):
        loss_fn = CenterLoss(num_classes=3, embedding_dim=16)
        assert isinstance(loss_fn.centers, torch.nn.Parameter)
        assert loss_fn.centers.shape == (3, 16)


class TestPrototypeConsistencyLoss:
    def test_forward(self):
        loss_fn = PrototypeConsistencyLoss(temperature=0.5)
        student = torch.randn(4, 16)
        teacher = torch.randn(4, 16)
        loss = loss_fn(student, teacher)
        assert loss.ndim == 0
        assert loss >= 0

    def test_identical_embeddings(self):
        loss_fn = PrototypeConsistencyLoss()
        emb = torch.randn(4, 16)
        loss = loss_fn(emb, emb)
        assert loss.item() == pytest.approx(0.0, abs=1e-4)


class TestPrototypeDiversityLoss:
    def test_forward(self):
        loss_fn = PrototypeDiversityLoss(margin=0.5)
        protos = _make_protos(5, dim=16)
        loss = loss_fn(protos)
        assert loss.ndim == 0
        assert loss >= 0

    def test_fewer_than_two_prototypes(self):
        loss_fn = PrototypeDiversityLoss()
        loss = loss_fn([_make_proto(dim=16)])
        assert loss.item() == 0.0

    def test_no_prototypes(self):
        loss_fn = PrototypeDiversityLoss()
        loss = loss_fn([])
        assert loss.item() == 0.0


# ---------------------------------------------------------------------------
# TestPrototypeMetrics
# ---------------------------------------------------------------------------


class TestPrototypeMetrics:
    def test_intra_class_distance(self):
        protos = [
            Prototype(embedding=torch.zeros(8), class_id=0),
            Prototype(embedding=torch.ones(8), class_id=0),
        ]
        metrics = PrototypeMetrics(protos)
        d = metrics.intra_class_distance(0)
        assert d > 0

    def test_intra_class_distance_single(self):
        metrics = PrototypeMetrics([_make_proto(class_id=0)])
        assert metrics.intra_class_distance(0) == 0.0

    def test_inter_class_distance(self):
        protos = [
            Prototype(embedding=torch.zeros(8), class_id=0),
            Prototype(embedding=torch.ones(8) * 10, class_id=1),
        ]
        metrics = PrototypeMetrics(protos)
        d = metrics.inter_class_distance(0, 1)
        assert d > 0

    def test_inter_class_distance_no_protos(self):
        metrics = PrototypeMetrics([])
        assert metrics.inter_class_distance(0, 1) == 0.0

    def test_prototype_purity(self):
        protos = [
            Prototype(embedding=torch.ones(8), class_id=0),
            Prototype(embedding=torch.ones(8), class_id=0),
        ]
        metrics = PrototypeMetrics(protos)
        p = metrics.prototype_purity(0)
        assert p > 0

    def test_prototype_purity_single(self):
        metrics = PrototypeMetrics([_make_proto(class_id=0)])
        assert metrics.prototype_purity(0) == 1.0

    def test_prototype_purity_no_protos(self):
        metrics = PrototypeMetrics([])
        assert metrics.prototype_purity(0) == 0.0

    def test_prototype_coverage(self):
        protos = _make_protos(6)
        metrics = PrototypeMetrics(protos)
        cov = metrics.prototype_coverage()
        assert len(cov) == 3
        assert all(0 < v <= 1.0 for v in cov.values())

    def test_prototype_coverage_empty(self):
        metrics = PrototypeMetrics([])
        assert metrics.prototype_coverage() == {}

    def test_prototype_variance(self):
        protos = [
            Prototype(embedding=torch.randn(8), class_id=0),
            Prototype(embedding=torch.randn(8), class_id=0),
        ]
        metrics = PrototypeMetrics(protos)
        var = metrics.prototype_variance()
        assert 0 in var

    def test_prototype_variance_single(self):
        metrics = PrototypeMetrics([_make_proto(class_id=0)])
        var = metrics.prototype_variance()
        assert var[0] == 0.0

    def test_prototype_drift(self):
        old = _make_protos(3)
        new = [p.clone() for p in old]
        new[0].embedding = torch.randn(16)
        metrics = PrototypeMetrics(old)
        drift = metrics.prototype_drift(old, new)
        assert drift >= 0

    def test_prototype_drift_no_common(self):
        old = [_make_proto()]
        new = [_make_proto()]
        metrics = PrototypeMetrics(old)
        assert metrics.prototype_drift(old, new) == 0.0

    def test_average_confidence(self):
        protos = [
            Prototype(embedding=_make_emb(), class_id=0, confidence=0.8),
            Prototype(embedding=_make_emb(), class_id=1, confidence=0.6),
        ]
        metrics = PrototypeMetrics(protos)
        assert metrics.average_confidence() == pytest.approx(0.7, abs=1e-6)

    def test_average_confidence_empty(self):
        metrics = PrototypeMetrics([])
        assert metrics.average_confidence() == 0.0

    def test_to_dict(self):
        protos = _make_protos(6)
        metrics = PrototypeMetrics(protos)
        d = metrics.to_dict()
        assert d["num_prototypes"] == 6
        assert d["num_classes"] == 3
        assert "prototype_coverage" in d
        assert "per_class" in d
        assert "average_confidence" in d


# ---------------------------------------------------------------------------
# TestPrototypeFactory
# ---------------------------------------------------------------------------


class TestPrototypeFactory:
    def test_create_prototype(self):
        p = PrototypeFactory.create_prototype(_make_emb(), class_id=5)
        assert isinstance(p, Prototype)
        assert p.class_id == 5

    def test_create_repository(self):
        repo = PrototypeFactory.create_repository()
        assert isinstance(repo, PrototypeRepository)

    def test_create_generator(self):
        gen = PrototypeFactory.create_generator(strategy="centroid")
        assert isinstance(gen, PrototypeGenerator)
        assert gen.strategy == "centroid"

    def test_create_memory(self):
        mem = PrototypeFactory.create_memory(max_global=500, max_local=50)
        assert isinstance(mem, PrototypeMemory)

    def test_create_updater(self):
        updater = PrototypeFactory.create_updater(strategy="ema")
        assert isinstance(updater, PrototypeUpdater)
        assert updater.strategy == "ema"

    def test_create_similarity(self):
        sim = PrototypeFactory.create_similarity(metric="cosine")
        assert isinstance(sim, SimilarityEngine)
        assert sim.metric == "cosine"

    def test_create_confidence(self):
        conf = PrototypeFactory.create_confidence(metric="cosine")
        assert isinstance(conf, ConfidenceEstimator)

    def test_create_clustering(self):
        clusterer = PrototypeFactory.create_clustering(strategy="kmeans")
        assert isinstance(clusterer, PrototypeClustering)

    def test_create_matcher(self):
        repo = PrototypeRepository()
        matcher = PrototypeFactory.create_matcher(repo)
        assert isinstance(matcher, PrototypeMatcher)
        assert matcher.repository is repo

    def test_create_metrics(self):
        protos = _make_protos(3)
        metrics = PrototypeFactory.create_metrics(protos)
        assert isinstance(metrics, PrototypeMetrics)

    def test_create_loss_compactness(self):
        loss = PrototypeFactory.create_loss("compactness")
        assert isinstance(loss, PrototypeCompactnessLoss)

    def test_create_loss_separation(self):
        loss = PrototypeFactory.create_loss("separation")
        assert isinstance(loss, PrototypeSeparationLoss)

    def test_create_loss_center(self):
        loss = PrototypeFactory.create_loss("center", num_classes=3, embedding_dim=16)
        assert isinstance(loss, CenterLoss)

    def test_create_loss_consistency(self):
        loss = PrototypeFactory.create_loss("consistency")
        assert isinstance(loss, PrototypeConsistencyLoss)

    def test_create_loss_diversity(self):
        loss = PrototypeFactory.create_loss("diversity")
        assert isinstance(loss, PrototypeDiversityLoss)

    def test_create_loss_invalid(self):
        with pytest.raises(ValueError):
            PrototypeFactory.create_loss("invalid")

    def test_create_matcher_with_defaults(self):
        matcher = PrototypeFactory.create_matcher_with_defaults()
        assert isinstance(matcher, PrototypeMatcher)
        assert isinstance(matcher.repository, PrototypeRepository)

    def test_default_system(self):
        system = PrototypeFactory.default_system()
        assert "repository" in system
        assert "similarity" in system
        assert "matcher" in system
        assert "confidence" in system
        assert "generator" in system
        assert "updater" in system
        assert "memory" in system

    def test_register_and_get(self):
        PrototypeFactory.register("test_comp", "dummy_value")
        assert PrototypeFactory.get("test_comp") == "dummy_value"

    def test_get_unknown(self):
        with pytest.raises(ValueError):
            PrototypeFactory.get("unknown_component")


# ---------------------------------------------------------------------------
# TestUtils
# ---------------------------------------------------------------------------


class TestValidateEmbedding:
    def test_valid(self):
        validate_embedding(torch.randn(16))

    def test_not_tensor(self):
        with pytest.raises(TypeError):
            validate_embedding([1, 2, 3])

    def test_not_1d(self):
        with pytest.raises(ValueError):
            validate_embedding(torch.randn(2, 16))

    def test_empty(self):
        with pytest.raises(ValueError):
            validate_embedding(torch.tensor([]))


class TestCheckNan:
    def test_no_nan(self):
        check_nan(torch.randn(16), "test")

    def test_with_nan(self):
        with pytest.raises(ValueError):
            check_nan(torch.tensor([float("nan"), 1.0]), "test")

    def test_with_inf(self):
        with pytest.raises(ValueError):
            check_nan(torch.tensor([float("inf"), 1.0]), "test")


class TestValidateClassId:
    def test_valid(self):
        validate_class_id(0)
        validate_class_id(5, num_classes=10)

    def test_not_int(self):
        with pytest.raises(TypeError):
            validate_class_id(1.0)

    def test_negative(self):
        with pytest.raises(ValueError):
            validate_class_id(-1)

    def test_out_of_range(self):
        with pytest.raises(ValueError):
            validate_class_id(10, num_classes=10)


class TestValidatePrototypeList:
    def test_valid(self):
        validate_prototype_list(_make_protos(3))

    def test_invalid_item(self):
        with pytest.raises(TypeError):
            validate_prototype_list([_make_proto(), "not_a_prototype"])


class TestValidateSimilarityMetric:
    def test_valid(self):
        validate_similarity_metric("cosine")

    def test_invalid(self):
        with pytest.raises(ValueError):
            validate_similarity_metric("invalid")


class TestMatrixUtils:
    def test_cosine_similarity_matrix(self):
        a = torch.randn(4, 8)
        b = torch.randn(3, 8)
        mat = cosine_similarity_matrix(a, b)
        assert mat.shape == (4, 3)

    def test_euclidean_distance_matrix(self):
        a = torch.randn(4, 8)
        b = torch.randn(3, 8)
        mat = euclidean_distance_matrix(a, b)
        assert mat.shape == (4, 3)


class TestTimer:
    def test_timer_elapsed(self):
        timer = Timer("test", log_on_exit=False)
        with timer:
            pass
        assert timer.elapsed >= 0

    def test_timer_no_start(self):
        timer = Timer()
        assert timer.elapsed == 0.0

    def test_timer_context_manager(self):
        with Timer("ctx", log_on_exit=False) as t:
            pass
        assert t.elapsed >= 0
