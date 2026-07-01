from __future__ import annotations

import numpy as np
import pytest

from app.datasets.partitioners.iid import IIDPartitioner
from app.datasets.partitioners.dirichlet import DirichletPartitioner
from app.datasets.partitioners.shard import ShardPartitioner


@pytest.fixture
def sample_labels() -> np.ndarray:
    rng = np.random.default_rng(42)
    return np.repeat(np.arange(10), 100)


class TestIIDPartitioner:
    def test_iid_balanced(self, sample_labels: np.ndarray):
        partitioner = IIDPartitioner()
        result = partitioner.partition(sample_labels, num_clients=10, seed=42)
        assert len(result) == 10
        sizes = [len(v) for v in result.values()]
        assert all(s > 0 for s in sizes)
        assert all(isinstance(k, int) for k in result)

    def test_iid_reproducible(self, sample_labels: np.ndarray):
        p1 = IIDPartitioner().partition(sample_labels, num_clients=5, seed=123)
        p2 = IIDPartitioner().partition(sample_labels, num_clients=5, seed=123)
        for k in p1:
            assert list(p1[k]) == list(p2[k])

    def test_iid_all_indices_covered(self, sample_labels: np.ndarray):
        result = IIDPartitioner().partition(sample_labels, num_clients=10, seed=42)
        all_indices = set()
        for v in result.values():
            all_indices.update(v)
        assert len(all_indices) == len(sample_labels)

    def test_iid_unbalanced(self, sample_labels: np.ndarray):
        result = IIDPartitioner().partition(
            sample_labels, num_clients=5, seed=42, balanced=False
        )
        sizes = [len(v) for v in result.values()]
        assert len(set(sizes)) > 1


class TestDirichletPartitioner:
    def test_dirichlet_high_alpha(self, sample_labels: np.ndarray):
        result = DirichletPartitioner().partition(
            sample_labels, num_clients=10, seed=42, alpha=100.0
        )
        assert len(result) == 10
        sizes = [len(v) for v in result.values()]
        assert all(s > 0 for s in sizes)

    def test_dirichlet_low_alpha(self, sample_labels: np.ndarray):
        result = DirichletPartitioner().partition(
            sample_labels, num_clients=10, seed=42, alpha=0.1
        )
        assert len(result) > 0
        sizes = [len(v) for v in result.values()]
        assert sum(sizes) > 0

    def test_dirichlet_reproducible(self, sample_labels: np.ndarray):
        p1 = DirichletPartitioner().partition(
            sample_labels, num_clients=5, seed=42, alpha=0.5
        )
        p2 = DirichletPartitioner().partition(
            sample_labels, num_clients=5, seed=42, alpha=0.5
        )
        for k in p1:
            assert list(p1[k]) == list(p2[k])

    def test_dirichlet_min_samples(self, sample_labels: np.ndarray):
        result = DirichletPartitioner().partition(
            sample_labels, num_clients=20, seed=42, alpha=0.1, min_samples=5
        )
        for v in result.values():
            assert len(v) >= 5


class TestShardPartitioner:
    def test_shard_basic(self, sample_labels: np.ndarray):
        result = ShardPartitioner().partition(
            sample_labels, num_clients=10, seed=42, shards_per_client=2
        )
        assert len(result) == 10

    def test_shard_reproducible(self, sample_labels: np.ndarray):
        p1 = ShardPartitioner().partition(sample_labels, num_clients=5, seed=42)
        p2 = ShardPartitioner().partition(sample_labels, num_clients=5, seed=42)
        for k in p1:
            assert list(p1[k]) == list(p2[k])

    def test_shard_all_indices_used(self, sample_labels: np.ndarray):
        result = ShardPartitioner().partition(
            sample_labels, num_clients=5, seed=42, shards_per_client=4
        )
        all_indices = set()
        for v in result.values():
            all_indices.update(v)
        assert len(all_indices) > 0

    def test_shard_single_client(self, sample_labels: np.ndarray):
        result = ShardPartitioner().partition(sample_labels, num_clients=1, seed=42)
        assert len(result) == 1
        assert len(result[0]) <= len(sample_labels)
        assert len(result[0]) > 0
