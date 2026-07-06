from __future__ import annotations

import json
import os
import tempfile
import time
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from app.evaluation import (
    AblationRegistry,
    AblationStudy,
    Baseline,
    BaselineFactory,
    BaselineRegistry,
    Benchmark,
    ClassificationMetrics,
    CommunicationMetrics,
    EvaluationEngine,
    EvaluationFactory,
    ExperimentRegistry,
    Exporter,
    KnowledgeTransferMetrics,
    Leaderboard,
    MetricFactory,
    MetricRegistry,
    PersonalizationMetrics,
    PrototypeMetrics,
    ReportGenerator,
    StatisticalAnalysis,
    TrainingMetrics,
    VisualizationDataGenerator,
    without_adaptive_weighting,
    without_aggregation,
    without_knowledge_transfer,
    without_personalization,
    without_prototype_memory,
    without_prototypes,
)
from app.evaluation.experiment_runner import ExperimentRunner


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def simple_model():
    return nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))


@pytest.fixture
def simple_dataloader():
    data = torch.randn(50, 10)
    labels = torch.randint(0, 5, (50,))
    dataset = TensorDataset(data, labels)
    return DataLoader(dataset, batch_size=10)


@pytest.fixture
def sample_outputs_targets():
    outputs = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
    targets = torch.tensor([1, 0, 1, 0])
    return outputs, targets


@pytest.fixture
def sample_config():
    return {
        "rounds": 5,
        "clients": {"num_clients": 2},
        "model": {"type": "simple"},
        "dataset": {"name": "generic", "batch_size": 16},
        "modalities": {"image": 64, "text": 64},
        "mappings": [("image", "text")],
        "num_classes": 5,
        "optimizer": {"type": "adam", "lr": 1e-3},
    }


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp


# ============================================================
# Test: Registry
# ============================================================


class TestMetricRegistry:
    def test_register_and_get(self):
        MetricRegistry.register("test_metric", lambda x: x + 1)
        fn = MetricRegistry.get("test_metric")
        assert fn(5) == 6
        MetricRegistry.unregister("test_metric")

    def test_get_unknown(self):
        with pytest.raises(ValueError):
            MetricRegistry.get("nonexistent")

    def test_list(self):
        assert "accuracy" in MetricRegistry.list()

    def test_contains(self):
        assert MetricRegistry.contains("accuracy")
        assert not MetricRegistry.contains("nonexistent")

    def test_unregister(self):
        MetricRegistry.register("temp_metric", lambda x: x)
        assert MetricRegistry.contains("temp_metric")
        MetricRegistry.unregister("temp_metric")
        assert not MetricRegistry.contains("temp_metric")

    def test_clear(self):
        MetricRegistry.register("temp_clear_metric", lambda x: x)
        assert MetricRegistry.contains("temp_clear_metric")
        MetricRegistry.unregister("temp_clear_metric")
        assert not MetricRegistry.contains("temp_clear_metric")

    def test_clear_removes_all(self):
        MetricRegistry.register("tmp_clear_test", lambda x: x)
        assert MetricRegistry.contains("tmp_clear_test")
        MetricRegistry.unregister("tmp_clear_test")
        assert not MetricRegistry.contains("tmp_clear_test")


class TestBaselineRegistry:
    def test_register_and_get(self):
        cls = BaselineRegistry.get("fedavg")
        assert issubclass(cls, Baseline)

    def test_get_unknown(self):
        with pytest.raises(ValueError):
            BaselineRegistry.get("nonexistent")

    def test_list(self):
        assert "fedavg" in BaselineRegistry.list()

    def test_contains(self):
        assert BaselineRegistry.contains("fedavg")
        assert not BaselineRegistry.contains("nonexistent")

    def test_unregister(self):
        BaselineRegistry.register("temp_base", FedAvgBaselineStub)
        assert BaselineRegistry.contains("temp_base")
        BaselineRegistry.unregister("temp_base")
        assert not BaselineRegistry.contains("temp_base")

    def test_clear(self):
        BaselineRegistry.register("temp_clear_base", FedAvgBaselineStub)
        assert BaselineRegistry.contains("temp_clear_base")
        BaselineRegistry.unregister("temp_clear_base")
        assert not BaselineRegistry.contains("temp_clear_base")


class FedAvgBaselineStub(Baseline):
    def name(self) -> str:
        return "stub"

    def train_round(self, round_id, clients, server_model, dataloaders):
        return {"round_id": round_id, "accuracy": 0.0}


class TestAblationRegistry:
    def test_register_and_get(self):
        fn = AblationRegistry.get("without_prototypes")
        assert callable(fn)

    def test_get_unknown(self):
        with pytest.raises(ValueError):
            AblationRegistry.get("nonexistent")

    def test_list(self):
        assert "without_prototypes" in AblationRegistry.list()

    def test_contains(self):
        assert AblationRegistry.contains("without_prototypes")

    def test_clear(self):
        AblationRegistry.register("temp_clear_ablation", lambda c: c)
        assert AblationRegistry.contains("temp_clear_ablation")
        AblationRegistry.unregister("temp_clear_ablation")
        assert not AblationRegistry.contains("temp_clear_ablation")


class TestExperimentRegistry:
    def test_register_and_get(self):
        ExperimentRegistry.register("test_exp", {"key": "value"})
        cfg = ExperimentRegistry.get("test_exp")
        assert cfg["key"] == "value"
        ExperimentRegistry.unregister("test_exp")

    def test_get_unknown(self):
        with pytest.raises(ValueError):
            ExperimentRegistry.get("nonexistent")

    def test_list(self):
        ExperimentRegistry.register("test_exp_a", {})
        assert "test_exp_a" in ExperimentRegistry.list()
        ExperimentRegistry.unregister("test_exp_a")

    def test_contains(self):
        ExperimentRegistry.register("test_exp_b", {})
        assert ExperimentRegistry.contains("test_exp_b")
        ExperimentRegistry.unregister("test_exp_b")

    def test_count(self):
        ExperimentRegistry.register("test_exp_c", {})
        ExperimentRegistry.register("test_exp_d", {})
        assert ExperimentRegistry.count() >= 2
        ExperimentRegistry.unregister("test_exp_c")
        ExperimentRegistry.unregister("test_exp_d")

    def test_clear(self):
        ExperimentRegistry.register("test_exp_e", {})
        ExperimentRegistry.clear()
        assert ExperimentRegistry.count() == 0


# ============================================================
# Test: Factory
# ============================================================


class TestEvaluationFactory:
    def test_create_engine(self):
        engine = EvaluationFactory.create_engine({"rounds": 10})
        assert isinstance(engine, EvaluationEngine)

    def test_create_metric(self):
        fn = EvaluationFactory.create_metric("accuracy")
        assert callable(fn)

    def test_create_baseline(self):
        base = EvaluationFactory.create_baseline("fedavg", {})
        assert isinstance(base, Baseline)
        assert base.name() == "fedavg"

    def test_list_available_metrics(self):
        metrics = EvaluationFactory.list_available_metrics()
        assert "accuracy" in metrics

    def test_list_available_baselines(self):
        baselines = EvaluationFactory.list_available_baselines()
        assert "fedavg" in baselines

    def test_list_available_ablations(self):
        ablations = EvaluationFactory.list_available_ablations()
        assert "without_prototypes" in ablations

    def test_from_config(self):
        result = EvaluationFactory.from_config({"metrics": ["accuracy"]})
        assert "engine" in result


class TestMetricFactory:
    def test_create_accuracy(self):
        fn = MetricFactory.create("accuracy")
        assert callable(fn)

    def test_compute_accuracy(self):
        outputs = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
        targets = torch.tensor([1, 0])
        acc = MetricFactory.compute("accuracy", outputs, targets)
        assert acc == 1.0

    def test_create_unknown(self):
        with pytest.raises(ValueError):
            MetricFactory.create("nonexistent")


# ============================================================
# Test: ClassificationMetrics
# ============================================================


class TestClassificationMetrics:
    def test_accuracy_perfect(self, sample_outputs_targets):
        outputs, targets = sample_outputs_targets
        acc = ClassificationMetrics.accuracy(outputs, targets)
        assert acc == 1.0

    def test_accuracy_half(self):
        outputs = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
        targets = torch.tensor([0, 0])
        acc = ClassificationMetrics.accuracy(outputs, targets)
        assert acc == 0.5

    def test_accuracy_numpy(self):
        outputs = np.array([[0.1, 0.9], [0.8, 0.2]])
        targets = np.array([1, 0])
        acc = ClassificationMetrics.accuracy(outputs, targets)
        assert acc == 1.0

    def test_precision_macro(self, sample_outputs_targets):
        outputs, targets = sample_outputs_targets
        prec = ClassificationMetrics.precision(outputs, targets)
        assert prec == 1.0

    def test_precision_micro(self, sample_outputs_targets):
        outputs, targets = sample_outputs_targets
        prec = ClassificationMetrics.precision(outputs, targets, average="micro")
        assert prec == 1.0

    def test_precision_weighted(self, sample_outputs_targets):
        outputs, targets = sample_outputs_targets
        prec = ClassificationMetrics.precision(outputs, targets, average="weighted")
        assert prec == 1.0

    def test_recall_macro(self, sample_outputs_targets):
        outputs, targets = sample_outputs_targets
        rec = ClassificationMetrics.recall(outputs, targets)
        assert rec == 1.0

    def test_recall_micro(self, sample_outputs_targets):
        outputs, targets = sample_outputs_targets
        rec = ClassificationMetrics.recall(outputs, targets, average="micro")
        assert rec == 1.0

    def test_recall_weighted(self, sample_outputs_targets):
        outputs, targets = sample_outputs_targets
        rec = ClassificationMetrics.recall(outputs, targets, average="weighted")
        assert rec == 1.0

    def test_f1_macro(self, sample_outputs_targets):
        outputs, targets = sample_outputs_targets
        f1 = ClassificationMetrics.f1_score(outputs, targets)
        assert f1 == 1.0

    def test_f1_micro(self, sample_outputs_targets):
        outputs, targets = sample_outputs_targets
        f1 = ClassificationMetrics.f1_score(outputs, targets, average="micro")
        assert f1 == 1.0

    def test_f1_weighted(self, sample_outputs_targets):
        outputs, targets = sample_outputs_targets
        f1 = ClassificationMetrics.f1_score(outputs, targets, average="weighted")
        assert f1 == 1.0

    def test_f1_numpy(self):
        outputs = np.array([[0.1, 0.9], [0.8, 0.2]])
        targets = np.array([1, 0])
        f1 = ClassificationMetrics.f1_score(outputs, targets)
        assert f1 == 1.0

    def test_f1_with_errors(self):
        outputs = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
        targets = torch.tensor([0, 0])
        f1 = ClassificationMetrics.f1_score(outputs, targets)
        assert f1 < 1.0

    def test_balanced_accuracy(self):
        outputs = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
        targets = torch.tensor([1, 0])
        ba = ClassificationMetrics.balanced_accuracy(outputs, targets)
        assert ba == 1.0

    def test_confusion_matrix(self):
        outputs = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
        targets = torch.tensor([1, 0])
        cm = ClassificationMetrics.confusion_matrix(outputs, targets)
        assert cm.shape == (2, 2)
        assert cm[0, 0] == 1
        assert cm[1, 1] == 1

    def test_confusion_matrix_with_num_classes(self):
        outputs = torch.tensor([[0.1, 0.9, 0.0], [0.8, 0.2, 0.0]])
        targets = torch.tensor([1, 0])
        cm = ClassificationMetrics.confusion_matrix(outputs, targets, num_classes=3)
        assert cm.shape == (3, 3)

    def test_roc_auc_binary(self):
        outputs = torch.tensor([[0.1, 0.9], [0.9, 0.1], [0.2, 0.8], [0.8, 0.2]])
        targets = torch.tensor([1, 0, 1, 0])
        auc = ClassificationMetrics.roc_auc(outputs, targets)
        assert auc >= 0.5

    def test_roc_auc_multiclass(self):
        outputs = torch.tensor(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
            ]
        )
        targets = torch.tensor([0, 1, 2])
        auc = ClassificationMetrics.roc_auc(outputs, targets)
        assert auc >= 0.5

    def test_compute_all(self, sample_outputs_targets):
        outputs, targets = sample_outputs_targets
        metrics = ClassificationMetrics.compute_all(outputs, targets)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "macro_f1" in metrics
        assert "micro_f1" in metrics
        assert "balanced_accuracy" in metrics

    def test_empty_outputs(self):
        outputs = torch.zeros(0, 2)
        targets = torch.zeros(0, dtype=torch.long)
        acc = ClassificationMetrics.accuracy(outputs, targets)
        assert acc == 0.0

    def test_single_class_f1(self):
        outputs = torch.tensor([[0.1, 0.9], [0.2, 0.8]])
        targets = torch.tensor([1, 1])
        f1 = ClassificationMetrics.f1_score(outputs, targets)
        assert f1 == 0.5


# ============================================================
# Test: CommunicationMetrics
# ============================================================


class TestCommunicationMetrics:
    def test_communication_cost(self):
        cost = CommunicationMetrics.communication_cost(
            bytes_sent=100, bytes_received=50
        )
        assert cost == 150.0

    def test_bandwidth(self):
        bw = CommunicationMetrics.bandwidth(bytes_sent=1000, duration_seconds=2.0)
        assert bw == 500.0

    def test_bandwidth_zero_duration(self):
        bw = CommunicationMetrics.bandwidth(bytes_sent=1000, duration_seconds=0)
        assert bw == 0.0

    def test_latency(self):
        lat = CommunicationMetrics.latency(send_time=1.0, receive_time=2.0)
        assert lat == 1.0

    def test_latency_negative(self):
        lat = CommunicationMetrics.latency(send_time=5.0, receive_time=2.0)
        assert lat == 0.0

    def test_bytes_transferred(self):
        messages = [{"payload": "hello"}, {"payload": "world"}]
        bt = CommunicationMetrics.bytes_transferred(messages)
        assert bt > 0

    def test_bytes_transferred_empty(self):
        bt = CommunicationMetrics.bytes_transferred(None)
        assert bt == 0

    def test_compute_all(self):
        metrics = CommunicationMetrics.compute_all(
            bytes_sent=100, bytes_received=50, duration_seconds=2.0
        )
        assert "communication_cost" in metrics
        assert "bandwidth" in metrics
        assert "bytes_transferred" in metrics


# ============================================================
# Test: TrainingMetrics
# ============================================================


class TestTrainingMetrics:
    def test_training_time(self):
        tt = TrainingMetrics.training_time(start_time=1.0, end_time=3.0)
        assert tt == 2.0

    def test_training_time_negative(self):
        tt = TrainingMetrics.training_time(start_time=5.0, end_time=3.0)
        assert tt == 0.0

    def test_inference_time(self, simple_model):
        inputs = torch.randn(1, 10)
        it = TrainingMetrics.inference_time(simple_model, inputs, repetitions=5)
        assert it >= 0

    def test_compute_all(self):
        metrics = TrainingMetrics.compute_all(start_time=1.0, end_time=3.0)
        assert "training_time" in metrics
        assert metrics["training_time"] == 2.0


# ============================================================
# Test: PrototypeMetrics
# ============================================================


class TestEvalPrototypeMetrics:
    def test_prototype_drift(self):
        old = [torch.randn(10) for _ in range(5)]
        new = [torch.randn(10) for _ in range(5)]
        drift = PrototypeMetrics.prototype_drift(old, new)
        assert isinstance(drift, float)

    def test_prototype_drift_empty_old(self):
        drift = PrototypeMetrics.prototype_drift([], [torch.randn(10)])
        assert drift == 0.0

    def test_prototype_drift_tensors(self):
        old = torch.randn(5, 10)
        new = torch.randn(5, 10)
        drift = PrototypeMetrics.prototype_drift(old, new)
        assert isinstance(drift, float)

    def test_prototype_diversity(self):
        embs = [torch.randn(10) for _ in range(10)]
        div = PrototypeMetrics.prototype_diversity(embs)
        assert 0.0 <= div <= 1.0

    def test_prototype_diversity_single(self):
        div = PrototypeMetrics.prototype_diversity([torch.randn(10)])
        assert div == 0.0

    def test_prototype_diversity_tensor(self):
        embs = torch.randn(10, 10)
        div = PrototypeMetrics.prototype_diversity(embs)
        assert 0.0 <= div <= 1.0

    def test_prototype_stability(self):
        stable = PrototypeMetrics.prototype_stability([0.5, 0.5, 0.5])
        assert stable == 1.0

    def test_prototype_stability_single(self):
        stable = PrototypeMetrics.prototype_stability([0.5])
        assert stable == 1.0

    def test_prototype_similarity(self):
        e1 = torch.randn(10)
        e2 = torch.randn(10)
        sim = PrototypeMetrics.prototype_similarity(e1, e2)
        assert -1.0 <= sim <= 1.0

    def test_prototype_compactness(self):
        embs = [torch.randn(10) for _ in range(10)]
        comp = PrototypeMetrics.prototype_compactness(embs)
        assert comp >= 0

    def test_prototype_compactness_empty(self):
        comp = PrototypeMetrics.prototype_compactness([])
        assert comp == 0.0

    def test_prototype_compactness_with_centers(self):
        embs = torch.randn(10, 10)
        centers = torch.randn(3, 10)
        comp = PrototypeMetrics.prototype_compactness(embs, centers)
        assert comp >= 0

    def test_compute_all(self):
        old = [torch.randn(10) for _ in range(5)]
        new = [torch.randn(10) for _ in range(5)]
        all_embs = [torch.randn(10) for _ in range(10)]
        metrics = PrototypeMetrics.compute_all(
            old_embeddings=old,
            new_embeddings=new,
            all_embeddings=all_embs,
            history=[0.5, 0.6],
        )
        assert "prototype_drift" in metrics
        assert "prototype_diversity" in metrics
        assert "prototype_stability" in metrics

    def test_compute_all_empty(self):
        metrics = PrototypeMetrics.compute_all()
        assert metrics == {}


# ============================================================
# Test: KnowledgeTransferMetrics
# ============================================================


class TestKnowledgeTransferMetrics:
    def test_alignment_score(self):
        src = torch.randn(10, 10)
        tgt = torch.randn(10, 10)
        score = KnowledgeTransferMetrics.alignment_score(src, tgt)
        assert -1.0 <= score <= 1.0

    def test_transfer_accuracy(self):
        pred = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
        tgt = torch.tensor([1, 0])
        acc = KnowledgeTransferMetrics.transfer_accuracy(pred, tgt)
        assert acc == 1.0

    def test_transfer_success_rate(self):
        rate = KnowledgeTransferMetrics.transfer_success_rate(8, 10)
        assert rate == 0.8

    def test_transfer_success_rate_zero(self):
        rate = KnowledgeTransferMetrics.transfer_success_rate(0, 0)
        assert rate == 0.0

    def test_cross_modal_similarity(self):
        a = torch.randn(10, 10)
        b = torch.randn(10, 10)
        sim = KnowledgeTransferMetrics.cross_modal_similarity(a, b)
        assert -1.0 <= sim <= 1.0

    def test_cross_modal_similarity_mismatched_sizes(self):
        a = torch.randn(10, 10)
        b = torch.randn(5, 10)
        sim = KnowledgeTransferMetrics.cross_modal_similarity(a, b)
        assert -1.0 <= sim <= 1.0

    def test_compute_all(self):
        src = torch.randn(10, 10)
        tgt = torch.randn(10, 10)
        pred = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
        targets = torch.tensor([1, 0])
        metrics = KnowledgeTransferMetrics.compute_all(
            source_embeddings=src,
            target_embeddings=tgt,
            predicted=pred,
            targets=targets,
            valid_transfers=8,
            total_attempts=10,
            mod_a_embeddings=src,
            mod_b_embeddings=tgt,
        )
        assert "alignment_score" in metrics
        assert "transfer_accuracy" in metrics
        assert "transfer_success_rate" in metrics
        assert "cross_modal_similarity" in metrics

    def test_compute_all_partial(self):
        metrics = KnowledgeTransferMetrics.compute_all(
            valid_transfers=5, total_attempts=10
        )
        assert "transfer_success_rate" in metrics


# ============================================================
# Test: PersonalizationMetrics
# ============================================================


class TestEvalPersonalizationMetrics:
    def test_personalization_gain(self):
        personalized = torch.randn(10)
        global_p = torch.randn(10)
        gain = PersonalizationMetrics.personalization_gain(personalized, global_p)
        assert isinstance(gain, float)

    def test_client_adaptation_score(self):
        pre = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
        post = torch.tensor([[0.1, 0.9], [0.2, 0.8]])
        targets = torch.tensor([1, 0])
        score = PersonalizationMetrics.client_adaptation_score(pre, post, targets)
        assert score >= 0.0

    def test_prototype_fusion_quality(self):
        fw = {"local": 0.5, "global": 0.3, "cross_modal": 0.2}
        quality = PersonalizationMetrics.prototype_fusion_quality(fw)
        assert 0.0 <= quality <= 1.0

    def test_prototype_fusion_quality_empty(self):
        quality = PersonalizationMetrics.prototype_fusion_quality({})
        assert quality == 0.0

    def test_prototype_fusion_quality_single(self):
        quality = PersonalizationMetrics.prototype_fusion_quality({"local": 1.0})
        assert quality == 1.0

    def test_confidence_calibration(self):
        confs = [0.9, 0.8, 0.7, 0.6]
        accs = [1.0, 0.8, 0.6, 0.5]
        ece = PersonalizationMetrics.confidence_calibration(confs, accs)
        assert ece >= 0.0

    def test_confidence_calibration_empty(self):
        ece = PersonalizationMetrics.confidence_calibration([], [])
        assert ece == 0.0

    def test_confidence_calibration_mismatched(self):
        ece = PersonalizationMetrics.confidence_calibration([0.9, 0.8], [1.0])
        assert ece >= 0.0

    def test_compute_all(self):
        personalized = torch.randn(10)
        global_p = torch.randn(10)
        pre = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
        post = torch.tensor([[0.1, 0.9], [0.2, 0.8]])
        targets = torch.tensor([1, 0])
        metrics = PersonalizationMetrics.compute_all(
            personalized=personalized,
            global_prototype=global_p,
            pre_adaptation=pre,
            post_adaptation=post,
            targets=targets,
            fusion_weights={"local": 0.5, "global": 0.5},
            confidences=[0.9, 0.8],
            accuracies=[1.0, 0.8],
        )
        assert "personalization_gain" in metrics
        assert "client_adaptation_score" in metrics
        assert "prototype_fusion_quality" in metrics
        assert "confidence_calibration" in metrics

    def test_compute_all_empty(self):
        metrics = PersonalizationMetrics.compute_all()
        assert metrics == {}


# ============================================================
# Test: Baselines
# ============================================================


class TestFedAvgBaseline:
    def test_name(self):
        base = BaselineFactory.create("fedavg", {})
        assert base.name() == "fedavg"

    def test_train_round(self, simple_model, simple_dataloader):
        base = BaselineFactory.create("fedavg", {})
        clients = [{"lr": 0.01}, {"lr": 0.01}]
        loaders = [simple_dataloader, simple_dataloader]
        metric = base.train_round(1, clients, simple_model, loaders)
        assert "round_id" in metric
        assert "accuracy" in metric
        assert metric["round_id"] == 1

    def test_reset(self, simple_model, simple_dataloader):
        base = BaselineFactory.create("fedavg", {})
        clients = [{"lr": 0.01}]
        base.train_round(1, clients, simple_model, [simple_dataloader])
        assert len(base.round_metrics) == 1
        base.reset()
        assert len(base.round_metrics) == 0


class TestFedProxBaseline:
    def test_name(self):
        base = BaselineFactory.create("fedprox", {"fedprox_mu": 0.01})
        assert base.name() == "fedprox"

    def test_train_round(self, simple_model, simple_dataloader):
        base = BaselineFactory.create("fedprox", {"fedprox_mu": 0.01})
        clients = [{"lr": 0.01}]
        metric = base.train_round(1, clients, simple_model, [simple_dataloader])
        assert "round_id" in metric
        assert metric["round_id"] == 1


class TestSCAFFOLDBaseline:
    def test_name(self):
        base = BaselineFactory.create("scaffold", {})
        assert base.name() == "scaffold"

    def test_train_round(self, simple_model, simple_dataloader):
        base = BaselineFactory.create("scaffold", {})
        clients = [{"lr": 0.01}]
        metric = base.train_round(1, clients, simple_model, [simple_dataloader])
        assert "round_id" in metric

    def test_reset(self):
        base = BaselineFactory.create("scaffold", {})
        base.reset()
        assert base._server_control is None


class TestOtherBaselines:
    def test_prototype_only(self):
        base = BaselineFactory.create("prototype_only", {})
        assert base.name() == "prototype_only"
        metric = base.train_round(1, [], None, [])
        assert metric["accuracy"] == 0.0

    def test_without_personalization(self):
        base = BaselineFactory.create("without_personalization", {})
        assert base.name() == "without_personalization"
        metric = base.train_round(1, [], None, [])
        assert metric["accuracy"] == 0.0

    def test_without_knowledge_transfer(self):
        base = BaselineFactory.create("without_knowledge_transfer", {})
        assert base.name() == "without_knowledge_transfer"
        metric = base.train_round(1, [], None, [])
        assert metric["accuracy"] == 0.0

    def test_without_adaptive_aggregation(self):
        base = BaselineFactory.create("without_adaptive_aggregation", {})
        assert base.name() == "without_adaptive_aggregation"
        metric = base.train_round(1, [], None, [])
        assert metric["accuracy"] == 0.0

    def test_full_pp_mfl(self):
        base = BaselineFactory.create("full_pp_mfl", {})
        assert base.name() == "full_pp_mfl"
        metric = base.train_round(1, [], None, [])
        assert metric["accuracy"] == 0.0

    def test_baseline_factory_create_unknown(self):
        with pytest.raises(ValueError):
            BaselineFactory.create("nonexistent", {})

    def test_baseline_factory_list(self):
        available = BaselineFactory.list_available()
        assert "fedavg" in available
        assert "fedprox" in available


# ============================================================
# Test: Ablation
# ============================================================


class TestAblationFunctions:
    def test_without_prototypes(self):
        config = {"key": "value"}
        result = without_prototypes(config)
        assert result["ablation"]["disable_prototypes"] is True
        assert result["key"] == "value"

    def test_without_aggregation(self):
        result = without_aggregation({})
        assert result["ablation"]["disable_aggregation"] is True

    def test_without_knowledge_transfer(self):
        result = without_knowledge_transfer({})
        assert result["ablation"]["disable_knowledge_transfer"] is True

    def test_without_personalization(self):
        result = without_personalization({})
        assert result["ablation"]["disable_personalization"] is True

    def test_without_adaptive_weighting(self):
        result = without_adaptive_weighting({})
        assert result["ablation"]["disable_adaptive_weighting"] is True

    def test_without_prototype_memory(self):
        result = without_prototype_memory({})
        assert result["ablation"]["disable_prototype_memory"] is True


class TestAblationStudy:
    def test_initialization(self):
        study = AblationStudy({"rounds": 10, "clients": {"num_clients": 2}})
        assert study._base_config["rounds"] == 10

    def test_modify_config(self):
        study = AblationStudy({})
        modified = study._modify_config({"key": "val"}, "without_prototypes")
        assert modified["ablation"]["disable_prototypes"] is True

    def test_get_default_ablations(self):
        study = AblationStudy({})
        ablations = study._get_default_ablations()
        assert "without_prototypes" in ablations
        assert "without_aggregation" in ablations
        assert "without_knowledge_transfer" in ablations
        assert "without_personalization" in ablations

    def test_run_with_mock_runner(self):
        study = AblationStudy({"key": "val"})
        mock_runner = MagicMock()
        mock_runner.run_single.return_value = {"accuracy": 0.9, "loss": 0.1}
        results = study.run(mock_runner, ablations=["without_prototypes"])
        assert "full" in results
        assert "without_prototypes" in results

    def test_comparison_table(self):
        study = AblationStudy({})
        study._results = {
            "full": {"accuracy": 0.9, "loss": 0.1},
            "without_prototypes": {"accuracy": 0.7, "loss": 0.3},
        }
        table = study.comparison_table()
        assert "full" in table
        assert "without_prototypes" in table

    def test_summary(self):
        study = AblationStudy({})
        study._results = {
            "full": {"accuracy": 0.9},
            "without_prototypes": {"accuracy": 0.7},
        }
        summary = study.summary()
        assert summary["num_ablations"] == 2
        assert "full_metrics" in summary

    def test_to_dict(self):
        study = AblationStudy({})
        study._results = {"full": {"accuracy": 0.9}}
        d = study.to_dict()
        assert "results" in d
        assert "comparison" in d
        assert "summary" in d


# ============================================================
# Test: StatisticalAnalysis
# ============================================================


class TestStatisticalAnalysis:
    def test_mean(self):
        assert StatisticalAnalysis.mean([1, 2, 3, 4, 5]) == 3.0

    def test_mean_empty(self):
        assert StatisticalAnalysis.mean([]) == 0.0

    def test_median(self):
        assert StatisticalAnalysis.median([1, 2, 3, 4, 5]) == 3.0

    def test_median_empty(self):
        assert StatisticalAnalysis.median([]) == 0.0

    def test_variance(self):
        var = StatisticalAnalysis.variance([1, 2, 3, 4, 5])
        assert var > 0

    def test_variance_single(self):
        assert StatisticalAnalysis.variance([5]) == 0.0

    def test_std(self):
        std = StatisticalAnalysis.std([1, 2, 3, 4, 5])
        assert std > 0

    def test_std_single(self):
        assert StatisticalAnalysis.std([5]) == 0.0

    def test_confidence_interval(self):
        lo, hi = StatisticalAnalysis.confidence_interval([1, 2, 3, 4, 5])
        assert lo < hi

    def test_confidence_interval_single(self):
        lo, hi = StatisticalAnalysis.confidence_interval([5])
        assert lo == 0.0 and hi == 0.0

    def test_paired_ttest(self):
        before = [0.5, 0.6, 0.7, 0.6, 0.8]
        after = [0.6, 0.7, 0.8, 0.7, 0.9]
        result = StatisticalAnalysis.paired_ttest(before, after)
        assert "t_statistic" in result
        assert "p_value" in result
        assert "degrees_of_freedom" in result

    def test_paired_ttest_single(self):
        result = StatisticalAnalysis.paired_ttest([0.5], [0.6])
        assert result["p_value"] == 1.0

    def test_paired_ttest_mismatched(self):
        result = StatisticalAnalysis.paired_ttest([0.5, 0.6], [0.6])
        assert "t_statistic" in result

    def test_wilcoxon(self):
        before = [0.5, 0.6, 0.7, 0.6, 0.8]
        after = [0.6, 0.7, 0.8, 0.7, 0.9]
        result = StatisticalAnalysis.wilcoxon_signed_rank(before, after)
        assert "statistic" in result
        assert "p_value" in result

    def test_wilcoxon_single(self):
        result = StatisticalAnalysis.wilcoxon_signed_rank([0.5], [0.6])
        assert result["p_value"] == 1.0

    def test_effect_size(self):
        before = [0.5, 0.6, 0.7, 0.6, 0.8]
        after = [0.6, 0.7, 0.8, 0.7, 0.9]
        result = StatisticalAnalysis.effect_size(before, after)
        assert "cohens_d" in result
        assert "hedges_g" in result

    def test_effect_size_single(self):
        result = StatisticalAnalysis.effect_size([0.5], [0.6])
        assert result["cohens_d"] == 0.0

    def test_effect_size_identical(self):
        result = StatisticalAnalysis.effect_size([1.0, 1.0], [1.0, 1.0])
        assert result["cohens_d"] == 0.0

    def test_describe(self):
        d = StatisticalAnalysis.describe([1, 2, 3, 4, 5])
        assert d["count"] == 5.0
        assert d["mean"] == 3.0
        assert d["min"] == 1.0
        assert d["max"] == 5.0

    def test_compare_groups(self):
        result = StatisticalAnalysis.compare_groups([0.5, 0.6], [0.7, 0.8])
        assert "group_a" in result
        assert "group_b" in result
        assert "mean_difference" in result
        assert "effect_size" in result
        assert "test_results" in result

    def test_compare_groups_wilcoxon(self):
        result = StatisticalAnalysis.compare_groups(
            [0.5, 0.6, 0.7], [0.6, 0.7, 0.8], test="wilcoxon"
        )
        assert result["test"] == "wilcoxon_signed_rank"

    def test_compare_groups_relative_improvement(self):
        result = StatisticalAnalysis.compare_groups([0.5, 0.5], [0.6, 0.6])
        assert result["relative_improvement"] > 0


# ============================================================
# Test: EvaluationEngine
# ============================================================


class TestEvaluationEngine:
    def test_initialization(self):
        engine = EvaluationEngine({"rounds": 10})
        assert engine.config == {"rounds": 10}
        assert engine.eval_history == []

    def test_evaluate_training(self, simple_model, simple_dataloader):
        engine = EvaluationEngine()
        metrics = engine.evaluate_training(simple_model, simple_dataloader)
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics

    def test_evaluate_training_with_loss(self, simple_model, simple_dataloader):
        engine = EvaluationEngine()
        loss_fn = nn.CrossEntropyLoss()
        metrics = engine.evaluate_training(
            simple_model, simple_dataloader, loss_fn=loss_fn
        )
        assert metrics["loss"] >= 0

    def test_evaluate_validation(self, simple_model, simple_dataloader):
        engine = EvaluationEngine()
        metrics = engine.evaluate_validation(simple_model, simple_dataloader)
        assert "accuracy" in metrics

    def test_evaluate_testing(self, simple_model, simple_dataloader):
        engine = EvaluationEngine()
        metrics = engine.evaluate_testing(simple_model, simple_dataloader)
        assert "accuracy" in metrics

    def test_evaluate_single_client(self, simple_model, simple_dataloader):
        engine = EvaluationEngine()
        metrics = engine.evaluate_single_client("c1", simple_model, simple_dataloader)
        assert metrics["client_id"] == "c1"

    def test_evaluate_multi_client(self, simple_model, simple_dataloader):
        engine = EvaluationEngine()
        clients = {
            "c1": (simple_model, simple_dataloader),
            "c2": (simple_model, simple_dataloader),
        }
        results = engine.evaluate_multi_client(clients)
        assert "c1" in results
        assert "c2" in results

    def test_evaluate_single_modality(self, simple_model, simple_dataloader):
        engine = EvaluationEngine()
        metrics = engine.evaluate_single_modality(
            simple_model, simple_dataloader, modality_name="image"
        )
        assert metrics["modality"] == "image"

    def test_evaluate_multi_modality(self, simple_model, simple_dataloader):
        engine = EvaluationEngine()
        loaders = {"image": simple_dataloader, "text": simple_dataloader}
        results = engine.evaluate_multi_modality(simple_model, loaders)
        assert "image" in results
        assert "text" in results

    def test_evaluate_missing_modality(self, simple_model, simple_dataloader):
        engine = EvaluationEngine()
        loaders = {"image": simple_dataloader, "text": simple_dataloader}
        metrics = engine.evaluate_missing_modality(
            simple_model, loaders, missing_modalities=["audio"]
        )
        assert metrics["missing_modalities"] == 1.0

    def test_evaluate_personalized(self, simple_model, simple_dataloader):
        engine = EvaluationEngine()
        p_models = {"c1": (simple_model, simple_dataloader)}
        results = engine.evaluate_personalized(p_models, global_model=simple_model)
        assert "c1" in results
        assert "_summary" in results

    def test_evaluate_prototypes(self):
        engine = EvaluationEngine()
        old = [torch.randn(10) for _ in range(5)]
        new = [torch.randn(10) for _ in range(5)]
        metrics = engine.evaluate_prototypes(old_embeddings=old, new_embeddings=new)
        assert "prototype_drift" in metrics

    def test_evaluate_prototypes_empty(self):
        engine = EvaluationEngine()
        metrics = engine.evaluate_prototypes()
        assert metrics == {}

    def test_evaluate_knowledge_transfer(self):
        engine = EvaluationEngine()
        src = torch.randn(10, 10)
        tgt = torch.randn(10, 10)
        metrics = engine.evaluate_knowledge_transfer(
            source_embeddings=src, target_embeddings=tgt
        )
        assert "alignment_score" in metrics

    def test_evaluate_communication(self):
        engine = EvaluationEngine()
        metrics = engine.evaluate_communication(bytes_sent=100, bytes_received=50)
        assert "communication_cost" in metrics

    def test_evaluate_personalization_metrics(self):
        engine = EvaluationEngine()
        metrics = engine.evaluate_personalization(
            personalized=torch.randn(10),
            global_prototype=torch.randn(10),
            fusion_weights={"local": 0.5, "global": 0.5},
        )
        assert "personalization_gain" in metrics
        assert "prototype_fusion_quality" in metrics

    def test_evaluate_all(self, simple_model, simple_dataloader):
        engine = EvaluationEngine()
        metrics = engine.evaluate_all(
            round_id=1,
            model=simple_model,
            dataloader=simple_dataloader,
            extra_metric=0.5,
        )
        assert metrics["round_id"] == 1.0
        assert "accuracy" in metrics
        assert "extra_metric" in metrics

    def test_compute_metric(self):
        engine = EvaluationEngine()
        from app.evaluation.metrics import MetricRegistry

        MetricRegistry.register("accuracy", ClassificationMetrics.accuracy)
        acc = engine.compute_metric(
            "accuracy",
            torch.tensor([[0.1, 0.9], [0.8, 0.2]]),
            torch.tensor([1, 0]),
        )
        assert acc == 1.0

    def test_compute_metric_unknown(self):
        engine = EvaluationEngine()
        result = engine.compute_metric("nonexistent_metric")
        assert result == 0.0

    def test_compute_training_time(self):
        engine = EvaluationEngine()
        tt = engine.compute_training_time(1.0, 3.0)
        assert tt == 2.0

    def test_compute_inference_time(self, simple_model):
        engine = EvaluationEngine()
        it = engine.compute_inference_time(
            simple_model, torch.randn(1, 10), repetitions=5
        )
        assert it >= 0

    def test_record_resource_usage(self):
        engine = EvaluationEngine()
        engine.record_resource_usage(gpu_usage=0.5, cpu_usage=0.3, memory_usage=0.8)
        summary = engine.get_resource_summary()
        assert summary["avg_gpu_usage"] == 0.5

    def test_get_resource_summary_empty(self):
        engine = EvaluationEngine()
        summary = engine.get_resource_summary()
        assert summary["avg_gpu_usage"] == 0.0

    def test_summary(self, simple_model, simple_dataloader):
        engine = EvaluationEngine()
        engine.evaluate_all(
            round_id=1, model=simple_model, dataloader=simple_dataloader
        )
        engine.evaluate_all(
            round_id=2, model=simple_model, dataloader=simple_dataloader
        )
        s = engine.summary()
        assert s["num_evaluations"] == 2
        assert "accuracy_mean" in s

    def test_summary_empty(self):
        engine = EvaluationEngine()
        s = engine.summary()
        assert s["num_evaluations"] == 0

    def test_clear(self, simple_model, simple_dataloader):
        engine = EvaluationEngine()
        engine.evaluate_all(
            round_id=1, model=simple_model, dataloader=simple_dataloader
        )
        assert len(engine.eval_history) == 1
        engine.clear()
        assert len(engine.eval_history) == 0

    def test_evaluate_training_with_dict_batch(self):
        engine = EvaluationEngine()
        model = nn.Linear(10, 5)

        class DictDataset(torch.utils.data.Dataset):
            def __init__(self):
                self.data = torch.randn(10, 10)
                self.labels = torch.randint(0, 5, (10,))

            def __len__(self):
                return 10

            def __getitem__(self, idx):
                return {"data": self.data[idx], "label": self.labels[idx]}

        loader = DataLoader(DictDataset(), batch_size=5)
        metrics = engine.evaluate_training(model, loader)
        assert "accuracy" in metrics

    def test_evaluate_training_empty_loader(self):
        engine = EvaluationEngine()
        model = nn.Linear(10, 5)
        data = torch.randn(0, 10)
        labels = torch.zeros(0, dtype=torch.long)
        dataset = TensorDataset(data, labels)
        loader = DataLoader(dataset, batch_size=5)
        metrics = engine.evaluate_training(model, loader)
        assert metrics["accuracy"] == 0.0

    def test_evaluate_personalized_no_global(self, simple_model, simple_dataloader):
        engine = EvaluationEngine()
        p_models = {"c1": (simple_model, simple_dataloader)}
        results = engine.evaluate_personalized(p_models)
        assert "c1" in results
        assert "_summary" in results


# ============================================================
# Test: VisualizationDataGenerator
# ============================================================


class TestVisualizationDataGenerator:
    def test_training_loss(self):
        data = VisualizationDataGenerator.training_loss(
            [
                {"round_id": 1, "loss": 0.5},
                {"round_id": 2, "loss": 0.3},
            ]
        )
        assert len(data) == 2
        assert data[0]["loss"] == 0.5

    def test_accuracy(self):
        data = VisualizationDataGenerator.accuracy(
            [
                {"round_id": 1, "accuracy": 0.8},
            ]
        )
        assert data[0]["accuracy"] == 0.8

    def test_communication_rounds(self):
        data = VisualizationDataGenerator.communication_rounds(
            [
                {"round_id": 1, "communication_cost": 100},
            ]
        )
        assert data[0]["communication_cost"] == 100.0

    def test_prototype_drift(self):
        data = VisualizationDataGenerator.prototype_drift(
            [
                {"drift": 0.1},
                {"drift": 0.2},
            ]
        )
        assert len(data) == 2

    def test_prototype_evolution(self):
        data = VisualizationDataGenerator.prototype_evolution(
            [
                {
                    "round": 1,
                    "class_id": 0,
                    "modality": "image",
                    "confidence": 0.9,
                    "embedding": [0.1, 0.2, 0.3],
                },
            ]
        )
        assert len(data) == 1
        assert data[0]["x"] == 0.1

    def test_prototype_evolution_tensor(self):
        data = VisualizationDataGenerator.prototype_evolution(
            [
                {
                    "round": 1,
                    "class_id": 0,
                    "modality": "image",
                    "confidence": 0.9,
                    "embedding": torch.tensor([0.1, 0.2]),
                },
            ]
        )
        assert abs(data[0]["x"] - 0.1) < 1e-6

    def test_prototype_evolution_no_embedding(self):
        data = VisualizationDataGenerator.prototype_evolution(
            [
                {"round": 1, "class_id": 0, "modality": "image", "confidence": 0.9},
            ]
        )
        assert "x" not in data[0]

    def test_client_participation(self):
        data = VisualizationDataGenerator.client_participation(
            {
                "c1": [{"round": 1}, {"round": 2}],
                "c2": [{"round": 1}],
            }
        )
        assert len(data) == 3

    def test_knowledge_transfer_quality(self):
        data = VisualizationDataGenerator.knowledge_transfer_quality(
            [
                {"round_id": 1, "alignment_score": 0.8},
            ]
        )
        assert data[0]["alignment_score"] == 0.8

    def test_personalization_gain(self):
        data = VisualizationDataGenerator.personalization_gain(
            [
                {"round_id": 1, "personalization_gain": 0.1},
            ]
        )
        assert data[0]["personalization_gain"] == 0.1

    def test_all_metrics(self):
        data = VisualizationDataGenerator.all_metrics(
            round_metrics=[{"round_id": 1, "accuracy": 0.9}],
            prototype_history=[{"drift": 0.1}],
            transfer_metrics=[{"round_id": 1, "alignment_score": 0.8}],
            personalization_metrics=[{"round_id": 1, "personalization_gain": 0.1}],
            client_histories={"c1": [{"round": 1}]},
        )
        assert "training_loss" in data
        assert "accuracy" in data
        assert "prototype_drift" in data
        assert "knowledge_transfer_quality" in data
        assert "personalization_gain" in data
        assert "client_participation" in data


# ============================================================
# Test: Exporter
# ============================================================


class TestExporter:
    def test_export_json(self, temp_dir):
        exporter = Exporter(output_dir=temp_dir)
        path = exporter.export_json({"key": "value"}, "test.json")
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert data["key"] == "value"

    def test_export_csv_dict(self, temp_dir):
        exporter = Exporter(output_dir=temp_dir)
        path = exporter.export_csv(
            {"exp1": {"acc": 0.9}, "exp2": {"acc": 0.8}}, "test.csv"
        )
        assert os.path.exists(path)

    def test_export_csv_list(self, temp_dir):
        exporter = Exporter(output_dir=temp_dir)
        path = exporter.export_csv([{"acc": 0.9}, {"acc": 0.8}], "test.csv")
        assert os.path.exists(path)

    def test_export_csv_empty(self, temp_dir):
        exporter = Exporter(output_dir=temp_dir)
        path = exporter.export_csv([], "empty.csv")
        assert os.path.exists(path)

    def test_export_excel_list(self, temp_dir):
        exporter = Exporter(output_dir=temp_dir)
        path = exporter.export_excel([{"acc": 0.9}], "test.xlsx")
        assert os.path.exists(path)

    def test_export_excel_dict_flat(self, temp_dir):
        exporter = Exporter(output_dir=temp_dir)
        path = exporter.export_excel({"acc": 0.9, "loss": 0.1}, "test_flat.xlsx")
        assert os.path.exists(path)

    def test_export_excel_dict_nested(self, temp_dir):
        exporter = Exporter(output_dir=temp_dir)
        path = exporter.export_excel(
            {"metrics": [{"acc": 0.9}], "info": {"version": "1.0"}}, "test_nested.xlsx"
        )
        assert os.path.exists(path)

    def test_export_markdown(self, temp_dir):
        exporter = Exporter(output_dir=temp_dir)
        path = exporter.export_markdown("# Test Report", "test.md")
        assert os.path.exists(path)
        with open(path) as f:
            content = f.read()
        assert "# Test Report" in content

    def test_export_latex(self, temp_dir):
        exporter = Exporter(output_dir=temp_dir)
        data = [{"accuracy": 0.9, "loss": 0.1}]
        path = exporter.export_latex(data, "test.tex")
        assert os.path.exists(path)
        with open(path) as f:
            content = f.read()
        assert "\\begin{table}" in content

    def test_export_latex_empty(self, temp_dir):
        exporter = Exporter(output_dir=temp_dir)
        path = exporter.export_latex([], "empty.tex")
        assert os.path.exists(path)

    def test_export_all(self, temp_dir):
        exporter = Exporter(output_dir=temp_dir)
        paths = exporter.export_all(
            {"key": "value"}, "all_test", formats=["json", "csv", "md", "tex"]
        )
        assert "json" in paths
        assert "csv" in paths
        assert "markdown" in paths
        assert "latex" in paths

    def test_export_dataframe(self, temp_dir):
        import pandas as pd

        exporter = Exporter(output_dir=temp_dir)
        df = pd.DataFrame({"a": [1, 2, 3]})
        path = exporter.export_dataframe(df, "df.csv")
        assert os.path.exists(path)

    def test_clear(self, temp_dir):
        exporter = Exporter(output_dir=temp_dir)
        exporter.export_json({"key": "value"}, "test.json")
        assert len(list(exporter._output_dir.iterdir())) > 0
        exporter.clear()
        assert len(list(exporter._output_dir.iterdir())) == 0

    def test_output_dir(self, temp_dir):
        exporter = Exporter(output_dir=temp_dir)
        assert str(exporter.output_dir) == str(temp_dir)

    def test_export_latex_with_dict(self, temp_dir):
        exporter = Exporter(output_dir=temp_dir)
        path = exporter.export_latex({"acc": 0.9}, "dict.tex")
        assert os.path.exists(path)


# ============================================================
# Test: ReportGenerator
# ============================================================


class TestReportGenerator:
    def test_generate(self):
        gen = ReportGenerator()
        report = gen.generate(
            experiment_summary={"num_experiments": 1},
            dataset_info={"name": "test"},
            config={"rounds": 10},
            metrics={"accuracy": 0.9},
            communication_stats={"bytes": 100},
            prototype_stats={"drift": 0.1},
            knowledge_transfer_stats={"alignment": 0.8},
            personalization_stats={"gain": 0.2},
            best_model={"id": "exp1"},
            conclusions=["Test complete."],
        )
        assert "# PP-MFL Experiment Report" in report
        assert "## 1. Experiment Summary" in report
        assert "## 10. Conclusions" in report
        assert "Test complete." in report

    def test_generate_minimal(self):
        gen = ReportGenerator()
        report = gen.generate(experiment_summary={"num_experiments": 0})
        assert "# PP-MFL Experiment Report" in report

    def test_generate_from_runner(self):
        gen = ReportGenerator()
        runner = MagicMock()
        runner._results = {"exp1": {"final_metrics": {"accuracy": 0.9}}}
        runner._config = {"rounds": 10}
        runner.engine.summary.return_value = {"accuracy_mean": 0.9}
        runner.summarize_experiments.return_value = {"num_experiments": 1}
        runner.get_best_experiment.return_value = ("exp1", 0.9)
        report = gen.generate_from_runner(runner)
        assert "# PP-MFL Experiment Report" in report

    def test_generate_from_runner_empty(self):
        gen = ReportGenerator()
        runner = MagicMock()
        runner._results = {}
        runner._config = {}
        runner.engine = MagicMock()
        runner.engine.summary.return_value = {}
        report = gen.generate_from_runner(runner)
        assert "# PP-MFL Experiment Report" in report

    def test_custom_title(self):
        gen = ReportGenerator(title="Custom Report")
        report = gen.generate(experiment_summary={"test": True})
        assert "# Custom Report" in report

    def test_section_with_list(self):
        gen = ReportGenerator()
        report = gen.generate(experiment_summary=["item1", "item2"])
        assert "item1" in report
        assert "item2" in report

    def test_section_with_none(self):
        gen = ReportGenerator()
        report = gen.generate(experiment_summary=None)
        assert "No data available" in report


# ============================================================
# Test: Leaderboard
# ============================================================


class TestLeaderboard:
    def test_add_entry(self):
        lb = Leaderboard()
        lb.add_entry("exp1", {"accuracy": 0.9, "f1_score": 0.8})
        assert len(lb.entries) == 1

    def test_add_entries(self):
        lb = Leaderboard()
        lb.add_entries(
            [
                {"experiment_id": "exp1", "accuracy": 0.9},
                {"experiment_id": "exp2", "accuracy": 0.8},
            ]
        )
        assert len(lb.entries) == 2

    def test_rank_by(self):
        lb = Leaderboard()
        lb.add_entry("exp1", {"accuracy": 0.8})
        lb.add_entry("exp2", {"accuracy": 0.9})
        ranked = lb.rank_by("accuracy")
        assert ranked[0]["rank"] == 1
        assert ranked[0]["experiment_id"] == "exp2"

    def test_rank_by_ascending(self):
        lb = Leaderboard()
        lb.add_entry("exp1", {"accuracy": 0.8})
        lb.add_entry("exp2", {"accuracy": 0.9})
        ranked = lb.rank_by("accuracy", ascending=True)
        assert ranked[0]["experiment_id"] == "exp1"

    def test_rank_by_top_k(self):
        lb = Leaderboard()
        lb.add_entry("exp1", {"accuracy": 0.8})
        lb.add_entry("exp2", {"accuracy": 0.9})
        lb.add_entry("exp3", {"accuracy": 0.7})
        ranked = lb.rank_by("accuracy", top_k=2)
        assert len(ranked) == 2

    def test_rank_by_accuracy(self):
        lb = Leaderboard()
        lb.add_entry("exp1", {"accuracy": 0.9})
        ranked = lb.rank_by_accuracy()
        assert ranked[0]["experiment_id"] == "exp1"

    def test_rank_by_f1(self):
        lb = Leaderboard()
        lb.add_entry("exp1", {"f1_score": 0.9})
        ranked = lb.rank_by_f1()
        assert ranked[0]["experiment_id"] == "exp1"

    def test_rank_by_communication_cost(self):
        lb = Leaderboard()
        lb.add_entry("exp1", {"communication_cost": 100})
        lb.add_entry("exp2", {"communication_cost": 50})
        ranked = lb.rank_by_communication_cost()
        assert ranked[0]["experiment_id"] == "exp2"

    def test_rank_by_training_time(self):
        lb = Leaderboard()
        lb.add_entry("exp1", {"training_time": 100})
        lb.add_entry("exp2", {"training_time": 50})
        ranked = lb.rank_by_training_time()
        assert ranked[0]["experiment_id"] == "exp2"

    def test_rank_by_prototype_quality(self):
        lb = Leaderboard()
        lb.add_entry("exp1", {"prototype_fusion_quality": 0.9})
        ranked = lb.rank_by_prototype_quality()
        assert ranked[0]["experiment_id"] == "exp1"

    def test_rank_by_personalization_gain(self):
        lb = Leaderboard()
        lb.add_entry("exp1", {"personalization_gain": 0.5})
        ranked = lb.rank_by_personalization_gain()
        assert ranked[0]["experiment_id"] == "exp1"

    def test_get_best(self):
        lb = Leaderboard()
        lb.add_entry("exp1", {"accuracy": 0.8})
        lb.add_entry("exp2", {"accuracy": 0.9})
        best = lb.get_best()
        assert best is not None
        assert best["experiment_id"] == "exp2"

    def test_get_best_no_entries(self):
        lb = Leaderboard()
        best = lb.get_best()
        assert best is None

    def test_get_worst(self):
        lb = Leaderboard()
        lb.add_entry("exp1", {"accuracy": 0.8})
        lb.add_entry("exp2", {"accuracy": 0.9})
        worst = lb.get_worst()
        assert worst["experiment_id"] == "exp1"

    def test_summary(self):
        lb = Leaderboard()
        lb.add_entry("exp1", {"accuracy": 0.9})
        s = lb.summary()
        assert s["total_entries"] == 1

    def test_clear(self):
        lb = Leaderboard()
        lb.add_entry("exp1", {"accuracy": 0.9})
        lb.clear()
        assert len(lb.entries) == 0

    def test_remove_entry(self):
        lb = Leaderboard()
        lb.add_entry("exp1", {"accuracy": 0.9})
        lb.add_entry("exp2", {"accuracy": 0.8})
        lb.remove_entry("exp1")
        assert len(lb.entries) == 1
        assert lb.entries[0]["experiment_id"] == "exp2"

    def test_rank_by_missing_metric(self):
        lb = Leaderboard()
        lb.add_entry("exp1", {"accuracy": 0.9})
        ranked = lb.rank_by("nonexistent")
        assert ranked == []


# ============================================================
# Test: Benchmark
# ============================================================


class TestBenchmark:
    def test_initialization(self):
        bm = Benchmark({})
        assert isinstance(bm.engine, EvaluationEngine)
        assert bm.results == {}

    def test_benchmark_model(self, simple_model, simple_dataloader):
        bm = Benchmark()
        result = bm.benchmark_model(
            simple_model, simple_dataloader, model_name="test_model"
        )
        assert "accuracy" in result
        assert result["model_name"] == "test_model"
        assert "inference_time_ms" in result
        assert "num_parameters" in result

    def test_benchmark_inference(self, simple_model):
        bm = Benchmark()
        result = bm.benchmark_inference(
            simple_model, (10,), batch_size=1, repetitions=5
        )
        assert "avg_inference_time_ms" in result
        assert "throughput_items_per_sec" in result

    def test_benchmark_training_speed(self, simple_model, simple_dataloader):
        bm = Benchmark()
        opt = torch.optim.SGD(simple_model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()
        result = bm.benchmark_training_speed(
            simple_model, simple_dataloader, loss_fn, opt, epochs=2
        )
        assert "samples_per_second" in result
        assert "seconds_per_epoch" in result

    def test_summary(self, simple_model, simple_dataloader):
        bm = Benchmark()
        bm.benchmark_model(simple_model, simple_dataloader, model_name="m1")
        s = bm.summary()
        assert s["num_benchmarks"] == 1

    def test_clear(self, simple_model, simple_dataloader):
        bm = Benchmark()
        bm.benchmark_model(simple_model, simple_dataloader, model_name="m1")
        bm.clear()
        assert bm.results == {}

    def test_benchmark_baseline(self, simple_model, simple_dataloader):
        bm = Benchmark()
        result = bm.benchmark_baseline(
            "fedavg",
            num_rounds=2,
            server_model=simple_model,
            dataloaders=[simple_dataloader, simple_dataloader],
        )
        assert len(result) > 0


# ============================================================
# Test: ExperimentRunner
# ============================================================


class TestExperimentRunner:
    def test_initialization(self):
        runner = ExperimentRunner({"rounds": 5})
        assert isinstance(runner.engine, EvaluationEngine)

    def test_run_baselines(self, simple_model, simple_dataloader):
        runner = ExperimentRunner()
        results = runner.run_baselines(
            ["fedavg"],
            {},
            num_rounds=2,
            server_model=simple_model,
            dataloaders=[simple_dataloader, simple_dataloader],
        )
        assert "fedavg" in results
        assert len(results["fedavg"]) == 2

    def test_run_baselines_unknown(self):
        runner = ExperimentRunner()
        results = runner.run_baselines(["nonexistent"], {})
        assert "nonexistent" in results

    def test_run_baselines_no_model(self):
        runner = ExperimentRunner()
        results = runner.run_baselines(["fedavg"], {}, num_rounds=1)
        assert "fedavg" in results

    def test_run_ablation(self):
        runner = ExperimentRunner()
        with patch.object(runner, "run_single", return_value={"accuracy": 0.9}):
            result = runner.run_ablation(
                {"rounds": 2, "clients": {"num_clients": 2}},
                ablations=["without_prototypes"],
            )
        assert "results" in result
        assert "comparison" in result

    def test_run_with_multiple_seeds(self):
        runner = ExperimentRunner()
        with patch.object(runner, "run_single", return_value={"accuracy": 0.9}):
            results = runner.run_with_multiple_seeds(
                {"rounds": 2, "clients": {"num_clients": 2}},
                seeds=[42, 123],
            )
        assert "seed_42" in results
        assert "seed_123" in results

    def test_run_with_multiple_datasets(self):
        runner = ExperimentRunner()
        with patch.object(runner, "run_single", return_value={"accuracy": 0.9}):
            results = runner.run_with_multiple_datasets(
                {"rounds": 2, "clients": {"num_clients": 2}},
                datasets=["ds1", "ds2"],
            )
        assert "dataset_ds1" in results
        assert "dataset_ds2" in results

    def test_run_batch(self):
        runner = ExperimentRunner()
        with patch.object(runner, "run_single", return_value={"accuracy": 0.9}):
            results = runner.run_batch(
                [
                    {"rounds": 2, "clients": {"num_clients": 2}},
                    {"rounds": 2, "clients": {"num_clients": 2}},
                ]
            )
        assert len(results) == 2

    def test_run_batch_with_error(self):
        runner = ExperimentRunner()
        with patch.object(runner, "run_single", side_effect=ValueError("fail")):
            results = runner.run_batch(
                [
                    {"rounds": 2, "clients": {"num_clients": 2}},
                ]
            )
        assert "error" in list(results.values())[0]

    def test_save_and_load_results(self, temp_dir):
        runner = ExperimentRunner(output_dir=temp_dir)
        runner._results = {"exp1": {"final_metrics": {"accuracy": 0.9}}}
        path = runner.save_results(os.path.join(temp_dir, "test_results.json"))
        assert os.path.exists(path)
        runner._results = {}
        loaded = runner.load_results(path)
        assert "exp1" in loaded

    def test_get_best_experiment(self):
        runner = ExperimentRunner()
        runner._results = {
            "exp1": {"final_metrics": {"accuracy": 0.8}},
            "exp2": {"final_metrics": {"accuracy": 0.9}},
        }
        best_id, best_val = runner.get_best_experiment()
        assert best_id == "exp2"
        assert best_val == 0.9

    def test_get_best_experiment_empty(self):
        runner = ExperimentRunner()
        best_id, best_val = runner.get_best_experiment()
        assert best_id == ""

    def test_summarize_experiments(self):
        runner = ExperimentRunner()
        runner._results = {
            "exp1": {"final_metrics": {"accuracy": 0.8}},
            "exp2": {"final_metrics": {"accuracy": 0.9}},
        }
        s = runner.summarize_experiments()
        assert s["num_experiments"] == 2
        assert "accuracy_mean" in s

    def test_summarize_experiments_empty(self):
        runner = ExperimentRunner()
        s = runner.summarize_experiments()
        assert s["num_experiments"] == 0

    def test_clear(self):
        runner = ExperimentRunner()
        runner._results = {"exp1": {"final_metrics": {"accuracy": 0.9}}}
        runner.clear()
        assert runner._results == {}

    def test_resume_experiment_no_registry(self):
        runner = ExperimentRunner(
            config={
                "rounds": 5,
                "clients": {"num_clients": 2},
                "model": {"type": "simple"},
                "dataset": {"name": "generic"},
                "modalities": {"image": 64},
                "mappings": [],
                "num_classes": 5,
            }
        )
        result = runner.resume_experiment("nonexistent")
        assert "completed" in result

    def test_run_with_multiple_seeds_error(self):
        runner = ExperimentRunner()
        with patch.object(runner, "run_single", side_effect=ValueError("fail")):
            results = runner.run_with_multiple_seeds({"rounds": 2}, seeds=[42])
        assert "error" in results["seed_42"]

    def test_run_with_multiple_datasets_error(self):
        runner = ExperimentRunner()
        with patch.object(runner, "run_single", side_effect=ValueError("fail")):
            results = runner.run_with_multiple_datasets({"rounds": 2}, datasets=["ds1"])
        assert "error" in results["dataset_ds1"]


# ============================================================
# Test: Edge Cases
# ============================================================


class TestEdgeCases:
    def test_classification_metrics_empty_outputs(self):
        outputs = torch.zeros(0, 2)
        targets = torch.zeros(0, dtype=torch.long)
        acc = ClassificationMetrics.accuracy(outputs, targets)
        assert acc == 0.0

    def test_classification_metrics_single_class(self):
        outputs = torch.tensor([[0.1, 0.9], [0.2, 0.8]])
        targets = torch.tensor([1, 1])
        prec = ClassificationMetrics.precision(outputs, targets)
        assert prec == 0.5
        f1 = ClassificationMetrics.f1_score(outputs, targets)
        assert abs(f1 - 0.5) < 1e-6

    def test_prototype_drift_empty_lists(self):
        drift = PrototypeMetrics.prototype_drift([], [])
        assert drift == 0.0

    def test_prototype_diversity_empty(self):
        div = PrototypeMetrics.prototype_diversity([])
        assert div == 0.0

    def test_statistical_analysis_all_empty(self):
        assert StatisticalAnalysis.mean([]) == 0.0
        assert StatisticalAnalysis.median([]) == 0.0
        assert StatisticalAnalysis.std([]) == 0.0

    def test_leaderboard_empty_rankings(self):
        lb = Leaderboard()
        assert lb.rank_by_accuracy() == []
        assert lb.get_best() is None

    def test_exporter_empty_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            exporter = Exporter(output_dir=tmp)
            path = exporter.export_csv([], "empty.csv")
            assert os.path.exists(path)

    def test_benchmark_empty_model(self):
        bm = Benchmark()
        result = bm.benchmark_inference(
            nn.Linear(10, 5), (10,), batch_size=1, repetitions=2
        )
        assert "avg_inference_time_ms" in result

    def test_visualization_empty_metrics(self):
        data = VisualizationDataGenerator.all_metrics([])
        assert "training_loss" in data
        assert "accuracy" in data

    def test_registry_clear_metrics(self):
        MetricRegistry.register("tmp_edge_metric", lambda x: x)
        assert MetricRegistry.contains("tmp_edge_metric")
        MetricRegistry.unregister("tmp_edge_metric")
        assert not MetricRegistry.contains("tmp_edge_metric")

    def test_evaluation_engine_dict_batch_with_missing_keys(self):
        engine = EvaluationEngine()
        model = nn.Linear(10, 5)

        class BadDictDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 5

            def __getitem__(self, idx):
                return {
                    "wrong_key": torch.randn(10),
                    "bad_target": torch.randint(0, 5, (1,)).item(),
                }

        loader = DataLoader(BadDictDataset(), batch_size=5)
        metrics = engine.evaluate_training(model, loader)
        assert metrics["accuracy"] == 0.0

    def test_evaluation_engine_invalid_batch_structure(self):
        engine = EvaluationEngine()
        model = nn.Linear(10, 5)

        class BadBatchDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 3

            def __getitem__(self, idx):
                return torch.randn(10)

        loader = DataLoader(BadBatchDataset(), batch_size=3)
        metrics = engine.evaluate_training(model, loader)
        assert metrics["accuracy"] == 0.0

    def test_evaluation_engine_single_element_batch(self):
        engine = EvaluationEngine()
        model = nn.Linear(10, 5)

        class SingleElementDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 1

            def __getitem__(self, idx):
                return (torch.randn(10), torch.randint(0, 5, (1,)).item())

        loader = DataLoader(SingleElementDataset(), batch_size=1)
        metrics = engine.evaluate_training(model, loader)
        assert "accuracy" in metrics

    def test_fedavg_with_zero_samples(self):
        base = BaselineFactory.create("fedavg", {})
        model = nn.Linear(10, 5)
        data = torch.randn(0, 10)
        labels = torch.zeros(0, dtype=torch.long)
        dataset = TensorDataset(data, labels)
        loader = DataLoader(dataset, batch_size=1)
        clients = [{"lr": 0.01}]
        metrics = base.train_round(1, clients, model, [loader])
        assert "accuracy" in metrics

    def test_scaffold_multi_round(self):
        base = BaselineFactory.create("scaffold", {})
        model = nn.Linear(10, 5)
        data = torch.randn(10, 10)
        labels = torch.randint(0, 5, (10,))
        dataset = TensorDataset(data, labels)
        loader = DataLoader(dataset, batch_size=5)
        clients = [{"lr": 0.01}, {"lr": 0.01}]
        m1 = base.train_round(1, clients, model, [loader, loader])
        m2 = base.train_round(2, clients, model, [loader, loader])
        assert m1["round_id"] == 1
        assert m2["round_id"] == 2
