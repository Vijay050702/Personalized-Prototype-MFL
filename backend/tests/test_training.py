from __future__ import annotations

import copy
import json
import os
import tempfile
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from app.federated.models import (
    AggregatedPrototype,
    ClientPrototypePackage,
)
from app.knowledge_transfer.inference import InferenceOutput
from app.prototypes.prototype import Prototype
from app.personalization.personalized_prototype import (
    PersonalizedPrototype,
)
from app.personalization.profile import ClientProfile


# ============================================================
# fixtures
# ============================================================


@pytest.fixture
def simple_model():
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5),
    )


@pytest.fixture
def simple_dataloader():
    data = torch.randn(50, 10)
    labels = torch.randint(0, 5, (50,))
    dataset = TensorDataset(data, labels)
    return DataLoader(dataset, batch_size=10)


@pytest.fixture
def config():
    return {
        "rounds": 10,
        "clients": {"num_clients": 3},
        "model": {"type": "simple"},
        "dataset": {"name": "generic", "batch_size": 16},
        "modalities": {"image": 64, "text": 64},
        "mappings": [("image", "text")],
        "num_classes": 5,
        "optimizer": {"type": "adam", "lr": 1e-3},
        "scheduler": {"type": "cosine_annealing", "kwargs": {"t_max": 10}},
        "personalization": {
            "fusion": {"strategy": "weighted_sum"},
            "selector": {"confidence_threshold": 0.3},
            "adaptation": {"strategy": "ema"},
        },
        "federated": {},
        "checkpoint_dir": tempfile.mkdtemp(),
    }


# ============================================================
# Test: state.py
# ============================================================


class TestTrainingState:
    def test_initial_state(self):
        state = TrainingState(experiment_id="test")
        assert state.experiment_id == "test"
        assert state.phase == "initialized"
        assert state.current_round == 0

    def test_register_client(self):
        state = TrainingState(experiment_id="test")
        cs = state.register_client("client_0")
        assert cs.client_id == "client_0"
        assert "client_0" in state.clients
        assert "client_0" in state.server.total_clients_ever

    def test_register_client_duplicate(self):
        state = TrainingState(experiment_id="test")
        state.register_client("client_0")
        state.register_client("client_0")
        assert len(state.clients) == 1

    def test_get_client(self):
        state = TrainingState(experiment_id="test")
        state.register_client("client_0")
        cs = state.get_client("client_0")
        assert cs.client_id == "client_0"

    def test_get_client_raises(self):
        state = TrainingState(experiment_id="test")
        with pytest.raises(KeyError):
            state.get_client("nonexistent")

    def test_active_clients(self):
        state = TrainingState(experiment_id="test")
        state.register_client("client_0")
        state.register_client("client_1")
        state.clients["client_0"].is_active = False
        active = state.active_clients()
        assert len(active) == 1

    def test_client_ids(self):
        state = TrainingState(experiment_id="test")
        state.register_client("c1")
        state.register_client("c2")
        assert sorted(state.client_ids()) == ["c1", "c2"]

    def test_record_round_metrics(self):
        state = TrainingState(experiment_id="test")
        state.record_round_metrics(1, {"accuracy": 0.9, "loss": 0.1})
        assert state.round_metrics[1] == {"accuracy": 0.9, "loss": 0.1}
        assert state.best_round == 1
        assert state.best_metric == 0.9
        assert state.current_round == 1
        assert state.server.current_round == 1
        assert state.server.rounds_completed == 1

    def test_record_round_metrics_best(self):
        state = TrainingState(experiment_id="test")
        state.record_round_metrics(1, {"accuracy": 0.5})
        state.record_round_metrics(2, {"accuracy": 0.8})
        assert state.best_round == 2
        assert state.best_metric == 0.8

    def test_mark_completed(self):
        state = TrainingState(experiment_id="test")
        state.mark_completed()
        assert state.phase == "completed"
        assert state.end_time is not None

    def test_mark_failed(self):
        state = TrainingState(experiment_id="test")
        state.mark_failed()
        assert state.phase == "failed"
        assert state.end_time is not None

    def test_elapsed(self):
        state = TrainingState(experiment_id="test")
        assert state.elapsed >= 0

    def test_elapsed_completed(self):
        state = TrainingState(experiment_id="test")
        state.mark_completed()
        assert state.elapsed >= 0

    def test_is_completed(self):
        state = TrainingState(experiment_id="test")
        assert not state.is_completed
        state.mark_completed()
        assert state.is_completed

    def test_is_running(self):
        state = TrainingState(experiment_id="test")
        assert not state.is_running
        state.phase = "running"
        assert state.is_running

    def test_to_dict(self):
        state = TrainingState(experiment_id="test")
        state.register_client("c1")
        d = state.to_dict()
        assert d["experiment_id"] == "test"
        assert "c1" in d["clients"]

    def test_client_state_to_dict(self):
        cs = ClientState(client_id="c1")
        cs.loss_history = [0.5, 0.3]
        cs.accuracy_history = [0.8, 0.9]
        d = cs.to_dict()
        assert d["client_id"] == "c1"
        assert d["loss_history"] == [0.5, 0.3]

    def test_server_state_to_dict(self):
        ss = ServerState()
        ss.total_clients_ever.add("c1")
        d = ss.to_dict()
        assert d["total_clients_ever"] == 1


# ============================================================
# Test: utils.py
# ============================================================


class TestUtils:
    def test_compute_accuracy(self):
        outputs = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
        targets = torch.tensor([1, 0])
        acc = compute_accuracy(outputs, targets)
        assert acc.item() == 1.0

    def test_compute_accuracy_half(self):
        outputs = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
        targets = torch.tensor([0, 0])
        acc = compute_accuracy(outputs, targets)
        assert acc.item() == 0.5

    def test_to_device_tensor(self):
        t = torch.tensor([1.0])
        result = to_device(t, torch.device("cpu"))
        assert torch.equal(result, t)

    def test_to_device_dict(self):
        d = {"a": torch.tensor([1.0])}
        result = to_device(d, torch.device("cpu"))
        assert torch.equal(result["a"], d["a"])

    def test_to_device_list(self):
        l = [torch.tensor([1.0])]
        result = to_device(l, torch.device("cpu"))
        assert torch.equal(result[0], l[0])

    def test_to_device_tuple(self):
        t = (torch.tensor([1.0]),)
        result = to_device(t, torch.device("cpu"))
        assert torch.equal(result[0], t[0])

    def test_count_parameters(self, simple_model):
        counts = count_parameters(simple_model)
        assert counts["total"] > 0
        assert counts["trainable"] > 0

    def test_compute_grad_norm(self, simple_model):
        out = simple_model(torch.randn(2, 10))
        out.sum().backward()
        norm = compute_grad_norm(simple_model)
        assert norm > 0

    def test_compute_grad_norm_zero(self):
        model = nn.Linear(10, 5)
        norm = compute_grad_norm(model)
        assert norm == 0.0

    def test_clip_gradients(self, simple_model):
        out = simple_model(torch.randn(2, 10))
        out.sum().backward()
        clip_gradients(simple_model, max_norm=1.0)
        norm = compute_grad_norm(simple_model)
        assert norm <= 1.0 + 1e-6

    def test_flatten_model_state(self, simple_model):
        state = simple_model.state_dict()
        flat = flatten_model_state(state)
        assert flat.dim() == 1
        assert flat.size(0) == sum(p.numel() for p in simple_model.parameters())

    def test_unflatten_model_state(self, simple_model):
        state = simple_model.state_dict()
        flat = flatten_model_state(state)
        reconstructed = unflatten_model_state(flat, state)
        for key in state:
            assert torch.equal(reconstructed[key], state[key])

    def test_timer(self):
        t = Timer()
        t.start()
        time.sleep(0.01)
        elapsed = t.stop()
        assert elapsed >= 0.01

    def test_timer_reset(self):
        t = Timer()
        t.start()
        t.stop()
        t.reset()
        assert t.elapsed == 0.0

    def test_timer_elapsed_running(self):
        t = Timer()
        t.start()
        time.sleep(0.005)
        assert t.elapsed >= 0.005
        t.stop()

    def test_merge_configs(self):
        base = {"a": 1, "b": {"c": 2}}
        override = {"b": {"d": 3}, "e": 4}
        merged = merge_configs(base, override)
        assert merged["a"] == 1
        assert merged["b"]["c"] == 2
        assert merged["b"]["d"] == 3
        assert merged["e"] == 4

    def test_validate_config_valid(self):
        cfg = {"rounds": 10, "clients": {}, "model": {}, "dataset": {}}
        errors = validate_config(cfg)
        assert len(errors) == 0

    def test_validate_config_missing(self):
        errors = validate_config({})
        assert len(errors) == 4


# ============================================================
# Test: events.py
# ============================================================


class TestEvents:
    def test_event_creation(self):
        event = Event(EventType.ROUND_START, {"round_id": 1})
        assert event.type == EventType.ROUND_START
        assert event.data["round_id"] == 1

    def test_event_dispatcher_register(self):
        d = EventDispatcher()
        handler = MagicMock()
        d.register(EventType.ROUND_START, handler)
        assert d.handler_count[EventType.ROUND_START] == 1

    def test_event_dispatcher_dispatch(self):
        d = EventDispatcher()
        handler = MagicMock()
        d.register(EventType.ROUND_START, handler)
        d.dispatch(Event(EventType.ROUND_START, {"round_id": 1}))
        handler.assert_called_once()

    def test_event_dispatcher_dispatch_simple(self):
        d = EventDispatcher()
        handler = MagicMock()
        d.register(EventType.ROUND_START, handler)
        d.dispatch_simple(EventType.ROUND_START, {"round_id": 1})
        handler.assert_called_once()

    def test_event_dispatcher_unregister(self):
        d = EventDispatcher()
        handler = MagicMock()
        d.register(EventType.ROUND_START, handler)
        d.unregister(EventType.ROUND_START, handler)
        d.dispatch(Event(EventType.ROUND_START))
        handler.assert_not_called()

    def test_event_dispatcher_clear(self):
        d = EventDispatcher()
        d.register(EventType.ROUND_START, MagicMock())
        d.clear()
        assert d.handler_count == {}

    def test_dispatcher_dispatch_wrong_type(self):
        d = EventDispatcher()
        handler = MagicMock()
        d.register(EventType.ROUND_START, handler)
        d.dispatch(Event(EventType.ROUND_END))
        handler.assert_not_called()

    def test_event_type_enum(self):
        assert EventType.ROUND_START.value == 3


# ============================================================
# Test: hooks.py
# ============================================================


class TestHooks:
    def test_hook_context(self):
        ctx = HookContext(round_id=1, client_id="c1")
        assert ctx.round_id == 1
        assert ctx.client_id == "c1"

    def test_hook_context_metrics(self):
        ctx = HookContext()
        ctx.metrics = {"accuracy": 0.9}
        assert ctx.metrics["accuracy"] == 0.9

    def test_hook_manager_register(self):
        hm = HookManager()
        hook = MagicMock()
        hm.register("before_round", hook)
        assert hm.hook_count("before_round") == 1

    def test_hook_manager_execute(self):
        hm = HookManager()
        hook = MagicMock()
        hm.register("before_round", hook)
        ctx = HookContext(round_id=1)
        hm.execute("before_round", ctx)
        hook.execute.assert_called_once_with(ctx)

    def test_hook_manager_execute_error(self):
        hm = HookManager()
        hook = MagicMock()
        hook.execute.side_effect = ValueError("test error")
        hm.register("before_round", hook)
        ctx = HookContext(round_id=1)
        hm.execute("before_round", ctx)
        assert ctx.error is not None

    def test_hook_manager_unknown_point(self):
        hm = HookManager()
        with pytest.raises(ValueError):
            hm.register("unknown", MagicMock())

    def test_hook_manager_unregister(self):
        hm = HookManager()
        hook = MagicMock()
        hm.register("before_round", hook)
        hm.unregister("before_round", hook)
        assert hm.hook_count("before_round") == 0

    def test_available_hook_points(self):
        hm = HookManager()
        assert "before_round" in hm.available_hook_points()
        assert "on_error" in hm.available_hook_points()

    def test_total_hooks(self):
        hm = HookManager()
        hm.register("before_round", MagicMock())
        hm.register("after_round", MagicMock())
        assert hm.total_hooks() == 2

    def test_clear(self):
        hm = HookManager()
        hm.register("before_round", MagicMock())
        hm.clear()
        assert hm.total_hooks() == 0


# ============================================================
# Test: callbacks.py
# ============================================================


class TestCallbacks:
    def test_early_stopping_not_stopped(self):
        es = EarlyStopping(patience=3)
        ctx = HookContext()
        ctx.metrics = {"accuracy": 0.5}
        es.execute(ctx)
        assert not es.stopped

    def test_early_stopping_triggers(self):
        es = EarlyStopping(patience=2, metric="accuracy")
        ctx = HookContext()
        ctx.metrics = {"accuracy": 0.5}
        es.execute(ctx)
        ctx.metrics = {"accuracy": 0.5}
        es.execute(ctx)
        ctx.metrics = {"accuracy": 0.5}
        es.execute(ctx)
        assert es.stopped

    def test_early_stopping_improves(self):
        es = EarlyStopping(patience=2, metric="accuracy")
        ctx = HookContext()
        ctx.metrics = {"accuracy": 0.5}
        es.execute(ctx)
        ctx.metrics = {"accuracy": 0.9}
        es.execute(ctx)
        assert es._counter == 0
        assert not es.stopped

    def test_early_stopping_reset(self):
        es = EarlyStopping(patience=1)
        ctx = HookContext()
        ctx.metrics = {"accuracy": 0.5}
        es.execute(ctx)
        ctx.metrics = {"accuracy": 0.5}
        es.execute(ctx)
        assert es.stopped
        es.reset()
        assert not es.stopped

    def test_early_stopping_min_mode(self):
        es = EarlyStopping(patience=1, metric="loss", mode="min")
        ctx = HookContext()
        ctx.metrics = {"loss": 0.5}
        es.execute(ctx)
        ctx.metrics = {"loss": 0.5}
        es.execute(ctx)
        assert es.stopped

    def test_checkpoint_saving(self):
        ckpt_mock = MagicMock()
        cs = CheckpointSaving(ckpt_mock, interval=1, save_best=True)
        ctx = HookContext(round_id=1)
        ctx.metrics = {"accuracy": 0.9}
        cs.execute(ctx)
        assert ckpt_mock.save_latest.called
        assert ckpt_mock.save_best.called

    def test_checkpoint_saving_not_best(self):
        ckpt_mock = MagicMock()
        cs = CheckpointSaving(ckpt_mock, interval=5, save_best=True)
        cs._best_value = 1.0
        ctx = HookContext(round_id=1)
        ctx.metrics = {"accuracy": 0.5}
        cs.execute(ctx)
        assert not ckpt_mock.save_latest.called

    def test_logging_hook_before_round(self):
        logger = MagicMock()
        lh = LoggingHook(logger)
        ctx = HookContext(round_id=1)
        ctx.data = {"hook_point": "before_round", "num_clients": 3}
        lh.execute(ctx)
        logger.log_round_start.assert_called_once_with(round_id=1, num_clients=3)

    def test_logging_hook_after_round(self):
        logger = MagicMock()
        lh = LoggingHook(logger)
        ctx = HookContext(round_id=1)
        ctx.data = {"hook_point": "after_round"}
        ctx.metrics = {"accuracy": 0.9}
        lh.execute(ctx)
        logger.log_round_end.assert_called_once()

    def test_logging_hook_error(self):
        logger = MagicMock()
        lh = LoggingHook(logger)
        ctx = HookContext(round_id=1)
        ctx.data = {"hook_point": "on_error"}
        ctx.error = ValueError("test")
        lh.execute(ctx)
        logger.log_error.assert_called_once()

    def test_metric_recording(self):
        mr = MetricRecording()
        ctx = HookContext(round_id=1)
        ctx.metrics = {"accuracy": 0.9, "loss": 0.1}
        mr.execute(ctx)
        assert len(mr.history["accuracy"]) == 1

    def test_metric_recording_get_metric(self):
        mr = MetricRecording()
        ctx = HookContext(round_id=1)
        ctx.metrics = {"accuracy": 0.9}
        mr.execute(ctx)
        assert mr.get_metric("accuracy") == [(1, 0.9)]

    def test_metric_recording_latest(self):
        mr = MetricRecording()
        ctx = HookContext(round_id=1)
        ctx.metrics = {"accuracy": 0.9}
        mr.execute(ctx)
        ctx2 = HookContext(round_id=2)
        ctx2.metrics = {"accuracy": 0.95}
        mr.execute(ctx2)
        assert mr.latest("accuracy") == 0.95

    def test_metric_recording_latest_none(self):
        mr = MetricRecording()
        assert mr.latest("accuracy") is None

    def test_metric_recording_reset(self):
        mr = MetricRecording()
        ctx = HookContext(round_id=1)
        ctx.metrics = {"accuracy": 0.9}
        mr.execute(ctx)
        mr.reset()
        assert mr.history == {}

    def test_lr_update_hook(self):
        scheduler = MagicMock()
        lr = LRUpdateHook(scheduler)
        lr.execute(HookContext())
        scheduler.step.assert_called_once()


# ============================================================
# Test: logger.py
# ============================================================


class TestTrainingLogger:
    def test_logger_creation(self):
        logger = TrainingLogger(experiment_id="test")
        assert logger.experiment_id == "test"

    def test_log_experiment_start(self):
        logger = TrainingLogger("test")
        logger.log_experiment_start({"rounds": 10})
        assert logger.log_count > 0

    def test_log_experiment_end(self):
        logger = TrainingLogger("test")
        logger.log_experiment_end("completed")
        assert logger.log_count > 0

    def test_log_round_start(self):
        logger = TrainingLogger("test")
        logger.log_round_start(round_id=1, num_clients=3)
        assert logger.log_count > 0

    def test_log_round_end(self):
        logger = TrainingLogger("test")
        logger.log_round_end(round_id=1, metrics={"acc": 0.9})
        assert logger.log_count > 0

    def test_log_client_update(self):
        logger = TrainingLogger("test")
        logger.log_client_update("c1", 1, loss=0.5, accuracy=0.9, num_samples=100)
        assert logger.log_count > 0

    def test_log_aggregation(self):
        logger = TrainingLogger("test")
        logger.log_aggregation(1, num_clients=3, num_prototypes=10)
        assert logger.log_count > 0

    def test_log_knowledge_transfer(self):
        logger = TrainingLogger("test")
        logger.log_knowledge_transfer(1, num_synthesized=5)
        assert logger.log_count > 0

    def test_log_personalization(self):
        logger = TrainingLogger("test")
        logger.log_personalization(1, num_personalized=10)
        assert logger.log_count > 0

    def test_log_evaluation(self):
        logger = TrainingLogger("test")
        logger.log_evaluation(1, {"accuracy": 0.9})
        assert logger.log_count > 0

    def test_log_checkpoint(self):
        logger = TrainingLogger("test")
        logger.log_checkpoint(1, "/tmp/ckpt.pt")
        assert logger.log_count > 0

    def test_log_error(self):
        logger = TrainingLogger("test")
        logger.log_error("something went wrong", round_id=1)
        assert logger.log_count > 0

    def test_log_warning(self):
        logger = TrainingLogger("test")
        logger.log_warning("warning")
        assert logger.log_count > 0

    def test_log_synchronization(self):
        logger = TrainingLogger("test")
        logger.log_synchronization(1, num_clients=3)
        assert logger.log_count > 0

    def test_get_recent_logs(self):
        logger = TrainingLogger("test")
        logger.log_round_start(1, 3)
        logger.log_round_end(1)
        recent = logger.get_recent_logs(1)
        assert len(recent) == 1

    def test_get_logs_by_level(self):
        logger = TrainingLogger("test")
        logger.log_error("err")
        errs = logger.get_logs_by_level("error")
        assert len(errs) == 1

    def test_summary(self):
        logger = TrainingLogger("test")
        logger.log_error("err")
        logger.log_warning("warn")
        s = logger.summary()
        assert s["errors"] == 1
        assert s["warnings"] == 1

    def test_clear(self):
        logger = TrainingLogger("test")
        logger.log_round_start(1, 3)
        logger.clear()
        assert logger.log_count == 0


# ============================================================
# Test: monitor.py
# ============================================================


class TestMonitor:
    def test_record(self):
        m = ResourceMonitor()
        m.record(round_id=1, training_time=1.0, prototype_count=10, client_count=3)
        assert m.round_count == 1

    def test_total_elapsed(self):
        m = ResourceMonitor()
        assert m.total_elapsed >= 0

    def test_average_prototype_count(self):
        m = ResourceMonitor()
        m.record(1, prototype_count=10)
        m.record(2, prototype_count=20)
        assert m.average_prototype_count() == 15.0

    def test_average_client_count(self):
        m = ResourceMonitor()
        m.record(1, client_count=3)
        m.record(2, client_count=5)
        assert m.average_client_count() == 4.0

    def test_totals(self):
        m = ResourceMonitor()
        m.record(
            1,
            training_time=1.0,
            communication_time=0.5,
            aggregation_time=0.3,
            personalization_time=0.2,
            evaluation_time=0.1,
            payload_size_bytes=1000,
        )
        assert m.total_training_time() == 1.0
        assert m.total_communication_time() == 0.5
        assert m.total_aggregation_time() == 0.3
        assert m.total_personalization_time() == 0.2
        assert m.total_evaluation_time() == 0.1
        assert m.total_payload_bytes() == 1000

    def test_average_payload_bytes(self):
        m = ResourceMonitor()
        m.record(1, payload_size_bytes=1000)
        m.record(2, payload_size_bytes=2000)
        assert m.average_payload_bytes() == 1500.0

    def test_statistics(self):
        m = ResourceMonitor()
        m.record(1, training_time=1.0, prototype_count=10, client_count=3)
        stats = m.statistics()
        assert stats["round_count"] == 1

    def test_get_measurements(self):
        m = ResourceMonitor()
        m.record(1, training_time=1.0)
        meas = m.get_measurements(1)
        assert meas is not None
        assert meas["round_id"] == 1

    def test_get_measurements_missing(self):
        m = ResourceMonitor()
        assert m.get_measurements(99) is None

    def test_clear(self):
        m = ResourceMonitor()
        m.record(1)
        m.clear()
        assert m.round_count == 0

    def test_empty_averages(self):
        m = ResourceMonitor()
        assert m.average_prototype_count() == 0.0
        assert m.average_client_count() == 0.0
        assert m.average_payload_bytes() == 0.0
        assert m.total_payload_bytes() == 0


# ============================================================
# Test: registry.py
# ============================================================


class TestTrainingRegistry:
    def test_optimizer_registration(self):
        TrainingRegistry.register_optimizer("test_opt", optim.SGD)
        assert "test_opt" in TrainingRegistry.list_optimizers()

    def test_optimizer_get(self):
        cls = TrainingRegistry.get_optimizer("adam")
        assert cls == optim.Adam

    def test_optimizer_get_unknown(self):
        with pytest.raises(ValueError):
            TrainingRegistry.get_optimizer("unknown")

    def test_scheduler_registration(self):
        TrainingRegistry.register_scheduler("test_sched", optim.lr_scheduler.StepLR)
        assert "test_sched" in TrainingRegistry.list_schedulers()

    def test_scheduler_get(self):
        cls = TrainingRegistry.get_scheduler("step_lr")
        assert cls == optim.lr_scheduler.StepLR

    def test_scheduler_get_unknown(self):
        with pytest.raises(ValueError):
            TrainingRegistry.get_scheduler("unknown")

    def test_loss_registration(self):
        TrainingRegistry.register_loss("test_loss", nn.CrossEntropyLoss)
        assert "test_loss" in TrainingRegistry.list_losses()

    def test_loss_get(self):
        with pytest.raises(ValueError):
            TrainingRegistry.get_loss("unknown")

    def test_metric_registration(self):
        TrainingRegistry.register_metric("test_metric", lambda x: x)
        assert "test_metric" in TrainingRegistry.list_metrics()

    def test_metric_get(self):
        with pytest.raises(ValueError):
            TrainingRegistry.get_metric("unknown")


# ============================================================
# Test: optimizer.py
# ============================================================


class TestOptimizerFactory:
    def test_create_adam(self, simple_model):
        opt = OptimizerFactory.create(simple_model, "adam", lr=1e-3)
        assert isinstance(opt, optim.Adam)

    def test_create_sgd(self, simple_model):
        opt = OptimizerFactory.create(simple_model, "sgd", lr=0.01, momentum=0.9)
        assert isinstance(opt, optim.SGD)

    def test_create_adamw(self, simple_model):
        opt = OptimizerFactory.create(simple_model, "adamw", lr=1e-3)
        assert isinstance(opt, optim.AdamW)

    def test_create_fedprox(self, simple_model):
        opt = OptimizerFactory.create(simple_model, "fedprox", lr=1e-3, mu=0.01)
        assert isinstance(opt, FedProxOptimizer)

    def test_create_fedprox_static(self, simple_model):
        opt = OptimizerFactory.create_fedprox(simple_model, lr=1e-3, mu=0.01)
        assert isinstance(opt, FedProxOptimizer)

    def test_fedprox_step(self, simple_model):
        opt = FedProxOptimizer(simple_model.parameters(), lr=1e-3, mu=0.01)
        out = simple_model(torch.randn(2, 10))
        out.sum().backward()
        opt.step()

    def test_set_global_params(self, simple_model):
        opt = OptimizerFactory.create(simple_model, "fedprox")
        OptimizerFactory.set_global_params(opt, simple_model)
        for group in opt.param_groups:
            assert "global_params" in group

    def test_set_global_params_non_fedprox(self, simple_model):
        opt = OptimizerFactory.create(simple_model, "adam")
        OptimizerFactory.set_global_params(opt, simple_model)


# ============================================================
# Test: scheduler.py
# ============================================================


class TestSchedulerFactory:
    def test_create_step_lr(self, simple_model):
        opt = optim.SGD(simple_model.parameters(), lr=0.01)
        sched = SchedulerFactory.create(opt, "step_lr", step_size=10, gamma=0.1)
        assert isinstance(sched, optim.lr_scheduler.StepLR)

    def test_create_cosine(self, simple_model):
        opt = optim.SGD(simple_model.parameters(), lr=0.01)
        sched = SchedulerFactory.create(opt, "cosine_annealing", t_max=10)
        assert isinstance(sched, optim.lr_scheduler.CosineAnnealingLR)

    def test_create_reduce_on_plateau(self, simple_model):
        opt = optim.SGD(simple_model.parameters(), lr=0.01)
        sched = SchedulerFactory.create(opt, "reduce_on_plateau")
        assert isinstance(sched, optim.lr_scheduler.ReduceLROnPlateau)

    def test_create_with_warmup(self, simple_model):
        opt = optim.SGD(simple_model.parameters(), lr=0.01)
        wrapped = SchedulerFactory.create(
            opt, "cosine_annealing", warmup_steps=5, warmup_lr=1e-6, t_max=10
        )
        assert isinstance(wrapped, WarmupWrapper)

    def test_warmup_wrapper_step(self, simple_model):
        opt = optim.SGD(simple_model.parameters(), lr=0.01)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        wrapped = WarmupWrapper(sched, warmup_steps=3, warmup_lr=1e-6)
        wrapped.step()
        assert wrapped._step_count == 1

    def test_warmup_wrapper_after_warmup(self, simple_model):
        opt = optim.SGD(simple_model.parameters(), lr=0.01)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        wrapped = WarmupWrapper(sched, warmup_steps=3, warmup_lr=1e-6)
        for _ in range(5):
            wrapped.step()
        assert wrapped._step_count == 5

    def test_warmup_state_dict(self, simple_model):
        opt = optim.SGD(simple_model.parameters(), lr=0.01)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        wrapped = WarmupWrapper(sched, warmup_steps=3, warmup_lr=1e-6)
        wrapped.step()
        sd = wrapped.state_dict()
        assert "warmup_steps" in sd
        assert "step_count" in sd

    def test_warmup_load_state_dict(self, simple_model):
        opt = optim.SGD(simple_model.parameters(), lr=0.01)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        wrapped = WarmupWrapper(sched, warmup_steps=3, warmup_lr=1e-6)
        wrapped.step()
        sd = wrapped.state_dict()
        wrapped2 = WarmupWrapper(
            optim.lr_scheduler.CosineAnnealingLR(
                optim.SGD(simple_model.parameters(), lr=0.01), T_max=10
            ),
            warmup_steps=0,
        )
        wrapped2.load_state_dict(sd)
        assert wrapped2._step_count == 1


# ============================================================
# Test: communication.py
# ============================================================


class TestCommunication:
    def test_send_message(self):
        comm = CommunicationLayer()
        msg = comm.send("c1", "server", 1, "prototype", {"data": [1.0, 2.0]})
        assert msg.sender_id == "c1"
        assert msg.receiver_id == "server"
        assert comm.messages_sent_count == 1

    def test_receive_message(self):
        comm = CommunicationLayer()
        msg = comm.send("c1", "server", 1, "prototype", {"data": [1.0]})
        payload = comm.receive(msg)
        assert payload == {"data": [1.0]}
        assert comm.messages_received_count == 1

    def test_checksum_mismatch(self):
        comm = CommunicationLayer()
        msg = comm.send("c1", "server", 1, "prototype", {"data": [1.0]})
        msg.checksum = "bad"
        with pytest.raises(ValueError, match="Checksum mismatch"):
            comm.receive(msg)

    def test_send_batch(self):
        comm = CommunicationLayer()
        msgs = comm.send_batch(
            "c1",
            "server",
            1,
            [
                ("type1", {"a": 1}),
                ("type2", {"b": 2}),
            ],
        )
        assert len(msgs) == 2
        assert comm.messages_sent_count == 2

    def test_receive_batch(self):
        comm = CommunicationLayer()
        msgs = comm.send_batch(
            "c1",
            "server",
            1,
            [
                ("t1", {"a": 1}),
            ],
        )
        payloads = comm.receive_batch(msgs)
        assert len(payloads) == 1

    def test_latency(self):
        comm = CommunicationLayer()
        msg = comm.send("c1", "server", 1, "test", {"x": 1})
        comm.receive(msg)
        assert comm.messages_received_count == 1

    def test_max_latency(self):
        comm = CommunicationLayer()
        assert comm.max_latency == 0.0
        msg = comm.send("c1", "server", 1, "test", {"x": 1})
        comm.receive(msg)
        assert comm.messages_received_count == 1

    def test_compression_disabled(self):
        comm = CommunicationLayer()
        comm.set_compression(False)
        msg = comm.send("c1", "server", 1, "test", {"x": "y" * 1000})
        assert msg.compression == "none"

    def test_compression_enabled(self):
        comm = CommunicationLayer()
        comm.set_compression(True)
        msg = comm.send("c1", "server", 1, "test", {"x": "y" * 2000})
        assert msg.compression == "zlib"

    def test_checksum_disabled(self):
        comm = CommunicationLayer()
        comm.set_checksum(False)
        msg = comm.send("c1", "server", 1, "test", {"x": 1})
        assert msg.checksum == ""

    def test_payload_statistics_empty(self):
        comm = CommunicationLayer()
        stats = comm.payload_statistics()
        assert stats["total_messages"] == 0

    def test_payload_statistics(self):
        comm = CommunicationLayer()
        comm.send("c1", "server", 1, "test", {"x": 1})
        stats = comm.payload_statistics()
        assert stats["total_messages"] == 1
        assert stats["total_bytes"] > 0

    def test_message_verify_checksum(self):
        payload = {"data": [1.0, 2.0]}
        payload_str = json.dumps(payload, sort_keys=True)
        import hashlib

        expected = hashlib.sha256(payload_str.encode()).hexdigest()
        msg = Message(
            sender_id="c1",
            receiver_id="server",
            round_id=1,
            message_type="test",
            payload=payload,
            checksum=expected,
        )
        assert msg.verify_checksum()

    def test_clear(self):
        comm = CommunicationLayer()
        comm.send("c1", "server", 1, "test", {})
        comm.clear()
        assert comm.messages_sent_count == 0

    def test_total_bytes(self):
        comm = CommunicationLayer()
        comm.send("c1", "server", 1, "test", {"x": [1.0] * 100})
        assert comm.total_bytes_sent > 0
        assert comm.total_bytes_compressed > 0


# ============================================================
# Test: synchronization.py
# ============================================================


class TestSynchronization:
    def test_sync_model_state(self, simple_model):
        target = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
        sm = SynchronizationManager()
        result = sm.sync_model_state(simple_model, target)
        assert result["type"] == "model_state"
        assert sm.sync_count == 1

    def test_sync_model_state_dict(self, simple_model):
        target = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
        sm = SynchronizationManager()
        result = sm.sync_model_state_dict(simple_model.state_dict(), target)
        assert result["type"] == "model_state_dict"
        for p1, p2 in zip(simple_model.parameters(), target.parameters()):
            assert torch.equal(p1.data, p2.data)

    def test_sync_prototype_repository_export(self):
        sm = SynchronizationManager()
        source = MagicMock()
        target = MagicMock()
        source.export_state.return_value = {"protos": []}
        result = sm.sync_prototype_repository(source, target)
        assert result["exported"]

    def test_sync_prototype_repository_list(self):
        sm = SynchronizationManager()
        source = MagicMock(spec=["list_global_prototypes", "store_global_prototype"])
        target = MagicMock(spec=["list_global_prototypes", "store_global_prototype"])
        source.list_global_prototypes.return_value = []
        target.store_global_prototype.return_value = None
        result = sm.sync_prototype_repository(source, target)
        assert result["type"] == "prototype_repository"

    def test_sync_prototype_repository_error(self):
        sm = SynchronizationManager()
        source = object()
        target = object()
        with pytest.raises(ValueError):
            sm.sync_prototype_repository(source, target)

    def test_sync_knowledge_transfer(self):
        sm = SynchronizationManager()
        source = MagicMock()
        target = MagicMock()
        source.mapper = MagicMock()
        target.mapper = MagicMock()
        source.mapper.available_mappings.return_value = [("a", "b")]
        result = sm.sync_knowledge_transfer_state(source, target)
        assert result["type"] == "knowledge_transfer"

    def test_sync_personalized_prototypes(self):
        sm = SynchronizationManager()
        source = MagicMock()
        target = MagicMock()
        source.statistics.return_value = {}
        source.retrieve_all.return_value = []
        result = sm.sync_personalized_prototypes(source, target)
        assert result["type"] == "personalized_prototypes"

    def test_sync_optimizer_states(self, simple_model):
        sm = SynchronizationManager()
        opt1 = optim.SGD(simple_model.parameters(), lr=0.01)
        opt2 = optim.SGD(
            nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5)).parameters(),
            lr=0.01,
        )
        result = sm.sync_optimizer_states(opt1, opt2)
        assert result["type"] == "optimizer_state"

    def test_sync_all(self, simple_model):
        sm = SynchronizationManager()
        target = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
        opt = optim.SGD(simple_model.parameters(), lr=0.01)
        results = sm.sync_all(
            models={"source": simple_model, "target": target},
            repositories={},
            optimizers={},
        )
        assert "model" in results

    def test_history(self, simple_model):
        sm = SynchronizationManager()
        target = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
        sm.sync_model_state(simple_model, target)
        h = sm.history()
        assert len(h) == 1

    def test_clear(self, simple_model):
        sm = SynchronizationManager()
        target = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
        sm.sync_model_state(simple_model, target)
        sm.clear()
        assert sm.sync_count == 0


# ============================================================
# Test: checkpoint.py
# ============================================================


class TestCheckpoint:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = CheckpointManager(checkpoint_dir=tmpdir, experiment_id="test")
            state = TrainingState(experiment_id="test")
            path = cm.save(state, round_id=1)
            assert path.exists()

            loaded = cm.load("latest")
            assert loaded["round_id"] == 1
            assert loaded["experiment_id"] == "test"

    def test_save_best(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = CheckpointManager(checkpoint_dir=tmpdir, experiment_id="test")
            path = cm.save_best(round_id=1, metrics={"acc": 0.9})
            assert path.exists()
            assert cm.has_checkpoint("best")

    def test_save_latest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = CheckpointManager(checkpoint_dir=tmpdir, experiment_id="test")
            path = cm.save_latest(round_id=1, metrics={"acc": 0.9})
            assert path.exists()

    def test_latest_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = CheckpointManager(checkpoint_dir=tmpdir, experiment_id="test")
            cm.save_latest(round_id=1)
            cm.save_latest(round_id=2)
            latest = cm.latest_checkpoint()
            assert latest is not None
            assert latest["round_id"] == 2

    def test_has_checkpoint_false(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = CheckpointManager(checkpoint_dir=tmpdir, experiment_id="test")
            assert not cm.has_checkpoint("best")

    def test_list_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = CheckpointManager(checkpoint_dir=tmpdir, experiment_id="test")
            cm.save_latest(round_id=1)
            cps = cm.list_checkpoints()
            assert len(cps) == 1

    def test_resume(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = CheckpointManager(checkpoint_dir=tmpdir, experiment_id="test")
            cm.save_latest(round_id=5)
            checkpoint = cm.resume()
            assert checkpoint["round_id"] == 5

    def test_load_round_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = CheckpointManager(checkpoint_dir=tmpdir, experiment_id="test")
            with pytest.raises(FileNotFoundError):
                cm.load("latest")

    def test_enforce_max_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = CheckpointManager(
                checkpoint_dir=tmpdir, experiment_id="test", max_checkpoints=2
            )
            for i in range(5):
                cm.save_latest(round_id=i)
            assert len(cm._checkpoints) <= 2

    def test_clear(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = CheckpointManager(checkpoint_dir=tmpdir, experiment_id="test")
            cm.save_latest(round_id=1)
            cm.clear()
            assert not cm._checkpoints

    def test_checkpoint_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = CheckpointManager(checkpoint_dir=tmpdir, experiment_id="test")
            assert cm.checkpoint_dir() == str(cm._base_dir)

    def test_save_with_full_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = CheckpointManager(checkpoint_dir=tmpdir, experiment_id="test")
            state = TrainingState(experiment_id="test")
            path = cm.save(
                state=state,
                round_id=1,
                server_state={"key": "val"},
                client_states={"c1": {"loss": 0.5}},
                model_state={"layer.weight": torch.tensor([1.0])},
                optimizer_state={"param_groups": []},
                prototype_repo_state={"protos": []},
                personalization_state={"data": {}},
                extra={"note": "test"},
            )
            assert path.exists()

    def test_save_best_with_kwargs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = CheckpointManager(checkpoint_dir=tmpdir, experiment_id="test")
            path = cm.save_best(
                round_id=1,
                metrics={"acc": 0.9},
                model_state={"w": torch.tensor([1.0])},
            )
            loaded = torch.load(path, weights_only=False)
            assert "model_state" in loaded


# ============================================================
# Test: trainer.py
# ============================================================


class TestTrainer:
    def test_train_one_epoch(self, simple_model, simple_dataloader):
        opt = optim.SGD(simple_model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()
        trainer = Trainer(simple_model, loss_fn, opt)
        metrics = trainer.train_one_epoch(simple_dataloader, epoch=0)
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert metrics["epoch"] == 0

    def test_validate(self, simple_model, simple_dataloader):
        opt = optim.SGD(simple_model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()
        trainer = Trainer(simple_model, loss_fn, opt)
        metrics = trainer.validate(simple_dataloader)
        assert "loss" in metrics
        assert "accuracy" in metrics

    def test_predict(self, simple_model, simple_dataloader):
        opt = optim.SGD(simple_model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()
        trainer = Trainer(simple_model, loss_fn, opt)
        outputs, targets = trainer.predict(simple_dataloader)
        assert len(outputs) > 0
        assert len(targets) > 0

    def test_train_local(self, simple_model, simple_dataloader):
        opt = optim.SGD(simple_model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()
        trainer = Trainer(simple_model, loss_fn, opt)
        results = trainer.train_local(simple_dataloader, epochs=2)
        assert len(results["epoch_metrics"]) == 2

    def test_gradient_clipping(self, simple_model):
        opt = optim.SGD(simple_model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()
        trainer = Trainer(simple_model, loss_fn, opt, max_grad_norm=1.0)
        inputs = torch.randn(2, 10)
        targets = torch.randint(0, 5, (2,))
        loss, acc = trainer._train_step(inputs, targets)
        assert loss >= 0

    def test_device_property(self, simple_model):
        opt = optim.SGD(simple_model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()
        trainer = Trainer(simple_model, loss_fn, opt)
        assert trainer.device.type == "cpu"

    def test_model_property(self, simple_model):
        opt = optim.SGD(simple_model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()
        trainer = Trainer(simple_model, loss_fn, opt)
        assert trainer.model is simple_model

    def test_optimizer_property(self, simple_model):
        opt = optim.SGD(simple_model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()
        trainer = Trainer(simple_model, loss_fn, opt)
        assert trainer.optimizer is opt

    def test_prepare_batch_2_elements(self, simple_model):
        opt = optim.SGD(simple_model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()
        trainer = Trainer(simple_model, loss_fn, opt)
        inputs, targets = trainer._prepare_batch(
            (torch.randn(2, 10), torch.randint(0, 5, (2,)))
        )
        assert inputs.shape == (2, 10)

    def test_prepare_batch_3_elements(self, simple_model):
        opt = optim.SGD(simple_model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()
        trainer = Trainer(simple_model, loss_fn, opt)
        inputs, targets = trainer._prepare_batch(
            (torch.randn(2, 10), torch.randint(0, 5, (2,)), torch.randn(2))
        )
        assert inputs.shape == (2, 10)

    def test_prepare_batch_invalid(self, simple_model):
        opt = optim.SGD(simple_model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()
        trainer = Trainer(simple_model, loss_fn, opt)
        with pytest.raises(ValueError):
            trainer._prepare_batch((torch.randn(2, 10),))


# ============================================================
# Test: local_training.py
# ============================================================


class TestLocalTraining:
    def test_train(self, simple_model, simple_dataloader):
        opt = optim.SGD(simple_model.parameters(), lr=0.01)
        lt = LocalTraining(simple_model, nn.CrossEntropyLoss(), opt)
        results = lt.train(simple_dataloader, epochs=2)
        assert "epoch_metrics" in results
        assert len(lt.history) == 1

    def test_validate(self, simple_model, simple_dataloader):
        opt = optim.SGD(simple_model.parameters(), lr=0.01)
        lt = LocalTraining(simple_model, nn.CrossEntropyLoss(), opt)
        metrics = lt.validate(simple_dataloader)
        assert "loss" in metrics

    def test_predict(self, simple_model, simple_dataloader):
        opt = optim.SGD(simple_model.parameters(), lr=0.01)
        lt = LocalTraining(simple_model, nn.CrossEntropyLoss(), opt)
        outputs, targets = lt.predict(simple_dataloader)
        assert len(outputs) > 0

    def test_get_embeddings(self, simple_model, simple_dataloader):
        opt = optim.SGD(simple_model.parameters(), lr=0.01)
        lt = LocalTraining(simple_model, nn.CrossEntropyLoss(), opt)
        embs = lt.get_embeddings(simple_dataloader)
        assert isinstance(embs, dict)

    def test_clear_history(self, simple_model, simple_dataloader):
        opt = optim.SGD(simple_model.parameters(), lr=0.01)
        lt = LocalTraining(simple_model, nn.CrossEntropyLoss(), opt)
        lt.train(simple_dataloader)
        lt.clear_history()
        assert lt.history == []

    def test_trainer_property(self, simple_model):
        opt = optim.SGD(simple_model.parameters(), lr=0.01)
        lt = LocalTraining(simple_model, nn.CrossEntropyLoss(), opt)
        assert lt.trainer is not None

    def test_model_property(self, simple_model):
        opt = optim.SGD(simple_model.parameters(), lr=0.01)
        lt = LocalTraining(simple_model, nn.CrossEntropyLoss(), opt)
        assert lt.model is simple_model


# ============================================================
# Test: client.py
# ============================================================


class TestClient:
    @pytest.fixture
    def client(self, simple_model, simple_dataloader):
        opt = optim.SGD(simple_model.parameters(), lr=0.01)
        from app.training.client import Client

        c = Client(
            client_id="test_client",
            model=simple_model,
            loss_fn=nn.CrossEntropyLoss(),
            optimizer=opt,
        )
        c.load_local_dataset(
            simple_dataloader,
            modalities={"image", "text"},
            all_modalities={"image", "text"},
        )
        return c

    def test_client_id(self, client):
        assert client.client_id == "test_client"

    def test_train(self, client):
        results = client.train(epochs=1)
        assert "final_loss" in results
        assert client.state.epochs_completed == 1

    def test_train_no_dataset(self, simple_model):
        opt = optim.SGD(simple_model.parameters(), lr=0.01)
        c = Client("c1", simple_model, nn.CrossEntropyLoss(), opt)
        with pytest.raises(ValueError):
            c.train()

    def test_generate_prototypes_no_generator(self, client):
        with pytest.raises(ValueError):
            client.generate_prototypes()

    def test_generate_prototypes_no_dataset(self, simple_model):
        opt = optim.SGD(simple_model.parameters(), lr=0.01)
        c = Client("c1", simple_model, nn.CrossEntropyLoss(), opt)
        with pytest.raises(ValueError):
            c.generate_prototypes()

    def test_upload_results(self, client):
        packages = client.upload_results(round_id=1)
        assert isinstance(packages, list)
        assert client.state.current_round == 1

    def test_receive_updates(self, client):
        client.receive_updates(
            global_prototypes=[],
            transferred=[],
        )

    def test_personalize_no_selector(self, client):
        with pytest.raises(ValueError):
            client.personalize()

    def test_update_model(self, client, simple_model):
        state_dict = simple_model.state_dict()
        client.update_model(state_dict)

    def test_get_model_state(self, client):
        state = client.get_model_state()
        assert isinstance(state, dict)

    def test_get_optimizer_state(self, client):
        state = client.get_optimizer_state()
        assert isinstance(state, dict)

    def test_set_optimizer_state(self, client):
        state = client.get_optimizer_state()
        client.set_optimizer_state(state)

    def test_set_current_round(self, client):
        client.set_current_round(5)
        assert client.state.current_round == 5

    def test_to_dict(self, client):
        d = client.to_dict()
        assert d["client_id"] == "test_client"

    def test_state_property(self, client):
        assert client.state.client_id == "test_client"

    def test_profile_property(self, client):
        assert client.profile.client_id == "test_client"

    def test_prototype_memory_property(self, client):
        assert client.prototype_memory is not None

    def test_collect_class_ids(self, client):
        from app.prototypes.prototype import Prototype

        local = [Prototype(embedding=torch.randn(10), class_id=1, modality="image")]
        global_p = [
            AggregatedPrototype(
                class_id=2,
                modality="text",
                prototype_vector=[0.0] * 10,
                embedding_dim=10,
                sample_count=1,
                confidence=0.9,
            )
        ]
        transferred = [
            InferenceOutput(
                modality="text",
                class_id=3,
                prototype_vector=[0.0] * 10,
                embedding_dim=10,
                confidence=0.8,
                source_modality="image",
                path=["image", "text"],
            )
        ]
        ids = client._collect_class_ids(local, global_p, transferred)
        assert 1 in ids
        assert 2 in ids
        assert 3 in ids

    def test_personalize_with_components(self, simple_model, simple_dataloader):
        from app.personalization.prototype_selector import (
            PrototypeSelector,
        )
        from app.personalization.fusion_engine import FusionEngine
        from app.personalization.weighting import WeightCalculator

        opt = optim.SGD(simple_model.parameters(), lr=0.01)
        selector = PrototypeSelector(confidence_threshold=0.0)
        wc = WeightCalculator(
            strategy="fixed",
            fixed_weights={
                "local": 0.5,
                "global": 0.5,
                "cross_modal": 1.0,
            },
        )
        fusion = FusionEngine(strategy="weighted_sum", weight_calculator=wc)

        c = Client(
            client_id="test",
            model=simple_model,
            loss_fn=nn.CrossEntropyLoss(),
            optimizer=opt,
            prototype_selector=selector,
            fusion_engine=fusion,
        )
        c.load_local_dataset(
            simple_dataloader,
            modalities={"image"},
            all_modalities={"image", "text"},
        )

        local = [
            Prototype(
                embedding=torch.randn(10),
                class_id=1,
                modality="image",
            )
        ]
        global_p = [
            AggregatedPrototype(
                class_id=1,
                modality="image",
                prototype_vector=[0.5] * 10,
                embedding_dim=10,
                sample_count=5,
                confidence=0.8,
            )
        ]
        transferred = [
            InferenceOutput(
                modality="text",
                class_id=1,
                prototype_vector=[0.3] * 10,
                embedding_dim=10,
                confidence=0.7,
                source_modality="image",
                path=["image", "text"],
            )
        ]

        result = c.personalize(
            local_prototypes=local,
            global_prototypes=global_p,
            transferred=transferred,
        )
        assert len(result) > 0
        for pp in result:
            assert pp.personalized_prototype is not None


# ============================================================
# Test: server.py
# ============================================================


class TestServer:
    @pytest.fixture
    def server(self):
        from app.training.server import Server

        return Server()

    def test_server_id(self, server):
        assert server.server_id == "server"

    def test_collect_client_results(self, server):
        results = {"c1": [MagicMock(), MagicMock()]}
        total = server.collect_client_results(results)
        assert total == 2

    def test_aggregate_prototypes_no_aggregator(self, server):
        with pytest.raises(ValueError):
            server.aggregate_prototypes(round_id=1)

    def test_broadcast_global_prototypes(self, server):
        protos = server.broadcast_global_prototypes()
        assert protos == []

    def test_broadcast_synthesized_prototypes(self, server):
        protos = server.broadcast_synthesized_prototypes()
        assert protos == []

    def test_get_aggregated_for_modality(self, server):
        result = server.get_aggregated_for_modality("image", 1)
        assert result is None

    def test_get_state_dict(self, server):
        sd = server.get_state_dict()
        assert sd["server_id"] == "server"

    def test_load_state_dict(self, server):
        sd = server.get_state_dict()
        server.load_state_dict(sd)

    def test_knowledge_transfer_no_engine(self, server):
        syn, inf = server.run_knowledge_transfer(1)
        assert syn == []
        assert inf == []

    def test_knowledge_transfer_with_engine(self):
        from app.training.server import Server

        inference = MagicMock()
        inference.infer_missing_modalities.return_value = []
        server = Server(inference_engine=inference)
        syn, inf = server.run_knowledge_transfer(1, all_modalities={"image"})
        assert syn == []
        assert inf == []

    def test_state_property(self, server):
        assert server.state.current_round == 0

    def test_repository_property(self, server):
        assert server.repository is not None

    def test_aggregated_prototypes_property(self, server):
        assert server.aggregated_prototypes == {}


# ============================================================
# Test: evaluator.py
# ============================================================


class TestEvaluator:
    def test_evaluate_basic(self):
        ev = Evaluator()
        metrics = ev.evaluate(
            loss=0.1, accuracy=0.9, precision=0.8, recall=0.7, f1_score=0.75
        )
        assert metrics["loss"] == 0.1
        assert metrics["accuracy"] == 0.9

    def test_evaluate_classification(self):
        ev = Evaluator()
        outputs = [torch.tensor([[0.1, 0.9], [0.8, 0.2]])]
        targets = [torch.tensor([1, 0])]
        metrics = ev.evaluate_classification(outputs, targets)
        assert metrics["accuracy"] == 1.0

    def test_evaluate_classification_empty(self):
        ev = Evaluator()
        metrics = ev.evaluate_classification([], [])
        assert metrics["accuracy"] == 0.0

    def test_evaluate_prototypes(self):
        ev = Evaluator()
        protos = [
            AggregatedPrototype(
                class_id=1,
                modality="image",
                prototype_vector=[0.5] * 10,
                embedding_dim=10,
                sample_count=5,
                confidence=0.8,
            )
        ]
        metrics = ev.evaluate_prototypes(protos)
        assert metrics["prototype_count"] == 1.0
        assert metrics["avg_confidence"] == 0.8

    def test_evaluate_prototypes_empty(self):
        ev = Evaluator()
        metrics = ev.evaluate_prototypes([])
        assert metrics["prototype_count"] == 0.0

    def test_evaluate_personalization(self):
        ev = Evaluator()
        pps = [
            PersonalizedPrototype(
                client_id="c1",
                class_id=1,
                modality="image",
                personalized_prototype=[0.5] * 10,
                global_prototype=[0.3] * 10,
                local_prototype=[0.7] * 10,
                fusion_weights={"local": 0.5, "global": 0.5},
            )
        ]
        metrics = ev.evaluate_personalization(pps)
        assert "personalization_gain" in metrics

    def test_evaluate_personalization_empty(self):
        ev = Evaluator()
        metrics = ev.evaluate_personalization([])
        assert metrics["personalization_gain"] == 0.0

    def test_compute_communication_statistics(self):
        ev = Evaluator()
        stats = ev.compute_communication_statistics(
            total_bytes_sent=1000,
            total_messages=10,
            avg_latency=0.05,
            compression_ratio=0.5,
        )
        assert stats["total_messages"] == 10.0

    def test_evaluate_all(self):
        ev = Evaluator()
        metrics = ev.evaluate_all(
            round_id=1,
            loss=0.1,
            accuracy=0.9,
            personalized_prototypes=[
                PersonalizedPrototype(
                    client_id="c1",
                    class_id=1,
                    modality="image",
                    personalized_prototype=[0.5] * 10,
                    global_prototype=[0.3] * 10,
                    fusion_weights={"local": 1.0},
                )
            ],
        )
        assert "accuracy" in metrics
        assert len(ev.eval_history) == 1

    def test_get_round_metrics(self):
        ev = Evaluator()
        ev.evaluate_all(round_id=1, loss=0.1, accuracy=0.9)
        metrics = ev.get_round_metrics(1)
        assert metrics is not None
        assert metrics["accuracy"] == 0.9

    def test_get_round_metrics_none(self):
        ev = Evaluator()
        assert ev.get_round_metrics(99) is None

    def test_clear(self):
        ev = Evaluator()
        ev.evaluate_all(round_id=1)
        ev.clear()
        assert ev.eval_history == []


# ============================================================
# Test: round_manager.py
# ============================================================


class TestRoundManager:
    @pytest.fixture
    def round_manager(self, simple_model, simple_dataloader):
        from unittest.mock import MagicMock

        from app.federated.aggregator import FederatedAggregator
        from app.personalization.fusion_engine import FusionEngine
        from app.personalization.prototype_selector import (
            PrototypeSelector,
        )
        from app.personalization.weighting import WeightCalculator
        from app.prototypes.generator import PrototypeGenerator
        from app.training.client import Client
        from app.training.round_manager import RoundManager
        from app.training.server import Server
        from app.training.state import TrainingState

        mock_aggregator = MagicMock(spec=FederatedAggregator)
        mock_aggregator.run_round.return_value = {}
        server = Server(federated_aggregator=mock_aggregator)

        selector = PrototypeSelector(confidence_threshold=0.0)
        wc = WeightCalculator(strategy="fixed")
        fusion = FusionEngine(strategy="weighted_sum", weight_calculator=wc)

        clients = []
        for i in range(2):
            client_model = copy.deepcopy(simple_model)
            opt = optim.SGD(client_model.parameters(), lr=0.01)
            proto_gen = PrototypeGenerator(strategy="centroid")
            c = Client(
                client_id=f"client_{i}",
                model=client_model,
                loss_fn=nn.CrossEntropyLoss(),
                optimizer=opt,
                prototype_generator=proto_gen,
                prototype_selector=selector,
                fusion_engine=fusion,
            )
            c.load_local_dataset(
                simple_dataloader, modalities={"image"}, all_modalities={"image"}
            )
            clients.append(c)

        state = TrainingState(experiment_id="test", total_rounds=5)
        return RoundManager(server=server, clients=clients, training_state=state)

    def test_run_round(self, round_manager):
        result = round_manager.run_round(round_id=1)
        assert result["round_id"] == 1
        assert "metrics" in result
        assert "num_clients" in result

    def test_should_stop_early(self, round_manager):
        assert not round_manager.should_stop()

    def test_should_stop_max_rounds(self):
        from app.training.round_manager import RoundManager
        from app.training.server import Server
        from app.training.state import TrainingState

        state = TrainingState(experiment_id="test", total_rounds=1)
        rm = RoundManager(server=Server(), clients=[], training_state=state)
        state.current_round = 1
        assert rm.should_stop()

    def test_monitor_property(self, round_manager):
        assert round_manager.monitor is not None

    def test_evaluator_property(self, round_manager):
        assert round_manager.evaluator is not None

    def test_metric_recording_property(self, round_manager):
        assert round_manager.metric_recording is not None


# ============================================================
# Test: coordinator.py
# ============================================================


class TestCoordinator:
    @pytest.fixture
    def coordinator(self, simple_model, simple_dataloader):
        from app.training.coordinator import Coordinator
        from app.training.server import Server
        from app.training.client import Client
        from app.training.state import TrainingState
        from app.prototypes.generator import PrototypeGenerator

        server = Server()
        clients = []
        for i in range(2):
            client_model = copy.deepcopy(simple_model)
            opt = optim.SGD(client_model.parameters(), lr=0.01)
            proto_gen = PrototypeGenerator(strategy="centroid")
            c = Client(
                client_id=f"client_{i}",
                model=client_model,
                loss_fn=nn.CrossEntropyLoss(),
                optimizer=opt,
                prototype_generator=proto_gen,
            )
            c.load_local_dataset(
                simple_dataloader, modalities={"image"}, all_modalities={"image"}
            )
            clients.append(c)

        state = TrainingState(experiment_id="test", total_rounds=3)
        for c in clients:
            state.register_client(c.client_id)

        return Coordinator(server=server, clients=clients, training_state=state)

    def test_initialize(self, coordinator):
        coordinator.initialize()
        assert coordinator.state.phase == "initialized"

    def test_run(self, coordinator):
        state = coordinator.run()
        assert state.phase == "completed"

    def test_finalize(self, coordinator):
        coordinator.initialize()
        coordinator.finalize()
        assert coordinator.state.phase == "completed"

    def test_state_property(self, coordinator):
        assert coordinator.state is not None

    def test_round_manager_property(self, coordinator):
        assert coordinator.round_manager is not None

    def test_event_dispatcher_property(self, coordinator):
        assert coordinator.event_dispatcher is not None

    def test_hook_manager_property(self, coordinator):
        assert coordinator.hook_manager is not None

    def test_default_hooks_installed(self, coordinator):
        assert coordinator.hook_manager.total_hooks() > 0

    def test_run_with_early_stopping(self, simple_model, simple_dataloader):
        from app.training.coordinator import Coordinator
        from app.training.server import Server
        from app.training.client import Client
        from app.training.state import TrainingState
        from app.training.callbacks import EarlyStopping
        from app.prototypes.generator import PrototypeGenerator

        server = Server()
        clients = []
        for i in range(2):
            client_model = copy.deepcopy(simple_model)
            opt = optim.SGD(client_model.parameters(), lr=0.01)
            proto_gen = PrototypeGenerator(strategy="centroid")
            c = Client(
                client_id=f"client_{i}",
                model=client_model,
                loss_fn=nn.CrossEntropyLoss(),
                optimizer=opt,
                prototype_generator=proto_gen,
            )
            c.load_local_dataset(
                simple_dataloader, modalities={"image"}, all_modalities={"image"}
            )
            clients.append(c)

        state = TrainingState(experiment_id="test", total_rounds=10)
        for c in clients:
            state.register_client(c.client_id)

        es = EarlyStopping(patience=1, metric="loss", mode="min")
        coord = Coordinator(
            server=server,
            clients=clients,
            training_state=state,
            early_stopping=es,
        )
        state = coord.run()
        assert state.phase == "completed"


# ============================================================
# Test: experiment.py
# ============================================================


class TestExperiment:
    def test_create_experiment(self, config):
        exp = Experiment(experiment_id="test_exp", config=config)
        assert exp.experiment_id == "test_exp"

    def test_validate_config_missing(self):
        with pytest.raises(ValueError):
            Experiment(experiment_id="bad", config={})

    def test_initialize(self, config):
        exp = Experiment(experiment_id="test", config=config)
        exp.initialize()
        assert exp.state is not None
        assert exp.state.phase == "initialized"

    def test_run(self, config):
        exp = Experiment(experiment_id="test", config=config)
        exp.initialize()
        state = exp.run()
        assert state.phase == "completed"

    def test_resume(self, config):
        exp = Experiment(experiment_id="test", config=config)
        exp.initialize()
        state = exp.resume()
        assert state.phase == "completed"

    def test_logger_property(self, config):
        exp = Experiment(experiment_id="test", config=config)
        assert exp.logger is not None

    def test_run_uninitialized(self, config):
        exp = Experiment(experiment_id="test", config=config)
        with pytest.raises(ValueError):
            exp.run()

    def test_state_uninitialized(self, config):
        exp = Experiment(experiment_id="test", config=config)
        with pytest.raises(ValueError):
            _ = exp.state

    def test_cleanup(self, config):
        exp = Experiment(experiment_id="test", config=config)
        exp.initialize()
        exp.cleanup()

    def test_create_models(self, config):
        exp = Experiment(experiment_id="test", config=config)
        exp.initialize()
        assert hasattr(exp, "_model")

    def test_config_with_early_stopping(self, config):
        config["early_stopping"] = {
            "enabled": True,
            "patience": 5,
            "metric": "accuracy",
        }
        exp = Experiment(experiment_id="test", config=config)
        exp.initialize()
        assert exp.coordinator._early_stopping is not None

    def test_config_with_dataset_name(self, config):
        config["dataset"] = {"name": "nonexistent", "batch_size": 16}
        exp = Experiment(experiment_id="test", config=config)
        exp.initialize()


# ============================================================
# Test: factory.py
# ============================================================


class TestTrainingFactory:
    def test_create_experiment(self, config):
        exp = TrainingFactory.create_experiment("test", config)
        assert isinstance(exp, Experiment)

    def test_create_coordinator(self, simple_model, simple_dataloader):
        from app.training.server import Server
        from app.training.client import Client
        from app.training.state import TrainingState

        server = Server()
        clients = [
            Client(
                client_id="c1",
                model=simple_model,
                loss_fn=nn.CrossEntropyLoss(),
                optimizer=optim.SGD(simple_model.parameters(), lr=0.01),
            )
        ]
        state = TrainingState(experiment_id="test")
        coord = TrainingFactory.create_coordinator(server, clients, state)
        assert isinstance(coord, Coordinator)

    def test_create_client(self, simple_model):
        opt = optim.SGD(simple_model.parameters(), lr=0.01)
        client = TrainingFactory.create_client(
            "c1", simple_model, nn.CrossEntropyLoss(), opt
        )
        assert isinstance(client, Client)
        assert client.client_id == "c1"

    def test_create_server(self):
        server = TrainingFactory.create_server()
        assert isinstance(server, Server)

    def test_create_evaluator(self):
        evaluator = TrainingFactory.create_evaluator()
        from app.training.evaluator import Evaluator

        assert isinstance(evaluator, Evaluator)

    def test_create_checkpoint_manager(self):
        cm = TrainingFactory.create_checkpoint_manager(
            checkpoint_dir=tempfile.mkdtemp(), experiment_id="test"
        )
        from app.training.checkpoint import CheckpointManager

        assert isinstance(cm, CheckpointManager)

    def test_create_monitor(self):
        m = TrainingFactory.create_monitor()
        from app.training.monitor import ResourceMonitor

        assert isinstance(m, ResourceMonitor)

    def test_create_logger(self):
        logger = TrainingFactory.create_logger(experiment_id="test")
        assert logger.experiment_id == "test"

    def test_create_event_dispatcher(self):
        d = TrainingFactory.create_event_dispatcher()
        from app.training.events import EventDispatcher

        assert isinstance(d, EventDispatcher)

    def test_create_hook_manager(self):
        hm = TrainingFactory.create_hook_manager()
        from app.training.hooks import HookManager

        assert isinstance(hm, HookManager)

    def test_create_from_config(self, config):
        result = TrainingFactory.create_from_config("test", config)
        assert "coordinator" in result
        assert "server" in result
        assert "clients" in result
        assert "state" in result
        assert "logger" in result
        assert "checkpoint_manager" in result
        assert "model" in result
        assert len(result["clients"]) == 3


# ============================================================
# Test: __init__.py exports
# ============================================================


class TestInitExports:
    def test_all_exports_exist(self):
        from app.training import (
            TrainingState,
            ClientState,
            ServerState,
            Trainer,
            LocalTraining,
            Client,
            Server,
            Coordinator,
            RoundManager,
            Experiment,
            Evaluator,
            CheckpointManager,
            CommunicationLayer,
            Message,
            SynchronizationManager,
            OptimizerFactory,
            FedProxOptimizer,
            SchedulerFactory,
            WarmupWrapper,
            Event,
            EventType,
            EventDispatcher,
            Hook,
            HookContext,
            HookManager,
            EarlyStopping,
            CheckpointSaving,
            LoggingHook,
            LRUpdateHook,
            MetricRecording,
            ResourceMonitor,
            TrainingLogger,
            TrainingRegistry,
            TrainingFactory,
            Timer,
            compute_accuracy,
            clip_gradients,
            compute_grad_norm,
            count_parameters,
            flatten_model_state,
            unflatten_model_state,
            to_device,
            merge_configs,
            validate_config,
        )

        assert TrainingState is not None
        assert Client is not None
        assert Server is not None
        assert Coordinator is not None
        assert TrainingFactory is not None


# Import at end to avoid circular issues
from app.training.state import TrainingState, ClientState, ServerState
from app.training.utils import (
    compute_accuracy,
    to_device,
    count_parameters,
    compute_grad_norm,
    clip_gradients,
    flatten_model_state,
    unflatten_model_state,
    Timer,
    merge_configs,
    validate_config,
)
from app.training.events import Event, EventType, EventDispatcher
from app.training.hooks import Hook, HookContext, HookManager
from app.training.callbacks import (
    EarlyStopping,
    CheckpointSaving,
    LoggingHook,
    LRUpdateHook,
    MetricRecording,
)
from app.training.logger import TrainingLogger
from app.training.monitor import ResourceMonitor
from app.training.registry import TrainingRegistry
from app.training.optimizer import OptimizerFactory, FedProxOptimizer
from app.training.scheduler import SchedulerFactory, WarmupWrapper
from app.training.communication import CommunicationLayer, Message
from app.training.synchronization import SynchronizationManager
from app.training.checkpoint import CheckpointManager
from app.training.trainer import Trainer
from app.training.local_training import LocalTraining
from app.training.client import Client
from app.training.server import Server
from app.training.evaluator import Evaluator
from app.training.round_manager import RoundManager
from app.training.coordinator import Coordinator
from app.training.experiment import Experiment
from app.training.factory import TrainingFactory
