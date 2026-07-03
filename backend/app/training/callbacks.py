from __future__ import annotations

from typing import Any

from app.training.events import Event, EventDispatcher, EventType
from app.training.hooks import Hook, HookContext, HookManager
from app.training.logger import TrainingLogger


class EarlyStopping(Hook):
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        metric: str = "accuracy",
        mode: str = "max",
    ) -> None:
        self._patience = patience
        self._min_delta = min_delta
        self._metric = metric
        self._mode = mode
        self._best_value: float | None = None
        self._counter: int = 0
        self._stopped: bool = False

    @property
    def stopped(self) -> bool:
        return self._stopped

    def execute(self, context: HookContext) -> None:
        current = context.metrics.get(self._metric, 0.0)
        if self._best_value is None:
            self._best_value = current
            return
        improved = (
            (current > self._best_value + self._min_delta)
            if self._mode == "max"
            else (current < self._best_value - self._min_delta)
        )
        if improved:
            self._best_value = current
            self._counter = 0
        else:
            self._counter += 1
            if self._counter >= self._patience:
                self._stopped = True

    def reset(self) -> None:
        self._best_value = None
        self._counter = 0
        self._stopped = False


class CheckpointSaving(Hook):
    def __init__(
        self,
        checkpoint_manager: Any,
        interval: int = 1,
        save_best: bool = True,
        metric: str = "accuracy",
        mode: str = "max",
    ) -> None:
        self._checkpoint_manager = checkpoint_manager
        self._interval = interval
        self._save_best = save_best
        self._metric = metric
        self._mode = mode
        self._best_value: float | None = None

    def execute(self, context: HookContext) -> None:
        should_save = False
        if context.round_id % self._interval == 0:
            should_save = True
        if self._save_best:
            current = context.metrics.get(self._metric, 0.0)
            if self._mode == "max" and (
                self._best_value is None or current > self._best_value
            ):
                self._best_value = current
                should_save = True
            elif self._mode == "min" and (
                self._best_value is None or current < self._best_value
            ):
                self._best_value = current
                should_save = True
        if should_save:
            self._checkpoint_manager.save_latest(
                round_id=context.round_id,
                metrics=context.metrics,
            )
            if self._save_best:
                self._checkpoint_manager.save_best(
                    round_id=context.round_id,
                    metrics=context.metrics,
                )


class LoggingHook(Hook):
    def __init__(self, logger: TrainingLogger) -> None:
        self._logger = logger

    def execute(self, context: HookContext) -> None:
        hook_point = context.data.get("hook_point", "unknown")
        if hook_point == "before_round":
            self._logger.log_round_start(
                round_id=context.round_id,
                num_clients=context.data.get("num_clients", 0),
            )
        elif hook_point == "after_round":
            self._logger.log_round_end(
                round_id=context.round_id,
                metrics=context.metrics,
            )
        elif hook_point == "on_error":
            self._logger.log_error(
                message=str(context.error) if context.error else "Unknown error",
                round_id=context.round_id,
            )


class MetricRecording(Hook):
    def __init__(self) -> None:
        self._history: dict[str, list[tuple[int, float]]] = {}

    @property
    def history(self) -> dict[str, list[tuple[int, float]]]:
        return dict(self._history)

    def execute(self, context: HookContext) -> None:
        for key, value in context.metrics.items():
            if key not in self._history:
                self._history[key] = []
            self._history[key].append((context.round_id, value))

    def get_metric(self, name: str) -> list[tuple[int, float]]:
        return self._history.get(name, [])

    def latest(self, name: str) -> float | None:
        values = self._history.get(name)
        if not values:
            return None
        return values[-1][1]

    def reset(self) -> None:
        self._history.clear()


class LRUpdateHook(Hook):
    def __init__(self, scheduler: Any) -> None:
        self._scheduler = scheduler

    def execute(self, context: HookContext) -> None:
        if hasattr(self._scheduler, "step"):
            self._scheduler.step()
