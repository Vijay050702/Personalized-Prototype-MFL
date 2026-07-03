from __future__ import annotations

from typing import Any

from app.core.logging import logger
from app.training.callbacks import (
    CheckpointSaving,
    EarlyStopping,
    LoggingHook,
    LRUpdateHook,
    MetricRecording,
)
from app.training.client import Client
from app.training.events import EventDispatcher, EventType
from app.training.hooks import HookContext, HookManager
from app.training.logger import TrainingLogger
from app.training.monitor import ResourceMonitor
from app.training.round_manager import RoundManager
from app.training.server import Server
from app.training.state import TrainingState


class Coordinator:
    def __init__(
        self,
        server: Server,
        clients: list[Client],
        training_state: TrainingState,
        event_dispatcher: EventDispatcher | None = None,
        hook_manager: HookManager | None = None,
        logger_instance: TrainingLogger | None = None,
        monitor: ResourceMonitor | None = None,
        checkpoint_manager: Any | None = None,
        early_stopping: EarlyStopping | None = None,
    ) -> None:
        self._server = server
        self._clients = clients
        self._state = training_state
        self._event_dispatcher = event_dispatcher or EventDispatcher()
        self._hook_manager = hook_manager or HookManager()
        self._logger = logger_instance or TrainingLogger()
        self._monitor = monitor or ResourceMonitor()
        self._checkpoint_manager = checkpoint_manager
        self._early_stopping = early_stopping

        self._metric_recording = MetricRecording()
        self._round_manager = RoundManager(
            server=server,
            clients=clients,
            training_state=training_state,
            event_dispatcher=event_dispatcher,
            hook_manager=hook_manager,
            evaluator=None,
            monitor=monitor,
            logger_instance=logger_instance,
            checkpoint_manager=checkpoint_manager,
            early_stopping=early_stopping,
            metric_recording=self._metric_recording,
            synchronization_manager=None,
        )

        self._setup_default_hooks()

    def _setup_default_hooks(self) -> None:
        logging_hook = LoggingHook(self._logger)
        self._hook_manager.register("before_round", logging_hook)
        self._hook_manager.register("after_round", logging_hook)
        self._hook_manager.register("on_error", logging_hook)

        if self._checkpoint_manager is not None:
            checkpoint_hook = CheckpointSaving(
                checkpoint_manager=self._checkpoint_manager,
                interval=5,
                save_best=True,
            )
            self._hook_manager.register("after_round", checkpoint_hook)

        if self._early_stopping is not None:
            self._hook_manager.register("after_round", self._early_stopping)

        self._hook_manager.register("after_round", self._metric_recording)

    def initialize(self) -> None:
        self._state.phase = "initialized"
        self._logger.log_experiment_start(self._state.config)
        for client_id in self._state.client_ids():
            self._state.register_client(client_id)
        logger.info(
            f"Coordinator initialized with {len(self._clients)} clients, "
            f"{self._state.total_rounds} rounds"
        )

    def run(self) -> TrainingState:
        self.initialize()
        self._state.phase = "running"

        self._event_dispatcher.dispatch_simple(
            EventType.EXPERIMENT_START,
            {
                "num_clients": len(self._clients),
                "total_rounds": self._state.total_rounds,
            },
        )

        start_round = self._state.current_round + 1
        for round_id in range(start_round, self._state.total_rounds + 1):
            if self._round_manager.should_stop():
                break

            try:
                result = self._round_manager.run_round(
                    round_id=round_id,
                    epochs=self._state.config.get("epochs", 1),
                )
                logger.info(
                    f"Round {round_id} completed: "
                    f"acc={result['metrics'].get('accuracy', 0.0):.4f}, "
                    f"loss={result['metrics'].get('loss', 0.0):.4f}, "
                    f"clients={result['num_clients']}"
                )
            except Exception as e:
                logger.error(f"Round {round_id} failed: {e}")
                self._event_dispatcher.dispatch_simple(
                    EventType.ERROR_OCCURRED,
                    {"round_id": round_id, "error": str(e)},
                )
                ctx = HookContext(round_id=round_id)
                ctx.error = e
                self._hook_manager.execute("on_error", ctx)
                if self._early_stopping is not None:
                    self._early_stopping._stopped = True
                break

        self.finalize()
        return self._state

    def finalize(self) -> None:
        self._state.mark_completed()
        self._event_dispatcher.dispatch_simple(EventType.EXPERIMENT_END)
        self._logger.log_experiment_end(status=self._state.phase)

        summary = self._state.to_dict()
        summary["monitor"] = self._monitor.statistics()
        summary["round_manager"] = {
            "hook_count": self._hook_manager.total_hooks(),
            "events_dispatched": len(self._event_dispatcher._handlers),
        }
        logger.info(
            f"Experiment completed: {self._state.total_rounds} rounds, "
            f"{len(self._clients)} clients, "
            f"best round {self._state.best_round} "
            f"(acc={self._state.best_metric:.4f})"
        )

    @property
    def state(self) -> TrainingState:
        return self._state

    @property
    def round_manager(self) -> RoundManager:
        return self._round_manager

    @property
    def event_dispatcher(self) -> EventDispatcher:
        return self._event_dispatcher

    @property
    def hook_manager(self) -> HookManager:
        return self._hook_manager
