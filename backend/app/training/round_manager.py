from __future__ import annotations

from typing import Any

from app.core.logging import logger
from app.training.callbacks import CheckpointSaving, EarlyStopping, MetricRecording
from app.training.client import Client
from app.training.evaluator import Evaluator
from app.training.events import Event, EventDispatcher, EventType
from app.training.hooks import HookContext, HookManager
from app.training.logger import TrainingLogger
from app.training.monitor import ResourceMonitor
from app.training.scheduler import SchedulerFactory
from app.training.server import Server
from app.training.state import TrainingState
from app.training.synchronization import SynchronizationManager
from app.training.utils import Timer


class RoundManager:
    def __init__(
        self,
        server: Server,
        clients: list[Client],
        training_state: TrainingState,
        event_dispatcher: EventDispatcher | None = None,
        hook_manager: HookManager | None = None,
        evaluator: Evaluator | None = None,
        synchronization_manager: SynchronizationManager | None = None,
        monitor: ResourceMonitor | None = None,
        logger_instance: TrainingLogger | None = None,
        checkpoint_manager: Any | None = None,
        early_stopping: EarlyStopping | None = None,
        metric_recording: MetricRecording | None = None,
    ) -> None:
        self._server = server
        self._clients = {c.client_id: c for c in clients}
        self._state = training_state
        self._event_dispatcher = event_dispatcher or EventDispatcher()
        self._hook_manager = hook_manager or HookManager()
        self._evaluator = evaluator or Evaluator()
        self._sync_manager = synchronization_manager or SynchronizationManager()
        self._monitor = monitor or ResourceMonitor()
        self._logger = logger_instance or TrainingLogger()
        self._checkpoint_manager = checkpoint_manager
        self._early_stopping = early_stopping
        self._metric_recording = metric_recording or MetricRecording()
        self._timer = Timer()

    def run_round(
        self,
        round_id: int,
        selected_client_ids: list[str] | None = None,
        epochs: int = 1,
    ) -> dict[str, Any]:
        self._timer.start()
        selected = selected_client_ids or list(self._clients.keys())

        self._state.current_round = round_id
        self._state.phase = "running"

        self._event_dispatcher.dispatch_simple(
            EventType.ROUND_START,
            {"round_id": round_id, "num_clients": len(selected)},
        )

        ctx = HookContext(round_id=round_id)
        self._hook_manager.execute("before_round", ctx)

        self._event_dispatcher.dispatch_simple(
            EventType.CLIENT_SELECTED,
            {"round_id": round_id, "clients": selected},
        )

        # 1. Client Selection
        active_clients = [
            self._clients[cid] for cid in selected if cid in self._clients
        ]

        # 2-3. Local Training + Prototype Generation
        client_updates: dict[str, Any] = {}
        for client in active_clients:
            client.set_current_round(round_id)

            self._hook_manager.execute(
                "before_training",
                HookContext(round_id=round_id, client_id=client.client_id),
            )

            train_result = client.train(epochs=epochs)

            self._hook_manager.execute(
                "after_training",
                HookContext(
                    round_id=round_id,
                    client_id=client.client_id,
                    metrics={
                        "loss": train_result.get("final_loss", 0.0),
                        "accuracy": train_result.get("final_accuracy", 0.0),
                    },
                ),
            )

            # 4. Local Prototype Generation
            prototypes = client.generate_prototypes()

            # 5. Prototype Upload
            packages = client.upload_results(round_id)
            client_updates[client.client_id] = packages

        # 6. Server Adaptive Aggregation
        self._event_dispatcher.dispatch_simple(
            EventType.AGGREGATION_START, {"round_id": round_id}
        )
        self._hook_manager.execute("before_aggregation", ctx)

        self._server.collect_client_results(client_updates)
        aggregated = self._server.aggregate_prototypes(round_id)

        self._hook_manager.execute("after_aggregation", ctx)
        self._event_dispatcher.dispatch_simple(
            EventType.AGGREGATION_END,
            {"round_id": round_id, "num_aggregated": len(aggregated)},
        )

        # 7. Cross-Modal Prototype Generation (Knowledge Transfer)
        self._event_dispatcher.dispatch_simple(
            EventType.KNOWLEDGE_TRANSFER_START, {"round_id": round_id}
        )
        self._hook_manager.execute("before_knowledge_transfer", ctx)

        synthesized, inferred = self._server.run_knowledge_transfer(round_id)

        self._hook_manager.execute("after_knowledge_transfer", ctx)
        self._event_dispatcher.dispatch_simple(
            EventType.KNOWLEDGE_TRANSFER_END, {"round_id": round_id}
        )

        # 8. Personalized Prototype Fusion
        self._event_dispatcher.dispatch_simple(
            EventType.PERSONALIZATION_START, {"round_id": round_id}
        )
        self._hook_manager.execute("before_personalization", ctx)

        global_protos = self._server.broadcast_global_prototypes()
        inferred_outputs = self._server.broadcast_synthesized_prototypes()

        all_personalized: list[Any] = []
        for client in active_clients:
            client.receive_updates(
                global_prototypes=global_protos,
                transferred=inferred_outputs,
            )
            personalized = client.personalize()
            all_personalized.extend(personalized)

        self._hook_manager.execute("after_personalization", ctx)
        self._event_dispatcher.dispatch_simple(
            EventType.PERSONALIZATION_END, {"round_id": round_id}
        )

        # 9. Broadcast Updated Information to Clients
        for client in active_clients:
            client.receive_updates(
                global_prototypes=global_protos,
                transferred=inferred_outputs,
            )

        # 10. Synchronize Clients
        self._event_dispatcher.dispatch_simple(
            EventType.SYNCHRONIZATION_START, {"round_id": round_id}
        )
        self._hook_manager.execute("before_synchronization", ctx)
        self._hook_manager.execute("after_synchronization", ctx)
        self._event_dispatcher.dispatch_simple(
            EventType.SYNCHRONIZATION_END, {"round_id": round_id}
        )

        # 11. Evaluation
        self._event_dispatcher.dispatch_simple(
            EventType.EVALUATION_START, {"round_id": round_id}
        )
        self._hook_manager.execute("before_evaluation", ctx)

        profiles_dict = {c.client_id: c.profile for c in active_clients}
        eval_metrics = self._evaluator.evaluate_all(
            round_id=round_id,
            personalized_prototypes=all_personalized,
            profiles=profiles_dict,
        )

        ctx.metrics = eval_metrics
        self._hook_manager.execute("after_evaluation", ctx)
        self._event_dispatcher.dispatch_simple(
            EventType.EVALUATION_END,
            {"round_id": round_id, "metrics": eval_metrics},
        )

        # Record metrics
        self._metric_recording.execute(ctx)
        self._state.record_round_metrics(round_id, eval_metrics)

        # 12. Checkpoint
        self._hook_manager.execute("before_checkpoint", ctx)
        if self._checkpoint_manager is not None:
            self._checkpoint_manager.save_latest(
                round_id=round_id, metrics=eval_metrics
            )
            self._event_dispatcher.dispatch_simple(
                EventType.CHECKPOINT_SAVED,
                {"round_id": round_id},
            )
        self._hook_manager.execute("after_checkpoint", ctx)

        # Round end hooks
        self._hook_manager.execute("after_round", ctx)
        self._event_dispatcher.dispatch_simple(
            EventType.ROUND_END,
            {"round_id": round_id, "metrics": eval_metrics},
        )

        round_time = self._timer.stop()

        # Monitor record
        self._monitor.record(
            round_id=round_id,
            training_time=round_time,
            prototype_count=sum(
                len(c.prototype_memory.local_repo.list()) for c in active_clients
            ),
            client_count=len(active_clients),
        )

        return {
            "round_id": round_id,
            "metrics": eval_metrics,
            "num_clients": len(active_clients),
            "num_aggregated": len(aggregated),
            "num_personalized": len(all_personalized),
            "duration": round_time,
        }

    def should_stop(self) -> bool:
        if self._early_stopping is not None and self._early_stopping.stopped:
            logger.info("Early stopping triggered")
            return True
        if self._state.current_round >= self._state.total_rounds:
            return True
        return False

    @property
    def evaluator(self) -> Evaluator:
        return self._evaluator

    @property
    def monitor(self) -> ResourceMonitor:
        return self._monitor

    @property
    def metric_recording(self) -> MetricRecording:
        return self._metric_recording
