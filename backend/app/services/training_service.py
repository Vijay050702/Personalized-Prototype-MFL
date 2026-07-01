from app.schemas.training import TrainingStatusResponse


class TrainingService:
    def get_training_status(self) -> TrainingStatusResponse:
        return TrainingStatusResponse(
            status="running",
            current_round=47,
            total_rounds=100,
            epochs_completed=3,
            total_epochs=5,
            current_loss=0.2341,
            current_accuracy=0.8734,
            learning_rate=0.001,
            clients_participating=12,
            aggregation_algorithm="FedAvg",
            time_elapsed_seconds=8423.5,
            estimated_time_remaining=9567.3,
        )


training_service = TrainingService()
