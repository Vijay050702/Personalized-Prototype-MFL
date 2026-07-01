from pydantic import BaseModel


class TrainingStatusResponse(BaseModel):
    status: str
    current_round: int
    total_rounds: int
    epochs_completed: int
    total_epochs: int
    current_loss: float
    current_accuracy: float
    learning_rate: float
    clients_participating: int
    aggregation_algorithm: str
    time_elapsed_seconds: float
    estimated_time_remaining: float


class TrainingStatusSummary(BaseModel):
    status: str
    message: str
    data: TrainingStatusResponse
