from datetime import datetime

from pydantic import BaseModel


class ExperimentResponse(BaseModel):
    id: str
    name: str
    status: str
    algorithm: str
    num_clients: int
    total_rounds: int
    current_round: int
    best_accuracy: float
    started_at: datetime
    completed_at: datetime | None


class ExperimentListResponse(BaseModel):
    status: str
    message: str
    data: list[ExperimentResponse]
    total: int
