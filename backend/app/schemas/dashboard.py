from datetime import datetime

from pydantic import BaseModel


class DashboardResponse(BaseModel):
    active_clients: int
    total_clients: int
    current_round: int
    total_rounds: int
    global_accuracy: float
    global_loss: float
    training_status: str
    experiments_running: int
    uptime_hours: float
    last_updated: datetime


class DashboardSummary(BaseModel):
    status: str
    message: str
    data: DashboardResponse
