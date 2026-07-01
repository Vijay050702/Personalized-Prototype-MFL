from datetime import datetime

from pydantic import BaseModel


class ClientResponse(BaseModel):
    id: str
    name: str
    status: str
    accuracy: float
    loss: float
    data_size: int
    last_round: int
    device: str
    region: str
    joined_at: datetime
    last_communication: datetime


class ClientListResponse(BaseModel):
    status: str
    message: str
    data: list[ClientResponse]
    total: int
