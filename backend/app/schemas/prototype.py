from pydantic import BaseModel


class PrototypeResponse(BaseModel):
    id: str
    modality: str
    dimension: int
    cluster_id: int
    quality_score: float
    client_id: str
    created_round: int


class PrototypeListResponse(BaseModel):
    status: str
    message: str
    data: list[PrototypeResponse]
    total: int
