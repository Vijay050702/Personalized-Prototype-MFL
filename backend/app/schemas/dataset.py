from pydantic import BaseModel


class DatasetResponse(BaseModel):
    id: str
    name: str
    type: str
    modality: str
    size_mb: float
    samples: int
    classes: int
    client_id: str
    distribution: str


class DatasetListResponse(BaseModel):
    status: str
    message: str
    data: list[DatasetResponse]
    total: int
