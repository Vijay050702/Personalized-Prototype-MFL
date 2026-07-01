from datetime import datetime

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


class DatasetMetadataResponse(BaseModel):
    dataset_name: str
    modalities: list[str]
    classes: list[str]
    num_classes: int
    input_shapes: dict[str, list[int]]
    num_samples: int
    client_count: int = 0
    missing_modality_ratio: float = 0.0
    download_status: str = "not_downloaded"
    preprocessing_status: str = "not_preprocessed"
    partition_status: str = "not_partitioned"


class DatasetRegistrationRequest(BaseModel):
    name: str
    modality: str = "image"
    modalities: list[str] | None = None
    path: str | None = None


class PartitionRequest(BaseModel):
    dataset_name: str
    strategy: str = "iid"
    num_clients: int = 10
    alpha: float | None = 0.5
    min_samples: int = 1
    seed: int = 42
    balanced: bool = True
    shards_per_client: int = 2


class PartitionResponse(BaseModel):
    status: str
    dataset_name: str
    strategy: str
    num_clients: int
    client_distributions: list[dict]
    seed: int


class DownloadRequest(BaseModel):
    dataset_name: str
    force: bool = False


class PreprocessRequest(BaseModel):
    dataset_name: str
    force: bool = False


class MissingModalityRequest(BaseModel):
    dataset_name: str
    strategy: str = "random"
    missing_ratio: float = 0.3
    modalities: list[str] | None = None
    seed: int = 42


class ValidationResponse(BaseModel):
    status: str
    dataset_name: str
    is_valid: bool
    errors: list[str]
    warnings: list[str]
    class_distribution: dict | None = None


class DatasetDetailResponse(BaseModel):
    status: str
    message: str
    data: DatasetMetadataResponse


class OperationResponse(BaseModel):
    status: str
    message: str
    dataset_name: str
    operation: str
