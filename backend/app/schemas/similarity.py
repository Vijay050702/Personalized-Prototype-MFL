from datetime import datetime

from pydantic import BaseModel


class SimilarityAnalysis(BaseModel):
    analysis_id: str
    source_client: str
    target_client: str
    source_prototype: str
    target_prototype: str
    source_modality: str
    target_modality: str
    similarity_metric: str
    cosine_similarity: float
    euclidean_distance: float
    prototype_distance: float
    transfer_confidence: float
    aggregation_round: int
    cluster_id: int
    analysis_status: str
    created_at: datetime


class SimilarityListResponse(BaseModel):
    status: str
    message: str
    data: list[SimilarityAnalysis]
    total: int


class SimilarityMatrixEntry(BaseModel):
    source: str
    target: str
    similarity: float
    count: int


class SimilarityMatrixResponse(BaseModel):
    status: str
    message: str
    data: list[SimilarityMatrixEntry]


class SimilarityStatisticsResponseData(BaseModel):
    average_similarity: float
    maximum_similarity: float
    minimum_similarity: float
    average_distance: float
    cluster_count: int
    client_groups: int
    prototype_groups: int
    communication_round: int


class SimilarityStatisticsResponse(BaseModel):
    status: str
    message: str
    data: SimilarityStatisticsResponseData


class SimilarityHistoryResponse(BaseModel):
    status: str
    message: str
    data: list[SimilarityAnalysis]
    total: int
