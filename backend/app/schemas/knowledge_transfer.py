from datetime import datetime

from pydantic import BaseModel


class KnowledgeTransferResponse(BaseModel):
    transfer_id: str
    source_client: str
    target_client: str
    source_prototype: str
    target_prototype: str
    source_modality: str
    target_modality: str
    transfer_strategy: str
    cross_modal_mapping: str
    alignment_method: str
    transfer_loss: float
    similarity_score: float
    confidence: float
    communication_round: int
    transfer_status: str
    execution_time: float
    created_at: datetime


class KnowledgeTransferListResponse(BaseModel):
    status: str
    message: str
    data: list[KnowledgeTransferResponse]
    total: int


class KnowledgeTransferStatisticsResponseData(BaseModel):
    total_transfers: int
    successful_transfers: int
    failed_transfers: int
    average_similarity: float
    average_confidence: float
    average_transfer_loss: float
    average_execution_time: float
    communication_efficiency: float


class KnowledgeTransferStatisticsResponse(BaseModel):
    status: str
    message: str
    data: KnowledgeTransferStatisticsResponseData


class KnowledgeTransferHistoryResponse(BaseModel):
    status: str
    message: str
    data: list[KnowledgeTransferResponse]
    total: int
