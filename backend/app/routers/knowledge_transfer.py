from datetime import datetime, timezone

from fastapi import APIRouter

from app.knowledge_transfer.registry import TransferRegistry
from app.schemas.knowledge_transfer import (
    KnowledgeTransferHistoryResponse,
    KnowledgeTransferListResponse,
    KnowledgeTransferResponse,
    KnowledgeTransferStatisticsResponse,
    KnowledgeTransferStatisticsResponseData,
)

router = APIRouter(tags=["Knowledge Transfer"])

KNOWLEDGE_TRANSFERS = [
    KnowledgeTransferResponse(
        transfer_id="kt-001",
        source_client="client-001",
        target_client="client-002",
        source_prototype="proto-001",
        target_prototype="proto-003",
        source_modality="visual",
        target_modality="acoustic",
        transfer_strategy="cross_modal_mapping",
        cross_modal_mapping="visual->acoustic",
        alignment_method="contrastive",
        transfer_loss=0.2341,
        similarity_score=0.8734,
        confidence=0.9123,
        communication_round=15,
        transfer_status="completed",
        execution_time=1.234,
        created_at=datetime.now(timezone.utc),
    ),
    KnowledgeTransferResponse(
        transfer_id="kt-002",
        source_client="client-002",
        target_client="client-003",
        source_prototype="proto-004",
        target_prototype="proto-005",
        source_modality="linguistic",
        target_modality="multimodal",
        transfer_strategy="prototype_synthesis",
        cross_modal_mapping="linguistic->multimodal",
        alignment_method="info_nce",
        transfer_loss=0.1876,
        similarity_score=0.9213,
        confidence=0.9456,
        communication_round=22,
        transfer_status="completed",
        execution_time=2.456,
        created_at=datetime.now(timezone.utc),
    ),
    KnowledgeTransferResponse(
        transfer_id="kt-003",
        source_client="client-001",
        target_client="client-003",
        source_prototype="proto-002",
        target_prototype="proto-005",
        source_modality="visual",
        target_modality="multimodal",
        transfer_strategy="cross_modal_mapping",
        cross_modal_mapping="visual->multimodal",
        alignment_method="triplet",
        transfer_loss=0.3123,
        similarity_score=0.8456,
        confidence=0.8876,
        communication_round=28,
        transfer_status="completed",
        execution_time=1.876,
        created_at=datetime.now(timezone.utc),
    ),
]


@router.get("/knowledge-transfer", response_model=KnowledgeTransferListResponse)
def get_knowledge_transfers():
    return KnowledgeTransferListResponse(
        status="success",
        message="Knowledge transfers retrieved",
        data=KNOWLEDGE_TRANSFERS,
        total=len(KNOWLEDGE_TRANSFERS),
    )


@router.get(
    "/knowledge-transfer/statistics",
    response_model=KnowledgeTransferStatisticsResponse,
)
def get_knowledge_transfer_statistics():
    registry = TransferRegistry()
    config = registry.to_config()
    completed = [t for t in KNOWLEDGE_TRANSFERS if t.transfer_status == "completed"]
    similarities = [t.similarity_score for t in completed]
    confidences = [t.confidence for t in completed]
    losses = [t.transfer_loss for t in completed]
    times = [t.execution_time for t in completed]

    return KnowledgeTransferStatisticsResponse(
        status="success",
        message="Knowledge transfer statistics retrieved",
        data=KnowledgeTransferStatisticsResponseData(
            total_transfers=len(KNOWLEDGE_TRANSFERS),
            successful_transfers=len(completed),
            failed_transfers=sum(
                1 for t in KNOWLEDGE_TRANSFERS if t.transfer_status == "failed"
            ),
            average_similarity=(sum(similarities) / len(similarities))
            if similarities
            else 0.0,
            average_confidence=(sum(confidences) / len(confidences))
            if confidences
            else 0.0,
            average_transfer_loss=(sum(losses) / len(losses)) if losses else 0.0,
            average_execution_time=(sum(times) / len(times)) if times else 0.0,
            communication_efficiency=0.9234,
        ),
    )


@router.get(
    "/knowledge-transfer/history",
    response_model=KnowledgeTransferHistoryResponse,
)
def get_knowledge_transfer_history():
    return KnowledgeTransferHistoryResponse(
        status="success",
        message="Knowledge transfer history retrieved",
        data=KNOWLEDGE_TRANSFERS,
        total=len(KNOWLEDGE_TRANSFERS),
    )


@router.get(
    "/knowledge-transfer/{transfer_id}",
    response_model=KnowledgeTransferResponse,
)
def get_knowledge_transfer_detail(transfer_id: str):
    for kt in KNOWLEDGE_TRANSFERS:
        if kt.transfer_id == transfer_id:
            return kt
    from fastapi import HTTPException

    raise HTTPException(
        status_code=404, detail=f"Knowledge transfer '{transfer_id}' not found"
    )
