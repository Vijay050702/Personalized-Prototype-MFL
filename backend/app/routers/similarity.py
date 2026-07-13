from datetime import datetime, timezone

from fastapi import APIRouter

from app.schemas.similarity import (
    SimilarityAnalysis,
    SimilarityHistoryResponse,
    SimilarityListResponse,
    SimilarityMatrixEntry,
    SimilarityMatrixResponse,
    SimilarityStatisticsResponse,
    SimilarityStatisticsResponseData,
)

router = APIRouter(tags=["Similarity"])

SIMILARITY_ANALYSES = [
    SimilarityAnalysis(
        analysis_id="sim-001",
        source_client="client-001",
        target_client="client-002",
        source_prototype="proto-001",
        target_prototype="proto-003",
        source_modality="visual",
        target_modality="acoustic",
        similarity_metric="cosine",
        cosine_similarity=0.8734,
        euclidean_distance=0.4521,
        prototype_distance=0.2341,
        transfer_confidence=0.9123,
        aggregation_round=15,
        cluster_id=0,
        analysis_status="completed",
        created_at=datetime.now(timezone.utc),
    ),
    SimilarityAnalysis(
        analysis_id="sim-002",
        source_client="client-002",
        target_client="client-003",
        source_prototype="proto-004",
        target_prototype="proto-005",
        source_modality="linguistic",
        target_modality="multimodal",
        similarity_metric="cosine",
        cosine_similarity=0.9213,
        euclidean_distance=0.3123,
        prototype_distance=0.1876,
        transfer_confidence=0.9456,
        aggregation_round=22,
        cluster_id=1,
        analysis_status="completed",
        created_at=datetime.now(timezone.utc),
    ),
    SimilarityAnalysis(
        analysis_id="sim-003",
        source_client="client-001",
        target_client="client-003",
        source_prototype="proto-002",
        target_prototype="proto-005",
        source_modality="visual",
        target_modality="multimodal",
        similarity_metric="cosine",
        cosine_similarity=0.8456,
        euclidean_distance=0.5234,
        prototype_distance=0.3123,
        transfer_confidence=0.8876,
        aggregation_round=28,
        cluster_id=2,
        analysis_status="completed",
        created_at=datetime.now(timezone.utc),
    ),
    SimilarityAnalysis(
        analysis_id="sim-004",
        source_client="client-002",
        target_client="client-001",
        source_prototype="proto-003",
        target_prototype="proto-001",
        source_modality="acoustic",
        target_modality="visual",
        similarity_metric="euclidean",
        cosine_similarity=0.7567,
        euclidean_distance=0.6543,
        prototype_distance=0.4123,
        transfer_confidence=0.8234,
        aggregation_round=15,
        cluster_id=0,
        analysis_status="completed",
        created_at=datetime.now(timezone.utc),
    ),
]


@router.get("/similarity", response_model=SimilarityListResponse)
def get_similarity_analyses():
    return SimilarityListResponse(
        status="success",
        message="Similarity analyses retrieved",
        data=SIMILARITY_ANALYSES,
        total=len(SIMILARITY_ANALYSES),
    )


@router.get("/similarity/statistics", response_model=SimilarityStatisticsResponse)
def get_similarity_statistics():
    similarities = [a.cosine_similarity for a in SIMILARITY_ANALYSES]
    distances = [a.euclidean_distance for a in SIMILARITY_ANALYSES]
    return SimilarityStatisticsResponse(
        status="success",
        message="Similarity statistics retrieved",
        data=SimilarityStatisticsResponseData(
            average_similarity=(sum(similarities) / len(similarities))
            if similarities
            else 0.0,
            maximum_similarity=max(similarities) if similarities else 0.0,
            minimum_similarity=min(similarities) if similarities else 0.0,
            average_distance=(sum(distances) / len(distances)) if distances else 0.0,
            cluster_count=3,
            client_groups=2,
            prototype_groups=4,
            communication_round=28,
        ),
    )


@router.get("/similarity/matrix", response_model=SimilarityMatrixResponse)
def get_similarity_matrix():
    return SimilarityMatrixResponse(
        status="success",
        message="Similarity matrix retrieved",
        data=[
            SimilarityMatrixEntry(
                source="visual", target="acoustic", similarity=0.8734, count=2
            ),
            SimilarityMatrixEntry(
                source="visual", target="linguistic", similarity=0.8123, count=1
            ),
            SimilarityMatrixEntry(
                source="visual", target="multimodal", similarity=0.8456, count=1
            ),
            SimilarityMatrixEntry(
                source="acoustic", target="visual", similarity=0.7567, count=1
            ),
            SimilarityMatrixEntry(
                source="acoustic", target="linguistic", similarity=0.7234, count=1
            ),
            SimilarityMatrixEntry(
                source="acoustic", target="multimodal", similarity=0.7891, count=1
            ),
            SimilarityMatrixEntry(
                source="linguistic", target="multimodal", similarity=0.9213, count=1
            ),
            SimilarityMatrixEntry(
                source="multimodal", target="visual", similarity=0.8345, count=1
            ),
        ],
    )


@router.get("/similarity/history", response_model=SimilarityHistoryResponse)
def get_similarity_history():
    return SimilarityHistoryResponse(
        status="success",
        message="Similarity history retrieved",
        data=SIMILARITY_ANALYSES,
        total=len(SIMILARITY_ANALYSES),
    )


@router.get(
    "/similarity/{analysis_id}",
    response_model=SimilarityAnalysis,
)
def get_similarity_detail(analysis_id: str):
    for analysis in SIMILARITY_ANALYSES:
        if analysis.analysis_id == analysis_id:
            return analysis
    from fastapi import HTTPException

    raise HTTPException(
        status_code=404, detail=f"Similarity analysis '{analysis_id}' not found"
    )
