from fastapi import APIRouter, HTTPException

from app.schemas.prototype import PrototypeListResponse, PrototypeResponse

router = APIRouter(tags=["Prototypes"])

PROTOTYPES = [
    PrototypeResponse(
        id="proto-001",
        modality="visual",
        dimension=512,
        cluster_id=0,
        quality_score=0.9234,
        client_id="client-001",
        created_round=10,
    ),
    PrototypeResponse(
        id="proto-002",
        modality="visual",
        dimension=512,
        cluster_id=1,
        quality_score=0.8876,
        client_id="client-001",
        created_round=10,
    ),
    PrototypeResponse(
        id="proto-003",
        modality="acoustic",
        dimension=256,
        cluster_id=0,
        quality_score=0.8456,
        client_id="client-002",
        created_round=15,
    ),
    PrototypeResponse(
        id="proto-004",
        modality="linguistic",
        dimension=768,
        cluster_id=2,
        quality_score=0.9123,
        client_id="client-002",
        created_round=20,
    ),
    PrototypeResponse(
        id="proto-005",
        modality="multimodal",
        dimension=1024,
        cluster_id=0,
        quality_score=0.9567,
        client_id="client-003",
        created_round=25,
    ),
]


@router.get("/prototypes", response_model=PrototypeListResponse)
def get_prototypes():
    return PrototypeListResponse(
        status="success",
        message="Prototypes retrieved",
        data=PROTOTYPES,
        total=len(PROTOTYPES),
    )


@router.get("/prototypes/{prototype_id}", response_model=PrototypeResponse)
def get_prototype_detail(prototype_id: str):
    for proto in PROTOTYPES:
        if proto.id == prototype_id:
            return proto
    raise HTTPException(status_code=404, detail=f"Prototype '{prototype_id}' not found")
