from fastapi import APIRouter

from app.schemas.dataset import DatasetListResponse
from app.services.dataset_service import dataset_service

router = APIRouter(tags=["Datasets"])


@router.get("/datasets", response_model=DatasetListResponse)
def get_datasets():
    datasets = dataset_service.get_datasets()
    return DatasetListResponse(
        status="success",
        message="Datasets retrieved",
        data=datasets,
        total=len(datasets),
    )
