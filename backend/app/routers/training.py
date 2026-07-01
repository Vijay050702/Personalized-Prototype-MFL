from fastapi import APIRouter

from app.schemas.training import TrainingStatusSummary
from app.services.training_service import training_service

router = APIRouter(tags=["Training"])


@router.get("/training/status", response_model=TrainingStatusSummary)
def get_training_status():
    data = training_service.get_training_status()
    return TrainingStatusSummary(
        status="success", message="Training status retrieved", data=data
    )
