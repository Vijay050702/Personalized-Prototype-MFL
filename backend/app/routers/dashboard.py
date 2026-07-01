from fastapi import APIRouter

from app.schemas.dashboard import DashboardSummary
from app.services.dashboard_service import dashboard_service

router = APIRouter(tags=["Dashboard"])


@router.get("/dashboard", response_model=DashboardSummary)
def get_dashboard():
    data = dashboard_service.get_dashboard()
    return DashboardSummary(
        status="success", message="Dashboard data retrieved", data=data
    )
