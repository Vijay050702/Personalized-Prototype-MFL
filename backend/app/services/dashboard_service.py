from datetime import datetime, timezone

from app.schemas.dashboard import DashboardResponse


class DashboardService:
    def get_dashboard(self) -> DashboardResponse:
        return DashboardResponse(
            active_clients=12,
            total_clients=20,
            current_round=47,
            total_rounds=100,
            global_accuracy=0.8734,
            global_loss=0.2341,
            training_status="running",
            experiments_running=3,
            uptime_hours=127.5,
            last_updated=datetime.now(timezone.utc),
        )


dashboard_service = DashboardService()
