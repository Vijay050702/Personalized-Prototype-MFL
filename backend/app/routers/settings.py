from fastapi import APIRouter

from app.schemas.settings import SettingsResponse, SettingsSummary

router = APIRouter(tags=["Settings"])


@router.get("/settings", response_model=SettingsSummary)
def get_settings():
    data = SettingsResponse(
        federation_strategy="personalized",
        aggregation_algorithm="FedAvg",
        learning_rate=0.001,
        batch_size=64,
        local_epochs=5,
        total_rounds=100,
        clients_per_round=10,
        min_clients=3,
        model_architecture="ResNet-18",
        prototype_dimension=512,
        communication_protocol="gRPC",
        encryption_enabled=True,
    )
    return SettingsSummary(status="success", message="Settings retrieved", data=data)
