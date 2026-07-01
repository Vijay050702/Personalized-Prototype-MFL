from pydantic import BaseModel


class SettingsResponse(BaseModel):
    federation_strategy: str
    aggregation_algorithm: str
    learning_rate: float
    batch_size: int
    local_epochs: int
    total_rounds: int
    clients_per_round: int
    min_clients: int
    model_architecture: str
    prototype_dimension: int
    communication_protocol: str
    encryption_enabled: bool


class SettingsSummary(BaseModel):
    status: str
    message: str
    data: SettingsResponse
