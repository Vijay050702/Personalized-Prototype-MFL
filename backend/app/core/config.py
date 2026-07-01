from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False
    )

    project_name: str = "Personalized Prototype-Based Multimodal Federated Learning"
    project_name_short: str = "PP-MFL"
    version: str = "0.1.0"
    debug: bool = True
    api_prefix: str = "/api/v1"

    cors_origins: list[str] = ["*"]
    cors_allow_credentials: bool = True
    cors_allow_methods: list[str] = ["*"]
    cors_allow_headers: list[str] = ["*"]

    log_level: str = "DEBUG"
    log_format: str = "json"
    log_file: str = "logs/app.log"

    secret_key: str = "change-me-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30


settings = Settings()
