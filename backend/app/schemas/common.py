from pydantic import BaseModel


class ErrorResponse(BaseModel):
    status: str = "error"
    message: str
    details: dict | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    service: str
