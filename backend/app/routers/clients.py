from fastapi import APIRouter

from app.schemas.client import ClientListResponse
from app.services.client_service import client_service

router = APIRouter(tags=["Clients"])


@router.get("/clients", response_model=ClientListResponse)
def get_clients():
    clients = client_service.get_clients()
    return ClientListResponse(
        status="success", message="Clients retrieved", data=clients, total=len(clients)
    )
