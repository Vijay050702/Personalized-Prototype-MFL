from fastapi import APIRouter

from app.core.constants import API_PREFIX
from app.routers import (
    dashboard,
    clients,
    datasets,
    training,
    prototypes,
    evaluation,
    experiments,
    settings,
)

api_router = APIRouter(prefix=API_PREFIX)

api_router.include_router(dashboard.router)
api_router.include_router(clients.router)
api_router.include_router(datasets.router)
api_router.include_router(training.router)
api_router.include_router(prototypes.router)
api_router.include_router(evaluation.router)
api_router.include_router(experiments.router)
api_router.include_router(settings.router)
