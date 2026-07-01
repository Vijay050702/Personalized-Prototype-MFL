from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.api import api_router
from app.core.config import settings
from app.core.constants import MSG_HEALTH_OK
from app.core.error_handlers import register_exception_handlers
from app.core.logging import logger
from app.schemas.common import HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.project_name} v{settings.version}")
    yield
    logger.info("Shutting down application")


app = FastAPI(
    title=settings.project_name,
    version=settings.version,
    description="Backend API for Personalized Prototype-Based Multimodal Federated Learning",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)

register_exception_handlers(app)

app.include_router(api_router)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    return HealthResponse(
        status="ok", version=settings.version, service=settings.project_name_short
    )
