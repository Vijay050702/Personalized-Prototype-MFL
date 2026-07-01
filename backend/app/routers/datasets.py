from fastapi import APIRouter, HTTPException

from app.datasets.errors import (
    DatasetAlreadyExistsError,
    DatasetNotFoundError,
    DatasetValidationError,
    DownloadError,
    InvalidModalityError,
)
from app.schemas.dataset import (
    DatasetDetailResponse,
    DatasetListResponse,
    DatasetMetadataResponse,
    DatasetRegistrationRequest,
    DatasetResponse,
    DownloadRequest,
    MissingModalityRequest,
    OperationResponse,
    PartitionRequest,
    PartitionResponse,
    PreprocessRequest,
    ValidationResponse,
)
from app.services.dataset_service import dataset_service

router = APIRouter(tags=["Datasets"])


@router.get("/datasets", response_model=DatasetListResponse)
def get_datasets():
    datasets = dataset_service.get_datasets()
    return DatasetListResponse(
        status="success",
        message="Datasets retrieved",
        data=datasets,
        total=len(datasets),
    )


@router.get("/datasets/{name}", response_model=DatasetDetailResponse)
def get_dataset_detail(name: str):
    try:
        detail = dataset_service.get_dataset_detail(name)
        return DatasetDetailResponse(
            status="success", message=f"Dataset '{name}' details retrieved", data=detail
        )
    except DatasetNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/datasets/register", response_model=DatasetDetailResponse)
def register_dataset(req: DatasetRegistrationRequest):
    try:
        detail = dataset_service.register_dataset(
            name=req.name,
            modalities=req.modalities or [req.modality],
        )
        return DatasetDetailResponse(
            status="success", message=f"Dataset '{req.name}' registered", data=detail
        )
    except DatasetNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except DatasetAlreadyExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.post("/datasets/download", response_model=OperationResponse)
def download_dataset(req: DownloadRequest):
    try:
        result = dataset_service.download_dataset(req.dataset_name, force=req.force)
        return OperationResponse(
            status="success",
            message=result["message"],
            dataset_name=req.dataset_name,
            operation="download",
        )
    except DatasetNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except DownloadError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/datasets/preprocess", response_model=OperationResponse)
def preprocess_dataset(req: PreprocessRequest):
    try:
        result = dataset_service.preprocess_dataset(req.dataset_name, force=req.force)
        return OperationResponse(
            status="success",
            message=result["message"],
            dataset_name=req.dataset_name,
            operation="preprocess",
        )
    except DatasetNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/datasets/partition", response_model=PartitionResponse)
def partition_dataset(req: PartitionRequest):
    try:
        result = dataset_service.partition_dataset(
            dataset_name=req.dataset_name,
            strategy=req.strategy,
            num_clients=req.num_clients,
            alpha=req.alpha if req.strategy == "dirichlet" else None,
            seed=req.seed,
            balanced=req.balanced,
            shards_per_client=req.shards_per_client,
            min_samples=req.min_samples,
        )
        return PartitionResponse(**result)
    except DatasetNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/datasets/missing-modality", response_model=OperationResponse)
def apply_missing_modality(req: MissingModalityRequest):
    try:
        result = dataset_service.apply_missing_modality(
            dataset_name=req.dataset_name,
            strategy=req.strategy,
            missing_ratio=req.missing_ratio,
            modalities=req.modalities,
            seed=req.seed,
        )
        return OperationResponse(
            status="success",
            message=f"Missing modality applied: {result['num_samples_affected']} samples affected",
            dataset_name=req.dataset_name,
            operation="missing_modality",
        )
    except DatasetNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except InvalidModalityError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/datasets/{name}/validate", response_model=ValidationResponse)
def validate_dataset(name: str):
    try:
        result = dataset_service.validate_dataset(name)
        return ValidationResponse(**result)
    except DatasetNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/datasets/{name}/metadata", response_model=DatasetDetailResponse)
def get_dataset_metadata(name: str):
    try:
        meta = dataset_service.get_metadata(name)
        data = DatasetMetadataResponse(
            dataset_name=meta.get("dataset_name", name),
            modalities=meta.get("modalities", []),
            classes=meta.get("classes", []),
            num_classes=meta.get("num_classes", 0),
            input_shapes=meta.get("input_shapes", {}),
            num_samples=meta.get("num_samples", 0),
        )
        return DatasetDetailResponse(
            status="success", message=f"Metadata for '{name}'", data=data
        )
    except DatasetNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/datasets/{name}", response_model=OperationResponse)
def delete_dataset(name: str):
    try:
        dataset_service.delete_dataset(name)
        return OperationResponse(
            status="success",
            message=f"Dataset '{name}' deleted",
            dataset_name=name,
            operation="delete",
        )
    except DatasetNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
