from app.datasets.cache import DatasetCache
from app.datasets.dataset_factory import DatasetFactory
from app.datasets.dataset_registry import DatasetRegistry
from app.datasets.errors import (
    CacheError,
    ChecksumError,
    DatasetAlreadyExistsError,
    DatasetError,
    DatasetNotFoundError,
    DatasetValidationError,
    DownloadError,
    InvalidModalityError,
    PartitionError,
    PreprocessingError,
)

dataset_cache = DatasetCache()
dataset_registry = DatasetRegistry()
dataset_factory = DatasetFactory()

__all__ = [
    "dataset_cache",
    "dataset_registry",
    "dataset_factory",
    "DatasetError",
    "DatasetNotFoundError",
    "DatasetAlreadyExistsError",
    "DatasetValidationError",
    "DownloadError",
    "ChecksumError",
    "PartitionError",
    "PreprocessingError",
    "InvalidModalityError",
    "CacheError",
]
