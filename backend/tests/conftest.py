import shutil
from pathlib import Path

import pytest

from app.core.config import settings


@pytest.fixture(autouse=True)
def clear_dataset_cache():
    cache_dir = Path(settings.datasets_root) / "cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    yield
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
