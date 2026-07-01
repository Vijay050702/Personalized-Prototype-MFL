from __future__ import annotations

import hashlib
import zipfile
import tarfile
from pathlib import Path
from typing import Callable

import httpx

from app.core.logging import logger
from app.datasets.errors import ChecksumError, DownloadError


def compute_checksum(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_checksum(path: Path, expected: str, algorithm: str = "sha256") -> bool:
    actual = compute_checksum(path, algorithm)
    return actual == expected


class DownloadManager:
    def __init__(self, progress_callback: Callable[[int, int], None] | None = None):
        self.progress_callback = progress_callback

    def download(
        self,
        url: str,
        destination: Path,
        expected_checksum: str | None = None,
        force: bool = False,
    ) -> Path:
        if destination.exists() and not force:
            logger.info(f"File already exists, skipping download: {destination}")
            if expected_checksum:
                if verify_checksum(destination, expected_checksum):
                    return destination
                logger.warning(f"Checksum mismatch, re-downloading: {destination}")
            else:
                return destination

        destination.parent.mkdir(parents=True, exist_ok=True)

        try:
            logger.info(f"Downloading {url} -> {destination}")
            with httpx.Client(follow_redirects=True, timeout=300.0) as client:
                with client.stream("GET", url) as response:
                    if response.status_code != 200:
                        raise DownloadError(
                            f"Download failed with status {response.status_code}: {url}"
                        )
                    total = int(response.headers.get("content-length", 0))
                    downloaded = 0
                    with open(destination, "wb") as f:
                        for chunk in response.iter_bytes(chunk_size=65536):
                            f.write(chunk)
                            downloaded += len(chunk)
                            if self.progress_callback and total > 0:
                                self.progress_callback(downloaded, total)
            logger.info(
                f"Download complete: {destination} ({downloaded / 1024 / 1024:.1f} MB)"
            )
        except httpx.HTTPError as e:
            if destination.exists():
                destination.unlink()
            raise DownloadError(f"Download failed: {e}") from e

        if expected_checksum:
            if not verify_checksum(destination, expected_checksum):
                destination.unlink()
                raise ChecksumError(f"Checksum mismatch for {destination}")

        return destination

    def extract(self, archive_path: Path, destination: Path) -> Path:
        destination.mkdir(parents=True, exist_ok=True)
        logger.info(f"Extracting {archive_path} -> {destination}")

        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(destination)
        elif tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path, "r:*") as tf:
                tf.extractall(destination)
        else:
            raise DownloadError(f"Unsupported archive format: {archive_path.suffix}")

        logger.info(f"Extraction complete: {destination}")
        return destination


download_manager = DownloadManager()
