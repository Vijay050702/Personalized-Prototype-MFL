from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.datasets.adapters.uci_har import UCIHARAdapter
from app.main import app

client = TestClient(app)


@pytest.fixture(scope="session", autouse=True)
def generate_sample_data():
    ds_root = Path(settings.datasets_root)
    adapter = UCIHARAdapter()
    adapter.generate_sample(ds_root, num_samples=50)
    yield


class TestDatasetsAPI:
    def test_get_datasets_list(self):
        response = client.get("/api/v1/datasets")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert len(data["data"]) > 0

    def test_get_dataset_detail(self):
        response = client.get("/api/v1/datasets/uci_har")
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["dataset_name"] == "uci_har"
        assert data["data"]["num_classes"] == 6

    def test_get_dataset_detail_not_found(self):
        response = client.get("/api/v1/datasets/nonexistent_dataset")
        assert response.status_code == 404

    def test_register_dataset(self):
        response = client.post(
            "/api/v1/datasets/register",
            json={
                "name": "my_custom_data",
                "modality": "image",
                "modalities": ["image"],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["dataset_name"] == "my_custom_data"

    def test_register_duplicate(self):
        client.post(
            "/api/v1/datasets/register",
            json={"name": "dup_test", "modality": "image"},
        )
        response = client.post(
            "/api/v1/datasets/register",
            json={"name": "dup_test", "modality": "image"},
        )
        assert response.status_code == 409

    def test_download_dataset(self):
        response = client.post(
            "/api/v1/datasets/download",
            json={"dataset_name": "uci_har"},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "success"

    def test_download_nonexistent(self):
        response = client.post(
            "/api/v1/datasets/download",
            json={"dataset_name": "nonexistent"},
        )
        assert response.status_code == 404

    def test_preprocess_dataset(self):
        response = client.post(
            "/api/v1/datasets/preprocess",
            json={"dataset_name": "uci_har"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["operation"] == "preprocess"

    def test_partition_iid(self):
        response = client.post(
            "/api/v1/datasets/partition",
            json={
                "dataset_name": "uci_har",
                "strategy": "iid",
                "num_clients": 5,
                "seed": 42,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["strategy"] == "iid"
        assert data["num_clients"] == 5

    def test_partition_dirichlet(self):
        response = client.post(
            "/api/v1/datasets/partition",
            json={
                "dataset_name": "uci_har",
                "strategy": "dirichlet",
                "num_clients": 5,
                "alpha": 0.5,
                "seed": 42,
            },
        )
        assert response.status_code == 200
        assert response.json()["strategy"] == "dirichlet"

    def test_partition_shard(self):
        response = client.post(
            "/api/v1/datasets/partition",
            json={
                "dataset_name": "uci_har",
                "strategy": "shard",
                "num_clients": 5,
                "shards_per_client": 2,
                "seed": 42,
            },
        )
        assert response.status_code == 200
        assert response.json()["strategy"] == "shard"

    def test_partition_invalid_strategy(self):
        response = client.post(
            "/api/v1/datasets/partition",
            json={
                "dataset_name": "uci_har",
                "strategy": "invalid",
                "num_clients": 5,
            },
        )
        assert response.status_code == 400

    def test_missing_modality(self):
        response = client.post(
            "/api/v1/datasets/missing-modality",
            json={
                "dataset_name": "uci_har",
                "strategy": "random",
                "missing_ratio": 0.5,
                "seed": 42,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["operation"] == "missing_modality"

    def test_missing_modality_invalid_strategy(self):
        response = client.post(
            "/api/v1/datasets/missing-modality",
            json={
                "dataset_name": "uci_har",
                "strategy": "invalid",
                "missing_ratio": 0.5,
            },
        )
        assert response.status_code == 400

    def test_validate_dataset(self):
        response = client.get("/api/v1/datasets/uci_har/validate")
        assert response.status_code == 200
        data = response.json()
        assert "is_valid" in data

    def test_get_metadata(self):
        response = client.get("/api/v1/datasets/uci_har/metadata")
        assert response.status_code == 200
        data = response.json()
        assert "modalities" in str(data)

    def test_delete_dataset(self):
        client.post(
            "/api/v1/datasets/register",
            json={"name": "to_delete", "modality": "sensor"},
        )
        response = client.delete("/api/v1/datasets/to_delete")
        assert response.status_code == 200
        assert response.json()["operation"] == "delete"

    def test_delete_nonexistent(self):
        response = client.delete("/api/v1/datasets/does_not_exist")
        assert response.status_code == 200


class TestPreprocessorsAPI:
    def test_preprocessors_image(self):
        from app.datasets.preprocessors.image import ImagePreprocessor

        p = ImagePreprocessor(target_size=(32, 32))
        data = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = p.process(data)
        assert result.shape[1] == 32
        assert result.shape[2] == 32

    def test_preprocessors_text(self):
        from app.datasets.preprocessors.text import TextPreprocessor

        p = TextPreprocessor(max_length=10, max_vocab_size=50)
        p.fit([{"text": np.array("hello world test")}])
        result = p.process(np.array("hello world"))
        assert len(result) == 10

    def test_preprocessors_audio(self):
        from app.datasets.preprocessors.audio import AudioPreprocessor

        p = AudioPreprocessor(target_sr=16000, n_mels=64)
        audio = np.sin(np.linspace(0, 100, 16000))
        result = p.process(audio)
        assert result.shape[0] == 64

    def test_preprocessors_sensor(self):
        from app.datasets.preprocessors.sensor import SensorPreprocessor

        p = SensorPreprocessor(window_size=16, stride=8)
        data = np.random.rand(100, 6).astype(np.float64)
        result = p.process(data)
        assert result.shape[1] == 16
        assert result.shape[2] == 6
