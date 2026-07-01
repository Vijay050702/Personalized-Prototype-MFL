from app.schemas.dataset import DatasetResponse


class DatasetService:
    def get_datasets(self) -> list[DatasetResponse]:
        return [
            DatasetResponse(
                id="ds-001",
                name="CIFAR-10-Client-A",
                type="image",
                modality="visual",
                size_mb=187.5,
                samples=5000,
                classes=10,
                client_id="client-001",
                distribution="non-iid",
            ),
            DatasetResponse(
                id="ds-002",
                name="Speech-Commands-B",
                type="audio",
                modality="acoustic",
                size_mb=342.1,
                samples=12000,
                classes=35,
                client_id="client-001",
                distribution="iid",
            ),
            DatasetResponse(
                id="ds-003",
                name="EMNIST-Client-C",
                type="image",
                modality="visual",
                size_mb=95.3,
                samples=8000,
                classes=62,
                client_id="client-002",
                distribution="non-iid",
            ),
            DatasetResponse(
                id="ds-004",
                name="Text-Classification-D",
                type="text",
                modality="linguistic",
                size_mb=64.8,
                samples=15000,
                classes=5,
                client_id="client-002",
                distribution="iid",
            ),
            DatasetResponse(
                id="ds-005",
                name="Sensor-Fusion-E",
                type="tabular",
                modality="multimodal",
                size_mb=512.0,
                samples=45000,
                classes=12,
                client_id="client-003",
                distribution="non-iid",
            ),
        ]


dataset_service = DatasetService()
