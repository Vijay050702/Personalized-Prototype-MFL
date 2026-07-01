from datetime import datetime, timezone, timedelta

from app.schemas.client import ClientResponse


class ClientService:
    def get_clients(self) -> list[ClientResponse]:
        now = datetime.now(timezone.utc)
        return [
            ClientResponse(
                id="client-001",
                name="Edge-Device-Alpha",
                status="active",
                accuracy=0.9123,
                loss=0.1876,
                data_size=15234,
                last_round=47,
                device="NVIDIA Jetson Xavier",
                region="us-east-1",
                joined_at=now - timedelta(days=14),
                last_communication=now - timedelta(minutes=3),
            ),
            ClientResponse(
                id="client-002",
                name="Mobile-Unit-Beta",
                status="active",
                accuracy=0.8845,
                loss=0.2103,
                data_size=9821,
                last_round=47,
                device="Google Pixel 8",
                region="eu-west-1",
                joined_at=now - timedelta(days=10),
                last_communication=now - timedelta(minutes=1),
            ),
            ClientResponse(
                id="client-003",
                name="IoT-Sensor-Gamma",
                status="inactive",
                accuracy=0.7654,
                loss=0.3421,
                data_size=4532,
                last_round=32,
                device="Raspberry Pi 5",
                region="ap-southeast-1",
                joined_at=now - timedelta(days=21),
                last_communication=now - timedelta(hours=5),
            ),
            ClientResponse(
                id="client-004",
                name="Workstation-Delta",
                status="active",
                accuracy=0.9456,
                loss=0.1209,
                data_size=28765,
                last_round=47,
                device="Linux Workstation RTX 4090",
                region="us-west-2",
                joined_at=now - timedelta(days=7),
                last_communication=now - timedelta(minutes=2),
            ),
            ClientResponse(
                id="client-005",
                name="Server-Epsilon",
                status="active",
                accuracy=0.9234,
                loss=0.1654,
                data_size=32100,
                last_round=46,
                device="DGX A100",
                region="eu-central-1",
                joined_at=now - timedelta(days=30),
                last_communication=now - timedelta(minutes=10),
            ),
        ]


client_service = ClientService()
