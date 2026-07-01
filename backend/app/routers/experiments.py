from datetime import datetime, timezone, timedelta

from fastapi import APIRouter

from app.schemas.experiment import ExperimentListResponse, ExperimentResponse

router = APIRouter(tags=["Experiments"])


@router.get("/experiments", response_model=ExperimentListResponse)
def get_experiments():
    now = datetime.now(timezone.utc)
    experiments = [
        ExperimentResponse(
            id="exp-001",
            name="FedAvg-Baseline-CIFAR10",
            status="running",
            algorithm="FedAvg",
            num_clients=10,
            total_rounds=100,
            current_round=47,
            best_accuracy=0.8734,
            started_at=now - timedelta(hours=48),
            completed_at=None,
        ),
        ExperimentResponse(
            id="exp-002",
            name="FedProx-NonIID-EMNIST",
            status="completed",
            algorithm="FedProx",
            num_clients=15,
            total_rounds=80,
            current_round=80,
            best_accuracy=0.9213,
            started_at=now - timedelta(days=5),
            completed_at=now - timedelta(days=1),
        ),
        ExperimentResponse(
            id="exp-003",
            name="SCAFFOLD-Heterogeneous",
            status="completed",
            algorithm="SCAFFOLD",
            num_clients=20,
            total_rounds=120,
            current_round=120,
            best_accuracy=0.9432,
            started_at=now - timedelta(days=10),
            completed_at=now - timedelta(days=3),
        ),
        ExperimentResponse(
            id="exp-004",
            name="Personalized-FL-Prototype",
            status="pending",
            algorithm="pFedProto",
            num_clients=12,
            total_rounds=150,
            current_round=0,
            best_accuracy=0.0,
            started_at=now - timedelta(hours=2),
            completed_at=None,
        ),
    ]
    return ExperimentListResponse(
        status="success",
        message="Experiments retrieved",
        data=experiments,
        total=len(experiments),
    )
