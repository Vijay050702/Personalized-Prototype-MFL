from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_clients_endpoint():
    response = client.get("/api/v1/clients")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["data"]) == 5
    assert data["total"] == 5
    assert data["data"][0]["id"] == "client-001"


def test_datasets_endpoint():
    response = client.get("/api/v1/datasets")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["data"]) == 5


def test_training_status_endpoint():
    response = client.get("/api/v1/training/status")
    assert response.status_code == 200
    data = response.json()
    assert data["data"]["status"] == "running"
    assert data["data"]["current_round"] == 47


def test_prototypes_endpoint():
    response = client.get("/api/v1/prototypes")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["data"]) == 5


def test_evaluation_endpoint():
    response = client.get("/api/v1/evaluation")
    assert response.status_code == 200
    data = response.json()
    assert data["data"]["accuracy"] == 0.8734


def test_experiments_endpoint():
    response = client.get("/api/v1/experiments")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["data"]) == 4


def test_settings_endpoint():
    response = client.get("/api/v1/settings")
    assert response.status_code == 200
    data = response.json()
    assert data["data"]["federation_strategy"] == "personalized"
    assert data["data"]["aggregation_algorithm"] == "FedAvg"
