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


def test_training_config_not_found():
    response = client.get("/api/v1/training/config")
    assert response.status_code == 404


def test_knowledge_transfer_list():
    response = client.get("/api/v1/knowledge-transfer")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["data"]) >= 1
    assert data["data"][0]["transfer_id"] == "kt-001"
    assert data["data"][0]["transfer_status"] == "completed"


def test_knowledge_transfer_detail():
    response = client.get("/api/v1/knowledge-transfer/kt-001")
    assert response.status_code == 200
    data = response.json()
    assert data["transfer_id"] == "kt-001"
    assert data["source_modality"] == "visual"
    assert data["target_modality"] == "acoustic"


def test_knowledge_transfer_detail_not_found():
    response = client.get("/api/v1/knowledge-transfer/nonexistent")
    assert response.status_code == 404
    data = response.json()
    assert data["status"] == "error"


def test_knowledge_transfer_statistics():
    response = client.get("/api/v1/knowledge-transfer/statistics")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["data"]["total_transfers"] >= 1
    assert isinstance(data["data"]["average_similarity"], float)
    assert isinstance(data["data"]["communication_efficiency"], float)


def test_knowledge_transfer_history():
    response = client.get("/api/v1/knowledge-transfer/history")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total"] >= 1
    assert len(data["data"]) >= 1


def test_prototype_detail():
    response = client.get("/api/v1/prototypes/proto-001")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "proto-001"
    assert data["modality"] == "visual"
    assert data["dimension"] == 512


def test_prototype_detail_not_found():
    response = client.get("/api/v1/prototypes/nonexistent")
    assert response.status_code == 404
    data = response.json()
    assert data["status"] == "error"


def test_similarity_list():
    response = client.get("/api/v1/similarity")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["data"]) >= 1
    assert data["data"][0]["analysis_id"] == "sim-001"


def test_similarity_detail():
    response = client.get("/api/v1/similarity/sim-001")
    assert response.status_code == 200
    data = response.json()
    assert data["analysis_id"] == "sim-001"
    assert data["source_modality"] == "visual"
    assert data["target_modality"] == "acoustic"


def test_similarity_detail_not_found():
    response = client.get("/api/v1/similarity/nonexistent")
    assert response.status_code == 404
    data = response.json()
    assert data["status"] == "error"


def test_similarity_statistics():
    response = client.get("/api/v1/similarity/statistics")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert isinstance(data["data"]["average_similarity"], float)
    assert data["data"]["cluster_count"] >= 1
    assert data["data"]["communication_round"] >= 1


def test_similarity_matrix():
    response = client.get("/api/v1/similarity/matrix")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["data"]) >= 1
    assert data["data"][0]["source"] is not None
    assert data["data"][0]["similarity"] is not None


def test_similarity_history():
    response = client.get("/api/v1/similarity/history")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total"] >= 1
    assert len(data["data"]) >= 1


def test_swagger_knowledge_transfer_endpoints_registered():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    spec = response.json()
    paths = spec["paths"]
    assert "/api/v1/knowledge-transfer" in paths
    assert "/api/v1/knowledge-transfer/statistics" in paths
    assert "/api/v1/knowledge-transfer/history" in paths
    assert "/api/v1/knowledge-transfer/{transfer_id}" in paths


def test_swagger_similarity_endpoints_registered():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    spec = response.json()
    paths = spec["paths"]
    assert "/api/v1/similarity" in paths
    assert "/api/v1/similarity/statistics" in paths
    assert "/api/v1/similarity/matrix" in paths
    assert "/api/v1/similarity/history" in paths
    assert "/api/v1/similarity/{analysis_id}" in paths
