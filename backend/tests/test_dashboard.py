from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_dashboard_endpoint():
    response = client.get("/api/v1/dashboard")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "data" in data
    assert data["data"]["active_clients"] == 12
    assert data["data"]["total_clients"] == 20
    assert data["data"]["current_round"] == 47
    assert data["data"]["training_status"] == "running"
