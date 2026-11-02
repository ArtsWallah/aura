import pytest
import pandas as pd
from fastapi.testclient import TestClient
from app.main import app
from app.core.config import settings

client = TestClient(app, client=("127.0.0.1", 54321))

@pytest.fixture
def api_test_csv():
    p = settings.UPLOADS_DIR / "api_test_dataset.csv.enc"
    df = pd.DataFrame({
        "category": ["A", "B", "C", "A", "B"],
        "revenue": [100.0, 200.0, 300.0, 150.0, 250.0]
    })
    from app.privacy.encryption import encryption_service
    p.write_bytes(encryption_service().encrypt_bytes(df.to_csv(index=False).encode("utf-8")))
    yield str(p)
    if p.exists():
        p.unlink()

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["registered_tools_count"] > 0
    assert data["registered_models_count"] > 0

def test_list_tools():
    response = client.get("/api/tools")
    assert response.status_code == 200
    tools = response.json()
    assert len(tools) >= 5
    tool_names = [t["name"] for t in tools]
    assert "inspect_dataset" in tool_names
    assert "descriptive_statistics" in tool_names

def test_list_models_and_classify():
    models_res = client.get("/api/models")
    assert models_res.status_code == 200
    models = models_res.json()
    assert len(models) >= 2

    # Intent classification endpoint
    intent_res = client.post("/api/models/classify-intent", json={"text": "Please analyze this sales csv and plot a chart"})
    assert intent_res.status_code == 200
    intent_data = intent_res.json()
    assert intent_data["intent"] == "data_analysis"
    assert intent_data["confidence"] > 0.5

    # Embedding endpoint
    embed_res = client.post("/api/models/embed", json={"text": "autonomous agent reasoning"})
    assert embed_res.status_code == 200
    embed_data = embed_res.json()
    assert len(embed_data["vector"]) > 50

def test_settings_endpoint():
    res = client.get("/api/settings", headers={"Origin": "http://localhost:3000"})
    assert res.status_code == 200
    data = res.json()
    assert "default_privacy_mode" in data
    # Ensure API key is masked
    if data["gemini_configured"]:
        assert data["gemini_api_key_masked"] == "Configured"

def test_create_task_api(api_test_csv):
    res = client.post(
        "/api/tasks",
        json={
            "objective": "Analyze dataset and detect underperforming categories",
            "file_path": api_test_csv,
            "privacy_mode": "local_only"
        }
    )
    assert res.status_code == 200
    task_data = res.json()
    assert task_data["status"] == "completed"
    assert task_data["final_deliverable"] is not None

