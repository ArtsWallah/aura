import pytest
import pandas as pd
import time
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


def test_queued_task_can_be_cancelled_through_task_api():
    submitted = client.post("/api/tasks", json={
        "objective": "cancel this queued task",
        "privacy_mode": "local_only",
    })
    assert submitted.status_code == 202
    task_id = submitted.json()["task_id"]

    cancelled = client.post(f"/api/tasks/{task_id}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    refreshed = client.get(f"/api/tasks/{task_id}")
    assert refreshed.status_code == 200
    assert refreshed.json()["status"] == "cancelled"

def test_create_task_api(api_test_csv):
    objective_marker = "private-objective-marker-9137"
    file_marker = api_test_csv
    with TestClient(app, client=("127.0.0.1", 54321)) as task_client:
        res = task_client.post(
            "/api/tasks",
            json={
                "objective": f"Analyze dataset and detect underperforming categories {objective_marker}",
                "file_path": api_test_csv,
                "privacy_mode": "local_only"
            }
        )
        assert res.status_code == 202
        accepted = res.json()
        assert accepted["status"] == "queued"
        assert set(accepted) == {"task_id", "status", "privacy_mode", "created_at", "updated_at"}
        assert objective_marker not in res.text and file_marker not in res.text
        task_id = accepted["task_id"]

        final = None
        for _ in range(200):
            response = task_client.get(f"/api/tasks/{task_id}")
            assert response.status_code == 200
            final = response.json()
            assert set(final) == {"task_id", "status", "privacy_mode", "created_at", "updated_at"}
            assert objective_marker not in response.text and file_marker not in response.text
            if final["status"] in {"completed", "failed"}:
                break
            time.sleep(0.05)

        assert final["task_id"] == task_id
        assert final["status"] == "completed"

        listed = task_client.get("/api/tasks")
        assert listed.status_code == 200
        assert all(set(item) == {"task_id", "status", "privacy_mode", "created_at", "updated_at"} for item in listed.json())
        assert objective_marker not in listed.text and file_marker not in listed.text

        artifacts = task_client.get(f"/api/tasks/{task_id}/artifacts")
        assert artifacts.status_code == 200
        manifest = artifacts.json()
        assert set(manifest) == {"task_id", "title", "html_url", "markdown_url", "chart_urls"}
        assert manifest["task_id"] == task_id
        assert manifest["html_url"] == f"/api/tasks/{task_id}/artifacts/html"
        assert manifest["markdown_url"] == f"/api/tasks/{task_id}/artifacts/markdown"
        assert objective_marker not in artifacts.text and file_marker not in artifacts.text
        assert "markdown_content" not in manifest and "artifacts" not in manifest

        report = task_client.get(manifest["markdown_url"])
        assert report.status_code == 200
        assert set(report.json()) == {"task_id", "artifact_type", "content"}
        assert report.json()["task_id"] == task_id
        assert report.json()["artifact_type"] == "markdown"
        html = task_client.get(manifest["html_url"])
        assert html.status_code == 200
        assert "sandbox" in html.headers["content-security-policy"]

        stream = task_client.get(f"/api/tasks/{task_id}/events")
        assert stream.status_code == 200
        event_text = stream.text
        assert "event: task.created" in event_text
        assert "event: task_completed" in event_text
        assert "objective\"" not in event_text
        assert "file_path" not in event_text
        sequence_ids = [int(line[4:]) for line in event_text.splitlines() if line.startswith("id: ")]
        assert sequence_ids == sorted(set(sequence_ids))

        replay = task_client.get(
            f"/api/tasks/{task_id}/events",
            headers={"Last-Event-ID": "1"},
        )
        assert replay.status_code == 200
        assert "event: task.created" not in replay.text
        assert all(int(line[4:]) > 1 for line in replay.text.splitlines() if line.startswith("id: "))

        latest = max(sequence_ids)
        terminal_reconnect = task_client.get(
            f"/api/tasks/{task_id}/events", headers={"Last-Event-ID": str(latest)}
        )
        assert terminal_reconnect.status_code == 200
        assert terminal_reconnect.text == ""
        future_reconnect = task_client.get(
            f"/api/tasks/{task_id}/events", headers={"Last-Event-ID": str(latest + 1)}
        )
        assert future_reconnect.status_code == 400

        bad_cursor = task_client.get(
            f"/api/tasks/{task_id}/events",
            headers={"Last-Event-ID": "not-a-number"},
        )
        assert bad_cursor.status_code == 400

