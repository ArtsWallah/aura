import pytest
import pandas as pd
import time
import asyncio
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from starlette.requests import Request
from app.main import app
from app.main import sanitized_request_validation_error
from app.core.config import settings
from app.agent.state import TaskState, TaskStatus
from app.agent.checkpoint import SQLiteCheckpointer
from app.tools.registry import tool_registry
from app.privacy.encryption import encryption_service
import app.api.routes.tasks as task_routes

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
        "objective": "Check system health",
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


def test_request_validation_errors_never_echo_submitted_values():
    objective_marker = "PRIVATE-NESTED-OBJECTIVE-718243"
    path_marker = "PRIVATE-FILE-PATH-918364"
    response = client.post("/api/tasks", json={
        "objective": {"nested": {"secret": objective_marker}},
        "file_path": [path_marker],
    })

    assert response.status_code == 422
    assert objective_marker not in response.text
    assert path_marker not in response.text
    assert "input" not in response.text
    payload = response.json()
    assert payload["message"] == "Request validation failed."
    assert payload["detail"] == [
        {"location": ["body", "field"], "category": "invalid_type"},
        {"location": ["body", "field"], "category": "invalid_type"},
    ]

    unknown_task = client.get(f"/api/tasks/{objective_marker}")
    unknown_tool = client.get(f"/api/tools/{path_marker}")
    assert unknown_task.status_code == unknown_tool.status_code == 404
    assert objective_marker not in unknown_task.text
    assert path_marker not in unknown_tool.text


def test_nested_request_validation_location_is_sanitized():
    marker = "PRIVATE-NESTED-ARGUMENT-718293"
    error = RequestValidationError([{
        "type": "string_type",
        "loc": ("body", "arguments", "private-field", 0),
        "msg": f"Invalid value {marker}",
        "input": {"secret": marker},
    }], body={"arguments": {"private-field": marker}})
    request = Request({
        "type": "http",
        "method": "POST",
        "path": "/api/tools/execute",
        "headers": [],
        "query_string": b"",
        "server": ("testserver", 80),
        "client": ("127.0.0.1", 54321),
        "scheme": "http",
    })

    response = asyncio.run(sanitized_request_validation_error(request, error))
    body = response.body.decode("utf-8")
    assert response.status_code == 422
    assert marker not in body
    assert "private-field" not in body
    assert json.loads(body)["detail"] == [
        {"location": ["body", "field"], "category": "invalid_type"}
    ]

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
        # This request plans dataset inspection, not report generation.
        assert artifacts.status_code == 404

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


@pytest.mark.asyncio
async def test_artifacts_are_manifest_scoped_and_legacy_filename_route_is_disabled(
    tmp_path, monkeypatch
):
    reports_dir = tmp_path / "reports"
    charts_dir = tmp_path / "charts"
    reports_dir.mkdir()
    charts_dir.mkdir()
    monkeypatch.setattr(settings, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(settings, "CHARTS_DIR", charts_dir)

    states = {}
    report_executor = tool_registry._executors["generate_report"]
    report_input = tool_registry._input_models["generate_report"]
    chart_raw_by_task = {}
    task_specs = (
        ("task_scope_a", "report_aaaaaaaaaa", "chart_aaaaaaaaaa", "TASK_A_ONLY"),
        ("task_scope_b", "report_bbbbbbbbbb", "chart_bbbbbbbbbb", "TASK_B_ONLY"),
    )
    for task_id, report_id, chart_id, marker in task_specs:
        chart_raw = f"PNG_BYTES_{marker}".encode()
        chart_path = charts_dir / f"{chart_id}.png.enc"
        chart_path.write_bytes(encryption_service().encrypt_bytes(chart_raw))
        chart_raw_by_task[task_id] = chart_raw
        report = report_executor(report_input(
            title=marker,
            executive_summary="Scoped artifact verification.",
            findings=["Safe fixture."],
            chart_paths=[str(chart_path)],
            task_id=task_id,
        ))

        generated_html = Path(report["html_path"])
        generated_markdown = Path(report["markdown_path"])
        html_path = reports_dir / f"{report_id}.html.enc"
        markdown_path = reports_dir / f"{report_id}.md.enc"
        html_path.write_bytes(generated_html.read_bytes())
        markdown_path.write_bytes(generated_markdown.read_bytes())
        generated_html.unlink()
        generated_markdown.unlink()
        report.update({
            "report_id": report_id,
            "html_path": str(html_path),
            "markdown_path": str(markdown_path),
            "artifacts": [str(markdown_path), str(html_path)],
        })
        states[task_id] = TaskState(
            task_id=task_id,
            objective="private",
            status=TaskStatus.COMPLETED,
            final_deliverable=report,
            artifacts=[str(chart_path), str(markdown_path), str(html_path)],
        )

    class StateLookup:
        def get_task_state(self, task_id):
            return states.get(task_id)

    monkeypatch.setattr(task_routes, "orchestrator", StateLookup())
    task_a = "task_scope_a"
    manifest_response = client.get(f"/api/tasks/{task_a}/artifacts")
    assert manifest_response.status_code == 200
    manifest = manifest_response.json()
    assert manifest["html_url"] == f"/api/tasks/{task_a}/artifacts/html"
    assert manifest["chart_urls"] == [f"/api/tasks/{task_a}/artifacts/chart/0"]

    html = client.get(manifest["html_url"])
    assert html.status_code == 200
    assert "sandbox" in html.headers["content-security-policy"]
    assert f"/api/tasks/{task_a}/artifacts/chart/0" in html.text
    assert "/api/artifacts/" not in html.text
    chart = client.get(manifest["chart_urls"][0])
    assert chart.status_code == 200
    assert encryption_service().decrypt_bytes(
        (charts_dir / "chart_aaaaaaaaaa.png.enc").read_bytes()
    ) == chart_raw_by_task[task_a]

    markdown = client.get(manifest["markdown_url"])
    assert markdown.status_code == 200
    assert f"/api/tasks/{task_a}/artifacts/chart/0" in markdown.json()["content"]
    assert "/api/artifacts/" not in markdown.json()["content"]

    # A filename query cannot switch task A's resolved artifact to task B's.
    query_filename = client.get(
        f"/api/tasks/{task_a}/artifacts/html?filename=report_bbbbbbbbbb.html.enc"
    )
    assert query_filename.status_code == 200
    assert "TASK_A_ONLY" in query_filename.text
    assert "TASK_B_ONLY" not in query_filename.text
    assert client.get(f"/api/tasks/{task_a}/artifacts/chart/1").status_code == 404

    original_report_id = states[task_a].final_deliverable["report_id"]
    states[task_a].final_deliverable["report_id"] = states["task_scope_b"].final_deliverable["report_id"]
    assert client.get(f"/api/tasks/{task_a}/artifacts/html").status_code == 404
    states[task_a].final_deliverable["report_id"] = original_report_id

    assert client.get("/api/artifacts/report_aaaaaaaaaa.html.enc").status_code == 404
    for traversal_url in (
        f"/api/tasks/{task_a}/artifacts/chart/%2e%2e%2f0",
        f"/api/tasks/{task_a}/artifacts/html/C:%5cprivate.html",
        "/api/artifacts/%2e%2e%2freport_aaaaaaaaaa.html.enc",
    ):
        assert client.get(traversal_url).status_code == 404


def test_encrypted_artifact_manifest_survives_restart_and_stays_task_scoped(tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"
    charts_dir = tmp_path / "charts"
    reports_dir.mkdir()
    charts_dir.mkdir()
    monkeypatch.setattr(settings, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(settings, "CHARTS_DIR", charts_dir)
    db_path = tmp_path / "artifact-manifests.db"
    store = SQLiteCheckpointer(db_path)

    for task_id, suffix, marker in (
        ("task_manifest_a", "aaaaaaaaaa", "TASK_A_SECRET_OBJECTIVE"),
        ("task_manifest_b", "bbbbbbbbbb", "TASK_B_SECRET_OBJECTIVE"),
    ):
        report_id = f"report_{suffix}"
        chart_name = f"chart_{suffix}.png.enc"
        html_name = f"{report_id}.html.enc"
        markdown_name = f"{report_id}.md.enc"
        (charts_dir / chart_name).write_bytes(encryption_service().encrypt_bytes(b"PNG" + marker.encode()))
        (reports_dir / html_name).write_bytes(encryption_service().encrypt_bytes(b"<html>safe</html>"))
        (reports_dir / markdown_name).write_bytes(encryption_service().encrypt_bytes(b"# safe report"))
        state = TaskState(
            task_id=task_id,
            objective=marker,
            status=TaskStatus.COMPLETED,
            final_deliverable={
                "report_id": report_id,
                "title": marker,
                "html_path": str(reports_dir / html_name),
                "markdown_path": str(reports_dir / markdown_name),
            },
            artifacts=[
                str(charts_dir / chart_name),
                str(reports_dir / markdown_name),
                str(reports_dir / html_name),
            ],
        )
        store.save_checkpoint(state)

    with closing(sqlite3.connect(db_path)) as conn:
        raw_manifest = conn.execute(
            "SELECT artifact_manifest FROM task_checkpoints WHERE task_id = ?",
            ("task_manifest_a",),
        ).fetchone()[0]
    assert "report_aaaaaaaaaa" not in raw_manifest
    assert "chart_aaaaaaaaaa" not in raw_manifest
    decrypted_manifest = json.loads(
        encryption_service().decrypt_bytes(raw_manifest.encode("ascii"))
    )
    assert {entry["artifact_type"] for entry in decrypted_manifest} == {
        "chart_png", "report_html", "report_markdown"
    }
    assert all(set(entry) == {"artifact_id", "task_id", "artifact_type", "relative_path"} for entry in decrypted_manifest)
    assert all(str(tmp_path) not in entry["relative_path"] for entry in decrypted_manifest)
    assert all(entry["task_id"] == "task_manifest_a" for entry in decrypted_manifest)

    # A new checkpointer and orchestrator lookup represent a backend restart.
    restarted = SQLiteCheckpointer(db_path)
    restored_a = restarted.load_checkpoint("task_manifest_a")
    assert restored_a is not None
    assert restored_a.objective == ""
    assert restored_a.final_deliverable == {
        "report_id": "report_aaaaaaaaaa",
        "title": "Analytical Deliverable",
    }
    assert len(restored_a.artifacts) == 3

    class RestartedLookup:
        def get_task_state(self, task_id):
            return restarted.load_checkpoint(task_id)

    monkeypatch.setattr(task_routes, "orchestrator", RestartedLookup())
    manifest_response = client.get("/api/tasks/task_manifest_a/artifacts")
    assert manifest_response.status_code == 200
    manifest = manifest_response.json()
    assert manifest["chart_urls"] == ["/api/tasks/task_manifest_a/artifacts/chart/0"]
    assert client.get(manifest["html_url"]).content == b"<html>safe</html>"
    assert client.get(manifest["markdown_url"]).json()["content"] == "# safe report"
    chart_response = client.get(manifest["chart_urls"][0])
    assert encryption_service().decrypt_bytes(
        (charts_dir / "chart_aaaaaaaaaa.png.enc").read_bytes()
    ) == b"PNGTASK_A_SECRET_OBJECTIVE"
    assert chart_response.status_code == 200
    assert chart_response.content == b"PNGTASK_A_SECRET_OBJECTIVE"
    assert b"TASK_B_SECRET_OBJECTIVE" not in client.get(manifest["html_url"]).content
    assert client.get("/api/tasks/task_manifest_a/artifacts/chart/1").status_code == 404


@pytest.mark.parametrize("bad_manifest", [
    None,
    "not-a-valid-fernet-token",
    [{"artifact_id": "report_aaaaaaaaaa_html", "task_id": "another_task", "artifact_type": "report_html", "relative_path": "report_aaaaaaaaaa.html.enc"}],
    [{"artifact_id": "report_aaaaaaaaaa_html", "task_id": "task_tampered_manifest", "artifact_type": "report_html", "relative_path": "../reports/report_aaaaaaaaaa.html.enc"}],
    [{"artifact_id": "report_aaaaaaaaaa_html", "task_id": "task_tampered_manifest", "artifact_type": "report_html", "relative_path": "C:\\private\\report_aaaaaaaaaa.html.enc"}],
])
def test_missing_corrupt_and_unsafe_manifests_fail_closed(tmp_path, monkeypatch, bad_manifest):
    reports_dir = tmp_path / "reports"
    charts_dir = tmp_path / "charts"
    reports_dir.mkdir()
    charts_dir.mkdir()
    monkeypatch.setattr(settings, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(settings, "CHARTS_DIR", charts_dir)
    store = SQLiteCheckpointer(tmp_path / "invalid-manifest.db")
    state = TaskState(task_id="task_tampered_manifest", objective="private", status=TaskStatus.COMPLETED)
    store.save_checkpoint(state)
    if bad_manifest is None:
        encoded = None
    elif isinstance(bad_manifest, str):
        encoded = bad_manifest
    else:
        encoded = encryption_service().encrypt_bytes(json.dumps(bad_manifest).encode()).decode("ascii")
    with closing(sqlite3.connect(store.db_path)) as conn:
        conn.execute(
            "UPDATE task_checkpoints SET artifact_manifest = ? WHERE task_id = ?",
            (encoded, state.task_id),
        )
        conn.commit()

    loaded = SQLiteCheckpointer(store.db_path).load_checkpoint(state.task_id)
    assert loaded is not None
    assert loaded.status == TaskStatus.COMPLETED
    assert loaded.artifacts == []
    assert loaded.final_deliverable is None

    class StateLookup:
        def get_task_state(self, task_id):
            return loaded if task_id == state.task_id else None

    monkeypatch.setattr(task_routes, "orchestrator", StateLookup())
    assert client.get(f"/api/tasks/{state.task_id}/artifacts").status_code == 404


def test_restored_artifact_manifest_rejects_symlink_escape_when_supported(tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"
    charts_dir = tmp_path / "charts"
    reports_dir.mkdir()
    charts_dir.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside = outside_dir / "report_aaaaaaaaaa.html.enc"
    outside.write_bytes(encryption_service().encrypt_bytes(b"private"))
    link = reports_dir / outside.name
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Symlink creation unavailable: {exc}")

    monkeypatch.setattr(settings, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(settings, "CHARTS_DIR", charts_dir)
    store = SQLiteCheckpointer(tmp_path / "symlink-manifest.db")
    state = TaskState(task_id="task_symlink_manifest", objective="private", status=TaskStatus.COMPLETED)
    store.save_checkpoint(state)
    crafted = [{
        "artifact_id": "report_aaaaaaaaaa_html",
        "task_id": state.task_id,
        "artifact_type": "report_html",
        "relative_path": outside.name,
    }]
    encrypted = encryption_service().encrypt_bytes(json.dumps(crafted).encode()).decode("ascii")
    with closing(sqlite3.connect(store.db_path)) as conn:
        conn.execute(
            "UPDATE task_checkpoints SET artifact_manifest = ? WHERE task_id = ?",
            (encrypted, state.task_id),
        )
        conn.commit()

    restored = store.load_checkpoint(state.task_id)
    assert restored is not None
    assert restored.artifacts == []
    assert restored.final_deliverable is None

