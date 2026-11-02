import json
import logging
import sqlite3
from pathlib import Path
from types import SimpleNamespace
import uuid

import httpx
import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from app.agent.checkpoint import SQLiteCheckpointer
from app.agent.executor import executor
from app.agent.state import ExecutionStep, TaskState
from app.llm.gemini import GeminiProvider
from app.tools.research_tools import WebResearchInput, web_research
from app.tools.schemas import ToolResult
from app.core.config import settings
from app.core.security import validate_safe_path
from app.core.config import Settings
from app.main import app
from app.privacy.encryption import AtRestEncryption, encryption_service
from app.privacy.outbound import PrivacyAwareAsyncClient, privacy_scope
from app.tools.registry import tool_registry
from app.api.routes import files as files_route

client = TestClient(app, client=("127.0.0.1", 54321))
UI_ORIGIN = "http://localhost:3000"


@pytest.fixture
def managed_test_dir():
    path = settings.DATA_DIR / f"phase1_test_{uuid.uuid4().hex}"
    path.mkdir()
    yield path
    for item in path.iterdir():
        if item.is_file():
            item.unlink()
    path.rmdir()


@pytest.mark.asyncio
async def test_local_only_blocks_http_request_at_client_boundary():
    sent = False

    async def transport(request):
        nonlocal sent
        sent = True
        return httpx.Response(200)

    async with PrivacyAwareAsyncClient(transport=httpx.MockTransport(transport)) as web:
        with privacy_scope("local_only"):
            with pytest.raises(PermissionError, match="local_only"):
                await web.get("https://example.com/private")
            with pytest.raises(PermissionError, match="local_only"):
                await web.send(web.build_request("GET", "https://example.com/private"))
    assert sent is False


@pytest.mark.asyncio
async def test_gemini_and_research_integrations_obey_local_only_policy():
    with privacy_scope("local_only"):
        with pytest.raises(PermissionError, match="local_only"):
            await GeminiProvider(api_key="test-api-key").chat([])
        with pytest.raises(PermissionError, match="local_only"):
            await web_research(WebResearchInput(query="private query"))


@pytest.mark.asyncio
async def test_hybrid_redacts_query_and_json_before_transmission():
    received = {}

    async def transport(request):
        received["query"] = request.url.query.decode()
        received["json"] = json.loads(request.content)
        return httpx.Response(200, json={"ok": True})

    async with PrivacyAwareAsyncClient(transport=httpx.MockTransport(transport)) as web:
        with privacy_scope("hybrid"):
            await web.post("https://example.com/search?email=user%40example.com", json={"prompt": "Contact user@example.com", "rows": [{"x": 1}], "sample_data": ["private"]})
    assert "user%40example.com" not in received["query"]
    assert "%5BREDACTED_EMAIL%5D" in received["query"]
    assert "user@example.com" not in json.dumps(received["json"])
    assert "rows" not in received["json"]
    assert "sample_data" not in received["json"]


@pytest.mark.asyncio
async def test_hybrid_blocks_uninspectable_binary_payloads():
    async with PrivacyAwareAsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200))) as web:
        with privacy_scope("hybrid"):
            with pytest.raises(PermissionError, match="binary"):
                await web.post("https://example.com/upload", content=b"\xff\x00\xfe")


def test_invalid_privacy_modes_fail_closed():
    with pytest.raises(ValueError, match="Unknown privacy mode"):
        with privacy_scope("typo"):
            pass


def test_server_host_must_be_loopback():
    with pytest.raises(ValueError, match="loopback"):
        Settings(HOST="0.0.0.0")


def test_fernet_encryption_round_trip_and_invalid_keys():
    key = Fernet.generate_key()
    cipher = AtRestEncryption(key)
    encrypted = cipher.encrypt_bytes(b"private row data")
    assert b"private row data" not in encrypted
    assert cipher.decrypt_bytes(encrypted) == b"private row data"
    with pytest.raises(ValueError, match="Fernet key"):
        AtRestEncryption("hardcoded-short-key")
    with pytest.raises(ValueError, match="authenticated"):
        AtRestEncryption(Fernet.generate_key()).decrypt_bytes(encrypted)


def test_encryption_key_survives_service_recreation_and_wrong_key_fails(monkeypatch):
    key = Fernet.generate_key().decode("ascii")
    monkeypatch.setattr(settings, "ENCRYPTION_KEY", key)
    encrypted = encryption_service().encrypt_bytes(b"persistent across restarts")

    # A new service instance represents a new application process using persisted configuration.
    monkeypatch.setattr(settings, "ENCRYPTION_KEY", key)
    assert encryption_service().decrypt_bytes(encrypted) == b"persistent across restarts"

    monkeypatch.setattr(settings, "ENCRYPTION_KEY", Fernet.generate_key().decode("ascii"))
    with pytest.raises(ValueError, match="authenticated"):
        encryption_service().decrypt_bytes(encrypted)


@pytest.mark.asyncio
async def test_validation_and_agent_logs_do_not_expose_submitted_values(monkeypatch, caplog):
    marker = "PRIVATE-SUBMITTED-VALUE-938271"
    with caplog.at_level(logging.INFO):
        invalid = await tool_registry.execute_tool(
            "inspect_dataset", {"file_path": "managed.csv.enc", "sample_rows": marker}
        )
        assert marker not in (invalid.error or "")

        async def failed_tool(*args, **kwargs):
            return ToolResult(tool_name="inspect_dataset", success=False, error=marker)

        async def no_wait(_seconds):
            return None

        monkeypatch.setattr(tool_registry, "execute_tool", failed_tool)
        monkeypatch.setattr("app.agent.executor.asyncio.sleep", no_wait)
        step = ExecutionStep(
            step_id="private-step", title=marker, description="test", capability="dataset",
            tool_name="inspect_dataset", parameters={"file_path": "managed.csv.enc"}, max_retries=0
        )
        await executor.execute_step(step, TaskState(task_id="private-log-test", objective="safe"))

    assert marker not in caplog.text


def test_task_checkpoint_is_encrypted_at_rest(managed_test_dir):
    store = SQLiteCheckpointer(managed_test_dir / "private.db")
    state = TaskState(task_id="private-test", objective="user@example.com confidential")
    store.save_checkpoint(state)
    conn = sqlite3.connect(store.db_path)
    raw = conn.execute("SELECT state_json FROM task_checkpoints").fetchone()[0]
    conn.close()
    assert "user@example.com" not in raw
    decrypted = encryption_service().decrypt_bytes(raw.encode("ascii")).decode("utf-8")
    assert "user@example.com" not in decrypted
    assert store.load_checkpoint(state.task_id).objective == ""
    assert "objective" not in store.list_checkpoints()[0]


def test_upload_is_encrypted_unique_and_never_writes_plaintext(managed_test_dir, monkeypatch):
    monkeypatch.setattr(settings, "UPLOADS_DIR", managed_test_dir)
    raw = b"category,revenue\nA,100\n"
    first = client.post("/api/files/upload", files={"file": ("sales.csv", raw, "text/csv")})
    second = client.post("/api/files/upload", files={"file": ("sales.csv", raw, "text/csv")})
    assert first.status_code == second.status_code == 200
    one, two = first.json(), second.json()
    assert one["file_path"] != two["file_path"]
    assert one["encrypted_at_rest"] is True
    stored = Path(one["file_path"]).read_bytes()
    assert raw not in stored
    assert encryption_service().decrypt_bytes(stored) == raw
    assert not (managed_test_dir / "sales.csv").exists()


def test_upload_rejects_invalid_type_and_size(managed_test_dir, monkeypatch):
    monkeypatch.setattr(settings, "UPLOADS_DIR", managed_test_dir)
    bad_type = client.post("/api/files/upload", files={"file": ("program.py", b"print(1)", "text/plain")})
    assert bad_type.status_code == 400
    monkeypatch.setattr(settings, "MAX_UPLOAD_BYTES", 3)
    too_large = client.post("/api/files/upload", files={"file": ("large.csv", b"1234", "text/csv")})
    assert too_large.status_code == 413


def test_upload_collision_does_not_delete_preexisting_staging_file(managed_test_dir, monkeypatch):
    monkeypatch.setattr(settings, "UPLOADS_DIR", managed_test_dir)
    collision_id = "fixedcollision"
    staging = managed_test_dir / f".upload_{collision_id}.csv.enc.tmp"
    sentinel = b"pre-existing staging file must be preserved"
    staging.write_bytes(sentinel)
    monkeypatch.setattr(files_route.uuid, "uuid4", lambda: SimpleNamespace(hex=collision_id))
    safe_client = TestClient(app, client=("127.0.0.1", 54321), raise_server_exceptions=False)

    response = safe_client.post("/api/files/upload", files={"file": ("sales.csv", b"x,y\n1,2\n", "text/csv")})

    assert response.status_code == 500
    assert staging.read_bytes() == sentinel


def test_upload_fails_closed_without_encryption_key(managed_test_dir, monkeypatch):
    monkeypatch.setattr(settings, "UPLOADS_DIR", managed_test_dir)
    monkeypatch.setattr(settings, "ENCRYPTION_KEY", None)
    response = client.post("/api/files/upload", files={"file": ("sales.csv", b"x,y\n1,2\n", "text/csv")})
    assert response.status_code == 503
    assert list(managed_test_dir.iterdir()) == []


def test_remote_clients_and_unexpected_origins_are_rejected():
    remote = TestClient(app, client=("192.0.2.10", 54321))
    assert remote.get("/health").status_code == 403
    assert client.get("/health", headers={"Origin": "https://attacker.example"}).status_code == 403
    assert client.post("/api/settings", json={"privacy_mode": "local_only"}).status_code == 403
    assert client.get("/api/settings").status_code == 403
    assert client.get("/api/settings", headers={"Origin": UI_ORIGIN}).status_code == 200


def test_direct_tool_execution_is_restricted():
    response = client.post("/api/tools/execute", headers={"Origin": UI_ORIGIN}, json={"tool_name": "generate_report", "arguments": {"title": "x", "executive_summary": "x", "findings": []}, "user_approval_granted": True})
    assert response.status_code == 403


def test_file_boundary_rejects_traversal_and_arbitrary_paths(managed_test_dir):
    outside = settings.DATA_DIR / f"secret_{uuid.uuid4().hex}.csv"
    outside.write_text("private", encoding="utf-8")
    try:
        with pytest.raises(PermissionError):
            validate_safe_path(outside)
        with pytest.raises(PermissionError):
            validate_safe_path(settings.UPLOADS_DIR / ".." / outside.name)
    finally:
        outside.unlink()


def test_file_boundary_rejects_symlink_escape_when_supported(managed_test_dir, monkeypatch):
    monkeypatch.setattr(settings, "UPLOADS_DIR", managed_test_dir)
    outside = managed_test_dir.parent / f"outside_{uuid.uuid4().hex}.csv"
    outside.write_text("private", encoding="utf-8")
    link = managed_test_dir / "linked.csv"
    try:
        try:
            link.symlink_to(outside)
        except (OSError, NotImplementedError) as exc:
            pytest.skip(f"Symlink creation is unavailable on this Windows environment: {exc}")
        with pytest.raises(PermissionError):
            validate_safe_path(link)
    finally:
        if link.is_symlink():
            link.unlink()
        if outside.exists():
            outside.unlink()


def test_report_escapes_script_tags_handlers_attributes_and_is_encrypted():
    malicious = '<script>alert(1)</script><img src=x onerror="alert(2)"><div title="x">ok</div>'
    result = tool_registry._executors["generate_report"](
        tool_registry._input_models["generate_report"](
            title=malicious,
            executive_summary=malicious,
            findings=[malicious],
            recommendations=[malicious],
            assumptions_and_limitations=[malicious],
        )
    )
    html_bytes = Path(result["html_path"]).read_bytes()
    assert malicious.encode() not in html_bytes
    html = encryption_service().decrypt_bytes(html_bytes).decode()
    assert "&lt;script&gt;" in html
    assert "&lt;img src=x onerror=&quot;alert(2)&quot;&gt;" in html
    assert "<script>alert(1)</script>" not in html
    response = client.get(f"/api/artifacts/{Path(result['html_path']).name}")
    assert response.status_code == 200
    assert "sandbox" in response.headers["content-security-policy"]
    assert "<script>alert(1)</script>" not in response.text


def test_encrypted_uploaded_dataset_remains_usable_by_analytics(managed_test_dir, monkeypatch):
    monkeypatch.setattr(settings, "UPLOADS_DIR", managed_test_dir)
    raw = b"group,value\nA,1\nB,3\n"
    path = managed_test_dir / "analytics.csv.enc"
    path.write_bytes(encryption_service().encrypt_bytes(raw))
    result = tool_registry._executors["descriptive_statistics"](
        tool_registry._input_models["descriptive_statistics"](file_path=str(path))
    )
    assert result["statistics"]["value"]["mean"] == 2

