"""Central outbound HTTP policy for all AURA initiated HTTPX requests."""
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx
import json

from app.core.config import settings
from app.privacy.redaction import data_redactor

_active_mode: ContextVar[str | None] = ContextVar("aura_privacy_mode", default=None)


def current_privacy_mode() -> str:
    return _active_mode.get() or settings.DEFAULT_PRIVACY_MODE


@contextmanager
def privacy_scope(mode: str):
    if mode not in {"local_only", "hybrid", "cloud_enabled"}:
        raise ValueError("Unknown privacy mode; refusing to select a less restrictive policy.")
    token = _active_mode.set(mode)
    try:
        yield
    finally:
        _active_mode.reset(token)


def redact_external_text(value: str) -> str:
    return data_redactor.redact_text(value)[0]


def _sanitize(value: Any) -> Any:
    if isinstance(value, str):
        return redact_external_text(value)
    if isinstance(value, list):
        return [_sanitize(item) for item in value[:100]]
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items() if key.lower() not in {"sample_data", "raw_rows", "rows", "records", "data_uri"}}
    if isinstance(value, tuple):
        return tuple(_sanitize(item) for item in value[:100])
    return value


def _sanitize_url(url: str) -> str:
    parts = urlsplit(url)
    credential_parameters = {"key", "api_key", "apikey", "token", "access_token", "authorization"}
    query = urlencode([
        (key, value if key.lower() in credential_parameters else redact_external_text(value))
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
    ])
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, parts.fragment))


class PrivacyAwareAsyncClient(httpx.AsyncClient):
    """Refuse all HTTP egress in local_only and sanitize hybrid payloads centrally."""
    async def request(self, method: str, url: Any, **kwargs: Any) -> httpx.Response:
        mode = current_privacy_mode()
        if mode == "local_only":
            raise PermissionError("Outbound network access is blocked by local_only privacy mode.")
        if mode not in {"hybrid", "cloud_enabled"}:
            raise PermissionError("Outbound network access is blocked by an invalid privacy mode.")
        parsed = urlsplit(str(url))
        if parsed.scheme not in {"https", "http"} or not parsed.netloc:
            raise PermissionError("Outbound requests require an explicit HTTP(S) destination.")
        if mode == "hybrid":
            url = _sanitize_url(str(url))
            if "json" in kwargs:
                kwargs["json"] = _sanitize(kwargs["json"])
            if kwargs.get("params") is not None:
                kwargs["params"] = _sanitize(kwargs["params"])
            if isinstance(kwargs.get("data"), (str, dict, list, tuple)):
                kwargs["data"] = _sanitize(kwargs["data"])
            if isinstance(kwargs.get("content"), str):
                kwargs["content"] = redact_external_text(kwargs["content"])
            elif isinstance(kwargs.get("content"), bytes):
                try:
                    kwargs["content"] = redact_external_text(kwargs["content"].decode("utf-8"))
                except UnicodeDecodeError as exc:
                    raise PermissionError("Hybrid mode blocks uninspectable binary request bodies.") from exc
            if kwargs.get("files"):
                raise PermissionError("Hybrid mode blocks multipart uploads to external providers.")
        return await super().request(method, url, **kwargs)

    async def send(self, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        """Apply policy at HTTPX's final client boundary, including prebuilt Requests."""
        mode = current_privacy_mode()
        if mode == "local_only":
            raise PermissionError("Outbound network access is blocked by local_only privacy mode.")
        if mode not in {"hybrid", "cloud_enabled"}:
            raise PermissionError("Outbound network access is blocked by an invalid privacy mode.")
        if request.url.scheme not in {"https", "http"} or not request.url.host:
            raise PermissionError("Outbound requests require an explicit HTTP(S) destination.")
        if mode == "hybrid":
            request.url = httpx.URL(_sanitize_url(str(request.url)))
            content_type = request.headers.get("content-type", "").lower()
            if "multipart/form-data" in content_type:
                raise PermissionError("Hybrid mode blocks multipart uploads to external providers.")
            try:
                body = await request.aread()
            except Exception as exc:
                raise PermissionError("Hybrid mode blocks uninspectable request bodies.") from exc
            if body:
                if "application/json" in content_type:
                    try:
                        body = json.dumps(_sanitize(json.loads(body.decode("utf-8")))).encode("utf-8")
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise PermissionError("Hybrid mode blocks uninspectable JSON request bodies.") from exc
                else:
                    try:
                        body = redact_external_text(body.decode("utf-8")).encode("utf-8")
                    except UnicodeDecodeError as exc:
                        raise PermissionError("Hybrid mode blocks uninspectable binary request bodies.") from exc
                request._content = body
                request.stream = httpx.ByteStream(body)
                request.headers["content-length"] = str(len(body))
        return await super().send(request, **kwargs)
