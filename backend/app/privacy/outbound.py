"""Central outbound HTTP policy for all AURA initiated HTTPX requests."""
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any
from urllib.parse import urlsplit

import httpx
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
    """Apply recognized-pattern redaction; this is not safe for arbitrary text release."""
    return data_redactor.redact_text(value)[0]


class PrivacyAwareAsyncClient(httpx.AsyncClient):
    """Enforce the egress policy at the HTTPX boundary for every request path."""
    async def request(self, method: str, url: Any, **kwargs: Any) -> httpx.Response:
        mode = current_privacy_mode()
        if mode == "local_only":
            raise PermissionError("Outbound network access is blocked by local_only privacy mode.")
        if mode == "hybrid":
            raise PermissionError("Hybrid mode blocks external transmission until content release can be explicitly approved.")
        if mode != "cloud_enabled":
            raise PermissionError("Outbound network access is blocked by an invalid privacy mode.")
        parsed = urlsplit(str(url))
        if parsed.scheme not in {"https", "http"} or not parsed.netloc:
            raise PermissionError("Outbound requests require an explicit HTTP(S) destination.")
        return await super().request(method, url, **kwargs)

    async def send(self, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        """Apply policy at HTTPX's final client boundary, including prebuilt Requests."""
        mode = current_privacy_mode()
        if mode == "local_only":
            raise PermissionError("Outbound network access is blocked by local_only privacy mode.")
        if mode == "hybrid":
            raise PermissionError("Hybrid mode blocks external transmission until content release can be explicitly approved.")
        if mode != "cloud_enabled":
            raise PermissionError("Outbound network access is blocked by an invalid privacy mode.")
        if request.url.scheme not in {"https", "http"} or not request.url.host:
            raise PermissionError("Outbound requests require an explicit HTTP(S) destination.")
        return await super().send(request, **kwargs)
