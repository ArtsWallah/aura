from fastapi import APIRouter
from typing import Literal
from pydantic import BaseModel
from app.core.config import settings
from app.core.security import mask_secret
from app.llm.router import llm_router
from app.llm.gemini import GeminiProvider

router = APIRouter(prefix="/settings", tags=["Settings"])

class UpdateSettingsRequest(BaseModel):
    privacy_mode: Literal["local_only", "hybrid", "cloud_enabled"] | None = None
    gemini_api_key: str | None = None
    gemini_model: str | None = None
    default_llm_provider: Literal["gemini", "test_reasoner"] | None = None

@router.get("")
def get_settings():
    return {
        "project_name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "default_privacy_mode": settings.DEFAULT_PRIVACY_MODE,
        "default_llm_provider": settings.DEFAULT_LLM_PROVIDER,
        "gemini_configured": bool(settings.GEMINI_API_KEY),
        "gemini_api_key_masked": mask_secret(settings.GEMINI_API_KEY),
        "gemini_model": settings.GEMINI_MODEL,
        "data_retention_hours": settings.DATA_RETENTION_HOURS,
        "available_providers": llm_router.list_providers()
    }

@router.post("")
def update_settings(req: UpdateSettingsRequest):
    if req.privacy_mode:
        settings.DEFAULT_PRIVACY_MODE = req.privacy_mode
    if req.gemini_api_key is not None:
        settings.GEMINI_API_KEY = req.gemini_api_key.strip() or None
        llm_router._providers["gemini"] = GeminiProvider()
    if req.gemini_model:
        settings.GEMINI_MODEL = req.gemini_model
        llm_router._providers["gemini"] = GeminiProvider()
    if req.default_llm_provider:
        settings.DEFAULT_LLM_PROVIDER = req.default_llm_provider

    return {
        "message": "Settings updated successfully.",
        "settings": get_settings()
    }

