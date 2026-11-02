import sys
from fastapi import APIRouter
from app.core.config import settings
from app.tools.registry import tool_registry
from app.models.registry import model_registry

router = APIRouter(tags=["Health"])

@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "python_version": sys.version.split()[0],
        "registered_tools_count": len(tool_registry.list_tools()),
        "registered_models_count": len(model_registry.list_models()),
        "default_privacy_mode": settings.DEFAULT_PRIVACY_MODE,
        "environment": settings.ENVIRONMENT
    }
