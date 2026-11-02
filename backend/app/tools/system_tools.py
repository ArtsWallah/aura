import sys
import shutil
from typing import Any
from pydantic import BaseModel, Field

from app.tools.schemas import ToolCategory, ExecutionLocation, PermissionLevel
from app.tools.registry import tool_registry
from app.core.config import settings

class SystemHealthInput(BaseModel):
    check_storage: bool = Field(default=True, description="Whether to check available disk storage.")

@tool_registry.register(
    name="system_health",
    category=ToolCategory.SYSTEM,
    description="Reports system operational status, Python environment version, available tool counts, and storage metrics.",
    input_model=SystemHealthInput,
    execution_location=ExecutionLocation.LOCAL,
    permission_level=PermissionLevel.READ_ONLY,
    retry_safe=True,
)
def system_health(params: SystemHealthInput) -> dict[str, Any]:
    storage_info = {}
    if params.check_storage:
        total, used, free = shutil.disk_usage(settings.DATA_DIR)
        storage_info = {
            "total_gb": round(total / (1024 ** 3), 2),
            "used_gb": round(used / (1024 ** 3), 2),
            "free_gb": round(free / (1024 ** 3), 2)
        }

    return {
        "status": "healthy",
        "python_version": sys.version.split()[0],
        "registered_tools_count": len(tool_registry.list_tools()),
        "privacy_mode_default": settings.DEFAULT_PRIVACY_MODE,
        "storage": storage_info,
        "directories_initialized": True
    }

