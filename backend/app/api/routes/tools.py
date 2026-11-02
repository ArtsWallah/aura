from fastapi import APIRouter, HTTPException
from typing import Any
from pydantic import BaseModel
from app.tools.registry import tool_registry
from app.tools.schemas import ToolDefinition, ToolCategory
from app.tools.schemas import PermissionLevel, ExecutionLocation

router = APIRouter(prefix="/tools", tags=["Tools"])
DIRECT_TOOL_ALLOWLIST = {
    "inspect_dataset", "descriptive_statistics", "group_comparison",
    "correlation_analysis", "detect_outliers", "read_document",
    "search_document_chunks", "system_health",
}

class ExecuteToolRequest(BaseModel):
    tool_name: str
    arguments: dict[str, Any]
    user_approval_granted: bool = False

@router.get("", response_model=list[ToolDefinition])
def list_tools(category: ToolCategory | None = None):
    return tool_registry.list_tools(category=category)

@router.get("/{tool_name}", response_model=ToolDefinition)
def get_tool(tool_name: str):
    tool = tool_registry.get_tool_definition(tool_name)
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found.")
    return tool

@router.post("/execute")
async def execute_tool_direct(request: ExecuteToolRequest):
    definition = tool_registry.get_tool_definition(request.tool_name)
    if (
        request.tool_name not in DIRECT_TOOL_ALLOWLIST
        or definition is None
        or definition.permission_level != PermissionLevel.READ_ONLY
        or definition.execution_location != ExecutionLocation.LOCAL
    ):
        raise HTTPException(status_code=403, detail="This tool is not available for direct execution.")
    result = await tool_registry.execute_tool(
        name=request.tool_name,
        arguments=request.arguments,
        user_approval_granted=False
    )
    return result

