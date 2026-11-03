from fastapi import APIRouter, HTTPException
from typing import Any
from pydantic import BaseModel
from app.tools.registry import tool_registry
from app.tools.schemas import ToolDefinition, ToolCategory

router = APIRouter(prefix="/tools", tags=["Tools"])

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
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found.")
    return tool

@router.post("/execute")
async def execute_tool_direct(request: ExecuteToolRequest):
    result = await tool_registry.execute_tool(
        name=request.tool_name,
        arguments=request.arguments,
        user_approval_granted=request.user_approval_granted
    )
    return result

