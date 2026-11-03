from enum import Enum
from typing import Any, Type
from pydantic import BaseModel, Field

class ExecutionLocation(str, Enum):
    LOCAL = "local"
    BROWSER = "browser"
    CLOUD = "cloud"

class PermissionLevel(str, Enum):
    READ_ONLY = "read_only"
    EXECUTE = "execute"
    PRIVILEGED = "privileged"

class ToolCategory(str, Enum):
    DATASET = "dataset"
    ANALYTICS = "analytics"
    VISUALIZATION = "visualization"
    RESEARCH = "research"
    DOCUMENT = "document"
    REPORT = "report"
    SYSTEM = "system"

class ToolParameterSchema(BaseModel):
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None

class ToolDefinition(BaseModel):
    name: str
    category: ToolCategory
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    execution_location: ExecutionLocation = ExecutionLocation.LOCAL
    permission_level: PermissionLevel = PermissionLevel.READ_ONLY
    requires_approval: bool = False
    timeout_seconds: int = 60
    version: str = "1.0.0"

class ToolResult(BaseModel):
    tool_name: str
    success: bool
    data: Any = None
    error: str | None = None
    execution_time_ms: float = 0.0
    artifacts: list[str] = Field(default_factory=list) # paths to charts/reports generated
    metadata: dict[str, Any] = Field(default_factory=dict)

