import time
import inspect
from typing import Callable, Any, Type
from pydantic import BaseModel, ValidationError

from app.tools.schemas import (
    ToolDefinition,
    ToolResult,
    ToolCategory,
    ExecutionLocation,
    PermissionLevel
)
from app.core.logging import logger

class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}
        self._executors: dict[str, Callable] = {}
        self._input_models: dict[str, Type[BaseModel]] = {}

    def register(
        self,
        name: str,
        category: ToolCategory,
        description: str,
        input_model: Type[BaseModel],
        output_model: Type[BaseModel] | None = None,
        execution_location: ExecutionLocation = ExecutionLocation.LOCAL,
        permission_level: PermissionLevel = PermissionLevel.READ_ONLY,
        requires_approval: bool = False,
        timeout_seconds: int = 60,
        version: str = "1.0.0"
    ):
        """Decorator or method to register a typed tool."""
        def decorator(func: Callable):
            input_schema = input_model.model_json_schema()
            output_schema = output_model.model_json_schema() if output_model else {"type": "object"}
            
            tool_def = ToolDefinition(
                name=name,
                category=category,
                description=description,
                input_schema=input_schema,
                output_schema=output_schema,
                execution_location=execution_location,
                permission_level=permission_level,
                requires_approval=requires_approval,
                timeout_seconds=timeout_seconds,
                version=version
            )
            
            self._tools[name] = tool_def
            self._executors[name] = func
            self._input_models[name] = input_model
            logger.info(f"Registered tool '{name}' [{category.value}] - {description[:50]}...")
            return func
            
        return decorator

    def get_tool_definition(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def list_tools(self, category: ToolCategory | None = None) -> list[ToolDefinition]:
        if category:
            return [t for t in self._tools.values() if t.category == category]
        return list(self._tools.values())

    def get_tool_schemas_for_llm(self) -> list[dict[str, Any]]:
        """Returns tool declarations formatted for LLM function/tool calling."""
        declarations = []
        for name, tool_def in self._tools.items():
            declarations.append({
                "name": name,
                "description": tool_def.description,
                "parameters": tool_def.input_schema
            })
        return declarations

    async def execute_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        user_approval_granted: bool = False
    ) -> ToolResult:
        """
        Validates arguments against the Pydantic schema and executes the tool with latency tracking.
        """
        if name not in self._tools:
            return ToolResult(
                tool_name=name,
                success=False,
                error=f"Tool '{name}' is not registered in AURA tool registry."
            )
            
        tool_def = self._tools[name]
        executor = self._executors[name]
        input_model = self._input_models[name]
        
        # Check approval
        if tool_def.requires_approval and not user_approval_granted:
            return ToolResult(
                tool_name=name,
                success=False,
                error=f"Tool '{name}' requires explicit user approval before execution."
            )

        # Validate input schema
        try:
            validated_input = input_model(**arguments)
        except ValidationError as e:
            return ToolResult(
                tool_name=name,
                success=False,
                error=f"Input validation error for tool '{name}': {e.errors()}"
            )
        except Exception as e:
            return ToolResult(
                tool_name=name,
                success=False,
                error=f"Malformed arguments for tool '{name}': {str(e)}"
            )

        # Execute with latency tracking
        start_time = time.perf_counter()
        try:
            # Check if executor is async or sync
            if inspect.iscoroutinefunction(executor):
                raw_result = await executor(validated_input)
            else:
                raw_result = executor(validated_input)
                
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            
            # Format output
            if isinstance(raw_result, ToolResult):
                raw_result.execution_time_ms = elapsed_ms
                return raw_result
                
            artifacts = []
            if isinstance(raw_result, dict) and "artifacts" in raw_result:
                artifacts = raw_result.get("artifacts", [])
                
            return ToolResult(
                tool_name=name,
                success=True,
                data=raw_result,
                execution_time_ms=elapsed_ms,
                artifacts=artifacts
            )
            
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(f"Error executing tool '{name}': {str(e)}", exc_info=True)
            return ToolResult(
                tool_name=name,
                success=False,
                error=f"Execution error in '{name}': {str(e)}",
                execution_time_ms=elapsed_ms
            )

# Global singleton
tool_registry = ToolRegistry()

