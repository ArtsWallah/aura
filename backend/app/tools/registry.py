import asyncio
import time
import inspect
import uuid
import weakref
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
from app.core.config import settings
from app.privacy.outbound import current_privacy_mode

class ToolRegistry:
    def __init__(self):
        # Process-local identity lets planning, handoff, and execution prove
        # they validated against the same registry instance.
        self.registry_id = uuid.uuid4().hex
        self._tools: dict[str, ToolDefinition] = {}
        self._executors: dict[str, Callable] = {}
        self._input_models: dict[str, Type[BaseModel]] = {}
        self._loop_semaphores: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

    def _execution_semaphore(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        semaphore = self._loop_semaphores.get(loop)
        if semaphore is None:
            semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_TOOL_EXECUTIONS)
            self._loop_semaphores[loop] = semaphore
        return semaphore

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
        retry_safe: bool = False,
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
                retry_safe=retry_safe,
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

    def validate_arguments(self, name: str, arguments: dict[str, Any]) -> BaseModel:
        """Validate a tool payload without exposing rejected input values."""
        input_model = self._input_models.get(name)
        if input_model is None:
            raise KeyError("Tool is not registered.")
        return input_model.model_validate(arguments)

    async def execute_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        user_approval_granted: bool = False,
        cancellation_check: Callable[[], bool] | None = None,
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

        if tool_def.execution_location == ExecutionLocation.BROWSER:
            return ToolResult(tool_name=name, success=False, error=f"Tool '{name}' cannot run in the backend.")
        if tool_def.execution_location == ExecutionLocation.CLOUD and current_privacy_mode() == "local_only":
            return ToolResult(tool_name=name, success=False, error=f"Tool '{name}' is blocked by local-only privacy mode.")
        if tool_def.permission_level != PermissionLevel.READ_ONLY and not user_approval_granted:
            return ToolResult(tool_name=name, success=False, error=f"Tool '{name}' requires execution approval.")
        
        # Check approval
        if tool_def.requires_approval and not user_approval_granted:
            return ToolResult(
                tool_name=name,
                success=False,
                error=f"Tool '{name}' requires explicit user approval before execution."
            )

        # Validate input schema
        try:
            validated_input = input_model.model_validate(arguments)
        except ValidationError:
            return ToolResult(
                tool_name=name,
                success=False,
                error=f"Input validation failed for tool '{name}'."
            )
        except Exception:
            return ToolResult(
                tool_name=name,
                success=False,
                error=f"Malformed arguments for tool '{name}'."
            )

        # Execute with latency tracking
        start_time = time.perf_counter()
        try:
            timeout_seconds = min(tool_def.timeout_seconds, settings.STEP_TIMEOUT_SECONDS)
            async with self._execution_semaphore():
                if cancellation_check is not None and cancellation_check():
                    return ToolResult(tool_name=name, success=False, error="Tool execution cancelled before start.")
                if inspect.iscoroutinefunction(executor):
                    raw_result = await asyncio.wait_for(executor(validated_input), timeout=timeout_seconds)
                else:
                    # A worker thread keeps sync tools off the API event loop. Threads
                    # cannot be killed safely, so timeout/cancellation waits for this
                    # invocation to reach a safe return boundary before releasing capacity.
                    invocation = asyncio.create_task(asyncio.to_thread(executor, validated_input))
                    try:
                        raw_result = await asyncio.wait_for(asyncio.shield(invocation), timeout=timeout_seconds)
                    except asyncio.TimeoutError as exc:
                        try:
                            await asyncio.shield(invocation)
                        except Exception:
                            pass
                        raise TimeoutError("Tool execution exceeded its configured deadline.") from exc
                    except asyncio.CancelledError:
                        try:
                            await asyncio.shield(invocation)
                        except Exception:
                            pass
                        raise
                
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
            
        except TimeoutError:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(f"Tool '{name}' exceeded its configured timeout.")
            return ToolResult(
                tool_name=name,
                success=False,
                error=f"Tool '{name}' timed out.",
                execution_time_ms=elapsed_ms
            )
        except Exception:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(f"Error executing tool '{name}'.")
            return ToolResult(
                tool_name=name,
                success=False,
                error=f"Execution error in '{name}'.",
                execution_time_ms=elapsed_ms
            )

# Global singleton
tool_registry = ToolRegistry()

