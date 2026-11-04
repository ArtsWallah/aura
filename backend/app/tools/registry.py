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


DIRECT_READ_ONLY_TOOL_ALLOWLIST = frozenset({
    "inspect_dataset", "descriptive_statistics", "group_comparison",
    "correlation_analysis", "detect_outliers", "read_document",
    "search_document_chunks", "system_health",
})

class ToolRegistry:
    def __init__(self):
        # Process-local identity lets planning, handoff, and execution prove
        # they validated against the same registry instance.
        self.registry_id = uuid.uuid4().hex
        self._tools: dict[str, ToolDefinition] = {}
        self._executors: dict[str, Callable] = {}
        self._input_models: dict[str, Type[BaseModel]] = {}
        self._output_models: dict[str, Type[BaseModel]] = {}
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
        """Decorator or method to register a typed tool with mandatory input and output contracts."""
        if input_model is None or not (isinstance(input_model, type) and issubclass(input_model, BaseModel)):
            raise TypeError(
                f"Tool '{name}' input_model must be an explicit subclass of pydantic.BaseModel, got {input_model!r}."
            )
        if output_model is None:
            raise TypeError(f"Tool '{name}' must declare an explicit output_model.")
        if not (isinstance(output_model, type) and issubclass(output_model, BaseModel)):
            raise TypeError(
                f"Tool '{name}' output_model must be an explicit subclass of pydantic.BaseModel, got {output_model!r}."
            )

        def decorator(func: Callable):
            input_schema = input_model.model_json_schema()
            output_schema = output_model.model_json_schema()
            
            tool_def = ToolDefinition(
                name=name,
                category=category,
                description=description,
                input_schema=input_schema,
                output_schema=output_schema,
                output_model=output_model,
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
            self._output_models[name] = output_model
            logger.info(f"Registered tool '{name}' [{category.value}] - {description[:50]}...")
            return func
            
        return decorator

    def get_output_model(self, name: str) -> Type[BaseModel] | None:
        return self._output_models.get(name)

    def get_input_model(self, name: str) -> Type[BaseModel] | None:
        return self._input_models.get(name)

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
        from app.agent.execution_handoff import (
            require_authorized_tool_arguments,
            test_legacy_tool_invocation,
        )
        from app.core.config import test_harness_enabled

        if test_harness_enabled():
            with test_legacy_tool_invocation(name, arguments):
                return await self._execute_tool(
                    name=name,
                    arguments=arguments,
                    user_approval_granted=user_approval_granted,
                    cancellation_check=cancellation_check,
                )
        require_authorized_tool_arguments(name, arguments)
        return await self._execute_tool(
            name=name,
            arguments=arguments,
            user_approval_granted=user_approval_granted,
            cancellation_check=cancellation_check,
        )

    async def execute_direct_read_only_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        """Run only the explicit standalone read-only tools exposed by the API."""
        tool_def = self._tools.get(name)
        if (
            name not in DIRECT_READ_ONLY_TOOL_ALLOWLIST
            or tool_def is None
            or tool_def.permission_level != PermissionLevel.READ_ONLY
            or tool_def.execution_location != ExecutionLocation.LOCAL
            or tool_def.requires_approval
        ):
            raise PermissionError("This tool is not available for direct execution.")
        from app.agent.execution_handoff import direct_read_only_tool_invocation

        with direct_read_only_tool_invocation(name, arguments):
            return await self._execute_tool(name=name, arguments=arguments, user_approval_granted=False)

    async def _execute_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        user_approval_granted: bool = False,
        cancellation_check: Callable[[], bool] | None = None,
    ) -> ToolResult:
        """Validated execution implementation behind the handoff/read-only gates."""
        from app.agent.execution_handoff import require_authorized_tool_arguments

        require_authorized_tool_arguments(name, arguments, consume=True)
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

            # If the tool returned an explicit failed ToolResult, preserve it directly.
            is_wrapped_result = isinstance(raw_result, ToolResult)
            if is_wrapped_result and not raw_result.success:
                raw_result.execution_time_ms = elapsed_ms
                return raw_result

            candidate_payload = raw_result.data if is_wrapped_result else raw_result

            # Retrieve registered output model
            output_model = self._output_models.get(name)
            if output_model is None:
                logger.error("Tool '%s' lacks a registered output model.", name)
                return ToolResult(
                    tool_name=name,
                    success=False,
                    error=f"Output validation failed for tool '{name}'.",
                    execution_time_ms=elapsed_ms,
                )

            # Validate returned value against registered output contract
            try:
                validated_output = output_model.model_validate(candidate_payload)
            except ValidationError:
                logger.error("Tool '%s' output failed contract validation.", name)
                return ToolResult(
                    tool_name=name,
                    success=False,
                    error=f"Output validation failed for tool '{name}'.",
                    execution_time_ms=elapsed_ms,
                )
            except Exception:
                logger.error("Malformed output returned by tool '%s'.", name)
                return ToolResult(
                    tool_name=name,
                    success=False,
                    error=f"Malformed output returned by tool '{name}'.",
                    execution_time_ms=elapsed_ms,
                )

            # Convert validated model to JSON-compatible format
            validated_data = validated_output.model_dump(mode="json")

            artifacts = []
            if isinstance(validated_data, dict) and "artifacts" in validated_data:
                raw_artifacts = validated_data.get("artifacts", [])
                if isinstance(raw_artifacts, list):
                    artifacts = [str(a) for a in raw_artifacts]
            elif is_wrapped_result and raw_result.artifacts:
                artifacts = list(raw_result.artifacts)

            if is_wrapped_result:
                raw_result.data = validated_data
                raw_result.artifacts = artifacts
                raw_result.execution_time_ms = elapsed_ms
                return raw_result

            return ToolResult(
                tool_name=name,
                success=True,
                data=validated_data,
                execution_time_ms=elapsed_ms,
                artifacts=artifacts,
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

