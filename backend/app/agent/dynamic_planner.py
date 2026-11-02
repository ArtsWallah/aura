"""Independent, validated plan construction for Phase 3.1.3.

The planner is not connected to task execution or orchestration. Its local
path is deterministic and only emits plans when registered tool arguments can
be filled from the objective and explicitly supplied, authorized metadata.
"""

from __future__ import annotations

import json
import re
import uuid
from enum import Enum
from typing import Any, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.agent.intelligence_contracts import (
    IntentAssessment,
    IntentCategory,
    PlanningResponse,
    PlanningResponseType,
    ProviderIdentity,
    RequiredInputKind,
)
from app.agent.plan_validation import PlanValidationError, validate_execution_plan
from app.agent.state import ExecutionPlan, ExecutionStep
from app.core.config import settings
from app.core.security import managed_file_suffix, validate_safe_path
from app.llm.base import BaseLLMProvider, LLMMessage
from app.privacy.outbound import privacy_scope
from app.tools.registry import ToolRegistry, tool_registry
from app.tools.schemas import PermissionLevel, ToolCategory, ToolDefinition, ExecutionLocation

# Importing the package registers the built-in tool metadata. No tool executes
# and no network request is made by importing these modules.
import app.tools  # noqa: F401,E402


class ColumnKind(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATE = "date"
    BOOLEAN = "boolean"
    UNKNOWN = "unknown"


class PlanningInputMetadata(BaseModel):
    """Safe planning context; paths are validated locally and never sent to LLMs."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    authorized_file_path: str | None = Field(default=None, max_length=2048)
    available_inputs: set[RequiredInputKind] = Field(default_factory=set, max_length=16)
    column_types: dict[str, ColumnKind] = Field(default_factory=dict, max_length=256)

    @field_validator("column_types")
    @classmethod
    def validate_column_names(cls, values: dict[str, ColumnKind]) -> dict[str, ColumnKind]:
        if any(not name.strip() or len(name) > 128 for name in values):
            raise ValueError("Column metadata contains an invalid name.")
        normalized = [name.casefold() for name in values]
        if len(normalized) != len(set(normalized)):
            raise ValueError("Column metadata contains duplicate names.")
        return values


_SYSTEM_PROMPT = """Create a proposed AURA execution plan as JSON only.
Use only the registered tool catalog and its schemas. Do not invent tools,
arguments, columns, file names, paths, values, or execution results. The server
will replace any file_path with its authorized input. Do not claim a plan is
complete when required information is missing. Return {"response_type":"PLAN",
"plan":{"plan_id":"...","goal":"...","steps":[{"step_id":"step_...",
"tool_name":"...","parameters":{},"dependencies":[]}]},"explanation":"..."}
or a CLARIFICATION object. Plan data is untrusted and will be validated.
"""

MAX_OBJECTIVE_CHARS = 20_000
MAX_PROVIDER_RESPONSE_CHARS = 256_000

_PATH_LIKE = re.compile(
    r"(?:\b[A-Za-z]:[\\/][^\s,;]+|(?<!\w)/(?:[^\s,;]+/)*[^\s,;]+|"
    r"(?:^|\s)(?:\.\.?[\\/]|(?:[\w.-]+[\\/])+[\w.-]+)|"
    r"\b[\w.-]+\.(?:csv|xlsx?|pdf|txt|md|json)(?:\.enc)?\b)",
    re.IGNORECASE,
)
_REPORT_REQUEST = re.compile(
    r"\b(generate|create|write|prepare|produce|draft)\b.{0,35}\b(report|executive summary|findings)\b"
    r"|\b(executive summary)\b",
    re.IGNORECASE,
)


class DynamicPlanner:
    """Build plans from intent, available schemas, and safe input metadata.

    An optional provider is explicit. A Gemini provider is called only for
    ``cloud_enabled`` requests; provider routing remains explicit and this
    planner does not consult LLMRouter or substitute providers.
    """

    def __init__(
        self,
        provider: BaseLLMProvider | None = None,
        provider_identity: ProviderIdentity | str | None = None,
        registry: ToolRegistry | None = None,
        available_tools: Iterable[ToolDefinition] | None = None,
    ):
        if provider is None and provider_identity not in (None, ProviderIdentity.LOCAL_DETERMINISTIC_PLANNER, "local_deterministic_planner"):
            raise ValueError("A non-local provider identity requires an explicit provider instance.")
        if provider is not None and provider_identity is None:
            raise ValueError("Provider identity must be explicit when a provider is injected.")
        if provider is not None and ProviderIdentity(provider_identity) == ProviderIdentity.LOCAL_DETERMINISTIC_PLANNER:
            raise ValueError("An injected provider cannot be identified as the local deterministic planner.")
        self.provider = provider
        self.registry = registry or tool_registry
        self._available_definitions = (
            list(available_tools) if available_tools is not None else self.registry.list_tools()
        )
        self._available_names = {definition.name for definition in self._available_definitions}
        self.last_provider_identity = (
            ProviderIdentity(provider_identity)
            if provider_identity is not None
            else ProviderIdentity.LOCAL_DETERMINISTIC_PLANNER
        )
        self.last_failure_code: str | None = None

    async def create_plan(
        self,
        objective: str,
        intent: IntentAssessment,
        input_metadata: PlanningInputMetadata | dict[str, Any] | None = None,
        privacy_mode: Literal["local_only", "hybrid", "cloud_enabled"] = "local_only",
    ) -> PlanningResponse:
        """Return a strict PLAN or a safe CLARIFICATION; never execute tools."""
        self.last_failure_code = None
        try:
            assessment = IntentAssessment.model_validate(intent)
            metadata = PlanningInputMetadata.model_validate(input_metadata or {})
        except ValidationError:
            return self._clarification("Required planning information is invalid.", "The available inputs could not be safely validated.")
        if not isinstance(objective, str) or not objective.strip():
            return self._clarification("What would you like AURA to plan?", "A planning objective is required.")
        if len(objective) > MAX_OBJECTIVE_CHARS:
            return self._clarification("Please shorten the request and keep only the task objective.", "The planning objective exceeds the configured size limit.")
        if privacy_mode not in {"local_only", "hybrid", "cloud_enabled"}:
            return self._clarification("Which privacy mode should be used?", "The requested privacy mode is invalid.")
        if assessment.category == IntentCategory.UNKNOWN or assessment.confidence < 0.5:
            return self._clarification("Could you clarify the task and the result you want?", "The request does not map clearly to a supported registered capability.")

        unresolved = self._unresolved_intent_inputs(assessment, metadata, objective)
        if unresolved:
            return self._missing_input_clarification(unresolved)

        if self.provider is not None:
            identity = self.last_provider_identity
            if identity == ProviderIdentity.GEMINI:
                if privacy_mode != "cloud_enabled":
                    return self._clarification("Would you like to use the local planner, or explicitly enable Cloud Enabled mode?", "Gemini planning requires explicit Cloud Enabled mode.")
                if _PATH_LIKE.search(objective):
                    return self._clarification("Please submit the request without a file path; the server will attach the authorized upload separately.", "File paths are never included in provider planning requests.")
                return await self._provider_plan(objective, assessment, metadata, privacy_mode)
            if identity == ProviderIdentity.TEST_MOCK_PROVIDER:
                if settings.ENVIRONMENT != "test":
                    return self._clarification("The test provider is available only in test configuration.", "A mock provider cannot replace a configured production provider.")
                return await self._provider_plan(objective, assessment, metadata, privacy_mode)
            return self._clarification("The selected planning provider is not supported.", "No safe plan was produced.")

        self.last_provider_identity = ProviderIdentity.LOCAL_DETERMINISTIC_PLANNER
        return self._local_plan(objective, assessment, metadata, privacy_mode)

    @staticmethod
    def _clarification(question: str, explanation: str) -> PlanningResponse:
        return PlanningResponse(
            response_type=PlanningResponseType.CLARIFICATION,
            provider_identity=ProviderIdentity.LOCAL_DETERMINISTIC_PLANNER,
            clarification_question=question,
            explanation=explanation,
        )

    def _definition(self, name: str) -> ToolDefinition | None:
        if name not in self._available_names:
            return None
        definition = self.registry.get_tool_definition(name)
        if definition is None:
            return None
        declared = next((item for item in self._available_definitions if item.name == name), None)
        if declared is None or declared != definition:
            return None
        return definition

    def _require_definition(self, name: str, category: ToolCategory | None = None) -> ToolDefinition | None:
        definition = self._definition(name)
        if definition is None:
            return None
        if category is not None and definition.category != category:
            return None
        if definition.execution_location == ExecutionLocation.BROWSER:
            return None
        if definition.permission_level == PermissionLevel.PRIVILEGED or definition.requires_approval:
            return None
        return definition

    def _authorized_path(
        self,
        metadata: PlanningInputMetadata,
        kinds: set[RequiredInputKind],
    ) -> str | None:
        if metadata.authorized_file_path is None:
            return None
        try:
            resolved = validate_safe_path(metadata.authorized_file_path)
        except (OSError, PermissionError, ValueError):
            return None
        try:
            suffix = managed_file_suffix(resolved)
        except (OSError, PermissionError, ValueError):
            return None
        if RequiredInputKind.DATASET in kinds:
            if suffix not in {".csv", ".xlsx", ".xls"}:
                return None
        if RequiredInputKind.DOCUMENT in kinds:
            if suffix not in {".txt", ".md", ".json"}:
                return None
        return str(resolved)

    def _unresolved_intent_inputs(
        self,
        intent: IntentAssessment,
        metadata: PlanningInputMetadata,
        objective: str,
    ) -> set[RequiredInputKind]:
        supplied = set(metadata.available_inputs)
        if intent.category == IntentCategory.DOCUMENT_ANALYSIS:
            if self._authorized_path(metadata, {RequiredInputKind.DOCUMENT}) is not None:
                supplied.add(RequiredInputKind.DOCUMENT)
        elif intent.category in {
            IntentCategory.DATA_ANALYSIS,
            IntentCategory.STATISTICAL_ANALYSIS,
            IntentCategory.VISUALIZATION,
            IntentCategory.MULTI_STEP_ANALYSIS,
        }:
            if self._authorized_path(metadata, {RequiredInputKind.DATASET}) is not None:
                supplied.add(RequiredInputKind.DATASET)
        if RequiredInputKind.RESEARCH_TOPIC in {
            item.kind for item in intent.required_inputs if item.required
        }:
            if self._objective_has_research_topic(objective):
                supplied.add(RequiredInputKind.RESEARCH_TOPIC)
            else:
                supplied.discard(RequiredInputKind.RESEARCH_TOPIC)
        missing = {item.kind for item in intent.missing_inputs if item.required} - supplied
        if (
            any(item.kind == RequiredInputKind.RESEARCH_TOPIC and item.required for item in intent.missing_inputs)
            and not self._objective_has_research_topic(objective)
        ):
            missing.add(RequiredInputKind.RESEARCH_TOPIC)
        required = {item.kind for item in intent.required_inputs if item.required}
        missing |= required - supplied - {RequiredInputKind.ANALYSIS_GOAL}
        return missing

    @staticmethod
    def _objective_has_research_topic(objective: str) -> bool:
        generic = {
            "research", "web", "online", "internet", "about", "on", "find", "look", "up",
            "the", "this", "that", "a", "an", "it", "something", "please", "conduct",
            "investigate", "current", "latest", "recent", "for", "sources", "and", "prepare",
        }
        topic_words = [word for word in re.findall(r"[a-z0-9]+", objective.casefold()) if word not in generic]
        return len(topic_words) >= 2

    def _local_plan(
        self,
        objective: str,
        intent: IntentAssessment,
        metadata: PlanningInputMetadata,
        privacy_mode: str,
    ) -> PlanningResponse:
        text = objective.casefold()
        if intent.category == IntentCategory.REPORT_GENERATION or _REPORT_REQUEST.search(text):
            return self._clarification(
                "Please provide completed findings or wait until the requested analysis has produced results.",
                "The registered report tool requires findings that are not available during planning.",
            )

        if intent.category == IntentCategory.MULTI_STEP_ANALYSIS or self._objective_requires_multiple_tools(text):
            selected = self._multi_step_tools(text, metadata, privacy_mode)
            if isinstance(selected, PlanningResponse):
                return selected
            names, params, dependencies = selected
            return self._validated_response(objective, intent, names, params, dependencies, privacy_mode)

        if intent.category == IntentCategory.SYSTEM_HEALTH:
            definition = self._require_definition("system_health", ToolCategory.SYSTEM)
            if definition is None:
                return self._unsupported_capability("System diagnostics")
            return self._validated_response(objective, intent, ["system_health"], [{"check_storage": True}], [[]], privacy_mode)

        if intent.category == IntentCategory.DATA_ANALYSIS:
            path = self._authorized_path(metadata, {RequiredInputKind.DATASET})
            if path is None:
                return self._missing_input_clarification({RequiredInputKind.DATASET})
            return self._validated_response(
                objective, intent, ["inspect_dataset"], [{"file_path": path, "sample_rows": 5}], [[]], privacy_mode
            )

        if intent.category == IntentCategory.DOCUMENT_ANALYSIS:
            path = self._authorized_path(metadata, {RequiredInputKind.DOCUMENT})
            if path is None:
                return self._missing_input_clarification({RequiredInputKind.DOCUMENT})
            if re.search(r"\b(search|find|question|summari[sz]e|analy[sz]e|review)\b", text):
                return self._clarification(
                    "Would text extraction be sufficient, or do you need document question answering or summarization?",
                    "The registered document tools extract text, but cannot perform semantic analysis without execution-time document content.",
                )
            return self._validated_response(
                objective, intent, ["read_document"], [{"file_path": path}], [[]], privacy_mode
            )

        if intent.category == IntentCategory.WEB_RESEARCH:
            if privacy_mode != "cloud_enabled":
                return self._clarification(
                    "Would you like to enable Cloud Enabled mode for external research?",
                    "The registered web research tool requires explicit Cloud Enabled mode.",
                )
            return self._validated_response(
                objective, intent, ["web_research"], [{"query": objective}], [[]], privacy_mode
            )

        if intent.category == IntentCategory.STATISTICAL_ANALYSIS:
            path = self._authorized_path(metadata, {RequiredInputKind.DATASET})
            if path is None:
                return self._missing_input_clarification({RequiredInputKind.DATASET})
            choice = self._statistical_tool(text, path, metadata)
            if isinstance(choice, PlanningResponse):
                return choice
            name, params = choice
            return self._validated_response(objective, intent, [name], [params], [[]], privacy_mode)

        if intent.category == IntentCategory.VISUALIZATION:
            path = self._authorized_path(metadata, {RequiredInputKind.DATASET})
            if path is None:
                return self._missing_input_clarification({RequiredInputKind.DATASET})
            choice = self._visualization_params(text, path, metadata)
            if isinstance(choice, PlanningResponse):
                return choice
            return self._validated_response(objective, intent, ["generate_chart"], [choice], [[]], privacy_mode)

        return self._unsupported_capability(intent.category.value)

    def _statistical_tool(
        self,
        text: str,
        path: str,
        metadata: PlanningInputMetadata,
    ) -> tuple[str, dict[str, Any]] | PlanningResponse:
        if self._has_tool_signal(text, ("correlation_analysis", "correlation", "correlate", "correlations")):
            if self._require_definition("correlation_analysis", ToolCategory.ANALYTICS) is None:
                return self._unsupported_capability("Correlation analysis")
            return "correlation_analysis", {"file_path": path}
        if self._has_tool_signal(text, ("detect_outliers", "outlier", "outliers", "anomaly", "anomalies")):
            if self._require_definition("detect_outliers", ToolCategory.ANALYTICS) is None:
                return self._unsupported_capability("Outlier detection")
            column = self._mentioned_column(text, metadata, {ColumnKind.NUMERIC})
            if column is None:
                return self._clarification(
                    "Which numeric column should be checked for outliers?",
                    "Outlier detection requires a specific numeric column from the supplied schema metadata.",
                )
            return "detect_outliers", {"file_path": path, "column": column}
        if self._has_tool_signal(text, ("group_comparison", "compare", "comparison", "across", "groups")):
            if self._require_definition("group_comparison", ToolCategory.ANALYTICS) is None:
                return self._unsupported_capability("Group comparison")
            columns = self._comparison_columns(text, metadata)
            if columns is None:
                return self._clarification(
                    "Which categorical column should define the groups, and which numeric column should be compared?",
                    "Group comparison needs exact column names and types from authorized dataset schema metadata.",
                )
            group_by, metric = columns
            aggregation = "mean" if re.search(r"\b(average|avg|mean)\b", text) else "sum"
            return "group_comparison", {
                "file_path": path,
                "group_by": group_by,
                "metric_column": metric,
                "aggregations": [aggregation],
                "sort_by": aggregation,
            }
        if self._require_definition("descriptive_statistics", ToolCategory.ANALYTICS) is None:
            return self._unsupported_capability("Descriptive statistics")
        if re.search(r"\b(descriptive|summary)\s+statistics\b|\b(mean|median|quartiles?|distribution|statistical summary)\b", text):
            return "descriptive_statistics", {"file_path": path}
        return self._clarification(
            "Which statistical question should the plan answer?",
            "The request does not identify a supported statistical operation.",
        )

    def _visualization_params(
        self,
        text: str,
        path: str,
        metadata: PlanningInputMetadata,
    ) -> dict[str, Any] | PlanningResponse:
        if self._require_definition("generate_chart", ToolCategory.VISUALIZATION) is None:
            return self._unsupported_capability("Visualization")
        title = "Requested data visualization"
        chart_type = next(
            (name for name in ("heatmap", "histogram", "scatter", "line", "box", "bar") if re.search(rf"\b{name}\b", text)),
            None,
        )
        if chart_type is None:
            return self._clarification("What type of chart would you like?", "The request does not specify a chart type that can be safely inferred.")
        if chart_type == "heatmap":
            numeric = [name for name, kind in metadata.column_types.items() if kind == ColumnKind.NUMERIC]
            if len(numeric) < 2:
                return self._clarification("Which numeric columns should be included in the heatmap?", "A heatmap requires at least two numeric columns in the supplied schema metadata.")
            x_column = numeric[0]
            y_column = numeric[1]
        elif chart_type == "histogram":
            x_column = self._mentioned_column(text, metadata, {ColumnKind.NUMERIC})
            if x_column is None:
                numeric = [name for name, kind in metadata.column_types.items() if kind == ColumnKind.NUMERIC]
                x_column = numeric[0] if len(numeric) == 1 else None
            y_column = None
            if x_column is None:
                return self._clarification("Which numeric column should the histogram show?", "The histogram column cannot be determined from the request and supplied schema metadata.")
        else:
            columns = self._comparison_columns(text, metadata)
            if columns is None:
                return self._clarification("Which available categorical and numeric columns should the chart use?", "The chart arguments cannot be safely inferred from the request and supplied schema metadata.")
            x_column, y_column = columns
        params: dict[str, Any] = {
            "file_path": path,
            "chart_type": chart_type,
            "title": title,
            "x_column": x_column,
        }
        if y_column is not None:
            params["y_column"] = y_column
        return params

    def _multi_step_tools(
        self,
        text: str,
        metadata: PlanningInputMetadata,
        privacy_mode: str,
    ) -> tuple[list[str], list[dict[str, Any]], list[list[str]]] | PlanningResponse:
        # Multi-step composition is intentionally limited to combinations whose
        # current tool schemas can be satisfied without inventing future output.
        if "research" in text and _REPORT_REQUEST.search(text):
            return self._clarification(
                "Would you like to run the research step first and provide its completed findings before generating a report?",
                "The report tool requires concrete findings; downstream output references are not defined by the current executor schema.",
            )
        if any(token in text for token in ("correlation", "correlate", "correlations")) and any(
            token in text for token in ("visualize", "visualise", "chart", "heatmap")
        ):
            path = self._authorized_path(metadata, {RequiredInputKind.DATASET})
            if path is None:
                return self._missing_input_clarification({RequiredInputKind.DATASET})
            numeric = [name for name, kind in metadata.column_types.items() if kind == ColumnKind.NUMERIC]
            if len(numeric) < 2:
                return self._clarification(
                    "Which numeric columns should be shown in the heatmap?",
                    "The correlation chart requires at least two numeric columns in the supplied schema metadata.",
                )
            return (
                ["correlation_analysis", "generate_chart"],
                [
                    {"file_path": path},
                    {"file_path": path, "chart_type": "heatmap", "title": "Correlation heatmap", "x_column": numeric[0], "y_column": numeric[1]},
                ],
                [[], ["step_correlation_analysis"]],
            )
        if any(token in text for token in ("statistics", "statistical", "mean", "median")) and any(
            token in text for token in ("inspect", "profile", "schema", "columns")
        ):
            path = self._authorized_path(metadata, {RequiredInputKind.DATASET})
            if path is None:
                return self._missing_input_clarification({RequiredInputKind.DATASET})
            if self._require_definition("inspect_dataset", ToolCategory.DATASET) is None or self._require_definition("descriptive_statistics", ToolCategory.ANALYTICS) is None:
                return self._unsupported_capability("Dataset inspection and descriptive statistics")
            return (
                ["inspect_dataset", "descriptive_statistics"],
                [{"file_path": path, "sample_rows": 5}, {"file_path": path}],
                [[], ["step_inspect_dataset"]],
            )
        if _REPORT_REQUEST.search(text):
            return self._clarification(
                "Please provide completed findings before planning report generation.",
                "The report tool requires results that are not available during planning.",
            )
        return self._clarification(
            "Could you narrow the requested workflow to supported steps and provide any missing input details?",
            "The requested multi-step combination cannot be represented safely by current registered tool schemas.",
        )

    @staticmethod
    def _objective_requires_multiple_tools(text: str) -> bool:
        correlation = any(word in text for word in ("correlation", "correlate"))
        visualization = any(word in text for word in ("visualize", "visualise", "chart", "heatmap", "plot"))
        inspection = any(word in text for word in ("inspect", "profile", "schema", "columns"))
        statistics = any(word in text for word in ("statistics", "statistical", "mean", "median"))
        research = any(word in text for word in ("research", "investigate"))
        report = bool(_REPORT_REQUEST.search(text))
        return (correlation and visualization) or (inspection and statistics) or (research and report)

    @staticmethod
    def _has_tool_signal(text: str, words: tuple[str, ...]) -> bool:
        return any(re.search(rf"\b{re.escape(word)}\b", text) for word in words)

    @staticmethod
    def _mentioned_column(
        text: str,
        metadata: PlanningInputMetadata,
        allowed: set[ColumnKind],
    ) -> str | None:
        matches: list[str] = []
        words = set(re.findall(r"[a-z0-9]+", text))
        for name, kind in metadata.column_types.items():
            if kind not in allowed:
                continue
            tokens = re.findall(r"[a-z0-9]+", name.casefold())
            normalized = " ".join(tokens)
            exact = bool(normalized and re.search(rf"\b{re.escape(normalized)}s?\b", text))
            token_match = len(tokens) == 1 and (tokens[0] in words or f"{tokens[0]}s" in words)
            if exact or token_match:
                matches.append(name)
        return matches[0] if len(matches) == 1 else None

    @classmethod
    def _comparison_columns(
        cls,
        text: str,
        metadata: PlanningInputMetadata,
    ) -> tuple[str, str] | None:
        group_by = cls._mentioned_column(text, metadata, {ColumnKind.CATEGORICAL})
        metric = cls._mentioned_column(text, metadata, {ColumnKind.NUMERIC})
        if group_by is not None and metric is not None:
            return group_by, metric
        # Column role cues can disambiguate when the objective names only one
        # field: a noun following "across/by/among" is the grouping field.
        role_match = re.search(r"\b(?:by|across|among)\s+([a-z0-9 _-]{2,60})", text)
        if role_match:
            role_words = set(re.findall(r"[a-z0-9]+", role_match.group(1)))
            groups = [
                name for name, kind in metadata.column_types.items()
                if kind == ColumnKind.CATEGORICAL
                and role_words.intersection(re.findall(r"[a-z0-9]+", name.casefold()))
            ]
            if len(groups) == 1:
                group_by = groups[0]
        if group_by is not None and metric is None:
            numeric = [name for name, kind in metadata.column_types.items() if kind == ColumnKind.NUMERIC]
            if len(numeric) == 1 and re.search(r"\b(sales|revenue|profit|amount|value|average|mean|sum)\b", text):
                metric = numeric[0]
        return (group_by, metric) if group_by and metric else None

    def _validated_response(
        self,
        objective: str,
        intent: IntentAssessment,
        tool_names: list[str],
        parameters: list[dict[str, Any]],
        dependencies: list[list[str]],
        privacy_mode: str,
        step_ids: list[str] | None = None,
        provider_identity: ProviderIdentity = ProviderIdentity.LOCAL_DETERMINISTIC_PLANNER,
    ) -> PlanningResponse:
        if not (len(tool_names) == len(parameters) == len(dependencies)):
            self.last_failure_code = "invalid_plan"
            return self._clarification("Could you clarify the requested steps?", "The candidate plan is structurally incomplete.")
        if step_ids is not None and (len(step_ids) != len(tool_names) or len(set(step_ids)) != len(step_ids)):
            self.last_failure_code = "invalid_plan"
            return self._clarification("Could you retry with unique step identifiers?", "The candidate plan contains invalid or duplicate step IDs.")
        steps: list[ExecutionStep] = []
        for index, (name, params, deps) in enumerate(zip(tool_names, parameters, dependencies), start=1):
            definition = self._require_definition(name)
            if definition is None:
                self.last_failure_code = "invalid_plan"
                return self._unsupported_capability(name)
            try:
                # Reject undeclared keys even if a Pydantic tool input model
                # currently ignores extras.
                properties = definition.input_schema.get("properties", {})
                if set(params) - set(properties):
                    raise ValueError("Undeclared arguments.")
                self.registry.validate_arguments(name, params)
                steps.append(ExecutionStep(
                    step_id=step_ids[index - 1] if step_ids is not None else (f"step_{name}" if index == 1 else f"step_{name}_{index}"),
                    title=name.replace("_", " ").title(),
                    description=definition.description,
                    capability=definition.category.value,
                    tool_name=name,
                    parameters=params,
                    dependencies=deps,
                ))
            except (ValidationError, ValueError, KeyError):
                self.last_failure_code = "invalid_plan"
                return self._clarification("Could you provide the missing tool input details?", "The requested plan does not satisfy the registered tool schema.")

        if not self._intent_matches_plan(intent, steps, objective):
            self.last_failure_code = "invalid_plan"
            return self._clarification("Could you clarify which supported operation should be planned?", "The candidate tools do not match the assessed intent.")
        plan = ExecutionPlan(plan_id=f"plan_{uuid.uuid4().hex[:12]}", goal=objective, steps=steps)
        try:
            validate_execution_plan(plan, privacy_mode=privacy_mode, max_steps=settings.MAX_PLAN_STEPS)
        except (PlanValidationError, ValidationError, ValueError):
            self.last_failure_code = "invalid_plan"
            return self._clarification("Could you clarify the required inputs or privacy mode?", "The candidate plan did not pass structural or policy validation.")
        return PlanningResponse(
            response_type=PlanningResponseType.PLAN,
            provider_identity=provider_identity,
            plan=plan,
            explanation="The plan uses registered tools and passed schema, dependency, and privacy validation.",
        )

    @staticmethod
    def _intent_matches_plan(intent: IntentAssessment, steps: list[ExecutionStep], objective: str) -> bool:
        categories = {ToolCategory(step.capability) for step in steps}
        text = objective.casefold()
        expected = {
            IntentCategory.DATA_ANALYSIS: {ToolCategory.DATASET},
            IntentCategory.STATISTICAL_ANALYSIS: {ToolCategory.ANALYTICS},
            IntentCategory.VISUALIZATION: {ToolCategory.VISUALIZATION},
            IntentCategory.DOCUMENT_ANALYSIS: {ToolCategory.DOCUMENT},
            IntentCategory.WEB_RESEARCH: {ToolCategory.RESEARCH},
            IntentCategory.REPORT_GENERATION: {ToolCategory.REPORT},
            IntentCategory.SYSTEM_HEALTH: {ToolCategory.SYSTEM},
            IntentCategory.MULTI_STEP_ANALYSIS: set(ToolCategory),
        }.get(intent.category, set())
        if intent.category == IntentCategory.MULTI_STEP_ANALYSIS and (len(steps) < 2 or len(categories) < 2):
            return False
        tools = {step.tool_name for step in steps}
        if intent.category == IntentCategory.MULTI_STEP_ANALYSIS:
            if "research" in text and "report" in text:
                return False
            if any(word in text for word in ("correlation", "correlate")) and any(
                word in text for word in ("visualize", "visualise", "chart", "heatmap", "plot")
            ):
                return {"correlation_analysis", "generate_chart"}.issubset(tools)
            if any(word in text for word in ("inspect", "profile", "schema", "columns")) and any(
                word in text for word in ("statistics", "statistical", "mean", "median")
            ):
                return {"inspect_dataset", "descriptive_statistics"}.issubset(tools)
        if intent.category == IntentCategory.STATISTICAL_ANALYSIS:
            if any(word in text for word in ("correlation", "correlate")):
                return "correlation_analysis" in tools
            if any(word in text for word in ("outlier", "outliers", "anomaly", "anomalies")):
                return "detect_outliers" in tools
            if any(word in text for word in ("compare", "comparison", "across", "groups")):
                return "group_comparison" in tools
            return "descriptive_statistics" in tools
        if intent.category == IntentCategory.DOCUMENT_ANALYSIS:
            if re.search(r"\b(search|find|question|summari[sz]e|analy[sz]e|review)\b", text):
                return False
        return bool(categories and categories.intersection(expected))

    async def _provider_plan(
        self,
        objective: str,
        intent: IntentAssessment,
        metadata: PlanningInputMetadata,
        privacy_mode: str,
    ) -> PlanningResponse:
        expected_provider = {
            ProviderIdentity.GEMINI: "gemini",
            ProviderIdentity.TEST_MOCK_PROVIDER: "test_reasoner",
        }.get(self.last_provider_identity)
        if expected_provider is None or self.provider is None:
            return self._clarification("Please choose an available planning provider.", "No provider was available for the requested planning mode.")
        try:
            if not self.provider.is_available():
                self.last_failure_code = "provider_unavailable"
                return self._clarification("Would you like to use local deterministic planning instead?", "The explicitly selected provider is unavailable; no fallback was used.")
            catalog = []
            for name in sorted(self._available_names):
                definition = self._require_definition(name)
                if definition is None:
                    continue
                catalog.append({
                    "name": definition.name,
                    "category": definition.category.value,
                    "description": definition.description,
                    "input_schema": definition.input_schema,
                })
            request_data = {
                "objective": objective,
                "intent": intent.model_dump(mode="json"),
                "available_input_kinds": sorted(
                    kind.value for kind in (
                        metadata.available_inputs
                        | ({RequiredInputKind.DATASET} if metadata.authorized_file_path and intent.category != IntentCategory.DOCUMENT_ANALYSIS else set())
                        | ({RequiredInputKind.DOCUMENT} if metadata.authorized_file_path and intent.category == IntentCategory.DOCUMENT_ANALYSIS else set())
                    )
                ),
                "registered_tools": catalog,
            }
            with privacy_scope(privacy_mode):
                response = await self.provider.chat(
                    [
                        LLMMessage(role="system", content=_SYSTEM_PROMPT),
                        LLMMessage(role="user", content=json.dumps(request_data, ensure_ascii=True)),
                    ],
                    response_format_json=True,
                    temperature=0,
                    max_tokens=2048,
                )
            if response.provider != expected_provider:
                self.last_failure_code = "invalid_provider_response"
                return self._clarification("Would you like to retry with a correctly configured provider?", "The provider identity did not match the explicit selection.")
            if response.tool_calls or not response.content or len(response.content) > MAX_PROVIDER_RESPONSE_CHARS:
                self.last_failure_code = "invalid_provider_response"
                return self._clarification("Could you narrow the request to a plan the selected provider can express?", "The provider did not return a structured planning response.")
            return self._parse_provider_plan(
                response.content, objective, intent, metadata, privacy_mode,
                provider_identity=self.last_provider_identity,
            )
        except Exception as exc:
            # Provider error bodies and response payloads are deliberately not
            # logged or exposed through the planning contract.
            code = getattr(exc, "code", None)
            if isinstance(exc, TimeoutError) or code == "timeout":
                self.last_failure_code = "provider_timeout"
            elif code == "missing_api_key":
                self.last_failure_code = "provider_unavailable"
            else:
                self.last_failure_code = "provider_failure"
            return self._clarification("Would you like to retry later or use deterministic local planning?", "The selected provider failed; no provider fallback was attempted.")

    def _parse_provider_plan(
        self,
        content: str,
        objective: str,
        intent: IntentAssessment,
        metadata: PlanningInputMetadata,
        privacy_mode: str,
        provider_identity: ProviderIdentity = ProviderIdentity.GEMINI,
    ) -> PlanningResponse:
        try:
            payload = json.loads(content)
            if not isinstance(payload, dict) or set(payload) - {"response_type", "plan", "clarification_question", "explanation"}:
                raise ValueError("Invalid planning response shape.")
            if payload.get("response_type") != "PLAN":
                return self._clarification("Could you provide the requested clarification?", "The provider did not return a validated executable plan.")
            raw_plan = payload.get("plan")
            if not isinstance(raw_plan, dict) or set(raw_plan) - {"plan_id", "goal", "created_at", "steps"}:
                raise ValueError("Invalid plan shape.")
            raw_steps = raw_plan.get("steps")
            if not isinstance(raw_steps, list) or not raw_steps or len(raw_steps) > settings.MAX_PLAN_STEPS:
                raise ValueError("Invalid step list.")
            allowed_step_fields = {"step_id", "tool_name", "parameters", "dependencies"}
            names: list[str] = []
            params_list: list[dict[str, Any]] = []
            dependencies: list[list[str]] = []
            step_ids: list[str] = []
            for raw_step in raw_steps:
                if not isinstance(raw_step, dict) or set(raw_step) - allowed_step_fields:
                    raise ValueError("Invalid step fields.")
                name = raw_step.get("tool_name")
                step_id = raw_step.get("step_id")
                if not isinstance(step_id, str) or not re.fullmatch(r"^step_[A-Za-z0-9][A-Za-z0-9_-]{0,63}$", step_id):
                    raise ValueError("Invalid step identifier.")
                definition = self._require_definition(name) if isinstance(name, str) else None
                if definition is None or name in {"generate_report", "search_document_chunks"}:
                    raise ValueError("Unavailable tool.")
                params = raw_step.get("parameters", {})
                deps = raw_step.get("dependencies", [])
                if not isinstance(params, dict) or not isinstance(deps, list) or any(not isinstance(dep, str) for dep in deps):
                    raise ValueError("Malformed tool arguments or dependencies.")
                required_fields = set(definition.input_schema.get("required", []))
                if "file_path" in required_fields:
                    kind = RequiredInputKind.DOCUMENT if definition.category == ToolCategory.DOCUMENT else RequiredInputKind.DATASET
                    safe_path = self._authorized_path(metadata, {kind})
                    if safe_path is None:
                        return self._missing_input_clarification({
                            kind
                        })
                    params["file_path"] = safe_path
                if "file_path" in params and "file_path" not in required_fields:
                    raise ValueError("Unexpected file path argument.")
                for field_name in ("group_by", "metric_column", "column", "x_column", "y_column", "group_column"):
                    column = params.get(field_name)
                    if column is not None and column not in metadata.column_types:
                        raise ValueError("Unrecognized column.")
                requested_columns = params.get("columns")
                if requested_columns is not None and (
                    not isinstance(requested_columns, list)
                    or any(column not in metadata.column_types for column in requested_columns)
                ):
                    raise ValueError("Unrecognized column list.")
                step_ids.append(step_id)
                names.append(name)
                params_list.append(params)
                dependencies.append(deps)
            return self._validated_response(
                objective, intent, names, params_list, dependencies, privacy_mode,
                step_ids, provider_identity,
            )
        except (json.JSONDecodeError, TypeError, ValueError, ValidationError):
            self.last_failure_code = "invalid_provider_response"
            return self._clarification("Could you clarify the supported tools or required inputs?", "The provider response was malformed, unsafe, or failed plan validation.")

    def _missing_input_clarification(self, missing: set[RequiredInputKind]) -> PlanningResponse:
        questions = {
            RequiredInputKind.DATASET: "Please select an authorized dataset upload.",
            RequiredInputKind.DOCUMENT: "Please select an authorized document upload.",
            RequiredInputKind.DATABASE: "Please provide an authorized database connection.",
            RequiredInputKind.RESEARCH_TOPIC: "What specific topic should be researched?",
            RequiredInputKind.ANALYSIS_GOAL: "What result should the analysis or report provide?",
            RequiredInputKind.CLARIFICATION: "Could you clarify the task and the result you want?",
        }
        first = sorted(missing, key=lambda item: item.value)[0]
        return self._clarification(questions[first], "A required input is missing from the safe planning context.")

    def _unsupported_capability(self, capability: str) -> PlanningResponse:
        # Do not echo arbitrary provider or objective text into this response.
        allowed = {category.value for category in ToolCategory}
        safe_label = capability if capability in allowed else "requested operation"
        return self._clarification(
            "Would you like to choose an operation supported by the registered tools?",
            f"No available registered tool safely supports {safe_label}.",
        )
