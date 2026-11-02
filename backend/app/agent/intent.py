"""Deterministic, local-only intent assessment backed by registered tools.

This module does not read user files, log objectives, invoke providers, or
change task execution. It returns only the typed, value-free assessment
contract used by the later planning layer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from app.agent.intelligence_contracts import (
    InputRequirement,
    IntentAssessment,
    IntentCategory,
    RequiredInputKind,
)
from app.tools.registry import ToolRegistry, tool_registry
from app.tools.schemas import ToolCategory, ToolDefinition

# Importing the tools package performs the existing, local registration of
# built-in tool metadata. It does not execute a tool or make network requests.
import app.tools  # noqa: F401,E402


@dataclass(frozen=True)
class _IntentRule:
    category: IntentCategory
    tool_names: tuple[str, ...]
    patterns: tuple[str, ...]
    requires: tuple[RequiredInputKind, ...] = ()


_RULES: tuple[_IntentRule, ...] = (
    _IntentRule(
        IntentCategory.SYSTEM_HEALTH,
        ("system_health",),
        (r"\b(system|service|application)\s+(health|status|diagnostics?)\b",
         r"\b(check|diagnose)\s+(system|service|storage|memory|cpu)\b",
         r"\b(storage|cpu|memory)\s+(health|status|usage)\b"),
    ),
    _IntentRule(
        IntentCategory.DOCUMENT_ANALYSIS,
        ("read_document", "search_document_chunks"),
        (r"\b(analy[sz]e|summari[sz]e|review|extract|search|read)\b.{0,50}\b(document|pdf|contract|text file)\b",
         r"\b(document|pdf|contract)\s+(analysis|summary|review|search|contents?)\b",
         r"\b(find|search)\b.{0,40}\b(in|within)\b.{0,20}\b(document|pdf|contract)\b"),
        (RequiredInputKind.DOCUMENT,),
    ),
    _IntentRule(
        IntentCategory.WEB_RESEARCH,
        ("web_research",),
        (r"\b(web|online|internet)\s+(research|search|sources?)\b",
         r"\b(research|investigate|look up|find sources for)\b",
         r"\b(current|latest|recent)\b.{0,35}\b(competitor|industry|market|news|trend|regulation)\b"),
        (RequiredInputKind.RESEARCH_TOPIC,),
    ),
    _IntentRule(
        IntentCategory.REPORT_GENERATION,
        ("generate_report",),
        (r"\b(generate|create|write|prepare|produce|draft)\b.{0,35}\b(report|executive summary|findings)\b",
         r"\b(report|executive summary)\s+(on|about|for|from)\b"),
        (RequiredInputKind.ANALYSIS_GOAL,),
    ),
    _IntentRule(
        IntentCategory.VISUALIZATION,
        ("generate_chart",),
        (r"\b(create|make|generate|draw|show|plot)\b.{0,30}\b(chart|graph|plot|visuali[sz]ation)\b",
         r"\b(visuali[sz]e|chart|graph|plot)\b.{0,30}\b(data|dataset|trend|relationship|distribution)\b",
         r"\b(histogram|heatmap|scatterplot|bar chart|line chart)\b"),
        (RequiredInputKind.DATASET,),
    ),
    _IntentRule(
        IntentCategory.STATISTICAL_ANALYSIS,
        ("group_comparison",),
        (r"\b(compare|comparison|difference)\b.{0,45}\b(groups?|segments?|categories|by)\b",
         r"\b(group|segment)\s+(comparison|analysis)\b"),
        (RequiredInputKind.DATASET,),
    ),
    _IntentRule(
        IntentCategory.STATISTICAL_ANALYSIS,
        ("correlation_analysis",),
        (r"\b(correlation|correlate|correlated|relationship between|association between)\b",
         r"\b(which|how)\b.{0,35}\b(variables|metrics|measures)\b.{0,20}\b(move together|relate)\b"),
        (RequiredInputKind.DATASET,),
    ),
    _IntentRule(
        IntentCategory.STATISTICAL_ANALYSIS,
        ("detect_outliers",),
        (r"\b(outliers?|anomal(?:y|ies)|extreme values?)\b",
         r"\b(detect|find|identify)\b.{0,25}\b(unusual|abnormal)\b.{0,15}\b(records?|values?|points?)\b"),
        (RequiredInputKind.DATASET,),
    ),
    _IntentRule(
        IntentCategory.STATISTICAL_ANALYSIS,
        ("descriptive_statistics",),
        (r"\b(descriptive|summary)\s+statistics\b",
         r"\b(mean|median|quartiles?|standard deviation|distribution)\b.{0,30}\b(data|dataset|columns?|fields?)\b",
         r"\b(statistical summary|summari[sz]e the numeric|numeric statistics)\b"),
        (RequiredInputKind.DATASET,),
    ),
    _IntentRule(
        IntentCategory.DATA_ANALYSIS,
        ("inspect_dataset",),
        (r"\b(inspect|profile|explore)\b.{0,30}\b(dataset|csv|spreadsheet|table|data)\b",
         r"\b(show|list)\b.{0,20}\b(columns|schema|data types|missing values|duplicates)\b",
         r"\b(data quality|dataset overview|dataset structure|tabular data)\b",
         r"\b(analy[sz]e|explore|assess)\b.{0,35}\b(dataset|csv|spreadsheet|sales data|table|data|metrics|revenue|profit|customers|churn)\b"),
        (RequiredInputKind.DATASET,),
    ),
)

_VAGUE_REQUEST = re.compile(
    r"^(?:analy[sz]e|compare|visuali[sz]e|help|look at|make sense of|research)"
    r"(?:\s+(?:this|it|that|the data|something))?\s*[.!?]*$",
    re.IGNORECASE,
)
_WORKFLOW_CUE = re.compile(r"\b(then|after that|also|and then|end[- ]to[- ]end|multi[- ]step|workflow)\b", re.I)


def _registered_tools(
    source: ToolRegistry | Iterable[ToolDefinition] | None,
) -> dict[str, ToolDefinition]:
    if source is None:
        definitions = tool_registry.list_tools()
    elif isinstance(source, ToolRegistry):
        definitions = source.list_tools()
    else:
        definitions = list(source)
    return {definition.name: definition for definition in definitions}


def _tool_supports_category(definition: ToolDefinition, category: IntentCategory) -> bool:
    expected = {
        IntentCategory.DATA_ANALYSIS: {ToolCategory.DATASET},
        IntentCategory.STATISTICAL_ANALYSIS: {ToolCategory.ANALYTICS},
        IntentCategory.VISUALIZATION: {ToolCategory.VISUALIZATION},
        IntentCategory.DOCUMENT_ANALYSIS: {ToolCategory.DOCUMENT},
        IntentCategory.WEB_RESEARCH: {ToolCategory.RESEARCH},
        IntentCategory.REPORT_GENERATION: {ToolCategory.REPORT},
        IntentCategory.SYSTEM_HEALTH: {ToolCategory.SYSTEM},
    }.get(category, set())
    description_markers = {
        ToolCategory.DATASET: ("tabular", "dataset", "schema", "data quality"),
        ToolCategory.ANALYTICS: ("statistics", "correlation", "groups data", "outlier", "aggregated metrics"),
        ToolCategory.VISUALIZATION: ("visualization", "chart", "plot"),
        ToolCategory.DOCUMENT: ("document", "text", "chunks"),
        ToolCategory.RESEARCH: ("search", "external information", "sources"),
        ToolCategory.REPORT: ("report", "analytical results", "findings"),
        ToolCategory.SYSTEM: ("system", "operational status", "storage metrics"),
    }
    description = definition.description.casefold()
    # Registry metadata determines both that the capability is installed and
    # that its description still matches the capability this rule represents.
    return (
        definition.category in expected
        and bool(description.strip())
        and any(marker in description for marker in description_markers.get(definition.category, ()))
    )


def _schema_requirement_kinds(
    tools: dict[str, ToolDefinition],
    selected_tool_names: set[str],
) -> set[RequiredInputKind]:
    required: set[RequiredInputKind] = set()
    for name in selected_tool_names:
        definition = tools.get(name)
        if definition is None:
            continue
        required_fields = set(definition.input_schema.get("required", []))
        for field_name in required_fields:
            if field_name == "file_path":
                if definition.category == ToolCategory.DOCUMENT:
                    required.add(RequiredInputKind.DOCUMENT)
                elif definition.category in {
                    ToolCategory.DATASET,
                    ToolCategory.ANALYTICS,
                    ToolCategory.VISUALIZATION,
                }:
                    required.add(RequiredInputKind.DATASET)
            elif field_name == "query" and definition.category == ToolCategory.RESEARCH:
                required.add(RequiredInputKind.RESEARCH_TOPIC)
    return required


def assess_intent(
    objective: str,
    available_tools: ToolRegistry | Iterable[ToolDefinition] | None = None,
    available_inputs: Iterable[RequiredInputKind | str] = (),
) -> IntentAssessment:
    """Assess request intent using local patterns and actual registered tools.

    ``available_inputs`` contains only safe presence metadata (for example,
    ``RequiredInputKind.DATASET``), never paths, filenames, rows, or contents.
    Objective text is consumed transiently and is never copied into the result.
    """
    if not isinstance(objective, str):
        objective = ""
    normalized = objective.casefold()[:20_000]
    tools = _registered_tools(available_tools)
    supplied: set[RequiredInputKind] = set()
    for value in available_inputs:
        try:
            supplied.add(RequiredInputKind(value))
        except (TypeError, ValueError):
            continue

    matches: list[tuple[_IntentRule, set[str]]] = []
    for rule in _RULES:
        if not any(re.search(pattern, normalized) for pattern in rule.patterns):
            continue
        matching_names = {
            name for name in rule.tool_names
            if name in tools and _tool_supports_category(tools[name], rule.category)
        }
        if matching_names:
            matches.append((rule, matching_names))

    if not matches:
        vague = bool(_VAGUE_REQUEST.search(normalized.strip()))
        missing = [InputRequirement(kind=RequiredInputKind.CLARIFICATION)] if vague else []
        return IntentAssessment(
            category=IntentCategory.UNKNOWN,
            confidence=0.25 if vague else 0.05,
            required_inputs=missing.copy(),
            missing_inputs=missing,
            is_multi_step=False,
        )

    matched_categories = {rule.category for rule, _ in matches}
    matched_tools = set().union(*(names for _, names in matches))
    explicit_workflow = bool(_WORKFLOW_CUE.search(normalized))
    multi_step = len(matched_tools) > 1 and (explicit_workflow or len(matched_categories) > 1)
    if multi_step:
        category = IntentCategory.MULTI_STEP_ANALYSIS
    else:
        # Prefer a specific registered capability over generic dataset inspection.
        category = next(
            (rule.category for rule, _ in matches if rule.category != IntentCategory.DATA_ANALYSIS),
            matches[0][0].category,
        )

    required_kinds = set().union(*(set(rule.requires) for rule, _ in matches))
    schema_kinds = _schema_requirement_kinds(tools, matched_tools)
    required_kinds.update(schema_kinds)
    # A research question in the objective supplies its topic. A bare request
    # to research has intent but not enough topic information.
    if RequiredInputKind.RESEARCH_TOPIC in required_kinds:
        topic_stop_words = {
            "research", "web", "online", "internet", "about", "on", "find", "look", "up",
            "the", "this", "that", "a", "an", "it", "something", "please", "conduct",
            "investigate", "current", "latest", "recent", "for", "sources", "and",
        }
        topic_words = [word for word in re.findall(r"[a-z0-9]+", normalized) if word not in topic_stop_words]
        if len(topic_words) >= 2:
            supplied.add(RequiredInputKind.RESEARCH_TOPIC)
    # The stated objective is itself the goal for a requested analysis/report.
    if normalized.strip():
        supplied.add(RequiredInputKind.ANALYSIS_GOAL)

    missing_kinds = required_kinds - supplied
    required = [InputRequirement(kind=kind) for kind in sorted(required_kinds, key=lambda item: item.value)]
    missing = [InputRequirement(kind=kind) for kind in sorted(missing_kinds, key=lambda item: item.value)]
    # Ambiguous or capability-missing requests retain lower confidence without
    # asserting facts about user data or inventing column names.
    confidence = 0.88 if len(matches) == 1 else 0.78
    if missing_kinds:
        confidence = min(confidence, 0.68)
    if multi_step:
        confidence = min(confidence, 0.82)

    return IntentAssessment(
        category=category,
        confidence=confidence,
        required_inputs=required,
        missing_inputs=missing,
        is_multi_step=multi_step,
    )


class IntentUnderstander:
    """Reusable wrapper for an injected registry or the built-in registry."""

    def __init__(self, registry: ToolRegistry | None = None):
        self.registry = registry or tool_registry

    def assess(
        self,
        objective: str,
        available_inputs: Iterable[RequiredInputKind | str] = (),
    ) -> IntentAssessment:
        return assess_intent(objective, self.registry, available_inputs)
