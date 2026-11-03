import json
import uuid
import re
from typing import Any
from app.llm.base import BaseLLMProvider, LLMMessage, LLMResponse, LLMToolCall
from app.core.logging import logger

class TestReasoningProvider(BaseLLMProvider):
    """
    [EXPLICIT MOCK / TEST REASONING PROVIDER]
    This provider operates entirely offline using deterministic rule-based reasoning.
    It satisfies the BaseLLMProvider interface so AURA can be tested and developed without
    cloud API keys. This is NOT a real neural LLM.
    """
    def __init__(self, model_name: str = "aura-test-reasoner-v1"):
        self.model_name = model_name

    def is_available(self) -> bool:
        return True

    async def chat(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        response_format_json: bool = False
    ) -> LLMResponse:
        logger.info("[TestReasoningProvider] Evaluating prompt via deterministic rule-based planner...")
        
        last_msg = messages[-1].content if messages else ""
        prompt_lower = last_msg.lower()

        # Check if caller expects structured JSON for Task Planning
        if response_format_json:
            # Generate deterministic plan based on user objective
            plan_obj = self._generate_deterministic_plan(prompt_lower, messages)
            return LLMResponse(
                content=json.dumps(plan_obj),
                tool_calls=[],
                prompt_tokens=len(last_msg) // 4,
                completion_tokens=len(json.dumps(plan_obj)) // 4,
                model=self.model_name,
                provider="test_reasoner"
            )

        # Direct text reasoning or tool calls
        return LLMResponse(
            content=f"[Deterministic Test Engine] Completed reasoning for: '{last_msg[:80]}...'",
            tool_calls=[],
            prompt_tokens=20,
            completion_tokens=30,
            model=self.model_name,
            provider="test_reasoner"
        )

    def _generate_deterministic_plan(self, prompt: str, messages: list[LLMMessage]) -> dict[str, Any]:
        """
        Decomposes instructions into structured execution steps.
        """
        # Look for file path mentioned in user prompt or previous messages
        file_path = None
        for msg in messages:
            match = re.search(r"['\"]?([\w\-\\/:.]+\.(?:csv|xlsx|xls|txt|md))['\"]?", msg.content)
            if match:
                file_path = match.group(1)
                break
                
        if not file_path:
            file_path = "data/sample_sales.csv"

        steps = []
        
        # Scenario: Data Analysis / Sales Dataset Request
        if any(w in prompt for w in ["analyze", "dataset", "sales", "csv", "excel", "trend", "underperforming"]):
            steps.append({
                "step_id": "step_1",
                "title": "Inspect Dataset Schema & Integrity",
                "description": "Inspect tabular columns, data types, null values, and sample records.",
                "capability": "dataset",
                "tool_name": "inspect_dataset",
                "parameters": {"file_path": file_path, "sample_rows": 5},
                "dependencies": []
            })
            steps.append({
                "step_id": "step_2",
                "title": "Compute Descriptive Statistics",
                "description": "Calculate statistical metrics (mean, median, quartiles, IQR) for key numeric columns.",
                "capability": "analytics",
                "tool_name": "descriptive_statistics",
                "parameters": {"file_path": file_path},
                "dependencies": ["step_1"]
            })
            steps.append({
                "step_id": "step_3",
                "title": "Analyze Category Performance & Detect Underperforming Segments",
                "description": "Perform group comparison to identify high-performing and underperforming categories.",
                "capability": "analytics",
                "tool_name": "group_comparison",
                "parameters": {
                    "file_path": file_path,
                    "group_by": "category",
                    "metric_column": "revenue",
                    "aggregations": ["sum", "mean", "count"]
                },
                "dependencies": ["step_1"]
            })
            steps.append({
                "step_id": "step_4",
                "title": "Generate Performance Visualization Chart",
                "description": "Render a bar chart visualising revenue distribution by product category.",
                "capability": "visualization",
                "tool_name": "generate_chart",
                "parameters": {
                    "file_path": file_path,
                    "chart_type": "bar",
                    "title": "Revenue Performance by Product Category",
                    "x_column": "category",
                    "y_column": "revenue"
                },
                "dependencies": ["step_3"]
            })
            steps.append({
                "step_id": "step_5",
                "title": "Synthesize Analytical Report",
                "description": "Consolidate findings, statistics, visualizations, and actionable recommendations into a publication-grade report.",
                "capability": "report",
                "tool_name": "generate_report",
                "parameters": {
                    "title": "Comprehensive Sales Performance & Category Analytics Report",
                    "executive_summary": "Autonomous analysis completed. Primary revenue contributors and underperforming product lines have been systematically identified.",
                    "findings": [
                        "Top-performing product categories contribute the vast majority of total enterprise revenue.",
                        "Underperforming segments have been flagged for marketing or inventory re-evaluation.",
                        "Statistical distributions indicate robust volume with identifiable outliers."
                    ],
                    "recommendations": [
                        "Reallocate inventory and promotional budget toward top quartile categories.",
                        "Perform root-cause analysis on underperforming low-margin items.",
                        "Establish continuous monitoring on revenue velocity."
                    ],
                    "chart_paths": []
                },
                "dependencies": ["step_2", "step_3", "step_4"]
            })
        elif any(w in prompt for w in ["research", "search", "market", "competitor"]):
            steps.append({
                "step_id": "step_1",
                "title": "Conduct Web Research",
                "description": "Query external sources for relevant industry intelligence and cited facts.",
                "capability": "research",
                "tool_name": "web_research",
                "parameters": {"query": prompt, "max_results": 5},
                "dependencies": []
            })
            steps.append({
                "step_id": "step_2",
                "title": "Generate Research Summary Report",
                "description": "Summarize research findings with citations and limitations.",
                "capability": "report",
                "tool_name": "generate_report",
                "parameters": {
                    "title": f"Market Research Intelligence: {prompt[:40]}",
                    "executive_summary": "Automated web research synthesis with verified source citations.",
                    "findings": ["External verified sources successfully extracted and indexed."],
                    "recommendations": ["Conduct deeper validation on high-relevance domain links."]
                },
                "dependencies": ["step_1"]
            })
        else:
            # Generic system diagnostic or inspection
            steps.append({
                "step_id": "step_1",
                "title": "Inspect System Diagnostic State",
                "description": "Check system operational health, tool registry status, and runtime environment.",
                "capability": "system",
                "tool_name": "system_health",
                "parameters": {"check_storage": True},
                "dependencies": []
            })

        return {
            "plan_id": f"plan_{uuid.uuid4().hex[:8]}",
            "goal": prompt,
            "steps": steps,
            "estimated_steps_count": len(steps)
        }

