import json
import uuid
from typing import Any
from app.agent.state import ExecutionPlan, ExecutionStep, StepStatus
from app.llm.base import BaseLLMProvider, LLMMessage
from app.tools.registry import tool_registry
from app.core.logging import logger

PLANNER_SYSTEM_PROMPT = """You are AURA's Master Task Planner & Decomposer.
Your responsibility is to decompose complex user objectives into a logical, ordered Directed Acyclic Graph (DAG) of execution steps.

RULES:
1. Every step must use an approved deterministic tool from the provided registry.
2. Clearly declare step dependencies. If Step 2 needs Step 1's dataset summary, declare dependencies: ["step_1"].
3. Do not invent tools. Only choose from the provided tool definitions.
4. Output STRICT JSON conforming to the requested schema. No conversational filler.

Schema expected:
{
  "plan_id": "plan_unique_id",
  "goal": "summarized user objective",
  "steps": [
    {
      "step_id": "step_1",
      "title": "Short title",
      "description": "Step purpose",
      "capability": "dataset",
      "tool_name": "inspect_dataset",
      "parameters": {"file_path": "...", ...},
      "dependencies": []
    }
  ]
}
"""

class TaskPlanner:
    def __init__(self, llm_provider: BaseLLMProvider):
        self.llm = llm_provider

    async def create_plan(
        self,
        objective: str,
        file_path: str | None = None,
        context: dict[str, Any] | None = None
    ) -> ExecutionPlan:
        tools_decl = tool_registry.get_tool_schemas_for_llm()
        tools_summary = "\n".join([f"- {t['name']}: {t['description']}" for t in tools_decl])
        
        user_prompt = f"""User Objective: {objective}
Attached File: {file_path or 'None'}

Available Registered Tools:
{tools_summary}

Create a structured execution plan to achieve the objective."""

        messages = [
            LLMMessage(role="system", content=PLANNER_SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_prompt)
        ]

        logger.info(f"Generating structured execution plan for objective: '{objective[:60]}...'")
        response = await self.llm.chat(messages, response_format_json=True)
        
        try:
            plan_dict = json.loads(response.content or "{}")
            raw_steps = plan_dict.get("steps", [])
            
            steps = []
            for s in raw_steps:
                steps.append(ExecutionStep(
                    step_id=s.get("step_id", f"step_{len(steps)+1}"),
                    title=s.get("title", "Execution Step"),
                    description=s.get("description", ""),
                    capability=s.get("capability", "analytics"),
                    tool_name=s.get("tool_name", ""),
                    parameters=s.get("parameters", {}),
                    dependencies=s.get("dependencies", []),
                    status=StepStatus.PENDING
                ))

            if not steps:
                raise ValueError("LLM generated an empty step list.")

            return ExecutionPlan(
                plan_id=plan_dict.get("plan_id", f"plan_{uuid.uuid4().hex[:8]}"),
                goal=objective,
                steps=steps
            )
        except Exception as e:
            logger.error(f"Failed to parse LLM plan output: {e}. Raw content: {response.content}")
            # Fallback deterministic plan construction
            return self._create_fallback_plan(objective, file_path)

    def _create_fallback_plan(self, objective: str, file_path: str | None) -> ExecutionPlan:
        target_file = file_path or "data/sample_sales.csv"
        return ExecutionPlan(
            plan_id=f"plan_fallback_{uuid.uuid4().hex[:6]}",
            goal=objective,
            steps=[
                ExecutionStep(
                    step_id="step_1",
                    title="Inspect Dataset Integrity",
                    description="Inspect dimensions, column types, and data health.",
                    capability="dataset",
                    tool_name="inspect_dataset",
                    parameters={"file_path": target_file, "sample_rows": 5},
                    dependencies=[]
                ),
                ExecutionStep(
                    step_id="step_2",
                    title="Calculate Descriptive Statistics",
                    description="Calculate distribution statistics for numeric attributes.",
                    capability="analytics",
                    tool_name="descriptive_statistics",
                    parameters={"file_path": target_file},
                    dependencies=["step_1"]
                ),
                ExecutionStep(
                    step_id="step_3",
                    title="Synthesize Analytical Report",
                    description="Synthesize findings into markdown/html deliverable.",
                    capability="report",
                    tool_name="generate_report",
                    parameters={
                        "title": "Automated Analytical Assessment",
                        "executive_summary": "Autonomous analysis performed on provided data.",
                        "findings": ["Dataset examined and statistical measures extracted."]
                    },
                    dependencies=["step_2"]
                )
            ]
        )

