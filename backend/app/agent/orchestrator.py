import asyncio
import time
import uuid
from typing import AsyncGenerator
from app.agent.state import TaskState, TaskStatus, StepStatus, TaskEvent, TaskMetrics
from app.agent.planner import TaskPlanner
from app.agent.executor import executor
from app.agent.evaluator import evaluator
from app.agent.checkpoint import checkpointer
from app.llm.router import llm_router
from app.core.logging import logger
from app.privacy.outbound import privacy_scope

class AgentOrchestrator:
    def __init__(self):
        self._active_tasks: dict[str, TaskState] = {}
        self._event_queues: dict[str, list[asyncio.Queue]] = {}

    def get_task_state(self, task_id: str) -> TaskState | None:
        if task_id in self._active_tasks:
            return self._active_tasks[task_id]
        # Fall back to database checkpoint
        return checkpointer.load_checkpoint(task_id)

    def register_event_listener(self, task_id: str) -> asyncio.Queue:
        q = asyncio.Queue()
        self._event_queues.setdefault(task_id, []).append(q)
        return q

    def remove_event_listener(self, task_id: str, q: asyncio.Queue):
        if task_id in self._event_queues and q in self._event_queues[task_id]:
            self._event_queues[task_id].remove(q)

    async def _emit_event(self, event: TaskEvent):
        queues = self._event_queues.get(event.task_id, [])
        for q in queues:
            await q.put(event)

    async def run_task(
        self,
        objective: str,
        file_path: str | None = None,
        privacy_mode: str = "hybrid",
        task_id: str | None = None
    ) -> TaskState:
        with privacy_scope(privacy_mode):
            return await self._run_task(objective, file_path, privacy_mode, task_id)

    async def _run_task(
        self,
        objective: str,
        file_path: str | None = None,
        privacy_mode: str = "hybrid",
        task_id: str | None = None
    ) -> TaskState:
        t_id = task_id or f"task_{uuid.uuid4().hex[:10]}"
        state = TaskState(
            task_id=t_id,
            objective=objective,
            privacy_mode=privacy_mode,
            status=TaskStatus.PLANNING
        )
        self._active_tasks[t_id] = state
        checkpointer.save_checkpoint(state)
        
        start_time = time.perf_counter()

        await self._emit_event(TaskEvent(
            event_type="task_started",
            task_id=t_id,
            message=f"Starting task execution under privacy mode: '{privacy_mode}'.",
            data={"objective": objective, "privacy_mode": privacy_mode}
        ))

        try:
            # 1. Routing & Planning
            provider = llm_router.get_provider(privacy_mode=privacy_mode)
            planner = TaskPlanner(provider)
            
            plan = await planner.create_plan(objective, file_path)
            state.plan = plan
            state.status = TaskStatus.EXECUTING
            state.metrics.llm_calls_count += 1
            checkpointer.save_checkpoint(state)

            await self._emit_event(TaskEvent(
                event_type="plan_created",
                task_id=t_id,
                message=f"Plan synthesized with {len(plan.steps)} steps.",
                data={"steps": [s.model_dump() for s in plan.steps]}
            ))

            # 2. Dependency-aware Execution Loop (DAG)
            queue_for_executor = asyncio.Queue()
            
            # Start background forwarder for executor events
            async def forward_events():
                while True:
                    ev = await queue_for_executor.get()
                    await self._emit_event(ev)
                    queue_for_executor.task_done()
                    
            forward_task = asyncio.create_task(forward_events())

            completed_step_ids: set[str] = set()
            total_steps = len(plan.steps)

            while len(completed_step_ids) < total_steps:
                # Find all pending steps whose dependencies are satisfied
                ready_steps = [
                    s for s in plan.steps
                    if s.status == StepStatus.PENDING and all(dep in completed_step_ids for dep in s.dependencies)
                ]

                if not ready_steps:
                    # Check if any steps failed or are stuck
                    failed = [s for s in plan.steps if s.status == StepStatus.FAILED]
                    if failed:
                        state.status = TaskStatus.FAILED
                        state.error_summary = f"Execution halted due to failed steps: {[s.step_id for s in failed]}"
                        break
                    else:
                        break # No more actionable steps

                # Execute ready steps in parallel when more than 1 are independent
                logger.info(f"Executing a batch of {len(ready_steps)} tool step(s).")
                batch_tasks = [
                    executor.execute_step(step, state, queue_for_executor)
                    for step in ready_steps
                ]
                
                await asyncio.gather(*batch_tasks)

                for step in ready_steps:
                    if step.status == StepStatus.COMPLETED:
                        completed_step_ids.add(step.step_id)
                    elif step.status == StepStatus.FAILED:
                        state.status = TaskStatus.FAILED
                        break

                checkpointer.save_checkpoint(state)
                if state.status == TaskStatus.FAILED:
                    break

            forward_task.cancel()

            # 3. Final Task Evaluation
            is_successful, eval_reason = evaluator.evaluate_task_completion(state)
            state.updated_at = time.time()
            state.metrics.total_execution_time_ms = (time.perf_counter() - start_time) * 1000.0

            if is_successful:
                state.status = TaskStatus.COMPLETED
                
                # Extract final deliverable
                report_step = next((s for s in reversed(plan.steps) if s.tool_name == "generate_report" and s.status == StepStatus.COMPLETED), None)
                if report_step and report_step.output:
                    state.final_deliverable = report_step.output
                else:
                    # Aggregate last output
                    last_step = next((s for s in reversed(plan.steps) if s.status == StepStatus.COMPLETED), None)
                    state.final_deliverable = last_step.output if last_step else {"message": "Execution finished."}

                await self._emit_event(TaskEvent(
                    event_type="task_completed",
                    task_id=t_id,
                    message="Task completed successfully.",
                    data={"deliverable": state.final_deliverable, "metrics": state.metrics.model_dump()}
                ))
            else:
                state.status = TaskStatus.FAILED
                state.error_summary = eval_reason
                await self._emit_event(TaskEvent(
                    event_type="task_failed",
                    task_id=t_id,
                    message=f"Task could not be verified: {eval_reason}",
                    data={"error": eval_reason}
                ))

        except Exception as e:
            state.status = TaskStatus.FAILED
            state.error_summary = "Internal task error."
            state.updated_at = time.time()
            state.metrics.total_execution_time_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error("Task execution failed with an internal error.")
            await self._emit_event(TaskEvent(
                event_type="task_failed",
                task_id=t_id,
                message="Task execution failed.",
                data={"error": "Internal task error."}
            ))

        checkpointer.save_checkpoint(state)
        return state

orchestrator = AgentOrchestrator()

