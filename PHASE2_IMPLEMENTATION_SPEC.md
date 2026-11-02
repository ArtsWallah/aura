# AURA Phase 2 Implementation Specification

Scope: reliable asynchronous orchestration on the existing FastAPI, SQLite, tool registry, and LLM provider architecture. No Redis, Celery, new provider framework, or Phase 3 work.

## Task state machine

Task status changes are performed through one validated transition operation in the application process. `COMPLETED`, `FAILED`, and `CANCELLED` are terminal. SQLite writes remain serialized with transactions; transition validation is not a cross-process compare-and-set.

| Current state | Allowed next states | Meaning |
| --- | --- | --- |
| `QUEUED` | `PLANNING`, `CANCELLED`, `FAILED` | Persisted, waiting for a managed worker. |
| `PLANNING` | `RUNNING`, `WAITING`, `CANCEL_REQUESTED`, `FAILED` | A plan is being created and validated. `WAITING` is for an explicit approval or recovery decision. |
| `RUNNING` | `WAITING`, `RETRYING`, `CANCEL_REQUESTED`, `COMPLETED`, `FAILED` | Validated steps are executing. |
| `WAITING` | `RUNNING`, `RETRYING`, `CANCEL_REQUESTED`, `CANCELLED`, `FAILED` | Execution is paused for approval or explicit recovery. |
| `RETRYING` | `RUNNING`, `WAITING`, `CANCEL_REQUESTED`, `FAILED` | A declared retry-safe operation is being scheduled again. |
| `CANCEL_REQUESTED` | `CANCELLED`, `FAILED` | Stop scheduling new work; wait for in-flight synchronous work to reach a safe boundary. |
| `COMPLETED` | — | Terminal success. |
| `FAILED` | — | Terminal failure; a new recovery action creates an auditable transition before execution resumes. |
| `CANCELLED` | — | Terminal cancellation, recorded only after active synchronous work has stopped. |

All transitions update the encrypted state and indexed status in one SQLite transaction. Invalid transitions are rejected. Legacy state values map as `pending → QUEUED`, `planning → PLANNING`, `executing/evaluating → RUNNING`; terminal legacy values retain their meaning.

## Persistent scheduling and recovery

SQLite durably stores task lifecycle metadata and atomically claims queued rows. Objectives, file paths, tool arguments, raw outputs, dataset rows, and deliverables are never written to SQLite, even encrypted; queued execution payloads exist only in process memory. Therefore, a queued task with no in-memory payload at restart is marked `FAILED` with `recovery_required`, and the user must submit it again. A running task interrupted during a step is also marked `FAILED` with recovery metadata; no tool is silently replayed. This is the ambiguous-outcome policy.

Shutdown stops accepting claims and waits for worker tasks and synchronous calls to finish. After the configured grace period it logs that it is still waiting for safe tool boundaries; Python cannot safely kill a running thread. A forced process termination can leave work ambiguous, which startup recovery marks failed without replay.

## Synchronous tool cancellation boundary

Synchronous tools run in a bounded thread pool, never on the FastAPI event loop. A cancellation request prevents future steps immediately. Python cannot safely terminate an arbitrary running thread, so a synchronous step remains in `CANCEL_REQUESTED` until its function returns; only then may the task become `CANCELLED`. Timeouts are recorded when a tool exceeds its deadline, but the worker does not retry or run dependent steps while the original call could still be active. Async tools can be cancelled at their await boundary. This limitation is surfaced in task state and events.

## Retry safety

Tool definitions gain an explicit `retry_safe` declaration, false by default. Built-in read-only analytics, file inspection, document lookup, system health, and idempotent GET research opt in explicitly. Chart/report creation and any future mutation remain non-retryable unless specifically proven safe. Retries are bounded by the lesser of per-step and global configuration, occur only for retry-safe tools and a classified transient failure or an evaluator-supplied safe parameter correction, and never occur after cancellation. Permission level does not imply retry safety.

## Plan validation

Every planner result, including deterministic fallback plans, must pass validation before any tool runs:

- Plan and step counts obey `MAX_PLAN_STEPS` and are non-empty.
- Step IDs are unique and conform to a bounded safe identifier format.
- Every dependency refers to a known, different step; no cycles exist.
- Each tool is registered and permitted for the task privacy mode and runtime location.
- Arguments validate against the registered Pydantic input model before execution.
- Required approval and permission policy is satisfied before scheduling.
- Failed prerequisites block dependents; no implicit continuation occurs.

Validation failures become structured task/step failures without exposing arguments in logs, API errors, or events.

## Configurable execution limits

Existing `MAX_PLAN_STEPS`, `MAX_STEP_RETRIES`, and `STEP_TIMEOUT_SECONDS` remain authoritative global ceilings. Add bounded settings for simultaneous tasks, simultaneous tools, and worker shutdown grace. Per-tool timeout metadata remains an upper bound and cannot exceed the global step timeout. Queue depth is durable in SQLite rather than an unbounded in-memory work queue.

## Durable privacy-safe SSE

Task creation commits `QUEUED` state and `task.created` before returning `202 Accepted` with the single generated task ID. Frontend subscription starts immediately. Events are persisted with monotonically increasing per-task IDs and emitted using the approved event names. `Last-Event-ID` replays later events; normal EventSource reconnect does not create another task.

Persisted and streamed event projections are allowlisted: event name/ID/time, task status, bounded step ordinal and registered tool name, duration, and error category. They never include objectives, prompts, plan arguments, raw outputs, file contents, dataset rows, exception text, or provider responses. Sensitive inputs and outputs remain in process memory only. Replay retention is bounded per task while preserving sequence monotonicity.

## Database/API migration and compatibility

Use idempotent additive SQLite migrations for worker claim metadata and an event table. Preserve the existing encrypted checkpoint table and rows. Status migration does not rewrite encrypted state. Legacy plaintext rows are encrypted in place with their original content preserved when a valid key is available; missing/invalid keys leave rows untouched. New checkpoint writes omit request text, file paths, tool arguments, raw outputs, dataset rows, and artifacts. Historical encrypted rows may retain previously stored values until an explicit operation updates that row; no bulk destructive scrub is run automatically. This is a migration/privacy tradeoff to preserve existing user data.

`POST /api/tasks` changes from synchronous completion (`200`) to queued acceptance (`202`). The request payload is held in process memory while queued because persisting it would violate the no-sensitive-payload persistence rule. A restart with a queued payload unavailable marks the task failed for resubmission; it is not silently replayed. Existing clients must poll or subscribe for the result. Task lookup/list/cancel and SSE retain their routes; cancellation reports `cancel_requested` while active work reaches a safe boundary. Frontend types and UI are updated without a visual redesign.

## Verification gates

Each implementation increment gets focused tests and a diff review before the next increment. Increment 6 runs the full backend suite, Phase 1 security tests, frontend lint/build, and the new orchestration acceptance tests. No Phase 3 features are included.
