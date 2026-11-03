import { ExecutionTrace, TaskStatus, TraceStep } from "@/lib/types";

const categoryLabel = (value: string | null | undefined) => (value ? value.replaceAll("_", " ") : "Unknown");

function outcomeMessage(status: ExecutionTrace["evaluation_status"], taskStatus: TaskStatus) {
  if (status === "succeeded") return "All planned steps completed with the expected evidence.";
  if (status === "partial_success") return "Some useful work completed, but one or more steps did not.";
  if (status === "execution_failed") return "Execution ended without useful completed work.";
  if (status === "inconclusive") return "Available evidence could not confirm the requested outcome.";
  if (taskStatus === "queued" || taskStatus === "planning") return "Waiting for a validated plan.";
  if (taskStatus === "cancel_requested") return "Cancellation was requested; active work is stopping at a safe boundary.";
  if (taskStatus === "cancelled") return "The task was cancelled before its outcome could be confirmed.";
  if (taskStatus === "failed") return "The task failed before a final evaluation was available.";
  if (taskStatus === "completed") return "The task completed; no final evaluation was available in this trace.";
  return "Execution is in progress.";
}

function statusClass(status: TraceStep["status"]) {
  if (status === "completed") return "text-emerald-300";
  if (status === "failed" || status === "interrupted") return "text-rose-300";
  if (status === "skipped") return "text-amber-300";
  if (status === "in_progress") return "text-indigo-300";
  return "text-slate-400";
}

interface Props {
  trace: ExecutionTrace | null;
  taskStatus: TaskStatus | null;
}

export default function ExecutionTracePanel({ trace, taskStatus }: Props) {
  return (
    <section aria-label="Execution trace" className="rounded-xl border border-slate-800 bg-slate-900/40 p-5 space-y-4">
      <div>
        <h2 className="text-sm font-semibold text-white">Execution trace</h2>
        <p className="mt-1 text-xs text-slate-400">
          Safe summary of AURA's interpretation, validated plan, and recorded step outcomes.
        </p>
      </div>
      {!trace ? (
        <p className="text-sm text-slate-400">
          {taskStatus === "queued" || taskStatus === "planning"
            ? "Waiting for the validated plan."
            : "No intelligence trace is available for this task."}
        </p>
      ) : (
        <>
          <div className="grid gap-3 sm:grid-cols-3">
            <div className="rounded-lg bg-slate-950/70 p-3">
              <p className="text-[10px] font-semibold uppercase text-slate-500">Understanding</p>
              <p className="mt-1 text-sm text-white">{categoryLabel(trace.intent_category)}</p>
              {trace.intent_confidence !== null && (
                <p className="text-xs text-slate-400">Confidence {Math.round(trace.intent_confidence * 100)}%</p>
              )}
              <p className="mt-1 text-xs text-slate-500">Planner: {categoryLabel(trace.provider_identity)}</p>
            </div>
            <div className="rounded-lg bg-slate-950/70 p-3 sm:col-span-2">
              <p className="text-[10px] font-semibold uppercase text-slate-500">Planning</p>
              <p className="mt-1 text-xs text-slate-300">
                {trace.step_statuses.length} ordered step{trace.step_statuses.length === 1 ? "" : "s"} in the validated plan
              </p>
              {trace.step_statuses.length > 0 && (
                <p className="mt-1 text-xs text-slate-400">
                  {trace.step_statuses.map((step) => step.tool_label || step.tool_name).join(" > ")}
                </p>
              )}
            </div>
          </div>

          <div className="space-y-2">
            <p className="text-[10px] font-semibold uppercase text-slate-500">Execution</p>
            {trace.step_statuses.map((step, index) => (
              <div key={step.step_id} className="flex flex-wrap items-center gap-x-3 gap-y-1 rounded-lg border border-slate-800 bg-slate-950/50 px-3 py-2 text-xs">
                <span className="w-5 text-slate-500">{index + 1}.</span>
                <span className="min-w-32 text-slate-200">{step.tool_label || step.tool_name}</span>
                <span className="text-slate-500">{categoryLabel(step.tool_category || "tool")}</span>
                <span className={`capitalize ${statusClass(step.status)}`}>{categoryLabel(step.status)}</span>
                <span className="text-slate-400">Retries {step.retry_count}/{step.retry_limit}</span>
                {step.failure_category && step.failure_category !== "none" && (
                  <span className="text-amber-300">{categoryLabel(step.failure_category)}</span>
                )}
                {step.artifacts.filter((artifact) => artifact.available).map((artifact) => (
                  <span key={artifact.artifact_type} className="text-cyan-300">
                    {categoryLabel(artifact.artifact_type)} x {artifact.count}
                  </span>
                ))}
              </div>
            ))}
          </div>

          <div className="rounded-lg border border-slate-800 bg-slate-950/70 p-3">
            <p className="text-[10px] font-semibold uppercase text-slate-500">Outcome</p>
            <p className="mt-1 text-sm text-white">
              {trace.evaluation_status ? categoryLabel(trace.evaluation_status) : categoryLabel(taskStatus || "waiting")}
            </p>
            <p className="mt-1 text-xs text-slate-400">{outcomeMessage(trace.evaluation_status, taskStatus || "queued")}</p>
          </div>
        </>
      )}
    </section>
  );
}
