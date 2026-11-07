import { ExecutionTrace, TaskStatus, TraceStep } from "@/lib/types";

const categoryLabel = (value: string | null | undefined) => (value ? value.replaceAll("_", " ") : "Unknown");

function outcomeMessage(status: ExecutionTrace["evaluation_status"], taskStatus: TaskStatus) {
  if (status === "succeeded") return "All planned steps completed with verified cryptographic evidence.";
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

function statusBadge(status: TraceStep["status"]) {
  if (status === "completed") {
    return "border border-black bg-black text-white font-mono font-bold shadow-sm";
  }
  if (status === "failed" || status === "interrupted") {
    return "border border-zinc-400 bg-zinc-200 text-zinc-800 font-mono";
  }
  if (status === "skipped") {
    return "border border-zinc-200 bg-zinc-100 text-zinc-400 font-mono";
  }
  if (status === "in_progress") {
    return "bg-black text-white font-mono font-bold animate-pulse shadow-[0_0_12px_rgba(0,0,0,0.3)]";
  }
  return "border border-zinc-200 bg-zinc-50 text-zinc-600 font-mono";
}

interface Props {
  trace: ExecutionTrace | null;
  taskStatus: TaskStatus | null;
}

export default function ExecutionTracePanel({ trace, taskStatus }: Props) {
  return (
    <section aria-label="Execution trace" className="rounded-2xl border border-black/10 bg-white/90 p-6 space-y-5 backdrop-blur-xl glass-panel shadow-sm">
      <div className="flex items-center justify-between border-b border-black/10 pb-4">
        <div>
          <h2 className="text-xs font-mono uppercase tracking-[0.2em] font-bold text-black flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-black animate-ping" />
            EXECUTION TELEMETRY & TRACE
          </h2>
          <p className="mt-1 text-xs text-zinc-600 font-sans">
            Cryptographic audit of intent interpretation, DAG validation, and verified step artifacts.
          </p>
        </div>
      </div>
      {!trace ? (
        <div className="rounded-xl border border-dashed border-black/15 bg-zinc-50/50 p-6 text-center">
          <p className="text-xs font-mono text-zinc-500">
            {taskStatus === "queued" || taskStatus === "planning"
              ? "Awaiting plan validation..."
              : "No intelligence trace registered for this task."}
          </p>
        </div>
      ) : (
        <>
          <div className="grid gap-3 sm:grid-cols-3 font-mono">
            <div className="rounded-xl border border-black/10 bg-zinc-50/80 p-4 transition-all hover:border-black/30 hover:bg-white hover:shadow-md">
              <p className="text-[10px] font-bold uppercase tracking-wider text-zinc-500">Understanding</p>
              <p className="mt-1.5 text-sm font-bold text-black uppercase">{categoryLabel(trace.intent_category)}</p>
              {trace.intent_confidence !== null && (
                <p className="mt-1 text-xs text-zinc-600">CONFIDENCE: {Math.round(trace.intent_confidence * 100)}%</p>
              )}
              <p className="mt-1 text-[11px] text-zinc-500">PLANNER: {categoryLabel(trace.provider_identity)}</p>
            </div>
            <div className="rounded-xl border border-black/10 bg-zinc-50/80 p-4 sm:col-span-2 transition-all hover:border-black/30 hover:bg-white hover:shadow-md">
              <p className="text-[10px] font-bold uppercase tracking-wider text-zinc-500">Validated Plan Structure</p>
              <p className="mt-1.5 text-xs text-zinc-900 font-semibold">
                {trace.step_statuses.length} ordered step{trace.step_statuses.length === 1 ? "" : "s"} locked in execution DAG
              </p>
              {trace.step_statuses.length > 0 && (
                <p className="mt-2 text-xs text-zinc-600 leading-relaxed">
                  {trace.step_statuses.map((step) => step.tool_label || step.tool_name).join("  →  ")}
                </p>
              )}
            </div>
          </div>

          <div className="space-y-2">
            <p className="text-[10px] font-mono font-bold uppercase tracking-[0.2em] text-zinc-500">Execution Steps</p>
            <div className="space-y-2">
              {trace.step_statuses.map((step, index) => (
                <div key={step.step_id} className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-black/10 bg-zinc-50/60 px-4 py-3 text-xs transition-all hover:border-black/30 hover:bg-white hover:shadow-sm">
                  <div className="flex items-center gap-3">
                    <span className="font-mono text-zinc-400 w-5">{String(index + 1).padStart(2, "0")}.</span>
                    <span className="font-bold text-black tracking-tight">{step.tool_label || step.tool_name}</span>
                    <span className="font-mono text-[10px] uppercase text-zinc-600 border border-black/10 bg-white rounded px-1.5 py-0.5">
                      {categoryLabel(step.tool_category || "tool")}
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className={`px-2.5 py-0.5 rounded text-[10px] tracking-wider uppercase ${statusBadge(step.status)}`}>
                      {categoryLabel(step.status)}
                    </span>
                    <span className="font-mono text-[11px] text-zinc-500">
                      R: {step.retry_count}/{step.retry_limit}
                    </span>
                    {step.artifacts.filter((artifact) => artifact.available).map((artifact) => (
                      <span key={artifact.artifact_type} className="font-mono text-[10px] uppercase border border-black/15 bg-white text-zinc-800 px-2 py-0.5 rounded font-semibold shadow-2xs">
                        {categoryLabel(artifact.artifact_type)}: {artifact.count}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-xl border border-black/10 bg-zinc-50/80 p-4 space-y-1">
            <p className="text-[10px] font-mono font-bold uppercase tracking-wider text-zinc-500">Final Outcome Evaluation</p>
            <p className="text-sm font-mono font-black text-black uppercase tracking-tight">
              {trace.evaluation_status ? categoryLabel(trace.evaluation_status) : categoryLabel(taskStatus || "waiting")}
            </p>
            <p className="text-xs text-zinc-600 font-sans">{outcomeMessage(trace.evaluation_status, taskStatus || "queued")}</p>
          </div>
        </>
      )}
    </section>
  );
}
