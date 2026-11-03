"use client";

import { useEffect, useState, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { 
  SearchCode, 
  Clock, 
  CheckCircle2, 
  XCircle, 
  ChevronRight, 
  Cpu, 
  Wrench, 
  ShieldCheck,
  RotateCw
} from "lucide-react";
import { fetchTasks, fetchTask } from "@/lib/api";
import { TaskState, ExecutionStep } from "@/lib/types";

function InspectorContent() {
  const searchParams = useSearchParams();
  const initialTaskId = searchParams.get("task_id");

  const [tasks, setTasks] = useState<any[]>([]);
  const [selectedTaskId, setSelectedTaskId] = useState<string>(initialTaskId || "");
  const [taskState, setTaskState] = useState<TaskState | null>(null);
  const [selectedStep, setSelectedStep] = useState<ExecutionStep | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  useEffect(() => {
    fetchTasks(20).then((list) => {
      setTasks(list);
      if (!selectedTaskId && list.length > 0) {
        setSelectedTaskId(list[0].task_id);
      }
    });
  }, [selectedTaskId]);

  useEffect(() => {
    if (selectedTaskId) {
      setLoading(true);
      fetchTask(selectedTaskId)
        .then((data) => {
          setTaskState(data);
          if (data.plan?.steps && data.plan.steps.length > 0) {
            setSelectedStep(data.plan.steps[0]);
          }
        })
        .finally(() => setLoading(false));
    }
  }, [selectedTaskId]);

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
            <SearchCode className="h-6 w-6 text-indigo-400" />
            Execution Inspector & Trace Audit
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Examine step execution timings, tool parameters, evaluation validations, and retry audits without exposing hidden model chain-of-thought.
          </p>
        </div>

        {/* Task Selector Dropdown */}
        <div className="flex items-center gap-2">
          <label className="text-xs text-slate-400 font-medium">Select Run:</label>
          <select
            value={selectedTaskId}
            onChange={(e) => setSelectedTaskId(e.target.value)}
            className="rounded-lg border border-slate-800 bg-slate-900 px-3 py-1.5 text-xs text-slate-200 focus:border-indigo-500 focus:outline-none"
          >
            {tasks.map((t) => (
              <option key={t.task_id} value={t.task_id}>
                {t.task_id.slice(0, 16)}... ({t.status})
              </option>
            ))}
          </select>
        </div>
      </div>

      {loading ? (
        <div className="p-12 text-center text-slate-400 text-sm">Loading execution trace...</div>
      ) : !taskState ? (
        <div className="p-12 text-center text-slate-400 text-sm">No task execution selected.</div>
      ) : (
        <div className="space-y-6">
          {/* Top Overview Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
            <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-3.5">
              <span className="text-[11px] font-medium text-slate-400 uppercase">Overall Status</span>
              <div className="mt-1 flex items-center gap-2">
                {taskState.status === "completed" ? (
                  <CheckCircle2 className="h-4 w-4 text-emerald-400" />
                ) : (
                  <XCircle className="h-4 w-4 text-rose-400" />
                )}
                <span className="text-sm font-bold text-white capitalize">{taskState.status}</span>
              </div>
            </div>

            <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-3.5">
              <span className="text-[11px] font-medium text-slate-400 uppercase">Execution Latency</span>
              <div className="mt-1 flex items-center gap-2 text-indigo-400 font-mono text-sm font-bold">
                <Clock className="h-4 w-4" />
                <span>{Math.round(taskState.metrics?.total_execution_time_ms || 0)} ms</span>
              </div>
            </div>

            <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-3.5">
              <span className="text-[11px] font-medium text-slate-400 uppercase">Tool Invocations</span>
              <div className="mt-1 flex items-center gap-2 text-blue-400 font-mono text-sm font-bold">
                <Wrench className="h-4 w-4" />
                <span>{taskState.metrics?.tool_calls_count || 0} calls</span>
              </div>
            </div>

            <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-3.5">
              <span className="text-[11px] font-medium text-slate-400 uppercase">Privacy Boundary</span>
              <div className="mt-1 flex items-center gap-2 text-emerald-400 text-sm font-bold uppercase font-mono">
                <ShieldCheck className="h-4 w-4" />
                <span>{taskState.privacy_mode}</span>
              </div>
            </div>
          </div>

          {/* Master Detail Split: Steps List & Step Inspector */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Step Selection List */}
            <div className="space-y-2">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                DAG Steps ({taskState.plan?.steps.length || 0})
              </span>
              <div className="space-y-1.5">
                {taskState.plan?.steps.map((step) => {
                  const isSelected = selectedStep?.step_id === step.step_id;
                  return (
                    <button
                      key={step.step_id}
                      onClick={() => setSelectedStep(step)}
                      className={`w-full text-left rounded-lg p-3 transition-all border flex items-center justify-between ${
                        isSelected
                          ? "border-indigo-500 bg-indigo-950/30 text-white"
                          : "border-slate-800 bg-slate-900/40 text-slate-300 hover:bg-slate-800/40"
                      }`}
                    >
                      <div>
                        <div className="flex items-center gap-2">
                          <span className="font-mono text-[10px] text-slate-400">{step.step_id}</span>
                          <span className="text-xs font-semibold truncate max-w-[180px]">{step.title}</span>
                        </div>
                        <div className="text-[11px] text-slate-400 font-mono mt-0.5">
                          tool: {step.tool_name}
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-[11px] font-mono text-slate-400">{Math.round(step.duration_ms)}ms</span>
                        <ChevronRight className="h-4 w-4 text-slate-500" />
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Selected Step Deep Inspector */}
            <div className="lg:col-span-2 rounded-xl border border-slate-800 bg-slate-900/50 p-5 space-y-4">
              {!selectedStep ? (
                <div className="text-slate-500 text-sm">Select a step to inspect its verifiable payload.</div>
              ) : (
                <div className="space-y-4">
                  <div className="border-b border-slate-800 pb-3 flex items-center justify-between">
                    <div>
                      <h3 className="text-base font-bold text-white">{selectedStep.title}</h3>
                      <p className="text-xs text-slate-400 mt-0.5">{selectedStep.description}</p>
                    </div>
                    <span className="rounded bg-emerald-500/10 border border-emerald-500/20 px-2 py-0.5 text-xs font-semibold text-emerald-400">
                      {selectedStep.status.toUpperCase()}
                    </span>
                  </div>

                  {/* Step Metadata Grid */}
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                    <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800/80">
                      <span className="text-slate-500 block">Tool Name</span>
                      <span className="font-mono text-indigo-300 font-semibold">{selectedStep.tool_name}</span>
                    </div>
                    <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800/80">
                      <span className="text-slate-500 block">Duration</span>
                      <span className="font-mono text-slate-200">{Math.round(selectedStep.duration_ms)} ms</span>
                    </div>
                    <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800/80">
                      <span className="text-slate-500 block">Retries</span>
                      <span className="font-mono text-slate-200">{selectedStep.retry_count} / {selectedStep.max_retries}</span>
                    </div>
                    <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800/80">
                      <span className="text-slate-500 block">Dependencies</span>
                      <span className="font-mono text-slate-200">
                        {selectedStep.dependencies.length > 0 ? selectedStep.dependencies.join(", ") : "None"}
                      </span>
                    </div>
                  </div>

                  {/* Input Parameters JSON */}
                  <div className="space-y-1.5">
                    <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                      Validated Input Arguments
                    </span>
                    <pre className="rounded-lg bg-slate-950 p-3 font-mono text-[11px] text-slate-300 overflow-x-auto border border-slate-800">
                      {JSON.stringify(selectedStep.parameters, null, 2)}
                    </pre>
                  </div>

                  {/* Output Result Payload */}
                  <div className="space-y-1.5">
                    <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                      Verified Tool Output Payload
                    </span>
                    <pre className="rounded-lg bg-slate-950 p-3 font-mono text-[11px] text-slate-300 overflow-x-auto max-h-60 border border-slate-800">
                      {JSON.stringify(selectedStep.output, null, 2)}
                    </pre>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default function ExecutionInspector() {
  return (
    <Suspense fallback={<div className="p-12 text-center text-slate-400">Loading Execution Inspector...</div>}>
      <InspectorContent />
    </Suspense>
  );
}
