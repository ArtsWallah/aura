"use client";

import { useEffect, useState, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { SearchCode, Clock, ShieldCheck } from "lucide-react";
import { fetchTasks, fetchTask } from "@/lib/api";
import { TaskState, TaskListItem } from "@/lib/types";
import ExecutionTracePanel from "@/components/ExecutionTracePanel";

function InspectorContent() {
  const searchParams = useSearchParams();
  const initialTaskId = searchParams.get("task_id");
  const [tasks, setTasks] = useState<TaskListItem[]>([]);
  const [selectedTaskId, setSelectedTaskId] = useState(initialTaskId || "");
  const [taskState, setTaskState] = useState<TaskState | null>(null);

  useEffect(() => {
    fetchTasks(20).then((list) => {
      setTasks(list);
      if (!selectedTaskId && list.length > 0) setSelectedTaskId(list[0].task_id);
    });
  }, [selectedTaskId]);

  useEffect(() => {
    if (selectedTaskId) fetchTask(selectedTaskId).then(setTaskState);
  }, [selectedTaskId]);

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-black/10 pb-5">
        <div>
          <div className="inline-flex items-center gap-2 rounded-full border border-black/10 bg-black/[0.04] px-3 py-0.5 text-[10px] font-mono uppercase tracking-[0.2em] text-zinc-700">
            <span className="h-1.5 w-1.5 rounded-full bg-black animate-pulse" />
            Audit Enclave
          </div>
          <h1 className="text-2xl md:text-3xl font-black font-mono tracking-tight text-black mt-1.5 flex items-center gap-2.5">
            <SearchCode className="h-6 w-6 text-black" />
            EXECUTION STATUS INSPECTOR
          </h1>
          <p className="text-xs text-zinc-600 font-sans mt-0.5">
            Review privacy-safe task telemetry. Prompts, paths, tool arguments, and raw outputs are strictly protected.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <label className="text-xs font-mono uppercase text-zinc-600 font-bold">Select Task:</label>
          <select
            value={selectedTaskId}
            onChange={(e) => setSelectedTaskId(e.target.value)}
            className="rounded-xl border border-black/15 bg-white px-4 py-2 text-xs font-mono text-black focus:border-black focus:outline-none transition-all shadow-2xs"
          >
            {tasks.map((task) => (
              <option key={task.task_id} value={task.task_id} className="bg-white text-black">
                {task.task_id.slice(0, 16)}... ({task.status.toUpperCase()})
              </option>
            ))}
          </select>
        </div>
      </div>

      {!taskState ? (
        <div className="rounded-2xl border border-dashed border-black/15 bg-zinc-50/50 p-16 text-center text-zinc-500 font-mono text-xs">
          NO TASK SELECTED FOR INSPECTION.
        </div>
      ) : (
        <div className="space-y-6">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 font-mono">
            <div className="rounded-2xl border border-black/10 bg-white/90 p-5 glass-panel transition-all hover:border-black/30 hover:shadow-md">
              <span className="text-[10px] uppercase tracking-[0.2em] text-zinc-500 font-bold">Task ID</span>
              <div className="mt-2 text-sm text-black font-black break-all">{taskState.task_id}</div>
            </div>
            <div className="rounded-2xl border border-black/10 bg-white/90 p-5 glass-panel transition-all hover:border-black/30 hover:shadow-md">
              <span className="text-[10px] uppercase tracking-[0.2em] text-zinc-500 font-bold">Status</span>
              <div className="mt-2">
                <span className="inline-block rounded-md bg-black text-white px-2.5 py-0.5 text-xs font-black uppercase tracking-wider shadow-sm">
                  {taskState.status}
                </span>
              </div>
            </div>
            <div className="rounded-2xl border border-black/10 bg-white/90 p-5 glass-panel transition-all hover:border-black/30 hover:shadow-md">
              <span className="text-[10px] uppercase tracking-[0.2em] text-zinc-500 font-bold">Privacy Mode</span>
              <div className="mt-2 flex items-center gap-2 text-black text-sm font-bold uppercase">
                <ShieldCheck className="h-4 w-4 text-black" />
                <span>{taskState.privacy_mode}</span>
              </div>
            </div>
            <div className="rounded-2xl border border-black/10 bg-white/90 p-5 glass-panel sm:col-span-3 transition-all hover:border-black/30 hover:shadow-md">
              <span className="text-[10px] uppercase tracking-[0.2em] text-zinc-500 font-bold">Timestamp & Latency</span>
              <div className="mt-2 flex items-center gap-2 text-zinc-700 text-xs font-medium">
                <Clock className="h-4 w-4 text-black" />
                <span>LAST MODIFIED: {new Date(taskState.updated_at * 1000).toUTCString()}</span>
              </div>
            </div>
          </div>
          <ExecutionTracePanel trace={taskState.execution_trace || null} taskStatus={taskState.status} />
        </div>
      )}
    </div>
  );
}

export default function ExecutionInspector() {
  return (
    <Suspense fallback={<div className="p-16 text-center text-zinc-500 font-mono text-xs">LOADING INSPECTOR...</div>}>
      <InspectorContent />
    </Suspense>
  );
}
