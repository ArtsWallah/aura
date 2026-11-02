"use client";

import { useEffect, useState, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { SearchCode, Clock, ShieldCheck } from "lucide-react";
import { fetchTasks, fetchTask } from "@/lib/api";
import { TaskState, TaskListItem } from "@/lib/types";

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
    <div className="space-y-6 max-w-7xl mx-auto">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
            <SearchCode className="h-6 w-6 text-indigo-400" />
            Execution Status Inspector
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Review privacy-safe task status. Prompts, paths, tool arguments, and raw outputs are not exposed by this view.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <label className="text-xs text-slate-400 font-medium">Select Task:</label>
          <select
            value={selectedTaskId}
            onChange={(e) => setSelectedTaskId(e.target.value)}
            className="rounded-lg border border-slate-800 bg-slate-900 px-3 py-1.5 text-xs text-slate-200"
          >
            {tasks.map((task) => (
              <option key={task.task_id} value={task.task_id}>
                {task.task_id.slice(0, 16)}... ({task.status})
              </option>
            ))}
          </select>
        </div>
      </div>

      {!taskState ? (
        <div className="p-12 text-center text-slate-400 text-sm">No task selected.</div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-4">
            <span className="text-[11px] font-medium text-slate-400 uppercase">Task ID</span>
            <div className="mt-1 text-sm font-mono text-white break-all">{taskState.task_id}</div>
          </div>
          <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-4">
            <span className="text-[11px] font-medium text-slate-400 uppercase">Status</span>
            <div className="mt-1 text-sm font-bold text-white capitalize">{taskState.status}</div>
          </div>
          <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-4">
            <span className="text-[11px] font-medium text-slate-400 uppercase">Privacy Mode</span>
            <div className="mt-1 flex items-center gap-2 text-emerald-400 text-sm font-bold uppercase font-mono">
              <ShieldCheck className="h-4 w-4" /> {taskState.privacy_mode}
            </div>
          </div>
          <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-4 sm:col-span-3">
            <span className="text-[11px] font-medium text-slate-400 uppercase">Last Updated</span>
            <div className="mt-1 flex items-center gap-2 text-indigo-300 font-mono text-sm">
              <Clock className="h-4 w-4" /> {new Date(taskState.updated_at * 1000).toLocaleString()}
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
