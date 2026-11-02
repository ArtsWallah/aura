"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { 
  Bot, 
  CheckCircle2, 
  XCircle, 
  Clock, 
  Wrench, 
  Cpu, 
  ShieldCheck, 
  ArrowRight,
  TrendingUp,
  FileSpreadsheet
} from "lucide-react";
import { fetchTasks, fetchHealth, fetchTools, fetchModels } from "@/lib/api";

export default function Dashboard() {
  const [tasks, setTasks] = useState<any[]>([]);
  const [health, setHealth] = useState<any>(null);
  const [toolsCount, setToolsCount] = useState<number>(0);
  const [modelsCount, setModelsCount] = useState<number>(0);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    Promise.all([
      fetchTasks(10).catch(() => []),
      fetchHealth().catch(() => null),
      fetchTools().then((t) => t.length).catch(() => 0),
      fetchModels().then((m) => m.length).catch(() => 0),
    ]).then(([tasksData, healthData, tCount, mCount]) => {
      setTasks(tasksData);
      setHealth(healthData);
      setToolsCount(tCount);
      setModelsCount(mCount);
      setLoading(false);
    });
  }, []);

  const totalTasks = tasks.length;
  const completedTasks = tasks.filter((t) => t.status === "completed").length;
  const failedTasks = tasks.filter((t) => t.status === "failed").length;
  const activeTasks = tasks.filter((t) => ["executing", "planning", "pending"].includes(t.status)).length;
  const successRate = totalTasks > 0 ? Math.round((completedTasks / totalTasks) * 100) : 100;

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-white">
            Operational Dashboard
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Real-time telemetry, model routing, and deterministic agent execution metrics.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Link
            href="/workspace"
            className="flex items-center gap-2 rounded-lg bg-indigo-600 px-4 py-2.5 text-sm font-semibold text-white shadow-lg shadow-indigo-600/30 hover:bg-indigo-500 transition-all"
          >
            <Bot className="h-4 w-4" />
            <span>Launch Agent Workspace</span>
            <ArrowRight className="h-4 w-4 ml-1" />
          </Link>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
          <div className="flex items-center justify-between text-slate-400">
            <span className="text-xs font-medium uppercase tracking-wider">Total Tasks</span>
            <Clock className="h-4 w-4 text-slate-400" />
          </div>
          <div className="mt-2 text-2xl font-bold text-white">{totalTasks}</div>
          <span className="text-[11px] text-slate-500">Autonomous workflow runs</span>
        </div>

        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
          <div className="flex items-center justify-between text-emerald-400">
            <span className="text-xs font-medium uppercase tracking-wider">Completed</span>
            <CheckCircle2 className="h-4 w-4 text-emerald-400" />
          </div>
          <div className="mt-2 text-2xl font-bold text-white">{completedTasks}</div>
          <span className="text-[11px] text-emerald-400/80">{successRate}% Success Rate</span>
        </div>

        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
          <div className="flex items-center justify-between text-rose-400">
            <span className="text-xs font-medium uppercase tracking-wider">Failed</span>
            <XCircle className="h-4 w-4 text-rose-400" />
          </div>
          <div className="mt-2 text-2xl font-bold text-white">{failedTasks}</div>
          <span className="text-[11px] text-slate-500">Self-correction halts</span>
        </div>

        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
          <div className="flex items-center justify-between text-blue-400">
            <span className="text-xs font-medium uppercase tracking-wider">Registered Tools</span>
            <Wrench className="h-4 w-4 text-blue-400" />
          </div>
          <div className="mt-2 text-2xl font-bold text-white">{toolsCount || 10}</div>
          <span className="text-[11px] text-slate-500">Deterministic engines</span>
        </div>

        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
          <div className="flex items-center justify-between text-purple-400">
            <span className="text-xs font-medium uppercase tracking-wider">Small AI Models</span>
            <Cpu className="h-4 w-4 text-purple-400" />
          </div>
          <div className="mt-2 text-2xl font-bold text-white">{modelsCount || 2} + WebGPU</div>
          <span className="text-[11px] text-slate-500">Edge & browser inference</span>
        </div>
      </div>

      {/* Main Grid: Recent Tasks & Quick Actions */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Recent Executions */}
        <div className="lg:col-span-2 rounded-xl border border-slate-800 bg-slate-900/40 p-5 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-base font-semibold text-white flex items-center gap-2">
              <TrendingUp className="h-4 w-4 text-indigo-400" />
              Recent Task History
            </h2>
            <Link href="/workspace" className="text-xs text-indigo-400 hover:text-indigo-300">
              View all &rarr;
            </Link>
          </div>

          {tasks.length === 0 ? (
            <div className="rounded-lg border border-dashed border-slate-800 p-8 text-center">
              <p className="text-sm text-slate-400">No execution history recorded yet.</p>
              <Link
                href="/workspace"
                className="mt-3 inline-flex items-center text-xs text-indigo-400 hover:underline"
              >
                Launch your first autonomous analytics task
              </Link>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs text-slate-300">
                <thead className="border-b border-slate-800 text-[11px] uppercase tracking-wider text-slate-400 bg-slate-900/60">
                  <tr>
                    <th className="py-2.5 px-3">Task ID</th>
                    <th className="py-2.5 px-3">Objective</th>
                    <th className="py-2.5 px-3">Privacy Mode</th>
                    <th className="py-2.5 px-3">Status</th>
                    <th className="py-2.5 px-3">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  {tasks.map((task) => (
                    <tr key={task.task_id} className="hover:bg-slate-800/30 transition-colors">
                      <td className="py-3 px-3 font-mono text-[11px] text-slate-400">
                        {task.task_id.slice(0, 14)}...
                      </td>
                      <td className="py-3 px-3 font-medium text-slate-200 max-w-xs truncate">
                        {task.objective}
                      </td>
                      <td className="py-3 px-3">
                        <span className="rounded bg-slate-800 px-2 py-0.5 text-[10px] font-mono text-slate-300">
                          {task.privacy_mode}
                        </span>
                      </td>
                      <td className="py-3 px-3">
                        <span
                          className={`rounded px-2 py-0.5 text-[10px] font-semibold ${
                            task.status === "completed"
                              ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
                              : task.status === "failed"
                              ? "bg-rose-500/10 text-rose-400 border border-rose-500/20"
                              : "bg-blue-500/10 text-blue-400 border border-blue-500/20"
                          }`}
                        >
                          {task.status.toUpperCase()}
                        </span>
                      </td>
                      <td className="py-3 px-3">
                        <Link
                          href={`/inspector?task_id=${task.task_id}`}
                          className="text-indigo-400 hover:text-indigo-300 font-medium"
                        >
                          Inspect &rarr;
                        </Link>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Right: Quick Launch & System Info */}
        <div className="space-y-6">
          {/* Quick Benchmark Prompt */}
          <div className="rounded-xl border border-indigo-500/20 bg-gradient-to-br from-indigo-950/40 via-slate-900/60 to-slate-950 p-5 space-y-3">
            <div className="flex items-center gap-2 text-indigo-400 font-semibold text-sm">
              <FileSpreadsheet className="h-4 w-4" />
              <span>Standard Acceptance Benchmark</span>
            </div>
            <p className="text-xs text-slate-300 leading-relaxed">
              Run the full autonomous workflow: inspect sales CSV, compute group metrics, render Matplotlib charts, detect underperforming product lines, and synthesize the report.
            </p>
            <Link
              href="/workspace?preset=sales"
              className="block w-full text-center rounded-lg bg-indigo-600 px-3 py-2 text-xs font-semibold text-white hover:bg-indigo-500 transition-colors shadow-md shadow-indigo-600/20"
            >
              Run Sales CSV Analysis Benchmark &rarr;
            </Link>
          </div>

          {/* Privacy Architecture Notice */}
          <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5 space-y-3">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <ShieldCheck className="h-4 w-4 text-emerald-400" />
              Privacy Modes & Security
            </h3>
            <div className="space-y-2 text-xs text-slate-400">
              <div className="p-2 rounded bg-slate-950/60 border border-slate-800/80">
                <span className="font-semibold text-emerald-400">Local Only:</span> Zero cloud calls. 100% on-device analytics & planning.
              </div>
              <div className="p-2 rounded bg-slate-950/60 border border-slate-800/80">
                <span className="font-semibold text-blue-400">Hybrid:</span> Proprietary row data stays local; abstract statistical schema sent to cloud LLM.
              </div>
              <div className="p-2 rounded bg-slate-950/60 border border-slate-800/80">
                <span className="font-semibold text-purple-400">WebGPU:</span> In-browser client-side neural classification & embeddings.
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
