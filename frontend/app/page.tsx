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
  Activity,
  FileSpreadsheet
} from "lucide-react";
import { fetchTasks, fetchHealth, fetchTools, fetchModels } from "@/lib/api";
import { useStartup } from "@/components/StartupLoader";

export default function Dashboard() {
  const { markStartupReady } = useStartup();
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
      markStartupReady();
    });
  }, [markStartupReady]);

  const totalTasks = tasks.length;
  const completedTasks = tasks.filter((t) => t.status === "completed").length;
  const failedTasks = tasks.filter((t) => t.status === "failed").length;
  const activeTasks = tasks.filter((t) => ["executing", "planning", "pending"].includes(t.status)).length;
  const successRate = totalTasks > 0 ? Math.round((completedTasks / totalTasks) * 100) : 100;

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      {/* Header Banner */}
      <div className="relative overflow-hidden rounded-3xl border border-black/10 bg-white/90 p-6 md:p-8 backdrop-blur-2xl glass-panel shadow-sm">
        <div className="absolute -top-24 -right-24 h-48 w-48 bg-black/[0.03] rounded-full blur-3xl pointer-events-none" />
        <div className="relative z-10 flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div className="space-y-1.5">
            <div className="inline-flex items-center gap-2 rounded-full border border-black/15 bg-black/[0.04] px-3 py-0.5 text-[10px] font-mono uppercase tracking-[0.2em] text-zinc-700 font-semibold">
              <span className="h-1.5 w-1.5 rounded-full bg-black animate-pulse" />
              Autonomous Engine Active
            </div>
            <h1 className="text-3xl md:text-4xl font-black tracking-tight text-black font-mono">
              OPERATIONAL COMMAND
            </h1>
            <p className="text-sm text-zinc-600 font-sans max-w-2xl leading-relaxed">
              Real-time telemetry, model routing, and deterministic agent execution metrics across verified local enclaves.
            </p>
          </div>
          <div className="flex items-center gap-3">
            <Link
              href="/workspace"
              className="group relative flex items-center gap-2.5 rounded-2xl bg-black px-6 py-3.5 text-xs font-mono font-bold text-white shadow-[0_4px_20px_rgba(0,0,0,0.18)] transition-all duration-300 hover:scale-[1.02] hover:bg-zinc-800 active:scale-[0.98]"
            >
              <Bot className="h-4 w-4 text-white transition-transform duration-300 group-hover:rotate-12" />
              <span className="tracking-wider">LAUNCH WORKSPACE</span>
              <ArrowRight className="h-4 w-4 ml-0.5 transition-transform duration-300 group-hover:translate-x-1" />
            </Link>
          </div>
        </div>
      </div>

      {/* KPI Cards (Classic High-Contrast B&W) */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
        {/* Total Tasks */}
        <div className="glass-panel glass-panel-hover rounded-2xl p-5 flex flex-col justify-between border-black/10 bg-white/90 shadow-xs">
          <div className="flex items-center justify-between text-zinc-500">
            <span className="text-[10px] font-mono uppercase tracking-[0.2em] font-bold">Total Tasks</span>
            <Clock className="h-4 w-4 text-zinc-700" />
          </div>
          <div className="mt-4">
            <div className="text-3xl font-mono font-black text-black tracking-tight">{totalTasks}</div>
            <span className="text-[11px] text-zinc-500 font-sans">Autonomous workflows</span>
          </div>
        </div>

        {/* Completed */}
        <div className="glass-panel glass-panel-hover rounded-2xl p-5 flex flex-col justify-between border-black/20 bg-white/90 shadow-xs">
          <div className="flex items-center justify-between text-black">
            <span className="text-[10px] font-mono uppercase tracking-[0.2em] font-bold">Completed</span>
            <CheckCircle2 className="h-4 w-4 text-black" />
          </div>
          <div className="mt-4">
            <div className="text-3xl font-mono font-black text-black tracking-tight">{completedTasks}</div>
            <span className="text-[11px] text-zinc-700 font-mono font-bold tracking-wide">{successRate}% SUCCESS RATE</span>
          </div>
        </div>

        {/* Failed */}
        <div className="glass-panel glass-panel-hover rounded-2xl p-5 flex flex-col justify-between border-black/10 bg-white/90 shadow-xs">
          <div className="flex items-center justify-between text-zinc-500">
            <span className="text-[10px] font-mono uppercase tracking-[0.2em] font-bold">Halted</span>
            <XCircle className="h-4 w-4 text-zinc-500" />
          </div>
          <div className="mt-4">
            <div className="text-3xl font-mono font-black text-zinc-700 tracking-tight">{failedTasks}</div>
            <span className="text-[11px] text-zinc-500 font-sans">Self-correction stops</span>
          </div>
        </div>

        {/* Tools Count */}
        <div className="glass-panel glass-panel-hover rounded-2xl p-5 flex flex-col justify-between border-black/10 bg-white/90 shadow-xs">
          <div className="flex items-center justify-between text-zinc-500">
            <span className="text-[10px] font-mono uppercase tracking-[0.2em] font-bold">Typed Tools</span>
            <Wrench className="h-4 w-4 text-zinc-700" />
          </div>
          <div className="mt-4">
            <div className="text-3xl font-mono font-black text-black tracking-tight">{toolsCount || 11}</div>
            <span className="text-[11px] text-zinc-500 font-sans">Deterministic engines</span>
          </div>
        </div>

        {/* Models Count */}
        <div className="glass-panel glass-panel-hover rounded-2xl p-5 flex flex-col justify-between border-black/10 bg-white/90 shadow-xs">
          <div className="flex items-center justify-between text-zinc-500">
            <span className="text-[10px] font-mono uppercase tracking-[0.2em] font-bold">AI Enclaves</span>
            <Cpu className="h-4 w-4 text-zinc-700" />
          </div>
          <div className="mt-4">
            <div className="text-3xl font-mono font-black text-black tracking-tight">{modelsCount || 2} <span className="text-xs font-normal text-zinc-500 font-mono">+ WebGPU</span></div>
            <span className="text-[11px] text-zinc-500 font-sans">Local & client inference</span>
          </div>
        </div>
      </div>

      {/* Main Grid: Recent Tasks & Quick Actions */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Recent Executions */}
        <div className="lg:col-span-2 rounded-3xl border border-black/10 bg-white/90 p-6 space-y-4 backdrop-blur-xl glass-panel shadow-sm">
          <div className="flex items-center justify-between border-b border-black/10 pb-4">
            <h2 className="text-xs font-mono uppercase tracking-[0.2em] font-bold text-black flex items-center gap-2.5">
              <Activity className="h-4 w-4 text-black" />
              Recent Task History
            </h2>
            <Link
              href="/workspace"
              className="text-xs font-mono uppercase tracking-wider text-zinc-600 hover:text-black font-semibold transition-colors"
            >
              View Workspace &rarr;
            </Link>
          </div>

          {tasks.length === 0 ? (
            <div className="rounded-2xl border border-dashed border-black/15 p-12 text-center space-y-3 bg-zinc-50/50">
              <p className="text-sm text-zinc-500 font-sans">No execution history recorded in active store.</p>
              <Link
                href="/workspace"
                className="inline-flex items-center gap-1.5 text-xs font-mono font-bold text-black underline underline-offset-4 hover:text-zinc-700 transition-colors"
              >
                Launch your first autonomous analytics task &rarr;
              </Link>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs text-zinc-800 font-sans">
                <thead className="border-b border-black/10 text-[10px] uppercase font-mono tracking-[0.2em] text-zinc-500 bg-zinc-50/70">
                  <tr>
                    <th className="py-3 px-3">Task Identifier</th>
                    <th className="py-3 px-3">Analytical Objective</th>
                    <th className="py-3 px-3">Privacy</th>
                    <th className="py-3 px-3">Status</th>
                    <th className="py-3 px-3 text-right">Inspect</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-black/5">
                  {tasks.map((task) => (
                    <tr key={task.task_id} className="hover:bg-zinc-50 transition-colors group">
                      <td className="py-3.5 px-3 font-mono text-[11px] text-zinc-500 group-hover:text-black font-medium">
                        {task.task_id.slice(0, 16)}...
                      </td>
                      <td className="py-3.5 px-3 font-medium text-black max-w-xs truncate">
                        {task.objective}
                      </td>
                      <td className="py-3.5 px-3">
                        <span className="rounded-md border border-black/10 bg-zinc-100 px-2 py-0.5 text-[10px] font-mono text-zinc-800 font-semibold">
                          {task.privacy_mode}
                        </span>
                      </td>
                      <td className="py-3.5 px-3">
                        <span
                          className={`rounded-md px-2.5 py-0.5 text-[10px] font-mono font-bold tracking-wider ${
                            task.status === "completed"
                              ? "bg-black text-white shadow-xs"
                              : task.status === "failed"
                              ? "border border-zinc-300 bg-zinc-100 text-zinc-600"
                              : "border border-black bg-black/10 text-black animate-pulse"
                          }`}
                        >
                          {task.status.toUpperCase()}
                        </span>
                      </td>
                      <td className="py-3.5 px-3 text-right">
                        <Link
                          href={`/inspector?task_id=${task.task_id}`}
                          className="font-mono text-[11px] text-black hover:underline underline-offset-4 font-bold"
                        >
                          INSPECT &rarr;
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
          {/* Acceptance Benchmark Card */}
          <div className="relative overflow-hidden rounded-3xl border border-black/15 bg-white/90 p-6 space-y-4 backdrop-blur-xl glass-panel shadow-sm">
            <div className="flex items-center gap-2 text-black font-mono text-xs uppercase tracking-wider font-bold">
              <FileSpreadsheet className="h-4 w-4 text-black" />
              <span>Standard Acceptance Benchmark</span>
            </div>
            <p className="text-xs text-zinc-600 leading-relaxed font-sans">
              Run the full autonomous workflow: inspect sales CSV, compute group metrics, render Matplotlib charts, detect underperforming product lines, and synthesize the report.
            </p>
            <Link
              href="/workspace?preset=sales"
              className="group flex w-full items-center justify-center gap-2 rounded-2xl bg-black px-4 py-3 text-xs font-mono font-bold text-white hover:bg-zinc-800 transition-all shadow-[0_4px_16px_rgba(0,0,0,0.18)] hover:scale-[1.02] active:scale-[0.98]"
            >
              <span>RUN SALES CSV BENCHMARK</span>
              <ArrowRight className="h-3.5 w-3.5 transition-transform duration-200 group-hover:translate-x-1" />
            </Link>
          </div>

          {/* Privacy Architecture Notice */}
          <div className="rounded-3xl border border-black/10 bg-white/90 p-6 space-y-3.5 backdrop-blur-xl glass-panel shadow-sm">
            <h3 className="text-xs font-mono uppercase tracking-[0.2em] font-bold text-black flex items-center gap-2">
              <ShieldCheck className="h-4 w-4 text-black" />
              Security & Privacy Enclaves
            </h3>
            <div className="space-y-2 text-xs font-sans text-zinc-700">
              <div className="p-3 rounded-xl bg-zinc-50 border border-black/10 transition-colors hover:border-black/20">
                <span className="font-mono font-bold text-black">LOCAL ONLY:</span> Zero external calls. 100% on-device analytics, encryption, and deterministic execution.
              </div>
              <div className="p-3 rounded-xl bg-zinc-50 border border-black/10 transition-colors hover:border-black/20">
                <span className="font-mono font-bold text-black">HYBRID:</span> Proprietary row data remains encrypted locally; abstract schemas sent to planner.
              </div>
              <div className="p-3 rounded-xl bg-zinc-50 border border-black/10 transition-colors hover:border-black/20">
                <span className="font-mono font-bold text-black">WEBGPU:</span> Client-side browser neural classification, embeddings, and similarity search.
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
