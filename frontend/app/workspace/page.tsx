"use client";

import { useState, useEffect, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { 
  Play, 
  Upload, 
  ShieldCheck, 
  Download, 
  BarChart3, 
  FileSpreadsheet,
  Terminal,
  RotateCw,
  Eye,
  CheckCircle2,
  XCircle
} from "lucide-react";
import { createTask, uploadDatasetFile, fetchTask, fetchTaskArtifacts, fetchTaskMarkdown, cancelTask, API_BASE } from "@/lib/api";
import { TaskState, TaskArtifactManifest, TaskClarificationResponse, PrivacyMode, ExecutionTrace, TraceStep } from "@/lib/types";
import ExecutionTracePanel from "@/components/ExecutionTracePanel";
import { traceFromTaskRefresh } from "@/lib/trace-refresh.mjs";

function WorkspaceContent() {
  const searchParams = useSearchParams();
  const preset = searchParams.get("preset");

  const [objective, setObjective] = useState<string>(
    "Analyze this dataset, identify sales trends, detect underperforming categories, and generate a report with charts."
  );
  const [filePath, setFilePath] = useState<string>("data/sample_sales.csv");
  const [privacyMode, setPrivacyMode] = useState<PrivacyMode>("hybrid");
  const [uploading, setUploading] = useState<boolean>(false);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
  const [cancellationPending, setCancellationPending] = useState<boolean>(false);

  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [currentTask, setCurrentTask] = useState<TaskState | null>(null);
  const [executionTrace, setExecutionTrace] = useState<ExecutionTrace | null>(null);
  const [clarification, setClarification] = useState<TaskClarificationResponse | null>(null);
  const [taskArtifacts, setTaskArtifacts] = useState<TaskArtifactManifest | null>(null);
  const [markdownReport, setMarkdownReport] = useState<string | null>(null);
  const [events, setEvents] = useState<Array<{ timestamp: string; message: string; type: string }>>([]);
  const [activeTab, setActiveTab] = useState<"plan" | "logs" | "report">("plan");

  useEffect(() => {
    if (preset === "sales") {
      setObjective("Analyze this dataset, identify sales trends, detect underperforming categories, and generate a report with charts.");
      setFilePath("data/sample_sales.csv");
    }
  }, [preset]);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    const file = e.target.files[0];
    setUploading(true);
    setUploadStatus("Uploading & validating file...");

    try {
      const res = await uploadDatasetFile(file, privacyMode === "local_only");
      setFilePath(res.file_path);
      setUploadStatus(`Verified: ${res.file_name} (${Math.round(res.size_bytes / 1024)} KB)`);
    } catch (err: unknown) {
      setUploadStatus(`Upload failed: ${err instanceof Error ? err.message : "Unknown error."}`);
    } finally {
      setUploading(false);
    }
  };

  const handleRunTask = async () => {
    if (!objective.trim()) return;

    setIsRunning(true);
    setEvents([]);
    setCurrentTask(null);
    setExecutionTrace(null);
    setClarification(null);
    setTaskArtifacts(null);
    setMarkdownReport(null);
    setActiveTab("plan");

    try {
      const submission = await createTask(objective, filePath, privacyMode);
      if ("outcome" in submission) {
        setClarification(submission);
        setIsRunning(false);
        return;
      }
      const initialTask = submission;
      setCurrentTask(initialTask);
      setExecutionTrace(initialTask.execution_trace || null);

      const applyTraceUpdate = (data: { trace_snapshot?: ExecutionTrace; trace_step?: TraceStep }) => {
        if (data.trace_snapshot) {
          setExecutionTrace(data.trace_snapshot);
        } else if (data.trace_step) {
          setExecutionTrace((trace) => trace ? {
            ...trace,
            step_statuses: trace.step_statuses.map((step) =>
              step.step_id === data.trace_step!.step_id ? data.trace_step! : step
            ),
          } : null);
        }
      };

      const syncTaskDetail = async (taskId: string) => {
        try {
          const detail = await fetchTask(taskId);
          setCurrentTask(detail);
          setExecutionTrace(traceFromTaskRefresh(detail));
        } catch {
          // ignore transient polling errors
        }
      };

      const es = new EventSource(`${API_BASE}/api/tasks/${initialTask.task_id}/events`);

      es.onmessage = async (e) => {
        try {
          const data = JSON.parse(e.data);
          setEvents((prev) => [
            ...prev,
            {
              timestamp: new Date().toLocaleTimeString(),
              message: data.message || JSON.stringify(data),
              type: data.type || "info",
            },
          ]);

          applyTraceUpdate(data);

          if (data.type === "step_completed" || data.type === "step_started" || data.type === "step_failed") {
            await syncTaskDetail(initialTask.task_id);
          }

          if (data.type === "task_completed") {
            es.close();
            setIsRunning(false);
            await syncTaskDetail(initialTask.task_id);

            const artifacts = await fetchTaskArtifacts(initialTask.task_id);
            setTaskArtifacts(artifacts);

            if (artifacts.markdown_url) {
              const md = await fetchTaskMarkdown(initialTask.task_id);
              setMarkdownReport(md.content);
            }
            setActiveTab("report");
          } else if (data.type === "task_failed" || data.type === "task_cancelled") {
            es.close();
            setIsRunning(false);
            await syncTaskDetail(initialTask.task_id);
          }
        } catch (parseErr) {
          console.error("SSE parse error", parseErr);
        }
      };

      es.onerror = () => {
        es.close();
        setIsRunning(false);
        syncTaskDetail(initialTask.task_id);
      };

    } catch (err: unknown) {
      setIsRunning(false);
      setEvents((prev) => [
        ...prev,
        {
          timestamp: new Date().toLocaleTimeString(),
          message: `Initialization failed: ${err instanceof Error ? err.message : "Unknown error."}`,
          type: "error",
        },
      ]);
    }
  };

  const handleCancelTask = async () => {
    if (!currentTask?.task_id || cancellationPending) return;
    setCancellationPending(true);
    try {
      const updated = await cancelTask(currentTask.task_id);
      setCurrentTask((task) => (task ? { ...task, status: updated.status } : task));
      const refreshed = await fetchTask(currentTask.task_id);
      setCurrentTask(refreshed);
      setExecutionTrace(traceFromTaskRefresh(refreshed));
      setEvents((prev) => [
        ...prev,
        {
          timestamp: new Date().toLocaleTimeString(),
          message: `Cancellation requested. Current state: ${updated.status}`,
          type: "task.cancellation_requested",
        },
      ]);
    } catch (err: unknown) {
      setEvents((prev) => [
        ...prev,
        {
          timestamp: new Date().toLocaleTimeString(),
          message: `Cancellation error: ${err instanceof Error ? err.message : "Unknown error."}`,
          type: "task.cancellation_failed",
        },
      ]);
    } finally {
      setCancellationPending(false);
    }
  };

  const handleDownloadMarkdown = () => {
    if (!markdownReport) return;
    const blob = new Blob([markdownReport], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `AURA_Analytical_Report_${currentTask?.task_id || "export"}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      {/* Top Banner / Breadcrumb */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-black/10 pb-5">
        <div>
          <div className="inline-flex items-center gap-2 rounded-full border border-black/10 bg-black/[0.04] px-3 py-0.5 text-[10px] font-mono uppercase tracking-[0.2em] text-zinc-700">
            <span className="h-1.5 w-1.5 rounded-full bg-black animate-pulse" />
            Autonomous Execution Lab
          </div>
          <h1 className="text-2xl md:text-3xl font-black font-mono tracking-tight text-black mt-1.5">
            AGENT WORKSPACE
          </h1>
          <p className="text-xs text-zinc-600 font-sans mt-0.5">
            Define objectives, configure privacy enclaves, and stream deterministic execution DAGs.
          </p>
        </div>

        {/* Privacy Selector Pills */}
        <div className="flex items-center gap-1.5 rounded-xl border border-black/10 bg-zinc-100/90 p-1 font-mono text-xs backdrop-blur-xl">
          {(["local_only", "hybrid", "cloud_enabled"] as PrivacyMode[]).map((mode) => (
            <button
              key={mode}
              onClick={() => setPrivacyMode(mode)}
              className={`rounded-lg px-3 py-1.5 text-xs font-semibold transition-all duration-200 ${
                privacyMode === mode
                  ? "bg-black text-white shadow-md scale-[1.02]"
                  : "text-zinc-600 hover:text-black hover:bg-white"
              }`}
            >
              {mode.replace("_", " ").toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      {/* Input Formulation Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left 2 Cols: Prompt & File */}
        <div className="lg:col-span-2 space-y-4">
          <div className="relative overflow-hidden rounded-2xl border border-black/10 bg-white/90 p-6 space-y-4 backdrop-blur-xl glass-panel shadow-sm">
            <label className="text-[10px] font-mono uppercase tracking-[0.22em] text-zinc-600 font-bold flex items-center justify-between">
              <span>High-Level Objective / Instructions</span>
              <span className="text-zinc-400 font-normal">Declarative Planner Input</span>
            </label>
            <textarea
              rows={3}
              value={objective}
              onChange={(e) => setObjective(e.target.value)}
              placeholder="e.g., Analyze this dataset, identify sales trends, detect underperforming categories, and generate a report with charts."
              className="w-full rounded-xl border border-black/15 bg-zinc-50 p-4 text-sm text-black placeholder-zinc-400 font-sans focus:border-black focus:outline-none focus:ring-1 focus:ring-black focus:bg-white transition-all leading-relaxed"
            />
            {clarification && (
              <div
                role="status"
                aria-live="polite"
                className="rounded-xl border border-black/20 bg-zinc-100 p-4 text-sm text-black font-mono space-y-1.5"
              >
                <p className="font-bold text-black uppercase tracking-wider text-xs">More Information Needed</p>
                <p className="font-sans text-xs text-zinc-700">{clarification.question}</p>
                {clarification.required_input && clarification.required_input !== "clarification" && (
                  <p className="text-[11px] text-zinc-600">
                    REQUIRED INPUT: {clarification.required_input.replaceAll("_", " ").toUpperCase()}
                  </p>
                )}
              </div>
            )}

            <div className="flex flex-wrap items-center justify-between gap-4 pt-2">
              <div className="flex items-center gap-3">
                <label className="cursor-pointer group flex items-center gap-2 rounded-xl border border-black/15 bg-white px-4 py-2 text-xs font-mono font-medium text-black hover:border-black hover:bg-zinc-50 shadow-2xs transition-all">
                  <Upload className="h-3.5 w-3.5 text-zinc-600 group-hover:text-black transition-colors" />
                  <span>UPLOAD DATASET</span>
                  <input
                    type="file"
                    accept=".csv,.xlsx,.xls,.txt,.md"
                    onChange={handleFileUpload}
                    disabled={uploading}
                    className="hidden"
                  />
                </label>
                {uploadStatus && (
                  <span className="text-[11px] font-mono text-zinc-700">
                    {uploadStatus}
                  </span>
                )}
              </div>

              <div className="flex items-center gap-3">
                {isRunning && currentTask && (
                  <button
                    onClick={handleCancelTask}
                    disabled={cancellationPending || currentTask.status === "cancel_requested" || ["completed", "failed", "cancelled"].includes(currentTask.status)}
                    className="rounded-xl border border-black/20 bg-white px-4 py-2.5 text-xs font-mono font-semibold text-black hover:bg-zinc-100 transition-all disabled:opacity-50"
                  >
                    {currentTask.status === "cancel_requested" || cancellationPending ? "STOPPING..." : "CANCEL TASK"}
                  </button>
                )}
                <button
                  onClick={handleRunTask}
                  disabled={isRunning}
                  className={`group relative flex items-center gap-2.5 rounded-xl px-6 py-2.5 text-xs font-mono font-bold tracking-wider transition-all duration-300 ${
                    isRunning
                      ? "bg-zinc-200 text-zinc-500 cursor-not-allowed border border-black/10"
                      : "bg-black text-white hover:bg-zinc-800 shadow-[0_4px_16px_rgba(0,0,0,0.18)] hover:scale-[1.01] active:scale-[0.99]"
                  }`}
                >
                  {isRunning ? (
                    <>
                      <RotateCw className="h-4 w-4 animate-spin text-zinc-400" />
                      <span>{currentTask?.status === "cancel_requested" ? "CANCELLING..." : "EXECUTING DAG..."}</span>
                    </>
                  ) : (
                    <>
                      <Play className="h-3.5 w-3.5 fill-white" />
                      <span>EXECUTE AUTONOMOUS TASK</span>
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Right 1 Col: Execution Boundary & File Target */}
        <div className="rounded-2xl border border-black/10 bg-white/90 p-6 space-y-4 backdrop-blur-xl glass-panel shadow-sm flex flex-col justify-between">
          <div className="space-y-3">
            <span className="text-[10px] font-mono uppercase tracking-[0.2em] font-bold text-zinc-500">
              Data Boundary & Target
            </span>
            <div className="rounded-xl bg-zinc-50 p-3.5 border border-black/10 font-mono text-xs text-black break-all flex items-center gap-2.5">
              <FileSpreadsheet className="h-4 w-4 shrink-0 text-zinc-600" />
              <span>{filePath || "No target attached"}</span>
            </div>

            <div className="text-[11px] text-zinc-600 font-sans space-y-2 pt-2">
              <div className="flex items-start gap-2 text-zinc-800">
                <ShieldCheck className="h-4 w-4 shrink-0 text-black mt-0.5" />
                <span className="leading-relaxed">
                  {privacyMode === "local_only"
                    ? "Local Only: 100% on-device deterministic execution in isolated enclave."
                    : privacyMode === "hybrid"
                    ? "Hybrid: User row content stays local; abstract schema sent to planner."
                    : "Cloud Enabled: Gemini cloud reasoning for unstructured multi-step analytics."}
                </span>
              </div>
            </div>
          </div>

          <div className="rounded-xl border border-black/10 bg-zinc-100/70 p-3 text-[10px] font-mono text-zinc-600 font-medium">
            ENCRYPTION: AES-128-CBC + HMAC-SHA256
          </div>
        </div>
      </div>

      {/* Tabs for Plan, Live Logs, and Final Report */}
      <div className="border-b border-black/10 flex items-center gap-6 font-mono text-xs">
        <button
          onClick={() => setActiveTab("plan")}
          className={`pb-3 font-semibold transition-all duration-200 border-b-2 tracking-wider ${
            activeTab === "plan"
              ? "border-black text-black font-bold"
              : "border-transparent text-zinc-500 hover:text-black"
          }`}
        >
          TASK STATUS & DAG
        </button>
        <button
          onClick={() => setActiveTab("logs")}
          className={`pb-3 font-semibold transition-all duration-200 border-b-2 tracking-wider ${
            activeTab === "logs"
              ? "border-black text-black font-bold"
              : "border-transparent text-zinc-500 hover:text-black"
          }`}
        >
          TELEMETRY & LOGS ({events.length})
        </button>
        <button
          onClick={() => setActiveTab("report")}
          className={`pb-3 font-semibold transition-all duration-200 border-b-2 tracking-wider ${
            activeTab === "report"
              ? "border-black text-black font-bold"
              : "border-transparent text-zinc-500 hover:text-black"
          }`}
        >
          FINAL DELIVERABLES & CHARTS
        </button>
      </div>

      {/* Tab 1: Privacy-safe task status */}
      {activeTab === "plan" && (
        <div className="space-y-4">
          <div className="rounded-2xl border border-black/10 bg-white/90 p-6 space-y-2 backdrop-blur-xl glass-panel shadow-sm font-mono text-xs">
            {!currentTask ? (
              <p className="text-zinc-500">Submit a task to stream its validated execution DAG.</p>
            ) : (
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div className="space-y-1">
                  <p className="text-zinc-600">TASK IDENTIFIER: <span className="text-black font-bold">{currentTask.task_id}</span></p>
                  <p className="text-zinc-600">CURRENT STATUS: <span className="text-black font-bold uppercase">{currentTask.status}</span></p>
                </div>
                <div className="text-[10px] text-zinc-700 font-semibold uppercase tracking-widest border border-black/10 bg-zinc-50 rounded-lg px-3 py-1">
                  Forensic Trace Verified
                </div>
              </div>
            )}
          </div>
          <ExecutionTracePanel trace={executionTrace} taskStatus={currentTask?.status || null} />
        </div>
      )}

      {/* Tab 2: Live Log Terminal */}
      {activeTab === "logs" && (
        <div className="rounded-2xl border border-black/20 bg-black p-6 font-mono text-xs space-y-3 max-h-[30rem] overflow-y-auto shadow-2xl glass-panel text-zinc-200">
          <div className="flex items-center justify-between pb-3 border-b border-zinc-800 text-zinc-400">
            <div className="flex items-center gap-2 text-white">
              <Terminal className="h-4 w-4 text-white" />
              <span className="font-bold tracking-wider">AURA SECURE EVENT STREAM</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="h-2 w-2 rounded-full bg-white animate-pulse" />
              <span className="text-[10px] text-zinc-400">BUFFER ACTIVE</span>
            </div>
          </div>
          {events.length === 0 ? (
            <p className="text-zinc-500 italic py-6 text-center">Awaiting execution runtime events...</p>
          ) : (
            <div className="space-y-2">
              {events.map((ev, i) => (
                <div key={i} className="flex items-start gap-3 text-zinc-300 font-mono text-[11px] leading-relaxed">
                  <span className="text-zinc-500 shrink-0">[{ev.timestamp}]</span>
                  <span
                    className={
                      ev.type === "step_completed" || ev.type === "task_completed"
                        ? "text-white font-bold"
                        : ev.type === "step_failed" || ev.type === "task_failed"
                        ? "text-zinc-400 line-through"
                        : ev.type === "step_started"
                        ? "text-zinc-200"
                        : "text-zinc-400"
                    }
                  >
                    {ev.message}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Tab 3: Final Deliverable & Embedded Charts */}
      {activeTab === "report" && (
        <div className="space-y-6">
          {!taskArtifacts ? (
            <div className="rounded-2xl border border-dashed border-black/15 bg-zinc-50/50 p-16 text-center text-zinc-500 text-xs font-mono space-y-2 glass-panel">
              <p className="font-bold">NO DELIVERABLE COMPILED YET.</p>
              <p className="text-zinc-500 font-sans">Execute a workflow to generate encrypted analytical reports and visualizations.</p>
            </div>
          ) : (
            <div className="space-y-6">
              {/* Header Actions */}
              <div className="flex flex-wrap items-center justify-between gap-4 bg-white/95 p-6 rounded-2xl border border-black/10 backdrop-blur-xl glass-panel shadow-sm">
                <div>
                  <h2 className="text-lg font-bold font-mono tracking-tight text-black uppercase">
                    {taskArtifacts.title || "Analytical Deliverable"}
                  </h2>
                  <p className="text-xs text-zinc-600 font-sans mt-0.5">
                    Synthesized with verified Python calculations and cryptographically authenticated charts.
                  </p>
                </div>
                <div className="flex items-center gap-3">
                  {taskArtifacts.html_url && (
                    <a
                      href={`${API_BASE}${taskArtifacts.html_url}`}
                      target="_blank"
                      rel="noreferrer"
                      className="group flex items-center gap-2 rounded-xl bg-black px-4 py-2 text-xs font-mono font-bold text-white hover:bg-zinc-800 transition-all shadow-[0_4px_16px_rgba(0,0,0,0.18)] hover:scale-[1.02]"
                    >
                      <Eye className="h-3.5 w-3.5 text-white" />
                      <span>VIEW STANDALONE HTML</span>
                    </a>
                  )}
                  {taskArtifacts.markdown_url && (
                    <button
                      onClick={handleDownloadMarkdown}
                      disabled={!markdownReport}
                      className="flex items-center gap-2 rounded-xl border border-black/20 bg-white px-4 py-2 text-xs font-mono font-semibold text-black hover:bg-zinc-50 transition-all"
                    >
                      <Download className="h-3.5 w-3.5 text-zinc-600" />
                      <span>DOWNLOAD .MD</span>
                    </button>
                  )}
                </div>
              </div>

              {/* Rendered Charts */}
              {taskArtifacts.chart_urls.length > 0 && (
                <div className="rounded-2xl border border-black/10 bg-white/90 p-6 space-y-4 backdrop-blur-xl glass-panel shadow-sm">
                  <h3 className="text-xs font-mono uppercase tracking-[0.2em] font-bold text-black flex items-center gap-2">
                    <BarChart3 className="h-4 w-4 text-black" />
                    Embedded Visualizations
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {taskArtifacts.chart_urls.map((url, i) => (
                      <div key={i} className="rounded-xl overflow-hidden border border-black/10 bg-zinc-50 p-2 transition-all hover:border-black/30 hover:shadow-md">
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img
                          src={`${API_BASE}${url}`}
                          alt="Generated Visualization"
                          className="w-full h-auto object-contain rounded-lg"
                        />
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Markdown Content Viewer */}
              {markdownReport && (
                <div className="rounded-2xl border border-black/10 bg-white/90 p-8 space-y-4 backdrop-blur-xl glass-panel shadow-sm">
                  <pre className="whitespace-pre-wrap font-sans text-sm text-zinc-900 leading-relaxed">
                    {markdownReport}
                  </pre>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default function AgentWorkspace() {
  return (
    <Suspense fallback={<div className="p-16 text-center text-zinc-500 font-mono text-xs">INITIALIZING WORKSPACE...</div>}>
      <WorkspaceContent />
    </Suspense>
  );
}
