"use client";

import { useState, useEffect, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { 
  Play, 
  Upload, 
  ShieldCheck, 
  Layers, 
  Download, 
  BarChart3, 
  FileSpreadsheet,
  Terminal,
  RotateCw,
  Eye
} from "lucide-react";
import { createTask, uploadDatasetFile, fetchTask, fetchTaskArtifacts, fetchTaskMarkdown, cancelTask, API_BASE } from "@/lib/api";
import { TaskState, TaskArtifactManifest, PrivacyMode } from "@/lib/types";

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
      setUploadStatus(`Uploaded: ${res.file_name} (${Math.round(res.size_bytes / 1024)} KB)`);
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
    setTaskArtifacts(null);
    setMarkdownReport(null);
    setActiveTab("plan");

    try {
      // The API returns 202 and the stable task ID before execution begins.
      const initialTask = await createTask(objective, filePath, privacyMode);
      setCurrentTask(initialTask);

      // Subscribe immediately; EventSource reconnects with Last-Event-ID and never
      // resubmits the task when the connection drops.
      const eventSource = new EventSource(`${API_BASE}/api/tasks/${initialTask.task_id}/events`);
      const terminalEvents = new Set(["task_completed", "task_failed", "task_cancelled"]);
      const eventNames = [
        "task.created", "task_started", "plan_created", "step_started", "step_completed",
        "step_failed", "step_retry", "task_completed", "task_failed", "task_cancelled", "task.snapshot",
        "task.history_gap",
      ];
      eventNames.forEach((eventName) => {
        eventSource.addEventListener(eventName, (event: MessageEvent) => {
          let data: { message?: string } = {};
          try {
            data = JSON.parse(event.data);
          } catch {
            // Ignore malformed event payloads; the server emits JSON projections.
          }
          setEvents((prev) => [
            ...prev,
            {
              timestamp: new Date().toLocaleTimeString(),
              message: data.message || "Task status updated.",
              type: eventName,
            },
          ]);
          fetchTask(initialTask.task_id).then(setCurrentTask);

          if (eventName === "task.history_gap") {
            fetchTask(initialTask.task_id).then((task) => {
              setCurrentTask(task);
              if (["completed", "failed", "cancelled"].includes(task.status)) {
                eventSource.close();
                setIsRunning(false);
                if (task.status === "completed") {
                  fetchTaskArtifacts(initialTask.task_id).then((artifacts) => {
                    setTaskArtifacts(artifacts);
                    return fetchTaskMarkdown(initialTask.task_id);
                  }).then((report) => setMarkdownReport(report.content)).catch(() => {
                    setTaskArtifacts(null);
                    setMarkdownReport(null);
                  });
                }
              }
            });
          }

          if (terminalEvents.has(eventName)) {
            eventSource.close();
            fetchTask(initialTask.task_id).then((task) => {
              setCurrentTask(task);
              if (eventName === "task_completed") {
                fetchTaskArtifacts(initialTask.task_id).then((artifacts) => {
                  setTaskArtifacts(artifacts);
                  return fetchTaskMarkdown(initialTask.task_id);
                }).then((report) => setMarkdownReport(report.content)).catch(() => {
                  setTaskArtifacts(null);
                  setMarkdownReport(null);
                });
              }
              setIsRunning(false);
              if (eventName === "task_completed") setActiveTab("report");
            });
          }
        });
      });
      eventSource.onerror = () => {
        // Leave the source open. The browser reconnects automatically and sends
        // Last-Event-ID so the server can replay events without task resubmission.
      };
    } catch (err: unknown) {
      alert(`Task initiation error: ${err instanceof Error ? err.message : "Unknown error."}`);
      setIsRunning(false);
    }
  };

  const handleCancelTask = async () => {
    if (!currentTask || cancellationPending || currentTask.status === "cancel_requested") return;
    setCancellationPending(true);
    try {
      const result = await cancelTask(currentTask.task_id);
      setCurrentTask((task) => task ? { ...task, status: result.status } : task);
      const refreshed = await fetchTask(currentTask.task_id);
      setCurrentTask(refreshed);
    } catch (err: unknown) {
      setUploadStatus(`Cancellation failed: ${err instanceof Error ? err.message : "Unknown error."}`);
    } finally {
      setCancellationPending(false);
    }
  };

  const handleDownloadMarkdown = () => {
    if (!markdownReport) return;
    const objectUrl = URL.createObjectURL(new Blob([markdownReport], { type: "text/markdown;charset=utf-8" }));
    const link = document.createElement("a");
    link.href = objectUrl;
    link.download = `${taskArtifacts?.task_id || "aura-report"}.md`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(objectUrl), 0);
  };

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Top Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
            <Layers className="h-6 w-6 text-indigo-400" />
            Agent Workspace
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Specify natural language objectives, configure privacy boundaries, and monitor autonomous DAG execution.
          </p>
        </div>

        {/* Privacy Selector */}
        <div className="flex items-center gap-2 bg-slate-900 border border-slate-800 p-1.5 rounded-xl">
          <span className="text-[11px] font-semibold uppercase text-slate-400 px-2">Privacy Mode:</span>
          {(["local_only", "hybrid", "cloud_enabled"] as PrivacyMode[]).map((mode) => (
            <button
              key={mode}
              onClick={() => setPrivacyMode(mode)}
              className={`rounded-lg px-3 py-1 text-xs font-medium transition-all ${
                privacyMode === mode
                  ? mode === "local_only"
                    ? "bg-emerald-600 text-white shadow"
                    : mode === "hybrid"
                    ? "bg-indigo-600 text-white shadow"
                    : "bg-blue-600 text-white shadow"
                  : "text-slate-400 hover:text-white"
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
          <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 space-y-3">
            <label className="text-xs font-semibold uppercase tracking-wider text-slate-300">
              High-Level Objective / Instructions
            </label>
            <textarea
              rows={3}
              value={objective}
              onChange={(e) => setObjective(e.target.value)}
              placeholder="e.g., Analyze this dataset, identify sales trends, detect underperforming categories, and generate a report with charts."
              className="w-full rounded-lg border border-slate-700 bg-slate-950 p-3 text-sm text-white placeholder-slate-500 focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
            />

            <div className="flex flex-wrap items-center justify-between gap-3 pt-2">
              <div className="flex items-center gap-2">
                <label className="cursor-pointer flex items-center gap-2 rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-xs font-medium text-slate-300 hover:bg-slate-700 transition-colors">
                  <Upload className="h-3.5 w-3.5" />
                  <span>Upload Dataset</span>
                  <input
                    type="file"
                    accept=".csv,.xlsx,.xls,.txt,.md"
                    onChange={handleFileUpload}
                    disabled={uploading}
                    className="hidden"
                  />
                </label>
                {uploadStatus && (
                  <span className="text-[11px] text-emerald-400 font-mono">
                    {uploadStatus}
                  </span>
                )}
              </div>

              <div className="flex items-center gap-2">
                {isRunning && currentTask && (
                  <button
                    onClick={handleCancelTask}
                    disabled={cancellationPending || currentTask.status === "cancel_requested" || ["completed", "failed", "cancelled"].includes(currentTask.status)}
                    className="rounded-lg border border-rose-700 px-3 py-2 text-xs font-semibold text-rose-300 disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    {currentTask.status === "cancel_requested" || cancellationPending ? "Cancellation requested" : "Cancel task"}
                  </button>
                )}
                <button
                  onClick={handleRunTask}
                  disabled={isRunning}
                  className={`flex items-center gap-2 rounded-lg px-5 py-2 text-sm font-semibold text-white shadow-lg transition-all ${
                    isRunning
                      ? "bg-indigo-800 cursor-not-allowed"
                      : "bg-indigo-600 hover:bg-indigo-500 shadow-indigo-600/30"
                  }`}
                >
                  {isRunning ? (
                    <>
                      <RotateCw className="h-4 w-4 animate-spin" />
                      <span>{currentTask?.status === "cancel_requested" ? "Waiting for safe cancellation..." : "Executing DAG..."}</span>
                    </>
                  ) : (
                    <>
                      <Play className="h-4 w-4 fill-white" />
                      <span>Execute Autonomous Task</span>
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Right 1 Col: Execution Boundary & File Target */}
        <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-4 space-y-3">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            Target File & Boundary
          </span>
          <div className="rounded-lg bg-slate-950 p-3 border border-slate-800 font-mono text-xs text-indigo-300 break-all flex items-center gap-2">
            <FileSpreadsheet className="h-4 w-4 shrink-0 text-slate-400" />
            <span>{filePath || "No file attached"}</span>
          </div>

          <div className="text-[11px] text-slate-400 space-y-1.5 pt-1">
            <div className="flex items-center gap-2 text-emerald-400">
              <ShieldCheck className="h-3.5 w-3.5" />
              <span>
                {privacyMode === "local_only"
                  ? "Local Only: 100% on-device deterministic execution."
                  : privacyMode === "hybrid"
                  ? "Hybrid: User content stays local; generic planning uses the local fallback."
                  : "Cloud Enabled: Authorized Gemini API cloud reasoning."}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Tabs for Plan, Live Logs, and Final Report */}
      <div className="border-b border-slate-800 flex items-center gap-4">
        <button
          onClick={() => setActiveTab("plan")}
          className={`pb-3 text-sm font-semibold transition-colors border-b-2 ${
            activeTab === "plan"
              ? "border-indigo-500 text-indigo-400"
              : "border-transparent text-slate-400 hover:text-slate-300"
          }`}
        >
          Task Status
        </button>
        <button
          onClick={() => setActiveTab("logs")}
          className={`pb-3 text-sm font-semibold transition-colors border-b-2 ${
            activeTab === "logs"
              ? "border-indigo-500 text-indigo-400"
              : "border-transparent text-slate-400 hover:text-slate-300"
          }`}
        >
          Live Telemetry & Logs ({events.length})
        </button>
        <button
          onClick={() => setActiveTab("report")}
          className={`pb-3 text-sm font-semibold transition-colors border-b-2 ${
            activeTab === "report"
              ? "border-indigo-500 text-indigo-400"
              : "border-transparent text-slate-400 hover:text-slate-300"
          }`}
        >
          Final Deliverable & Visualizations
        </button>
      </div>

      {/* Tab 1: Privacy-safe task status */}
      {activeTab === "plan" && (
        <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-6 space-y-3">
          {!currentTask ? (
            <p className="text-sm text-slate-400">Submit a task to view its status.</p>
          ) : (
            <>
              <p className="text-sm text-slate-300">Task <span className="font-mono">{currentTask.task_id}</span></p>
              <p className="text-sm text-slate-300">Status: <span className="font-semibold capitalize">{currentTask.status}</span></p>
              <p className="text-xs text-slate-500">Execution plans and tool data remain private and are not included in task status responses.</p>
            </>
          )}
        </div>
      )}

      {/* Tab 2: Live Log Terminal */}
      {activeTab === "logs" && (
        <div className="rounded-xl border border-slate-800 bg-slate-950 p-4 font-mono text-xs space-y-2 max-h-96 overflow-y-auto">
          <div className="flex items-center gap-2 text-slate-400 pb-2 border-b border-slate-800">
            <Terminal className="h-4 w-4 text-indigo-400" />
            <span>AURA Execution Event Stream</span>
          </div>
          {events.length === 0 ? (
            <p className="text-slate-600">Waiting for task execution events...</p>
          ) : (
            events.map((ev, i) => (
              <div key={i} className="flex items-start gap-2 text-slate-300">
                <span className="text-slate-500 shrink-0">[{ev.timestamp}]</span>
                <span
                  className={
                    ev.type === "step_completed"
                      ? "text-emerald-400"
                      : ev.type === "step_failed"
                      ? "text-rose-400"
                      : ev.type === "step_started"
                      ? "text-indigo-400"
                      : "text-slate-300"
                  }
                >
                  {ev.message}
                </span>
              </div>
            ))
          )}
        </div>
      )}

      {/* Tab 3: Final Deliverable & Embedded Charts */}
      {activeTab === "report" && (
        <div className="space-y-6">
          {!taskArtifacts ? (
            <div className="rounded-xl border border-slate-800 bg-slate-900/30 p-12 text-center text-slate-400 text-sm">
              No deliverable generated yet. Run a task to generate analytical reports.
            </div>
          ) : (
            <div className="space-y-6">
              {/* Header Actions */}
              <div className="flex items-center justify-between bg-slate-900/60 p-4 rounded-xl border border-slate-800">
                <div>
                  <h2 className="text-lg font-bold text-white">
                    {taskArtifacts.title || "Analytical Deliverable"}
                  </h2>
                  <p className="text-xs text-slate-400">
                    Generated autonomously with verified Python calculations and Matplotlib charts.
                  </p>
                </div>
                <div className="flex items-center gap-3">
                  {taskArtifacts.html_url && (
                    <a
                      href={`${API_BASE}${taskArtifacts.html_url}`}
                      target="_blank"
                      rel="noreferrer"
                      className="flex items-center gap-2 rounded-lg bg-indigo-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-indigo-500 shadow"
                    >
                      <Eye className="h-3.5 w-3.5" />
                      <span>View Standalone HTML</span>
                    </a>
                  )}
                  {taskArtifacts.markdown_url && (
                    <button
                      onClick={handleDownloadMarkdown}
                      disabled={!markdownReport}
                      className="flex items-center gap-2 rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-xs font-semibold text-slate-200 hover:bg-slate-700"
                    >
                      <Download className="h-3.5 w-3.5" />
                      <span>Download .md</span>
                    </button>
                  )}
                </div>
              </div>

              {/* Rendered Charts */}
              {taskArtifacts.chart_urls.length > 0 && (
                <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5 space-y-4">
                  <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                    <BarChart3 className="h-4 w-4 text-indigo-400" />
                    Embedded Visualizations
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {taskArtifacts.chart_urls.map((url, i) => {
                      return (
                        <div key={i} className="rounded-lg overflow-hidden border border-slate-800 bg-slate-950">
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img
                            src={`${API_BASE}${url}`}
                            alt="Generated Visualization"
                            className="w-full h-auto object-contain"
                          />
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Markdown Content Viewer */}
              {markdownReport && (
                <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-6 space-y-4">
                  <pre className="whitespace-pre-wrap font-sans text-sm text-slate-200 leading-relaxed">
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
    <Suspense fallback={<div className="p-12 text-center text-slate-400">Loading Agent Workspace...</div>}>
      <WorkspaceContent />
    </Suspense>
  );
}
