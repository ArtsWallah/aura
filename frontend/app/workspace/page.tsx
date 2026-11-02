"use client";

import { useState, useEffect, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { 
  Play, 
  Upload, 
  FileText, 
  ShieldCheck, 
  CheckCircle2, 
  Clock, 
  AlertCircle, 
  Layers, 
  Download, 
  BarChart3, 
  FileSpreadsheet,
  Terminal,
  RotateCw,
  Eye
} from "lucide-react";
import { createTask, uploadDatasetFile, fetchTask, API_BASE } from "@/lib/api";
import { TaskState, PrivacyMode, StepStatus } from "@/lib/types";

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

  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [currentTask, setCurrentTask] = useState<TaskState | null>(null);
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
    } catch (err: any) {
      setUploadStatus(`Upload failed: ${err.message}`);
    } finally {
      setUploading(false);
    }
  };

  const handleRunTask = async () => {
    if (!objective.trim()) return;

    setIsRunning(true);
    setEvents([]);
    setCurrentTask(null);
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

          if (terminalEvents.has(eventName)) {
            eventSource.close();
            fetchTask(initialTask.task_id).then((task) => {
              setCurrentTask(task);
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
    } catch (err: any) {
      alert(`Task initiation error: ${err.message}`);
      setIsRunning(false);
    }
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
                    className="hidden"
                  />
                </label>
                {uploadStatus && (
                  <span className="text-[11px] text-emerald-400 font-mono">
                    {uploadStatus}
                  </span>
                )}
              </div>

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
                    <span>Executing DAG...</span>
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
                  ? "Hybrid: Proprietary rows retained locally; minimized schemas to cloud."
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
          Execution Plan & DAG ({currentTask?.plan?.steps.length || 0} Steps)
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

      {/* Tab 1: Execution Plan Visualizer */}
      {activeTab === "plan" && (
        <div className="space-y-4">
          {!currentTask?.plan ? (
            <div className="rounded-xl border border-slate-800 bg-slate-900/30 p-12 text-center text-slate-400 text-sm">
              Click &quot;Execute Autonomous Task&quot; above to synthesize the structured execution plan.
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {currentTask.plan.steps.map((step, idx) => {
                const isCompleted = step.status === "completed";
                const isFailed = step.status === "failed";
                const isInProgress = step.status === "in_progress";

                return (
                  <div
                    key={step.step_id}
                    className={`rounded-xl border p-4 space-y-2.5 transition-all ${
                      isCompleted
                        ? "border-emerald-500/30 bg-emerald-950/10"
                        : isFailed
                        ? "border-rose-500/30 bg-rose-950/10"
                        : isInProgress
                        ? "border-indigo-500 bg-indigo-950/20 shadow-lg shadow-indigo-500/10"
                        : "border-slate-800 bg-slate-900/40"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-mono text-[11px] text-slate-400 font-semibold uppercase">
                        Step {idx + 1} &bull; {step.step_id}
                      </span>
                      <span
                        className={`rounded px-2 py-0.5 text-[10px] font-bold ${
                          isCompleted
                            ? "bg-emerald-500/20 text-emerald-300"
                            : isFailed
                            ? "bg-rose-500/20 text-rose-300"
                            : isInProgress
                            ? "bg-indigo-500/20 text-indigo-300 animate-pulse"
                            : "bg-slate-800 text-slate-400"
                        }`}
                      >
                        {step.status.toUpperCase()}
                      </span>
                    </div>

                    <h3 className="text-sm font-semibold text-white leading-snug">
                      {step.title}
                    </h3>
                    <p className="text-xs text-slate-400 leading-relaxed">
                      {step.description}
                    </p>

                    <div className="pt-2 border-t border-slate-800/80 flex items-center justify-between text-[11px] text-slate-400">
                      <span className="font-mono text-indigo-300">tool: {step.tool_name}</span>
                      {step.duration_ms > 0 && (
                        <span>{Math.round(step.duration_ms)} ms</span>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
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
          {!currentTask?.final_deliverable ? (
            <div className="rounded-xl border border-slate-800 bg-slate-900/30 p-12 text-center text-slate-400 text-sm">
              No deliverable generated yet. Run a task to generate analytical reports.
            </div>
          ) : (
            <div className="space-y-6">
              {/* Header Actions */}
              <div className="flex items-center justify-between bg-slate-900/60 p-4 rounded-xl border border-slate-800">
                <div>
                  <h2 className="text-lg font-bold text-white">
                    {currentTask.final_deliverable.title || "Analytical Deliverable"}
                  </h2>
                  <p className="text-xs text-slate-400">
                    Generated autonomously with verified Python calculations and Matplotlib charts.
                  </p>
                </div>
                <div className="flex items-center gap-3">
                  {currentTask.final_deliverable.relative_html_url && (
                    <a
                      href={`${API_BASE}${currentTask.final_deliverable.relative_html_url}`}
                      target="_blank"
                      rel="noreferrer"
                      className="flex items-center gap-2 rounded-lg bg-indigo-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-indigo-500 shadow"
                    >
                      <Eye className="h-3.5 w-3.5" />
                      <span>View Standalone HTML</span>
                    </a>
                  )}
                  {currentTask.final_deliverable.relative_markdown_url && (
                    <a
                      href={`${API_BASE}${currentTask.final_deliverable.relative_markdown_url}`}
                      download
                      className="flex items-center gap-2 rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-xs font-semibold text-slate-200 hover:bg-slate-700"
                    >
                      <Download className="h-3.5 w-3.5" />
                      <span>Download .md</span>
                    </a>
                  )}
                </div>
              </div>

              {/* Rendered Charts */}
              {currentTask.artifacts && currentTask.artifacts.filter(a => a.endsWith(".png")).length > 0 && (
                <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5 space-y-4">
                  <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                    <BarChart3 className="h-4 w-4 text-indigo-400" />
                    Embedded Visualizations
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {currentTask.artifacts.filter(a => a.endsWith(".png")).map((art, i) => {
                      const filename = art.split(/[\\/]/).pop();
                      return (
                        <div key={i} className="rounded-lg overflow-hidden border border-slate-800 bg-slate-950">
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img
                            src={`${API_BASE}/data/charts/${filename}`}
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
              {currentTask.final_deliverable.markdown_content && (
                <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-6 space-y-4">
                  <pre className="whitespace-pre-wrap font-sans text-sm text-slate-200 leading-relaxed">
                    {currentTask.final_deliverable.markdown_content}
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
