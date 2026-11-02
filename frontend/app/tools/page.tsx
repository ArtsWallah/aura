"use client";

import { useEffect, useState } from "react";
import { 
  Wrench, 
  Play, 
  ShieldCheck, 
  Code, 
  Clock, 
  CheckCircle2, 
  Terminal,
  FileSpreadsheet,
  BarChart3,
  Search,
  FileText
} from "lucide-react";
import { fetchTools, executeToolDirect } from "@/lib/api";
import { ToolDefinition } from "@/lib/types";

export default function ToolRegistryPage() {
  const [tools, setTools] = useState<ToolDefinition[]>([]);
  const [selectedTool, setSelectedTool] = useState<ToolDefinition | null>(null);
  const [customArgs, setCustomArgs] = useState<string>("{\n  \"file_path\": \"data/sample_sales.csv\"\n}");
  const [execResult, setExecResult] = useState<any>(null);
  const [executing, setExecuting] = useState<boolean>(false);

  useEffect(() => {
    fetchTools().then((list) => {
      setTools(list);
      if (list.length > 0) {
        setSelectedTool(list[0]);
      }
    });
  }, []);

  const handleSelectTool = (tool: ToolDefinition) => {
    setSelectedTool(tool);
    // Provide sensible default JSON based on tool
    if (tool.name === "inspect_dataset") {
      setCustomArgs(JSON.stringify({ file_path: "data/sample_sales.csv", sample_rows: 5 }, null, 2));
    } else if (tool.name === "descriptive_statistics") {
      setCustomArgs(JSON.stringify({ file_path: "data/sample_sales.csv" }, null, 2));
    } else if (tool.name === "group_comparison") {
      setCustomArgs(JSON.stringify({ file_path: "data/sample_sales.csv", group_by: "category", metric_column: "revenue" }, null, 2));
    } else if (tool.name === "generate_chart") {
      setCustomArgs(JSON.stringify({ file_path: "data/sample_sales.csv", chart_type: "bar", title: "Revenue by Category", x_column: "category", y_column: "revenue" }, null, 2));
    } else if (tool.name === "web_research") {
      setCustomArgs(JSON.stringify({ query: "enterprise cloud artificial intelligence adoption 2026", max_results: 3 }, null, 2));
    } else if (tool.name === "system_health") {
      setCustomArgs(JSON.stringify({ check_storage: true }, null, 2));
    } else {
      setCustomArgs("{\n  \n}");
    }
    setExecResult(null);
  };

  const handleExecute = async () => {
    if (!selectedTool) return;
    setExecuting(true);
    setExecResult(null);

    try {
      const parsedArgs = JSON.parse(customArgs);
      const res = await executeToolDirect(selectedTool.name, parsedArgs);
      setExecResult(res);
    } catch (err: any) {
      setExecResult({ success: false, error: `Invalid JSON or execution error: ${err.message}` });
    } finally {
      setExecuting(false);
    }
  };

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      {/* Header */}
      <div className="border-b border-slate-800 pb-4">
        <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
          <Wrench className="h-6 w-6 text-indigo-400" />
          Deterministic Tool Registry
        </h1>
        <p className="text-xs text-slate-400 mt-1">
          Strongly-typed Python execution engines registered with validated Pydantic schemas, permissions, and timeout controls.
        </p>
      </div>

      {/* Main Layout: Tools Catalog & Interactive Runner */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Tools List */}
        <div className="space-y-2.5">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            Registered Engines ({tools.length})
          </span>
          <div className="space-y-2">
            {tools.map((t) => {
              const isSelected = selectedTool?.name === t.name;
              return (
                <button
                  key={t.name}
                  onClick={() => handleSelectTool(t)}
                  className={`w-full text-left rounded-xl p-3.5 border transition-all ${
                    isSelected
                      ? "border-indigo-500 bg-indigo-950/30 text-white shadow"
                      : "border-slate-800 bg-slate-900/40 text-slate-300 hover:bg-slate-800/40"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-xs font-bold text-indigo-400">{t.name}</span>
                    <span className="rounded bg-slate-800 px-1.5 py-0.5 text-[10px] font-mono text-slate-300">
                      {t.category.toUpperCase()}
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-400 mt-1 line-clamp-2 leading-relaxed">
                    {t.description}
                  </p>
                </button>
              );
            })}
          </div>
        </div>

        {/* Right: Selected Tool Spec & Direct Runner */}
        <div className="lg:col-span-2 space-y-6">
          {!selectedTool ? (
            <div className="rounded-xl border border-slate-800 bg-slate-900/30 p-12 text-center text-slate-400 text-sm">
              Select a tool to inspect its specification and test execution.
            </div>
          ) : (
            <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-6 space-y-6">
              {/* Tool Header */}
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-4">
                <div>
                  <h2 className="text-lg font-bold text-white font-mono">{selectedTool.name}</h2>
                  <p className="text-xs text-slate-400 mt-1">{selectedTool.description}</p>
                </div>
                <div className="flex items-center gap-2">
                  <span className="rounded bg-emerald-500/10 border border-emerald-500/20 px-2 py-0.5 text-[11px] font-mono font-medium text-emerald-400">
                    {selectedTool.execution_location.toUpperCase()}
                  </span>
                  <span className="rounded bg-slate-800 border border-slate-700 px-2 py-0.5 text-[11px] font-mono text-slate-300">
                    timeout: {selectedTool.timeout_seconds}s
                  </span>
                </div>
              </div>

              {/* Direct Runner Sandbox */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <label className="text-xs font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-2">
                    <Terminal className="h-4 w-4 text-indigo-400" />
                    Input Arguments (JSON)
                  </label>
                  <button
                    onClick={handleExecute}
                    disabled={executing}
                    className="flex items-center gap-1.5 rounded-lg bg-indigo-600 px-4 py-1.5 text-xs font-semibold text-white hover:bg-indigo-500 shadow"
                  >
                    <Play className="h-3 w-3 fill-white" />
                    <span>{executing ? "Executing..." : "Execute Tool"}</span>
                  </button>
                </div>

                <textarea
                  rows={5}
                  value={customArgs}
                  onChange={(e) => setCustomArgs(e.target.value)}
                  className="w-full rounded-lg border border-slate-800 bg-slate-950 p-3 font-mono text-xs text-indigo-300 focus:outline-none focus:border-indigo-500"
                />
              </div>

              {/* Tool Execution Result */}
              {execResult && (
                <div className="space-y-2 pt-2 border-t border-slate-800">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                      Execution Result ({execResult.execution_time_ms ? `${Math.round(execResult.execution_time_ms)}ms` : "0ms"})
                    </span>
                    <span
                      className={`text-xs font-bold ${
                        execResult.success ? "text-emerald-400" : "text-rose-400"
                      }`}
                    >
                      {execResult.success ? "SUCCESS" : "ERROR"}
                    </span>
                  </div>

                  <pre className="rounded-lg bg-slate-950 p-3 font-mono text-[11px] text-slate-300 overflow-x-auto max-h-72 border border-slate-800">
                    {JSON.stringify(execResult.data || execResult.error, null, 2)}
                  </pre>
                </div>
              )}

              {/* Input Schema Inspector */}
              <div className="space-y-2 pt-2 border-t border-slate-800">
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                  Pydantic JSON Schema
                </span>
                <pre className="rounded-lg bg-slate-950/80 p-3 font-mono text-[10px] text-slate-500 overflow-x-auto max-h-40 border border-slate-800/80">
                  {JSON.stringify(selectedTool.input_schema, null, 2)}
                </pre>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
