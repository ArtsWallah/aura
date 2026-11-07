"use client";

import { useEffect, useState } from "react";
import { 
  Wrench, 
  Play, 
  Terminal
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
      <div className="border-b border-black/10 pb-5">
        <div className="inline-flex items-center gap-2 rounded-full border border-black/10 bg-black/[0.04] px-3 py-0.5 text-[10px] font-mono uppercase tracking-[0.2em] text-zinc-700">
          <span className="h-1.5 w-1.5 rounded-full bg-black animate-pulse" />
          Deterministic Sandbox
        </div>
        <h1 className="text-2xl md:text-3xl font-black font-mono tracking-tight text-black mt-1.5 flex items-center gap-2.5">
          <Wrench className="h-6 w-6 text-black" />
          DETERMINISTIC TOOL REGISTRY
        </h1>
        <p className="text-xs text-zinc-600 font-sans mt-0.5">
          Strongly-typed Python execution engines registered with validated Pydantic schemas, permissions, and timeout controls.
        </p>
      </div>

      {/* Main Layout: Tools Catalog & Interactive Runner */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Tools List */}
        <div className="space-y-3">
          <span className="text-[10px] font-mono uppercase tracking-[0.2em] font-bold text-zinc-500">
            Registered Engines ({tools.length})
          </span>
          <div className="space-y-2">
            {tools.map((t) => {
              const isSelected = selectedTool?.name === t.name;
              return (
                <button
                  key={t.name}
                  onClick={() => handleSelectTool(t)}
                  className={`w-full text-left rounded-2xl p-4 border transition-all duration-200 glass-panel ${isSelected
                      ? "border-black bg-black text-white shadow-md scale-[1.01]"
                      : "border-black/10 bg-white/90 text-zinc-900 hover:border-black/30 hover:bg-zinc-50 shadow-2xs"
                    }`}
                >
                  <div className="flex items-center justify-between">
                    <span className={`font-mono text-xs font-bold tracking-tight ${isSelected ? "text-white" : "text-black"}`}>{t.name}</span>
                    <span className={`rounded-lg px-2 py-0.5 text-[9px] font-mono uppercase font-semibold ${isSelected ? "border border-white/20 bg-white/10 text-white" : "border border-black/10 bg-zinc-100 text-zinc-700"}`}>
                      {t.category}
                    </span>
                  </div>
                  <p className={`text-[11px] mt-1.5 line-clamp-2 leading-relaxed font-sans ${isSelected ? "text-zinc-300" : "text-zinc-600"}`}>
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
            <div className="rounded-2xl border border-dashed border-black/15 bg-zinc-50/50 p-16 text-center text-zinc-500 font-mono text-xs glass-panel">
              SELECT AN ENGINE TO INSPECT CONTRACTS AND TEST EXECUTION.
            </div>
          ) : (
            <div className="rounded-2xl border border-black/10 bg-white/90 p-6 space-y-6 backdrop-blur-xl glass-panel shadow-sm">
              {/* Tool Header */}
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-black/10 pb-5">
                <div>
                  <h2 className="text-xl font-black text-black font-mono tracking-tight">{selectedTool.name}</h2>
                  <p className="text-xs text-zinc-600 mt-1 font-sans">{selectedTool.description}</p>
                </div>
                <div className="flex items-center gap-2">
                  <span className="rounded-lg border border-black/15 bg-zinc-100 px-2.5 py-1 text-[10px] font-mono font-bold text-black uppercase tracking-wider">
                    {selectedTool.execution_location}
                  </span>
                  <span className="rounded-lg border border-black/10 bg-zinc-50 px-2.5 py-1 text-[10px] font-mono text-zinc-600 font-medium">
                    TIMEOUT: {selectedTool.timeout_seconds}S
                  </span>
                </div>
              </div>

              {/* Direct Runner Sandbox */}
              <div className="space-y-3 font-mono">
                <div className="flex items-center justify-between">
                  <label className="text-[10px] uppercase tracking-[0.2em] text-zinc-600 font-bold flex items-center gap-2">
                    <Terminal className="h-4 w-4 text-black" />
                    Input Arguments (JSON Payload)
                  </label>
                  <button
                    onClick={handleExecute}
                    disabled={executing}
                    className="flex items-center gap-1.5 rounded-xl bg-black px-4 py-1.5 text-xs font-mono font-bold text-white hover:bg-zinc-800 transition-all shadow-[0_4px_16px_rgba(0,0,0,0.18)] hover:scale-[1.02] active:scale-[0.98]"
                  >
                    <Play className="h-3 w-3 fill-white" />
                    <span>{executing ? "COMPUTING..." : "EXECUTE DIRECT"}</span>
                  </button>
                </div>

                <textarea
                  rows={5}
                  value={customArgs}
                  onChange={(e) => setCustomArgs(e.target.value)}
                  className="w-full rounded-xl border border-black/15 bg-zinc-50 p-4 font-mono text-xs text-black focus:outline-none focus:border-black focus:bg-white shadow-2xs transition-all"
                />
              </div>

              {/* Tool Execution Result */}
              {execResult && (
                <div className="space-y-2.5 pt-3 border-t border-black/10 font-mono">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] uppercase tracking-[0.2em] font-bold text-zinc-600">
                      Execution Result ({execResult.execution_time_ms ? `${Math.round(execResult.execution_time_ms)}ms` : "0ms"})
                    </span>
                    <span
                      className={`text-xs font-bold px-2.5 py-0.5 rounded-md ${
                        execResult.success ? "bg-black text-white font-black" : "border border-zinc-400 bg-zinc-200 text-zinc-800"
                      }`}
                    >
                      {execResult.success ? "SUCCESS" : "ERROR"}
                    </span>
                  </div>

                  <pre className="rounded-xl bg-zinc-50 p-4 font-mono text-[11px] text-zinc-900 overflow-x-auto max-h-72 border border-black/10 shadow-2xs">
                    {JSON.stringify(execResult.data || execResult.error, null, 2)}
                  </pre>
                </div>
              )}

              {/* Input Schema Inspector */}
              <div className="space-y-2 pt-3 border-t border-black/10">
                <span className="text-[10px] font-mono uppercase tracking-[0.2em] font-bold text-zinc-500">
                  Pydantic JSON Contract Definition
                </span>
                <pre className="rounded-xl bg-zinc-50 p-4 font-mono text-[10px] text-zinc-700 overflow-x-auto max-h-40 border border-black/10">
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
