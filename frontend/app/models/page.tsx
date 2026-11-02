"use client";

import { useEffect, useState } from "react";
import { 
  Cpu, 
  Zap, 
  Activity, 
  CheckCircle2, 
  XCircle, 
  Play, 
  Layers, 
  Sparkles,
  ArrowRight
} from "lucide-react";
import { fetchModels, classifyIntent, generateEmbedding } from "@/lib/api";
import { checkWebGPUAvailability, runBrowserLocalInference, WebGPUStatus } from "@/lib/webgpu";
import { ModelMetadata } from "@/lib/types";

export default function ModelRegistryPage() {
  const [models, setModels] = useState<ModelMetadata[]>([]);
  const [webGpuStatus, setWebGpuStatus] = useState<WebGPUStatus | null>(null);

  // Playground state
  const [intentInput, setIntentInput] = useState<string>("Analyze the sales dataset, detect anomalies and plot a bar chart");
  const [intentOutput, setIntentOutput] = useState<any>(null);
  const [classifying, setClassifying] = useState<boolean>(false);

  const [embedInput, setEmbedInput] = useState<string>("Autonomous agent reasoning and memory persistence");
  const [embedOutput, setEmbedOutput] = useState<any>(null);
  const [embedding, setEmbedding] = useState<boolean>(false);

  // WebGPU in-browser test state
  const [browserInput, setBrowserInput] = useState<string>("Research global semiconductor manufacturing supply chains");
  const [browserResult, setBrowserResult] = useState<any>(null);
  const [runningBrowser, setRunningBrowser] = useState<boolean>(false);

  useEffect(() => {
    fetchModels().then(setModels).catch(() => []);
    checkWebGPUAvailability().then(setWebGpuStatus);
  }, []);

  const handleClassify = async () => {
    setClassifying(true);
    try {
      const res = await classifyIntent(intentInput);
      setIntentOutput(res);
    } finally {
      setClassifying(false);
    }
  };

  const handleEmbed = async () => {
    setEmbedding(true);
    try {
      const res = await generateEmbedding(embedInput);
      setEmbedOutput(res);
    } finally {
      setEmbedding(false);
    }
  };

  const handleBrowserInference = async () => {
    setRunningBrowser(true);
    try {
      const res = await runBrowserLocalInference(browserInput);
      setBrowserResult(res);
    } finally {
      setRunningBrowser(false);
    }
  };

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      {/* Header */}
      <div className="border-b border-slate-800 pb-4">
        <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
          <Cpu className="h-6 w-6 text-indigo-400" />
          Specialized Small AI Models Registry
        </h1>
        <p className="text-xs text-slate-400 mt-1">
          Catalog of specialized narrow models, intent classifiers, semantic embedding extractors, and browser-local WebGPU neural execution.
        </p>
      </div>

      {/* WebGPU Status Banner */}
      <div className="rounded-xl border border-indigo-500/20 bg-gradient-to-r from-indigo-950/40 via-slate-900/60 to-slate-950 p-5 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="flex h-2.5 w-2.5 rounded-full bg-cyan-400 shadow-sm shadow-cyan-400" />
            <h3 className="text-sm font-bold text-white uppercase tracking-wider">
              Browser WebGPU Subsystem
            </h3>
          </div>
          <p className="text-xs text-slate-300 mt-1">
            {webGpuStatus?.details || "Checking WebGPU hardware support..."}
          </p>
          {webGpuStatus?.adapterName && (
            <span className="inline-block mt-2 rounded bg-cyan-500/10 border border-cyan-500/30 px-2 py-0.5 text-[11px] font-mono text-cyan-300">
              Adapter: {webGpuStatus.adapterName}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <span className="rounded-full bg-slate-800 border border-slate-700 px-3 py-1 text-xs font-mono text-slate-300">
            Backend: {webGpuStatus?.backend.toUpperCase() || "CHECKING"}
          </span>
        </div>
      </div>

      {/* Model Catalog Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
        {models.map((m) => (
          <div
            key={m.model_id}
            className="rounded-xl border border-slate-800 bg-slate-900/50 p-5 space-y-3 flex flex-col justify-between"
          >
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="rounded bg-indigo-500/10 border border-indigo-500/20 px-2 py-0.5 text-[10px] font-mono font-semibold text-indigo-400 uppercase">
                  {m.category}
                </span>
                <span className="text-[11px] font-mono text-slate-400">
                  {m.execution_location.toUpperCase()}
                </span>
              </div>
              <h3 className="text-base font-bold text-white">{m.name}</h3>
              <p className="text-xs text-slate-400 leading-relaxed">{m.description}</p>
            </div>

            <div className="pt-3 border-t border-slate-800/80 space-y-1.5 text-xs font-mono">
              <div className="flex justify-between text-slate-400">
                <span>Parameters:</span>
                <span className="text-slate-200">{m.parameters_size}</span>
              </div>
              <div className="flex justify-between text-slate-400">
                <span>Average Latency:</span>
                <span className="text-emerald-400 font-bold">{m.average_latency_ms.toFixed(2)} ms</span>
              </div>
              <div className="flex justify-between text-slate-400">
                <span>Inferences Run:</span>
                <span className="text-slate-200">{m.total_inferences}</span>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Interactive Testing Sandboxes */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 pt-2">
        {/* Sandbox 1: Intent Classifier */}
        <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <Zap className="h-4 w-4 text-amber-400" />
              Intent Classifier Playground (Local DistilBERT Hybrid)
            </h3>
          </div>
          <p className="text-xs text-slate-400">
            Categorizes requests into deterministic routing domains before invoking LLMs.
          </p>

          <div className="flex gap-2">
            <input
              type="text"
              value={intentInput}
              onChange={(e) => setIntentInput(e.target.value)}
              className="flex-1 rounded-lg border border-slate-700 bg-slate-950 px-3 py-1.5 text-xs text-white focus:outline-none focus:border-indigo-500"
            />
            <button
              onClick={handleClassify}
              disabled={classifying}
              className="rounded-lg bg-indigo-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-indigo-500"
            >
              {classifying ? "Classifying..." : "Classify"}
            </button>
          </div>

          {intentOutput && (
            <div className="rounded-lg bg-slate-950 p-3 border border-slate-800 text-xs font-mono space-y-1 text-slate-300">
              <div>
                <span className="text-slate-500">Predicted Intent:</span>{" "}
                <span className="text-emerald-400 font-bold">{intentOutput.intent}</span>
              </div>
              <div>
                <span className="text-slate-500">Confidence:</span> {(intentOutput.confidence * 100).toFixed(1)}%
              </div>
              <div>
                <span className="text-slate-500">Inference Latency:</span> {intentOutput.latency_ms} ms
              </div>
              <div>
                <span className="text-slate-500">Requires LLM Fallback:</span>{" "}
                {intentOutput.requires_llm_fallback ? "Yes (Ambiguous)" : "No (Deterministic Route)"}
              </div>
            </div>
          )}
        </div>

        {/* Sandbox 2: Browser-Local WebGPU Execution */}
        <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-cyan-400" />
              Browser-Local WebGPU Neural Runner
            </h3>
          </div>
          <p className="text-xs text-slate-400">
            Runs client-side feature extraction & intent classification in your browser without transmitting raw text.
          </p>

          <div className="flex gap-2">
            <input
              type="text"
              value={browserInput}
              onChange={(e) => setBrowserInput(e.target.value)}
              className="flex-1 rounded-lg border border-slate-700 bg-slate-950 px-3 py-1.5 text-xs text-white focus:outline-none focus:border-cyan-500"
            />
            <button
              onClick={handleBrowserInference}
              disabled={runningBrowser}
              className="rounded-lg bg-cyan-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-cyan-500"
            >
              {runningBrowser ? "Computing..." : "Run WebGPU"}
            </button>
          </div>

          {browserResult && (
            <div className="rounded-lg bg-slate-950 p-3 border border-slate-800 text-xs font-mono space-y-1 text-slate-300">
              <div>
                <span className="text-slate-500">Execution Hardware:</span>{" "}
                <span className="text-cyan-400 font-bold uppercase">{browserResult.backend}</span>
              </div>
              <div>
                <span className="text-slate-500">Predicted Intent:</span> {browserResult.intent}
              </div>
              <div>
                <span className="text-slate-500">Local Latency:</span> {browserResult.latencyMs} ms
              </div>
              <div>
                <span className="text-slate-500">64-Dim Vector Preview:</span> [
                {browserResult.embedding.slice(0, 4).map((n: number) => n.toFixed(3)).join(", ")}...]
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

