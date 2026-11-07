"use client";

import { useEffect, useState } from "react";
import { 
  Cpu, 
  Zap, 
  Sparkles,
  ShieldCheck,
  Terminal
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
      <div className="border-b border-black/10 pb-5">
        <div className="inline-flex items-center gap-2 rounded-full border border-black/10 bg-black/[0.04] px-3 py-0.5 text-[10px] font-mono uppercase tracking-[0.2em] text-zinc-700">
          <span className="h-1.5 w-1.5 rounded-full bg-black animate-pulse" />
          Neural Execution Subsystem
        </div>
        <h1 className="text-2xl md:text-3xl font-black font-mono tracking-tight text-black mt-1.5 flex items-center gap-2.5">
          <Cpu className="h-6 w-6 text-black" />
          SPECIALIZED AI MODEL ENCLAVES
        </h1>
        <p className="text-xs text-zinc-600 font-sans mt-0.5">
          Catalog of specialized narrow models, intent classifiers, semantic embedding extractors, and browser-local WebGPU neural execution.
        </p>
      </div>

      {/* WebGPU Status Banner */}
      <div className="relative overflow-hidden rounded-2xl border border-black/10 bg-white/90 p-6 flex flex-col sm:flex-row sm:items-center justify-between gap-4 backdrop-blur-xl glass-panel shadow-sm">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <span className="flex h-2 w-2 rounded-full bg-black shadow-xs animate-ping" />
            <h3 className="text-xs font-mono font-bold text-black uppercase tracking-[0.2em]">
              Browser WebGPU Subsystem
            </h3>
          </div>
          <p className="text-xs text-zinc-600 font-sans">
            {webGpuStatus?.details || "Checking WebGPU hardware support..."}
          </p>
          {webGpuStatus?.adapterName && (
            <span className="inline-block mt-1.5 rounded-lg border border-black/10 bg-zinc-50 px-2.5 py-0.5 text-[10px] font-mono text-zinc-800 font-semibold">
              ADAPTER: {webGpuStatus.adapterName}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <span className="rounded-xl border border-black/15 bg-zinc-100 px-3.5 py-1.5 text-xs font-mono text-black font-bold">
            BACKEND: {webGpuStatus?.backend.toUpperCase() || "CHECKING"}
          </span>
        </div>
      </div>

      {/* Model Catalog Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
        {models.map((m) => (
          <div
            key={m.model_id}
            className="rounded-2xl border border-black/10 bg-white/90 p-6 space-y-4 flex flex-col justify-between backdrop-blur-xl glass-panel transition-all hover:border-black/30 hover:shadow-md group"
          >
            <div className="space-y-2.5">
              <div className="flex items-center justify-between">
                <span className="rounded-lg border border-black/15 bg-zinc-100 px-2 py-0.5 text-[9px] font-mono font-bold text-black uppercase tracking-wider">
                  {m.category}
                </span>
                <span className="text-[10px] font-mono text-zinc-500 uppercase tracking-widest font-semibold">
                  {m.execution_location}
                </span>
              </div>
              <h3 className="text-base font-bold font-mono tracking-tight text-black group-hover:text-zinc-700">{m.name}</h3>
              <p className="text-xs text-zinc-600 leading-relaxed font-sans">{m.description}</p>
            </div>

            <div className="pt-3 border-t border-black/10 space-y-2 text-xs font-mono">
              <div className="flex justify-between text-zinc-600">
                <span className="font-semibold text-zinc-500">PARAMETERS:</span>
                <span className="text-black font-bold">{m.parameters_size}</span>
              </div>
              <div className="flex justify-between text-zinc-600">
                <span className="font-semibold text-zinc-500">AVERAGE LATENCY:</span>
                <span className="text-black font-bold">{m.average_latency_ms.toFixed(2)} MS</span>
              </div>
              <div className="flex justify-between text-zinc-600">
                <span className="font-semibold text-zinc-500">TOTAL INFERENCES:</span>
                <span className="text-zinc-800 font-semibold">{m.total_inferences}</span>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Interactive Testing Sandboxes */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 pt-2">
        {/* Sandbox 1: Intent Classifier */}
        <div className="rounded-2xl border border-black/10 bg-white/90 p-6 space-y-4 backdrop-blur-xl glass-panel shadow-sm">
          <div>
            <h3 className="text-xs font-mono uppercase tracking-[0.2em] font-bold text-black flex items-center gap-2">
              <Zap className="h-4 w-4 text-black" />
              Intent Classifier Sandbox (Local DistilBERT Hybrid)
            </h3>
            <p className="text-xs text-zinc-600 font-sans mt-1">
              Deterministic routing domain classifier running locally before cloud dispatch.
            </p>
          </div>

          <div className="flex gap-2">
            <input
              type="text"
              value={intentInput}
              onChange={(e) => setIntentInput(e.target.value)}
              className="flex-1 rounded-xl border border-black/15 bg-zinc-50 px-4 py-2 text-xs font-sans text-black focus:outline-none focus:border-black focus:bg-white transition-all shadow-2xs"
            />
            <button
              onClick={handleClassify}
              disabled={classifying}
              className="rounded-xl bg-black px-4 py-2 text-xs font-mono font-bold text-white hover:bg-zinc-800 transition-all shadow-[0_4px_16px_rgba(0,0,0,0.18)] hover:scale-[1.02] active:scale-[0.98]"
            >
              {classifying ? "COMPUTING..." : "CLASSIFY"}
            </button>
          </div>

          {intentOutput && (
            <div className="rounded-xl bg-zinc-50 p-4 border border-black/10 text-xs font-mono space-y-1.5 text-zinc-800 shadow-2xs">
              <div>
                <span className="text-zinc-500 font-bold">PREDICTED INTENT:</span>{" "}
                <span className="text-black font-black">{intentOutput.intent}</span>
              </div>
              <div>
                <span className="text-zinc-500 font-bold">CONFIDENCE:</span> {(intentOutput.confidence * 100).toFixed(1)}%
              </div>
              <div>
                <span className="text-zinc-500 font-bold">INFERENCE LATENCY:</span> {intentOutput.latency_ms} ms
              </div>
              <div>
                <span className="text-zinc-500 font-bold">LLM FALLBACK REQUIRED:</span>{" "}
                <span className="text-black font-bold">
                  {intentOutput.requires_llm_fallback ? "YES (AMBIGUOUS INTENT)" : "NO (DETERMINISTIC ROUTE)"}
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Sandbox 2: Browser-Local WebGPU Execution */}
        <div className="rounded-2xl border border-black/10 bg-white/90 p-6 space-y-4 backdrop-blur-xl glass-panel shadow-sm">
          <div>
            <h3 className="text-xs font-mono uppercase tracking-[0.2em] font-bold text-black flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-black" />
              Browser-Local WebGPU Neural Runner
            </h3>
            <p className="text-xs text-zinc-600 font-sans mt-1">
              Client-side feature extraction & intent classification in your browser without transmitting raw text.
            </p>
          </div>

          <div className="flex gap-2">
            <input
              type="text"
              value={browserInput}
              onChange={(e) => setBrowserInput(e.target.value)}
              className="flex-1 rounded-xl border border-black/15 bg-zinc-50 px-4 py-2 text-xs font-sans text-black focus:outline-none focus:border-black focus:bg-white transition-all shadow-2xs"
            />
            <button
              onClick={handleBrowserInference}
              disabled={runningBrowser}
              className="rounded-xl bg-black px-4 py-2 text-xs font-mono font-bold text-white hover:bg-zinc-800 transition-all shadow-[0_4px_16px_rgba(0,0,0,0.18)] hover:scale-[1.02] active:scale-[0.98]"
            >
              {runningBrowser ? "COMPUTING..." : "RUN WEBGPU"}
            </button>
          </div>

          {browserResult && (
            <div className="rounded-xl bg-zinc-50 p-4 border border-black/10 text-xs font-mono space-y-1.5 text-zinc-800 shadow-2xs">
              <div>
                <span className="text-zinc-500 font-bold">HARDWARE BACKEND:</span>{" "}
                <span className="text-black font-black uppercase">{browserResult.backend}</span>
              </div>
              <div>
                <span className="text-zinc-500 font-bold">PREDICTED INTENT:</span> {browserResult.intent}
              </div>
              <div>
                <span className="text-zinc-500 font-bold">LOCAL LATENCY:</span> {browserResult.latencyMs} ms
              </div>
              <div>
                <span className="text-zinc-500 font-bold">VECTOR EMBEDDING PREVIEW:</span> [
                {browserResult.embedding.slice(0, 4).map((n: number) => n.toFixed(3)).join(", ")}...]
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
