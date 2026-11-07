"use client";

import { useEffect, useState } from "react";
import { Settings as SettingsIcon, ShieldCheck, Key, Database, Save, CheckCircle2 } from "lucide-react";
import { fetchSettings, updateSettings } from "@/lib/api";
import { SystemSettings, PrivacyMode } from "@/lib/types";

export default function SettingsPage() {
  const [settings, setSettings] = useState<SystemSettings | null>(null);
  const [apiKeyInput, setApiKeyInput] = useState<string>("");
  const [modelInput, setModelInput] = useState<string>("gemini-2.0-flash");
  const [privacyInput, setPrivacyInput] = useState<PrivacyMode>("hybrid");
  const [saving, setSaving] = useState<boolean>(false);
  const [saveSuccess, setSaveSuccess] = useState<boolean>(false);

  useEffect(() => {
    fetchSettings().then((data) => {
      setSettings(data);
      setModelInput(data.gemini_model || "gemini-2.0-flash");
      setPrivacyInput(data.default_privacy_mode || "hybrid");
    });
  }, []);

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);
    setSaveSuccess(false);

    try {
      const payload: any = {
        privacy_mode: privacyInput,
        gemini_model: modelInput,
      };
      if (apiKeyInput.trim()) {
        payload.gemini_api_key = apiKeyInput.trim();
      }

      const res = await updateSettings(payload);
      setSettings(res.settings);
      setApiKeyInput("");
      setSaveSuccess(true);
      setTimeout(() => setSaveSuccess(false), 3000);
    } catch (err: any) {
      alert(`Error updating settings: ${err.message}`);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="space-y-8 max-w-4xl mx-auto">
      {/* Header */}
      <div className="border-b border-black/10 pb-5">
        <div className="inline-flex items-center gap-2 rounded-full border border-black/10 bg-black/[0.04] px-3 py-0.5 text-[10px] font-mono uppercase tracking-[0.2em] text-zinc-700">
          <span className="h-1.5 w-1.5 rounded-full bg-black animate-pulse" />
          Enclave Controls
        </div>
        <h1 className="text-2xl md:text-3xl font-black font-mono tracking-tight text-black mt-1.5 flex items-center gap-2.5">
          <SettingsIcon className="h-6 w-6 text-black" />
          PLATFORM & PRIVACY CONFIGURATION
        </h1>
        <p className="text-xs text-zinc-600 font-sans mt-0.5">
          Configure cloud LLM credentials, default execution boundaries, and local encryption parameters.
        </p>
      </div>

      <form onSubmit={handleSave} className="space-y-6">
        {/* Privacy Modes Configuration */}
        <div className="rounded-2xl border border-black/10 bg-white/90 p-6 space-y-4 backdrop-blur-xl glass-panel shadow-sm">
          <h2 className="text-xs font-mono uppercase tracking-[0.2em] font-bold text-black flex items-center gap-2">
            <ShieldCheck className="h-4 w-4 text-black" />
            Default Privacy Execution Mode
          </h2>
          <p className="text-xs text-zinc-600 font-sans">
            Determines whether tasks are permitted to transmit minimized statistical schemas to cloud LLMs or must remain strictly on-device.
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {[
              {
                id: "local_only",
                title: "Local Only",
                desc: "Strictly zero external network calls. Offline deterministic reasoning and local Python tools.",
              },
              {
                id: "hybrid",
                title: "Hybrid (Standard)",
                desc: "Raw proprietary files stay local; only sanitized metadata & statistical schemas sent to cloud.",
              },
              {
                id: "cloud_enabled",
                title: "Cloud Enabled",
                desc: "Cloud reasoning via Gemini API enabled for general tasks with telemetry logging.",
              },
            ].map((opt) => (
              <label
                key={opt.id}
                className={`cursor-pointer rounded-xl border p-4 transition-all duration-200 block ${
                  privacyInput === opt.id
                    ? "border-black bg-black text-white shadow-md scale-[1.01]"
                    : "border-black/10 bg-zinc-50 hover:border-black/30 hover:bg-white text-zinc-900 shadow-2xs"
                }`}
              >
                <div className="flex items-center gap-2.5">
                  <input
                    type="radio"
                    name="privacy_mode"
                    value={opt.id}
                    checked={privacyInput === opt.id}
                    onChange={() => setPrivacyInput(opt.id as PrivacyMode)}
                    className="accent-black"
                  />
                  <span className={`text-xs font-mono font-bold uppercase ${privacyInput === opt.id ? "text-white" : "text-black"}`}>
                    {opt.title}
                  </span>
                </div>
                <p className={`text-[11px] mt-2 leading-relaxed font-sans ${privacyInput === opt.id ? "text-zinc-300" : "text-zinc-600"}`}>
                  {opt.desc}
                </p>
              </label>
            ))}
          </div>
        </div>

        {/* Cloud LLM Provider Credentials */}
        <div className="rounded-2xl border border-black/10 bg-white/90 p-6 space-y-4 backdrop-blur-xl glass-panel shadow-sm">
          <h2 className="text-xs font-mono uppercase tracking-[0.2em] font-bold text-black flex items-center gap-2">
            <Key className="h-4 w-4 text-black" />
            Cloud LLM Provider (Google Gemini API)
          </h2>
          <p className="text-xs text-zinc-600 font-sans">
            API keys are stored securely in environment variables or masked on disk. Keys are never exposed in frontend responses.
          </p>

          <div className="space-y-4 text-xs font-mono">
            <div>
              <label className="text-zinc-700 font-bold uppercase tracking-wider block mb-1.5">
                Gemini API Key
              </label>
              <input
                type="password"
                placeholder={settings?.gemini_configured ? `Configured (${settings.gemini_api_key_masked})` : "Enter Google Gemini API Key"}
                value={apiKeyInput}
                onChange={(e) => setApiKeyInput(e.target.value)}
                className="w-full rounded-xl border border-black/15 bg-zinc-50 px-4 py-2.5 text-black placeholder-zinc-400 focus:outline-none focus:border-black focus:bg-white shadow-2xs font-mono text-xs transition-all"
              />
              <span className="text-[10px] text-zinc-500 mt-1.5 block font-sans">
                Leave blank to preserve existing key. If no key is set, AURA operates via the deterministic test reasoning engine.
              </span>
            </div>

            <div>
              <label className="text-zinc-700 font-bold uppercase tracking-wider block mb-1.5">
                Gemini Model Architecture
              </label>
              <select
                value={modelInput}
                onChange={(e) => setModelInput(e.target.value)}
                className="w-full rounded-xl border border-black/15 bg-zinc-50 px-4 py-2.5 text-black focus:outline-none focus:border-black focus:bg-white shadow-2xs font-mono text-xs transition-all"
              >
                <option value="gemini-2.0-flash" className="bg-white text-black">gemini-2.0-flash (Fast, Low Latency, High Reasoning)</option>
                <option value="gemini-1.5-pro" className="bg-white text-black">gemini-1.5-pro (Complex Multi-Step DAG Planning)</option>
                <option value="gemini-1.5-flash" className="bg-white text-black">gemini-1.5-flash (Standard)</option>
              </select>
            </div>
          </div>
        </div>

        {/* Data Retention & Storage Policy */}
        <div className="rounded-2xl border border-black/10 bg-white/90 p-6 space-y-3 backdrop-blur-xl glass-panel shadow-sm">
          <h2 className="text-xs font-mono uppercase tracking-[0.2em] font-bold text-black flex items-center gap-2">
            <Database className="h-4 w-4 text-black" />
            Data Retention & Checkpoint Storage
          </h2>
          <div className="grid grid-cols-2 gap-4 text-xs font-mono">
            <div className="p-4 bg-zinc-50 rounded-xl border border-black/10">
              <span className="text-zinc-500 block uppercase tracking-wider text-[10px] font-bold">Database Storage:</span>
              <span className="text-black font-black mt-1 block">SQLite Checkpointer</span>
            </div>
            <div className="p-4 bg-zinc-50 rounded-xl border border-black/10">
              <span className="text-zinc-500 block uppercase tracking-wider text-[10px] font-bold">Retention Window:</span>
              <span className="text-black font-black mt-1 block">{settings?.data_retention_hours || 72} HOURS</span>
            </div>
          </div>
        </div>

        {/* Save Button */}
        <div className="flex items-center justify-between pt-2">
          {saveSuccess && (
            <div className="flex items-center gap-2 text-xs font-mono text-black font-bold">
              <CheckCircle2 className="h-4 w-4 text-black" />
              <span>SETTINGS PERSISTED SUCCESSFULLY</span>
            </div>
          )}
          <button
            type="submit"
            disabled={saving}
            className="ml-auto flex items-center gap-2 rounded-xl bg-black px-6 py-2.5 text-xs font-mono font-bold text-white hover:bg-zinc-800 transition-all shadow-[0_4px_16px_rgba(0,0,0,0.18)] hover:scale-[1.02] active:scale-[0.98]"
          >
            <Save className="h-4 w-4" />
            <span>{saving ? "SAVING..." : "SAVE SETTINGS"}</span>
          </button>
        </div>
      </form>
    </div>
  );
}
