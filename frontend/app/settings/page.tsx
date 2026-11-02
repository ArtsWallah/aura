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
      <div className="border-b border-slate-800 pb-4">
        <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
          <SettingsIcon className="h-6 w-6 text-indigo-400" />
          Platform Configuration & Privacy Policies
        </h1>
        <p className="text-xs text-slate-400 mt-1">
          Configure cloud LLM credentials, default execution boundaries, and local encryption parameters.
        </p>
      </div>

      <form onSubmit={handleSave} className="space-y-6">
        {/* Privacy Modes Configuration */}
        <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-6 space-y-4">
          <h2 className="text-sm font-semibold text-white flex items-center gap-2">
            <ShieldCheck className="h-4 w-4 text-emerald-400" />
            Default Privacy Execution Mode
          </h2>
          <p className="text-xs text-slate-400">
            Determines whether tasks are allowed to transmit minimized context to cloud LLMs or must remain strictly on-device.
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
                title: "Hybrid (Recommended)",
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
                className={`cursor-pointer rounded-xl border p-4 transition-all block ${
                  privacyInput === opt.id
                    ? "border-indigo-500 bg-indigo-950/20 shadow-md"
                    : "border-slate-800 bg-slate-950 hover:bg-slate-900"
                }`}
              >
                <div className="flex items-center gap-2">
                  <input
                    type="radio"
                    name="privacy_mode"
                    value={opt.id}
                    checked={privacyInput === opt.id}
                    onChange={() => setPrivacyInput(opt.id as PrivacyMode)}
                    className="text-indigo-600 focus:ring-indigo-500"
                  />
                  <span className="text-xs font-bold text-white">{opt.title}</span>
                </div>
                <p className="text-[11px] text-slate-400 mt-2 leading-relaxed">{opt.desc}</p>
              </label>
            ))}
          </div>
        </div>

        {/* Cloud LLM Provider Credentials */}
        <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-6 space-y-4">
          <h2 className="text-sm font-semibold text-white flex items-center gap-2">
            <Key className="h-4 w-4 text-indigo-400" />
            Cloud LLM Provider (Google Gemini API)
          </h2>
          <p className="text-xs text-slate-400">
            API keys are stored securely in environment variables or masked on disk. Keys are never exposed in frontend responses.
          </p>

          <div className="space-y-4 text-xs">
            <div>
              <label className="text-slate-300 font-semibold block mb-1">
                Gemini API Key
              </label>
              <input
                type="password"
                placeholder={settings?.gemini_configured ? `Configured (${settings.gemini_api_key_masked})` : "Enter Google Gemini API Key"}
                value={apiKeyInput}
                onChange={(e) => setApiKeyInput(e.target.value)}
                className="w-full rounded-lg border border-slate-800 bg-slate-950 px-3 py-2 text-white placeholder-slate-500 focus:outline-none focus:border-indigo-500 font-mono"
              />
              <span className="text-[11px] text-slate-500 mt-1 block">
                Leave blank to keep existing key. If no key is set, AURA operates via the deterministic test reasoning engine.
              </span>
            </div>

            <div>
              <label className="text-slate-300 font-semibold block mb-1">
                Gemini Model Architecture
              </label>
              <select
                value={modelInput}
                onChange={(e) => setModelInput(e.target.value)}
                className="w-full rounded-lg border border-slate-800 bg-slate-950 px-3 py-2 text-white focus:outline-none focus:border-indigo-500 font-mono"
              >
                <option value="gemini-2.0-flash">gemini-2.0-flash (Fast, Low Latency, High Reasoning)</option>
                <option value="gemini-1.5-pro">gemini-1.5-pro (Complex Multi-Step DAG Planning)</option>
                <option value="gemini-1.5-flash">gemini-1.5-flash (Standard)</option>
              </select>
            </div>
          </div>
        </div>

        {/* Data Retention & Storage Policy */}
        <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-6 space-y-3">
          <h2 className="text-sm font-semibold text-white flex items-center gap-2">
            <Database className="h-4 w-4 text-blue-400" />
            Data Retention & Checkpoint Storage
          </h2>
          <div className="grid grid-cols-2 gap-4 text-xs font-mono">
            <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
              <span className="text-slate-500 block">Database:</span>
              <span className="text-slate-200">SQLite Checkpointer</span>
            </div>
            <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
              <span className="text-slate-500 block">Retention Window:</span>
              <span className="text-slate-200">{settings?.data_retention_hours || 72} Hours</span>
            </div>
          </div>
        </div>

        {/* Save Button */}
        <div className="flex items-center justify-between pt-2">
          {saveSuccess && (
            <div className="flex items-center gap-2 text-xs text-emerald-400 font-medium">
              <CheckCircle2 className="h-4 w-4" />
              <span>Settings updated successfully!</span>
            </div>
          )}
          <button
            type="submit"
            disabled={saving}
            className="ml-auto flex items-center gap-2 rounded-lg bg-indigo-600 px-5 py-2.5 text-xs font-bold text-white hover:bg-indigo-500 shadow-lg shadow-indigo-600/20"
          >
            <Save className="h-4 w-4" />
            <span>{saving ? "Saving..." : "Save Settings"}</span>
          </button>
        </div>
      </form>
    </div>
  );
}
