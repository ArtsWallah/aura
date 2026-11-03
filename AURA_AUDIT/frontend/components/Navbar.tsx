"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { ShieldCheck, Cpu, Activity, AlertCircle } from "lucide-react";
import { fetchHealth } from "@/lib/api";

export default function Navbar() {
  const [health, setHealth] = useState<any>(null);
  const [connected, setConnected] = useState<boolean>(false);

  useEffect(() => {
    fetchHealth()
      .then((data) => {
        setHealth(data);
        setConnected(true);
      })
      .catch(() => setConnected(false));
  }, []);

  return (
    <header className="sticky top-0 z-50 flex h-16 w-full items-center justify-between border-b border-slate-800 bg-slate-950/80 px-6 backdrop-blur-md">
      <div className="flex items-center gap-3">
        <Link href="/" className="flex items-center gap-2.5">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-tr from-blue-600 via-indigo-500 to-cyan-400 font-black text-white shadow-lg shadow-indigo-500/20">
            A
          </div>
          <div>
            <div className="text-lg font-bold tracking-tight text-white flex items-center gap-2">
              AURA
              <span className="rounded bg-indigo-500/20 px-1.5 py-0.5 text-[10px] font-semibold text-indigo-300 border border-indigo-500/30">
                v1.0.0
              </span>
            </div>
            <p className="text-[10px] text-slate-400 font-mono tracking-wider">
              AUTONOMOUS REASONING & ANALYTICS
            </p>
          </div>
        </Link>
      </div>

      <div className="flex items-center gap-4">
        {/* Privacy Mode Badge */}
        <div className="flex items-center gap-1.5 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-3 py-1 text-xs font-medium text-emerald-400">
          <ShieldCheck className="h-3.5 w-3.5" />
          <span>Privacy Mode: {health?.default_privacy_mode?.toUpperCase() || "HYBRID"}</span>
        </div>

        {/* Backend Connectivity Status */}
        <div className="flex items-center gap-2 rounded-full border border-slate-800 bg-slate-900/80 px-3 py-1 text-xs text-slate-300">
          <div
            className={`h-2 w-2 rounded-full ${
              connected ? "bg-emerald-400 shadow-sm shadow-emerald-400 animate-pulse" : "bg-red-400"
            }`}
          />
          <span>{connected ? "Backend Connected (8000)" : "Connecting..."}</span>
        </div>
      </div>
    </header>
  );
}

