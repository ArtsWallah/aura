"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { ShieldCheck, Activity, Terminal } from "lucide-react";
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
    <header className="sticky top-0 z-50 flex h-16 w-full items-center justify-between border-b border-black/10 bg-white/85 px-6 backdrop-blur-xl shadow-xs">
      <div className="flex items-center gap-3">
        <Link href="/" className="group flex items-center gap-3 transition-transform duration-300 hover:scale-[1.01]">
          <div className="relative flex h-9 w-9 items-center justify-center rounded-lg bg-black font-black text-white shadow-[0_4px_16px_rgba(0,0,0,0.18)] transition-all duration-300 group-hover:shadow-[0_4px_22px_rgba(0,0,0,0.28)]">
            <span className="text-sm font-black tracking-tighter">A</span>
            <div className="absolute inset-0 rounded-lg border border-black/20" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-lg font-black tracking-tight text-black font-mono">AURA</span>
              <span className="rounded border border-black/20 bg-black/5 px-1.5 py-0.5 text-[9px] font-mono font-bold tracking-wider text-black">
                v1.0.0
              </span>
            </div>
            <p className="text-[9px] font-mono tracking-[0.25em] text-zinc-500 uppercase font-semibold">
              Autonomous Reasoning & Analytics
            </p>
          </div>
        </Link>
      </div>

      <div className="flex items-center gap-3">
        {/* Privacy Mode Badge */}
        <div className="flex items-center gap-1.5 rounded-full border border-black/15 bg-zinc-50 px-3.5 py-1 text-xs font-mono font-medium text-zinc-900 backdrop-blur-md shadow-xs transition-all hover:border-black/30">
          <ShieldCheck className="h-3.5 w-3.5 text-black" />
          <span className="text-zinc-500">MODE:</span>
          <span className="text-black font-bold">{health?.default_privacy_mode?.toUpperCase() || "HYBRID"}</span>
        </div>

        {/* Backend Connectivity Status */}
        <div className="flex items-center gap-2.5 rounded-full border border-black/10 bg-white px-3.5 py-1 text-xs font-mono text-zinc-800 backdrop-blur-md shadow-xs transition-all hover:border-black/25">
          <div className="relative flex h-2 w-2 items-center justify-center">
            {connected && (
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-black opacity-30" />
            )}
            <span
              className={`relative inline-flex h-1.5 w-1.5 rounded-full ${
                connected ? "bg-black shadow-[0_0_8px_rgba(0,0,0,0.6)]" : "bg-zinc-400"
              }`}
            />
          </div>
          <span className="tracking-wide font-semibold">
            {connected ? "LIVE (8000)" : "CONNECTING..."}
          </span>
        </div>
      </div>
    </header>
  );
}
