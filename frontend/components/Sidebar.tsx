"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { 
  LayoutDashboard, 
  Bot, 
  SearchCode, 
  Cpu, 
  Wrench, 
  Settings,
  Shield,
  PanelLeftClose,
  PanelLeftOpen
} from "lucide-react";

const NAV_ITEMS = [
  { name: "Dashboard", href: "/", icon: LayoutDashboard },
  { name: "Agent Workspace", href: "/workspace", icon: Bot },
  { name: "Execution Inspector", href: "/inspector", icon: SearchCode },
  { name: "Model Registry", href: "/models", icon: Cpu },
  { name: "Tool Registry", href: "/tools", icon: Wrench },
  { name: "Settings", href: "/settings", icon: Settings },
];

export default function Sidebar() {
  const pathname = usePathname();
  const [isCollapsed, setIsCollapsed] = useState(false);

  useEffect(() => {
    try {
      const saved = localStorage.getItem("aura_sidebar_collapsed");
      if (saved !== null) {
        setIsCollapsed(saved === "true");
      }
    } catch {
      // storage unavailable in sandbox/private mode
    }
  }, []);

  const toggleSidebar = () => {
    setIsCollapsed((prev) => {
      const next = !prev;
      try {
        localStorage.setItem("aura_sidebar_collapsed", String(next));
      } catch {
        // storage error handling
      }
      return next;
    });
  };

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "b") {
        const target = e.target as HTMLElement;
        if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA")) return;
        e.preventDefault();
        toggleSidebar();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  return (
    <aside
      className={`border-r border-black/10 bg-white/75 backdrop-blur-2xl flex flex-col justify-between shrink-0 z-20 shadow-xs transition-all duration-300 ease-in-out ${
        isCollapsed ? "w-[72px] px-2.5 py-4" : "w-64 p-4"
      }`}
    >
      <div className="space-y-2">
        {/* Header / Toggle Button */}
        <div className={`flex items-center ${isCollapsed ? "justify-center" : "justify-between px-2"} pb-1`}>
          {!isCollapsed && (
            <span className="text-[10px] font-mono font-bold uppercase tracking-[0.22em] text-zinc-500">
              Navigation
            </span>
          )}
          <button
            onClick={toggleSidebar}
            type="button"
            aria-label={isCollapsed ? "Expand sidebar (Ctrl+B)" : "Collapse sidebar (Ctrl+B)"}
            title={isCollapsed ? "Expand sidebar (Ctrl+B)" : "Collapse sidebar (Ctrl+B)"}
            className="flex h-7 w-7 items-center justify-center rounded-lg border border-black/10 bg-black/5 text-zinc-700 transition-all hover:bg-black hover:text-white hover:border-black active:scale-95"
          >
            {isCollapsed ? (
              <PanelLeftOpen className="h-4 w-4" />
            ) : (
              <PanelLeftClose className="h-4 w-4" />
            )}
          </button>
        </div>

        {/* Navigation items */}
        <nav className="space-y-1">
          {NAV_ITEMS.map((item) => {
            const isActive = pathname === item.href;
            const Icon = item.icon;
            return (
              <Link
                key={item.href}
                href={item.href}
                title={isCollapsed ? item.name : undefined}
                className={`group flex items-center rounded-xl transition-all duration-200 ${
                  isCollapsed ? "justify-center p-2.5" : "gap-3 px-3.5 py-2.5"
                } text-sm font-medium ${
                  isActive
                    ? "bg-black text-white font-semibold shadow-[0_4px_20px_rgba(0,0,0,0.2)] scale-[1.01]"
                    : "text-zinc-600 hover:bg-zinc-100 hover:text-black border border-transparent hover:border-black/5"
                }`}
              >
                <Icon
                  className={`h-4 w-4 shrink-0 transition-transform duration-200 group-hover:scale-110 ${
                    isActive ? "text-white" : "text-zinc-500 group-hover:text-black"
                  }`}
                />
                {!isCollapsed && (
                  <>
                    <span className="tracking-tight truncate">{item.name}</span>
                    {isActive && (
                      <span className="ml-auto h-1.5 w-1.5 rounded-full bg-white shadow-[0_0_4px_rgba(255,255,255,0.8)]" />
                    )}
                  </>
                )}
              </Link>
            );
          })}
        </nav>
      </div>

      {/* Footer Info Card */}
      {isCollapsed ? (
        <div
          title="Deterministic Core — Reasoning routed to LLMs; execution isolated in typed local Python tools and WebGPU models."
          className="flex items-center justify-center h-10 w-10 mx-auto rounded-xl border border-black/10 bg-zinc-50/90 hover:border-black/25 shadow-xs transition-colors cursor-help"
        >
          <Shield className="h-4 w-4 text-black" />
        </div>
      ) : (
        <div className="relative overflow-hidden rounded-2xl border border-black/10 bg-zinc-50/90 p-4 space-y-2 backdrop-blur-md transition-all hover:border-black/20 shadow-xs">
          <div className="absolute -top-10 -right-10 h-20 w-20 bg-black/[0.03] rounded-full blur-xl pointer-events-none" />
          <div className="flex items-center gap-2 text-xs font-mono font-bold text-black">
            <Shield className="h-4 w-4 text-black" />
            <span>DETERMINISTIC CORE</span>
          </div>
          <p className="text-[11px] text-zinc-600 leading-relaxed font-sans">
            Reasoning routed to LLMs; execution isolated in typed local Python tools and WebGPU models.
          </p>
        </div>
      )}
    </aside>
  );
}
