"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { 
  LayoutDashboard, 
  Bot, 
  SearchCode, 
  Cpu, 
  Wrench, 
  Settings,
  Shield,
  Layers
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

  return (
    <aside className="w-64 border-r border-slate-800 bg-slate-950/60 p-4 flex flex-col justify-between shrink-0">
      <div className="space-y-1">
        <p className="px-3 pb-2 text-[11px] font-semibold uppercase tracking-wider text-slate-500 font-mono">
          Platform Navigation
        </p>
        {NAV_ITEMS.map((item) => {
          const isActive = pathname === item.href;
          const Icon = item.icon;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-all ${
                isActive
                  ? "bg-indigo-600 text-white shadow-md shadow-indigo-600/30"
                  : "text-slate-400 hover:bg-slate-900 hover:text-slate-200"
              }`}
            >
              <Icon className="h-4 w-4 shrink-0" />
              <span>{item.name}</span>
            </Link>
          );
        })}
      </div>

      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-3.5 space-y-2">
        <div className="flex items-center gap-2 text-xs font-semibold text-slate-200">
          <Shield className="h-4 w-4 text-emerald-400" />
          <span>Deterministic Core</span>
        </div>
        <p className="text-[11px] text-slate-400 leading-relaxed">
          Reasoning routed to LLMs; execution isolated in local Python tools & WebGPU models.
        </p>
      </div>
    </aside>
  );
}
