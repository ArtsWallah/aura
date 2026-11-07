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
  Shield
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
    <aside className="w-64 border-r border-black/10 bg-white/75 backdrop-blur-2xl p-4 flex flex-col justify-between shrink-0 z-20 shadow-xs">
      <div className="space-y-1.5">
        <p className="px-3 pb-2 text-[10px] font-mono font-bold uppercase tracking-[0.22em] text-zinc-500">
          Navigation
        </p>
        <div className="space-y-1">
          {NAV_ITEMS.map((item) => {
            const isActive = pathname === item.href;
            const Icon = item.icon;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`group flex items-center gap-3 rounded-xl px-3.5 py-2.5 text-sm font-medium transition-all duration-200 ${
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
                <span className="tracking-tight">{item.name}</span>
                {isActive && (
                  <span className="ml-auto h-1.5 w-1.5 rounded-full bg-white shadow-[0_0_4px_rgba(255,255,255,0.8)]" />
                )}
              </Link>
            );
          })}
        </div>
      </div>

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
    </aside>
  );
}
