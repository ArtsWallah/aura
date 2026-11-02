import type { Metadata } from "next";
import "./globals.css";
import Navbar from "@/components/Navbar";
import Sidebar from "@/components/Sidebar";

export const metadata: Metadata = {
  title: "AURA — Autonomous Understanding, Reasoning & Analytics",
  description: "Enterprise privacy-aware autonomous AI agent platform with specialized small models and typed tools.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="bg-slate-950 text-slate-100 min-h-screen flex flex-col font-sans antialiased selection:bg-indigo-500 selection:text-white">
        <Navbar />
        <div className="flex flex-1 overflow-hidden">
          <Sidebar />
          <main className="flex-1 overflow-y-auto p-6 bg-gradient-to-b from-slate-950 via-slate-900/40 to-slate-950">
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}
