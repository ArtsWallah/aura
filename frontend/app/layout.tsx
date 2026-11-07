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
    <html lang="en">
      <body className="bg-white text-black min-h-screen flex flex-col font-sans antialiased selection:bg-black selection:text-white relative">
        {/* Crisp dynamic background depth */}
        <div className="fixed inset-0 bg-grid-pattern pointer-events-none z-0 opacity-70" />
        <div className="fixed -top-32 left-1/4 w-[30rem] h-[30rem] bg-black/[0.04] rounded-full blur-[140px] pointer-events-none animate-orb z-0" />
        <div className="fixed bottom-0 right-1/4 w-[36rem] h-[36rem] bg-black/[0.03] rounded-full blur-[160px] pointer-events-none animate-orb z-0" />

        <div className="relative z-10 flex flex-col min-h-screen">
          <Navbar />
          <div className="flex flex-1 overflow-hidden">
            <Sidebar />
            <main className="flex-1 overflow-y-auto p-6 lg:p-8 bg-transparent">
              {children}
            </main>
          </div>
        </div>
      </body>
    </html>
  );
}
