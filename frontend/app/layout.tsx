import type { Metadata } from "next";
import "./globals.css";
import Navbar from "@/components/Navbar";
import Sidebar from "@/components/Sidebar";
import { StartupProvider } from "@/components/StartupLoader";
import GlobalBackground from "@/components/GlobalBackground";

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
        <StartupProvider>
          {/* Global animated video background */}
          <GlobalBackground />

          <div className="relative z-10 flex flex-col min-h-screen">
            <Navbar />
            <div className="flex flex-1 overflow-hidden">
              <Sidebar />
              <main className="flex-1 overflow-y-auto p-6 lg:p-8 bg-transparent">
                {children}
              </main>
            </div>
          </div>
        </StartupProvider>
      </body>
    </html>
  );
}
