import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";

test("startup loader asset exists in public directory and has valid size", () => {
  const assetPath = path.resolve("public/aura-startup.mp4");
  assert.ok(fs.existsSync(assetPath), "public/aura-startup.mp4 must exist");
  const stats = fs.statSync(assetPath);
  assert.ok(stats.size > 1_000_000, `Video asset size (${stats.size}) should be > 1MB`);
});

test("StartupLoader component implements all required visual and behavioral attributes", () => {
  const componentPath = path.resolve("components/StartupLoader.tsx");
  assert.ok(fs.existsSync(componentPath), "StartupLoader.tsx must exist");
  const content = fs.readFileSync(componentPath, "utf-8");

  // Visual requirements
  assert.ok(content.includes("bg-white"), "Must use white background");
  assert.ok(
    content.includes("object-cover") || content.includes("object-contain"),
    "Must specify full-screen object fit (object-cover or object-contain)"
  );
  assert.ok(content.includes("INITIALIZING AURA..."), "Must include exact label 'INITIALIZING AURA...'");

  // Video playback attributes
  assert.ok(content.includes("autoPlay"), "Must specify autoPlay");
  assert.ok(content.includes("muted"), "Must specify muted");
  assert.ok(
    content.includes("handleVideoEnded") || content.includes("onEnded"),
    "Must trigger fade away when video ends"
  );
  assert.ok(content.includes("playsInline"), "Must specify playsInline");

  // Accessibility and resilience
  assert.ok(content.includes("prefers-reduced-motion"), "Must handle prefers-reduced-motion");
  assert.ok(content.includes("playbackFailed"), "Must handle video playback failure");
  assert.ok(content.includes("sessionStorage"), "Must check session storage to avoid re-triggering on normal navigation");
  assert.ok(content.includes("safetyTimer"), "Must include a safety timeout so loader never remains stuck");
});

test("RootLayout includes StartupProvider and Dashboard integrates readiness signal", () => {
  const layoutPath = path.resolve("app/layout.tsx");
  const layoutContent = fs.readFileSync(layoutPath, "utf-8");
  assert.ok(layoutContent.includes("<StartupProvider>"), "RootLayout must wrap with StartupProvider");

  const dashboardPath = path.resolve("app/page.tsx");
  const dashboardContent = fs.readFileSync(dashboardPath, "utf-8");
  assert.ok(dashboardContent.includes("useStartup"), "Dashboard must use useStartup hook");
  assert.ok(dashboardContent.includes("markStartupReady"), "Dashboard must trigger markStartupReady when initial load completes");
});

