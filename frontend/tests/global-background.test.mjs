import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";

test("global background asset exists in public directory and has valid size", () => {
  const assetPath = path.resolve("public/aura-background.mp4");
  assert.ok(fs.existsSync(assetPath), "public/aura-background.mp4 must exist");
  const stats = fs.statSync(assetPath);
  assert.ok(stats.size > 1_000_000, `Background video asset size (${stats.size}) should be > 1MB`);

  // Ensure startup loader asset is preserved and untouched
  const startupPath = path.resolve("public/aura-startup.mp4");
  assert.ok(fs.existsSync(startupPath), "public/aura-startup.mp4 must be preserved");
  const startupStats = fs.statSync(startupPath);
  assert.ok(startupStats.size > 1_000_000, "Startup video asset must remain valid");
});

test("GlobalBackground component implements all required visual, behavioral, and accessibility attributes", () => {
  const componentPath = path.resolve("components/GlobalBackground.tsx");
  assert.ok(fs.existsSync(componentPath), "GlobalBackground.tsx must exist");
  const content = fs.readFileSync(componentPath, "utf-8");

  // Fixed, viewport-covering, non-interactive layer
  assert.ok(content.includes("fixed inset-0"), "Must be a fixed viewport-covering layer");
  assert.ok(content.includes("pointer-events-none"), "Must prevent intercepting pointer events");
  assert.ok(content.includes("z-0"), "Must stay behind application content (z-0)");
  assert.ok(content.includes("object-cover"), "Must use object-cover");

  // Subtle opacity in 15-25% range
  assert.ok(
    content.includes("opacity-20") || content.includes("opacity-[0.18]") || content.includes("opacity-[0.2]"),
    "Must use subtle opacity (15-25%)"
  );

  // Video playback attributes
  assert.ok(content.includes("autoPlay"), "Must specify autoPlay");
  assert.ok(content.includes("muted"), "Must specify muted");
  assert.ok(content.includes("loop"), "Must specify loop");
  assert.ok(content.includes("playsInline"), "Must specify playsInline");
  assert.ok(content.includes("/aura-background.mp4"), "Must reference /aura-background.mp4");

  // Accessibility and resilience
  assert.ok(content.includes("prefers-reduced-motion"), "Must handle prefers-reduced-motion");
  assert.ok(content.includes("playbackFailed"), "Must handle video playback failure with static fallback");
});

test("RootLayout integrates GlobalBackground and replaces legacy white grid background", () => {
  const layoutPath = path.resolve("app/layout.tsx");
  const layoutContent = fs.readFileSync(layoutPath, "utf-8");

  assert.ok(layoutContent.includes("GlobalBackground"), "RootLayout must import and render GlobalBackground");
  assert.ok(layoutContent.includes("<GlobalBackground />"), "RootLayout must render <GlobalBackground />");
  assert.ok(!layoutContent.includes("bg-grid-pattern"), "Legacy bg-grid-pattern must be removed from layout");
});

