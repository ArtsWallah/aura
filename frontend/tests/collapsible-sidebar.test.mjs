import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";

test("Sidebar component implements collapsible layout and accessibility features", () => {
  const sidebarPath = path.resolve("components/Sidebar.tsx");
  assert.ok(fs.existsSync(sidebarPath), "components/Sidebar.tsx must exist");
  const content = fs.readFileSync(sidebarPath, "utf-8");

  // State and toggle functionality
  assert.ok(content.includes("isCollapsed"), "Must have isCollapsed state");
  assert.ok(content.includes("toggleSidebar"), "Must have toggleSidebar handler");
  assert.ok(content.includes("PanelLeftClose"), "Must import PanelLeftClose icon");
  assert.ok(content.includes("PanelLeftOpen"), "Must import PanelLeftOpen icon");

  // Smooth layout widths
  assert.ok(content.includes("w-64"), "Must support expanded w-64 width");
  assert.ok(content.includes("w-[72px]"), "Must support collapsed w-[72px] width");
  assert.ok(content.includes("transition-all"), "Must animate width with smooth transitions");

  // Accessibility and tooltips
  assert.ok(content.includes("title={isCollapsed ? item.name : undefined}"), "Must provide title tooltips in collapsed state");
  assert.ok(content.includes("aria-label"), "Must have aria-label on collapse toggle button");

  // Persistence and shortcuts
  assert.ok(content.includes("aura_sidebar_collapsed"), "Must persist collapsed state in localStorage");
  assert.ok(content.includes("ctrlKey") || content.includes("metaKey"), "Must support keyboard shortcut toggle");
});

