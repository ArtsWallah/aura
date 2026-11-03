import assert from "node:assert/strict";
import test from "node:test";
import { traceFromTaskRefresh } from "../lib/trace-refresh.mjs";

test("refresh clears a previous trace when the task detail returns null", () => {
  const previousTrace = { selected_tools: ["system_health"] };
  assert.equal(traceFromTaskRefresh({ execution_trace: null }), null);
  assert.deepEqual(traceFromTaskRefresh({ execution_trace: previousTrace }), previousTrace);
});
