import type { ExecutionTrace } from "./types";

export function traceFromTaskRefresh(task: { execution_trace?: ExecutionTrace | null }): ExecutionTrace | null;
