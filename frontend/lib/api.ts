import { 
  TaskState, 
  TaskClarificationResponse,
  TaskSubmissionResponse,
  TaskListItem,
  TaskArtifactManifest,
  TaskArtifactContent,
  TaskCancellationResponse,
  ToolDefinition, 
  ModelMetadata, 
  SystemSettings, 
  PrivacyMode 
} from "./types";

export const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export async function fetchHealth() {
  const res = await fetch(`${API_BASE}/health`);
  return res.json();
}

export async function fetchTools(): Promise<ToolDefinition[]> {
  const res = await fetch(`${API_BASE}/api/tools`);
  return res.json();
}

export async function executeToolDirect(tool_name: string, args: Record<string, any>) {
  const res = await fetch(`${API_BASE}/api/tools/execute`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tool_name, arguments: args, user_approval_granted: true }),
  });
  return res.json();
}

export async function fetchModels(): Promise<ModelMetadata[]> {
  const res = await fetch(`${API_BASE}/api/models`);
  return res.json();
}

export async function classifyIntent(text: string) {
  const res = await fetch(`${API_BASE}/api/models/classify-intent`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  return res.json();
}

export async function generateEmbedding(text: string) {
  const res = await fetch(`${API_BASE}/api/models/embed`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  return res.json();
}

export async function fetchTasks(limit = 25): Promise<TaskListItem[]> {
  const res = await fetch(`${API_BASE}/api/tasks?limit=${limit}`);
  return res.json();
}

export async function fetchTask(task_id: string): Promise<TaskState> {
  const res = await fetch(`${API_BASE}/api/tasks/${task_id}`);
  return res.json();
}

export async function fetchTaskArtifacts(task_id: string): Promise<TaskArtifactManifest> {
  const res = await fetch(`${API_BASE}/api/tasks/${task_id}/artifacts`);
  if (!res.ok) throw new Error(`Task artifacts are unavailable (${res.status}).`);
  return res.json();
}

export async function fetchTaskMarkdown(task_id: string): Promise<TaskArtifactContent> {
  const res = await fetch(`${API_BASE}/api/tasks/${task_id}/artifacts/markdown`);
  if (!res.ok) throw new Error(`Task report is unavailable (${res.status}).`);
  return res.json();
}

export async function cancelTask(task_id: string): Promise<TaskCancellationResponse> {
  const res = await fetch(`${API_BASE}/api/tasks/${task_id}/cancel`, { method: "POST" });
  if (!res.ok) throw new Error(`Task cancellation failed (${res.status}).`);
  return res.json();
}

export async function createTask(
  objective: string,
  file_path?: string,
  privacy_mode: PrivacyMode = "hybrid"
): Promise<TaskState | TaskClarificationResponse> {
  const res = await fetch(`${API_BASE}/api/tasks`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ objective, file_path, privacy_mode }),
  });
  const result = (await res.json()) as TaskSubmissionResponse;
  if ("outcome" in result) {
    if (result.outcome === "clarification_required") {
      return result;
    }
    throw new Error(result.message);
  }
  if (!res.ok) throw new Error(`Task submission failed (${res.status}).`);
  return result;
}

export async function uploadDatasetFile(file: File, encryptAtRest = false) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("encrypt_at_rest", String(encryptAtRest));

  const res = await fetch(`${API_BASE}/api/files/upload`, {
    method: "POST",
    body: formData,
  });
  return res.json();
}

export async function fetchSettings(): Promise<SystemSettings> {
  const res = await fetch(`${API_BASE}/api/settings`);
  return res.json();
}

export async function updateSettings(data: {
  privacy_mode?: PrivacyMode;
  gemini_api_key?: string;
  gemini_model?: string;
  default_llm_provider?: string;
}) {
  const res = await fetch(`${API_BASE}/api/settings`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  return res.json();
}

