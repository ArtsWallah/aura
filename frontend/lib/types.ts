export type PrivacyMode = "local_only" | "hybrid" | "cloud_enabled";

export type TaskStatus = 
  | "queued"
  | "planning" 
  | "running"
  | "waiting"
  | "retrying"
  | "cancel_requested"
  | "completed" 
  | "failed" 
  | "cancelled";

export type StepStatus = "pending" | "in_progress" | "completed" | "failed" | "skipped" | "interrupted";
export type IntentCategory =
  | "data_analysis" | "statistical_analysis" | "visualization" | "sql_query"
  | "document_analysis" | "web_research" | "report_generation" | "multi_step_analysis"
  | "system_health" | "unknown";
export type ProviderIdentity = "gemini" | "local_deterministic_planner" | "test_mock_provider";
export type TraceEvaluationStatus =
  | "plan_valid" | "incomplete_coverage" | "invalid_plan" | "execution_failed"
  | "partial_success" | "succeeded" | "inconclusive";
export type TraceFailureCategory =
  | "none" | "tool_failure" | "dependency_blocked" | "cancelled" | "interrupted" | "unknown";

export interface TraceArtifact {
  artifact_type: "chart_png" | "report_html" | "report_markdown";
  available: boolean;
  count: number;
}

export interface TraceStep {
  step_id: string;
  tool_name: string;
  status: StepStatus;
  tool_label: string | null;
  tool_category: string | null;
  retry_count: number;
  retry_limit: number;
  failure_category: TraceFailureCategory | null;
  artifacts: TraceArtifact[];
  duration_ms: number;
}

export interface ExecutionTrace {
  intent_category: IntentCategory;
  intent_confidence: number | null;
  provider_identity: ProviderIdentity;
  selected_tools: string[];
  step_statuses: TraceStep[];
  validation_outcome: "accepted" | "rejected";
  evaluation_status: TraceEvaluationStatus | null;
  timing: { planning_ms: number; validation_ms: number; execution_ms: number };
}

export interface ExecutionStep {
  step_id: string;
  title: string;
  description: string;
  capability: string;
  tool_name: string;
  parameters: Record<string, any>;
  dependencies: string[];
  output?: any;
  error?: string | null;
  retry_count: number;
  max_retries: number;
  status: StepStatus;
  duration_ms: number;
}

export interface ExecutionPlan {
  plan_id: string;
  goal: string;
  steps: ExecutionStep[];
  created_at: number;
}

export interface TaskMetrics {
  total_execution_time_ms: number;
  tool_calls_count: number;
  llm_calls_count: number;
  prompt_tokens: number;
  completion_tokens: number;
  retries_count: number;
  success_rate: number;
}

export interface TaskState {
  task_id: string;
  privacy_mode: PrivacyMode;
  status: TaskStatus;
  created_at: number;
  updated_at: number;
  execution_trace?: ExecutionTrace | null;
}

export interface TaskClarificationResponse {
  outcome: "clarification_required";
  question: string;
  required_input: "dataset" | "document" | "database" | "research_topic" | "analysis_goal" | "clarification" | null;
}

export interface TaskRejectionResponse {
  outcome: "rejected";
  code: "invalid_request" | "unsupported_request" | "provider_unavailable" | "planning_failed";
  message: string;
}

export type TaskSubmissionResponse =
  | TaskState
  | TaskClarificationResponse
  | TaskRejectionResponse;

export interface TaskArtifactManifest {
  task_id: string;
  title: string;
  html_url?: string | null;
  markdown_url?: string | null;
  chart_urls: string[];
}

export interface TaskArtifactContent {
  task_id: string;
  artifact_type: "markdown";
  content: string;
}

// Task lists are intentionally summary-only; fetchTask returns the trace detail.
export type TaskListItem = Omit<TaskState, "execution_trace">;

export interface TaskCancellationResponse {
  status: TaskStatus;
  message: string;
}

export interface ToolDefinition {
  name: string;
  category: string;
  description: string;
  input_schema: Record<string, any>;
  output_schema: Record<string, any>;
  execution_location: string;
  permission_level: string;
  requires_approval: boolean;
  retry_safe: boolean;
  timeout_seconds: number;
  version: string;
}

export interface ModelMetadata {
  model_id: string;
  name: string;
  category: string;
  execution_location: string;
  parameters_size: string;
  is_loaded: boolean;
  average_latency_ms: number;
  total_inferences: number;
  description: string;
}

export interface TaskEvent {
  event_type: string;
  task_id: string;
  timestamp: number;
  step_id?: string | null;
  message: string;
  data: Record<string, any>;
}

export interface SystemSettings {
  project_name: string;
  version: string;
  environment: string;
  default_privacy_mode: PrivacyMode;
  default_llm_provider: string;
  gemini_configured: boolean;
  gemini_api_key_masked: string;
  gemini_model: string;
  data_retention_hours: number;
  available_providers: Array<{
    id: string;
    name: string;
    available: boolean;
    model: string;
    is_cloud: boolean;
    requires_api_key: boolean;
  }>;
}

