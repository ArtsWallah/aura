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
  objective: string;
  privacy_mode: PrivacyMode;
  status: TaskStatus;
  plan?: ExecutionPlan | null;
  step_results: Record<string, any>;
  final_deliverable?: any;
  artifacts: string[];
  metrics: TaskMetrics;
  error_summary?: string | null;
  recovery_required?: boolean;
  recovery_reason?: string | null;
  created_at: number;
  updated_at: number;
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

