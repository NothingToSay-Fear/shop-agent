// API 响应类型由运营工作台状态与请求客户端共用。
export interface Conversation {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface AuthUser {
  id: string;
  username: string;
  display_name: string;
  is_admin: boolean;
  created_at: string;
}

export interface AuthSession {
  access_token: string;
  token_type: "bearer";
  user: AuthUser;
}

export type UserMemoryType =
  | "analysis_preference"
  | "answer_preference"
  | "focus_topic"
  | "work_profile"
  | "focus_direction";

export interface MemoryCandidate {
  id: string;
  conversation_id: string;
  source_message_id: string;
  agent_message_id: string;
  memory_type: UserMemoryType;
  content: string;
  confidence: number;
  expires_at: string | null;
  created_at: string;
}

export interface Message {
  id: string;
  conversation_id: string;
  sender_type: "user" | "agent";
  content: string;
  data_references: string | null;
  status: string;
  created_at: string;
}

export interface ToolCallAudit {
  tool_name: string;
  input_summary: string | null;
  result_summary: string | null;
  reference_ids: string[];
  status: string;
  duration_ms: number | null;
  error_code: string | null;
  created_at: string;
}

export interface AgentRunAudit {
  id: string;
  question_summary: string;
  route_mode: string | null;
  route_confidence: number | null;
  route_fallback: boolean;
  context_summary: string | null;
  context_actions: string[];
  context_snapshot: Record<string, unknown>;
  memory_summary: string | null;
  memory_ids: string[];
  memory_selection: Array<{ memory_id: string; memory_type: string; reason: string }>;
  conversation_summary_version: number | null;
  conversation_summary_used: boolean;
  conversation_history_ids: string[];
  conversation_history_used: boolean;
  status: string;
  answer_summary: string | null;
  reference_ids: string[];
  execution_plan: string[];
  total_duration_ms: number | null;
  error_code: string | null;
  created_at: string;
  completed_at: string | null;
  tool_calls: ToolCallAudit[];
}

export interface KnowledgeDocument {
  id: string;
  space: "private" | "team";
  title: string;
  original_filename: string;
  file_type: string;
  status: string;
  error_message: string | null;
  chunk_count: number;
  index_status: string | null;
  index_stage: string | null;
  processed_chunks: number;
  total_chunks: number;
  index_error_message: string | null;
  retrieval_enabled: boolean;
  created_at: string;
  updated_at: string;
}

export interface KnowledgeDocumentContent extends KnowledgeDocument {
  content: string;
}
