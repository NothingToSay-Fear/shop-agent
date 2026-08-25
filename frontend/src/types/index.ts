// API 响应类型由运营工作台状态与请求客户端共用。
export interface Conversation {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
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
  title: string;
  original_filename: string;
  file_type: string;
  group_name: string;
  status: string;
  error_message: string | null;
  chunk_count: number;
  created_at: string;
  updated_at: string;
}

export interface KnowledgeDocumentContent extends KnowledgeDocument {
  content: string;
}

export interface KnowledgeGroup {
  name: string;
  document_count: number;
}
