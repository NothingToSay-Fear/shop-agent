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
