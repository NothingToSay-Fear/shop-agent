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
