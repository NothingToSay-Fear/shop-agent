import type {
  AgentRunAudit,
  AuthSession,
  AuthUser,
  Conversation,
  KnowledgeDocument,
  KnowledgeDocumentContent,
  Message,
  MemoryCandidate,
} from "../types";

// Docker 会在构建期注入该地址；本地开发时回退到默认 API 端口。
const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";
const ACCESS_TOKEN_STORAGE_KEY = "shop-agent-access-token";

let accessToken = localStorage.getItem(ACCESS_TOKEN_STORAGE_KEY);

function buildHeaders(init?: RequestInit): Headers {
  const headers = new Headers(init?.headers);
  if (accessToken) headers.set("Authorization", `Bearer ${accessToken}`);
  return headers;
}

function saveAccessToken(token: string | null) {
  accessToken = token;
  if (token) localStorage.setItem(ACCESS_TOKEN_STORAGE_KEY, token);
  else localStorage.removeItem(ACCESS_TOKEN_STORAGE_KEY);
}

function handleUnauthorized(status: number) {
  if (status === 401) {
    saveAccessToken(null);
    window.dispatchEvent(new Event("shop-agent-auth-expired"));
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  // 统一处理 REST 请求的 API 基地址、JSON 请求头和异常状态。
  const headers = buildHeaders(init);
  if (!headers.has("Content-Type")) headers.set("Content-Type", "application/json");
  const response = await fetch(`${API_URL}${path}`, { ...init, headers });
  if (!response.ok) {
    handleUnauthorized(response.status);
    const payload = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(payload?.detail ?? `请求失败：${response.status}`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  getAccessToken: () => accessToken,
  async register(username: string, displayName: string, password: string): Promise<AuthSession> {
    const session = await request<AuthSession>("/api/auth/register", {
      method: "POST",
      body: JSON.stringify({ username, display_name: displayName, password }),
    });
    saveAccessToken(session.access_token);
    return session;
  },
  async login(username: string, password: string): Promise<AuthSession> {
    const session = await request<AuthSession>("/api/auth/login", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    });
    saveAccessToken(session.access_token);
    return session;
  },
  getCurrentUser: () => request<AuthUser>("/api/auth/me"),
  async logout(): Promise<void> {
    const response = await fetch(`${API_URL}/api/auth/logout`, {
      method: "POST",
      headers: buildHeaders(),
    });
    saveAccessToken(null);
    if (!response.ok && response.status !== 401) {
      const payload = (await response.json().catch(() => null)) as { detail?: string } | null;
      throw new Error(payload?.detail ?? `退出登录失败：${response.status}`);
    }
  },
  listMemoryCandidates: (conversationId: string) =>
    request<MemoryCandidate[]>(`/api/memory-candidates?conversation_id=${encodeURIComponent(conversationId)}`),
  async resolveMemoryCandidate(candidateId: string, action: "accept" | "dismiss"): Promise<void> {
    const response = await fetch(`${API_URL}/api/memory-candidates/${candidateId}/${action}`, {
      method: "POST",
      headers: buildHeaders(),
    });
    handleUnauthorized(response.status);
    if (!response.ok) {
      const payload = (await response.json().catch(() => null)) as { detail?: string } | null;
      throw new Error(payload?.detail ?? `处理记忆候选失败：${response.status}`);
    }
  },
  acceptMemoryCandidate: (candidateId: string) => api.resolveMemoryCandidate(candidateId, "accept"),
  dismissMemoryCandidate: (candidateId: string) => api.resolveMemoryCandidate(candidateId, "dismiss"),
  listConversations: () => request<Conversation[]>("/api/conversations"),
  createConversation: (title = "新会话") =>
    request<Conversation>("/api/conversations", {
      method: "POST",
      body: JSON.stringify({ title }),
    }),
  listMessages: (conversationId: string) =>
    request<Message[]>(`/api/conversations/${conversationId}/messages`),
  async resetConversationContext(conversationId: string): Promise<void> {
    const response = await fetch(`${API_URL}/api/conversations/${conversationId}/context`, {
      method: "DELETE",
      headers: buildHeaders(),
    });
    handleUnauthorized(response.status);
    if (!response.ok) throw new Error(`重置会话条件失败：${response.status}`);
  },
  getMessageAudit: (conversationId: string, messageId: string) =>
    request<AgentRunAudit>(`/api/conversations/${conversationId}/messages/${messageId}/audit`),
  listKnowledgeDocuments: () => request<KnowledgeDocument[]>("/api/knowledge/documents"),
  getKnowledgeDocument: (documentId: string) =>
    request<KnowledgeDocumentContent>(`/api/knowledge/documents/${documentId}`),
  async uploadKnowledgeDocument(
    file: File, space: "private" | "team" = "private",
  ): Promise<KnowledgeDocument> {
    // 文件上传不能使用全局 JSON 请求头，否则浏览器无法携带 multipart 边界。
    const body = new FormData();
    body.append("file", file);
    body.append("space", space);
    const response = await fetch(`${API_URL}/api/knowledge/documents`, {
      method: "POST",
      headers: buildHeaders(),
      body,
    });
    if (!response.ok) {
      const payload = (await response.json().catch(() => null)) as { detail?: string } | null;
      throw new Error(payload?.detail ?? `上传失败：${response.status}`);
    }
    return response.json() as Promise<KnowledgeDocument>;
  },
  async deleteKnowledgeDocument(documentId: string): Promise<void> {
    const response = await fetch(`${API_URL}/api/knowledge/documents/${documentId}`, {
      method: "DELETE",
      headers: buildHeaders(),
    });
    if (!response.ok) {
      const payload = (await response.json().catch(() => null)) as { detail?: string } | null;
      throw new Error(payload?.detail ?? `删除失败：${response.status}`);
    }
  },
  async updateKnowledgeDocumentRetrieval(
    documentId: string, retrievalEnabled: boolean,
  ): Promise<KnowledgeDocument> {
    return request<KnowledgeDocument>(`/api/knowledge/documents/${documentId}/retrieval`, {
      method: "PUT",
      body: JSON.stringify({ retrieval_enabled: retrievalEnabled }),
    });
  },
  async downloadKnowledgeDocument(documentId: string): Promise<Blob> {
    const response = await fetch(`${API_URL}/api/knowledge/documents/${documentId}/download`, {
      headers: buildHeaders(),
    });
    handleUnauthorized(response.status);
    if (!response.ok) {
      const payload = (await response.json().catch(() => null)) as { detail?: string } | null;
      throw new Error(payload?.detail ?? `下载失败：${response.status}`);
    }
    return response.blob();
  },
  async streamMessage(
    conversationId: string,
    content: string,
    mode: "hybrid" | "metrics" | "knowledge" | "web",
    onStatus: (content: string, phase: string | undefined) => void,
    onChunk: (chunk: string) => void,
    onMemoryCandidate: (candidate: MemoryCandidate) => void,
    onDone: (messageId: string) => void,
  ): Promise<void> {
    // 该 SSE 接口是 POST 请求，因此使用 fetch 而非仅支持 GET 的 EventSource。
    const response = await fetch(`${API_URL}/api/conversations/${conversationId}/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...Object.fromEntries(buildHeaders()) },
      body: JSON.stringify({ content, mode }),
    });
    handleUnauthorized(response.status);
    if (!response.ok || !response.body) {
      const payload = (await response.json().catch(() => null)) as { detail?: string } | null;
      throw new Error(payload?.detail ?? "无法建立流式连接");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      // 单次网络读取可能包含半个或多个事件，保留末尾不完整片段以供下次拼接。
      const events = buffer.split("\n\n");
      buffer = events.pop() ?? "";
      for (const event of events) {
        const type = event.match(/^event: (.+)$/m)?.[1];
        const data = event.match(/^data: (.+)$/m)?.[1];
        if (!type || !data) continue;
        const payload = JSON.parse(data) as {
          content?: string;
          phase?: string;
          message_id?: string;
          message?: string;
          candidate?: MemoryCandidate;
        };
        if (type === "status" && payload.content) onStatus(payload.content, payload.phase);
        if (type === "chunk" && payload.content) onChunk(payload.content);
        if (type === "memory_candidate" && payload.candidate) onMemoryCandidate(payload.candidate);
        if (type === "done" && payload.message_id) onDone(payload.message_id);
        if (type === "error") throw new Error(payload.message ?? "生成失败");
      }
    }
  },
};
