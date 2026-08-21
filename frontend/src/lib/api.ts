import type { Conversation, Message, Task } from "../types";

// Docker 会在构建期注入该地址；本地开发时回退到默认 API 端口。
const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  // 统一处理 REST 请求的 API 基地址、JSON 请求头和异常状态。
  const response = await fetch(`${API_URL}${path}`, {
    headers: { "Content-Type": "application/json", ...init?.headers },
    ...init,
  });
  if (!response.ok) {
    throw new Error(`请求失败：${response.status}`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  listConversations: () => request<Conversation[]>("/api/conversations"),
  createConversation: (title = "新会话") =>
    request<Conversation>("/api/conversations", {
      method: "POST",
      body: JSON.stringify({ title }),
    }),
  listMessages: (conversationId: string) =>
    request<Message[]>(`/api/conversations/${conversationId}/messages`),
  listTasks: () => request<Task[]>("/api/tasks"),
  createTask: (title: string, sourceMessageId?: string) =>
    request<Task>("/api/tasks", {
      method: "POST",
      body: JSON.stringify({ title, source_message_id: sourceMessageId }),
    }),
  sendFeedback: (messageId: string, feedbackType: "up" | "down") =>
    request<{ id: string; message: string }>(`/api/messages/${messageId}/feedback`, {
      method: "POST",
      body: JSON.stringify({ feedback_type: feedbackType }),
    }),
  updateTask: (id: string, status: Task["status"]) =>
    request<Task>(`/api/tasks/${id}`, {
      method: "PATCH",
      body: JSON.stringify({ status }),
    }),
  async streamMessage(
    conversationId: string,
    content: string,
    onChunk: (chunk: string) => void,
    onDone: (messageId: string) => void,
  ): Promise<void> {
    // 该 SSE 接口是 POST 请求，因此使用 fetch 而非仅支持 GET 的 EventSource。
    const response = await fetch(`${API_URL}/api/conversations/${conversationId}/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content }),
    });
    if (!response.ok || !response.body) throw new Error("无法建立流式连接");

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
        const payload = JSON.parse(data) as { content?: string; message_id?: string; message?: string };
        if (type === "chunk" && payload.content) onChunk(payload.content);
        if (type === "done" && payload.message_id) onDone(payload.message_id);
        if (type === "error") throw new Error(payload.message ?? "生成失败");
      }
    }
  },
};
