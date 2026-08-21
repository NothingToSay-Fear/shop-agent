import { useEffect, useMemo, useState } from "react";
import {
  CheckCircleOutlined,
  FileTextOutlined,
  LikeOutlined,
  MessageOutlined,
  PlusOutlined,
  SendOutlined,
  ProfileOutlined,
  UserOutlined,
} from "@ant-design/icons";
import { Button, Card, Input, Layout, List, Select, Space, Spin, Tag, Typography, message } from "antd";

import { api } from "./lib/api";
import type { Conversation, Message, Task } from "./types";

const { Sider, Content } = Layout;

const examples = [
  "分析本周 GMV 环比下降原因",
  "生成一份秋季上新活动方案",
  "为商品 A 写 3 个小红书标题",
];

export function App() {
  // MVP 阶段将会话和任务状态集中在此处；页面增多后再引入全局状态管理。
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string>();
  const [messages, setMessages] = useState<Message[]>([]);
  const [tasks, setTasks] = useState<Task[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(true);
  const [streaming, setStreaming] = useState(false);

  const activeConversation = useMemo(
    () => conversations.find((item) => item.id === activeConversationId),
    [activeConversationId, conversations],
  );

  useEffect(() => {
    // 页面首次挂载时只加载一次工作台初始数据。
    void bootstrap();
  }, []);

  useEffect(() => {
    // 切换会话时刷新该会话已持久化的消息记录。
    if (activeConversationId) void loadMessages(activeConversationId);
  }, [activeConversationId]);

  async function bootstrap() {
    try {
      const [existingConversations, existingTasks] = await Promise.all([
        api.listConversations(),
        api.listTasks(),
      ]);
      if (existingConversations.length === 0) {
        // 首次访问时创建空会话，保证用户可以立即发送消息。
        const firstConversation = await api.createConversation();
        setConversations([firstConversation]);
        setActiveConversationId(firstConversation.id);
      } else {
        setConversations(existingConversations);
        setActiveConversationId(existingConversations[0].id);
      }
      setTasks(existingTasks);
    } catch {
      message.error("无法连接后端，请确认 API 服务已启动。");
    } finally {
      setLoading(false);
    }
  }

  async function loadMessages(conversationId: string) {
    try {
      setMessages(await api.listMessages(conversationId));
    } catch {
      message.error("加载会话失败。");
    }
  }

  async function createConversation() {
    try {
      const conversation = await api.createConversation();
      setConversations((items) => [conversation, ...items]);
      setActiveConversationId(conversation.id);
      setMessages([]);
    } catch {
      message.error("创建会话失败。");
    }
  }

  async function sendMessage(content = input) {
    const trimmedContent = content.trim();
    if (!trimmedContent || !activeConversationId || streaming) return;

    // 在 POST/SSE 请求进行期间先插入临时消息，以提升对话的即时反馈体验。
    const pendingUser: Message = {
      id: crypto.randomUUID(),
      conversation_id: activeConversationId,
      sender_type: "user",
      content: trimmedContent,
      data_references: null,
      status: "completed",
      created_at: new Date().toISOString(),
    };
    const pendingAgent: Message = {
      id: "streaming",
      conversation_id: activeConversationId,
      sender_type: "agent",
      content: "",
      data_references: null,
      status: "streaming",
      created_at: new Date().toISOString(),
    };
    setMessages((items) => [...items, pendingUser, pendingAgent]);
    setInput("");
    setStreaming(true);

    try {
      await api.streamMessage(
        activeConversationId,
        trimmedContent,
        (chunk) => {
          setMessages((items) =>
            items.map((item) => (item.id === "streaming" ? { ...item, content: item.content + chunk } : item)),
          );
        },
        () => undefined,
      );
      // 收到 `done` 后重新加载，用数据库生成的 ID 和时间替换临时消息。
      await Promise.all([loadMessages(activeConversationId), refreshConversations()]);
    } catch (error) {
      message.error(error instanceof Error ? error.message : "生成回答失败。");
      setMessages((items) => items.filter((item) => item.id !== "streaming"));
    } finally {
      setStreaming(false);
    }
  }

  async function refreshConversations() {
    setConversations(await api.listConversations());
  }

  async function createTask(source: Message) {
    try {
      const task = await api.createTask(`跟进：${source.content.slice(0, 36)}`, source.id);
      setTasks((items) => [task, ...items]);
      message.success("已创建运营任务。");
    } catch {
      message.error("创建任务失败。");
    }
  }

  async function updateTaskStatus(task: Task, status: Task["status"]) {
    try {
      const updatedTask = await api.updateTask(task.id, status);
      setTasks((items) => items.map((item) => (item.id === task.id ? updatedTask : item)));
    } catch {
      message.error("更新任务失败。");
    }
  }

  async function feedback(messageId: string, type: "up" | "down") {
    try {
      await api.sendFeedback(messageId, type);
      message.success("感谢你的反馈。");
    } catch {
      message.error("反馈提交失败。");
    }
  }

  if (loading) {
    return <Spin className="page-spinner" size="large" />;
  }

  return (
    <Layout className="app-shell">
      <Sider width={260} theme="light" className="conversation-sider">
        <div className="brand"><span>✦</span> Shop Agent</div>
        <Button block type="primary" icon={<PlusOutlined />} onClick={() => void createConversation()}>
          新建会话
        </Button>
        <Typography.Text type="secondary" className="sider-label">最近会话</Typography.Text>
        <List
          dataSource={conversations}
          locale={{ emptyText: "暂无会话" }}
          renderItem={(item) => (
            <List.Item
              className={`conversation-item ${item.id === activeConversationId ? "active" : ""}`}
              onClick={() => setActiveConversationId(item.id)}
            >
              <MessageOutlined />
              <Typography.Text ellipsis>{item.title}</Typography.Text>
            </List.Item>
          )}
        />
      </Sider>

      <Content className="workspace">
        <header className="workspace-header">
          <div>
            <Typography.Title level={4}>{activeConversation?.title ?? "运营工作台"}</Typography.Title>
            <Typography.Text type="secondary">数据分析、内容创作与任务协同</Typography.Text>
          </div>
          <Tag color="blue">MVP 演示模式</Tag>
        </header>
        <main className="conversation-content">
          {messages.length === 0 ? (
            <section className="welcome">
              <Typography.Title level={2}>今天想推进哪项运营工作？</Typography.Title>
              <Typography.Paragraph type="secondary">
                我可以帮助你分析经营数据、生成商品内容、规划活动并创建任务。
              </Typography.Paragraph>
              <Space wrap>
                {examples.map((example) => (
                  <Button key={example} icon={<FileTextOutlined />} onClick={() => void sendMessage(example)}>
                    {example}
                  </Button>
                ))}
              </Space>
            </section>
          ) : (
            <section className="messages">
              {messages.map((item) => (
                <Card key={item.id} className={`message-card ${item.sender_type}`} size="small">
                  <div className="message-title">
                    {item.sender_type === "user" ? <UserOutlined /> : <CheckCircleOutlined />}
                    <Typography.Text strong>{item.sender_type === "user" ? "运营人员" : "Shop Agent"}</Typography.Text>
                  </div>
                  <div className="message-body">{item.content || "正在思考…"}</div>
                  {item.sender_type === "agent" && item.id !== "streaming" && (
                    <Space className="message-actions">
                        <Button type="link" size="small" icon={<ProfileOutlined />} onClick={() => void createTask(item)}>
                        创建任务
                      </Button>
                      <Button type="link" size="small" icon={<LikeOutlined />} onClick={() => void feedback(item.id, "up")}>
                        有帮助
                      </Button>
                    </Space>
                  )}
                </Card>
              ))}
            </section>
          )}
        </main>
        <footer className="composer">
          <Input.TextArea
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onPressEnter={(event) => {
              if (!event.shiftKey) {
                event.preventDefault();
                void sendMessage();
              }
            }}
            placeholder="例如：分析本周 GMV 环比下降原因"
            autoSize={{ minRows: 2, maxRows: 5 }}
            disabled={streaming}
          />
          <Button type="primary" icon={<SendOutlined />} loading={streaming} onClick={() => void sendMessage()}>
            发送
          </Button>
        </footer>
      </Content>

      <Sider width={300} theme="light" className="task-sider">
        <div className="task-heading"><ProfileOutlined /> 运营任务</div>
        <List
          dataSource={tasks}
          locale={{ emptyText: "从 Agent 建议创建任务" }}
          renderItem={(task) => (
            <List.Item className="task-item">
              <Typography.Text>{task.title}</Typography.Text>
              <Select
                size="small"
                value={task.status}
                onChange={(status: Task["status"]) => void updateTaskStatus(task, status)}
                options={[
                  { value: "todo", label: "待处理" },
                  { value: "in_progress", label: "进行中" },
                  { value: "completed", label: "已完成" },
                ]}
              />
            </List.Item>
          )}
        />
      </Sider>
    </Layout>
  );
}
