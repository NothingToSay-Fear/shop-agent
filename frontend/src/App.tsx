import { useEffect, useMemo, useState } from "react";
import {
  CheckCircleOutlined,
  DatabaseOutlined,
  DeleteOutlined,
  DownloadOutlined,
  EyeOutlined,
  FileTextOutlined,
  LogoutOutlined,
  MessageOutlined,
  PlusOutlined,
  SendOutlined,
  UploadOutlined,
  UserOutlined,
} from "@ant-design/icons";
import {
  Button,
  Card,
  Drawer,
  Input,
  Layout,
  List,
  Modal,
  Popconfirm,
  Select,
  Space,
  Spin,
  Tag,
  Tooltip,
  Typography,
  message,
} from "antd";

import { api } from "./lib/api";
import type {
  AgentRunAudit,
  AuthUser,
  Conversation,
  KnowledgeDocument,
  KnowledgeDocumentContent,
  KnowledgeGroup,
  Message,
  MemoryCandidate,
} from "./types";

const { Sider, Content } = Layout;

const examples = [
  "分析本周 GMV 环比下降原因",
  "生成一份秋季上新活动方案",
  "为商品 A 写 3 个小红书标题",
];

const memoryTypeOptions: { value: MemoryCandidate["memory_type"]; label: string }[] = [
  { value: "analysis_preference", label: "分析习惯" },
  { value: "answer_preference", label: "回答偏好" },
  { value: "focus_topic", label: "关注主题" },
];

function memoryTypeLabel(memoryType: MemoryCandidate["memory_type"]) {
  return memoryTypeOptions.find((item) => item.value === memoryType)?.label ?? memoryType;
}

function documentStatusTag(status: string) {
  if (status === "ready") return <Tag color="success">索引已完成，可检索</Tag>;
  if (status === "processing") return <Tag color="processing">正在建立索引</Tag>;
  return <Tag color="warning">待向量化</Tag>;
}

export function App() {
  // MVP 阶段将会话和页面状态集中在此处；页面增多后再引入全局状态管理。
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string>();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(true);
  const [streaming, setStreaming] = useState(false);
  const [streamingStatus, setStreamingStatus] = useState("");
  const [knowledgeGroup, setKnowledgeGroup] = useState<string>();
  const [knowledgeOpen, setKnowledgeOpen] = useState(false);
  const [knowledgeDocuments, setKnowledgeDocuments] = useState<KnowledgeDocument[]>([]);
  const [knowledgeGroups, setKnowledgeGroups] = useState<KnowledgeGroup[]>([]);
  const [uploadGroup, setUploadGroup] = useState("活动规则");
  const [uploadFile, setUploadFile] = useState<File>();
  const [uploading, setUploading] = useState(false);
  const [deletingDocumentId, setDeletingDocumentId] = useState<string>();
  const [previewDocument, setPreviewDocument] = useState<KnowledgeDocumentContent>();
  const [auditOpen, setAuditOpen] = useState(false);
  const [auditRecord, setAuditRecord] = useState<AgentRunAudit>();
  const [currentUser, setCurrentUser] = useState<AuthUser>();
  const [authMode, setAuthMode] = useState<"login" | "register">("login");
  const [username, setUsername] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [password, setPassword] = useState("");
  const [authSubmitting, setAuthSubmitting] = useState(false);
  const [memoryCandidates, setMemoryCandidates] = useState<MemoryCandidate[]>([]);
  const [resolvingMemoryCandidateId, setResolvingMemoryCandidateId] = useState<string>();

  const activeConversation = useMemo(
    () => conversations.find((item) => item.id === activeConversationId),
    [activeConversationId, conversations],
  );

  useEffect(() => {
    // 仅在已恢复登录状态后加载用户自己的会话，避免未认证请求访问工作台内容。
    const handleAuthExpired = () => clearWorkspaceForLogout("登录状态已过期，请重新登录。");
    window.addEventListener("shop-agent-auth-expired", handleAuthExpired);
    if (api.getAccessToken()) void restoreSession();
    else setLoading(false);
    return () => window.removeEventListener("shop-agent-auth-expired", handleAuthExpired);
  }, []);

  useEffect(() => {
    // 切换会话时刷新该会话已持久化的消息记录。
    if (activeConversationId) {
      void loadMessages(activeConversationId);
      void loadMemoryCandidates(activeConversationId);
    }
  }, [activeConversationId]);

  async function bootstrap() {
    try {
      const existingConversations = await api.listConversations();
      if (existingConversations.length === 0) {
        // 首次访问时创建空会话，保证用户可以立即发送消息。
        const firstConversation = await api.createConversation();
        setConversations([firstConversation]);
        setActiveConversationId(firstConversation.id);
      } else {
        setConversations(existingConversations);
        setActiveConversationId(existingConversations[0].id);
      }
    } catch {
      message.error("无法连接后端，请确认 API 服务已启动。");
    } finally {
      setLoading(false);
    }
  }

  async function restoreSession() {
    try {
      setCurrentUser(await api.getCurrentUser());
      await bootstrap();
    } catch {
      clearWorkspaceForLogout();
    } finally {
      setLoading(false);
    }
  }

  async function submitAuthentication() {
    if (!username.trim() || !password) {
      message.warning("请填写账号和密码。");
      return;
    }
    if (authMode === "register" && !displayName.trim()) {
      message.warning("请填写显示名称。");
      return;
    }
    setAuthSubmitting(true);
    try {
      const session = authMode === "register"
        ? await api.register(username, displayName, password)
        : await api.login(username, password);
      setCurrentUser(session.user);
      setPassword("");
      setLoading(true);
      await bootstrap();
      message.success(authMode === "register" ? "账号创建成功，已登录。" : "登录成功。");
    } catch (error) {
      message.error(error instanceof Error ? error.message : "认证失败，请稍后重试。");
    } finally {
      setAuthSubmitting(false);
      setLoading(false);
    }
  }

  async function logout() {
    try {
      await api.logout();
    } catch (error) {
      message.error(error instanceof Error ? error.message : "退出登录失败。");
    } finally {
      clearWorkspaceForLogout();
    }
  }

  function clearWorkspaceForLogout(notice?: string) {
    setCurrentUser(undefined);
    setConversations([]);
    setActiveConversationId(undefined);
    setMessages([]);
    setKnowledgeGroup(undefined);
    setMemoryCandidates([]);
    setLoading(false);
    if (notice) message.warning(notice);
  }

  async function loadMessages(conversationId: string) {
    try {
      setMessages(await api.listMessages(conversationId));
    } catch {
      message.error("加载会话失败。");
    }
  }

  async function loadMemoryCandidates(conversationId: string) {
    try {
      setMemoryCandidates(await api.listMemoryCandidates(conversationId));
    } catch (error) {
      message.error(error instanceof Error ? error.message : "加载记忆候选失败。");
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

  async function loadKnowledge() {
    try {
      const [documents, groups] = await Promise.all([
        api.listKnowledgeDocuments(),
        api.listKnowledgeGroups(),
      ]);
      setKnowledgeDocuments(documents);
      setKnowledgeGroups(groups);
    } catch {
      message.error("加载知识库失败。");
    }
  }

  async function resolveMemoryCandidate(candidate: MemoryCandidate, action: "accept" | "dismiss") {
    setResolvingMemoryCandidateId(candidate.id);
    try {
      if (action === "accept") {
        await api.acceptMemoryCandidate(candidate.id);
        message.success("已记住，会在后续相关问题中参考。");
      } else {
        await api.dismissMemoryCandidate(candidate.id);
      }
      setMemoryCandidates((items) => items.filter((item) => item.id !== candidate.id));
    } catch (error) {
      message.error(error instanceof Error ? error.message : "处理记忆候选失败。");
    } finally {
      setResolvingMemoryCandidateId(undefined);
    }
  }

  function openKnowledge() {
    setKnowledgeOpen(true);
    void loadKnowledge();
  }

  async function uploadKnowledgeDocument() {
    if (!uploadFile || !uploadGroup.trim()) {
      message.warning("请选择文件并填写分组名称。");
      return;
    }
    setUploading(true);
    try {
      const document = await api.uploadKnowledgeDocument(uploadFile, uploadGroup.trim());
      message.success(document.status === "ready" ? "文件已完成索引，可立即用于综合分析。" : "文件已上传，等待向量索引。" );
      setUploadFile(undefined);
      await loadKnowledge();
    } catch (error) {
      message.error(error instanceof Error ? error.message : "文件上传失败。");
    } finally {
      setUploading(false);
    }
  }

  async function previewKnowledgeDocument(documentId: string) {
    try {
      setPreviewDocument(await api.getKnowledgeDocument(documentId));
    } catch {
      message.error("加载文件预览失败。");
    }
  }

  async function deleteKnowledgeDocument(document: KnowledgeDocument) {
    setDeletingDocumentId(document.id);
    try {
      await api.deleteKnowledgeDocument(document.id);
      if (previewDocument?.id === document.id) setPreviewDocument(undefined);
      message.success("文件、索引片段和向量已删除。");
      await loadKnowledge();
    } catch (error) {
      message.error(error instanceof Error ? error.message : "删除文件失败。");
    } finally {
      setDeletingDocumentId(undefined);
    }
  }

  async function openMessageAudit(messageId: string) {
    if (!activeConversationId) return;
    setAuditOpen(true);
    setAuditRecord(undefined);
    try {
      setAuditRecord(await api.getMessageAudit(activeConversationId, messageId));
    } catch {
      setAuditOpen(false);
      message.error("未找到该回答的执行审计记录。");
    }
  }

  async function resetConversationContext() {
    if (!activeConversationId) return;
    try {
      await api.resetConversationContext(activeConversationId);
      setKnowledgeGroup(undefined);
      message.success("本会话的活动、时间、指标和资料分组条件已重置。");
    } catch (error) {
      message.error(error instanceof Error ? error.message : "重置会话条件失败。");
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
    setStreamingStatus("正在判断问题类型…");

    try {
      await api.streamMessage(
        activeConversationId,
        trimmedContent,
        "hybrid",
        knowledgeGroup,
        (content) => setStreamingStatus(content),
        (chunk) => {
          setMessages((items) =>
            items.map((item) => (item.id === "streaming" ? { ...item, content: item.content + chunk } : item)),
          );
        },
        (candidate) => {
          setMemoryCandidates((items) => [
            ...items.filter((item) => item.id !== candidate.id),
            candidate,
          ]);
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
      setStreamingStatus("");
    }
  }

  async function refreshConversations() {
    setConversations(await api.listConversations());
  }

  if (loading) {
    return <Spin className="page-spinner" size="large" />;
  }

  if (!currentUser) {
    return (
      <main className="auth-page">
        <Card className="auth-card">
          <Typography.Title level={2}>Shop Agent</Typography.Title>
          <Typography.Paragraph type="secondary">
            登录后可访问你自己的会话与后续长期记忆。
          </Typography.Paragraph>
          <Space.Compact block>
            <Button type={authMode === "login" ? "primary" : "default"} onClick={() => setAuthMode("login")}>登录</Button>
            <Button type={authMode === "register" ? "primary" : "default"} onClick={() => setAuthMode("register")}>注册</Button>
          </Space.Compact>
          <section className="auth-fields">
            <Input value={username} maxLength={50} onChange={(event) => setUsername(event.target.value)} placeholder="账号：字母、数字、_ 或 -" />
            {authMode === "register" && (
              <Input value={displayName} maxLength={100} onChange={(event) => setDisplayName(event.target.value)} placeholder="显示名称" />
            )}
            <Input.Password value={password} maxLength={128} onChange={(event) => setPassword(event.target.value)} onPressEnter={() => void submitAuthentication()} placeholder="密码至少 8 位" />
            <Button type="primary" block loading={authSubmitting} onClick={() => void submitAuthentication()}>
              {authMode === "login" ? "登录" : "创建账号并登录"}
            </Button>
          </section>
        </Card>
      </main>
    );
  }

  return (
    <Layout className="app-shell">
      <Sider width={260} theme="light" className="conversation-sider">
        <div className="brand"><span>✦</span> Shop Agent</div>
        <div className="current-user">
          <UserOutlined />
          <Typography.Text ellipsis title={currentUser.display_name}>{currentUser.display_name}</Typography.Text>
          <Tooltip title="退出登录">
            <Button type="text" size="small" shape="circle" icon={<LogoutOutlined />} onClick={() => void logout()} />
          </Tooltip>
        </div>
        <Button block type="primary" icon={<PlusOutlined />} onClick={() => void createConversation()}>
          新建会话
        </Button>
        <Button block icon={<DatabaseOutlined />} className="knowledge-button" onClick={openKnowledge}>
          知识库管理
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
            <Typography.Text type="secondary">数据分析、资料问答与内容创作</Typography.Text>
          </div>
          <Tag color="blue">MVP 演示模式</Tag>
        </header>
        <main className="conversation-content">
          {messages.length === 0 ? (
            <section className="welcome">
              <Typography.Title level={2}>今天想推进哪项运营工作？</Typography.Title>
              <Typography.Paragraph type="secondary">
                我可以帮助你分析经营数据、生成商品内容并规划活动。
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
              {messages.map((item) => {
                const candidate = item.sender_type === "agent"
                  ? memoryCandidates.find((value) => value.agent_message_id === item.id)
                  : undefined;
                return (
                <Card key={item.id} className={`message-card ${item.sender_type}`} size="small">
                  <div className="message-title">
                    {item.sender_type === "user" ? <UserOutlined /> : <CheckCircleOutlined />}
                    <Typography.Text strong>{item.sender_type === "user" ? "运营人员" : "Shop Agent"}</Typography.Text>
                  </div>
                  {item.id === "streaming" && (
                    <div className="message-progress"><Spin size="small" /> {streamingStatus || "正在处理…"}</div>
                  )}
                  <div className="message-body">{item.content || (item.id === "streaming" ? "" : "正在思考…")}</div>
                  {item.sender_type === "agent" && item.data_references && (
                    <Typography.Text type="secondary" className="message-reference">
                      依据：{item.data_references}
                    </Typography.Text>
                  )}
                  {item.sender_type === "agent" && item.status === "completed" && (
                    <Button type="link" size="small" onClick={() => void openMessageAudit(item.id)}>
                      查看本次执行依据
                    </Button>
                  )}
                  {candidate && (
                    <section className="memory-candidate">
                      <Typography.Text type="secondary">
                        检测到你可能希望长期保留这项偏好：{candidate.content}
                      </Typography.Text>
                      <Space size={8}>
                        <Button
                          type="link"
                          size="small"
                          loading={resolvingMemoryCandidateId === candidate.id}
                          onClick={() => void resolveMemoryCandidate(candidate, "accept")}
                        >
                          记住
                        </Button>
                        <Button
                          type="link"
                          size="small"
                          disabled={resolvingMemoryCandidateId === candidate.id}
                          onClick={() => void resolveMemoryCandidate(candidate, "dismiss")}
                        >
                          暂不
                        </Button>
                        <Tag>{memoryTypeLabel(candidate.memory_type)}</Tag>
                      </Space>
                    </section>
                  )}
                </Card>
                );
              })}
            </section>
          )}
        </main>
        <footer className="composer">
          <div className="composer-knowledge">
            <Tag color="blue">综合分析</Tag>
            <Select
              allowClear
              placeholder="全部知识库分组"
              value={knowledgeGroup}
              onChange={setKnowledgeGroup}
              options={knowledgeGroups.map((group) => ({ value: group.name, label: group.name }))}
            />
            <Popconfirm
              title="重置本会话的继承条件？"
              description="不会删除历史消息或运行审计。"
              okText="重置"
              cancelText="取消"
              onConfirm={() => void resetConversationContext()}
            >
              <Button type="link" size="small">重置会话条件</Button>
            </Popconfirm>
          </div>
          <Input.TextArea
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onPressEnter={(event) => {
              if (!event.shiftKey) {
                event.preventDefault();
                void sendMessage();
              }
            }}
            placeholder="例如：本周 GMV 下滑后，可参考活动规则和历史复盘采取哪些动作？"
            autoSize={{ minRows: 2, maxRows: 5 }}
            disabled={streaming}
          />
          <Button type="primary" icon={<SendOutlined />} loading={streaming} onClick={() => void sendMessage()}>
            发送
          </Button>
        </footer>
      </Content>

      <Drawer title="知识库管理" width={560} open={knowledgeOpen} onClose={() => setKnowledgeOpen(false)}>
        <section className="knowledge-upload">
          <Typography.Text strong>上传运营资料</Typography.Text>
          <Input
            value={uploadGroup}
            maxLength={100}
            onChange={(event) => setUploadGroup(event.target.value)}
            placeholder="分组，例如：活动规则"
          />
          <input
            key={uploadFile?.name ?? "empty"}
            className="file-input"
            type="file"
            accept=".pdf,.docx,.md,.txt"
            onChange={(event) => setUploadFile(event.target.files?.[0])}
          />
          <Typography.Text type="secondary">
            支持 PDF、DOCX、Markdown、TXT，单个文件最大 20MB。
          </Typography.Text>
          <Button type="primary" icon={<UploadOutlined />} loading={uploading} onClick={() => void uploadKnowledgeDocument()}>
            {uploading ? "正在上传并建立索引…" : "上传并建立索引"}
          </Button>
          {uploading && <Spin size="small" tip="正在解析文件、切分文本并生成向量，请勿关闭此页面。" />}
        </section>
        <Typography.Title level={5}>已上传资料</Typography.Title>
        <List
          dataSource={knowledgeDocuments}
          locale={{ emptyText: "暂无资料" }}
          renderItem={(item) => (
            <List.Item
              className="knowledge-document-item"
              actions={[
                <Tooltip key="preview" title="预览">
                  <Button
                    aria-label="预览"
                    type="text"
                    size="small"
                    shape="circle"
                    icon={<EyeOutlined />}
                    onClick={() => void previewKnowledgeDocument(item.id)}
                  />
                </Tooltip>,
                <Tooltip key="download" title="下载原文件">
                  <Button
                    aria-label="下载原文件"
                    type="text"
                    size="small"
                    shape="circle"
                    icon={<DownloadOutlined />}
                    href={api.getKnowledgeDownloadUrl(item.id)}
                  />
                </Tooltip>,
                <Popconfirm
                  key="delete"
                  title="删除此知识库文件？"
                  description="将同时删除原始文件、索引片段和向量，无法恢复。"
                  okText="删除"
                  cancelText="取消"
                  onConfirm={() => void deleteKnowledgeDocument(item)}
                >
                  <Button
                    aria-label="删除"
                    title="删除"
                    danger
                    type="text"
                    size="small"
                    shape="circle"
                    icon={<DeleteOutlined />}
                    loading={deletingDocumentId === item.id}
                  />
                </Popconfirm>,
              ]}
            >
              <List.Item.Meta
                title={
                  <div className="knowledge-document-title">
                    <Typography.Text className="knowledge-document-name" ellipsis={{ tooltip: item.title }}>
                      {item.title}
                    </Typography.Text>
                    <Tooltip title={item.group_name}>
                      <Tag className="knowledge-document-group">{item.group_name}</Tag>
                    </Tooltip>
                    {documentStatusTag(item.status)}
                  </div>
                }
                description={`${item.file_type.toUpperCase()} · ${item.chunk_count} 个片段`}
              />
            </List.Item>
          )}
        />
      </Drawer>
      <Drawer
        title="本次执行依据"
        width={560}
        open={auditOpen}
        onClose={() => setAuditOpen(false)}
      >
        {!auditRecord ? (
          <Spin tip="正在加载执行记录…" />
        ) : (
          <Space direction="vertical" size="middle" className="audit-drawer-content">
            <Typography.Text type="secondary">运行 ID：{auditRecord.id}</Typography.Text>
            <Typography.Text>问题摘要：{auditRecord.question_summary}</Typography.Text>
            <Typography.Text>
              路由：{auditRecord.route_mode ?? "未记录"}
              {auditRecord.route_confidence !== null && `（置信度 ${auditRecord.route_confidence.toFixed(2)}）`}
              {auditRecord.route_fallback && "；已采用保守降级"}
            </Typography.Text>
            <Typography.Text>
              会话条件：{auditRecord.context_summary ?? "本轮未使用已确认条件"}
            </Typography.Text>
            <Typography.Text>
              长期记忆：{auditRecord.memory_summary ?? "本轮未采用长期记忆"}
            </Typography.Text>
            {auditRecord.memory_ids.length > 0 && (
              <Typography.Text type="secondary">记忆 ID：{auditRecord.memory_ids.join("、")}</Typography.Text>
            )}
            {auditRecord.context_actions.length > 0 && (
              <Typography.Text type="secondary">
                条件变更：{auditRecord.context_actions.join("；")}
              </Typography.Text>
            )}
            <Typography.Text>
              执行计划：{auditRecord.execution_plan.length > 0 ? auditRecord.execution_plan.join(" → ") : "无需检索工具"}
            </Typography.Text>
            <Typography.Text>
              状态：{auditRecord.status}；总耗时：{auditRecord.total_duration_ms ?? "-"} ms
            </Typography.Text>
            <Typography.Text>回答摘要：{auditRecord.answer_summary ?? "尚未生成"}</Typography.Text>
            <section>
              <Typography.Text strong>调用能力</Typography.Text>
              <List
                size="small"
                dataSource={auditRecord.tool_calls}
                locale={{ emptyText: "本轮未调用受控工具" }}
                renderItem={(call) => (
                  <List.Item>
                    <Space direction="vertical" size={2}>
                      <Typography.Text strong>{call.tool_name} · {call.status} · {call.duration_ms ?? "-"} ms</Typography.Text>
                      <Typography.Text type="secondary">{call.input_summary}</Typography.Text>
                      <Typography.Text>{call.result_summary}</Typography.Text>
                      {call.reference_ids.length > 0 && (
                        <Typography.Text type="secondary">引用 ID：{call.reference_ids.join("、")}</Typography.Text>
                      )}
                      {call.error_code && <Typography.Text type="warning">原因：{call.error_code}</Typography.Text>}
                    </Space>
                  </List.Item>
                )}
              />
            </section>
            {auditRecord.reference_ids.length > 0 && (
              <Typography.Text type="secondary">本轮全部引用 ID：{auditRecord.reference_ids.join("、")}</Typography.Text>
            )}
          </Space>
        )}
      </Drawer>
      <Modal
        title={previewDocument?.title}
        open={Boolean(previewDocument)}
        onCancel={() => setPreviewDocument(undefined)}
        footer={previewDocument ? <Button href={api.getKnowledgeDownloadUrl(previewDocument.id)} icon={<DownloadOutlined />}>下载原文件</Button> : null}
        width={760}
      >
        <Typography.Paragraph className="knowledge-preview">
          {previewDocument?.content}
        </Typography.Paragraph>
      </Modal>

    </Layout>
  );
}
