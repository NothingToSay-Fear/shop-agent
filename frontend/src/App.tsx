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
  Checkbox,
  Drawer,
  Input,
  Layout,
  List,
  Modal,
  Popconfirm,
  Radio,
  Space,
  Spin,
  Tabs,
  Tag,
  Tooltip,
  Typography,
  message,
} from "antd";

import { api } from "./lib/api";
import { MessageContent } from "./components/MessageContent";
import type {
  AgentRunAudit,
  AuthUser,
  Conversation,
  KnowledgeDocument,
  KnowledgeDocumentContent,
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
  { value: "work_profile", label: "工作背景" },
  { value: "focus_direction", label: "近期关注" },
];

function memoryTypeLabel(memoryType: MemoryCandidate["memory_type"]) {
  return memoryTypeOptions.find((item) => item.value === memoryType)?.label ?? memoryType;
}

function documentStatusTag(document: KnowledgeDocument) {
  if (document.status === "ready") return <Tag color="success">索引已完成，可检索</Tag>;
  if (document.status === "failed") return <Tooltip title={document.index_error_message ?? document.error_message ?? "索引失败"}><Tag color="error">索引失败，可删除后重传</Tag></Tooltip>;
  if (document.status === "queued") return <Tag color="processing">等待后台索引</Tag>;
  if (document.status === "processing") {
    const progress = document.total_chunks > 0
      ? `向量化 ${document.processed_chunks}/${document.total_chunks}`
      : "正在解析资料";
    return <Tag color="processing">{progress}</Tag>;
  }
  return <Tag color="warning">待处理</Tag>;
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
  const [knowledgeOpen, setKnowledgeOpen] = useState(false);
  const [knowledgeDocuments, setKnowledgeDocuments] = useState<KnowledgeDocument[]>([]);
  const [uploadSpace, setUploadSpace] = useState<"private" | "team">("private");
  const [knowledgeTab, setKnowledgeTab] = useState<"team" | "private">("team");
  const [uploadFile, setUploadFile] = useState<File>();
  const [uploading, setUploading] = useState(false);
  const [deletingDocumentId, setDeletingDocumentId] = useState<string>();
  const [deletingConversationId, setDeletingConversationId] = useState<string>();
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

  useEffect(() => {
    // 仅在抽屉打开且存在进行中的任务时轮询，索引完成后自动停止。
    const hasPendingIndex = knowledgeDocuments.some((document) =>
      document.status === "queued" || document.status === "processing",
    );
    if (!knowledgeOpen || !hasPendingIndex) return undefined;
    const timer = window.setInterval(() => void loadKnowledge(), 2000);
    return () => window.clearInterval(timer);
  }, [knowledgeOpen, knowledgeDocuments]);

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
    setUploadSpace("private");
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

  async function deleteConversation(conversation: Conversation) {
    if (streaming) return;
    setDeletingConversationId(conversation.id);
    try {
      await api.deleteConversation(conversation.id);
      const remaining = conversations.filter((item) => item.id !== conversation.id);
      setConversations(remaining);
      if (activeConversationId === conversation.id) {
        const nextConversationId = remaining[0]?.id;
        setActiveConversationId(nextConversationId);
        setMessages([]);
        setMemoryCandidates([]);
        setAuditOpen(false);
        setAuditRecord(undefined);
      }
      message.success("会话、消息和本次会话的运行审计已删除。");
    } catch (error) {
      message.error(error instanceof Error ? error.message : "删除会话失败。");
    } finally {
      setDeletingConversationId(undefined);
    }
  }

  async function loadKnowledge() {
    try {
      setKnowledgeDocuments(await api.listKnowledgeDocuments());
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
    if (!uploadFile) {
      message.warning("请选择要上传的文件。");
      return;
    }
    setUploading(true);
    try {
      const document = await api.uploadKnowledgeDocument(
        uploadFile,
        currentUser?.is_admin ? uploadSpace : "private",
      );
      message.success("文件已上传，正在后台建立索引。");
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

  async function updateKnowledgeDocumentRetrieval(document: KnowledgeDocument, enabled: boolean) {
    try {
      const updated = await api.updateKnowledgeDocumentRetrieval(document.id, enabled);
      setKnowledgeDocuments((items) => items.map((item) => (item.id === updated.id ? updated : item)));
      message.success(enabled ? "该资料已加入本人的问答检索" : "该资料已从本人的问答检索中移除");
    } catch (error) {
      message.error(error instanceof Error ? error.message : "更新资料检索设置失败");
    }
  }

  async function downloadKnowledgeDocument(document: KnowledgeDocument) {
    try {
      const blob = await api.downloadKnowledgeDocument(document.id);
      const url = URL.createObjectURL(blob);
      const anchor = window.document.createElement("a");
      anchor.href = url;
      anchor.download = document.original_filename;
      anchor.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      message.error(error instanceof Error ? error.message : "下载文件失败");
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
      message.success("本会话的活动、时间和指标条件已重置。");
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
    let bufferedChunk = "";
    let animationFrame: number | undefined;
    const flushBufferedChunk = () => {
      animationFrame = undefined;
      if (!bufferedChunk) return;
      const nextChunk = bufferedChunk;
      bufferedChunk = "";
      setMessages((items) =>
        items.map((item) => (item.id === "streaming" ? { ...item, content: item.content + nextChunk } : item)),
      );
    };

    try {
      await api.streamMessage(
        activeConversationId,
        trimmedContent,
        "hybrid",
        (content) => setStreamingStatus(content),
        (chunk) => {
          bufferedChunk += chunk;
          if (animationFrame === undefined) {
            animationFrame = window.requestAnimationFrame(flushBufferedChunk);
          }
        },
        (candidate) => {
          setMemoryCandidates((items) => [
            ...items.filter((item) => item.id !== candidate.id),
            candidate,
          ]);
        },
        () => undefined,
      );
      if (animationFrame !== undefined) window.cancelAnimationFrame(animationFrame);
      flushBufferedChunk();
      // 收到 `done` 后重新加载，用数据库生成的 ID 和时间替换临时消息。
      await Promise.all([loadMessages(activeConversationId), refreshConversations()]);
    } catch (error) {
      message.error(error instanceof Error ? error.message : "生成回答失败。");
      setMessages((items) => items.filter((item) => item.id !== "streaming"));
    } finally {
      if (animationFrame !== undefined) window.cancelAnimationFrame(animationFrame);
      flushBufferedChunk();
      setStreaming(false);
      setStreamingStatus("");
    }
  }

  async function refreshConversations() {
    setConversations(await api.listConversations());
  }

  function renderKnowledgeDocuments(documents: KnowledgeDocument[], emptyText: string) {
    return (
      <List
        dataSource={documents}
        locale={{ emptyText }}
        renderItem={(item) => {
          const canDelete = item.space === "private" || currentUser?.is_admin === true;
          return (
            <List.Item
              className="knowledge-document-item"
              actions={[
                <Tooltip key="retrieval" title="决定该资料是否参与你自己的后续问答检索">
                  <Checkbox
                    checked={item.retrieval_enabled}
                    onChange={(event) => void updateKnowledgeDocumentRetrieval(item, event.target.checked)}
                  >
                    加入问答
                  </Checkbox>
                </Tooltip>,
                <Tooltip key="preview" title="预览">
                  <Button
                    aria-label="预览"
                    type="text"
                    size="small"
                    shape="circle"
                    icon={<EyeOutlined />}
                    disabled={item.status !== "ready"}
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
                    onClick={() => void downloadKnowledgeDocument(item)}
                  />
                </Tooltip>,
                ...(canDelete ? [
                  <Popconfirm
                    key="delete"
                    title="删除此知识库文件？"
                    description="将同时删除原始文件、索引任务、片段和向量，无法恢复。"
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
                ] : []),
              ]}
            >
              <List.Item.Meta
                title={
                  <div className="knowledge-document-title">
                    <Typography.Text className="knowledge-document-name" ellipsis={{ tooltip: item.title }}>
                      {item.title}
                    </Typography.Text>
                    {documentStatusTag(item)}
                  </div>
                }
                description={`${item.file_type.toUpperCase()} · ${item.status === "processing" || item.status === "queued" ? "索引任务进行中" : `${item.chunk_count} 个片段`}`}
              />
            </List.Item>
          );
        }}
      />
    );
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
              actions={[
                <Popconfirm
                  key="delete-conversation"
                  title="删除此会话？"
                  description="会删除消息、会话条件、待确认记忆候选和运行审计，无法恢复。已确认的长期记忆不会删除。"
                  okText="删除"
                  cancelText="取消"
                  okButtonProps={{ danger: true, loading: deletingConversationId === item.id }}
                  disabled={streaming}
                  onConfirm={() => void deleteConversation(item)}
                >
                  <Tooltip title={streaming ? "生成回答期间不能删除会话" : "删除会话"}>
                    <Button
                      type="text"
                      danger
                      size="small"
                      shape="circle"
                      icon={<DeleteOutlined />}
                      aria-label="删除会话"
                      disabled={streaming}
                      onClick={(event) => event.stopPropagation()}
                    />
                  </Tooltip>
                </Popconfirm>,
              ]}
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
                  <div className="message-body">
                    {item.sender_type === "agent"
                      ? <MessageContent
                        content={item.content || (item.id === "streaming" ? "" : "正在思考…")}
                        streaming={item.id === "streaming"}
                      />
                      : item.content}
                  </div>
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
                        检测到你可能希望长期保留：{candidate.content}
                        {candidate.expires_at && `（有效至 ${new Date(candidate.expires_at).toLocaleDateString("zh-CN")}）`}
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
          {currentUser.is_admin ? (
            <Radio.Group
              value={uploadSpace}
              onChange={(event) => setUploadSpace(event.target.value as "private" | "team")}
            >
              <Radio value="private">上传至我的资料</Radio>
              <Radio value="team">上传至团队资料</Radio>
            </Radio.Group>
          ) : (
            <Typography.Text type="secondary">上传的资料仅自己可见，默认加入你的问答检索。</Typography.Text>
          )}
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
            {uploading ? "正在上传文件…" : "上传并后台建立索引"}
          </Button>
          {uploading && <Spin size="small" tip="正在保存文件…" />}
        </section>
        <Tabs
          activeKey={knowledgeTab}
          onChange={(key) => setKnowledgeTab(key as "team" | "private")}
          items={[
            {
              key: "team",
              label: `团队资料（${knowledgeDocuments.filter((item) => item.space === "team").length}）`,
              children: renderKnowledgeDocuments(
                knowledgeDocuments.filter((item) => item.space === "team"),
                "暂无团队资料",
              ),
            },
            {
              key: "private",
              label: `我的资料（${knowledgeDocuments.filter((item) => item.space === "private").length}）`,
              children: renderKnowledgeDocuments(
                knowledgeDocuments.filter((item) => item.space === "private"),
                "暂无私有资料",
              ),
            },
          ]}
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
              任务约束：{auditRecord.context_summary ?? "本轮未使用已确认约束"}
            </Typography.Text>
            <Typography.Text>
              长期记忆：{auditRecord.memory_summary ?? "本轮未采用长期记忆"}
            </Typography.Text>
            <Typography.Text>
              会话短期状态：{auditRecord.conversation_summary_used
                ? `已采用版本 ${auditRecord.conversation_summary_version ?? "-"}（仅用于保持对话连续性）`
                : "本轮未采用"}
            </Typography.Text>
            <Typography.Text>
              会话历史召回：{auditRecord.conversation_history_used
                ? `采用 ${auditRecord.conversation_history_ids.length} 条相关历史片段（仅用于保持对话连续性）`
                : "本轮未采用"}
            </Typography.Text>
            {auditRecord.memory_ids.length > 0 && (
              <Typography.Text type="secondary">记忆 ID：{auditRecord.memory_ids.join("、")}</Typography.Text>
            )}
            {auditRecord.context_actions.length > 0 && (
              <Typography.Text type="secondary">
                约束来源：{auditRecord.context_actions.join("；")}
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
        footer={previewDocument ? (
          <Button icon={<DownloadOutlined />} onClick={() => void downloadKnowledgeDocument(previewDocument)}>
            下载原文件
          </Button>
        ) : null}
        width={760}
      >
        <Typography.Paragraph className="knowledge-preview">
          {previewDocument?.content}
        </Typography.Paragraph>
      </Modal>

    </Layout>
  );
}
