import { useEffect, useMemo, useState } from "react";
import {
  CheckCircleOutlined,
  DatabaseOutlined,
  DeleteOutlined,
  DownloadOutlined,
  EyeOutlined,
  FileTextOutlined,
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
  Typography,
  message,
} from "antd";

import { api } from "./lib/api";
import type { Conversation, KnowledgeDocument, KnowledgeDocumentContent, KnowledgeGroup, Message } from "./types";

const { Sider, Content } = Layout;

const examples = [
  "分析本周 GMV 环比下降原因",
  "生成一份秋季上新活动方案",
  "为商品 A 写 3 个小红书标题",
];

function documentStatusTag(status: string) {
  if (status === "ready") return <Tag color="success">索引已完成，可检索</Tag>;
  if (status === "processing") return <Tag color="processing">正在建立索引</Tag>;
  return <Tag color="warning">待向量化</Tag>;
}

export function App() {
  // MVP 阶段将会话和任务状态集中在此处；页面增多后再引入全局状态管理。
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string>();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(true);
  const [streaming, setStreaming] = useState(false);
  const [knowledgeGroup, setKnowledgeGroup] = useState<string>();
  const [knowledgeOpen, setKnowledgeOpen] = useState(false);
  const [knowledgeDocuments, setKnowledgeDocuments] = useState<KnowledgeDocument[]>([]);
  const [knowledgeGroups, setKnowledgeGroups] = useState<KnowledgeGroup[]>([]);
  const [uploadGroup, setUploadGroup] = useState("活动规则");
  const [uploadFile, setUploadFile] = useState<File>();
  const [uploading, setUploading] = useState(false);
  const [deletingDocumentId, setDeletingDocumentId] = useState<string>();
  const [previewDocument, setPreviewDocument] = useState<KnowledgeDocumentContent>();

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
        "hybrid",
        knowledgeGroup,
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
              {messages.map((item) => (
                <Card key={item.id} className={`message-card ${item.sender_type}`} size="small">
                  <div className="message-title">
                    {item.sender_type === "user" ? <UserOutlined /> : <CheckCircleOutlined />}
                    <Typography.Text strong>{item.sender_type === "user" ? "运营人员" : "Shop Agent"}</Typography.Text>
                  </div>
                  <div className="message-body">{item.content || "正在思考…"}</div>
                  {item.sender_type === "agent" && item.data_references && (
                    <Typography.Text type="secondary" className="message-reference">
                      依据：{item.data_references}
                    </Typography.Text>
                  )}
                </Card>
              ))}
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
              actions={[
                <Button key="preview" type="link" icon={<EyeOutlined />} onClick={() => void previewKnowledgeDocument(item.id)}>预览</Button>,
                <Button key="download" type="link" icon={<DownloadOutlined />} href={api.getKnowledgeDownloadUrl(item.id)}>下载</Button>,
                <Popconfirm
                  key="delete"
                  title="删除此知识库文件？"
                  description="将同时删除原始文件、索引片段和向量，无法恢复。"
                  okText="删除"
                  cancelText="取消"
                  onConfirm={() => void deleteKnowledgeDocument(item)}
                >
                  <Button danger type="link" icon={<DeleteOutlined />} loading={deletingDocumentId === item.id}>删除</Button>
                </Popconfirm>,
              ]}
            >
              <List.Item.Meta
                title={<Space><span>{item.title}</span><Tag>{item.group_name}</Tag>{documentStatusTag(item.status)}</Space>}
                description={`${item.file_type.toUpperCase()} · ${item.chunk_count} 个片段`}
              />
            </List.Item>
          )}
        />
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
