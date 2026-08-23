# Shop Agent 技术方案

## 1. 架构概览

项目采用前后端分离架构。React 前端负责运营工作台；FastAPI 提供 API、SSE 流式响应和数据持久化接口；Agent 层使用 DeepAgent、LangChain 与 LangGraph 编排工具调用；PostgreSQL 保存业务状态。

```text
浏览器（React + TypeScript）
        │ HTTP / SSE
        ▼
FastAPI API 服务
 ├── 会话、知识库文件、审计接口
 ├── SQLAlchemy 异步数据访问
 └── OperationAgent（主 Agent）
       ├── 指标 RAG 工具 / 知识库 RAG 工具
       ├── 运营复盘子 Agent（复杂复盘与归因）
       ├── DeepAgent + LangChain + LangGraph（配置模型后）
       └── 演示模式（未配置模型时）
        │
        ├── PostgreSQL
        ├── PostgreSQL：会话、业务数据、指标与知识库向量
        ├── 宿主机 uploads 卷：知识库原始文件
        ├── 电商业务数据源（后续接入）
        └── 大模型服务（后续按环境配置）
```

## 1.1 指标 RAG 查询流程

经营数据查询采用“指标定义检索 + 受控 SQL 模板”的 RAG 方案，不让 LLM 直接生成或执行 SQL：

```text
用户问题
  -> 为问题生成嵌入向量（优先使用配置的通用嵌入模型）
  -> 在 metric_definitions 中与指标名称、描述、别名和向量进行相似度匹配
  -> 选出相关指标，递归补齐 dependency_codes 中的前置指标
  -> 只执行前置基础指标对应的受控 SQL 模板
  -> 计算派生指标，并将最小结果集作为 Agent 上下文
  -> Agent 基于问题和查询结果生成回答
```

`metric_definitions` 是指标知识的唯一入口。一条定义包含稳定编码、名称、业务描述、别名、前置指标编码、受控 SQL 模板或受控计算公式、嵌入向量和启用状态。基础指标通过模板查询 `daily_metrics`；如支付转化率、客单价和退款率等派生指标只读取其前置指标的结果后计算。

SQL 模板存储在表中以便维护指标口径，但执行前必须与后端登记的只读模板完全匹配；参数仅允许 `source`、`start_date`、`end_date` 绑定传入。计算公式同样只允许后端登记的公式，禁止对数据库中的文本使用 `eval` 或让模型输出 SQL。

嵌入配置采用手动引入的本地模型。模型文件由宿主机下载并挂载到 API 容器，运行时通过 Sentence Transformers 直接加载，不调用外部 Embeddings API。模型未加载或指标向量未重建时，指标 RAG 不查询数据，并记录明确日志。

## 1.2 知识库 RAG 查询流程

上传的 PDF、DOCX、Markdown、TXT 文件按用户填写的 `group_name` 分组，解析成纯文本后按约 800 字符切块（保留 120 字符重叠）；PDF 片段同时记录页码。上传请求在向量化完成前保持 loading 状态，完成后文档为 `ready` 并显示“索引已完成，可检索”；模型不可用时保留为 `pending_embedding`。原始文件持久化在宿主机 `uploads/`，解析正文和片段向量存入 PostgreSQL。

```text
上传文件 + 分组
  -> 校验格式与大小
  -> 提取文本并保存原件
  -> 切分 knowledge_chunks
  -> 本地模型向量化
  -> ready / pending_embedding

综合分析 + 可选知识库分组
  -> 问题向量化（每个问题一次）
  -> 与缓存的“指标 / 知识库 / 综合”意图原型比较相似度
  -> 高置信度且分差足够：只进入对应 RAG
  -> 低置信度或意图接近：同时进入两类 RAG
  -> 指标 RAG 按需执行受控 SQL / 知识库余弦相似度 Top-4 检索
  -> 数据结果 + 文档标题/页码作为并列依据
  -> LLM 归纳结论、建议与验证动作
```

意图路由复用与后续检索相同的本地问题向量，不额外调用 LLM，也不会引入外部 Embeddings API。“指标查询”“知识库问答”“综合分析”三段稳定的原型描述首次使用时向量化并按模型路径、模型标识和设备缓存在 API 进程内。以当前本地中文模型校准后，最高相似度低于 `0.45`，或第一、第二意图的分差小于 `0.06` 时，系统保守地降级为综合检索；这样能减少明确单一问题的无效检索，同时避免误判导致漏掉资料。用户仍可通过 API 的 `mode=metrics` 或 `mode=knowledge` 显式限定来源，前端默认使用自动路由。

本地模型未就绪时，文件仍会保存为 `pending_embedding`，不会参与知识库检索。模型下载完成后执行 `python -m app.reindex_embeddings`，可同时重建指标定义和知识库片段向量。删除资料时，数据库以外键级联删除 `knowledge_chunks`（含向量），随后删除 `uploads/` 中的原文件。

| 环境变量 | 是否必填 | 说明 |
| --- | --- | --- |
| `LOCAL_EMBEDDING_MODEL_PATH` | 使用本地语义检索时必填 | 容器中的模型目录，默认 `/models/bge-small-zh-v1.5` |
| `LOCAL_EMBEDDING_MODEL_ID` | 建议填写 | 模型名称或固定版本，用于判断是否需要重建向量 |
| `LOCAL_EMBEDDING_DEVICE` | 可选 | 推理设备，默认 `cpu` |

当前时间解析支持“最近 7 天”（默认）、“最近 14 天/近 14 天/两周”、“上周”，以及“2026 年 6 月 6 日至 6 月 18 日”这类完整年份的日期区间。新增更多时间范围、渠道或商品维度时，应扩展受控参数解析和模板注册表，不能直接把用户输入拼入 SQL。

## 2. 技术选型

| 层级 | 技术 | 作用 |
| --- | --- | --- |
| 前端 | React、TypeScript、Vite | 构建运营工作台与类型安全的前端代码 |
| 前端 UI | Ant Design | 表单、知识库管理、会话界面与基础数据展示 |
| 后端 | Python 3.12、FastAPI、Uvicorn | REST API、SSE 流式回答与健康检查 |
| Agent | DeepAgent、LangChain、LangGraph | Agent 执行、多步骤编排、模型与工具抽象 |
| 数据库 | PostgreSQL | 保存会话、消息、业务数据、指标定义与知识库片段 |
| ORM | SQLAlchemy、asyncpg、Alembic | 异步访问 PostgreSQL 与管理表结构迁移 |
| 部署 | Docker、Docker Compose | 单机容器化交付与本地一致运行环境 |
| 测试 | Pytest、HTTPX | API 与业务逻辑测试 |

## 3. 服务与职责

### 前端

- 管理会话、发送运营问题并流式展示回答。
- 展示回答中的数据依据、资料引用与建议。
- 通过 REST API 获取历史会话和知识库文件，通过 SSE 获取实时回答。

### API 服务

- 保存会话、消息、指标定义、知识库文件和检索片段。
- 将用户输入交给 `OperationAgent`，再以 SSE 转发回答片段。
- 暴露 `/health` 用于 Docker 健康检查。

### Agent 服务

- 主 Agent 根据本地语义路由调用 `query_metric_rag`、`query_knowledge_rag` 两个受控工具；工具自行创建短生命周期数据库会话，不向模型暴露连接或任意 SQL。
- 涉及活动复盘、经营归因、效果评估和优化建议时，主 Agent 通过 DeepAgent `task` 委派给 `operation_review_agent`；子 Agent 只拥有同一批 RAG 工具。项目同时显式覆盖 DeepAgent 默认的 `general-purpose` 子 Agent，防止框架自动附加更宽的能力。
- 主 Agent 和两个子 Agent 均把框架文件系统能力限制为只读 `read_file`；不提供文件写入、删除或命令执行工具。
- 配置 `LLM_API_KEY` 和 `LLM_MODEL` 后使用 DeepAgent 执行 LangChain/LangGraph Agent 流程；模型不可用时，主 Agent 仍按路由调用同一批工具后进入演示回答。
- 未配置模型时使用演示模式，保证本地开发和 Docker 验收不依赖密钥。
- 后续通过工具适配器接入商品、订单、流量及推广数据源；数据结论必须带数据范围与查询时间。

### 模型服务

项目不绑定任何特定模型或厂商。当前适配器使用 OpenAI 兼容协议，配置项如下：

| 环境变量 | 是否必填 | 说明 |
| --- | --- | --- |
| `LLM_API_KEY` | 启用真实 Agent 时必填 | 模型服务的访问密钥 |
| `LLM_MODEL` | 启用真实 Agent 时必填 | 服务端支持的模型名称 |
| `LLM_BASE_URL` | 可选 | 兼容模型服务的 API 地址；未设置时由 SDK 使用其默认地址 |

只有同时配置 `LLM_API_KEY` 和 `LLM_MODEL` 才会调用真实模型；否则进入不依赖密钥的演示模式。

## 4. 数据模型

| 表 | 用途 |
| --- | --- |
| `conversations` | 会话标题及创建、更新时间 |
| `messages` | 用户与 Agent 消息、回答状态及数据引用 |
| `tool_calls` | Agent 工具调用摘要、结果、耗时和错误信息 |
| `products` | 内置模拟商品资料，后续可替换为真实商品数据源 |
| `daily_metrics` | 近 14 天按商品和渠道汇总的模拟经营指标 |
| `metric_definitions` | 指标名称、描述、别名、依赖、受控模板、公式和 RAG 向量缓存 |
| `knowledge_documents` | 原始文件元信息、分组、解析正文、处理状态和片段数 |
| `knowledge_chunks` | 文件片段、PDF 页码、向量与所用模型标识 |

## 5. 配置原则

- 运行配置由环境变量注入，示例值存放于 `.env.example`。
- 不将模型密钥、数据库密码或外部数据源凭据提交到仓库。
- `DATABASE_URL` 默认指向 Docker Compose 中的 PostgreSQL 服务。
- `LLM_API_KEY`、`LLM_MODEL` 和可选的 `LLM_BASE_URL` 均通过环境变量配置；不预设任何模型。
- `KNOWLEDGE_UPLOAD_DIR` 默认 `/uploads`，由 Docker 映射为宿主机 `uploads/`；上传原件不写入镜像或数据库临时目录。
- 当前模型适配器采用 OpenAI 兼容协议，因此可配置支持该协议的模型服务地址和模型名称，而不绑定特定厂商；未配置密钥和模型时启用演示模式。

## 6. Docker 部署

Docker Compose 包含四个常驻服务和一个按需工具服务：

- `frontend`：构建并提供 React 静态页面。
- `api`：运行 FastAPI 与 Agent 服务。
- `adminer`：提供浏览器访问的 PostgreSQL 管理界面，仅用于本地查看和排查数据。
- `db`：运行 PostgreSQL，并使用命名数据卷保存数据。

前端镜像使用 `package-lock.json` 和 `npm ci` 安装依赖，以确保可复现构建。`frontend/.dockerignore` 会排除本地 `node_modules` 与 `dist`，避免将开发产物传入镜像构建上下文。

首次部署步骤：

1. 复制 `.env.example` 为 `.env`，按需填入模型配置。
2. 执行 `docker compose up -d --build`。
3. 访问 `http://localhost:5173`；API 健康检查地址为 `http://localhost:8000/health`；数据库管理界面为 `http://localhost:8081`。

Adminer 登录时选择 PostgreSQL，服务器填写 `db`，用户名和数据库名均为 `shop_agent`，密码使用 `.env` 中的 `POSTGRES_PASSWORD`。默认数据库端口不暴露到宿主机，Adminer 通过 Docker 内部网络连接数据库。

API 容器启动时会先执行数据库迁移，再运行 `python -m app.seed`。初始化脚本以增量、幂等方式补齐数据，不覆盖已有记录：通用模拟数据包含 5 个商品和最近 14 天的访客、支付订单、GMV、退款订单；另根据根目录 `test/` 的活动文档补充春季上新活动期数据（GMV 86,400 元、支付订单 312 单）、618 预热/正式/返场期数据（按规则节奏构造的模拟数值）及七夕礼赠活动期的礼盒、组合款渠道数据。可通过 `GET /api/metrics/overview` 查看最近两周汇总。知识库原始文件保存在 `uploads/` 挂载目录，重建 API 容器不会丢失；不要随意删除该目录。

本地语义向量模型不随镜像或 Git 仓库提交。首次使用依次执行 `docker compose build api model-download`、`docker compose run --rm model-download`、`docker compose up -d api` 和 `docker compose exec api python -m app.reindex_embeddings`。模型会下载到宿主机的 `models/bge-small-zh-v1.5` 并以只读卷挂载到 API 容器；下载完成后重建指标定义和知识库片段向量。API 运行期间不会下载模型或调用外部嵌入 API。

镜像为 CPU 推理预装 PyTorch CPU 轮子，不会安装 CUDA 运行库；如后续需要 GPU 推理，应单独提供 GPU 镜像与 `LOCAL_EMBEDDING_DEVICE` 配置，而不是在默认镜像中混入 CUDA 依赖。

停止服务使用 `docker compose down`。不要加 `-v`，否则会删除本地数据库卷。数据库备份、HTTPS 和生产域名在正式部署前再按目标环境补充。

## 7. 运行日志与排查

使用 `docker compose up -d` 会将服务放入后台，终端不会持续输出日志。使用以下命令查看：

```powershell
docker compose logs -f
docker compose logs -f api
docker compose logs -f frontend
docker compose logs -f db
```

- `api` 默认输出 Uvicorn 的启动、访问与异常日志。
- `frontend` 由 Nginx 提供静态页面，通常在浏览器发起请求后输出访问日志。
- `db` 输出 PostgreSQL 的启动与数据库错误日志。

健康检查与服务状态可通过 `docker compose ps` 和 `http://localhost:8000/health` 确认。

## 8. 后续演进

- 接入真实业务数据源，并以工具适配器隔离不同平台的查询差异。
- 根据并发与异步任务需求，再评估 Redis 与任务队列。
- 按 PRD 的范围补充复盘、运营指标和人工确认的外部操作流程。
