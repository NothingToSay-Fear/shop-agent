# Shop Agent 技术方案

## 1. 架构概览

项目采用前后端分离架构。React 前端负责运营工作台；FastAPI 提供 API、SSE 流式响应和数据持久化接口；Agent 层使用自建持久化 Plan-and-Execute、LangChain 与 LangGraph 编排受控能力；PostgreSQL 保存业务状态。

```text
浏览器（React + TypeScript）
        │ HTTP / SSE
        ▼
FastAPI API 服务
 ├── 本地账号认证、会话、知识库文件、单条回答审计接口
 ├── SQLAlchemy 异步数据访问
 └── OperationAgent（主 Agent）
       ├── 指标 RAG 工具 / 知识库 RAG 工具 / 联网搜索工具
       ├── 运营复盘子 Agent（复杂复盘与归因）
       ├── LangChain 聊天模型 + LangGraph（配置模型后）
       └── 受控证据回退（未配置模型或调用失败时）
        │
        ├── PostgreSQL
        ├── PostgreSQL：会话、业务数据、指标与知识库向量
        ├── 宿主机 uploads 卷：知识库原始文件
        ├── 电商业务数据源（后续接入）
        ├── 大模型服务（后续按环境配置）
        └── Tavily 搜索 API（按环境变量启用）
```

## 1.1 指标 RAG 查询流程

经营数据查询采用“数据查询计划 + 指标定义检索 + 受控 SQL 模板”的方案，不让 LLM 直接生成或执行 SQL：

```text
用户问题
  -> 已有跨轮任务时，主工作流委派 `DataQueryAgent.plan()` 从结构化约束生成指标、时间、维度与数据缺口计划
  -> 用户明确指定的指标作为执行契约完整保留，不再经过 RAG 排名截断
  -> 保留原始问题，并由已配置 LLM 最多生成 2 条受限检索改写（不可改变时间、活动或指标条件）
  -> 每条 Query 分别在 metric_definitions 中执行向量检索与中文 BM25 检索
  -> 使用 RRF 按排名融合、按 metric_code 去重，并保留业务别名的强命中信号
  -> 选出相关指标，递归补齐 dependency_codes 中的前置指标
  -> 从明确活动期或日期范围构造 1 至 4 个 MetricQueryUnit
  -> 对每个单元只执行前置基础指标对应的受控 SQL 模板
  -> 分别计算派生指标与跨单元确定性差异，并将最小结果集作为 Agent 上下文
  -> Agent 基于问题和查询结果生成回答
```

`metric_definitions` 是指标知识的唯一入口。一条定义包含稳定编码、名称、业务描述、别名、前置指标编码、受控 SQL 模板或受控计算公式、嵌入向量和启用状态。基础指标通过模板查询 `daily_metrics`；如支付转化率、客单价和退款率等派生指标只读取其前置指标的结果后计算。`DataQueryAgent` 是独立的内部子 Agent，提供 `plan()`、`execute()` 与 `observe()`：前者只产生指标编码、已经确认的 ISO 时间范围、所需维度、能力缺口与下钻授权；后者仅可调用已持久化 Action 授权的 `query_metric_rag` 只读工具；观察阶段只能提出图谱中已登记的后续指标，不能生成 SQL，也不能获得数据库连接。已明确的 `metric_codes` 直接驱动查询，只有自由问答中未形成任务约束时才使用指标 RAG 选择指标。对于“GMV 为什么下降”等已授权归因型指标问题，主 Agent 会委派数据子 Agent 扩展同口径的支付订单数、访客数、支付转化率与客单价，随后委派复盘子 Agent 依据已验证证据完成驱动拆解；纯指标对比不自动扩大到上述驱动指标。当前数据源支持按天、渠道、商品/SKU 聚合：计划识别“按渠道”“各 SKU”“高退款商品”等请求后，仅通过维度白名单构造聚合；渠道名、商品名和 SKU 均在运行时从 `daily_metrics`、`products` 的实际值匹配，不在代码中写死业务值。商品退款率排行可以直接计算，但订单退款原因、退款金额/时点及活动规则关联仍会随结果明确标为未接入能力。未给出日期且没有可继承会话条件时，系统以 `Asia/Shanghai` 当前日期为上限，从业务数据中选取最近可用日期并向前查询 7 天；未来日期或误入库数据不会成为默认范围。

指标归因关系存储在 `metric_analysis_drivers`：每条启用的有向边表示“目标指标 → 直接驱动指标”，并带有顺序和关系类型。默认边仅由 Alembic 迁移在首次部署时创建；运行期只读取该表来扩展因果查询计划，不会补写或恢复被运营配置删除的边。图谱边不授予 SQL 权限，所有指标仍须在 `metric_definitions` 中启用并通过既有受控模板或公式校验。

每轮任务型请求以 `ConversationTask.effective_constraints` 为唯一执行条件源：其中的 `execution_intent` 将“操作目标（查询/对比/归因/复盘）”“所需证据来源（指标/资料）”“输出范围”和“是否授权指标下钻”分开保存。主 Agent 据此编排子 Agent，而非用单一任务类型互斥路由；旧的 `conversation_contexts` 不再参与任务执行。会话条件重置会取消包括已完成任务在内的可继承任务，防止旧约束被后续输入重新启用。

SQL 模板存储在表中以便维护指标口径，但执行前必须与后端登记的只读模板完全匹配；参数仅允许 `source`、`start_date`、`end_date` 绑定传入。计算公式同样只允许后端登记的公式，禁止对数据库中的文本使用 `eval` 或让模型输出 SQL。

指标工具每轮仍只真实执行一次，但该次调用可承载多个受控查询单元。若问题明确包含多个日期范围，或同时提到多个已登记活动（当前为 618、七夕、春季上新），`MetricQueryPlan` 会按原文顺序构造对应范围；它优先于会话中继承的单一 `start_date/end_date`，避免“对比 618 和七夕”被旧会话条件缩窄为一个活动。单次最多 4 个单元，超过时整体返回可解释的空结果，不执行部分查询。每个单元使用同一批指标定义、依赖关系和受控 SQL 模板；结果包含各区间数值，以及后续区间相对首个区间的数值差异（比例指标使用百分点）。归因和建议仍只由 Agent 基于这些已验证依据生成。

嵌入配置采用手动引入的本地模型。模型文件由宿主机下载并挂载到 API 容器，运行时通过 Sentence Transformers 直接加载，不调用外部 Embeddings API。指标定义数量较少、别名与依赖关系属于强业务规则，因此不使用 CrossEncoder 精排；向量模型不可用时，仍可由 BM25 在已启用的指标定义中进行受限召回，最终 SQL 安全边界不变。

## 1.2 知识库 RAG 查询流程

上传的 PDF、DOCX、Markdown、TXT 文件归入 `private` 或 `team` 资料空间。私有资料仅上传者可见且默认加入其问答检索；团队资料由管理员上传和删除，所有已登录用户可见但默认不参与任何人的问答。解析器优先按 Markdown/DOCX 章节、段落、列表和表格分块；PDF 保留页码范围。片段以约 500～700 字符为目标、900 字符为上限，只有硬切超长内容时保留约 100 字符重叠。后台 Worker 还会利用同一份本地嵌入模型识别相邻结构块的主题跳变。原始文件持久化在宿主机 `uploads/`，解析正文、标题路径、内容类型、页码范围、pgvector 向量和中文分词词项存入 PostgreSQL。

```text
上传文件 + 资料空间
  -> 校验格式与大小
  -> 保存原件、knowledge_documents(queued) 与 knowledge_index_jobs(queued)
  -> 私有资料同时创建“上传者已勾选”记录；团队资料等待用户自行勾选
  -> API 立即返回，前端轮询任务进度
  -> 独立 knowledge-worker 领取任务、解析并分批向量化
  -> 原子标记 ready / failed；临时失败按退避策略重试

综合分析 + 当前用户已勾选资料
  -> 问题向量化（每个问题一次）
  -> 与缓存的“指标 / 知识库 / 综合 / 联网 / 内外部综合”意图原型比较相似度
  -> 高置信度且分差足够：只进入对应 RAG
  -> 低置信度或意图接近：同时进入两类 RAG
  -> 指标 RAG 按需执行受控 SQL / 知识库多 Query 混合检索与精排
  -> 数据结果 + 文档标题/页码作为并列依据
  -> LLM 归纳结论、建议与验证动作
```

意图路由复用与后续检索相同的本地问题向量，不额外调用 LLM，也不会引入外部 Embeddings API。执行计划包含指标或知识库工具时，工作流仅执行一次受限 Query 改写和扩展 Query 向量化，并将同一份请求级检索上下文注入两个工具；纯联网路径不创建该上下文。“指标查询”“知识库问答”“综合分析”“联网检索”“内外部综合分析”五段稳定的原型描述首次使用时向量化并按模型路径、模型标识和设备缓存在 API 进程内。高置信度的“内外部综合分析”会同时调用三类工具；最高相似度低于 `0.45`，或第一、第二意图的分差小于 `0.06` 时，系统保守地降级为仅检索内部的综合分析，避免不确定问题造成不必要的联网搜索消耗。用户仍可通过 API 的 `mode=metrics`、`mode=knowledge` 或 `mode=web` 显式限定来源，前端默认使用自动路由。

知识库检索的完整阶段为：保留原始问题 → 可选的最多 2 条 Query 改写 → 每条 Query 在 PostgreSQL 中执行 pgvector HNSW 向量候选召回与 GIN 中文分词全文候选召回（各 Top-40）→ 服务端 RRF（`k=60`）融合并按 `knowledge_chunk.id` 去重 → 仅加载最多 30 个候选完整文本进入本地 CrossEncoder 精排 → 过滤分数低于 `0.35` 的片段 → 返回 Top-4 片段。候选 SQL 始终同时限制当前用户、资料勾选状态和资料空间，不能绕过私有资料边界。改写仅影响“找什么资料”，不会改变原始问题或受控指标 SQL 的日期参数；LLM 未配置、改写失败或输出非 JSON 时仅使用原问题。`0.35` 是当前 `BAAI/bge-reranker-base` 的保守初值，应随真实标注评测集的分数分布调整；精排模型未挂载、加载失败时不阻断问答，保留 RRF 排序作为确定性降级。

本地模型未就绪时，文件仍会保存为 `pending_embedding`，不会参与知识库检索。模型下载完成后执行 `python -m app.scripts.rebuild_embeddings`，可同时重建指标定义向量，并从原始资料重新解析标题、段落、列表、表格和页码范围后重建知识库片段、向量与词面索引。删除资料时，数据库以外键级联删除 `knowledge_chunks`（含向量），随后删除 `uploads/` 中的原文件。

资料入库不在上传请求内执行。API 接收文件后，在同一数据库事务中创建 `knowledge_documents` 和 `knowledge_index_jobs`，随后返回 `202 Accepted`；`knowledge-worker` 使用 PostgreSQL 行锁（`FOR UPDATE SKIP LOCKED`）领取任务，支持多个 Worker 并行而不重复处理。任务依次记录 `queued`、`parsing`、`embedding`、`completed` 或 `failed` 状态及已处理片段数；Worker 重启时会将遗留 `running` 任务重新入队。向量按默认 32 个片段一批持久化，文档处于 `processing` 时不参与检索；删除文档后，Worker 在每批提交前检测其状态，避免继续写入已删除资料。临时错误最多重试 3 次并指数退避，解析错误等不可恢复错误直接标记失败。

知识库可见性与检索范围分离：`knowledge_documents.space` 决定当前用户能否查看资料，`owner_user_id` 保证私有资料只对上传者开放；`user_knowledge_document_settings` 则保存每位用户对每份可见资料的 `retrieval_enabled` 选择。知识库 RAG 在 SQL 查询中同时限定当前 `user_id`、已勾选状态，以及“团队资料或本人私有资料”条件；因此前端勾选只是交互入口，服务端检索本身不能跨越资料边界。

| 环境变量 | 是否必填 | 说明 |
| --- | --- | --- |
| `LOCAL_EMBEDDING_MODEL_PATH` | 使用本地语义检索时必填 | 容器中的模型目录，默认 `/models/bge-small-zh-v1.5` |
| `LOCAL_EMBEDDING_MODEL_ID` | 建议填写 | 模型名称或固定版本，用于判断是否需要重建向量 |
| `LOCAL_EMBEDDING_DEVICE` | 可选 | 推理设备，默认 `cpu` |
| `LOCAL_RERANKER_MODEL_PATH` | 启用知识库精排时必填 | 本地 CrossEncoder 目录，默认 `/models/bge-reranker-base` |
| `LOCAL_RERANKER_MODEL_ID` | 建议填写 | 精排模型名称或固定版本，默认 `BAAI/bge-reranker-base` |
| `LOCAL_RERANKER_DEVICE` | 可选 | 精排推理设备，默认 `cpu` |
| `KNOWLEDGE_RERANKER_MIN_SCORE` | 可选 | 知识库精排最低相关度，默认 `0.35`；仅在精排模型正常评分时生效 |
| `RAG_QUERY_EXPANSION_MAX_QUERIES` | 可选 | LLM 检索改写数量，默认 `2`；设为 `0` 时禁用改写 |
| `KNOWLEDGE_INDEX_BATCH_SIZE` | 可选 | Worker 每批向量化的片段数，默认 `32` |
| `KNOWLEDGE_CHUNK_SEMANTIC_SIMILARITY_THRESHOLD` | 可选 | 相邻结构块语义相似度低于该值、且当前片段已达 300 字符时提前切分，默认 `0.55`；嵌入模型不可用时跳过该辅助判断 |
| `KNOWLEDGE_INDEX_POLL_SECONDS` | 可选 | Worker 空闲轮询任务间隔，默认 `1` 秒 |
| `KNOWLEDGE_INDEX_MAX_ATTEMPTS` | 可选 | 索引任务最大尝试次数，默认 `3` |

未给出时间时，指标查询默认最近 7 个可用业务日。`temporal_interpreter` 先用本地嵌入模型将当前问题与“时间变化 / 高置信无时间续问”原型作语义比较：只有“无时间”分数达到 `0.58` 且领先时间分数至少 `0.08` 时跳过时间 LLM；不确定或疑似时间问题一律调用受限 LLM。LLM 返回 `no_time`、`resolved`、`clarify` 或 `forecast`，其中 `resolved` 必须携带 1 至 4 个 ISO 起止日期区间。提示词提供当前服务日期、最新可用业务数据日、自然周规则和必要的待续任务上下文，因此可理解任意时间说法而不依赖词表。后端只接受格式正确、起始不晚于结束、未晚于最新实际数据日且数量不超限的范围；随后将已验证的 `date` 参数交给受控指标计划。模型不可用、低置信度或输出异常时直接追问时间范围，绝不使用关键词推断日期，也不会把用户文本拼入 SQL。

## 1.3 联网搜索流程

联网搜索只用于补充具有时效性的公开信息，不替代内部业务数据或用户上传资料。首期通过现有 `httpx` 调用 Tavily 的 `POST /search` 接口，固定使用 `basic` 深度、按配置返回最多 20 条结果且不请求网页原文，避免模型自行选择搜索成本或抓取大量页面内容。Tavily API 采用 Bearer 密钥鉴权并返回按相关性排序的标题、链接和摘要。[官方接口说明](https://tavilyai.mintlify.app/documentation/api-reference/endpoint/search)

```text
用户问题
  -> 本地语义路由命中“联网检索”或“内外部综合分析”，或 API 显式指定 mode=web
  -> 纯外部问题只调用 search_web(question)；内外部综合问题并行调用三类受控工具
  -> Tavily 基础搜索（超时、数量均由环境变量限制）
  -> 过滤为 HTTP(S) 标题、链接、限长摘要
  -> 标记为“不可信网页摘要”后交给 Agent
  -> Agent 输出结论及可核验链接
```

网页片段可能包含提示注入、错误信息或过期内容，因此工具不返回网页原文，不执行网页中的指令、链接、下载或写操作；系统提示也要求模型仅把它作为外部参考。搜索结果仅存在于当前回答上下文和消息的来源摘要，不自动向 `knowledge_documents` 或 `knowledge_chunks` 写入内容。未配置密钥、网络超时、HTTP 错误或无结果时，工具返回资料不足，不阻断指标 RAG 或知识库 RAG。

| 环境变量 | 是否必填 | 说明 |
| --- | --- | --- |
| `WEB_SEARCH_PROVIDER` | 可选 | 联网搜索提供方；首期仅支持 `tavily`，默认该值 |
| `WEB_SEARCH_API_KEY` | 启用联网搜索时必填 | Tavily API 密钥；为空时工具禁用且不发起网络请求 |
| `WEB_SEARCH_MAX_RESULTS` | 可选 | 每次搜索结果数，默认 `5`，运行时限制为 `1` 至 `20` |
| `WEB_SEARCH_TIMEOUT_SECONDS` | 可选 | 单次搜索超时秒数，默认 `10` |

## 2. 技术选型

| 层级 | 技术 | 作用 |
| --- | --- | --- |
| 前端 | React、TypeScript、Vite | 构建运营工作台与类型安全的前端代码 |
| 前端 UI | Ant Design | 表单、知识库管理、会话界面与基础数据展示 |
| 后端 | Python 3.12、FastAPI、Uvicorn | REST API、SSE 流式回答与健康检查 |
| Agent | 自建 Plan-and-Execute、LangChain、LangGraph | 持久化任务执行、模型与工具抽象 |
| 联网检索 | Tavily Search API、HTTPX | 受控获取公开且有时效性的信息；不引入额外 SDK |
| 知识库候选召回 | PostgreSQL、pgvector、HNSW、GIN 全文索引、jieba | 在数据库按可见资料范围快速返回语义和词面候选，避免 API 读取并遍历全部切片向量 |
| 混合融合 | RRF | 按排名融合数据库返回的向量与词面候选，并按片段 ID 去重 |
| 本地精排 | Sentence Transformers CrossEncoder、`BAAI/bge-reranker-base` | 对 RRF 去重后的知识库候选进行问题—片段相关性判断；模型缺失时降级为 RRF 顺序 |
| 数据库 | PostgreSQL | 保存会话、消息、业务数据、指标定义与知识库片段 |
| 本地认证 | Python 标准库 `hashlib.scrypt`、FastAPI HTTP Bearer | 保存带盐密码散列，签发并校验可撤销的登录令牌；不新增认证依赖 |
| ORM | SQLAlchemy、asyncpg、Alembic | 异步访问 PostgreSQL 与管理表结构迁移 |
| 部署 | Docker、Docker Compose | 单机容器化交付与本地一致运行环境 |
| 测试 | Pytest、HTTPX | API、业务逻辑与固定 Agent 场景评估 |

## 3. 服务与职责

### 前端

- 管理会话、手动删除无效会话、发送运营问题并流式展示回答。
- 展示回答中的数据依据、资料引用与建议。
- 通过 REST API 获取历史会话和知识库文件，通过 SSE 获取实时回答。

### API 服务

- 提供账号注册、登录、退出和当前用户接口。浏览器以 `Authorization: Bearer <token>` 调用受保护接口；令牌原文只保存在浏览器本地存储，服务端仅保存 SHA-256 摘要。
- 保存会话、消息、会话结构化条件、指标定义、知识库文件和检索片段。
- 将用户输入交给 `OperationAgent`，再以 SSE 转发回答片段。
- 暴露 `/health` 用于 Docker 健康检查。

### Agent 服务

- `OperationAgent` 只负责调用正式工作流并实时转发模型生成中的 SSE 文本片段；不提供外部注入检索上下文的旁路，因此不直接处理路由、工具调用、模型创建或回答文案。
- `AgentWorkflow` 负责按持久化 Action Plan 执行、校验工具轨迹和来源。其“执行一批 Action → 评估证据 → 必要时重规划并执行下一批”的循环由 LangGraph `StateGraph` 驱动；图状态仅保存本次运行的待执行 Action、已完成键和受控观察结果，PostgreSQL 仍是步骤、预算、重规划版本与审计的唯一事实来源。`AnswerGenerator` 使用无工具的 LangChain 聊天模型完成最终表达与受控证据回退；`PromptBuilder` 只生成系统提示词与已执行计划约束。

```text
OperationAgent（流式输出）
  -> AgentWorkflow（路由、执行计划、工具调用、来源校验）
       -> PromptBuilder（系统提示词、已执行计划约束）
       -> AnswerGenerator（LangChain 聊天模型 / 受控证据回退）
       -> app/agent/execution_plan.py（计划与校验）
       -> app/agent/tools/（指标、知识库、联网搜索、工具注册表）
```

- 主 Agent 先生成并持久化受约束的 Action Plan；路由仅用于选择初始能力，不再直接决定固定工具清单。执行器按数据库中的依赖、顺序、能力要求和预算调度动作，任一工具失败不会阻断其他独立动作。
- `validate_execution_plan` 要求每个必调工具都有 `success`、`empty`、`skipped` 或 `failed` 终态；`success` 时还必须有与工具匹配的引用 ID（`metric:*`、`knowledge_chunk:*`、HTTP(S) URL）。校验失败时不会调用模型生成事实性结论。
- 工具注册表位于 `app/agent/tools/registry.py`，声明稳定工具名、每轮最多真实调用次数和引用规则。知识库与联网检索每轮最多一次；指标查询可在主 Agent 已追加的受控 Action 内执行有限次。超出登记额度的调用会返回 `tool_call_limit_reached`，不产生额外数据库查询或联网请求。
- 前两类工具自行创建短生命周期数据库会话，不向模型暴露连接或任意 SQL；联网工具只调用固定的 Tavily 搜索端点。工具代码按职责位于 `app/agent/tools/`：`metric_rag.py`、`knowledge_rag.py`、`web_search.py` 负责具体检索，`tracker.py` 汇总本轮依据与调用额度，`__init__.py` 仅组合工具供 Agent 使用。
- 涉及活动复盘、经营归因、效果评估和优化建议时，主工作流在指标与资料证据校验通过后，显式调用无工具的 `ReviewAgent`；它只能消费已验证证据包，不能重复查询、读写文件或补充外部事实。
- 配置 `LLM_API_KEY` 和 `LLM_MODEL` 后，最终生成层和复盘子 Agent 均使用 LangChain 聊天模型；模型只能在系统已完成并校验执行计划后基于受控上下文总结。模型未配置或调用失败时，主 Agent 不再生成模拟回答，仅返回同轮已验证的工具依据。
- `ContextBuilder` 在保存用户消息后合并会话中已确认的活动、日期、指标提示和分析目标。当前轮明确条件覆盖旧值；支持清除或重置；未出现的条件可继承。它统一识别中文、ISO 和 `8/16–8/22` 等月/日日期范围；时间继承是否清除不再由关键词决定，而是消费 `temporal_interpreter` 的受限判定，只给路由和检索提供“原问题 + 条件摘要”。
- `ConversationTask` 是独立于短期记忆的会话任务状态。它保存任务类型、状态、原任务问题的长度受限副本、后续补充、缺失槽位和来源消息 ID；当前覆盖指标查询/对比、知识库问答、综合分析和复盘。存在既有任务时，`task_interpreter` 先将“既有任务快照 + 当前问题”交给已配置 LLM，并只接受 `continue`、`revise`、`replace`、`cancel`、受限变更槽位和置信度组成的 JSON；低置信度、格式异常或模型不可用时退回确定性规则。`temporal_interpreter` 先由本地语义门控判断是否需要 LLM；需要时才接收受限 JSON，并可直接给出规整后的 ISO 查询区间，后端校验后写入任务帧。待补充任务的原问题会作为最小上下文传入时间解析器，使“本周是指 2026-09-09 至 2026-09-15”这类补充能够由模型完整列出比较所需的全部区间，后端不再派生上周或月度范围。新时间结果会关闭旧任务或旧会话时间范围，避免新问题继续使用上一轮日期。资料和综合任务的续问复用原任务目标与已完成路由，但总会重新执行本轮 RAG，绝不把上一轮检索结果拼成当前事实。新任务、取消或“重置会话条件”会关闭未完成任务。
- 会话中的 `start_date`、`end_date` 不再由指标 RAG 从摘要文本二次解析：工作流将其作为 `date` 类型参数传给 `query_metric_rag`，再由 `MetricQueryConstraints` 绑定到受控 SQL 模板。单一继承范围作为默认单元；本轮明确出现多个活动或多个日期范围时，`MetricQueryPlan` 以本轮多单元范围为准。条件摘要仅用于语义路由与可解释展示。
- 指标 RAG 不解析“今天”“本月”“自然周”等时间词，也不派生对比基期；它只消费时间解析器已经校验的显式日期，或在 `no_time` 时使用默认最近 7 个可用业务日。模型要求未来区间或返回超出可用数据边界时，后端在进入指标 RAG 前追问或拒绝，因此不会把半月数据冒充整月答案。
- `ConversationSummary` 是同会话短期状态，而非仅保存摘要：它保存压缩讨论摘要、主题/要点/待验证项和有限的最近消息窗口。每轮回答完成后，API 直接将本轮精简窗口写入该状态；它不反查 `messages` 表。状态行以 `FOR UPDATE` 串行化同一会话的追加，首次创建时先锁定会话父行，避免并发请求丢失轮次。状态估算 token 数达到 `CONVERSATION_MEMORY_COMPACT_THRESHOLD` 后创建 `ConversationSummaryJob`；独立 `conversation-summary-worker` 通过 `FOR UPDATE SKIP LOCKED` 领取任务，并绑定租约令牌。Worker 中断后，租约到期的任务会自动重新排队；执行异常按指数退避重试，达到 `CONVERSATION_SUMMARY_MAX_ATTEMPTS` 后才标记失败。模型不可用或返回非 JSON 时采用长度受限的确定性降级摘要，绝不阻塞本轮回答。每轮回答生成均读取短期状态，不依赖历史指代关键词；路由、问题扩展、RAG 候选召回、工具入参和 SQL 仍只使用当前问题与 `ConversationContext` 的结构化条件。`agent_runs` 记录短期状态版本和是否实际采用，压缩状态记录来源消息 ID 与 `run_id`，便于追溯。
- `ConversationHistoryUnit` 是独立于短期状态的会话内历史 RAG 索引：每个完成的一问一答生成一个长度受限单元，立即写入 jieba 词项，Worker 再异步补齐 pgvector 向量。仅当 `ConversationSummary` 已有压缩摘要时，API 才按当前问题在同一会话内查询 HNSW 向量候选和 GIN 全文候选、RRF 融合并按需 CrossEncoder 精排；没有精排或嵌入模型时分别保留 RRF 或全文召回。命中的最多两条历史单元仅传入生成提示词，不参与路由、查询扩展、工具入参和 SQL；`agent_runs` 只审计其 ID 与采用标记。首期不回填旧会话单元，也不承诺逐字恢复早期原文。
- 未配置模型时仍可完成受控检索与审计，但只返回已验证依据；没有依据时明确拒绝生成回答。本地开发和 Docker 验收可不依赖模型密钥。
- 后续通过工具适配器接入商品、订单、流量及推广数据源；数据结论必须带数据范围与查询时间。

### 模型服务

项目通过 `LLMProviderFactory` 将不同供应商收口为 LangChain 的统一 `BaseChatModel`，因此主 Agent、复盘 Agent、时间解析、任务关系判断、检索改写和会话摘要都复用同一模型入口；业务编排、RAG 和记忆模块不感知厂商差异。

支持 `openai`（默认，包含 OpenAI-compatible 服务）、`anthropic`、`google` 与 `ollama`。各模型在工具调用、结构化输出、上下文长度和推理表现上仍有差异，生产配置应选择已验证支持流式输出和工具调用的模型。

| 环境变量 | 是否必填 | 说明 |
| --- | --- | --- |
| `LLM_PROVIDER` | 可选 | `openai`、`anthropic`、`google` 或 `ollama`；默认 `openai` |
| `LLM_API_KEY` | 启用真实 Agent 时必填 | 模型服务的访问密钥 |
| `LLM_MODEL` | 启用真实 Agent 时必填 | 服务端支持的模型名称 |
| `LLM_BASE_URL` | 可选 | `openai` 兼容服务或 Ollama 的 API 地址；未设置时由 SDK 使用其默认地址 |

除 Ollama 可不配置密钥外，只有同时配置 `LLM_API_KEY` 和 `LLM_MODEL` 才会调用模型生成结论；否则系统只返回本轮已验证依据，不进行关键词模板生成。

## 4. 数据模型

| 表 | 用途 |
| --- | --- |
| `conversations` | 会话标题及创建、更新时间 |
| `users` | 本地登录账号、显示名称和 scrypt 密码散列 |
| `auth_tokens` | 可撤销的登录令牌摘要、归属用户及过期时间 |
| `conversation_contexts` | 会话内已确认的活动、时间范围、指标提示、分析目标及字段来源消息 ID |
| `conversation_tasks` | 可跨轮补充的任务类型、状态、路由、任务帧、缺失槽位、来源消息 ID 与完成时间；不保存工具结果 |
| `conversation_summaries` | 同一会话的短期状态：压缩摘要、主题、要点、待验证项、有限最近窗口、token 估算、来源和版本；仅用于回答连续性 |
| `conversation_summary_jobs` | 会话摘要异步生成任务的状态、来源回答、失败原因和时间 |
| `conversation_history_units` | 同会话一问一答的截断检索单元、中文词项、pgvector 向量及来源消息 ID；只用于按需补充回答背景 |
| `conversation_history_index_jobs` | 历史单元异步向量化任务状态；嵌入模型不可用时仍保留全文召回 |
| `messages` | 用户与 Agent 消息、回答状态及数据引用 |
| `agent_runs` | 每次 Agent 问答的 `run_id`、脱敏摘要、实际采用的结构化上下文快照、会话条件变化、短期状态版本/采用标记、历史单元 ID/采用标记、路由、状态、总耗时和引用 ID；保留 15 天 |
| `tool_calls` | 关联 `run_id` 的工具调用摘要、引用 ID、结果、耗时和错误信息；随运行记录级联清理 |
| `products` | 商品资料；生产环境可接入商品主数据或业务系统同步 |
| `daily_metrics` | 按日期、商品和渠道汇总的经营指标，另可包含活动专项渠道数据 |
| `metric_definitions` | 指标名称、描述、别名、依赖、受控模板、公式和 RAG 向量缓存 |
| `knowledge_documents` | 原始文件元信息、资料空间、解析正文、处理状态和片段数 |
| `knowledge_chunks` | 文件片段、标题路径、内容类型、PDF 页码范围、pgvector 向量、中文分词词项与所用模型标识 |

### 4.1 本地账号认证与数据隔离

本期采用最小本地账号体系，而非完整权限系统。注册账号时，后端用随机盐和 Python 标准库 `hashlib.scrypt` 生成密码散列；数据库不保存明文密码。登录成功后生成 32 字节随机访问令牌，数据库仅保存其 SHA-256 摘要及过期时间，默认有效期为 7 天，可由 `AUTH_TOKEN_TTL_DAYS` 修改。退出登录会删除当前摘要记录，因此令牌可立即撤销。

```text
注册 / 登录
  -> 验证账号与密码
  -> 创建 users / auth_tokens 记录
  -> 返回一次原始 Bearer 令牌
  -> 浏览器保存令牌并在后续请求携带 Authorization 请求头

访问会话接口
  -> 校验令牌摘要和过期时间
  -> 获取 current_user
  -> 所有 conversations 查询附加 user_id = current_user.id
  -> 未归属给当前用户的会话统一返回 404
```

`conversations.user_id` 是会话、消息、会话上下文、Agent 运行审计以及后续长期记忆的根归属字段。知识库文件当前仍是共享工作区资料，不按用户拆分。迁移时已有会话会保留到不可登录的历史归属账户，避免把旧数据错误分配给第一个新注册用户；新账号默认从空会话开始。

这套方案适用于本地和单机 MVP。首个真实注册账号会成为初始化管理员；已存在的部署在迁移时提升最早真实账号。令牌存于浏览器本地存储，尚未包含 HTTPS 强制、登录频率限制、密码找回、邮箱验证、跨设备会话管理、细粒度角色权限或企业 SSO；部署到公网或处理真实敏感数据前，应升级为 HTTPS、HttpOnly Cookie/短期令牌机制并接入成熟身份提供方。

### 4.2 运行审计与可观测性

每次 `POST /api/conversations/{id}/messages` 在保存用户消息后立即创建 `agent_runs` 记录，并将其 `id` 作为本轮 `run_id`。路由完成后写入路由类型、置信度和是否保守降级；三个受控工具在运行时记录实际调用的状态、耗时、结果摘要与引用 ID。回答完成后再关联 Agent 消息、写入回答长度/引用数量摘要和总耗时。

在创建运行记录前，API 会先调用 `ContextBuilder`：读取 `conversation_contexts`，提取本轮用户明确出现的条件，按“当前轮覆盖旧值、未出现则继承、显式日期优先于已登记活动期”的规则生成条件快照，并写回会话。仅有新日期且未提活动时会清除旧活动标签，避免出现矛盾范围。快照随运行记录保存为条件摘要、变化动作和结构化 JSON；前端审计抽屉可直接查看。它不保存历史问题副本，字段来源仅保存消息 ID。

```text
显式“记住 / 忘记 / 清除所有记忆”命令
  -> 在用户域内直接新增、删除或清空记忆
  -> 保存一条 memory_command 消息并结束，不创建 AgentRun

保存用户消息
  -> ContextBuilder 合并条件 + 读取上一轮受限结论摘要
  -> 创建 agent_run（running，问题长度摘要、结构化上下文快照）
  -> 路由并记录 route
  -> 读取持久化 Action Plan；指标动作直接接收已确认的时间范围与指标清单
  -> LangGraph 按动作依赖执行本批步骤，记录每次 Attempt
  -> 证据不足时进入 `replan` 条件分支，由主 Agent 在数据库预算内追加下一批 Action 后回到执行节点
  -> 校验工具终态、每轮调用上限与成功结果引用 ID
  -> 工具调用写入内存轨迹（不保存原始问题/全文结果）
  -> 保存 Agent 回答 + tool_calls + 完成状态
  -> SSE 返回 message_id、run_id
  -> 前端按需读取 /messages/{message_id}/audit
```

工具状态统一为 `success`、`empty`、`skipped` 和 `failed`。例如未配置 Tavily 时，`search_web` 会留下 `status=skipped`、`error_code=web_search_disabled`，因此可直接定位“为什么没有联网搜索”。应用日志同时输出带 `run_id` 的 key-value 摘要，适合通过 `docker compose logs -f api` 检索。

审计表只保存问题/回答的结构化摘要、工具输入/输出摘要、引用 ID、结构化条件快照、会话短期状态版本/采用标记及会话历史单元 ID/采用标记：不保存原始问题、回答副本、文档正文、网页正文、向量或密钥。短期状态与历史单元均限制在同一会话范围，不作为跨会话长期记忆。API 启动后执行一次过期清理，并每 24 小时删除创建时间早于 15 天的 `agent_runs`；关联 `tool_calls` 由数据库外键级联删除。用户也可手动删除自己的已完成会话；接口在同一事务内按依赖逆序清理消息、短期条件、滚动摘要/任务、历史单元/任务、待确认记忆候选、运行审计和工具轨迹，已确认的 `user_memories` 不会随会话删除。

### 4.2 执行阶段 SSE 与前端展示

`OperationAgent.stream_events` 通过内存队列将工作流进度转换为 SSE `status` 事件，并在 `AnswerGenerator` 或复盘模型生成时立即转发 `chunk` 事件；不再等待完整回答后进行固定长度切片。前端按动画帧合并短分片以降低重渲染频率，生成期间以纯文本显示，`done` 后再使用 Notion 式 Markdown 渲染完整回答。进度只包含稳定的阶段文案，不携带用户原文、检索正文或密钥：

```text
status：正在应用本会话已确认的查询条件
  -> status：正在判断问题类型
  -> status：已生成执行计划
  -> status：正在查询经营指标 / 检索资料 / 联网搜索
  -> status：正在校验检索依据
  -> status：正在基于已验证依据生成结论
  -> chunk：回答文本
```

前端在临时 Agent 消息中展示当前阶段；回答完成后可打开“本次执行依据”，其中的 `execution_plan` 由已持久化路由恢复，并与实际 `tool_calls` 状态并列展示。

### 4.2.1 跨轮任务状态与执行计划

`ConversationTask` 保存任务的结构化有效约束，而不是仅依赖原问题和补充文本。`continue`、`revise` 在同一任务 ID 上递增版本并按字段合并约束：用户明确说“保留、沿用、刚才、上述范围”时优先继承已确认区间，即使本轮将对比升级为归因；只有明确替换时间的已验证 `resolved` 结果才能覆盖旧区间，`clarify` 仅形成待补充问题。`replace` 时，任务解释器还必须返回经原文子串校验的 `next_task_input`，新任务的指标、活动、目标和时间只能从该片段重新解析，避免“取消旧复盘，改为对比”把旧任务词汇带入新任务。`cancel` 会留下最小审计事件。

活动时间由 `activity_periods` 作为任务层与指标 RAG 的共享日历解析；例如“比较七夕和 618”会在任务创建时写入两个受控日期区间，不依赖后续检索层补救。每个任务版本都创建 `TaskPlan`、`TaskPlanStep` 与 `TaskPlanStepAttempt`。`TaskPlanStep` 是实际驱动执行的 Action，保存输入、依赖、能力要求、成功标准和所属计划版本；每次执行均写入 Attempt。数据库子 Agent 只有在任务的 `allow_metric_drilldown=true` 时才能提出已登记指标的下一步候选动作，主 Agent 再在预算内追加新的计划版本；纯指标对比只查询用户要求的口径。计划、尝试和任务事件均不复制完整对话正文。

### 4.3 固定 Agent 场景评估

`backend/tests/evaluation_cases.py` 定义数据驱动的离线评估集，`test_evaluation_suite.py` 使用真实 `AgentWorkflow`、`ExecutionPlan`、`AgentToolTracker` 和确定性测试替身执行每条样例。评估工具只写入与活动资料一致的最小指标、知识片段或网页摘要，不连接 PostgreSQL、向量模型、Tavily 或 LLM 服务。因此评估结果可重复，不会受数据变动、网络或模型随机性的影响。

首批样例包含 618、七夕、春季三类活动的指标查询、知识库问答与综合复盘，另外覆盖公开网页检索、内外部综合分析、知识库未命中和联网未配置。每条样例统一断言：

- 初始 Action Plan 的能力边界、依赖与动作顺序；
- 每个动作均有可审计的执行尝试，且工具调用不超过计划预算；
- 成功结果的 `metric:*`、`knowledge_chunk:*` 或 HTTP(S) 引用类型；
- 离线回答中的固定事实提示，或资料不足/联网不可用时的诚实降级文案。

运行命令如下：

```powershell
cd backend
python -m pytest tests/test_evaluation_suite.py -q
```

该评估验证受控编排的回归，不把固定替身误当作真实检索质量评测。指标召回、知识库 Top-K 和 Tavily 请求分别继续由其单元测试覆盖。

### 4.4 真实 RAG 评测与质量门禁

`backend/evaluation/datasets/v1/` 是版本化评测集：固定运营资料位于 `documents/`，指标、知识库、综合、多查询单元、多轮继承和无答案样例以 JSONL 保存。知识依据使用“源文件 + 标题/关键文本锚点”标注，而不是易因重切分变化的 Chunk ID。

`app.scripts.run_rag_evaluation` 会在隔离评测库中创建专用评测用户、写入固定资料并使用生产同一套解析、结构化切分、嵌入、pgvector HNSW、GIN + `ts_rank_cd`、RRF、CrossEncoder、受控 SQL 和 `AgentWorkflow` 执行样例。评测过程不连接 Tavily、不需要 LLM 改写；多 Query 改写在评测中固定关闭，以消除外部模型随机性。每一条样例保留向量候选、全文候选、RRF 候选和最终精排结果对应的来源文件，便于定位召回在哪一层退化。

报告输出以下指标：路由准确率、工具计划准确率、指标选择命中率、向量/全文/RRF 的 Recall@40、精排后的 Recall@4、MRR、nDCG@4、Precision@4、无答案准确率、回答关键事实覆盖率、禁止事实未出现率及端到端 P95 时延。Markdown 报告会在每个数值旁展示中文名称、英文术语、计算含义与“越高/越低越好”的判断方向；JSON 报告同步提供 `metric_definitions`，便于前端或脚本读取。`manifest.json` 定义最低门槛；人工验收某次结果后可使用 `--write-baseline` 写入 `baseline.json`，后续运行若任一已记录指标下降超过 3 个百分点即失败。

```powershell
# 需要已下载本地嵌入和精排模型；只启动隔离评测数据库和一次性评测容器。
docker compose --profile evaluation run --rm evaluation

# 人工核验首份报告后，明确将该次结果写为版本化基线。
docker compose --profile evaluation run --rm evaluation sh -c "alembic upgrade head && python -m app.scripts.run_rag_evaluation --prepare --write-baseline --output-dir /reports"
```

报告写入根目录 `evaluation-reports/`，该目录被 Git 忽略。每条样例还会记录路由置信度、是否因低置信度/低分差保护性降级到 `hybrid`、向量或意图精排的判定原因及各候选得分；因此可以区分“确实需要综合检索”和“保守地多查了一路”。`evaluation-db` 使用独立数据卷，评测脚本仅允许数据库连接串包含 `evaluation` 时执行 `--prepare`；若需要在其他隔离环境写入，必须显式设置 `EVALUATION_ALLOW_DATABASE_WRITE=true`。常规单元回归仍执行：

```powershell
cd backend
python -m pytest tests/test_evaluation_suite.py tests/test_evaluation_framework.py -q
```

## 5. 配置原则

- 运行配置由环境变量注入，示例值存放于 `.env.example`。
- 不将模型密钥、数据库密码或外部数据源凭据提交到仓库。
- `DATABASE_URL` 默认指向 Docker Compose 中的 PostgreSQL 服务。
- `LLM_API_KEY`、`LLM_MODEL` 和可选的 `LLM_BASE_URL` 均通过环境变量配置；不预设任何模型。
- `WEB_SEARCH_API_KEY` 仅用于 Tavily 联网搜索；为空时该工具关闭，不会隐式访问外部网络。公开网页摘要不自动沉淀为知识库。
- `AUTH_TOKEN_TTL_DAYS` 控制本地登录令牌有效期，默认 `7`；修改后新签发令牌按新期限生效。
- `USER_MEMORY_FOCUS_DIRECTION_TTL_DAYS` 控制“近期关注方向”未明确指定有效期时的默认保留天数，默认 `90`；到期记录不会进入后续回答生成提示词。
- `USER_MEMORY_MAX_ACTIVE_RECORDS` 控制单个用户可保留的有效长期记忆上限，默认 `50`；达到上限时，用户需先通过“忘记 …”或“清除所有记忆”释放容量。
- `CONVERSATION_SUMMARY_POLL_SECONDS`（默认 `1`）控制短期状态压缩 Worker 的空闲轮询间隔；`CONVERSATION_SUMMARY_MAX_ATTEMPTS`（默认 `3`）控制任务异常或 Worker 中断后的最大领取次数；`CONVERSATION_SUMMARY_LEASE_SECONDS`（默认 `300`）是 Worker 领取任务后的租约时长，超时任务会被自动回收。`CONVERSATION_MEMORY_TOKEN_BUDGET`（默认 `12000`）是会话短期状态总预算，`CONVERSATION_MEMORY_COMPACT_THRESHOLD`（默认 `9600`）是触发异步压缩的估算 token 阈值，`CONVERSATION_MEMORY_RECENT_MESSAGE_LIMIT`（默认 `10`）是压缩后最多保留的最近消息数；`CONVERSATION_MEMORY_TURN_TEXT_LIMIT`（默认 `2000`）限制写入短期状态的单条消息长度。`CONVERSATION_SUMMARY_LLM_TEXT_LIMIT`（默认 `1200`）和 `CONVERSATION_SUMMARY_FALLBACK_TEXT_LIMIT`（默认 `2400`）分别限制正常 LLM 摘要与模型不可用时的确定性降级摘要长度。压缩阈值必须小于总预算，且 LLM 摘要上限不得超过降级摘要上限；最近窗口还会受 token 预算约束，避免少量超长消息阻止状态收缩。它们只影响短期状态大小，不会阻塞问答请求。
- `KNOWLEDGE_UPLOAD_DIR` 默认 `/uploads`，由 Docker 映射为宿主机 `uploads/`；上传原件不写入镜像或数据库临时目录。
- 当前模型适配器采用 OpenAI 兼容协议，因此可配置支持该协议的模型服务地址和模型名称，而不绑定特定厂商；未配置密钥和模型时仅提供受控检索证据，不生成模拟结论。

## 6. Docker 部署

Docker Compose 包含六个常驻服务和一个按需工具服务：

- `frontend`：构建并提供 React 静态页面。
- `api`：运行 FastAPI 与 Agent 服务。
- `knowledge-worker`：异步解析、切分和向量化知识库资料。
- `conversation-summary-worker`：异步压缩同会话短期状态，并为新历史单元补齐本地向量。
- `adminer`：提供浏览器访问的 PostgreSQL 管理界面，仅用于本地查看和排查数据。
- `db`：运行 PostgreSQL，并使用命名数据卷保存数据。

前端镜像使用 `package-lock.json` 和 `npm ci` 安装依赖，以确保可复现构建。`frontend/.dockerignore` 会排除本地 `node_modules` 与 `dist`，避免将开发产物传入镜像构建上下文。

首次部署步骤：

1. 复制 `.env.example` 为 `.env`，按需填入模型配置；需要联网搜索时再填写 `WEB_SEARCH_API_KEY`。
2. 执行 `docker compose up -d --build`。
3. 访问 `http://localhost:5173`；API 健康检查地址为 `http://localhost:8000/health`；数据库管理界面为 `http://localhost:8081`。

Adminer 登录时选择 PostgreSQL，服务器填写 `db`，用户名和数据库名均为 `shop_agent`，密码使用 `.env` 中的 `POSTGRES_PASSWORD`。默认数据库端口不暴露到宿主机，Adminer 通过 Docker 内部网络连接数据库。

API 容器启动时会先执行数据库迁移，再运行 `python -m app.scripts.seed_demo_data`。初始化脚本以增量、幂等方式补齐数据，不覆盖已有记录：通用模拟数据包含 5 个商品、2026 年 1 月 1 日至 12 月 31 日每天的两个通用渠道记录，共 3,650 条；另根据根目录 `test/` 的活动文档补充春季上新活动期数据（GMV 86,400 元、支付订单 312 单）、618 预热/正式/返场期数据（按规则节奏构造的模拟数值）及七夕礼赠活动期的礼盒、组合款渠道数据。全年基础数据与活动专项数据共 3,730 条，可通过 `GET /api/metrics/overview` 查看最近两周汇总。

如需重新生成测试业务数据，可在人工确认后执行 `docker compose exec api python -m app.scripts.seed_demo_data --reset-business-data`。该命令会永久删除 `products` 和 `daily_metrics` 中的全部记录，再写入上述 2026 年模拟商品和指标；不会删除账号、会话、记忆、指标定义或知识库文件，不能用于保留真实业务数据的环境。知识库原始文件保存在 `uploads/` 挂载目录，重建 API 容器不会丢失；不要随意删除该目录。

本地语义向量与精排模型不随镜像或 Git 仓库提交。首次使用依次执行 `docker compose build api knowledge-worker model-download`、`docker compose run --rm model-download`、`docker compose up -d api knowledge-worker conversation-summary-worker` 和 `docker compose exec api python -m app.scripts.rebuild_embeddings`。下载工具会将嵌入模型保存到 `models/bge-small-zh-v1.5`、精排模型保存到 `models/bge-reranker-base`，随后均以只读卷挂载给 API 与 Worker；下载完成后重建指标定义，并从原始文件重新解析结构化知识库片段、向量和词面索引。会话 Worker 使用同一嵌入模型为新历史单元异步补齐向量；未配置或不可用时保留全文召回。它仅在已配置 `LLM_API_KEY` 与 `LLM_MODEL` 时使用同一兼容模型服务压缩摘要，否则自动采用确定性降级。需要暂时跳过精排模型时可执行 `docker compose run --rm model-download python -m app.scripts.download_local_models --skip-reranker`。API 运行期间不会下载模型或调用外部嵌入 API。

镜像为 CPU 推理预装 PyTorch CPU 轮子，不会安装 CUDA 运行库；如后续需要 GPU 推理，应单独提供 GPU 镜像与 `LOCAL_EMBEDDING_DEVICE` 配置，而不是在默认镜像中混入 CUDA 依赖。

停止服务使用 `docker compose down`。不要加 `-v`，否则会删除本地数据库卷。数据库备份、HTTPS 和生产域名在正式部署前再按目标环境补充。

## 7. 运行日志与排查

Agent 执行采用轻量级后端日志排障，不依赖 Prometheus、Grafana 或额外监控容器。前端无需新增诊断页面：正常回答的 SSE `done` 事件已包含 `message_id` 与 `run_id`；失败 SSE `error` 事件也会带 `run_id`。后端日志以 `key=value` 输出，并且不会记录用户原文、完整回答、SQL 或工具完整结果。

一次 Agent 运行会依次输出 `agent_run_started`、`plan_step_started` / `plan_step_finished`、`tool_call_finished` 与 `agent_run_completed` 或 `agent_run_failed`。所有执行链路均以 `run_id` 为中心；开始与结束日志还带有 `conversation_id`、`user_message_id` 和可用时的 `agent_message_id`。工具失败时，`tool_call_failed` 会附带白名单化的 `diagnostic_input`：指标工具保留指标编码、起止日期、查询分段日期和维度；知识库及联网工具仅保留问题长度、是否使用预处理查询或搜索提供方。

正常回答可先按前端的 `message_id` 找到运行，再按 `run_id` 查看完整过程：

```powershell
docker compose logs api | Select-String "agent_message_id=<message-id>"
docker compose logs api | Select-String "run_id=<run-id>"
```

没有生成 Agent 消息时，可按前端会话 ID 和发生时间定位：

```powershell
docker compose logs api | Select-String "conversation_id=<conversation-id>"
```

使用 `docker compose up -d` 会将服务放入后台，终端不会持续输出日志。使用以下命令查看：

```powershell
docker compose logs -f
docker compose logs -f api
docker compose logs -f frontend
docker compose logs -f db
docker compose logs -f conversation-summary-worker
```

- `api` 默认输出 Uvicorn 的启动、访问与异常日志。
- `frontend` 由 Nginx 提供静态页面，通常在浏览器发起请求后输出访问日志。
- `db` 输出 PostgreSQL 的启动与数据库错误日志。
- `conversation-summary-worker` 输出短期状态压缩与历史单元向量化任务的启动、完成和失败日志；它不处理前端请求。

健康检查与服务状态可通过 `docker compose ps` 和 `http://localhost:8000/health` 确认。

## 9. 用户长期记忆

长期记忆采用独立的 `user_memories` 表，以 `user_id` 为隔离边界，不复用 `messages`、`conversation_contexts` 或同会话 `conversation_summaries` 存储。它保存用户确认过的偏好，而不是业务事实：

- `analysis_preference`：分析习惯，例如复盘时优先按渠道、品类拆分；
- `answer_preference`：回答呈现偏好，例如结论先行、保持简洁；
- `focus_topic`：关注主题，例如大促活动复盘。
- `work_profile`：稳定工作背景，例如负责美妆品类和直播渠道。
- `focus_direction`：近期关注方向，例如近期重点关注大促支付转化；默认在 `USER_MEMORY_FOCUS_DIRECTION_TTL_DAYS` 后失效，也可由用户显式指定日期。

不包含指标口径偏好、指标数值、业务数据、完整聊天记录或知识库正文。前端不再提供独立的“我的记忆”管理抽屉；用户可在对话中输入“记住 …”“忘记 …”“清除所有记忆”。显式命令在 `MemoryService` 内直接处理，不调用 Agent、RAG、受控 SQL 或联网搜索。

系统另以 `user_memory_candidates` 保存自然表达产生的 `pending` 候选。普通问答完成并落库后，服务仅根据有限且可解释的偏好、工作背景和近期关注句式生成候选；指标和业务数据关键词会被拒绝。近期关注候选会携带有效期，到期后不再展示或可被确认。候选通过已认证的 `GET /api/memory-candidates` 获取，`POST /api/memory-candidates/{id}/accept` 才会转为 `user_memories`，`dismiss` 只标记忽略。所有读写均带 `user_id = current_user.id`；跨用户 ID 访问统一返回 404。

```text
保存用户消息
  -> ContextBuilder 更新当前会话短期条件
  -> MemoryService 读取当前用户 active 且未过期的记忆
  -> 轻量排序并最多选择 3 条
  -> 记忆使用次数与最后采用时间随本轮事务更新
  -> Agent 路由和受控工具仍只使用原问题 + 会话条件
  -> 仅在回答生成提示词中注入长期偏好
  -> agent_runs 保存采用数量摘要和记忆 ID
  -> 回答落库后从自然偏好表达生成 pending 候选（如有）
  -> SSE 推送候选，前端在对应回答下方展示“记住 / 暂不”
```

对于通常数量很小的用户记忆，首期不引入向量化或额外模型：回答偏好始终优先，分析习惯在“分析、复盘、对比、诊断、原因、建议、优化”等问题中优先，稳定工作背景与未过期的近期关注方向次之，旧版关注主题按文本命中和最近更新时间排序。候选提取同样采用有限规则，而非调用模型，避免为一次偏好确认额外增加时延、费用或误存风险。

`PromptBuilder` 明确约束长期偏好只能影响回答呈现、分析角度或建议优先级，不能覆盖本轮问题、会话活动/时间/指标条件，更不能作为业务事实、数据或指标口径。即使使用长期记忆，受控 SQL、指标 RAG 与知识库 RAG 的检索入参都不会改变。

`agent_runs.memory_summary` 只保存“采用了几条长期记忆”，`memory_ids` 保存本轮采用的记录 ID；记忆正文仍只保留在用户自己的 `user_memories.content`。运行审计的 15 天清理不会删除长期记忆；用户删除记忆后，历史运行记录仍可保留其 ID 以便复盘。`user_memory_candidates` 是待确认交互记录，不参与召回；“清除所有记忆”会同时忽略仍待确认的候选。

长期记忆治理采用“用户显式更新优先”的策略：`更新工作背景：…` 或 `更新近期关注：…` 会将同类型 `active` 记录标记为 `superseded`，新记录才会继续参与回答。每个用户有效记忆默认上限为 `50`，到期记录在读取或写入前惰性标记为 `expired`。手机号、邮箱、证件号、密码、密钥和令牌会被拒绝写入或生成候选。`user_memory_events` 保存创建、确认、替代、过期和删除事件，以及候选/会话/消息定位 ID；不保存记忆正文或原始对话。`agent_runs.memory_selection` 记录本轮采用的类型与可解释原因，用于观测召回质量。

## 8. 后续演进

- 接入真实业务数据源，并以工具适配器隔离不同平台的查询差异。
- 根据并发与异步任务需求，再评估 Redis 与任务队列。
- 按 PRD 的范围补充复盘、运营指标和人工确认的外部操作流程。
