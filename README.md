# Shop Agent

面向电商运营人员的智能工作助手。用户可通过自然语言查询经营指标、分析经营变化、检索运营资料，并生成活动方案、商品文案和复盘建议。

> 项目当前提供本地/单机 Docker Compose 部署方案，并默认写入模拟经营数据，适合开发、演示与功能验证。接入真实业务数据或部署到公网前，请阅读[生产部署注意事项](#生产部署注意事项)。

## 核心能力

- **经营分析**：查询 GMV、订单量、支付转化率、客单价、退款率等指标，支持按时间、渠道和商品/SKU 下钻，并进行环比、同比与归因分析。
- **受控 Agent 工作流**：先进行意图路由和执行计划校验，再调用指标、知识库或联网搜索工具；最终回答仅基于已验证证据生成。
- **知识库问答**：支持 PDF、DOCX、Markdown、TXT 资料上传、异步解析、预览和引用式问答；支持团队与私有资料空间。
- **混合检索**：结合向量检索、BM25 全文检索、RRF 融合与本地 Cross-Encoder 精排。
- **多轮会话与记忆**：支持会话任务、短期上下文压缩、历史讨论召回和用户确认的长期偏好。
- **运营工作台**：提供注册登录、会话管理、SSE 流式回答、资料管理、运行审计和知识库索引进度展示。

## 技术栈

- 后端：Python 3.12、FastAPI、SQLAlchemy、Alembic、PostgreSQL、pgvector
- Agent：LangChain、LangGraph、受控工具调用、Plan-and-Execute
- 检索：Sentence Transformers、BAAI Embedding / Reranker、BM25、jieba、RRF
- 前端：React、TypeScript、Vite、Ant Design
- 交付与验证：Docker Compose、Pytest、隔离 RAG 离线评测

## 快速启动

### 1. 前置条件

- Docker Engine 及 Docker Compose v2
- 可访问 Docker 镜像仓库；首次下载本地检索模型时还需访问 Hugging Face
- 可选：兼容 OpenAI API 的模型服务，或 Anthropic、Google、Ollama 服务

确认 Docker 可用：

```powershell
docker compose version
```

### 2. 克隆并进入项目

```powershell
git clone <你的仓库地址> shop-agent
Set-Location shop-agent
```

### 3. 创建本地配置

```powershell
Copy-Item .env.example .env
```

使用编辑器打开 `.env`，至少将 `POSTGRES_PASSWORD` 改成自己的本地密码。

若要启用真实模型回答，填写以下配置：

```dotenv
LLM_API_KEY=<你的模型服务密钥>
LLM_MODEL=<模型名称>
# 使用 OpenAI 兼容服务时填写；使用官方默认地址时留空。
LLM_BASE_URL=
```

若未配置 `LLM_API_KEY` 与 `LLM_MODEL`，系统仍可启动、检索和审计，但不会生成未经验证的模拟结论；有可用证据时仅返回证据依据。

### 4. 下载本地检索模型（首次必须执行）

项目不会将模型文件提交到 Git，也不会在 API 运行时下载模型。首次启动请先下载本地 Embedding 与精排模型：

```powershell
docker compose build api knowledge-worker model-download
docker compose run --rm model-download
```

模型将保存到本地 `models/`，之后由 API 和 Worker 以只读方式挂载。

如仅想暂时体验检索流程、跳过精排模型，可执行：

```powershell
docker compose run --rm model-download python -m app.scripts.download_local_models --skip-reranker
```

### 5. 启动全部服务

```powershell
docker compose up -d --build
```

首次模型下载完成后，重建指标定义和知识库向量索引：

```powershell
docker compose exec api python -m app.scripts.rebuild_embeddings
```

查看服务状态：

```powershell
docker compose ps
```

### 6. 验证运行结果

- 前端工作台：<http://localhost:5173>
- API 健康检查：<http://localhost:8000/health>
- 指标概览：<http://localhost:8000/api/metrics/overview>
- 本地 Adminer：<http://localhost:8081>

首次打开前端后，注册一个本地账号即可开始使用。可以尝试：

```text
分析本周 GMV 环比下降原因
```

或上传根目录 `test/` 中的示例运营资料后提问：

```text
根据 618 活动规则，说明预热期的核心玩法。
```

## 配置说明

完整配置项见 [`.env.example`](.env.example)。最常用的变量如下：

| 变量 | 是否必填 | 说明 |
| --- | --- | --- |
| `POSTGRES_PASSWORD` | 是 | PostgreSQL 密码；不要保留默认值。 |
| `LLM_PROVIDER` | 否 | `openai`、`anthropic`、`google` 或 `ollama`；默认为 `openai`。当前 Compose 未将该变量传给容器，使用非 `openai` 提供方前需在 Compose 的 `api` 与摘要 Worker 环境变量中补充映射。 |
| `LLM_API_KEY` | 使用云端模型时是 | 模型服务密钥。Ollama 可不填。 |
| `LLM_MODEL` | 启用模型生成时是 | 服务端支持的模型名称。 |
| `LLM_BASE_URL` | 否 | OpenAI 兼容服务或 Ollama 的 API 地址。 |
| `WEB_SEARCH_API_KEY` | 启用联网搜索时是 | Tavily API 密钥；留空时联网搜索自动关闭。 |
| `LOCAL_EMBEDDING_DEVICE` | 否 | 默认 `cpu`；GPU 部署需使用独立镜像和对应运行环境。 |
| `CORS_ORIGINS` | 否 | 本地默认前端地址；当前 Compose 固定为 localhost，公网部署时需通过生产 Compose 覆盖为真实 HTTPS 域名。 |

## 服务说明

| 服务 | 职责 |
| --- | --- |
| `frontend` | 构建 React 工作台，并通过 Nginx 提供静态页面。 |
| `api` | FastAPI 服务；负责认证、会话、Agent 编排、SSE 输出、数据库迁移和本地演示数据初始化。 |
| `knowledge-worker` | 异步解析、切分并向量化知识库资料。 |
| `conversation-summary-worker` | 异步压缩会话短期状态，并为历史单元补齐向量。 |
| `db` | PostgreSQL + pgvector 数据库。 |
| `model-download` | 一次性模型下载工具，仅在首次初始化时运行。 |
| `adminer` | 本地数据库查看工具；仅用于开发排查，不应暴露到公网。 |

## 常用命令

### 查看日志

```powershell
docker compose logs -f
docker compose logs -f api
docker compose logs -f knowledge-worker
docker compose logs -f conversation-summary-worker
```

### 停止服务

```powershell
docker compose down
```

不要添加 `-v`，否则会删除 PostgreSQL 数据卷。

### 运行后端测试

```powershell
docker compose run --rm -v "${PWD}/backend/tests:/app/tests:ro" api pytest
```

### 运行隔离 RAG 评测

评测使用独立数据库和固定语料，不会写入业务数据库：

```powershell
docker compose --profile evaluation run --rm evaluation
```

报告会输出到根目录 `evaluation-reports/`。

## 常见问题

### 访问前端后无法连接 API

确认容器均已启动，并检查健康检查：

```powershell
docker compose ps
Invoke-WebRequest http://localhost:8000/health
```

随后查看 API 日志：

```powershell
docker compose logs --tail 200 api
```

### 知识库一直处于“排队”或“解析中”

确认 `knowledge-worker` 正常运行，且 `models/bge-small-zh-v1.5` 已通过“下载本地检索模型”步骤写入：

```powershell
docker compose ps knowledge-worker
docker compose logs --tail 200 knowledge-worker
```

### 回答中只有依据，没有自然语言结论

这通常表示未配置可用的 `LLM_API_KEY` 与 `LLM_MODEL`，或模型服务调用失败。系统在这种情况下会保留已验证证据并停止生成，避免编造结论。

### 如何定位一次 Agent 执行

正常 SSE 完成事件会返回 `message_id` 和 `run_id`。可通过 `run_id` 查询 API 日志：

```powershell
docker compose logs api | Select-String "run_id=<run-id>"
```

## 生产部署注意事项

当前 Compose 面向本地和单机 MVP。接入真实用户或真实经营数据前，至少应完成：

- 移除生产启动过程中的模拟数据初始化，并接入真实业务数据源；
- 使用 HTTPS，前端 API 地址与后端 `CORS_ORIGINS` 配置为真实域名；
- 使用密钥管理服务或受限环境文件管理数据库、LLM 和联网搜索密钥；
- 不向公网暴露 PostgreSQL、Adminer 和 API 裸端口；
- 升级当前本地令牌认证为企业 SSO 或安全的 HttpOnly Cookie/短期令牌方案，并增加登录限流；
- 为 PostgreSQL 和 `uploads/` 配置定期备份及恢复演练；
- 在灰度环境运行测试与 RAG 评测后再发布。

详细设计和部署说明见 [技术方案](docs/TECH.md)，业务范围见 [PRD](docs/PRD.md)。

## 目录结构

```text
backend/
  app/
    api/             # FastAPI 路由
    agent/           # 主 Agent、子 Agent、工具与 Plan-and-Execute 编排
    services/        # 指标、会话、知识库、检索、模型与认证服务
    workers/         # 知识库索引与会话摘要后台任务
    scripts/         # 数据初始化、模型下载、向量重建与离线评测
  migrations/        # Alembic 数据库迁移
  tests/             # 后端测试
frontend/            # React 运营工作台
docs/                # PRD 与技术方案
models/              # 本地 Embedding / Reranker 模型（不提交）
uploads/             # 知识库原始文件（持久化目录）
test/                # 演示运营资料
```
