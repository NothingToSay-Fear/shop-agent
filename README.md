# Shop Agent

面向电商运营人员的智能工作助手。项目说明见 [PRD](docs/PRD.md)，技术方案见 [TECH](docs/TECH.md)。

## 项目结构

```text
backend/
  app/
    api/           # FastAPI 路由
    agent/         # 主 Agent、子 Agent、工具与 Plan-Execute 编排
    services/
      analytics/   # 指标、日期、活动周期与数据查询规划
      conversations/ # 会话上下文、任务、摘要与计划持久化
      knowledge/   # 知识库解析、索引与检索
      memory/      # 用户长期记忆
      retrieval/   # 混合召回与查询扩展
      models/      # LLM、嵌入与精排模型适配
    workers/       # 后台索引与会话摘要任务
    scripts/       # 初始化、模型下载、向量重建与离线评测
  tests/           # 后端测试
  migrations/      # Alembic 数据库迁移
frontend/          # React 运营工作台
docs/              # PRD 与技术方案
```

## Docker 启动

```powershell
Copy-Item .env.example .env
docker compose up -d --build
```

前端默认地址：`http://localhost:5173`；后端健康检查：`http://localhost:8000/health`。

未填写 `LLM_API_KEY` 和 `LLM_MODEL` 时，系统以演示模式运行，仍可体验完整会话、任务与反馈流程。模型服务使用开放配置，不预设 OpenAI 模型。

首次启动时，API 会自动写入 3 个模拟商品和最近 14 天的经营数据。访问 `http://localhost:8000/api/metrics/overview` 可查看模拟数据汇总；在对话中输入“分析本周 GMV 环比下降原因”可体验基于模拟数据的回答。
