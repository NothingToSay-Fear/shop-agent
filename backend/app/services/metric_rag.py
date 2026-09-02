"""通过指标定义 RAG 缩小经营数据查询范围，并执行受控 SQL 模板。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.models import DailyMetric, MetricDefinition
from app.services.local_embeddings import embed_texts
from app.services.date_ranges import parse_explicit_date_range

# 每个模板均为只读聚合查询，只允许固定的日期和数据源绑定参数。
SUM_PAID_GMV_SQL = """
SELECT COALESCE(SUM(paid_gmv), 0) AS value
FROM daily_metrics
WHERE source = :source AND metric_date BETWEEN :start_date AND :end_date
""".strip()
SUM_PAID_ORDER_SQL = """
SELECT COALESCE(SUM(paid_order_count), 0) AS value
FROM daily_metrics
WHERE source = :source AND metric_date BETWEEN :start_date AND :end_date
""".strip()
SUM_VISITOR_SQL = """
SELECT COALESCE(SUM(visitor_count), 0) AS value
FROM daily_metrics
WHERE source = :source AND metric_date BETWEEN :start_date AND :end_date
""".strip()
SUM_REFUND_ORDER_SQL = """
SELECT COALESCE(SUM(refund_order_count), 0) AS value
FROM daily_metrics
WHERE source = :source AND metric_date BETWEEN :start_date AND :end_date
""".strip()

CONTROLLED_SQL_TEMPLATES = frozenset(
    {SUM_PAID_GMV_SQL, SUM_PAID_ORDER_SQL, SUM_VISITOR_SQL, SUM_REFUND_ORDER_SQL}
)

METRIC_DEFINITION_SEEDS = (
    {
        "metric_code": "paid_gmv",
        "name": "支付 GMV",
        "description": "指定时间范围内已支付订单的成交金额总和，用于衡量销售规模。",
        "aliases": ["GMV", "成交额", "销售额", "营业额", "支付金额"],
        "dependency_codes": [],
        "query_template": SUM_PAID_GMV_SQL,
        "calculation_formula": None,
    },
    {
        "metric_code": "paid_order_count",
        "name": "支付订单数",
        "description": "指定时间范围内已支付的订单总数，用于衡量成交量。",
        "aliases": ["订单量", "订单数", "成交订单", "支付订单", "销量"],
        "dependency_codes": [],
        "query_template": SUM_PAID_ORDER_SQL,
        "calculation_formula": None,
    },
    {
        "metric_code": "visitor_count",
        "name": "访客数",
        "description": "指定时间范围内访问商品或运营页面的去重访客数量，用于衡量流量规模。",
        "aliases": ["访客", "流量", "UV", "浏览人数", "访问量"],
        "dependency_codes": [],
        "query_template": SUM_VISITOR_SQL,
        "calculation_formula": None,
    },
    {
        "metric_code": "refund_order_count",
        "name": "退款订单数",
        "description": "指定时间范围内产生退款的订单总数，用于衡量售后风险。",
        "aliases": ["退款数", "退款订单", "售后订单"],
        "dependency_codes": [],
        "query_template": SUM_REFUND_ORDER_SQL,
        "calculation_formula": None,
    },
    {
        "metric_code": "conversion_rate",
        "name": "支付转化率",
        "description": "支付订单数除以访客数，衡量流量转化为支付订单的效率。",
        "aliases": ["转化率", "支付转化", "成交转化", "CVR"],
        "dependency_codes": ["paid_order_count", "visitor_count"],
        "query_template": None,
        "calculation_formula": "paid_order_count / visitor_count",
    },
    {
        "metric_code": "average_order_value",
        "name": "客单价",
        "description": "支付 GMV 除以支付订单数，衡量每笔支付订单的平均成交金额。",
        "aliases": ["客单价", "平均订单金额", "AOV"],
        "dependency_codes": ["paid_gmv", "paid_order_count"],
        "query_template": None,
        "calculation_formula": "paid_gmv / paid_order_count",
    },
    {
        "metric_code": "refund_rate",
        "name": "退款率",
        "description": "退款订单数除以支付订单数，衡量支付订单产生退款的比例。",
        "aliases": ["退款率", "退货率", "售后率"],
        "dependency_codes": ["refund_order_count", "paid_order_count"],
        "query_template": None,
        "calculation_formula": "refund_order_count / paid_order_count",
    },
)


@dataclass(frozen=True)
class MetricDocument:
    """脱离 ORM 的指标检索文档，便于复用和单元测试。"""

    metric_code: str
    name: str
    description: str
    aliases: list[str]
    embedding: list[float] | None

    @property
    def retrieval_text(self) -> str:
        return " ".join([self.name, self.description, *self.aliases, self.metric_code])


@dataclass(frozen=True)
class RetrievedMetric:
    """一个被 RAG 命中的指标及其相似度。"""

    metric_code: str
    score: float


@dataclass(frozen=True)
class MetricQueryContext:
    """供 Agent 引用的最小化查询结果。"""

    text: str
    metric_codes: tuple[str, ...]


@dataclass(frozen=True)
class MetricQueryConstraints:
    """由上下文工程提供给指标 SQL 的结构化筛选条件。"""

    start_date: date | None = None
    end_date: date | None = None

    @property
    def period(self) -> tuple[date, date] | None:
        if self.start_date is None or self.end_date is None or self.start_date > self.end_date:
            return None
        return self.start_date, self.end_date


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """计算同维向量余弦相似度，长度不一致时视为不可比较。"""
    if len(left) != len(right) or not left or not right:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True))


def retrieve_metrics(question: str, documents: list[MetricDocument], query_embedding: list[float]) -> list[RetrievedMetric]:
    """结合向量相似度和指标别名精确命中，返回最相关的一个或两个指标。"""
    normalized_question = question.lower()
    candidates: list[RetrievedMetric] = []
    for document in documents:
        # 没有由当前本地模型生成的向量时，不以其他算法替代语义检索。
        if document.embedding is None:
            continue
        embedding = document.embedding
        score = cosine_similarity(query_embedding, embedding)
        # 指标别名是业务人员常用说法；命中时提高分数，避免短问题被向量噪声淹没。
        alias_hit = any(alias.lower() in normalized_question for alias in [document.name, *document.aliases])
        if alias_hit:
            score = max(score, 0.85)
        if score >= 0.2:
            candidates.append(RetrievedMetric(metric_code=document.metric_code, score=score))
    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates[:2]


async def seed_metric_definitions(session: AsyncSession) -> None:
    """幂等写入内置指标知识；真实业务可通过后台管理流程维护同一张表。"""
    existing_codes = set(await session.scalars(select(MetricDefinition.metric_code)))
    for definition in METRIC_DEFINITION_SEEDS:
        if definition["metric_code"] in existing_codes:
            continue
        session.add(
            MetricDefinition(
                **definition,
                # 向量必须由已挂载的本地模型统一生成，初始化阶段不写入替代向量。
                embedding=None,
                embedding_model=None,
                enabled=True,
            )
        )


async def query_metrics_for_question(
    session: AsyncSession,
    question: str,
    settings: Settings | None = None,
    query_embedding: list[float] | None = None,
    constraints: MetricQueryConstraints | None = None,
) -> MetricQueryContext | None:
    """检索用户需要的指标，补齐依赖后仅执行对应基础指标的受控 SQL。"""
    definitions = list(
        await session.scalars(select(MetricDefinition).where(MetricDefinition.enabled.is_(True)))
    )
    if not definitions:
        return None

    active_settings = settings or get_settings()
    active_query_embedding = query_embedding or await _query_embedding(question, active_settings)
    if active_query_embedding is None:
        return None
    embeddings_ready = await _ensure_definition_embeddings(session, definitions, active_settings)
    if not embeddings_ready:
        return None
    documents = [_to_document(definition) for definition in definitions]
    requested = retrieve_metrics(question, documents, active_query_embedding)
    if not requested:
        return None

    definition_map = {definition.metric_code: definition for definition in definitions}
    requested_codes = [item.metric_code for item in requested]
    resolved_codes = _resolve_dependencies(requested_codes, definition_map)
    period = (constraints.period if constraints is not None else None) or await _resolve_period(session, question)
    if period is None:
        return None
    start_date, end_date = period

    values: dict[str, float] = {}
    for metric_code in resolved_codes:
        definition = definition_map[metric_code]
        if definition.query_template:
            values[metric_code] = await _execute_controlled_template(
                session, definition.query_template, start_date, end_date
            )
    for metric_code in resolved_codes:
        definition = definition_map[metric_code]
        if definition.calculation_formula:
            values[metric_code] = _calculate_metric(definition.calculation_formula, values)

    requested_definitions = [definition_map[code] for code in requested_codes]
    lines = [f"数据来源：内置模拟经营数据；查询范围：{start_date} 至 {end_date}。"]
    for definition in requested_definitions:
        value = values.get(definition.metric_code, 0.0)
        lines.append(f"{definition.name}：{_format_value(definition.metric_code, value)}。")
    return MetricQueryContext(text="\n".join(lines), metric_codes=tuple(requested_codes))


async def _query_embedding(question: str, settings: Settings) -> list[float] | None:
    """仅使用手动挂载的本地模型生成问题向量。"""
    vectors = await embed_texts([question], settings)
    if vectors:
        return vectors[0]
    return None


async def _ensure_definition_embeddings(
    session: AsyncSession,
    definitions: list[MetricDefinition],
    settings: Settings,
) -> bool:
    """当本地嵌入模型切换时，为指标知识重建向量并缓存到定义表。"""
    stale_definitions = [
        item for item in definitions if item.embedding_model != settings.local_embedding_model_id
    ]
    if not stale_definitions:
        return True
    vectors = await embed_texts(
        [_to_document(item).retrieval_text for item in stale_definitions], settings
    )
    if vectors is not None:
        for definition, vector in zip(stale_definitions, vectors, strict=True):
            definition.embedding = vector
            definition.embedding_model = settings.local_embedding_model_id
        await session.flush()
        return True
    return False


def _to_document(definition: MetricDefinition) -> MetricDocument:
    return MetricDocument(
        metric_code=definition.metric_code,
        name=definition.name,
        description=definition.description,
        aliases=definition.aliases,
        embedding=definition.embedding,
    )


def _resolve_dependencies(
    requested_codes: list[str], definition_map: dict[str, MetricDefinition]
) -> list[str]:
    """递归补齐派生指标的前置指标，同时检测不存在的依赖和循环依赖。"""
    resolved: list[str] = []
    visiting: set[str] = set()

    def visit(metric_code: str) -> None:
        if metric_code in resolved:
            return
        if metric_code in visiting:
            raise ValueError(f"指标依赖存在循环：{metric_code}")
        definition = definition_map.get(metric_code)
        if definition is None:
            raise ValueError(f"指标依赖不存在：{metric_code}")
        visiting.add(metric_code)
        for dependency_code in definition.dependency_codes:
            visit(dependency_code)
        visiting.remove(metric_code)
        resolved.append(metric_code)

    for code in requested_codes:
        visit(code)
    return resolved


async def _resolve_period(session: AsyncSession, question: str) -> tuple[date, date] | None:
    """将当前支持的自然语言时间范围映射为受控日期参数。"""
    explicit_period = parse_explicit_date_range(question)
    if explicit_period is not None:
        return explicit_period
    latest_date = await session.scalar(
        select(func.max(DailyMetric.metric_date)).where(DailyMetric.source == "demo")
    )
    if latest_date is None:
        return None
    lowered = question.lower()
    if any(text_value in lowered for text_value in ("最近14天", "近14天", "两周")):
        return latest_date - timedelta(days=13), latest_date
    if "上周" in lowered:
        return latest_date - timedelta(days=13), latest_date - timedelta(days=7)
    return latest_date - timedelta(days=6), latest_date


def _parse_explicit_date_range(question: str) -> tuple[date, date] | None:
    """兼容既有调用方；实际解析由统一日期模块完成。"""
    return parse_explicit_date_range(question)


async def _execute_controlled_template(
    session: AsyncSession, query_template: str, start_date: date, end_date: date
) -> float:
    """仅执行登记过的只读 SQL 模板，所有输入均使用绑定参数传递。"""
    normalized_template = query_template.strip()
    if normalized_template not in CONTROLLED_SQL_TEMPLATES:
        raise ValueError("指标定义包含未登记的 SQL 模板")
    result = await session.scalar(
        text(normalized_template),
        {"source": "demo", "start_date": start_date, "end_date": end_date},
    )
    return float(result or Decimal("0"))


def _calculate_metric(formula: str, values: dict[str, float]) -> float:
    """按已登记的派生公式计算指标，禁止将表中字符串作为表达式直接执行。"""
    if formula == "paid_order_count / visitor_count":
        return values.get("paid_order_count", 0.0) / values.get("visitor_count", 1.0) if values.get("visitor_count") else 0.0
    if formula == "paid_gmv / paid_order_count":
        return values.get("paid_gmv", 0.0) / values.get("paid_order_count", 1.0) if values.get("paid_order_count") else 0.0
    if formula == "refund_order_count / paid_order_count":
        return values.get("refund_order_count", 0.0) / values.get("paid_order_count", 1.0) if values.get("paid_order_count") else 0.0
    raise ValueError(f"指标定义包含未登记的计算公式：{formula}")


def _format_value(metric_code: str, value: float) -> str:
    """以指标口径格式化数值，避免模型误解金额和比例单位。"""
    if metric_code in {"conversion_rate", "refund_rate"}:
        return f"{value:.2%}"
    if metric_code in {"paid_gmv", "average_order_value"}:
        return f"{value:,.2f} 元"
    return f"{value:,.0f}"
