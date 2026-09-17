"""通过指标定义 RAG 缩小经营数据查询范围，并执行受控 SQL 模板。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.models import DailyMetric, MetricDefinition
from app.services.business_dates import current_business_date
from app.services.local_embeddings import embed_texts
from app.services.activity_periods import ACTIVITY_PERIODS, resolve_activity_periods
from app.services.date_ranges import parse_explicit_date_range, parse_explicit_date_ranges
from app.services.hybrid_retrieval import hybrid_retrieve
from app.services.query_expansion import (
    PreparedRetrievalQueries,
    embed_expanded_queries,
    expand_queries,
)

MAX_QUERY_UNITS = 4

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
    query_units: tuple["MetricQueryUnit", ...] = ()


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


@dataclass(frozen=True)
class MetricQueryUnit:
    """一组固定时间范围内、共享同一批指标的受控查询单元。"""

    label: str
    start_date: date
    end_date: date


@dataclass(frozen=True)
class MetricQueryPlan:
    """一个问题对应的一至多个查询单元，不携带任何动态 SQL。"""

    units: tuple[MetricQueryUnit, ...]


class MetricQueryPlanError(ValueError):
    """用户表达的时间单元超出当前受控计划上限。"""


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """计算同维向量余弦相似度，长度不一致时视为不可比较。"""
    if len(left) != len(right) or not left or not right:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True))


def retrieve_metrics(
    question: str, documents: list[MetricDocument], query_embedding: list[float]
) -> list[RetrievedMetric]:
    """兼容既有单 Query 调用；实际召回也统一经过向量、BM25 与 RRF。"""
    return retrieve_metrics_for_queries((question,), documents, [query_embedding])


def retrieve_metrics_for_queries(
    queries: tuple[str, ...],
    documents: list[MetricDocument],
    query_embeddings: list[list[float] | None],
) -> list[RetrievedMetric]:
    """多 Query 混合召回指标定义，保留别名这一强业务信号。"""
    document_by_code = {document.metric_code: document for document in documents}

    def dense_score(document: MetricDocument, query: str, vector: list[float]) -> float:
        score = cosine_similarity(vector, document.embedding or [])
        alias_hit = any(
            alias.lower() in query.lower() for alias in [document.name, *document.aliases]
        )
        return max(score, 0.85) if alias_hit else score

    fused = hybrid_retrieve(
        documents,
        queries,
        query_embeddings,
        get_id=lambda document: document.metric_code,
        get_text=lambda document: document.retrieval_text,
        get_embedding=lambda document: document.embedding,
        dense_score=dense_score,
    )
    return [
        RetrievedMetric(metric_code=candidate.item_id, score=candidate.rrf_score)
        for candidate in fused
        if candidate.item_id in document_by_code
    ][:2]


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
    await session.flush()


async def query_metrics_for_question(
    session: AsyncSession,
    question: str,
    settings: Settings | None = None,
    query_embedding: list[float] | None = None,
    constraints: MetricQueryConstraints | None = None,
    prepared_queries: PreparedRetrievalQueries | None = None,
) -> MetricQueryContext | None:
    """检索用户需要的指标，补齐依赖后仅执行对应基础指标的受控 SQL。"""
    # 先校验时间单元数量，避免超限问题继续消耗向量检索和数据库查询资源。
    plan = build_metric_query_plan(question, constraints)
    definitions = list(
        await session.scalars(select(MetricDefinition).where(MetricDefinition.enabled.is_(True)))
    )
    if not definitions:
        return None

    active_settings = settings or get_settings()
    use_prepared_queries = prepared_queries is not None and prepared_queries.question == question.strip()
    active_query_embedding = (
        prepared_queries.original_embedding
        if use_prepared_queries and prepared_queries is not None
        else query_embedding or await _query_embedding(question, active_settings)
    )
    # 只有向量检索可用时才尝试补齐指标向量；BM25 仍可在模型暂不可用时查询已启用定义。
    if active_query_embedding is not None:
        await _ensure_definition_embeddings(session, definitions, active_settings)
    documents = [_to_document(definition) for definition in definitions]
    if use_prepared_queries and prepared_queries is not None:
        queries = prepared_queries.queries
        query_embeddings = list(prepared_queries.query_embeddings)
    else:
        queries = await expand_queries(question, active_settings)
        if not queries:
            return None
        query_embeddings = await embed_expanded_queries(
            queries, active_settings, active_query_embedding
        )
    requested = retrieve_metrics_for_queries(queries, documents, query_embeddings)
    if not requested:
        return None

    definition_map = {definition.metric_code: definition for definition in definitions}
    requested_codes = [item.metric_code for item in requested]
    resolved_codes = _resolve_dependencies(requested_codes, definition_map)
    if not plan.units:
        fallback_period = await _resolve_period(session, question)
        if fallback_period is None:
            return None
        plan = MetricQueryPlan((MetricQueryUnit("当前查询范围", *fallback_period),))
    if not plan.units:
        return None

    values_by_unit: list[dict[str, float]] = []
    for unit in plan.units:
        values: dict[str, float] = {}
        for metric_code in resolved_codes:
            definition = definition_map[metric_code]
            if definition.query_template:
                values[metric_code] = await _execute_controlled_template(
                    session, definition.query_template, unit.start_date, unit.end_date
                )
        for metric_code in resolved_codes:
            definition = definition_map[metric_code]
            if definition.calculation_formula:
                values[metric_code] = _calculate_metric(definition.calculation_formula, values)
        values_by_unit.append(values)

    requested_definitions = [definition_map[code] for code in requested_codes]
    lines = ["数据来源：经营数据。"]
    for unit, values in zip(plan.units, values_by_unit, strict=True):
        lines.append(f"【{unit.label}：{unit.start_date} 至 {unit.end_date}】")
        for definition in requested_definitions:
            value = values.get(definition.metric_code, 0.0)
            lines.append(f"{definition.name}：{_format_value(definition.metric_code, value)}。")
    if len(plan.units) > 1:
        lines.append("【区间对比】")
        baseline_unit, baseline_values = plan.units[0], values_by_unit[0]
        baseline_days = _unit_days(baseline_unit)
        for unit, values in zip(plan.units[1:], values_by_unit[1:], strict=True):
            compared_days = _unit_days(unit)
            unequal_days = baseline_days != compared_days
            if unequal_days:
                lines.append(
                    f"【天数不等提示】{baseline_unit.label}共 {baseline_days} 天，"
                    f"{unit.label}共 {compared_days} 天；总量仅用于规模参考，"
                    "可比趋势请以日均值为准。"
                )
            for definition in requested_definitions:
                lines.append(
                    _format_unit_comparison(
                        definition,
                        baseline_unit,
                        baseline_values.get(definition.metric_code, 0.0),
                        unit,
                        values.get(definition.metric_code, 0.0),
                        unequal_days,
                    )
                )
                if unequal_days and definition.metric_code not in {
                    "conversion_rate",
                    "refund_rate",
                    "average_order_value",
                }:
                    lines.append(
                        _format_daily_average_comparison(
                            definition,
                            baseline_unit,
                            baseline_values.get(definition.metric_code, 0.0),
                            baseline_days,
                            unit,
                            values.get(definition.metric_code, 0.0),
                            compared_days,
                        )
                    )
    return MetricQueryContext(
        text="\n".join(lines),
        metric_codes=tuple(requested_codes),
        query_units=plan.units,
    )


async def query_metrics_for_codes(
    session: AsyncSession,
    metric_codes: tuple[str, ...],
    question: str,
    settings: Settings | None = None,
    constraints: MetricQueryConstraints | None = None,
    query_plan: MetricQueryPlan | None = None,
    capability_notes: tuple[str, ...] = (),
) -> MetricQueryContext | None:
    """按数据查询计划指定的指标执行，而不是再次对指标定义做 RAG 截断。"""
    requested_codes = list(dict.fromkeys(metric_codes))
    if not requested_codes:
        return None

    definitions = list(
        await session.scalars(select(MetricDefinition).where(MetricDefinition.enabled.is_(True)))
    )
    definition_map = {definition.metric_code: definition for definition in definitions}
    available_codes = [code for code in requested_codes if code in definition_map]
    if not available_codes:
        return None

    plan = query_plan or build_metric_query_plan(question, constraints)
    if not plan.units:
        fallback_period = await _resolve_period(session, question)
        if fallback_period is None:
            return None
        plan = MetricQueryPlan((MetricQueryUnit("当前查询范围", *fallback_period),))

    resolved_codes = _resolve_dependencies(available_codes, definition_map)
    values_by_unit: list[dict[str, float]] = []
    for unit in plan.units:
        values: dict[str, float] = {}
        for metric_code in resolved_codes:
            definition = definition_map[metric_code]
            if definition.query_template:
                values[metric_code] = await _execute_controlled_template(
                    session, definition.query_template, unit.start_date, unit.end_date
                )
        for metric_code in resolved_codes:
            definition = definition_map[metric_code]
            if definition.calculation_formula:
                values[metric_code] = _calculate_metric(definition.calculation_formula, values)
        values_by_unit.append(values)

    lines = ["数据来源：经营数据（按任务数据查询计划执行）"]
    for note in capability_notes:
        lines.append(f"【数据能力边界】{note}")
    for unit, values in zip(plan.units, values_by_unit, strict=True):
        lines.append(f"【{unit.label}：{unit.start_date} 至 {unit.end_date}】")
        for metric_code in available_codes:
            definition = definition_map[metric_code]
            lines.append(f"{definition.name}：{_format_value(metric_code, values[metric_code])}")
    if len(plan.units) > 1:
        lines.append("【区间对比】")
        baseline_unit, baseline_values = plan.units[0], values_by_unit[0]
        baseline_days = _unit_days(baseline_unit)
        for unit, values in zip(plan.units[1:], values_by_unit[1:], strict=True):
            compared_days = _unit_days(unit)
            unequal_days = baseline_days != compared_days
            if unequal_days:
                lines.append(
                    f"【天数不等提示】{baseline_unit.label} 共 {baseline_days} 天，"
                    f"{unit.label} 共 {compared_days} 天；总量仅用于规模参考，可比趋势请以日均值为准。"
                )
            for metric_code in available_codes:
                definition = definition_map[metric_code]
                lines.append(
                    _format_unit_comparison(
                        definition,
                        baseline_unit,
                        baseline_values[metric_code],
                        unit,
                        values[metric_code],
                        unequal_days,
                    )
                )
                if unequal_days and metric_code not in {
                    "conversion_rate",
                    "refund_rate",
                    "average_order_value",
                }:
                    lines.append(
                        _format_daily_average_comparison(
                            definition,
                            baseline_unit,
                            baseline_values[metric_code],
                            baseline_days,
                            unit,
                            values[metric_code],
                            compared_days,
                        )
                    )
    return MetricQueryContext(
        text="\n".join(lines), metric_codes=tuple(available_codes), query_units=plan.units
    )


def build_metric_query_plan(
    question: str, constraints: MetricQueryConstraints | None = None
) -> MetricQueryPlan:
    """从明确活动或日期范围构建多查询单元，优先保留用户本轮给出的多区间表达。"""
    explicit_units = _explicit_period_units(question)
    activity_units = _activity_period_units(question)
    explicit_or_activity_units = _sort_query_units_by_question_order(
        question, [*explicit_units, *activity_units]
    )
    if len(explicit_or_activity_units) > 1:
        return _make_query_plan(explicit_or_activity_units)
    if constraints is not None and constraints.period is not None:
        return _make_query_plan((MetricQueryUnit("已确认时间范围", *constraints.period),))
    if explicit_units:
        return _make_query_plan(explicit_units)
    if activity_units:
        return _make_query_plan(activity_units)
    return MetricQueryPlan(())


def _explicit_period_units(question: str) -> tuple[MetricQueryUnit, ...]:
    """将每段明确日期表达变成一个查询单元，并按原文出现顺序去重。"""
    units = [MetricQueryUnit(label, *period) for label, period in parse_explicit_date_ranges(question)]
    return _deduplicate_query_units(units)


def _activity_period_units(question: str) -> tuple[MetricQueryUnit, ...]:
    """识别问题中全部已登记活动，避免会话单一活动条件覆盖本轮对比对象。"""
    return _deduplicate_query_units(
        [MetricQueryUnit(activity, start_date, end_date) for activity, start_date, end_date in resolve_activity_periods(question)]
    )


def _deduplicate_query_units(units: list[MetricQueryUnit]) -> tuple[MetricQueryUnit, ...]:
    seen_periods: set[tuple[date, date]] = set()
    unique: list[MetricQueryUnit] = []
    for unit in units:
        period = (unit.start_date, unit.end_date)
        if period not in seen_periods:
            seen_periods.add(period)
            unique.append(unit)
    return tuple(unique)


def _sort_query_units_by_question_order(
    question: str, units: list[MetricQueryUnit] | tuple[MetricQueryUnit, ...]
) -> tuple[MetricQueryUnit, ...]:
    """合并活动期和显式日期时，尽量按用户表达顺序确定对比基准。"""
    lowered = question.lower()

    def position(unit: MetricQueryUnit) -> int:
        if unit.label in ACTIVITY_PERIODS:
            aliases = ACTIVITY_PERIODS[unit.label][0]
            matched_positions = [lowered.find(alias.lower()) for alias in aliases]
            visible_positions = [item for item in matched_positions if item >= 0]
            if visible_positions:
                return min(visible_positions)
        visible_position = lowered.find(unit.label.lower())
        return visible_position if visible_position >= 0 else len(question)

    return _deduplicate_query_units(sorted(units, key=position))


def _make_query_plan(units: tuple[MetricQueryUnit, ...]) -> MetricQueryPlan:
    if len(units) > MAX_QUERY_UNITS:
        raise MetricQueryPlanError(f"一次最多支持 {MAX_QUERY_UNITS} 个明确时间范围，请拆分后再查询。")
    return MetricQueryPlan(units)


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
    """将自然语言时间范围映射为受控日期参数，默认范围不使用未来数据。"""
    explicit_period = parse_explicit_date_range(question)
    if explicit_period is not None:
        return explicit_period
    latest_date = await session.scalar(
        select(func.max(DailyMetric.metric_date)).where(
            DailyMetric.source == "demo",
            DailyMetric.metric_date <= current_business_date(),
        )
    )
    if latest_date is None:
        return None
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


def _format_unit_comparison(
    definition: MetricDefinition,
    baseline_unit: MetricQueryUnit,
    baseline_value: float,
    compared_unit: MetricQueryUnit,
    compared_value: float,
    unequal_days: bool = False,
) -> str:
    """输出透明的确定性区间差异，归因和建议仍交由基于依据的回答层完成。"""
    change = compared_value - baseline_value
    if definition.metric_code in {"conversion_rate", "refund_rate"}:
        return (
            f"{compared_unit.label} 相比 {baseline_unit.label}，{definition.name}"
            f"变化 {change * 100:+.2f} 个百分点。"
        )
    if baseline_value:
        ratio = change / baseline_value
        ratio_text = f"（{ratio:+.2%}）"
    else:
        ratio_text = "（基准值为 0，无法计算比例变化）"
    suffix = "（统计天数不同，总量不作为可比涨跌结论）" if unequal_days else ""
    return (
        f"{compared_unit.label} 相比 {baseline_unit.label}，{definition.name} "
        f"变化 {_format_value(definition.metric_code, change)}{ratio_text}。{suffix}"
    )


def _format_daily_average_comparison(
    definition: MetricDefinition,
    baseline_unit: MetricQueryUnit,
    baseline_value: float,
    baseline_days: int,
    compared_unit: MetricQueryUnit,
    compared_value: float,
    compared_days: int,
) -> str:
    """当区间天数不等时，补充日均值变化，避免模型把整月和月累计的总量混为同口径。"""
    baseline_average = baseline_value / baseline_days
    compared_average = compared_value / compared_days
    change = compared_average - baseline_average
    if baseline_average:
        ratio_text = f"（{change / baseline_average:+.2%}）"
    else:
        ratio_text = "（基准日均值为 0，无法计算比例变化）"
    return (
        f"日均{definition.name}：{compared_unit.label} {_format_value(definition.metric_code, compared_average)}，"
        f"{baseline_unit.label} {_format_value(definition.metric_code, baseline_average)}，"
        f"变化 {_format_value(definition.metric_code, change)}{ratio_text}。"
    )


def _unit_days(unit: MetricQueryUnit) -> int:
    """查询单元均为闭区间，天数用于判断总量是否可以直接比较。"""
    return (unit.end_date - unit.start_date).days + 1


def _format_value(metric_code: str, value: float) -> str:
    """以指标口径格式化数值，避免模型误解金额和比例单位。"""
    if metric_code in {"conversion_rate", "refund_rate"}:
        return f"{value:.2%}"
    if metric_code in {"paid_gmv", "average_order_value"}:
        return f"{value:,.2f} 元"
    return f"{value:,.0f}"
