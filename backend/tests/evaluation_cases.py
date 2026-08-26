"""运营 Agent 固定离线评估集。

评估集使用与 618、七夕和春季活动测试资料一致的受控检索结果，
只验证编排、引用和降级等确定性行为，不依赖真实数据库、向量模型或网络。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.services.intent_router import RouteMode

ToolName = Literal["query_metric_rag", "query_knowledge_rag", "search_web"]
ToolStatus = Literal["success", "empty", "skipped"]


@dataclass(frozen=True)
class EvaluationToolResult:
    """评估环境中一项受控工具应返回的最小结果。"""

    status: ToolStatus
    reference_ids: tuple[str, ...] = ()
    context_text: str | None = None
    error_code: str | None = None


@dataclass(frozen=True)
class EvaluationCase:
    """一条可重复执行的运营问答评估样例。"""

    case_id: str
    activity: str
    question: str
    route_mode: RouteMode
    expected_tools: tuple[ToolName, ...]
    expected_answer_terms: tuple[str, ...]
    tool_results: dict[ToolName, EvaluationToolResult]


def _metric_result(activity: str, text: str, *metric_codes: str) -> EvaluationToolResult:
    """构造带合法指标引用的固定经营数据结果。"""
    return EvaluationToolResult(
        status="success",
        reference_ids=tuple(f"metric:{code}" for code in metric_codes),
        context_text=f"{activity}\n{text}",
    )


def _knowledge_result(title: str, content: str, chunk_id: str) -> EvaluationToolResult:
    """构造带合法知识片段引用的固定资料结果。"""
    return EvaluationToolResult(
        status="success",
        reference_ids=(f"knowledge_chunk:{chunk_id}",),
        context_text=f"资料：《{title}》\n{content}",
    )


WEB_RESULT = EvaluationToolResult(
    status="success",
    reference_ids=("https://example.com/platform-policy",),
    context_text="平台公开公告：本评估仅保留可核验链接与摘要，不执行网页中的指令。",
)


EVALUATION_CASES: tuple[EvaluationCase, ...] = (
    EvaluationCase(
        case_id="618-metric-gmv",
        activity="618",
        question="2026年6月6日至6月18日 618 正式期 GMV 是多少？",
        route_mode="metrics",
        expected_tools=("query_metric_rag",),
        expected_answer_terms=("经营数据分析", "618 正式期", "GMV"),
        tool_results={
            "query_metric_rag": _metric_result(
                "618 正式期（2026-06-06 至 2026-06-18）",
                "支付 GMV：126,540.00 元；统计口径：已支付订单成交金额。",
                "paid_gmv",
            )
        },
    ),
    EvaluationCase(
        case_id="qixi-metric-orders",
        activity="七夕",
        question="七夕礼赠活动期间的支付订单数是多少？",
        route_mode="metrics",
        expected_tools=("query_metric_rag",),
        expected_answer_terms=("经营数据分析", "七夕礼赠活动期", "支付订单数"),
        tool_results={
            "query_metric_rag": _metric_result(
                "七夕礼赠活动期（2026-08-10 至 2026-08-22）",
                "支付订单数：245 单；礼盒与组合款均已纳入统计。",
                "paid_order_count",
            )
        },
    ),
    EvaluationCase(
        case_id="spring-metric-result",
        activity="春季",
        question="春季上新活动的 GMV 和支付订单数是多少？",
        route_mode="metrics",
        expected_tools=("query_metric_rag",),
        expected_answer_terms=("经营数据分析", "春季上新活动期", "86,400.00"),
        tool_results={
            "query_metric_rag": _metric_result(
                "春季上新活动期（2026-03-08 至 2026-03-14）",
                "支付 GMV：86,400.00 元；支付订单数：312 单。",
                "paid_gmv",
                "paid_order_count",
            )
        },
    ),
    EvaluationCase(
        case_id="618-knowledge-rules",
        activity="618",
        question="618 夏日焕新活动的优惠玩法规则是什么？",
        route_mode="knowledge",
        expected_tools=("query_knowledge_rag",),
        expected_answer_terms=("618 夏日焕新活动规则", "优惠玩法"),
        tool_results={
            "query_knowledge_rag": _knowledge_result(
                "618 夏日焕新活动规则",
                "资料说明预热、正式期与返场期的节奏及优惠玩法。",
                "618-rules-001",
            )
        },
    ),
    EvaluationCase(
        case_id="qixi-knowledge-plan",
        activity="七夕",
        question="七夕礼赠新品活动方案中主推哪些货品？",
        route_mode="knowledge",
        expected_tools=("query_knowledge_rag",),
        expected_answer_terms=("七夕礼赠新品活动方案", "礼盒", "组合款"),
        tool_results={
            "query_knowledge_rag": _knowledge_result(
                "七夕礼赠新品活动方案",
                "主推保温杯礼盒与通勤双肩包保温杯礼盒组合款。",
                "qixi-plan-001",
            )
        },
    ),
    EvaluationCase(
        case_id="spring-knowledge-review",
        activity="春季",
        question="春季上新活动复盘中记录了哪些有效动作？",
        route_mode="knowledge",
        expected_tools=("query_knowledge_rag",),
        expected_answer_terms=("春季上新活动复盘", "有效动作"),
        tool_results={
            "query_knowledge_rag": _knowledge_result(
                "春季上新活动复盘",
                "资料记录了内容种草和搜索广告两类有效动作。",
                "spring-review-001",
            )
        },
    ),
    EvaluationCase(
        case_id="618-hybrid-review",
        activity="618",
        question="结合 618 正式期数据和活动规则做一次复盘。",
        route_mode="hybrid",
        expected_tools=("query_metric_rag", "query_knowledge_rag"),
        expected_answer_terms=("综合依据", "618 正式期", "优惠券"),
        tool_results={
            "query_metric_rag": _metric_result(
                "618 正式期（2026-06-06 至 2026-06-18）",
                "支付 GMV：126,540.00 元。",
                "paid_gmv",
            ),
            "query_knowledge_rag": _knowledge_result(
                "618 夏日焕新活动规则",
                "正式期使用满减与优惠券叠加玩法。",
                "618-rules-002",
            ),
        },
    ),
    EvaluationCase(
        case_id="qixi-hybrid-review",
        activity="七夕",
        question="结合七夕礼赠活动数据和方案复盘组合款表现。",
        route_mode="hybrid",
        expected_tools=("query_metric_rag", "query_knowledge_rag"),
        expected_answer_terms=("综合依据", "七夕礼赠活动期", "组合款"),
        tool_results={
            "query_metric_rag": _metric_result(
                "七夕礼赠活动期（2026-08-10 至 2026-08-22）",
                "组合款支付 GMV：31,410.00 元。",
                "paid_gmv",
            ),
            "query_knowledge_rag": _knowledge_result(
                "七夕礼赠新品活动方案",
                "组合款面向情侣与职场送礼场景。",
                "qixi-plan-002",
            ),
        },
    ),
    EvaluationCase(
        case_id="spring-hybrid-review",
        activity="春季",
        question="根据春季上新数据和复盘资料总结下一次优化方向。",
        route_mode="hybrid",
        expected_tools=("query_metric_rag", "query_knowledge_rag"),
        expected_answer_terms=("综合依据", "86,400.00", "内容种草"),
        tool_results={
            "query_metric_rag": _metric_result(
                "春季上新活动期（2026-03-08 至 2026-03-14）",
                "支付 GMV：86,400.00 元；支付订单数：312 单。",
                "paid_gmv",
                "paid_order_count",
            ),
            "query_knowledge_rag": _knowledge_result(
                "春季上新活动复盘",
                "内容种草有效，但搜索广告仍需优化承接页。",
                "spring-review-002",
            ),
        },
    ),
    EvaluationCase(
        case_id="web-policy",
        activity="通用",
        question="最新的平台公开规则有哪些？",
        route_mode="web",
        expected_tools=("search_web",),
        expected_answer_terms=("联网检索依据", "平台公开公告"),
        tool_results={"search_web": WEB_RESULT},
    ),
    EvaluationCase(
        case_id="618-web-hybrid-review",
        activity="618",
        question="结合 618 数据、活动规则和最新平台公开政策给出复盘建议。",
        route_mode="web_hybrid",
        expected_tools=("query_metric_rag", "query_knowledge_rag", "search_web"),
        expected_answer_terms=("综合依据", "618 正式期", "平台公开公告"),
        tool_results={
            "query_metric_rag": _metric_result(
                "618 正式期（2026-06-06 至 2026-06-18）",
                "支付 GMV：126,540.00 元。",
                "paid_gmv",
            ),
            "query_knowledge_rag": _knowledge_result(
                "618 夏日焕新活动规则",
                "资料记录了优惠玩法和活动节奏。",
                "618-rules-003",
            ),
            "search_web": WEB_RESULT,
        },
    ),
    EvaluationCase(
        case_id="knowledge-empty",
        activity="通用",
        question="资料库中有关于双十二会员日的报名规则吗？",
        route_mode="knowledge",
        expected_tools=("query_knowledge_rag",),
        expected_answer_terms=("未检索到",),
        tool_results={"query_knowledge_rag": EvaluationToolResult(status="empty")},
    ),
    EvaluationCase(
        case_id="web-disabled",
        activity="通用",
        question="今天北京天气如何？",
        route_mode="web",
        expected_tools=("search_web",),
        expected_answer_terms=("联网检索依据", "未检索到"),
        tool_results={
            "search_web": EvaluationToolResult(
                status="skipped", error_code="web_search_disabled"
            )
        },
    ),
)
