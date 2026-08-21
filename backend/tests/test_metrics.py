from datetime import date

from app.services.metrics import MetricPeriod, MetricsOverview


def test_metrics_overview_calculates_change_and_formats_agent_context() -> None:
    overview = MetricsOverview(
        current=MetricPeriod(date(2026, 8, 15), date(2026, 8, 21), 9200, 80, 2000, 2),
        previous=MetricPeriod(date(2026, 8, 8), date(2026, 8, 14), 10000, 75, 2200, 1),
    )

    payload = overview.as_dict()

    assert payload["gmv_change_rate"] == -0.08
    assert "GMV 9,200.00 元" in overview.to_agent_context()
    assert "GMV 环比 -8.00%" in overview.to_agent_context()
