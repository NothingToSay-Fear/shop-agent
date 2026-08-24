import json

import httpx
import pytest

from app.config import Settings
from app.services.web_search import TAVILY_SEARCH_URL, search_web_for_question


@pytest.mark.asyncio
async def test_web_search_returns_bounded_public_sources() -> None:
    """联网工具仅保留可引用的 HTTP(S) 结果，并固定使用基础检索。"""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "平台公告",
                        "url": "https://example.com/notice",
                        "content": "这是可供核验的公开摘要。",
                    },
                    {"title": "无效来源", "url": "file:///unsafe", "content": "不应保留。"},
                ]
            },
        )

    settings = Settings(web_search_api_key="test-key", web_search_max_results=3)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        context = await search_web_for_question("最新平台规则", settings, client)

    assert context is not None
    assert len(context.items) == 1
    assert "不可信的网页摘要" in context.text
    assert "https://example.com/notice" in context.references
    assert requests[0].url == httpx.URL(TAVILY_SEARCH_URL)
    payload = json.loads(requests[0].content)
    assert payload["search_depth"] == "basic"
    assert payload["include_raw_content"] is False


@pytest.mark.asyncio
async def test_web_search_is_disabled_without_an_api_key() -> None:
    """没有密钥时不得隐式访问外部网络。"""
    context = await search_web_for_question("最新平台规则", Settings(web_search_api_key=None))

    assert context is None
