"""搜索模块测试"""

import pytest
from rlwriter.search import (
    SearchManager,
    format_search_results,
    search_duckduckgo,
    search_wikipedia,
    search_tavily,
    search_exa,
)


class TestSearchManagerInit:
    """SearchManager 初始化测试"""

    def test_default_config(self):
        mgr = SearchManager({})
        assert mgr.enabled is True
        assert mgr.mode == "llm_select"
        assert mgr.source == "auto"
        assert mgr.max_results == 3
        assert mgr.api_keys == {}

    def test_custom_config(self):
        config = {
            "enabled": False,
            "mode": "manual",
            "source": "duckduckgo",
            "max_results": 10,
            "api_keys": {"tavily": "key1", "exa": "key2"},
        }
        mgr = SearchManager(config)
        assert mgr.enabled is False
        assert mgr.mode == "manual"
        assert mgr.source == "duckduckgo"
        assert mgr.max_results == 10
        assert mgr.api_keys == {"tavily": "key1", "exa": "key2"}

    def test_partial_config(self):
        mgr = SearchManager({"enabled": True, "mode": "auto"})
        assert mgr.enabled is True
        assert mgr.mode == "auto"
        assert mgr.source == "auto"  # default


class TestGetAvailableSources:
    """_get_available_sources 测试"""

    def test_with_all_keys(self):
        mgr = SearchManager({"api_keys": {"tavily": "k1", "exa": "k2"}})
        sources = mgr._get_available_sources()
        assert "tavily" in sources
        assert "exa" in sources
        assert "duckduckgo" in sources
        assert "wikipedia" in sources

    def test_with_no_keys(self):
        mgr = SearchManager({"api_keys": {}})
        sources = mgr._get_available_sources()
        assert "tavily" not in sources
        assert "exa" not in sources
        assert "duckduckgo" in sources
        assert "wikipedia" in sources

    def test_with_only_tavily(self):
        mgr = SearchManager({"api_keys": {"tavily": "k1"}})
        sources = mgr._get_available_sources()
        assert "tavily" in sources
        assert "exa" not in sources
        assert "duckduckgo" in sources


class TestSearchDisabled:
    """搜索禁用测试"""

    def test_disabled_returns_empty(self):
        mgr = SearchManager({"enabled": False})
        result = mgr.search("test query")
        assert result == []


class TestFormatSearchResults:
    """format_search_results 测试"""

    def test_empty_results(self):
        assert format_search_results([]) == "（无搜索结果）"

    def test_basic_format(self):
        results = [
            {"title": "Test Title", "url": "https://example.com", "snippet": "Test snippet"},
        ]
        output = format_search_results(results)
        assert "Test Title" in output
        assert "Test snippet" in output
        assert "https://example.com" in output

    def test_max_items_limit(self):
        results = [
            {"title": f"Title {i}", "url": f"https://example.com/{i}", "snippet": f"Snippet {i}"}
            for i in range(10)
        ]
        output = format_search_results(results, max_items=3)
        assert "Title 0" in output
        assert "Title 2" in output
        assert "Title 3" not in output

    def test_missing_fields(self):
        results = [{"title": "Only Title"}]
        output = format_search_results(results)
        assert "Only Title" in output
        assert "N/A" not in output  # title exists

    def test_snippet_truncation(self):
        results = [{"title": "T", "snippet": "A" * 500}]
        output = format_search_results(results)
        # snippet is truncated to 200 chars in output
        assert "A" * 200 in output


class TestSearchSourceGracefulDegradation:
    """搜索源优雅降级测试（无 API key）"""

    def test_tavily_no_key(self):
        result = search_tavily("test", api_key=None)
        assert result == []

    def test_exa_no_key(self):
        result = search_exa("test", api_key=None)
        assert result == []

    def test_tavily_empty_key(self):
        result = search_tavily("test", api_key="")
        assert result == []

    def test_exa_empty_key(self):
        result = search_exa("test", api_key="")
        assert result == []


class TestSearchFallback:
    """auto 模式降级测试"""

    def test_auto_fallback_no_api_keys(self):
        """没有 API keys 时，auto 模式应尝试 duckduckgo/wikipedia（mock 网络）"""
        from unittest.mock import patch
        mgr = SearchManager({
            "mode": "auto",
            "api_keys": {},
            "max_results": 1,
        })
        # mock 所有搜索源返回空，避免真实网络请求
        with patch("rlwriter.search.search_duckduckgo", return_value=[]), \
             patch("rlwriter.search.search_wikipedia", return_value=[]):
            result = mgr.search("Python programming")
            assert isinstance(result, list)


class TestLLMSelectSource:
    """llm_select_source 测试"""

    def test_returns_none_on_llm_failure(self):
        """LLM 调用失败时应返回 ('none', '')"""
        mgr = SearchManager({"api_keys": {}})

        class FakeClient:
            pass

        source, query = mgr.llm_select_source(FakeClient(), "some context")
        assert source == "none"
        assert query == ""
