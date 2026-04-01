"""搜索模块 — 四种搜索源：DuckDuckGo / Wikipedia / Tavily / Exa"""

import logging
from typing import List, Dict, Optional

log = logging.getLogger(__name__)

# 搜索结果统一格式: [{"title": str, "url": str, "snippet": str, "score": float}, ...]


def search_duckduckgo(query: str, max_results: int = 5) -> List[Dict]:
    """DuckDuckGo 搜索（免费，无需 API key）"""
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        log.warning("duckduckgo-search not installed. pip install duckduckgo-search")
        return []
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return [{"title": r["title"], "url": r["href"], "snippet": r["body"]} for r in results]
    except Exception as e:
        log.warning(f"DuckDuckGo search failed: {e}")
        return []


def search_wikipedia(query: str, max_results: int = 3) -> List[Dict]:
    """Wikipedia 搜索（免费）"""
    try:
        import wikipedia
    except ImportError:
        log.warning("wikipedia not installed. pip install wikipedia")
        return []
    try:
        wikipedia.set_lang("zh")
        search_results = wikipedia.search(query, results=max_results)
        items = []
        for title in search_results:
            try:
                summary = wikipedia.summary(title, sentences=3)
                page = wikipedia.page(title)
                items.append({"title": title, "url": page.url, "snippet": summary})
            except Exception:
                continue
        return items
    except Exception as e:
        log.warning(f"Wikipedia search failed: {e}")
        return []


def search_tavily(query: str, max_results: int = 5, api_key: str = None) -> List[Dict]:
    """Tavily 搜索"""
    if not api_key:
        log.warning("Tavily API key not set")
        return []
    try:
        from tavily import TavilyClient
    except ImportError:
        log.warning("tavily-python not installed. pip install tavily-python")
        return []
    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(query, max_results=max_results)
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", ""),
                "score": r.get("score", 0),
            }
            for r in response.get("results", [])
        ]
    except Exception as e:
        log.warning(f"Tavily search failed: {e}")
        return []


def search_exa(query: str, max_results: int = 5, api_key: str = None) -> List[Dict]:
    """Exa 神经搜索"""
    if not api_key:
        log.warning("Exa API key not set")
        return []
    try:
        from exa_py import Exa
    except ImportError:
        log.warning("exa-py not installed. pip install exa-py")
        return []
    try:
        client = Exa(api_key=api_key)
        response = client.search(query, num_results=max_results)
        return [
            {
                "title": r.title or "",
                "url": r.url or "",
                "snippet": (r.text or "")[:300],
                "score": r.score or 0,
            }
            for r in response.results
        ]
    except Exception as e:
        log.warning(f"Exa search failed: {e}")
        return []


class SearchManager:
    """搜索管理器，支持 auto 降级和 llm_select 智能选择"""

    SOURCE_FUNCS = {
        "duckduckgo": search_duckduckgo,
        "wikipedia": search_wikipedia,
    }
    # auto 模式下的降级优先级
    AUTO_PRIORITY = ["tavily", "exa", "duckduckgo", "wikipedia"]

    def __init__(self, config: dict):
        """
        config 格式:
        {
            "enabled": True,
            "mode": "llm_select",  # auto / manual / llm_select
            "source": "auto",
            "max_results": 3,
            "api_keys": {"tavily": "xxx", "exa": "xxx"}
        }
        """
        self.enabled = config.get("enabled", True)
        self.mode = config.get("mode", "llm_select")
        self.source = config.get("source", "auto")
        self.max_results = config.get("max_results", 3)
        self.api_keys = config.get("api_keys", {})

    def _get_available_sources(self) -> List[str]:
        """返回可用的搜索源列表"""
        available = []
        if self.api_keys.get("tavily"):
            available.append("tavily")
        if self.api_keys.get("exa"):
            available.append("exa")
        available.extend(["duckduckgo", "wikipedia"])  # 免费，总是可用
        return available

    def search(self, query: str, source: str = None, max_results: int = None) -> List[Dict]:
        """执行搜索。source 为 None 时按 mode 决定。"""
        if not self.enabled:
            return []

        max_results = max_results or self.max_results

        # 确定用哪个源
        if source:
            return self._do_search(source, query, max_results)

        if self.mode == "manual":
            return self._do_search(self.source, query, max_results)

        # auto / llm_select 都走降级
        for src in self.AUTO_PRIORITY:
            results = self._do_search(src, query, max_results)
            if results:
                log.info(f"Search: used {src} for '{query}'")
                return results

        log.warning(f"Search: all sources failed for '{query}'")
        return []

    def _do_search(self, source: str, query: str, max_results: int) -> List[Dict]:
        """执行单个搜索源"""
        try:
            if source == "duckduckgo":
                return search_duckduckgo(query, max_results)
            elif source == "wikipedia":
                return search_wikipedia(query, max_results)
            elif source == "tavily":
                return search_tavily(query, max_results, self.api_keys.get("tavily"))
            elif source == "exa":
                return search_exa(query, max_results, self.api_keys.get("exa"))
        except Exception as e:
            log.warning(f"Search [{source}] failed: {e}")
        return []

    def llm_select_source(self, llm_client, context: str, task_type: str = "chapter", llm_cfg=None) -> tuple:
        """让 LLM 选择搜索源和搜索词。

        Returns:
            (source, query) — source 可为 "none" 表示不需要搜索
        """
        available = self._get_available_sources()
        sources_desc = {
            "tavily": "高质量事实检索，适合历史/科学/现实题材",
            "exa": "神经搜索，适合创意灵感和语义检索",
            "duckduckgo": "通用网页搜索，免费",
            "wikipedia": "百科检索，适合历史/地理/人物背景",
        }
        available_desc = "\n".join(f"- {s}: {sources_desc.get(s, '')}" for s in available)

        prompt = f"""你是一位小说创作助手。当前任务：生成小说{task_type}。

背景信息：
{context[:500]}

可用搜索源：
{available_desc}

请判断：
1. 是否需要搜索背景信息？如果题材是纯架空/奇幻不需要真实背景，回答"不需要"。
2. 如果需要，选择最佳搜索源和搜索关键词。

输出格式（JSON）：
{{"need_search": true/false, "source": "搜索源名称", "query": "搜索关键词"}}
如果不需要搜索：{{"need_search": false, "source": "none", "query": ""}}"""

        try:
            import json
            import re

            from ..utils import llm_call, LLMConfig

            messages = [{"role": "user", "content": prompt}]
            _cfg = llm_cfg or LLMConfig()
            resp = llm_call(
                llm_client,
                _cfg,
                messages,
                temperature=0.3,
                max_tokens=500,
            )
            match = re.search(r'\{[\s\S]*\}', resp)
            if match:
                data = json.loads(match.group())
                if data.get("need_search") and data.get("source") in available:
                    return data["source"], data.get("query", "")
        except Exception as e:
            log.warning(f"LLM search selection failed: {e}")

        return "none", ""


def format_search_results(results: List[Dict], max_items: int = 5) -> str:
    """格式化搜索结果为文本"""
    if not results:
        return "（无搜索结果）"
    lines = []
    for i, r in enumerate(results[:max_items], 1):
        lines.append(f"{i}. {r.get('title', 'N/A')}")
        lines.append(f"   {r.get('snippet', '')[:200]}")
        if r.get("url"):
            lines.append(f"   来源: {r['url']}")
        lines.append("")
    return "\n".join(lines)
