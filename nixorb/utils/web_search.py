"""NixOrb web search utility.

Searches the web using DuckDuckGo and returns formatted results
for injection into the LLM prompt context.
"""
from __future__ import annotations

import logging
from urllib.parse import quote_plus

import aiohttp

log = logging.getLogger(__name__)

DUCKDUCKGO_URL = "https://html.duckduckgo.com/html/"
REQUEST_TIMEOUT = 15


async def search_formatted(query: str, max_results: int = 4) -> str:
    """Search the web and return formatted results.

    Args:
        query: Search query
        max_results: Maximum number of results to include

    Returns:
        Formatted search results for LLM context
    """
    try:
        results = await _search_duckduckgo(query, max_results)
        if not results:
            return "\n[Web search: No results found]\n"

        formatted = ["\n[Web search results]:"]
        for i, result in enumerate(results[:max_results], 1):
            formatted.append(f"{i}. {result['title']}")
            formatted.append(f"   {result['snippet']}")
            formatted.append(f"   URL: {result['url']}")

        return "\n".join(formatted) + "\n"

    except Exception as exc:
        log.warning("Web search failed: %s", exc)
        return f"\n[Web search: Error — {exc}]\n"


async def _search_duckduckgo(query: str, max_results: int) -> list[dict]:
    """Search DuckDuckGo and parse results."""
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    ) as session:
        params = {"q": query}
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        }

        async with session.get(
            DUCKDUCKGO_URL, params=params, headers=headers
        ) as resp:
            html = await resp.text()

    # Simple HTML parsing
    from html.parser import HTMLParser

    class ResultParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.results = []
            self._current = {}
            self._in_result = False
            self._in_title = False
            self._in_snippet = False
            self._tag_stack = []

        def handle_starttag(self, tag, attrs):
            attrs_dict = dict(attrs)
            self._tag_stack.append(tag)

            if tag == "div" and "result" in attrs_dict.get("class", ""):
                self._in_result = True
                self._current = {}

            if self._in_result:
                if tag == "a" and "result__a" in attrs_dict.get("class", ""):
                    self._in_title = True
                    self._current["url"] = attrs_dict.get("href", "")

                if tag == "a" and "result__snippet" in attrs_dict.get("class", ""):
                    self._in_snippet = True

        def handle_endtag(self, tag):
            if self._tag_stack:
                self._tag_stack.pop()

            if tag == "div" and self._in_result:
                if self._current.get("title") and self._current.get("snippet"):
                    self.results.append(self._current)
                self._in_result = False

            if tag == "a":
                self._in_title = False
                self._in_snippet = False

        def handle_data(self, data):
            if self._in_title:
                self._current["title"] = data.strip()
            elif self._in_snippet:
                self._current["snippet"] = data.strip()

    parser = ResultParser()
    parser.feed(html)
    return parser.results[:max_results]
