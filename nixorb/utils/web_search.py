"""nixorb/utils/web_search.py — Lightweight web search via DuckDuckGo HTML."""
from __future__ import annotations

import asyncio
import html
import logging
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass

log = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

_RESULT_RE = re.compile(
    r'<a class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?'
    r'<a class="result__snippet"[^>]*>(.*?)</a>',
    re.DOTALL,
)


@dataclass
class SearchResult:
    title: str
    url:   str
    snippet: str

    def __str__(self) -> str:
        return f"[{self.title}]({self.url})\n{self.snippet}"


async def search(query: str, max_results: int = 5) -> list[SearchResult]:
    """Search DuckDuckGo and return up to *max_results* results."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _search_sync, query, max_results)


def _search_sync(query: str, max_results: int) -> list[SearchResult]:
    encoded = urllib.parse.urlencode({"q": query, "kl": "us-en"})
    url = f"https://html.duckduckgo.com/html/?{encoded}"

    try:
        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=8) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        log.error("DuckDuckGo search failed: %s", exc)
        return []

    results: list[SearchResult] = []
    for m in _RESULT_RE.finditer(body):
        raw_url, raw_title, raw_snippet = m.group(1), m.group(2), m.group(3)
        results.append(SearchResult(
            title=html.unescape(re.sub(r"<[^>]+>", "", raw_title)).strip(),
            url=raw_url.strip(),
            snippet=html.unescape(re.sub(r"<[^>]+>", "", raw_snippet)).strip(),
        ))
        if len(results) >= max_results:
            break

    log.debug("web_search '%s' → %d results", query, len(results))
    return results


async def search_formatted(query: str, max_results: int = 4) -> str:
    """Return search results as a compact string for LLM injection."""
    results = await search(query, max_results)
    if not results:
        return f"No search results found for: {query}"
    lines = [f"Web search results for '{query}':"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r.title}\n   {r.url}\n   {r.snippet}")
    return "\n\n".join(lines)
