"""tests/test_web_search.py — Web search utility tests."""
from __future__ import annotations

from unittest.mock import patch

_RESULT_HTML = """
<div class="result">
  <a class="result__a" href="https://example.com">Example Title</a>
  <a class="result__snippet">This is a snippet about the topic.</a>
</div>
"""


def test_parse_results_extracts_fields():
    from nixorb.utils.web_search import parse_results

    results = parse_results(_RESULT_HTML, max_results=3)

    assert len(results) == 1
    assert results[0]["title"] == "Example Title"
    assert results[0]["url"] == "https://example.com"
    assert "snippet" in results[0]["snippet"]


def test_parse_results_honours_max():
    from nixorb.utils.web_search import parse_results

    results = parse_results(_RESULT_HTML * 5, max_results=2)
    assert len(results) == 2


def test_parse_results_on_junk_html():
    from nixorb.utils.web_search import parse_results

    assert parse_results("<html><body>nothing here</body></html>") == []


async def test_search_returns_list():
    from nixorb.utils.web_search import search

    with patch("nixorb.utils.web_search._fetch", return_value=_RESULT_HTML):
        results = await search("test query", max_results=3)

    assert isinstance(results, list)
    assert results[0]["title"] == "Example Title"


async def test_search_fails_gracefully():
    """A network error must degrade to no results, never propagate."""
    from nixorb.utils.web_search import search

    async def _fail(_query):
        raise OSError("network down")

    with patch("nixorb.utils.web_search._fetch", _fail):
        results = await search("anything")

    assert results == []


async def test_search_formatted_no_results():
    from nixorb.utils.web_search import search_formatted

    async def _none(_query, _max):
        return []

    with patch("nixorb.utils.web_search.search", _none):
        result = await search_formatted("xyz")

    assert "No search results" in result


async def test_search_formatted_includes_titles_and_urls():
    from nixorb.utils.web_search import search_formatted

    async def _one(_query, _max):
        return [{"title": "T", "snippet": "S", "url": "https://u"}]

    with patch("nixorb.utils.web_search.search", _one):
        result = await search_formatted("xyz")

    assert "T" in result and "https://u" in result


async def test_wants_web_detection():
    from nixorb.main import _wants_web

    assert _wants_web("what is the current price of bitcoin")
    assert _wants_web("search for Arch Linux news")
    assert _wants_web("who is Linus Torvalds")
    assert not _wants_web("open my terminal")


async def test_wants_screen_detection():
    from nixorb.main import _wants_screen

    assert _wants_screen("what am I looking at")
    assert _wants_screen("what's on my screen")
    assert _wants_screen("see my screen")
    assert not _wants_screen("play some music")
