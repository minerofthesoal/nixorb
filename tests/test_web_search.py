"""tests/test_web_search.py — Web search utility tests."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.asyncio


async def test_search_returns_list():
    from nixorb.utils.web_search import search

    mock_html = """
    <a class="result__a" href="https://example.com">Example Title</a>
    <a class="result__snippet">This is a snippet about the topic.</a>
    """

    def _fake_urlopen(*args, **kwargs):
        m = MagicMock()
        m.__enter__ = lambda s: s
        m.__exit__  = MagicMock(return_value=False)
        m.read      = lambda: mock_html.encode()
        return m

    with patch("urllib.request.urlopen", _fake_urlopen):
        results = await search("test query", max_results=3)

    assert isinstance(results, list)


async def test_search_fails_gracefully():
    from nixorb.utils.web_search import search

    def _fail(*args, **kwargs):
        raise OSError("network down")

    with patch("urllib.request.urlopen", _fail):
        results = await search("anything")

    assert results == []


async def test_search_formatted_no_results():
    from nixorb.utils.web_search import search_formatted

    with patch("nixorb.utils.web_search.search", return_value=[]):
        result = await search_formatted("xyz")

    assert "No search results" in result


async def test_wants_web_detection():
    from nixorb.main import _wants_web

    assert _wants_web("what is the current price of bitcoin")
    assert _wants_web("search for Arch Linux news")
    assert _wants_web("who is Linus Torvalds")
    assert not _wants_web("open my terminal")
    assert not _wants_web("what time is it locally")


async def test_wants_screen_detection():
    from nixorb.main import _wants_screen

    assert _wants_screen("what am I looking at")
    assert _wants_screen("what's on my screen")
    assert _wants_screen("see my screen")
    assert not _wants_screen("play some music")
