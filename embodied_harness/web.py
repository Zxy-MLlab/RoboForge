"""Minimal public-web discovery independent of any agent framework."""
from __future__ import annotations

from html import unescape
import re
from urllib.parse import quote_plus
from urllib.request import Request, urlopen


def search_public_web(query: str, *, limit: int = 5) -> dict:
    url = "https://html.duckduckgo.com/html/?q=" + quote_plus(query)
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 EmbodiedHarness/1.0"})
    with urlopen(request, timeout=20) as response:
        html = response.read().decode("utf-8", errors="replace")
    matches = re.findall(
        r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html,
        flags=re.DOTALL,
    )
    results = []
    for href, title in matches[:max(1, min(int(limit), 10))]:
        clean = unescape(re.sub(r"<[^>]+>", "", title)).strip()
        results.append({"title": clean, "url": unescape(href)})
    return {"query": query, "results": results, "source": "public_web"}


__all__ = ["search_public_web"]
