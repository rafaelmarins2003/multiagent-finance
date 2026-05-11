from __future__ import annotations

import json
import os
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from mafin.config import BRAVE_SEARCH_API_KEY
from mafin.data.news_policy import classify_news_bucket, extract_domain

BRAVE_NEWS_ENDPOINT = "https://api.search.brave.com/res/v1/news/search"


def _extract_source(item: dict[str, Any]) -> str | None:
    source = item.get("source")
    if isinstance(source, str):
        return source
    profile = item.get("profile")
    if isinstance(profile, dict):
        return profile.get("name")
    return None


def fetch_brave_news(
    query: str,
    *,
    api_key: str | None = None,
    count: int = 5,
    country: str = "br",
    search_lang: str = "pt-br",
    timeout: int = 20,
) -> dict[str, Any]:

    token = api_key or BRAVE_SEARCH_API_KEY or os.getenv("BRAVE_SEARCH_API_KEY")
    if not token:
        return {
            "provider": "brave",
            "query": query,
            "items": [],
            "data_gaps": ["BRAVE_SEARCH_API_KEY is not configured"],
        }

    params = {
        "q": query,
        "count": str(count),
        "country": country,
        "search_lang": search_lang,
    }
    request_url = f"{BRAVE_NEWS_ENDPOINT}?{urlencode(params)}"
    request = Request(
        request_url,
        headers={
            "Accept": "application/json",
            "User-Agent": "mafin-research/0.1",
            "X-Subscription-Token": token,
        },
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return {
            "provider": "brave",
            "query": query,
            "items": [],
            "data_gaps": [f"Brave fetch failed: {type(exc).__name__}: {exc}"],
        }

    items = []
    for item in payload.get("results", []):
        url = item.get("url")
        title = item.get("title")
        if not url or not title:
            continue
        source = _extract_source(item)
        items.append(
            {
                "title": title,
                "url": url,
                "source": source,
                "domain": extract_domain(url),
                "bucket": classify_news_bucket(url, source),
                "published_at": item.get("age") or item.get("page_age"),
                "description": item.get("description"),
                "provider": "brave",
                "query": query,
            }
        )

    return {
        "provider": "brave",
        "query": query,
        "items": items,
        "data_gaps": [] if items else ["Brave returned no news results"],
    }
