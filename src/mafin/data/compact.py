from __future__ import annotations

from typing import Any


def _truncate(text: str | None, limit: int) -> str | None:
    if text is None or len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}..."


def compact_market_data(market_data: dict[str, Any]) -> dict[str, Any]:
    allowed_fields = {
        "ticker",
        "provider_ticker",
        "source",
        "currency",
        "price",
        "previous_close",
        "ma50",
        "ma200",
        "vol_30d",
        "trend",
        "volume_avg",
        "history_start",
        "history_end",
        "data_gaps",
    }
    return {
        ticker: {key: value for key, value in payload.items() if key in allowed_fields}
        for ticker, payload in market_data.items()
        if isinstance(payload, dict)
    }


def compact_sentiment_data(
    sentiment_data: dict[str, Any],
    *,
    max_items_per_ticker: int = 8,
    max_description_chars: int = 240,
) -> dict[str, Any]:
    compacted: dict[str, Any] = {}

    for ticker, payload in sentiment_data.items():
        items = []
        for item in payload.get("items", [])[:max_items_per_ticker]:
            items.append(
                {
                    "title": item.get("title"),
                    "source": item.get("source"),
                    "domain": item.get("domain"),
                    "bucket": item.get("bucket"),
                    "published_at": item.get("published_at"),
                    "description": _truncate(item.get("description"), max_description_chars),
                    "query": item.get("query"),
                }
            )

        compacted[ticker] = {
            "provider": payload.get("provider"),
            "policy": payload.get("policy"),
            "bucket_counts": payload.get("bucket_counts", {}),
            "items": items,
            "data_gaps": payload.get("data_gaps", []),
        }

    return compacted


def compact_macro_data(macro_data: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": macro_data.get("source"),
        "start": macro_data.get("start"),
        "end": macro_data.get("end"),
        "latest": macro_data.get("latest", {}),
        "data_gaps": macro_data.get("data_gaps", []),
    }


def compact_fundamental_data(fundamental_data: dict[str, Any]) -> dict[str, Any]:
    allowed_fields = {
        "ticker",
        "provider_ticker",
        "source",
        "company_name",
        "reported_sector",
        "industry",
        "market_cap",
        "enterprise_value",
        "trailing_pe",
        "forward_pe",
        "price_to_book",
        "dividend_yield",
        "payout_ratio",
        "total_revenue",
        "revenue_growth",
        "gross_margins",
        "operating_margins",
        "profit_margins",
        "ebitda",
        "net_income_to_common",
        "total_debt",
        "debt_to_equity",
        "free_cashflow",
        "return_on_equity",
        "return_on_assets",
        "beta",
        "data_gaps",
    }
    return {
        ticker: {key: value for key, value in payload.items() if key in allowed_fields}
        for ticker, payload in fundamental_data.items()
    }
