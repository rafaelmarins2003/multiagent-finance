from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

from mafin.data.news_policy import (
    NEWS_BUCKET_TARGETS,
    balance_news_items,
    bucket_counts,
    build_news_queries,
)
from mafin.data.providers import (
    fetch_bcb_macro_snapshot,
    fetch_brave_news,
    fetch_yfinance_fundamentals,
    fetch_yfinance_market_data,
)
from mafin.data.snapshot import DataSourceRecord, PortfolioDataSnapshot


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _portfolio_tickers(portfolio: list[dict[str, Any]]) -> list[str]:
    return [str(holding["ticker"]).strip().upper() for holding in portfolio]


def build_portfolio_snapshot(
    *,
    portfolio: list[dict[str, Any]],
    profile: dict[str, Any],
    as_of_date: date | None = None,
    price_period: str = "1y",
    include_prices: bool = True,
    include_fundamentals: bool = True,
    include_macro: bool = True,
    include_news: bool = True,
    news_count: int = 5,
    news_bucket_targets: dict[str, int] | None = None,
) -> PortfolioDataSnapshot:
    created_at = _now_iso()
    snapshot_date = as_of_date or date.today()
    snapshot_id = f"portfolio-{snapshot_date.isoformat()}-{created_at[:19].replace(':', '')}"
    tickers = _portfolio_tickers(portfolio)

    sources: list[DataSourceRecord] = []
    data_gaps: list[str] = []
    news_targets = news_bucket_targets or NEWS_BUCKET_TARGETS
    news_policy: dict[str, Any] = {}

    market_data: dict[str, dict[str, Any]] = {}
    if include_prices:
        market_data = fetch_yfinance_market_data(tickers, period=price_period)
        sources.append(
            DataSourceRecord(
                name="yfinance",
                kind="prices",
                retrieved_at=_now_iso(),
                detail=f"period={price_period}",
            )
        )
        for ticker, values in market_data.items():
            for gap in values.get("data_gaps", []):
                data_gaps.append(f"{ticker}: {gap}")

    fundamental_data: dict[str, dict[str, Any]] = {}
    if include_fundamentals:
        fundamental_data = fetch_yfinance_fundamentals(tickers)
        sources.append(
            DataSourceRecord(
                name="yfinance",
                kind="fundamentals",
                retrieved_at=_now_iso(),
                detail="info selected fields",
            )
        )
        for ticker, values in fundamental_data.items():
            for gap in values.get("data_gaps", []):
                data_gaps.append(f"{ticker}: {gap}")

    macro_data: dict[str, Any] = {}
    if include_macro:
        macro_data = fetch_bcb_macro_snapshot()
        sources.append(
            DataSourceRecord(
                name="bcb_sgs",
                kind="macro",
                retrieved_at=_now_iso(),
                detail="default SGS series",
            )
        )
        data_gaps.extend([f"macro: {gap}" for gap in macro_data.get("data_gaps", [])])

    sentiment_data: dict[str, dict[str, Any]] = {}
    if include_news:
        for holding in portfolio:
            ticker = str(holding["ticker"]).strip().upper()
            queries = build_news_queries(ticker, holding.get("sector"))
            raw_items: list[dict[str, Any]] = []
            query_results = []

            for query_spec in queries:
                news = fetch_brave_news(query_spec["query"], count=news_count)
                raw_items.extend(news.get("items", []))
                query_results.append(
                    {
                        "bucket_hint": query_spec["bucket_hint"],
                        "query": query_spec["query"],
                        "items_returned": len(news.get("items", [])),
                        "data_gaps": news.get("data_gaps", []),
                    }
                )
                for gap in news.get("data_gaps", []):
                    data_gaps.append(f"{ticker}: {query_spec['bucket_hint']}: {gap}")

            balanced_items = balance_news_items(raw_items, bucket_targets=news_targets)
            sentiment_data[ticker] = {
                "provider": "brave",
                "policy": "balanced_source_buckets",
                "bucket_targets": news_targets,
                "bucket_counts": bucket_counts(balanced_items),
                "queries": query_results,
                "raw_items_returned": len(raw_items),
                "items": balanced_items,
                "data_gaps": []
                if balanced_items
                else ["no usable Brave news items after balanced selection"],
            }
            for gap in sentiment_data[ticker]["data_gaps"]:
                data_gaps.append(f"{ticker}: {gap}")

        news_policy = {
            "provider": "brave",
            "selection": "balanced_source_buckets",
            "bucket_targets": news_targets,
            "buckets": list(news_targets),
            "raw_results_per_query": news_count,
        }
        sources.append(
            DataSourceRecord(
                name="brave",
                kind="news",
                retrieved_at=_now_iso(),
                detail=f"raw_results_per_query={news_count}; balanced_source_buckets",
            )
        )

    return PortfolioDataSnapshot(
        snapshot_id=snapshot_id,
        created_at=created_at,
        as_of_date=snapshot_date.isoformat(),
        portfolio=portfolio,
        profile=profile,
        market_data=market_data,
        fundamental_data=fundamental_data,
        sentiment_data=sentiment_data,
        macro_data=macro_data,
        news_policy=news_policy,
        sources=sources,
        data_gaps=data_gaps,
    )
