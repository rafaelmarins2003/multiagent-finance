from __future__ import annotations

from typing import Any

import yfinance as yf

from mafin.data.providers.yfinance_prices import to_yahoo_ticker

INFO_FIELDS = {
    "marketCap": "market_cap",
    "enterpriseValue": "enterprise_value",
    "trailingPE": "trailing_pe",
    "forwardPE": "forward_pe",
    "priceToBook": "price_to_book",
    "dividendYield": "dividend_yield",
    "payoutRatio": "payout_ratio",
    "totalRevenue": "total_revenue",
    "revenueGrowth": "revenue_growth",
    "grossMargins": "gross_margins",
    "operatingMargins": "operating_margins",
    "profitMargins": "profit_margins",
    "ebitda": "ebitda",
    "netIncomeToCommon": "net_income_to_common",
    "totalDebt": "total_debt",
    "debtToEquity": "debt_to_equity",
    "freeCashflow": "free_cashflow",
    "returnOnEquity": "return_on_equity",
    "returnOnAssets": "return_on_assets",
    "beta": "beta",
    "sector": "reported_sector",
    "industry": "industry",
    "longName": "company_name",
}


def _safe_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, int | str | bool):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:
        return None
    return round(number, 6)


def fetch_yfinance_fundamentals(tickers: list[str]) -> dict[str, dict[str, Any]]:

    results: dict[str, dict[str, Any]] = {}

    for ticker in tickers:
        provider_ticker = to_yahoo_ticker(ticker)
        data_gaps: list[str] = []

        try:
            info = yf.Ticker(provider_ticker).get_info()
        except Exception as exc:
            results[ticker] = {
                "ticker": ticker,
                "provider_ticker": provider_ticker,
                "source": "yfinance_info",
                "data_gaps": [f"yfinance fundamentals fetch failed: {type(exc).__name__}: {exc}"],
            }
            continue

        fields = {
            output_name: _safe_value(info.get(input_name))
            for input_name, output_name in INFO_FIELDS.items()
        }
        fields = {key: value for key, value in fields.items() if value is not None}

        missing_core = [
            name
            for name in ["market_cap", "trailing_pe", "price_to_book", "profit_margins"]
            if name not in fields
        ]
        if missing_core:
            data_gaps.append(f"missing core fields: {', '.join(missing_core)}")

        results[ticker] = {
            "ticker": ticker,
            "provider_ticker": provider_ticker,
            "source": "yfinance_info",
            **fields,
            "data_gaps": data_gaps,
        }

    return results
