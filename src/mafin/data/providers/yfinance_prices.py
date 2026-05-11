from __future__ import annotations

from math import sqrt
from typing import Any

import yfinance as yf


def to_yahoo_ticker(ticker: str, default_suffix: str = ".SA") -> str:
    normalized = ticker.strip().upper()
    if normalized.startswith("^") or "." in normalized or "=" in normalized:
        return normalized
    return f"{normalized}{default_suffix}"


def _safe_float(value: Any, ndigits: int = 6) -> float | None:
    if value is None:
        return None
    try:
        if value != value:
            return None
        return round(float(value), ndigits)
    except (TypeError, ValueError):
        return None


def _iso_date(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "date"):
        return value.date().isoformat()
    return str(value)


def _classify_trend(price: float | None, ma50: float | None, ma200: float | None) -> str:
    if price is None or ma50 is None or ma200 is None:
        return "unknown"
    if price >= ma50 >= ma200:
        return "alta"
    if price <= ma50 <= ma200:
        return "baixa"
    return "lateral"


def fetch_yfinance_market_data(
    tickers: list[str],
    *,
    period: str = "1y",
    auto_adjust: bool = True,
) -> dict[str, dict[str, Any]]:
    """Fetch OHLC-derived data for each ticker using yfinance.

    Returned keys intentionally match what current agents already consume:
    `price`, `ma50`, `ma200`, `vol_30d`, and `trend`.
    """

    results: dict[str, dict[str, Any]] = {}

    for ticker in tickers:
        provider_ticker = to_yahoo_ticker(ticker)
        data_gaps: list[str] = []

        try:
            history = yf.Ticker(provider_ticker).history(
                period=period,
                interval="1d",
                auto_adjust=auto_adjust,
            )
        except Exception as exc:  # noqa: BLE001 - provider failures must be captured in snapshots
            results[ticker] = {
                "ticker": ticker,
                "provider_ticker": provider_ticker,
                "source": "yfinance",
                "data_gaps": [f"yfinance fetch failed: {type(exc).__name__}: {exc}"],
            }
            continue

        if history.empty or "Close" not in history:
            results[ticker] = {
                "ticker": ticker,
                "provider_ticker": provider_ticker,
                "source": "yfinance",
                "data_gaps": ["empty price history returned by yfinance"],
            }
            continue

        close = history["Close"].dropna()
        volume = history["Volume"].dropna() if "Volume" in history else None

        if close.empty:
            data_gaps.append("close price series is empty")

        price = _safe_float(close.iloc[-1] if not close.empty else None)
        ma50 = _safe_float(close.tail(50).mean() if len(close) >= 50 else None)
        ma200 = _safe_float(close.tail(200).mean() if len(close) >= 200 else None)

        if ma50 is None:
            data_gaps.append("not enough observations for ma50")
        if ma200 is None:
            data_gaps.append("not enough observations for ma200")

        returns = close.pct_change().dropna()
        vol_30d = _safe_float(returns.tail(30).std() * sqrt(252) if len(returns) >= 30 else None)
        if vol_30d is None:
            data_gaps.append("not enough observations for annualized 30-day volatility")

        volume_30d_avg = (
            _safe_float(volume.tail(30).mean())
            if volume is not None and len(volume)
            else None
        )

        results[ticker] = {
            "ticker": ticker,
            "provider_ticker": provider_ticker,
            "source": "yfinance",
            "price": price,
            "ma50": ma50,
            "ma200": ma200,
            "vol_30d": vol_30d,
            "trend": _classify_trend(price, ma50, ma200),
            "volume_30d_avg": volume_30d_avg,
            "history_start": _iso_date(close.index.min() if not close.empty else None),
            "history_end": _iso_date(close.index.max() if not close.empty else None),
            "observations": int(len(close)),
            "data_gaps": data_gaps,
        }

    return results
