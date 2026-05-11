"""Generate a frozen data snapshot for a portfolio.

Examples:
  uv run python scripts/generate_snapshot.py --output data/snapshots/sample.json
  uv run python scripts/generate_snapshot.py --input examples/sample_portfolio.json \
    --output data/snapshots/sample.json
  uv run python scripts/generate_snapshot.py --skip-prices --skip-macro --skip-news
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mafin.data.portfolio import Portfolio, UserProfile
from mafin.data.snapshot_builder import build_portfolio_snapshot

SAMPLE_PORTFOLIO = [
    {"ticker": "PETR4", "weight": 0.30, "sector": "Energia"},
    {"ticker": "VALE3", "weight": 0.25, "sector": "Mineração"},
    {"ticker": "ITUB4", "weight": 0.25, "sector": "Financeiro"},
    {"ticker": "WEGE3", "weight": 0.20, "sector": "Industrial"},
]

SAMPLE_PROFILE = {
    "risk_tolerance": "moderate",
    "horizon": "long",
    "objective": "valorização de capital com tolerância média a drawdown",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Portfolio request JSON with keys `portfolio` and `profile`.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the snapshot JSON. Prints to stdout when omitted.",
    )
    parser.add_argument(
        "--price-period",
        default="1y",
        help="Period passed to yfinance history, for example 6mo, 1y, 2y.",
    )
    parser.add_argument(
        "--news-count",
        type=int,
        default=5,
        help="Raw Brave news results per query before bucket balancing.",
    )
    parser.add_argument("--skip-prices", action="store_true", help="Skip yfinance collection.")
    parser.add_argument(
        "--skip-fundamentals",
        action="store_true",
        help="Skip yfinance fundamentals collection.",
    )
    parser.add_argument("--skip-macro", action="store_true", help="Skip BCB SGS collection.")
    parser.add_argument("--skip-news", action="store_true", help="Skip Brave news collection.")
    return parser.parse_args()


def load_portfolio_request(path: Path | None) -> tuple[list[dict], dict]:
    if path is None:
        return SAMPLE_PORTFOLIO, SAMPLE_PROFILE

    payload = json.loads(path.read_text(encoding="utf-8"))
    portfolio_payload = payload.get("portfolio") or payload.get("holdings")
    if not portfolio_payload:
        raise ValueError("input JSON must include `portfolio` or `holdings`")
    if "profile" not in payload:
        raise ValueError("input JSON must include `profile`")

    portfolio = Portfolio(holdings=portfolio_payload).model_dump()["holdings"]
    profile = UserProfile.model_validate(payload["profile"]).model_dump()
    return portfolio, profile


def main() -> None:
    args = parse_args()
    portfolio, profile = load_portfolio_request(args.input)
    snapshot = build_portfolio_snapshot(
        portfolio=portfolio,
        profile=profile,
        price_period=args.price_period,
        include_prices=not args.skip_prices,
        include_fundamentals=not args.skip_fundamentals,
        include_macro=not args.skip_macro,
        include_news=not args.skip_news,
        news_count=args.news_count,
    )
    payload = snapshot.model_dump_json(indent=2)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{payload}\n", encoding="utf-8")
        print(f"Wrote snapshot to {args.output}")
        return

    print(payload)


if __name__ == "__main__":
    main()
