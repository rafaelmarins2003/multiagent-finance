from __future__ import annotations

import random
from typing import Any

from mafin.data.portfolio import Portfolio

ASSET_UNIVERSE = [
    {"ticker": "TAEE11", "sector": "Utilidade Pública", "style": "defensive"},
    {"ticker": "EGIE3", "sector": "Utilidade Pública", "style": "defensive"},
    {"ticker": "BBSE3", "sector": "Seguros", "style": "defensive"},
    {"ticker": "VIVT3", "sector": "Telecomunicações", "style": "defensive"},
    {"ticker": "ABEV3", "sector": "Consumo não cíclico", "style": "defensive"},
    {"ticker": "ITUB4", "sector": "Financeiro", "style": "core"},
    {"ticker": "BBAS3", "sector": "Financeiro", "style": "core"},
    {"ticker": "B3SA3", "sector": "Financeiro", "style": "core"},
    {"ticker": "PETR4", "sector": "Energia", "style": "commodity"},
    {"ticker": "VALE3", "sector": "Mineração", "style": "commodity"},
    {"ticker": "SUZB3", "sector": "Papel e Celulose", "style": "commodity"},
    {"ticker": "WEGE3", "sector": "Industrial", "style": "growth"},
    {"ticker": "RENT3", "sector": "Locação", "style": "growth"},
    {"ticker": "RADL3", "sector": "Saúde e Varejo", "style": "growth"},
    {"ticker": "PRIO3", "sector": "Óleo e Gás", "style": "growth"},
    {"ticker": "EMBR3", "sector": "Aeroespacial", "style": "cyclical"},
    {"ticker": "JBSS3", "sector": "Alimentos", "style": "cyclical"},
    {"ticker": "LREN3", "sector": "Varejo", "style": "cyclical"},
]

PROFILES = {
    "conservative": {
        "risk_tolerance": "conservative",
        "horizon": "long",
        "objective": "preservação de capital com crescimento gradual e baixa tolerância a drawdown",
    },
    "moderate": {
        "risk_tolerance": "moderate",
        "horizon": "long",
        "objective": "valorização de capital com tolerância média a drawdown",
    },
    "aggressive": {
        "risk_tolerance": "aggressive",
        "horizon": "long",
        "objective": "crescimento de capital com alta tolerância a volatilidade e drawdown",
    },
}

STYLE_WEIGHTS = {
    "conservative": {
        "defensive": 5.0,
        "core": 2.5,
        "commodity": 0.8,
        "growth": 0.8,
        "cyclical": 0.5,
    },
    "moderate": {
        "defensive": 2.0,
        "core": 2.5,
        "commodity": 1.8,
        "growth": 1.6,
        "cyclical": 1.0,
    },
    "aggressive": {
        "defensive": 0.6,
        "core": 1.4,
        "commodity": 2.3,
        "growth": 3.0,
        "cyclical": 2.0,
    },
}


def _weighted_sample_assets(rng: random.Random, strategy: str, count: int) -> list[dict[str, str]]:
    available = ASSET_UNIVERSE.copy()
    selected = []
    for _ in range(count):
        weights = [STYLE_WEIGHTS[strategy][asset["style"]] for asset in available]
        asset = rng.choices(available, weights=weights, k=1)[0]
        selected.append(asset)
        available.remove(asset)
    return selected


def _normalize_weights(raw_weights: list[float]) -> list[float]:
    total = sum(raw_weights)
    rounded = [round(weight / total, 4) for weight in raw_weights]
    rounded[-1] = round(1.0 - sum(rounded[:-1]), 4)
    return rounded


def _portfolio_for_strategy(
    rng: random.Random,
    strategy: str,
    holdings_count: int,
) -> list[dict[str, Any]]:
    assets = _weighted_sample_assets(rng, strategy, holdings_count)

    if strategy == "conservative":
        raw_weights = [rng.uniform(0.8, 1.4) for _ in assets]
    elif strategy == "moderate":
        raw_weights = [rng.uniform(0.6, 1.8) for _ in assets]
    else:
        raw_weights = [rng.uniform(0.4, 2.6) for _ in assets]

    weights = _normalize_weights(raw_weights)
    holdings = [
        {"ticker": asset["ticker"], "weight": weight, "sector": asset["sector"]}
        for asset, weight in zip(assets, weights, strict=True)
    ]
    return Portfolio(holdings=holdings).model_dump()["holdings"]


def generate_workload(
    *,
    portfolios: int = 10,
    profiles: list[str] | None = None,
    seed: int = 42,
    min_holdings: int = 4,
    max_holdings: int = 8,
) -> dict[str, Any]:
    rng = random.Random(seed)
    selected_profiles = profiles or ["conservative", "moderate", "aggressive"]

    portfolio_specs = []
    cases = []

    for idx in range(portfolios):
        strategy = selected_profiles[idx % len(selected_profiles)]
        holdings_count = rng.randint(min_holdings, max_holdings)
        holdings = _portfolio_for_strategy(rng, strategy, holdings_count)
        portfolio_id = f"pf_{idx + 1:03d}_{strategy}"
        portfolio_specs.append(
            {
                "portfolio_id": portfolio_id,
                "construction_strategy": strategy,
                "holdings": holdings,
            }
        )

        for profile_name in selected_profiles:
            cases.append(
                {
                    "case_id": f"{portfolio_id}_{profile_name}",
                    "portfolio_id": portfolio_id,
                    "profile_name": profile_name,
                    "portfolio": holdings,
                    "profile": PROFILES[profile_name],
                }
            )

    return {
        "seed": seed,
        "portfolio_count": len(portfolio_specs),
        "case_count": len(cases),
        "profiles": selected_profiles,
        "portfolios": portfolio_specs,
        "cases": cases,
    }
