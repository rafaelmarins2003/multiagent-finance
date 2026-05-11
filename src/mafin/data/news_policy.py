from __future__ import annotations

from collections import defaultdict
from typing import Any
from urllib.parse import urlparse

NEWS_BUCKET_TARGETS = {
    "official": 2,
    "financial_media": 4,
    "general_media": 4,
    "international": 2,
    "other": 2,
}

DOMAIN_BUCKETS = {
    "b3.com.br": "official",
    "bcb.gov.br": "official",
    "cvm.gov.br": "official",
    "gov.br": "official",
    "investidorpetrobras.com.br": "official",
    "petrobras.com.br": "official",
    "vale.com": "official",
    "vale.com.br": "official",
    "itau.com.br": "official",
    "weg.net": "official",
    "infomoney.com.br": "financial_media",
    "valor.globo.com": "financial_media",
    "valorinveste.globo.com": "financial_media",
    "moneytimes.com.br": "financial_media",
    "braziljournal.com": "financial_media",
    "suno.com.br": "financial_media",
    "investing.com": "financial_media",
    "statusinvest.com.br": "financial_media",
    "exame.com": "financial_media",
    "einvestidor.estadao.com.br": "financial_media",
    "seudinheiro.com": "financial_media",
    "investidor10.com.br": "financial_media",
    "guiadoinvestidor.com.br": "financial_media",
    "analisedeacoes.com": "financial_media",
    "acionista.com.br": "financial_media",
    "g1.globo.com": "general_media",
    "uol.com.br": "general_media",
    "folha.uol.com.br": "general_media",
    "estadao.com.br": "general_media",
    "cnnbrasil.com.br": "general_media",
    "metropoles.com": "general_media",
    "poder360.com.br": "general_media",
    "reuters.com": "international",
    "bloomberg.com": "international",
    "ft.com": "international",
    "wsj.com": "international",
    "cnbc.com": "international",
    "marketwatch.com": "international",
    "bloomberglinea.com.br": "international",
}

SOURCE_NAME_BUCKETS = {
    "b3": "official",
    "cvm": "official",
    "banco central": "official",
    "relações com investidores": "official",
    "relacoes com investidores": "official",
    "petrobras": "official",
    "vale": "official",
    "itaú": "official",
    "itau": "official",
    "weg": "official",
    "infomoney": "financial_media",
    "valor": "financial_media",
    "valor inv": "financial_media",
    "money times": "financial_media",
    "brazil journal": "financial_media",
    "suno": "financial_media",
    "investing": "financial_media",
    "seu dinheiro": "financial_media",
    "investidor10": "financial_media",
    "guia do investidor": "financial_media",
    "exame": "financial_media",
    "g1": "general_media",
    "uol": "general_media",
    "folha": "general_media",
    "estadão": "general_media",
    "estadao": "general_media",
    "cnn brasil": "general_media",
    "metrópoles": "general_media",
    "metropoles": "general_media",
    "reuters": "international",
    "bloomberg": "international",
    "financial times": "international",
    "wall street journal": "international",
    "cnbc": "international",
}


def extract_domain(url: str | None) -> str | None:
    if not url:
        return None
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain or None


def classify_news_bucket(url: str | None, source: str | None = None) -> str:
    domain = extract_domain(url)
    if domain:
        for known_domain, bucket in DOMAIN_BUCKETS.items():
            if domain == known_domain or domain.endswith(f".{known_domain}"):
                return bucket

    normalized_source = (source or "").lower()
    for source_fragment, bucket in SOURCE_NAME_BUCKETS.items():
        if source_fragment in normalized_source:
            return bucket

    return "other"


def build_news_queries(ticker: str, sector: str | None = None) -> list[dict[str, str]]:
    base = ticker.strip().upper()
    sector_part = f' "{sector}"' if sector else ""
    return [
        {"bucket_hint": "official", "query": f"{base} B3 CVM RI fato relevante"},
        {
            "bucket_hint": "financial_media",
            "query": f"{base} ações B3 mercado financeiro{sector_part}",
        },
        {"bucket_hint": "general_media", "query": f"{base} empresa notícias Brasil{sector_part}"},
        {"bucket_hint": "international", "query": f"{base} Brazil stock Reuters Bloomberg"},
    ]


def annotate_news_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    annotated = []
    for item in items:
        domain = item.get("domain") or extract_domain(item.get("url"))
        source = item.get("source")
        annotated.append(
            {
                **item,
                "domain": domain,
                "bucket": item.get("bucket") or classify_news_bucket(item.get("url"), source),
            }
        )
    return annotated


def dedupe_news_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped = []
    for item in items:
        key = (item.get("url") or item.get("title") or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def balance_news_items(
    items: list[dict[str, Any]],
    *,
    bucket_targets: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    targets = bucket_targets or NEWS_BUCKET_TARGETS
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in dedupe_news_items(annotate_news_items(items)):
        grouped[item.get("bucket") or "other"].append(item)

    selected: list[dict[str, Any]] = []
    selected_urls: set[str] = set()

    for bucket, target in targets.items():
        for item in grouped.get(bucket, [])[:target]:
            selected.append(item)
            selected_urls.add(item["url"])

    total_target = sum(targets.values())
    if len(selected) < total_target:
        for item in dedupe_news_items(annotate_news_items(items)):
            if item["url"] in selected_urls:
                continue
            selected.append(item)
            selected_urls.add(item["url"])
            if len(selected) >= total_target:
                break

    return selected


def bucket_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts = {bucket: 0 for bucket in NEWS_BUCKET_TARGETS}
    for item in annotate_news_items(items):
        bucket = item.get("bucket") or "other"
        counts[bucket] = counts.get(bucket, 0) + 1
    return counts
