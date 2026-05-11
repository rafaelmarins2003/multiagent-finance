from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BCB_SGS_BASE_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"

DEFAULT_SGS_SERIES = {
    "selic_target": 432,
    "selic_over": 11,
    "cdi": 12,
    "ipca_monthly": 433,
    "ipca_12m": 13522,
    "usd_brl": 1,
    "ibc_br_seasonally_adjusted": 24364,
}


def _format_br_date(value: date) -> str:
    return value.strftime("%d/%m/%Y")


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).replace(",", "."))
    except ValueError:
        return None


def _fetch_json(url: str, params: dict[str, str], timeout: int) -> Any:
    request_url = f"{url}?{urlencode(params)}"
    request = Request(
        request_url,
        headers={"User-Agent": "mafin-research/0.1"},
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_bcb_macro_snapshot(
    *,
    start: date | None = None,
    end: date | None = None,
    series: dict[str, int] | None = None,
    timeout: int = 20,
) -> dict[str, Any]:
    """Fetch selected Brazilian macro time series from BCB SGS."""

    end = end or date.today()
    start = start or end - timedelta(days=365 * 2)
    selected_series = series or DEFAULT_SGS_SERIES

    macro_series: dict[str, list[dict[str, Any]]] = {}
    latest: dict[str, dict[str, Any]] = {}
    data_gaps: list[str] = []

    for name, code in selected_series.items():
        try:
            payload = _fetch_json(
                BCB_SGS_BASE_URL.format(code=code),
                {
                    "formato": "json",
                    "dataInicial": _format_br_date(start),
                    "dataFinal": _format_br_date(end),
                },
                timeout,
            )
        except Exception as exc:  # noqa: BLE001 - provider failures must be preserved
            data_gaps.append(f"{name} ({code}) fetch failed: {type(exc).__name__}: {exc}")
            macro_series[name] = []
            continue

        observations = [
            {
                "date": item.get("data"),
                "value": _safe_float(item.get("valor")),
                "series_code": code,
            }
            for item in payload
        ]
        observations = [item for item in observations if item["date"] and item["value"] is not None]

        macro_series[name] = observations
        if observations:
            latest[name] = observations[-1]
        else:
            data_gaps.append(f"{name} ({code}) returned no observations")

    return {
        "source": "bcb_sgs",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "series": macro_series,
        "latest": latest,
        "data_gaps": data_gaps,
    }
