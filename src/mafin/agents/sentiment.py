import json
from typing import Any

from mafin.agents.base import Agent
from mafin.data.compact import compact_sentiment_data
from mafin.schema import SentimentAnalysisOutput, TickerSentimentAnalysis

SENTIMENT_FIELDS = {
    "articles",
    "events",
    "headlines",
    "mentions",
    "news",
    "sentiment",
    "sentiment_score",
}

SYSTEM_PROMPT = """Você é um analista de sentimento para carteiras de ações brasileiras.

Seu papel: avaliar sinais de sentimento por ativo a partir de notícias, eventos,
menções ou campos textuais explicitamente fornecidos no input.

Restrições:
- Não recomende compra ou venda.
- Use apenas evidências presentes no input. Não invente notícias, eventos ou fontes.
- Não use preço, média móvel, volatilidade ou tendência técnica como proxy de sentimento.
- Se não houver notícias, eventos ou sinais textuais, classifique como unknown e registre a lacuna.
"""


def _extract_sentiment_data(market_data: dict[str, Any]) -> dict[str, Any]:
    return {
        ticker: {key: value for key, value in data.items() if key in SENTIMENT_FIELDS}
        for ticker, data in market_data.items()
        if isinstance(data, dict)
        and any(key in SENTIMENT_FIELDS for key in data)
    }


def _normalize_sentiment_output(
    output: SentimentAnalysisOutput,
    portfolio: list[dict[str, Any]],
) -> SentimentAnalysisOutput:
    expected_tickers = [str(holding["ticker"]).strip().upper() for holding in portfolio]
    by_ticker = {
        item.ticker.strip().upper(): item
        for item in output.per_ticker
        if item.ticker.strip().upper() in expected_tickers
    }

    normalized_items = []
    for ticker in expected_tickers:
        item = by_ticker.get(ticker)
        if item is None:
            item = TickerSentimentAnalysis(
                ticker=ticker,
                sentiment="unknown",
                summary="O agente não produziu análise válida para este ticker.",
                data_gaps=["saída ausente após normalização por ticker"],
            )
        else:
            item.ticker = ticker
        normalized_items.append(item)

    return SentimentAnalysisOutput(per_ticker=normalized_items, overall=output.overall)


class SentimentAgent(Agent):
    role = "sentiment"
    system_prompt = SYSTEM_PROMPT

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        portfolio = state["portfolio"]
        sentiment_data = state.get("sentiment_data") or _extract_sentiment_data(
            state.get("market_data", {})
        )

        if not sentiment_data:
            output = SentimentAnalysisOutput(
                per_ticker=[
                    TickerSentimentAnalysis(
                        ticker=holding["ticker"],
                        sentiment="unknown",
                        summary="Não há notícias, eventos ou menções fornecidos para este ativo.",
                        data_gaps=["notícias, eventos e menções ausentes no input"],
                    )
                    for holding in portfolio
                ],
                overall="Sem dados textuais de sentimento no input.",
            )
            return {
                "analyses": [
                    {
                        "role": self.role,
                        "model": self.model,
                        "output": output.model_dump(),
                    }
                ]
            }

        compacted_sentiment = json.dumps(
            compact_sentiment_data(sentiment_data),
            ensure_ascii=False,
            indent=2,
        )
        prompt = (
            f"Carteira (ticker, peso, setor):\n"
            f"{json.dumps(portfolio, ensure_ascii=False, indent=2)}\n\n"
            f"Dados de sentimento disponíveis por ticker:\n"
            f"{compacted_sentiment}\n\n"
            "Produza a análise de sentimento conforme o schema."
        )

        output = self._invoke_structured(prompt, schema=SentimentAnalysisOutput)
        output = _normalize_sentiment_output(output, portfolio)
        return {
            "analyses": [
                {
                    "role": self.role,
                    "model": self.model,
                    "output": output.model_dump(),
                }
            ]
        }
