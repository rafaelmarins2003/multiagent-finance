import json
from typing import Any

from mafin.agents.base import Agent
from mafin.data.compact import compact_fundamental_data
from mafin.schema import FundamentalAnalysisOutput, TickerFundamentalAnalysis

SYSTEM_PROMPT = """Você é um analista fundamentalista de ações brasileiras.

Seu papel: avaliar fundamentos por ativo apenas quando houver dados explícitos,
como receita, lucro, margens, dívida, valuation, dividendos ou indicadores similares.

Restrições:
- Não recomende compra ou venda.
- Não use conhecimento externo implícito sobre empresas.
- Se dados fundamentalistas não forem fornecidos, registre a lacuna em vez de inferir.
"""


def _normalize_fundamental_output(
    output: FundamentalAnalysisOutput,
    portfolio: list[dict[str, Any]],
) -> FundamentalAnalysisOutput:
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
            item = TickerFundamentalAnalysis(
                ticker=ticker,
                summary="O agente não produziu análise fundamentalista válida para este ticker.",
                data_gaps=["saída ausente após normalização por ticker"],
            )
        else:
            item.ticker = ticker
        normalized_items.append(item)

    return FundamentalAnalysisOutput(per_ticker=normalized_items, overall=output.overall)


class FundamentalAgent(Agent):
    role = "fundamental"
    system_prompt = SYSTEM_PROMPT

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        portfolio = state["portfolio"]
        fundamentals = state.get("fundamental_data", {})
        compacted_fundamentals = json.dumps(
            compact_fundamental_data(fundamentals),
            ensure_ascii=False,
            indent=2,
        )

        prompt = (
            f"Carteira (ticker, peso, setor):\n"
            f"{json.dumps(portfolio, ensure_ascii=False, indent=2)}\n\n"
            f"Dados fundamentalistas disponíveis por ticker:\n"
            f"{compacted_fundamentals}\n\n"
            "Produza a análise fundamentalista conforme o schema. O objeto de topo deve conter "
            "`per_ticker` com um item para cada ticker da carteira e `overall` com o resumo geral."
        )

        output = self._invoke_structured(prompt, schema=FundamentalAnalysisOutput)
        output = _normalize_fundamental_output(output, portfolio)
        return {
            "analyses": [
                {
                    "role": self.role,
                    "model": self.model,
                    "output": output.model_dump(),
                }
            ]
        }
