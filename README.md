# mafin — Multi-Agent Portfolio Diagnosis

Pesquisa para o artigo submetido ao ENIAC 2026. O sistema executa um pipeline
multiagente (LangGraph) que diagnostica carteiras da B3 alinhadas ao perfil do
investidor, com debate adversarial Bull/Bear opcional.

Estado atual: B4 (multiagente + debate) roda fim a fim. Os baselines B1, B2,
B3, B4-H e B4-R estão implementados, com tracing SQLite por execução, presets
de roteamento (`local`, `b4h`, `b4r`) e micro-benchmark de compatibilidade.
Provedores reais (`yfinance`, BCB, Brave Search) alimentam o pipeline de
snapshots congelados.

## Requisitos

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) para gestão do ambiente
- [Ollama](https://ollama.com/) local em `http://localhost:11434` **ou**
  Ollama Cloud (`https://ollama.com`) via `OLLAMA_API_KEY`
- Modelo padrão local: `granite4.1:8b`

## Setup

```bash
# 1. Instalar dependências
uv sync --group dev

# 2. (Local) Baixar o modelo default no Ollama
ollama pull granite4.1:8b

# 3. (Opcional) Variáveis em .env
cp .env.example .env  # se existir, ou criar manualmente
```

### Variáveis de ambiente

Todas opcionais. Convenções: `*_LLM` definem o modelo por papel; o sufixo
`:cloud` é removido automaticamente quando o host é `ollama.com`.

| Variável | Default | Uso |
|---|---|---|
| `DEFAULT_LLM` / `DEFAULT_MODEL` | `granite4.1:8b` | Modelo base para todos os papéis |
| `LOCAL_LLM` | `granite4.1:8b` | Modelo do preset `local` |
| `B4H_LLM` | `DEFAULT_LLM` | Modelo único usado pelo B4-H |
| `MODEL_ROUTE_PRESET` | `b4r` | Preset ativo: `local`, `b4h` ou `b4r` |
| `TECHNICAL_LLM`, `SENTIMENT_LLM`, `FUNDAMENTAL_LLM`, `MACRO_LLM`, `RISK_LLM`, `BULL_LLM`, `BEAR_LLM`, `MODERATOR_LLM` | `DEFAULT_LLM` | Rota heterogênea (B4-R) |
| `OLLAMA_API_KEY` | — | Quando definida, host muda para `https://ollama.com` |
| `OLLAMA_HOST` | `http://localhost:11434` | Override explícito do host |
| `DEBATE_MAX_ROUNDS` | `3` | Rodadas máximas Bull/Bear |
| `DEBATE_MIN_ROUNDS` | `2` | Rodadas mínimas antes de checar convergência |
| `DEBATE_CONVERGENCE_THRESHOLD` | `0.88` | Score lexical de parada |
| `TRACE_DB_PATH` | `data/traces/mafin.sqlite3` | SQLite onde runs/calls/resultados ficam |
| `BRAVE_SEARCH_API_KEY` | — | Necessária para coleta de notícias |

## Smoke test

Roda o grafo completo (especialistas → debate → moderador) com dados mockados
ou com um snapshot real congelado:

```bash
uv run python scripts/smoke.py
uv run python scripts/generate_snapshot.py --output data/snapshots/snap.json
uv run python scripts/smoke.py --snapshot data/snapshots/snap.json
```

## Testes automatizados

```bash
uv run pytest
```

A suite em `tests/` cobre schemas (`Portfolio`, `UserProfile`), compilação do
grafo, agregação de self-consistency, persistência no tracing SQLite e o
loader de workload. Não chama o Ollama. O `scripts/smoke.py` permanece como
teste de integração manual.

## Workload sintético

Gera carteiras estratificadas (perfis conservador/moderado/agressivo) para
avaliação reprodutível:

```bash
uv run python scripts/generate_workload.py --output data/workload.json
```

## Baselines experimentais

Cinco baselines compartilham o mesmo schema de saída (`PortfolioDiagnosisOutput`):

- **B1** — LLM única ingênua.
- **B2** — LLM única com raciocínio estruturado + self-consistency (`n=5`).
- **B3** — multiagente sequencial sem debate.
- **B4-H** — multiagente com debate, modelo homogêneo (`B4H_LLM`).
- **B4-R** — multiagente com debate, roteamento heterogêneo por papel.

Executar sobre um snapshot ou workload:

```bash
uv run python scripts/run_baselines.py --snapshot data/snapshots/snap.json
uv run python scripts/run_baselines.py --workload data/workload.json --baselines b1 b2 b3 b4h b4r
```

Resultados saem em `data/results/baselines_<timestamp>.json` e cada execução
é registrada em `TRACE_DB_PATH` (prompt, resposta bruta, resposta parseada,
latência, modelos, hash de input).

## Micro-benchmark de modelos

Valida compatibilidade de saída estruturada Pydantic, latência e parsing
antes de fixar a rota oficial:

```bash
uv run python scripts/benchmark_models.py --route-preset b4r --output data/benchmarks/b4r.json
```

## Layout

```
src/mafin/
  agents/        # Technical, Sentiment, Fundamental, Macro, Risk, Bull, Bear, Moderator
  baselines/     # B1/B2 (LLM única) + runner unificado para B1–B4R
  data/          # Portfolio, snapshot, workload, providers (yfinance, BCB, Brave)
  debate/        # Convergência lexical e orquestração de rodadas
  graph/         # Build do LangGraph, GraphState, instrumentação de métricas
  llm/           # Wrapper ChatOllama (local ou Ollama Cloud)
  tracing/       # SQLiteTraceStore + TraceContext por execução
scripts/         # smoke, generate_snapshot, generate_workload, run_baselines, benchmark_models
tests/           # Suite unitária (pytest, offline)
```
