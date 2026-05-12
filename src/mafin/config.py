import os
from dataclasses import asdict, dataclass

from dotenv import load_dotenv

load_dotenv()

OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
OLLAMA_API_KEY_CONFIGURED = bool(OLLAMA_API_KEY)
OLLAMA_HOST = os.getenv("OLLAMA_HOST") or (
    "https://ollama.com" if OLLAMA_API_KEY_CONFIGURED else "http://localhost:11434"
)


def is_direct_ollama_cloud() -> bool:
    return OLLAMA_HOST.rstrip("/").endswith("ollama.com")


def normalize_ollama_model(model: str) -> str:
    """Adapt local cloud tags to direct Ollama Cloud API model names."""
    if is_direct_ollama_cloud() and model.endswith(":cloud"):
        return model[: -len(":cloud")]
    return model


DEFAULT_LLM = normalize_ollama_model(
    os.getenv("DEFAULT_LLM") or os.getenv("DEFAULT_MODEL", "granite4.1:8b")
)
DEFAULT_MODEL = DEFAULT_LLM
LOCAL_LLM = normalize_ollama_model(os.getenv("LOCAL_LLM", "granite4.1:8b"))
B4H_LLM = normalize_ollama_model(os.getenv("B4H_LLM", DEFAULT_LLM))
MODEL_ROUTE_PRESET = os.getenv("MODEL_ROUTE_PRESET", "b4r")

BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY")
TRACE_DB_PATH = os.getenv("TRACE_DB_PATH", "data/traces/mafin.sqlite3")
STRUCTURED_OUTPUT_MODE = os.getenv("STRUCTURED_OUTPUT_MODE", "manual").strip().lower()
STRUCTURED_OUTPUT_MAX_RETRIES = max(1, int(os.getenv("STRUCTURED_OUTPUT_MAX_RETRIES", "2")))
STRUCTURED_OUTPUT_RETRY_BASE_SECONDS = max(
    0.0,
    float(os.getenv("STRUCTURED_OUTPUT_RETRY_BASE_SECONDS", "8")),
)
STRUCTURED_OUTPUT_RETRY_MAX_SECONDS = max(
    STRUCTURED_OUTPUT_RETRY_BASE_SECONDS,
    float(os.getenv("STRUCTURED_OUTPUT_RETRY_MAX_SECONDS", "90")),
)
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "240"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "2048"))
LLM_MAX_CONCURRENCY = max(1, int(os.getenv("LLM_MAX_CONCURRENCY", "1")))

_OLLAMA_REASONING_ENV = os.getenv("OLLAMA_REASONING", "false").strip().lower()
if _OLLAMA_REASONING_ENV in {"", "none", "null", "auto"}:
    OLLAMA_REASONING: bool | str | None = None
elif _OLLAMA_REASONING_ENV in {"1", "true", "yes", "on"}:
    OLLAMA_REASONING = True
elif _OLLAMA_REASONING_ENV in {"0", "false", "no", "off"}:
    OLLAMA_REASONING = False
else:
    OLLAMA_REASONING = _OLLAMA_REASONING_ENV


@dataclass(frozen=True)
class ModelRoute:
    """Roteamento de modelos por papel.

    `local` é usado para validação estrutural barata. `b4h` força um modelo
    homogêneo em todos os papéis. `b4r` lê os modelos por papel do ambiente.
    """

    technical: str = DEFAULT_LLM
    sentiment: str = DEFAULT_LLM
    fundamental: str = DEFAULT_LLM
    macro: str = DEFAULT_LLM
    risk: str = DEFAULT_LLM
    bull: str = DEFAULT_LLM
    bear: str = DEFAULT_LLM
    moderator: str = DEFAULT_LLM
    preset: str = "custom"

    @classmethod
    def homogeneous(cls, model: str, *, preset: str) -> "ModelRoute":
        return cls(
            technical=model,
            sentiment=model,
            fundamental=model,
            macro=model,
            risk=model,
            bull=model,
            bear=model,
            moderator=model,
            preset=preset,
        )

    @classmethod
    def from_env(cls) -> "ModelRoute":
        return cls(
            technical=normalize_ollama_model(os.getenv("TECHNICAL_LLM", DEFAULT_LLM)),
            sentiment=normalize_ollama_model(os.getenv("SENTIMENT_LLM", DEFAULT_LLM)),
            fundamental=normalize_ollama_model(os.getenv("FUNDAMENTAL_LLM", DEFAULT_LLM)),
            macro=normalize_ollama_model(os.getenv("MACRO_LLM", DEFAULT_LLM)),
            risk=normalize_ollama_model(os.getenv("RISK_LLM", DEFAULT_LLM)),
            bull=normalize_ollama_model(os.getenv("BULL_LLM", DEFAULT_LLM)),
            bear=normalize_ollama_model(os.getenv("BEAR_LLM", DEFAULT_LLM)),
            moderator=normalize_ollama_model(os.getenv("MODERATOR_LLM", DEFAULT_LLM)),
            preset="b4r",
        )

    def as_dict(self) -> dict[str, str]:
        return asdict(self)

    def unique_models(self) -> list[str]:
        models = [
            self.technical,
            self.sentiment,
            self.fundamental,
            self.macro,
            self.risk,
            self.bull,
            self.bear,
            self.moderator,
        ]
        return sorted(set(models))


def get_model_route(preset: str | None = None) -> ModelRoute:
    selected = (preset or MODEL_ROUTE_PRESET or "b4r").strip().lower()
    if selected == "env":
        selected = (MODEL_ROUTE_PRESET or "b4r").strip().lower()
        if selected == "env":
            selected = "b4r"

    if selected in {"role", "roles", "b4r", "heterogeneous"}:
        return ModelRoute.from_env()
    if selected in {"b4h", "homogeneous"}:
        return ModelRoute.homogeneous(B4H_LLM, preset="b4h")
    if selected == "local":
        return ModelRoute.homogeneous(LOCAL_LLM, preset="local")
    raise ValueError(f"Unknown model route preset: {preset!r}")


DEFAULT_TEMPERATURE = 0.2
ROUTES = get_model_route()


@dataclass(frozen=True)
class DebateConfig:
    max_rounds: int = int(os.getenv("DEBATE_MAX_ROUNDS", "3"))
    min_rounds: int = int(os.getenv("DEBATE_MIN_ROUNDS", "2"))
    convergence_threshold: float = float(os.getenv("DEBATE_CONVERGENCE_THRESHOLD", "0.88"))


DEBATE = DebateConfig()
