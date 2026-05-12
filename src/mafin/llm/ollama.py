from langchain_ollama import ChatOllama

from mafin.config import (
    DEFAULT_TEMPERATURE,
    OLLAMA_API_KEY,
    OLLAMA_HOST,
    OLLAMA_NUM_PREDICT,
    OLLAMA_REASONING,
    OLLAMA_TIMEOUT_SECONDS,
    normalize_ollama_model,
)


def _client_kwargs() -> dict:
    kwargs = {"timeout": OLLAMA_TIMEOUT_SECONDS}
    if not OLLAMA_API_KEY:
        return kwargs
    kwargs["headers"] = {"authorization": f"Bearer {OLLAMA_API_KEY}"}
    return kwargs


def make_chat(model: str, temperature: float = DEFAULT_TEMPERATURE) -> ChatOllama:
    return ChatOllama(
        model=normalize_ollama_model(model),
        temperature=temperature,
        reasoning=OLLAMA_REASONING,
        num_predict=OLLAMA_NUM_PREDICT,
        base_url=OLLAMA_HOST,
        client_kwargs=_client_kwargs(),
    )
