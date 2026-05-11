from langchain_ollama import ChatOllama

from mafin.config import (
    DEFAULT_TEMPERATURE,
    OLLAMA_API_KEY,
    OLLAMA_HOST,
    normalize_ollama_model,
)


def _client_kwargs() -> dict:
    if not OLLAMA_API_KEY:
        return {}
    return {"headers": {"authorization": f"Bearer {OLLAMA_API_KEY}"}}


def make_chat(model: str, temperature: float = DEFAULT_TEMPERATURE) -> ChatOllama:
    return ChatOllama(
        model=normalize_ollama_model(model),
        temperature=temperature,
        base_url=OLLAMA_HOST,
        client_kwargs=_client_kwargs(),
    )
