from langchain_ollama import ChatOllama

from mafin.config import DEFAULT_TEMPERATURE, OLLAMA_HOST


def make_chat(model: str, temperature: float = DEFAULT_TEMPERATURE) -> ChatOllama:
    return ChatOllama(
        model=model,
        temperature=temperature,
        base_url=OLLAMA_HOST,
    )
