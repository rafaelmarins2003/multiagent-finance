from abc import ABC, abstractmethod
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from mafin.config import DEFAULT_TEMPERATURE
from mafin.llm.ollama import make_chat


class Agent(ABC):
    role: str
    system_prompt: str

    def __init__(self, model: str, temperature: float = DEFAULT_TEMPERATURE):
        self.model = model
        self.llm = make_chat(model=model, temperature=temperature)

    def _invoke(self, user_prompt: str) -> str:
        msg = self.llm.invoke(
            [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        return msg.content if isinstance(msg.content, str) else str(msg.content)

    @abstractmethod
    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Read state, call LLM, return partial state update."""
