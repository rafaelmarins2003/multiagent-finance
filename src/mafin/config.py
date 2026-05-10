from dataclasses import dataclass

# Tag Ollama do modelo local. Usuário pediu Granite 4 8B; tag mais próxima
# disponível no granite4 é `tiny-h` (~7B). Trocar aqui se publicarem variante 8B.
DEFAULT_MODEL = "granite4:tiny-h"


@dataclass(frozen=True)
class ModelRoute:
    """Roteamento de modelos por papel. V1 100% local com Granite 4.

    Quando ligarmos APIs proprietárias na fase de avaliação, papéis de síntese
    (moderator) e adversariais (bull/bear) migram para modelo de fronteira.
    """

    technical: str = DEFAULT_MODEL
    sentiment: str = DEFAULT_MODEL
    fundamental: str = DEFAULT_MODEL
    macro: str = DEFAULT_MODEL
    risk: str = DEFAULT_MODEL
    bull: str = DEFAULT_MODEL
    bear: str = DEFAULT_MODEL
    moderator: str = DEFAULT_MODEL


OLLAMA_HOST = "http://localhost:11434"
DEFAULT_TEMPERATURE = 0.2
ROUTES = ModelRoute()
