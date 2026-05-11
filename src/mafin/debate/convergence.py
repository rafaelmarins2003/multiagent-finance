from __future__ import annotations

import re
from typing import Any

TOKEN_RE = re.compile(r"[a-zA-ZÀ-ÿ0-9_]+")


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text) if len(token) > 2}


def _argument_text(argument: dict[str, Any]) -> str:
    output = argument.get("output", argument)
    parts = [
        output.get("thesis", ""),
        " ".join(output.get("key_points", [])),
        " ".join(output.get("challenged_assumptions", [])),
        " ".join(output.get("residual_uncertainties", [])),
    ]
    return " ".join(parts)


def _round_text(round_record: dict[str, Any]) -> str:
    return " ".join(
        [
            _argument_text(round_record.get("bull", {})),
            _argument_text(round_record.get("bear", {})),
        ]
    )


def convergence_score(previous_round: dict[str, Any], current_round: dict[str, Any]) -> float:
    """Return lexical semantic proxy in [0, 1] for two debate rounds.

    This is intentionally deterministic and cheap. It approximates semantic
    convergence for orchestration; evaluation can later replace it with
    embeddings if needed.
    """

    previous_tokens = _tokens(_round_text(previous_round))
    current_tokens = _tokens(_round_text(current_round))
    if not previous_tokens and not current_tokens:
        return 1.0
    if not previous_tokens or not current_tokens:
        return 0.0
    return round(len(previous_tokens & current_tokens) / len(previous_tokens | current_tokens), 4)
