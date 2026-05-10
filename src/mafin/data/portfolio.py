from typing import Literal

from pydantic import BaseModel, Field, field_validator

RiskTolerance = Literal["conservative", "moderate", "aggressive"]
Horizon = Literal["short", "medium", "long"]


class Holding(BaseModel):
    ticker: str
    weight: float = Field(ge=0.0, le=1.0)
    sector: str | None = None


class Portfolio(BaseModel):
    holdings: list[Holding]

    @field_validator("holdings")
    @classmethod
    def weights_sum_to_one(cls, v: list[Holding]) -> list[Holding]:
        total = sum(h.weight for h in v)
        if not 0.99 <= total <= 1.01:
            raise ValueError(f"weights must sum to 1.0, got {total:.4f}")
        return v


class UserProfile(BaseModel):
    risk_tolerance: RiskTolerance
    horizon: Horizon
    objective: str
