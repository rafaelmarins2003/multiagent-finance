from typing import Any, Literal

from pydantic import BaseModel, Field

SentimentLabel = Literal["positive", "neutral", "negative", "mixed", "unknown"]
RiskLevel = Literal["low", "moderate", "high", "unknown"]
DebateStance = Literal["bull", "bear"]
DiagnosisClassification = Literal[
    "estavel",
    "atencao",
    "risco_elevado",
    "desalinhada",
    "inconclusiva",
]


class TickerTechnicalAnalysis(BaseModel):
    ticker: str
    summary: str
    signals: list[str] = Field(default_factory=list)


class TechnicalAnalysisOutput(BaseModel):
    per_ticker: list[TickerTechnicalAnalysis]
    overall: str


class TickerSentimentAnalysis(BaseModel):
    ticker: str
    sentiment: SentimentLabel
    summary: str
    evidence: list[str] = Field(default_factory=list)
    data_gaps: list[str] = Field(default_factory=list)


class SentimentAnalysisOutput(BaseModel):
    per_ticker: list[TickerSentimentAnalysis]
    overall: str


class TickerFundamentalAnalysis(BaseModel):
    ticker: str
    summary: str
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    data_gaps: list[str] = Field(default_factory=list)


class FundamentalAnalysisOutput(BaseModel):
    per_ticker: list[TickerFundamentalAnalysis]
    overall: str


class MacroAnalysisOutput(BaseModel):
    summary: str
    relevant_factors: list[str] = Field(default_factory=list)
    portfolio_implications: list[str] = Field(default_factory=list)
    data_gaps: list[str] = Field(default_factory=list)


class RiskAnalysisOutput(BaseModel):
    overall_risk: RiskLevel
    concentration_summary: str
    profile_alignment: str
    risk_factors: list[str] = Field(default_factory=list)
    mitigants: list[str] = Field(default_factory=list)
    data_gaps: list[str] = Field(default_factory=list)


class DebateArgumentOutput(BaseModel):
    stance: DebateStance
    thesis: str
    key_points: list[str] = Field(default_factory=list)
    challenged_assumptions: list[str] = Field(default_factory=list)
    evidence_used: list[str] = Field(default_factory=list)
    residual_uncertainties: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class DebateStatusOutput(BaseModel):
    stopped_by: str
    rounds_completed: int
    convergence_score: float = Field(ge=0.0, le=1.0)
    converged: bool
    summary: str


class PortfolioDiagnosisOutput(BaseModel):
    classification: DiagnosisClassification
    justification: str
    positive_factors: list[str] = Field(default_factory=list)
    negative_factors: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    profile_alignment: str


class AnalysisRecord(BaseModel):
    role: str
    model: str
    output: dict[str, Any]
