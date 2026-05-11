from typing import Any

from pydantic import BaseModel, Field


class DataSourceRecord(BaseModel):
    name: str
    kind: str
    retrieved_at: str
    detail: str | None = None


class NewsItem(BaseModel):
    title: str
    url: str
    source: str | None = None
    domain: str | None = None
    bucket: str | None = None
    published_at: str | None = None
    description: str | None = None
    provider: str
    query: str


class PortfolioDataSnapshot(BaseModel):
    snapshot_id: str
    created_at: str
    as_of_date: str
    portfolio: list[dict[str, Any]]
    profile: dict[str, Any]
    market_data: dict[str, dict[str, Any]] = Field(default_factory=dict)
    fundamental_data: dict[str, dict[str, Any]] = Field(default_factory=dict)
    sentiment_data: dict[str, dict[str, Any]] = Field(default_factory=dict)
    macro_data: dict[str, Any] = Field(default_factory=dict)
    news_policy: dict[str, Any] = Field(default_factory=dict)
    sources: list[DataSourceRecord] = Field(default_factory=list)
    data_gaps: list[str] = Field(default_factory=list)

    def to_graph_state(self) -> dict[str, Any]:
        return {
            "portfolio": self.portfolio,
            "profile": self.profile,
            "market_data": self.market_data,
            "fundamental_data": self.fundamental_data,
            "sentiment_data": self.sentiment_data,
            "macro_data": self.macro_data,
            "analyses": [],
            "debate_rounds": [],
            "debate_status": {},
            "diagnosis": None,
        }
