import pytest
from pydantic import ValidationError

from mafin.data.portfolio import Holding, Portfolio, UserProfile


def test_portfolio_accepts_weights_summing_to_one():
    portfolio = Portfolio(
        holdings=[
            Holding(ticker="PETR4", weight=0.5, sector="Energia"),
            Holding(ticker="ITUB4", weight=0.5, sector="Financeiro"),
        ]
    )
    assert len(portfolio.holdings) == 2


def test_portfolio_accepts_weights_within_tolerance():
    Portfolio(
        holdings=[
            Holding(ticker="A", weight=0.333),
            Holding(ticker="B", weight=0.333),
            Holding(ticker="C", weight=0.334),
        ]
    )


def test_portfolio_rejects_weights_not_summing_to_one():
    with pytest.raises(ValidationError, match="weights must sum to 1.0"):
        Portfolio(
            holdings=[
                Holding(ticker="PETR4", weight=0.4),
                Holding(ticker="ITUB4", weight=0.4),
            ]
        )


def test_holding_rejects_weight_above_one():
    with pytest.raises(ValidationError):
        Holding(ticker="PETR4", weight=1.5)


def test_holding_rejects_negative_weight():
    with pytest.raises(ValidationError):
        Holding(ticker="PETR4", weight=-0.1)


def test_holding_sector_optional():
    holding = Holding(ticker="PETR4", weight=0.3)
    assert holding.sector is None


def test_user_profile_valid_combination():
    profile = UserProfile(
        risk_tolerance="moderate",
        horizon="long",
        objective="valorização com tolerância média a drawdown",
    )
    assert profile.risk_tolerance == "moderate"
    assert profile.horizon == "long"


@pytest.mark.parametrize("risk", ["conservative", "moderate", "aggressive"])
def test_user_profile_accepts_each_risk_tolerance(risk):
    UserProfile(risk_tolerance=risk, horizon="long", objective="x")


@pytest.mark.parametrize("horizon", ["short", "medium", "long"])
def test_user_profile_accepts_each_horizon(horizon):
    UserProfile(risk_tolerance="moderate", horizon=horizon, objective="x")


def test_user_profile_rejects_invalid_risk():
    with pytest.raises(ValidationError):
        UserProfile(risk_tolerance="reckless", horizon="long", objective="x")


def test_user_profile_rejects_invalid_horizon():
    with pytest.raises(ValidationError):
        UserProfile(risk_tolerance="moderate", horizon="eternal", objective="x")
