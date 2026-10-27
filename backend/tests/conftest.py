"""Shared, deterministic fixtures. No network, no credentials, no real model."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.schemas import MLResult, ReputationResult, RuleResult


@pytest.fixture
def settings() -> Settings:
    # High rate limit so ordinary tests never trip it.
    return Settings(rate_limit_per_minute=1000)


@pytest.fixture
def client(settings: Settings):
    with TestClient(create_app(settings)) as test_client:
        yield test_client


# Helpers for building component results in decision tests -------------------
def rules_result(score: int = 0) -> RuleResult:
    return RuleResult(score=score, findings=[])


def ml_result(kind: str = "not_configured") -> MLResult:
    if kind == "phishing":
        return MLResult(status="predicted", prediction="phishing", score=0.9, detail="test")
    if kind == "legitimate":
        return MLResult(status="predicted", prediction="legitimate", score=0.1, detail="test")
    return MLResult(status=kind, detail="test")  # type: ignore[arg-type]


def rep_result(status: str = "not_configured") -> ReputationResult:
    provider = "test-provider" if status in {"malicious", "no_match"} else None
    return ReputationResult(status=status, provider=provider, detail="test")  # type: ignore[arg-type]
