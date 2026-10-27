"""Phase 1: reputation is a placeholder; "not looked up" must never read as "clean".

Phase 4 will extend this file with mocked-provider tests for malicious, no_match,
timeout, rate-limit, and caching behavior.
"""

from __future__ import annotations

from app.features import parse_url
from app.reputation import ReputationService


def test_no_provider_reports_not_configured():
    result = ReputationService().check(parse_url("https://example.com/"))
    assert result.status == "not_configured"
    assert result.provider is None


def test_not_configured_is_never_reported_as_no_match():
    # Regression guard for the central rule: absence of a lookup is not a clean result.
    result = ReputationService().check(parse_url("https://example.com/"))
    assert result.status != "no_match"


def test_info_reports_not_configured():
    info = ReputationService().info()
    assert info.name == "reputation" and info.status == "not_configured"
