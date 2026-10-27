"""Phase 1: the ML component is a placeholder and must never fabricate output.

Phase 3 will extend this file with artifact loading, feature-schema mismatch,
and real prediction tests.
"""

from __future__ import annotations

from app.features import parse_url
from app.model import ModelService


def test_missing_model_reports_not_configured():
    result = ModelService().predict(parse_url("https://example.com/"))
    assert result.status == "not_configured"


def test_missing_model_never_fabricates_a_prediction():
    result = ModelService().predict(parse_url("http://8.8.8.8/login"))
    assert result.prediction is None
    assert result.score is None
    assert result.model_version is None


def test_info_reports_not_configured():
    info = ModelService().info()
    assert info.name == "ml" and info.status == "not_configured"
