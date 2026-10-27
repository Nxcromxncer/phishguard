"""HTTP API: contracts, validation, error shape, limits, privacy, no-network."""

from __future__ import annotations

import logging
import socket

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def scan(client, url):
    return client.post("/scan", json={"url": url})


# --------------------------------------------------------------------------
# Info and health
# --------------------------------------------------------------------------
def test_root_info(client):
    body = client.get("/").json()
    assert body["name"] == "PhishGuard" and "POST /scan" in body["endpoints"]


def test_health_reports_honest_component_status(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    by_name = {c["name"]: c["status"] for c in body["components"]}
    assert by_name == {"rules": "available", "ml": "not_configured", "reputation": "not_configured"}


def test_openapi_docs_available(client):
    assert client.get("/openapi.json").status_code == 200
    assert "/scan" in client.get("/openapi.json").json()["paths"]


# --------------------------------------------------------------------------
# /scan success paths
# --------------------------------------------------------------------------
def test_scan_clean_looking_url_is_unknown_not_safe(client):
    response = scan(client, "https://www.example.com/")
    assert response.status_code == 200
    body = response.json()
    assert body["verdict"] == "unknown"
    assert body["risk_level"] == "unknown"
    assert body["ml"]["status"] == "not_configured" and body["ml"]["prediction"] is None
    assert body["reputation"]["status"] == "not_configured"
    assert "not a guarantee" in body["disclaimer"]
    assert {c["name"] for c in body["components"]} == {"rules", "ml", "reputation"}


def test_scan_suspicious_url_returns_findings_and_evidence(client):
    body = scan(client, "http://8.8.8.8/").json()
    assert body["verdict"] == "suspicious"
    assert body["risk_level"] == "medium"
    assert body["rules"]["score"] == 40
    finding = body["rules"]["findings"][0]
    assert finding["rule_id"] == "ip_host" and finding["evidence"] and finding["explanation"]


def test_scan_displays_punycode_host_and_unicode_form(client):
    body = scan(client, "http://аpple.com/").json()
    assert body["host"].startswith("xn--")
    assert body["host_unicode"] == "аpple.com"
    assert body["verdict"] == "suspicious"  # mixed-script is a strong rule


def test_url_display_drops_path_query_and_userinfo(client):
    body = scan(client, "https://alice:pw@www.example.com/secret/path?token=abc#frag").json()
    shown = body["url_display"]
    assert shown == "https://www.example.com/..."
    for leaked in ("secret", "token", "abc", "alice", "pw"):
        assert leaked not in shown


# --------------------------------------------------------------------------
# Validation and error shape
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("url", "reason"),
    [("ftp://example.com/", "unsupported_scheme"), ("not a url", "invalid_character"), ("example.com", "missing_scheme")],
)
def test_invalid_url_is_422_with_consistent_error(client, url, reason):
    response = scan(client, url)
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "invalid_url"
    assert error["details"] == [{"reason": reason}]


def test_error_responses_never_echo_the_submitted_url(client):
    response = scan(client, "ftp://very-secret-host.invalid/path?token=SECRET123")
    assert "very-secret-host" not in response.text and "SECRET123" not in response.text


def test_url_over_configured_length_is_rejected(client):
    response = scan(client, "http://example.com/" + "a" * 3000)
    assert response.status_code == 422
    assert response.json()["error"]["details"] == [{"reason": "url_too_long"}]


@pytest.mark.parametrize(
    "payload",
    [{}, {"url": ""}, {"url": 123}, {"url": None}, {"url": "http://a.com", "extra": 1}],
)
def test_schema_violations_are_422_validation_error(client, payload):
    response = client.post("/scan", json=payload)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_non_json_body_is_422_and_does_not_echo_it(client):
    response = client.post("/scan", content="url=SECRET-FORM-VALUE", headers={"content-type": "application/json"})
    assert response.status_code == 422
    assert "SECRET-FORM-VALUE" not in response.text


def test_unknown_route_and_wrong_method_use_error_shape(client):
    missing = client.get("/nope")
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "not_found"
    wrong = client.get("/scan")
    assert wrong.status_code == 405 and wrong.json()["error"]["code"] == "method_not_allowed"


# --------------------------------------------------------------------------
# Size limits and rate limiting
# --------------------------------------------------------------------------
def test_oversized_body_is_413(client):
    response = client.post("/scan", json={"url": "http://example.com/" + "a" * 20_000})
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"


def test_oversized_chunked_body_without_content_length_is_413(client):
    def chunks():
        for _ in range(20):
            yield b"x" * 1024

    response = client.post("/scan", content=chunks(), headers={"content-type": "application/json"})
    assert response.status_code == 413


def test_rate_limit_returns_429_with_retry_after(tmp_path):
    with TestClient(create_app(Settings(rate_limit_per_minute=3, model_dir=str(tmp_path)))) as limited:
        codes = [scan(limited, "https://example.com/").status_code for _ in range(4)]
        assert codes == [200, 200, 200, 429]
        blocked = scan(limited, "https://example.com/")
        assert blocked.json()["error"]["code"] == "rate_limited"
        assert int(blocked.headers["retry-after"]) >= 1


# --------------------------------------------------------------------------
# Security and privacy properties
# --------------------------------------------------------------------------
def test_backend_never_opens_network_connections_to_analyze_a_url(client, monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("the backend tried to connect to a network host")

    monkeypatch.setattr(socket.socket, "connect", boom)
    assert scan(client, "http://definitely-not-real.invalid/login").status_code == 200


def test_logs_do_not_contain_the_submitted_url(client, caplog):
    caplog.set_level(logging.INFO, logger="phishguard")
    scan(client, "https://very-secret-host.example.com/path?token=SECRET123")
    scan(client, "ftp://another-secret-host.example.com/x")
    assert "scan ref=" in caplog.text  # something was logged...
    assert "secret-host" not in caplog.text and "SECRET123" not in caplog.text  # ...but not the URL


def test_component_failure_is_unavailable_never_clean(settings):
    class ExplodingModel:
        def predict(self, parsed):
            raise RuntimeError("model blew up")

        def info(self):
            raise RuntimeError

    with TestClient(create_app(settings, ml_service=ExplodingModel())) as c:  # type: ignore[arg-type]
        body = scan(c, "https://www.example.com/").json()
    assert body["ml"]["status"] == "unavailable"
    assert body["verdict"] != "no_known_threat_detected"
    assert "unavailable" in " ".join(body["explanation"]).lower()


def test_unexpected_error_returns_generic_500_without_details(settings, monkeypatch):
    def broken(_parsed):
        raise RuntimeError("internal detail SECRET-INTERNAL")

    monkeypatch.setattr("app.main.evaluate_rules", broken)
    with TestClient(create_app(settings), raise_server_exceptions=False) as c:
        response = scan(c, "https://www.example.com/")
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert "SECRET-INTERNAL" not in response.text


def test_no_cors_headers_by_default(client):
    response = client.get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in response.headers


def test_cors_only_for_explicitly_configured_origin(tmp_path):
    cfg = Settings(rate_limit_per_minute=1000, cors_origins=("http://localhost:3000",), model_dir=str(tmp_path))
    with TestClient(create_app(cfg)) as c:
        allowed = c.get("/health", headers={"Origin": "http://localhost:3000"})
        denied = c.get("/health", headers={"Origin": "https://evil.example"})
    assert allowed.headers.get("access-control-allow-origin") == "http://localhost:3000"
    assert "access-control-allow-origin" not in denied.headers
