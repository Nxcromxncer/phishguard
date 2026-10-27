"""Rule engine: scoring, severity separation, and false-positive resistance."""

from __future__ import annotations

import pytest

from app.features import parse_url
from app.rules import MAX_SCORE, SEVERITY_SCORES, WEAK_SCORE_CAP, evaluate_rules
from app.schemas import Severity


def evaluate(url: str):
    return evaluate_rules(parse_url(url))


def ids(result) -> set[str]:
    return {f.rule_id for f in result.findings}


# --------------------------------------------------------------------------
# Benign URLs must stay quiet
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "url",
    [
        "https://www.example.com/",
        "https://en.wikipedia.org/wiki/Phishing",
        "https://github.com/anthropics",
        "https://www.amazon.de/dp/B000000000",
        "https://support.apple.com/en-us/HT201222",
        "https://example.com/login",  # one keyword alone proves nothing
        "https://example.com/account",
    ],
)
def test_benign_urls_produce_no_findings(url):
    result = evaluate(url)
    assert result.findings == [], [f.rule_id for f in result.findings]
    assert result.score == 0


def test_https_is_not_rewarded_or_penalized():
    assert evaluate("http://example.com/").score == evaluate("https://example.com/").score == 0


# --------------------------------------------------------------------------
# Weak indicators never add up to a warning on their own
# --------------------------------------------------------------------------
def test_many_weak_indicators_are_capped_below_suspicious():
    url = "http://a-b-c-d.e.f.g.h.example.com:8081/login/verify/account?x=%41%42%43%44%45"
    result = evaluate(url)
    assert len(result.findings) >= 5
    assert all(f.severity is Severity.weak for f in result.findings)
    assert result.score == WEAK_SCORE_CAP
    assert WEAK_SCORE_CAP < 30  # below the default suspicious threshold


def test_severity_scores_are_ordered():
    assert SEVERITY_SCORES[Severity.weak] < SEVERITY_SCORES[Severity.moderate] < SEVERITY_SCORES[Severity.strong]


# --------------------------------------------------------------------------
# Individual rules
# --------------------------------------------------------------------------
def test_public_ip_is_strong():
    result = evaluate("http://8.8.8.8/")
    assert ids(result) == {"ip_host"}
    assert result.findings[0].severity is Severity.strong
    assert result.score == 40


def test_private_ip_is_only_weak():
    result = evaluate("http://192.168.0.10/")
    assert ids(result) == {"ip_host_private"}
    assert result.findings[0].severity is Severity.weak


def test_obfuscated_ip_is_strong():
    result = evaluate("http://2130706433/")
    assert "ip_host_obfuscated" in ids(result)
    assert result.score >= 40


def test_userinfo_that_looks_like_a_domain_is_strong():
    result = evaluate("http://paypal.com@evil.example/signin")
    finding = next(f for f in result.findings if f.rule_id == "userinfo_in_url")
    assert finding.severity is Severity.strong
    assert any("evil.example" in e for e in finding.evidence)


def test_plain_userinfo_is_moderate():
    finding = next(f for f in evaluate("http://bob@example.com/").findings if f.rule_id == "userinfo_in_url")
    assert finding.severity is Severity.moderate


@pytest.mark.parametrize(
    "url",
    [
        "https://paypal.com.secure-check.net/",
        "https://paypal-support.example.org/",
        "https://amazon.evil.example/",
        "https://login.microsoft.account-verify.com/",
    ],
)
def test_brand_in_unrelated_domain_flagged(url):
    assert "brand_in_unrelated_domain" in ids(evaluate(url))


@pytest.mark.parametrize(
    "url",
    [
        "https://www.paypal.com/signin",
        "https://paypal.me/someone",
        "https://pineapple.com/",  # 'apple' is a substring, not a label token
        "https://www.amazon.co.uk/",
        "https://login.microsoftonline.com/",  # different, real domain - not in sample list
    ],
)
def test_legitimate_brand_domains_not_flagged(url):
    assert "brand_in_unrelated_domain" not in ids(evaluate(url))


def test_homograph_mixed_script_is_strong_but_plain_idn_is_weak():
    homograph = evaluate("http://аpple.com/")
    assert "mixed_script_host" in ids(homograph)
    assert homograph.score >= 40

    plain_idn = evaluate("http://bücher.de/")
    assert ids(plain_idn) == {"idn_host"}
    assert plain_idn.findings[0].severity is Severity.weak


def test_executable_download_is_moderate():
    finding = evaluate("https://example.com/files/setup.exe").findings[0]
    assert finding.rule_id == "executable_download"
    assert finding.severity is Severity.moderate


def test_embedded_url_and_long_url_are_weak():
    result = evaluate("https://example.com/r?u=https://other.example.net/" + "a" * 120)
    assert {"embedded_url", "long_url"} <= ids(result)
    assert all(f.severity is Severity.weak for f in result.findings)


def test_http_with_keywords_flagged_but_https_equivalent_is_not():
    assert "http_with_sensitive_keywords" in ids(evaluate("http://example.com/login"))
    assert "http_with_sensitive_keywords" not in ids(evaluate("https://example.com/login"))


# --------------------------------------------------------------------------
# Result structure
# --------------------------------------------------------------------------
def test_every_finding_is_explained_and_has_evidence():
    result = evaluate("http://paypal.com@8.8.8.8:8080/login/verify/account.exe")
    assert result.findings
    for f in result.findings:
        assert f.title and f.explanation and f.evidence
        assert f.score == SEVERITY_SCORES[f.severity]


def test_findings_sorted_strongest_first_and_score_capped():
    result = evaluate("http://paypal.com@8.8.8.8:8080/login/verify/account.exe")
    scores = [f.score for f in result.findings]
    assert scores == sorted(scores, reverse=True)
    assert 0 <= result.score <= MAX_SCORE
