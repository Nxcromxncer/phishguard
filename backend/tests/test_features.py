"""URL parsing and feature extraction: valid input, hostile input, determinism."""

from __future__ import annotations

import socket

import pytest

from app.features import (
    FEATURE_NAMES,
    URLValidationError,
    extract_features,
    features_to_vector,
    parse_url,
)


# --------------------------------------------------------------------------
# Valid URLs and normalization
# --------------------------------------------------------------------------
def test_parses_basic_url_with_public_suffix_awareness():
    p = parse_url("https://www.example.co.uk/a/b?x=1#frag")
    assert p.scheme == "https"
    assert p.host == "www.example.co.uk"
    assert p.subdomain_labels == ("www",)
    assert p.suffix == "co.uk"  # multi-label public suffix handled correctly
    assert p.registered_domain == "example.co.uk"
    assert p.path == "/a/b" and p.query == "x=1" and p.fragment == "frag"


def test_normalizes_case_scheme_and_trailing_dot():
    p = parse_url("  HTTP://EXAMPLE.COM./A  ")
    assert p.scheme == "http"
    assert p.host == "example.com"
    assert p.path == "/A"  # path case is significant and preserved


def test_unknown_tld_falls_back_to_last_label():
    p = parse_url("http://login.evil.example/")
    assert p.registered_domain == "evil.example"
    assert p.subdomain_labels == ("login",)


def test_userinfo_password_is_not_retained():
    p = parse_url("http://alice:s3cret@example.com/")
    assert p.has_userinfo
    assert p.userinfo_user == "alice"
    assert "s3cret" not in repr(p)  # password is dropped everywhere, incl. `normalized`
    assert "@" in p.normalized  # ...but the '@' itself is still visible to features


def test_userinfo_host_is_the_real_destination():
    p = parse_url("http://paypal.com@evil.example/login")
    assert p.host == "evil.example"
    assert p.userinfo_user == "paypal.com"


def test_ports():
    assert parse_url("http://example.com:80/").port == 80
    assert parse_url("http://example.com:8443/").port == 8443
    assert parse_url("http://example.com/").port is None


# --------------------------------------------------------------------------
# IP hosts
# --------------------------------------------------------------------------
def test_ipv4_host():
    p = parse_url("http://192.168.1.1/login")
    assert p.is_ip and not p.is_obfuscated_ip


def test_ipv6_host_with_port():
    p = parse_url("http://[::1]:8080/")
    assert p.is_ip and p.host == "::1" and p.port == 8080
    assert "[::1]:8080" in p.normalized


@pytest.mark.parametrize("url", ["http://2130706433/", "http://0x7f.0.0.1/", "http://0x7f000001/"])
def test_obfuscated_ip_hosts(url):
    p = parse_url(url)
    assert p.is_obfuscated_ip and not p.is_ip


# --------------------------------------------------------------------------
# Internationalized domains
# --------------------------------------------------------------------------
def test_idn_converted_to_punycode_and_back():
    p = parse_url("http://bücher.de/")
    assert p.host == "xn--bcher-kva.de"
    assert p.host_unicode == "bücher.de"
    assert p.has_punycode and p.has_non_ascii_host
    assert p.mixed_script_labels == ()  # single-script: not a homograph


def test_unicode_and_punycode_inputs_yield_identical_features():
    a = extract_features(parse_url("http://bücher.de/x"))
    b = extract_features(parse_url("http://xn--bcher-kva.de/x"))
    assert a["url_length"] == b["url_length"]
    assert features_to_vector(a)[:5] == features_to_vector(b)[:5]


# Regression: punycode adds hyphens/digits of its own ("xn--bcher-kva" has 3 hyphens)
# which made every internationalized domain look hyphen-stuffed.
def test_punycode_does_not_inflate_hyphen_or_digit_features():
    f = extract_features(parse_url("http://xn--bcher-kva.de/"))
    assert f["num_hyphens_host"] == 0.0
    assert f["num_digits_host"] == 0.0
    assert f["has_punycode"] == 1.0


def test_mixed_script_homograph_detected():
    # U+0430 is CYRILLIC SMALL LETTER A, visually identical to Latin 'a'.
    p = parse_url("http://аpple.com/")
    assert p.mixed_script_labels == ("аpple",)
    assert p.host.startswith("xn--")  # analysts see punycode, not the lookalike


# --------------------------------------------------------------------------
# Rejected input
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("url", "code"),
    [
        ("", "empty_url"),
        ("   ", "empty_url"),
        ("example.com", "missing_scheme"),
        ("ftp://example.com/", "unsupported_scheme"),
        ("javascript:alert(1)", "unsupported_scheme"),
        ("data:text/html,<script>1</script>", "unsupported_scheme"),
        ("file:///etc/passwd", "unsupported_scheme"),
        ("http://", "missing_host"),
        ("http:///path", "missing_host"),
        ("http://exa mple.com/", "invalid_character"),
        ("http://example.com/\x00", "invalid_character"),
        ("http://example.com/‮", "invalid_character"),  # bidi override
        ("http://example.com\\@evil.com/", "backslash"),
        ("http://[::1/", "malformed_url"),
        ("http://example.com:99999/", "invalid_port"),
        ("http://example.com:abc/", "invalid_port"),
        ("http://example.com:0/", "invalid_port"),
        ("http://-bad-.com/", "invalid_host"),
        ("http://a..b.com/", "invalid_host"),
        ("http://exa%6Dple.com/", "invalid_host"),
        ("http://" + "a" * 64 + ".com/", "invalid_host"),
    ],
)
def test_rejects_malformed_or_unsupported(url, code):
    with pytest.raises(URLValidationError) as exc_info:
        parse_url(url)
    assert exc_info.value.code == code


def test_rejects_non_string():
    with pytest.raises(URLValidationError) as exc_info:
        parse_url(123)  # type: ignore[arg-type]
    assert exc_info.value.code == "not_a_string"


def test_length_limit_boundary():
    prefix = "http://a.co/"
    ok = prefix + "x" * (50 - len(prefix))
    assert len(ok) == 50
    parse_url(ok, max_length=50)
    with pytest.raises(URLValidationError) as exc_info:
        parse_url(ok + "x", max_length=50)
    assert exc_info.value.code == "url_too_long"


def test_error_messages_never_contain_the_full_url():
    secret = "ftp://very-secret-host.invalid/private/token=abc123"
    with pytest.raises(URLValidationError) as exc_info:
        parse_url(secret)
    assert "very-secret-host" not in exc_info.value.message
    assert "abc123" not in exc_info.value.message


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
def test_feature_names_are_unique_and_vector_matches_order():
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)
    features = extract_features(parse_url("https://www.example.com/a/b?x=1&y=2"))
    assert tuple(features) == FEATURE_NAMES
    assert features_to_vector(features) == [features[n] for n in FEATURE_NAMES]
    assert all(isinstance(v, float) for v in features.values())


def test_extraction_is_deterministic():
    p = parse_url("http://login-secure.example.com:8080/verify%20me?next=http://x.com")
    assert extract_features(p) == extract_features(p)


def test_expected_feature_values():
    f = extract_features(parse_url("http://user@a-b.example.com:8080/one/two?x=1&y=2&z=3"))
    assert f["has_userinfo"] == 1.0
    assert f["num_at_symbols"] == 1.0
    assert f["num_hyphens_host"] == 1.0
    assert f["has_nonstandard_port"] == 1.0
    assert f["is_https"] == 0.0
    assert f["path_depth"] == 2.0
    assert f["num_query_params"] == 3.0
    assert f["num_subdomains"] == 1.0


def test_https_is_a_feature_but_changes_nothing_else():
    http = extract_features(parse_url("http://example.com/"))
    https = extract_features(parse_url("https://example.com/"))
    diff = {k for k in FEATURE_NAMES if http[k] != https[k]}
    assert diff == {"is_https", "url_length"}  # one extra character, nothing semantic


def test_ip_host_features():
    f = extract_features(parse_url("http://8.8.8.8/"))
    assert f["host_is_ip"] == 1.0


# Regression: tldextract must never fetch the public suffix list over the network.
def test_feature_extraction_makes_no_network_connections(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("network access attempted during parsing/feature extraction")

    monkeypatch.setattr(socket.socket, "connect", boom)
    extract_features(parse_url("https://www.example.co.uk/path"))
