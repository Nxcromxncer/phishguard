"""Explainable rule engine.

Every rule inspects a ``ParsedURL`` and either returns a ``RuleFinding`` (with a
severity, a human-readable explanation, and the evidence that triggered it) or
``None``. Rules never touch the network.

Severity and score definitions
------------------------------
============  =====  ==============================================================
Severity      Score  Meaning
============  =====  ==============================================================
weak          10     Common in legitimate URLs too. Never sufficient on its own.
moderate      25     Unusual and associated with phishing, but has legitimate uses.
strong        40     Rarely legitimate in a link a person would be asked to click.
============  =====  ==============================================================

Combining scores
----------------
``total = min(100, strong_and_moderate_sum + min(weak_sum, WEAK_SCORE_CAP))``

The weak-indicator cap (20) is deliberately BELOW the default suspicious
threshold (30), so any number of weak indicators can never, alone, push a URL to
"suspicious". Legitimate sites with long URLs, many subdomains, or the word
"account" in the path are not penalized into a warning.

All scores and thresholds are provisional until evaluated on real data (Phase 5).

Not covered (documented limitations)
------------------------------------
* Whole-script confusables (e.g. an all-Cyrillic lookalike of a Latin brand) are
  only flagged as a weak "internationalized domain" signal.
* The brand list is a small illustrative sample, not a maintained feed.
* Rules see only URL text, never page content.
"""

from __future__ import annotations

import ipaddress
from typing import Callable

from .features import (
    STANDARD_PORTS,
    ParsedURL,
    count_percent_encoded,
    has_embedded_url,
    path_has_executable_extension,
    sensitive_keywords_found,
)
from .schemas import RuleFinding, RuleResult, Severity

SEVERITY_SCORES: dict[Severity, int] = {
    Severity.weak: 10,
    Severity.moderate: 25,
    Severity.strong: 40,
}
WEAK_SCORE_CAP = 20
MAX_SCORE = 100

# Thresholds used by individual rules (documented here, not buried in code).
EXCESSIVE_SUBDOMAINS = 4
HYPHEN_HEAVY_HOST = 3
LONG_URL_LENGTH = 120
MIN_DISTINCT_KEYWORDS = 2
MIN_PERCENT_ENCODED = 5
HIGH_DIGIT_RATIO = 0.4

# Illustrative sample only. Maps a brand token to the registered domains that
# legitimately belong to it. Extend/replace with a maintained list in production.
BRAND_DOMAINS: dict[str, frozenset[str]] = {
    "paypal": frozenset({"paypal.com", "paypal.me"}),
    "apple": frozenset({"apple.com", "icloud.com"}),
    "microsoft": frozenset({"microsoft.com", "live.com", "office.com"}),
    "google": frozenset({"google.com", "gmail.com", "youtube.com", "google.co.uk", "google.co.in"}),
    "amazon": frozenset(
        {
            "amazon.com", "amazon.co.uk", "amazon.de", "amazon.fr", "amazon.it", "amazon.es",
            "amazon.in", "amazon.ca", "amazon.co.jp", "amazon.com.au", "amazon.com.br",
            "amazon.com.mx",
        }
    ),
    "netflix": frozenset({"netflix.com"}),
    "facebook": frozenset({"facebook.com", "fb.com"}),
    "instagram": frozenset({"instagram.com"}),
    "linkedin": frozenset({"linkedin.com"}),
    "dropbox": frozenset({"dropbox.com"}),
    "github": frozenset({"github.com", "github.io"}),
    "docusign": frozenset({"docusign.com", "docusign.net"}),
}


def _clip(text: str, limit: int = 120) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


def _finding(
    rule_id: str, severity: Severity, title: str, explanation: str, evidence: list[str]
) -> RuleFinding:
    return RuleFinding(
        rule_id=rule_id,
        severity=severity,
        score=SEVERITY_SCORES[severity],
        title=title,
        explanation=explanation,
        evidence=[_clip(item) for item in evidence],
    )


# --------------------------------------------------------------------------
# Individual rules
# --------------------------------------------------------------------------
def rule_ip_host(p: ParsedURL) -> RuleFinding | None:
    if p.is_obfuscated_ip:
        return _finding(
            "ip_host_obfuscated", Severity.strong, "Obfuscated numeric IP address as host",
            "The host is written as a decimal, hexadecimal, or octal number that browsers "
            "convert to an IP address. This notation is used to hide the destination.",
            [f"host: {p.host}"],
        )
    if p.is_ip:
        if not ipaddress.ip_address(p.host).is_global:
            return _finding(
                "ip_host_private", Severity.weak, "Private or local IP address as host",
                "The host is a private/loopback address. This is normal for development and "
                "home devices, but unexpected in a link sent to you.",
                [f"host: {p.host}"],
            )
        return _finding(
            "ip_host", Severity.strong, "Public IP address instead of a domain name",
            "Legitimate services almost always use domain names. A raw IP address in a "
            "link avoids domain reputation and is common in phishing.",
            [f"host: {p.host}"],
        )
    return None


def rule_userinfo(p: ParsedURL) -> RuleFinding | None:
    if not p.has_userinfo:
        return None
    looks_like_domain = "." in p.userinfo_user or "://" in p.userinfo_user
    severity = Severity.strong if looks_like_domain else Severity.moderate
    return _finding(
        "userinfo_in_url", severity, "Credentials-style prefix before '@' in the URL",
        "Everything before the '@' is ignored when connecting; the real destination is the "
        "host after it. Attackers put a trusted-looking name first to mislead readers.",
        [f"text before '@': {p.userinfo_user}", f"actual host: {p.host}"],
    )


def rule_mixed_script(p: ParsedURL) -> RuleFinding | None:
    if not p.mixed_script_labels:
        return None
    return _finding(
        "mixed_script_host", Severity.strong, "Domain mixes visually confusable alphabets",
        "A domain label combines letters from different scripts (for example Latin and "
        "Cyrillic). Lookalike characters can make a fake domain appear identical to a real one.",
        [f"label: {label}" for label in p.mixed_script_labels] + [f"ASCII form: {p.host}"],
    )


def rule_idn(p: ParsedURL) -> RuleFinding | None:
    if p.mixed_script_labels or not (p.has_punycode or p.has_non_ascii_host):
        return None
    return _finding(
        "idn_host", Severity.weak, "Internationalized (non-ASCII) domain name",
        "The domain uses non-ASCII characters. This is legitimate for many languages, but "
        "it is also how homograph lookalikes are built. Compare the ASCII form carefully.",
        [f"unicode: {p.host_unicode}", f"ASCII: {p.host}"],
    )


def rule_brand_impersonation(p: ParsedURL) -> RuleFinding | None:
    if p.is_ip or p.is_obfuscated_ip:
        return None
    tokens = set(p.host.replace("_", ".").replace("-", ".").split("."))
    hits = [
        brand
        for brand, legit in BRAND_DOMAINS.items()
        if brand in tokens and p.registered_domain not in legit
    ]
    if not hits:
        return None
    return _finding(
        "brand_in_unrelated_domain", Severity.moderate,
        "Well-known brand name appears in a domain the brand does not own",
        "The host contains a popular brand name, but the registered domain (the part you "
        "actually pay for) belongs to someone else. Only the registered domain shows who "
        "controls a site.",
        [f"brand token: {hits[0]}", f"registered domain: {p.registered_domain}", f"host: {p.host}"],
    )


def rule_excessive_subdomains(p: ParsedURL) -> RuleFinding | None:
    if len(p.subdomain_labels) < EXCESSIVE_SUBDOMAINS:
        return None
    return _finding(
        "excessive_subdomains", Severity.weak, "Unusually many subdomain levels",
        "Long subdomain chains push the real registered domain out of view, especially on "
        "small screens. Many legitimate sites also use deep subdomains.",
        [f"{len(p.subdomain_labels)} subdomain labels", f"registered domain: {p.registered_domain}"],
    )


def rule_hyphen_heavy_host(p: ParsedURL) -> RuleFinding | None:
    # Count on the decoded host: punycode adds its own hyphens ("xn--", "-kva")
    # that would otherwise make every internationalized domain look hyphen-heavy.
    count = p.host_unicode.count("-")
    if p.is_ip or count < HYPHEN_HEAVY_HOST:
        return None
    return _finding(
        "hyphen_heavy_host", Severity.weak, "Many hyphens in the host name",
        "Hyphen-stuffed names (e.g. 'secure-login-update-...') are a common way to fake "
        "official-looking domains, but some legitimate hosts use hyphens too.",
        [f"{count} hyphens in {p.host_unicode}"],
    )


def rule_digit_heavy_host(p: ParsedURL) -> RuleFinding | None:
    if p.is_ip or p.is_obfuscated_ip or not p.host:
        return None
    readable = p.host_unicode  # decoded, so punycode digits do not inflate the ratio
    ratio = sum(ch.isdigit() for ch in readable) / len(readable)
    if ratio < HIGH_DIGIT_RATIO:
        return None
    return _finding(
        "digit_heavy_host", Severity.weak, "Host name is mostly digits",
        "Auto-generated host names often contain many digits. Weak evidence by itself.",
        [f"digit ratio {ratio:.2f} in {readable}"],
    )


def rule_long_url(p: ParsedURL) -> RuleFinding | None:
    if len(p.normalized) < LONG_URL_LENGTH:
        return None
    return _finding(
        "long_url", Severity.weak, "Very long URL",
        "Long URLs can hide the important part of the address, but tracking parameters make "
        "many legitimate URLs long as well.",
        [f"length: {len(p.normalized)} characters"],
    )


def rule_nonstandard_port(p: ParsedURL) -> RuleFinding | None:
    if p.port is None or p.port in STANDARD_PORTS:
        return None
    return _finding(
        "nonstandard_port", Severity.weak, "Non-standard port",
        "The URL connects to an unusual port. Common for development servers and some "
        "services, uncommon for public login pages.",
        [f"port: {p.port}"],
    )


def rule_sensitive_keywords(p: ParsedURL) -> RuleFinding | None:
    found = sensitive_keywords_found(p)
    if len(found) < MIN_DISTINCT_KEYWORDS:
        return None
    return _finding(
        "sensitive_keywords", Severity.weak, "Several account-related keywords",
        f"At least {MIN_DISTINCT_KEYWORDS} distinct words typical of credential-theft pages "
        "appear. A single keyword is ignored because legitimate sites use them constantly.",
        [f"keywords: {', '.join(found)}"],
    )


def rule_http_with_keywords(p: ParsedURL) -> RuleFinding | None:
    """Plain HTTP combined with account keywords. Note: HTTPS is NOT rewarded."""
    if p.scheme != "http" or not sensitive_keywords_found(p):
        return None
    return _finding(
        "http_with_sensitive_keywords", Severity.weak, "Unencrypted HTTP on an account-style URL",
        "The page is not served over HTTPS yet looks like a login or account page. "
        "(The reverse is not reassuring: most phishing sites now use HTTPS.)",
        [f"scheme: {p.scheme}", f"keywords: {', '.join(sensitive_keywords_found(p))}"],
    )


def rule_percent_encoding(p: ParsedURL) -> RuleFinding | None:
    count = count_percent_encoded(p.normalized)
    if count < MIN_PERCENT_ENCODED:
        return None
    return _finding(
        "heavy_percent_encoding", Severity.weak, "Heavy percent-encoding",
        "Many %XX sequences can obscure the real content of a URL from a human reader.",
        [f"{count} percent-encoded sequences"],
    )


def rule_embedded_url(p: ParsedURL) -> RuleFinding | None:
    if not has_embedded_url(p):
        return None
    return _finding(
        "embedded_url", Severity.weak, "Another URL embedded in the path or query",
        "The link contains a second URL, which can indicate a redirect. Redirects are also "
        "used by many legitimate sites for sign-in and tracking.",
        [f"query/path: {_clip(p.path + '?' + p.query, 80)}"],
    )


def rule_executable_download(p: ParsedURL) -> RuleFinding | None:
    if not path_has_executable_extension(p):
        return None
    return _finding(
        "executable_download", Severity.moderate, "Link points to an executable or script file",
        "The path ends in a file type that can run code on your computer. Do not open files "
        "from links you did not expect.",
        [f"path: {p.path.rsplit('/', 1)[-1]}"],
    )


RULES: tuple[Callable[[ParsedURL], RuleFinding | None], ...] = (
    rule_ip_host,
    rule_userinfo,
    rule_mixed_script,
    rule_idn,
    rule_brand_impersonation,
    rule_executable_download,
    rule_excessive_subdomains,
    rule_hyphen_heavy_host,
    rule_digit_heavy_host,
    rule_long_url,
    rule_nonstandard_port,
    rule_sensitive_keywords,
    rule_http_with_keywords,
    rule_percent_encoding,
    rule_embedded_url,
)


def evaluate_rules(parsed: ParsedURL) -> RuleResult:
    """Run every rule and combine the scores (see module docstring)."""
    findings = [f for rule in RULES if (f := rule(parsed)) is not None]
    findings.sort(key=lambda f: (-f.score, f.rule_id))

    weak_sum = sum(f.score for f in findings if f.severity is Severity.weak)
    other_sum = sum(f.score for f in findings if f.severity is not Severity.weak)
    score = min(MAX_SCORE, other_sum + min(weak_sum, WEAK_SCORE_CAP))
    return RuleResult(score=score, findings=findings)
