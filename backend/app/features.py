"""Safe URL parsing and deterministic feature extraction.

This module is the single source of truth for turning a URL string into
structured data. The rule engine (rules.py) and, from Phase 3, the ML model
(model.py + training/) all consume ``ParsedURL`` and ``extract_features`` so that
training and inference can never drift apart.

Security properties
-------------------
* NO network access. ``tldextract`` is configured to use its bundled public
  suffix snapshot only (no download, no cache directory).
* Strict input validation: only http/https, no control or invisible characters,
  no backslashes (browsers and parsers disagree about them - a classic
  parser-differential trick), bounded length, valid port, valid hostname.
* Internationalized domains are converted to ASCII punycode for analysis and
  display, and checked for mixed-script (homograph) labels.
* HTTPS is recorded as a feature but NEVER treated as proof of legitimacy.
"""

from __future__ import annotations

import ipaddress
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlsplit, urlunsplit

import tldextract

FEATURE_SCHEMA_VERSION = 1
"""Bump when features are added/removed/reordered; model.py checks this."""

DEFAULT_MAX_URL_LENGTH = 2048
ALLOWED_SCHEMES = frozenset({"http", "https"})
STANDARD_PORTS = frozenset({80, 443})

# A single keyword proves nothing (many legitimate URLs contain "account").
# The rule engine only reacts when SEVERAL distinct keywords co-occur.
SENSITIVE_KEYWORDS: tuple[str, ...] = (
    "login", "signin", "sign-in", "verify", "verification", "account", "secure",
    "update", "confirm", "password", "banking", "wallet", "billing", "suspend",
)

EXECUTABLE_EXTENSIONS = frozenset(
    {"exe", "scr", "bat", "cmd", "msi", "apk", "jar", "vbs", "ps1", "dll"}
)

# Scripts whose letters are visually confusable with each other.
CONFUSABLE_SCRIPTS = frozenset({"LATIN", "CYRILLIC", "GREEK", "ARMENIAN"})

# Offline extractor: suffix_list_urls=() forbids fetching; cache_dir=None forbids
# writing a cache. It falls back to the snapshot bundled with the package.
_TLD_EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)

_LABEL_RE = re.compile(r"^[a-z0-9_](?:[a-z0-9_-]{0,61}[a-z0-9_])?$")
_OBFUSCATED_IP_RE = re.compile(r"^(?:0x[0-9a-f]+|\d+)(?:\.(?:0x[0-9a-f]+|\d+)){0,3}$")
_EMBEDDED_URL_RE = re.compile(r"https?(?::|%3a)(?:/|%2f)(?:/|%2f)")
_PERCENT_RE = re.compile(r"%[0-9a-fA-F]{2}")


class URLValidationError(ValueError):
    """Raised for malformed or unsupported URLs.

    ``code`` is a stable machine-readable reason; ``message`` is safe to show to
    users. Neither ever contains the full submitted URL.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class ParsedURL:
    """A validated, normalized URL. Immutable so it can be shared safely."""

    normalized: str
    scheme: str
    host: str  # ASCII / punycode, lowercase, no trailing dot
    host_unicode: str  # Unicode form (equals host when not internationalized)
    port: int | None
    has_userinfo: bool
    userinfo_user: str  # user part only (password intentionally dropped), clipped
    path: str
    query: str
    fragment: str
    subdomain_labels: tuple[str, ...]
    domain: str
    suffix: str
    registered_domain: str
    is_ip: bool
    is_obfuscated_ip: bool
    has_punycode: bool
    has_non_ascii_host: bool
    mixed_script_labels: tuple[str, ...]


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------
def _clip(text: str, limit: int = 20) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


def _label_scripts(label: str) -> set[str]:
    """Return the set of Unicode script names (first word of char name) in a label."""
    scripts: set[str] = set()
    for ch in label:
        if not ch.isalpha():
            continue
        try:
            scripts.add(unicodedata.name(ch).split(" ", 1)[0])
        except ValueError:  # unnamed code point
            continue
    return scripts


def _to_unicode_host(ascii_host: str) -> str:
    """Decode punycode labels for analysis; fall back to ASCII if undecodable."""
    if "xn--" not in ascii_host:
        return ascii_host
    try:
        return ascii_host.encode("ascii").decode("idna")
    except UnicodeError:
        return ascii_host


def parse_url(raw: str, *, max_length: int = DEFAULT_MAX_URL_LENGTH) -> ParsedURL:
    """Validate and normalize ``raw``; raise ``URLValidationError`` if unsafe/invalid."""
    if not isinstance(raw, str):
        raise URLValidationError("not_a_string", "URL must be a string.")

    text = raw.strip()
    if not text:
        raise URLValidationError("empty_url", "URL is empty.")
    if len(text) > max_length:
        raise URLValidationError(
            "url_too_long", f"URL exceeds the maximum length of {max_length} characters."
        )

    # Control (Cc) and format (Cf, e.g. zero-width / bidi override) characters
    # are used to disguise URLs. Trade-off: this also rejects the rare legitimate
    # ZWNJ in some scripts.
    for ch in text:
        if ch.isspace() or unicodedata.category(ch) in {"Cc", "Cf"}:
            raise URLValidationError(
                "invalid_character", "URL contains whitespace, control, or invisible characters."
            )
    if "\\" in text:
        raise URLValidationError(
            "backslash", "URL contains a backslash, which parsers interpret inconsistently."
        )

    try:
        split = urlsplit(text)
    except ValueError:
        raise URLValidationError("malformed_url", "URL is malformed.") from None

    scheme = split.scheme.lower()
    if not scheme:
        raise URLValidationError("missing_scheme", "URL must start with http:// or https://.")
    if scheme not in ALLOWED_SCHEMES:
        raise URLValidationError(
            "unsupported_scheme", f"Unsupported URL scheme {_clip(scheme)!r}; only http and https."
        )

    try:
        port = split.port
    except ValueError:
        raise URLValidationError("invalid_port", "URL has an invalid port.") from None
    if port == 0:
        raise URLValidationError("invalid_port", "URL has an invalid port.")

    raw_host = split.hostname
    if not raw_host:
        raise URLValidationError("missing_host", "URL has no host.")
    host = raw_host[:-1] if raw_host.endswith(".") else raw_host  # strip FQDN dot
    if not host:
        raise URLValidationError("missing_host", "URL has no host.")

    # --- Host classification and IDNA normalization -----------------------
    is_ip = False
    try:
        ipaddress.ip_address(host)
        is_ip = True
    except ValueError:
        pass

    is_obfuscated_ip = False
    has_non_ascii_host = not host.isascii()
    if is_ip:
        ascii_host = host
        unicode_host = host
    else:
        if has_non_ascii_host:
            try:
                ascii_host = host.encode("idna").decode("ascii")
            except UnicodeError:
                raise URLValidationError("invalid_host", "URL host is not a valid domain name.") from None
        else:
            ascii_host = host
        labels = ascii_host.split(".")
        if len(ascii_host) > 253 or any(not _LABEL_RE.match(label) for label in labels):
            raise URLValidationError("invalid_host", "URL host is not a valid domain name.")
        is_obfuscated_ip = bool(_OBFUSCATED_IP_RE.match(ascii_host))
        unicode_host = _to_unicode_host(ascii_host)

    has_punycode = (not is_ip) and any(label.startswith("xn--") for label in ascii_host.split("."))
    mixed_labels = (
        ()
        if is_ip
        else tuple(
            label
            for label in unicode_host.split(".")
            if len(_label_scripts(label) & CONFUSABLE_SCRIPTS) >= 2
        )
    )

    # --- Domain structure (offline public-suffix lookup) -------------------
    if is_ip or is_obfuscated_ip:
        subdomain_labels: tuple[str, ...] = ()
        domain, suffix, registered = ascii_host, "", ascii_host
    else:
        ext = _TLD_EXTRACT(ascii_host)
        subdomain_labels = tuple(part for part in ext.subdomain.split(".") if part)
        domain, suffix = ext.domain, ext.suffix
        host_labels = ascii_host.split(".")
        if not suffix and len(host_labels) >= 2:
            # TLD not in the public suffix list (e.g. ".example", new gTLDs).
            # Browsers treat the last label as the suffix; do the same.
            domain, suffix = host_labels[-2], host_labels[-1]
            subdomain_labels = tuple(host_labels[:-2])
        registered = f"{domain}.{suffix}" if domain and suffix else (domain or ascii_host)

    # --- Userinfo (the part before '@' in the authority) -------------------
    has_userinfo = "@" in split.netloc
    userinfo_user = ""
    if has_userinfo:
        userinfo_user = split.netloc.rpartition("@")[0].split(":", 1)[0][:64]

    # --- Rebuild a canonical string used for length/char-count features ----
    host_part = f"[{ascii_host}]" if ":" in ascii_host else ascii_host
    netloc = host_part + (f":{port}" if port is not None else "")
    if has_userinfo:
        # Keep the '@' (it is a feature) but never retain the password anywhere.
        netloc = userinfo_user + "@" + netloc
    normalized = urlunsplit((scheme, netloc, split.path, split.query, split.fragment))

    return ParsedURL(
        normalized=normalized,
        scheme=scheme,
        host=ascii_host,
        host_unicode=unicode_host,
        port=port,
        has_userinfo=has_userinfo,
        userinfo_user=userinfo_user,
        path=split.path,
        query=split.query,
        fragment=split.fragment,
        subdomain_labels=subdomain_labels,
        domain=domain,
        suffix=suffix,
        registered_domain=registered,
        is_ip=is_ip,
        is_obfuscated_ip=is_obfuscated_ip,
        has_punycode=has_punycode,
        has_non_ascii_host=has_non_ascii_host,
        mixed_script_labels=mixed_labels,
    )


# --------------------------------------------------------------------------
# Reusable URL signals (shared by feature extraction and the rule engine)
# --------------------------------------------------------------------------
def sensitive_keywords_found(parsed: ParsedURL) -> tuple[str, ...]:
    """Distinct sensitive keywords present in host, path, or query (sorted)."""
    haystack = f"{parsed.host}{parsed.path}?{parsed.query}".lower()
    return tuple(sorted(k for k in SENSITIVE_KEYWORDS if k in haystack))


def has_embedded_url(parsed: ParsedURL) -> bool:
    """True if another http(s):// URL appears inside the path or query."""
    return bool(_EMBEDDED_URL_RE.search(f"{parsed.path}?{parsed.query}".lower()))


def path_has_executable_extension(parsed: ParsedURL) -> bool:
    """True if the last path segment ends in an executable/script extension."""
    last = parsed.path.rsplit("/", 1)[-1].lower()
    return "." in last and last.rsplit(".", 1)[-1] in EXECUTABLE_EXTENSIONS


def count_percent_encoded(text: str) -> int:
    """Count %XX sequences."""
    return len(_PERCENT_RE.findall(text))


def _shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    total = len(text)
    return -sum((n / total) * math.log2(n / total) for n in Counter(text).values())


# --------------------------------------------------------------------------
# Feature vector (shared by training and inference)
# --------------------------------------------------------------------------
FEATURE_NAMES: tuple[str, ...] = (
    "url_length",
    "hostname_length",
    "path_length",
    "query_length",
    "num_dots_host",
    "num_hyphens_host",
    "num_digits_url",
    "num_digits_host",
    "host_digit_ratio",
    "host_entropy",
    "num_subdomains",
    "num_domain_labels",
    "host_is_ip",
    "has_userinfo",
    "num_at_symbols",
    "has_punycode",
    "has_non_ascii_host",
    "has_mixed_script",
    "num_percent_encoded",
    "has_nonstandard_port",
    "is_https",
    "path_depth",
    "num_query_params",
    "num_sensitive_keywords",
    "has_embedded_url",
    "path_has_executable_ext",
)
"""Canonical feature order. Changing it requires bumping FEATURE_SCHEMA_VERSION."""


def extract_features(parsed: ParsedURL) -> dict[str, float]:
    """Compute the numeric feature dict for a parsed URL.

    Pure and deterministic: same input -> same output, no I/O.
    """
    url = parsed.normalized
    host = parsed.host
    # Hyphen/digit counts use the decoded host: punycode ("xn--" + encoded tail)
    # adds hyphens and digits of its own that say nothing about the domain.
    readable_host = parsed.host_unicode
    host_digits = sum(ch.isdigit() for ch in readable_host)
    path_segments = [seg for seg in parsed.path.split("/") if seg]

    values: dict[str, float] = {
        "url_length": len(url),
        "hostname_length": len(host),
        "path_length": len(parsed.path),
        "query_length": len(parsed.query),
        "num_dots_host": host.count("."),
        "num_hyphens_host": readable_host.count("-"),
        "num_digits_url": sum(ch.isdigit() for ch in url),
        "num_digits_host": host_digits,
        "host_digit_ratio": host_digits / len(readable_host) if readable_host else 0.0,
        "host_entropy": _shannon_entropy(host),
        "num_subdomains": len(parsed.subdomain_labels),
        "num_domain_labels": len(host.split(".")),
        "host_is_ip": parsed.is_ip or parsed.is_obfuscated_ip,
        "has_userinfo": parsed.has_userinfo,
        "num_at_symbols": url.count("@"),
        "has_punycode": parsed.has_punycode,
        "has_non_ascii_host": parsed.has_non_ascii_host,
        "has_mixed_script": bool(parsed.mixed_script_labels),
        "num_percent_encoded": count_percent_encoded(url),
        "has_nonstandard_port": parsed.port is not None and parsed.port not in STANDARD_PORTS,
        "is_https": parsed.scheme == "https",
        "path_depth": len(path_segments),
        "num_query_params": len(parse_qsl(parsed.query, keep_blank_values=True)),
        "num_sensitive_keywords": len(sensitive_keywords_found(parsed)),
        "has_embedded_url": has_embedded_url(parsed),
        "path_has_executable_ext": path_has_executable_extension(parsed),
    }
    if set(values) != set(FEATURE_NAMES):  # guards against drift when editing
        raise RuntimeError("FEATURE_NAMES and extract_features are out of sync")
    return {name: float(values[name]) for name in FEATURE_NAMES}


def features_to_vector(features: dict[str, float]) -> list[float]:
    """Order a feature dict according to FEATURE_NAMES (the model's input order)."""
    return [features[name] for name in FEATURE_NAMES]
