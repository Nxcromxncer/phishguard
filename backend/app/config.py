"""Runtime configuration for the PhishGuard backend.

All tunables live here so that limits and thresholds are never scattered through
the code as magic numbers. Values come from ``PHISHGUARD_*`` environment
variables, with safe defaults.

Design notes
------------
* ``Settings`` is a frozen dataclass: once the app starts, configuration cannot
  be mutated by accident.
* Validation happens at construction, so a typo such as
  ``PHISHGUARD_HIGH_RISK_THRESHOLD=abc`` fails loudly at startup instead of
  silently falling back to a default.
* Decision thresholds are PROVISIONAL. They are placeholders until they are
  tuned on held-out evaluation data (Phase 5).
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _int_env(name: str, default: int, *, minimum: int = 1) -> int:
    """Read an integer environment variable, validating type and lower bound."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


@dataclass(frozen=True)
class Settings:
    """Validated, immutable backend settings."""

    # --- Input limits -----------------------------------------------------
    max_url_length: int = 2048
    """Longest URL (in characters) accepted by /scan."""

    max_body_bytes: int = 8192
    """Largest request body accepted by any endpoint (enforced before parsing)."""

    # --- Abuse protection -------------------------------------------------
    rate_limit_per_minute: int = 60
    """Max /scan requests per client address per 60 s (in-memory, per process)."""

    rate_limit_max_clients: int = 10_000
    """Upper bound on tracked clients so the limiter cannot grow without limit."""

    # --- CORS -------------------------------------------------------------
    cors_origins: tuple[str, ...] = ()
    """Allowed browser origins. Empty = no CORS headers are sent (safe default).

    Extension pages/service workers with ``host_permissions`` are not subject to
    CORS, so the extension does not need this. It exists for development tools.
    """

    # --- Decision thresholds (PROVISIONAL - see decision.py) --------------
    suspicious_threshold: int = 30
    """Rule score at/above which the rules count as a suspicious signal."""

    high_risk_threshold: int = 60
    """Rule score at/above which risk is reported as high."""

    # --- Logging ----------------------------------------------------------
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        if not 0 < self.suspicious_threshold <= 100:
            raise ValueError("suspicious_threshold must be in 1..100")
        if not self.suspicious_threshold <= self.high_risk_threshold <= 100:
            raise ValueError("high_risk_threshold must be in suspicious_threshold..100")
        if self.max_url_length < 8:
            raise ValueError("max_url_length is unreasonably small")
        if self.max_body_bytes < 64:
            raise ValueError("max_body_bytes is unreasonably small")
        if self.log_level.upper() not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"invalid log_level {self.log_level!r}")

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from ``PHISHGUARD_*`` environment variables."""
        origins_raw = os.environ.get("PHISHGUARD_CORS_ORIGINS", "")
        origins = tuple(o.strip() for o in origins_raw.split(",") if o.strip())
        defaults = cls()
        return cls(
            max_url_length=_int_env("PHISHGUARD_MAX_URL_LENGTH", defaults.max_url_length, minimum=8),
            max_body_bytes=_int_env("PHISHGUARD_MAX_BODY_BYTES", defaults.max_body_bytes, minimum=64),
            rate_limit_per_minute=_int_env(
                "PHISHGUARD_RATE_LIMIT_PER_MINUTE", defaults.rate_limit_per_minute
            ),
            cors_origins=origins,
            suspicious_threshold=_int_env(
                "PHISHGUARD_SUSPICIOUS_THRESHOLD", defaults.suspicious_threshold
            ),
            high_risk_threshold=_int_env(
                "PHISHGUARD_HIGH_RISK_THRESHOLD", defaults.high_risk_threshold
            ),
            log_level=os.environ.get("PHISHGUARD_LOG_LEVEL", defaults.log_level).upper(),
        )
