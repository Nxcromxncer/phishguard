"""Threat-intelligence interface.

*** PHASE 1 PLACEHOLDER ***
No provider is integrated yet. The provider abstraction, a Safe Browsing
integration, caching, and timeout/rate-limit handling arrive in Phase 4. Until
then this service reports ``not_configured`` and NEVER reports ``no_match``,
because "we did not look" must not read as "nothing was found".
"""

from __future__ import annotations

from .features import ParsedURL
from .schemas import ComponentInfo, ReputationResult


class ReputationService:
    """Phase 1: always reports that no provider is configured."""

    def check(self, parsed: ParsedURL) -> ReputationResult:
        return ReputationResult(
            status="not_configured",
            provider=None,
            detail="No threat-intelligence provider is configured (implemented in Phase 4).",
        )

    def info(self) -> ComponentInfo:
        return ComponentInfo(
            name="reputation",
            status="not_configured",
            limitation="No threat-intelligence provider is configured; known-bad URLs cannot be detected.",
        )
