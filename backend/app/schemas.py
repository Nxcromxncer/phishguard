"""API contracts: request/response models shared by every backend module.

Key principle: *missing information is never a clean result.* Every detection
component reports an explicit status, and the response lists the limitations of
each component so a client can show uncertainty honestly.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DISCLAIMER = (
    "PhishGuard provides a risk assessment, not a guarantee. A URL with no "
    "warnings may still be malicious, and a flagged URL may be legitimate."
)


# --------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------
class Verdict(str, Enum):
    """Overall outcome. Deliberately has no value that means 'safe'."""

    likely_phishing = "likely_phishing"
    suspicious = "suspicious"
    no_known_threat_detected = "no_known_threat_detected"
    unknown = "unknown"


class RiskLevel(str, Enum):
    """Risk level. ``unknown`` is distinct from ``low`` on purpose."""

    low = "low"
    medium = "medium"
    high = "high"
    unknown = "unknown"


class Severity(str, Enum):
    """Strength of a single rule finding (see rules.py for score values)."""

    weak = "weak"
    moderate = "moderate"
    strong = "strong"


# --------------------------------------------------------------------------
# Request
# --------------------------------------------------------------------------
class ScanRequest(BaseModel):
    """Body of ``POST /scan``.

    Length and syntax of the URL are validated in ``features.parse_url`` against
    the configured limit; the overall body size is capped by middleware before
    this model is even built.
    """

    model_config = ConfigDict(extra="forbid")

    url: str = Field(min_length=1, description="Absolute http(s) URL to assess.")


# --------------------------------------------------------------------------
# Component results
# --------------------------------------------------------------------------
class RuleFinding(BaseModel):
    """One triggered heuristic rule, with the evidence that triggered it."""

    rule_id: str
    severity: Severity
    score: int = Field(ge=0)
    title: str
    explanation: str
    evidence: list[str] = Field(default_factory=list)


class RuleResult(BaseModel):
    """Output of the rule engine."""

    status: Literal["evaluated"] = "evaluated"
    score: int = Field(ge=0, le=100, description="Combined rule score, 0-100.")
    findings: list[RuleFinding] = Field(default_factory=list)


class MLResult(BaseModel):
    """Output of the ML component (a trained model arrives in Phase 3)."""

    # Pydantic reserves the "model_" prefix; we use it for model_version.
    model_config = ConfigDict(protected_namespaces=())

    status: Literal["predicted", "not_configured", "unavailable"]
    prediction: Literal["phishing", "legitimate"] | None = None
    score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Raw classifier output. Uncalibrated: NOT a probability of phishing.",
    )
    model_version: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    detail: str


class ReputationResult(BaseModel):
    """Output of the threat-intelligence component (provider arrives in Phase 4)."""

    status: Literal["malicious", "no_match", "unavailable", "not_configured"]
    provider: str | None = None
    detail: str
    cached: bool = False


class ComponentInfo(BaseModel):
    """Availability and limitations of one detection component."""

    name: Literal["rules", "ml", "reputation"]
    status: str
    limitation: str


# --------------------------------------------------------------------------
# Responses
# --------------------------------------------------------------------------
class ScanResponse(BaseModel):
    """Body of a successful ``POST /scan`` response."""

    url_display: str = Field(
        description="Privacy-conscious form of the URL: scheme + ASCII host (+ port) only."
    )
    host: str = Field(description="ASCII (punycode) host. Show this, not the Unicode form.")
    host_unicode: str | None = Field(
        default=None, description="Unicode form of an internationalized host, if different."
    )
    verdict: Verdict
    risk_level: RiskLevel
    rules: RuleResult
    ml: MLResult
    reputation: ReputationResult
    explanation: list[str]
    components: list[ComponentInfo]
    disclaimer: str = DISCLAIMER


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    components: list[ComponentInfo]


class ApiInfo(BaseModel):
    name: str
    version: str
    docs_url: str
    endpoints: list[str]
    disclaimer: str = DISCLAIMER


class ErrorBody(BaseModel):
    code: str
    message: str
    details: list[dict[str, str]] | None = None


class ErrorResponse(BaseModel):
    """Every non-2xx response uses this shape."""

    error: ErrorBody
