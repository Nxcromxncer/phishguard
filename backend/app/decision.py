"""Hybrid decision engine.

Combines rule, ML, and reputation results with explicit, documented logic.

Evidence priority
-----------------
1. A confirmed malicious threat-intelligence match is strong evidence: it alone
   yields ``likely_phishing`` / ``high``.
2. ML and rule signals each count as one *independent* signal. Two agreeing
   signals -> ``likely_phishing``; one signal -> ``suspicious``.
3. Disagreement between components is stated in the explanation, never hidden.
4. An unavailable or unconfigured component NEVER counts as a negative
   detection. "No signals" yields ``no_known_threat_detected`` only if at least
   one component that can independently vouch (reputation or ML) actually ran
   and found nothing; otherwise the verdict is ``unknown`` with risk ``unknown``.
5. Every explanation reminds the reader that this is an assessment.

Rules alone cap at ``suspicious``: URL-text heuristics are not enough to declare
"likely phishing" without a second, independent signal.

All thresholds come from Settings and are PROVISIONAL until tuned on held-out
data (Phase 5).
"""

from __future__ import annotations

from dataclasses import dataclass

from .schemas import (
    ComponentInfo,
    MLResult,
    ReputationResult,
    RiskLevel,
    RuleResult,
    Verdict,
)


@dataclass(frozen=True)
class Decision:
    verdict: Verdict
    risk_level: RiskLevel
    explanation: list[str]
    components: list[ComponentInfo]


def describe_components(
    rules: RuleResult, ml: MLResult, reputation: ReputationResult
) -> list[ComponentInfo]:
    """Report each component's status and what its result can and cannot mean."""
    ml_limits = {
        "predicted": "The score is an uncalibrated classifier output, not a probability of phishing.",
        "not_configured": "No trained model is loaded; no ML assessment was made.",
        "unavailable": "The model failed to produce a prediction; the ML assessment is missing.",
    }
    rep_limits = {
        "malicious": "Reported by the provider's database; data may lag behind new threats.",
        "no_match": "No match means the URL is not in the provider's database right now, not that it is safe.",
        "unavailable": "The lookup failed; whether the URL is listed is unknown.",
        "not_configured": "No provider is configured; known-bad URLs cannot be detected.",
    }
    return [
        ComponentInfo(
            name="rules",
            status=rules.status,
            limitation="Rules inspect only the URL text, never the page. No findings is not evidence of safety.",
        ),
        ComponentInfo(name="ml", status=ml.status, limitation=ml_limits[ml.status]),
        ComponentInfo(name="reputation", status=reputation.status, limitation=rep_limits[reputation.status]),
    ]


def decide(
    rules: RuleResult,
    ml: MLResult,
    reputation: ReputationResult,
    *,
    suspicious_threshold: int,
    high_risk_threshold: int,
) -> Decision:
    """Combine component results into a verdict, risk level, and explanation."""
    notes: list[str] = []

    rep_malicious = reputation.status == "malicious"
    rep_clean = reputation.status == "no_match"
    ml_flag = ml.status == "predicted" and ml.prediction == "phishing"
    ml_clean = ml.status == "predicted" and ml.prediction == "legitimate"
    rules_flag = rules.score >= suspicious_threshold
    rules_high = rules.score >= high_risk_threshold

    # ---- Verdict ---------------------------------------------------------
    if rep_malicious:
        verdict, risk = Verdict.likely_phishing, RiskLevel.high
        provider = reputation.provider or "the threat-intelligence provider"
        notes.append(f"{provider} lists this URL as malicious. This takes priority over other signals.")
        if ml_clean:
            notes.append("The ML model did not flag it, but a confirmed reputation match outweighs that.")
        if rules.score == 0:
            notes.append("The URL-text rules found nothing unusual; known-bad URLs can look ordinary.")

    elif ml_flag and rules_flag:
        verdict, risk = Verdict.likely_phishing, RiskLevel.high
        notes.append("Both the ML model and the URL-text rules flag this URL.")

    elif ml_flag or rules_flag:
        verdict = Verdict.suspicious
        if ml_flag:
            risk = RiskLevel.medium
            notes.append("The ML model flags this URL, but the URL-text rules do not corroborate it.")
        else:
            risk = RiskLevel.high if rules_high else RiskLevel.medium
            notes.append(
                "The URL-text rules found suspicious characteristics "
                f"(rule score {rules.score}/100) without an independent confirmation."
            )
        if rep_clean:
            notes.append("The reputation provider has no record of this URL, which does not clear it: new phishing pages are often not yet listed.")
        if ml_clean and rules_flag:
            notes.append("The ML model and the rules disagree; the model did not flag this URL.")

    else:
        if rep_clean or ml_clean:
            verdict, risk = Verdict.no_known_threat_detected, RiskLevel.low
            notes.append("No component flagged this URL. That is the absence of a known warning, not proof of safety.")
        else:
            verdict, risk = Verdict.unknown, RiskLevel.unknown
            notes.append(
                "Only the URL-text rules ran, and they found no strong warning signs. "
                "With no ML or reputation check available, PhishGuard cannot tell whether this URL is safe."
            )
        if rules.findings:
            notes.append(
                f"Minor indicators were noted ({len(rules.findings)}) but are common in legitimate URLs "
                "and did not reach the warning threshold."
            )

    # ---- Honest handling of missing or failed components -------------------
    if reputation.status == "unavailable":
        notes.append("The threat-intelligence lookup was unavailable, so a known-bad listing could not be ruled out.")
    elif reputation.status == "not_configured" and verdict is not Verdict.unknown:
        notes.append("No threat-intelligence provider is configured, so known-bad listings were not checked.")
    if ml.status == "unavailable":
        notes.append("The ML model was unavailable; its assessment is missing, not negative.")

    return Decision(
        verdict=verdict,
        risk_level=risk,
        explanation=notes,
        components=describe_components(rules, ml, reputation),
    )
