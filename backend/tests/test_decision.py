"""Decision engine: evidence priority, conflicts, and honest handling of gaps.

Phase 5 will add threshold-tuning tests against held-out evaluation data.
"""

from __future__ import annotations

from app.decision import decide
from app.schemas import RiskLevel, Verdict

from .conftest import ml_result, rep_result, rules_result

SUSPICIOUS, HIGH = 30, 60


def run(rules=0, ml="not_configured", rep="not_configured"):
    return decide(
        rules_result(rules),
        ml_result(ml),
        rep_result(rep),
        suspicious_threshold=SUSPICIOUS,
        high_risk_threshold=HIGH,
    )


def text(decision) -> str:
    return " ".join(decision.explanation).lower()


# --- Reputation priority ----------------------------------------------------
def test_confirmed_malicious_match_is_likely_phishing_even_with_clean_signals():
    d = run(rules=0, ml="legitimate", rep="malicious")
    assert d.verdict is Verdict.likely_phishing
    assert d.risk_level is RiskLevel.high
    assert "outweighs" in text(d)  # the conflict is stated, not hidden


def test_malicious_match_with_nothing_else():
    d = run(rep="malicious")
    assert d.verdict is Verdict.likely_phishing and d.risk_level is RiskLevel.high


# --- Gaps are not negatives (central regression tests) ----------------------
def test_nothing_available_and_no_findings_is_unknown_not_clean():
    d = run()
    assert d.verdict is Verdict.unknown
    assert d.risk_level is RiskLevel.unknown
    assert "cannot tell" in text(d)


def test_unavailable_reputation_never_becomes_no_known_threat():
    d = run(rules=0, ml="not_configured", rep="unavailable")
    assert d.verdict is Verdict.unknown
    assert "unavailable" in text(d)


def test_unavailable_ml_and_reputation_with_no_findings_is_unknown():
    assert run(ml="unavailable", rep="unavailable").verdict is Verdict.unknown


def test_unavailable_reputation_with_suspicious_rules_still_flags_and_says_so():
    d = run(rules=50, rep="unavailable")
    assert d.verdict is Verdict.suspicious
    assert "could not be ruled out" in text(d)


# --- Clean outcomes ---------------------------------------------------------
def test_no_match_with_no_findings_is_no_known_threat_low():
    d = run(rep="no_match")
    assert d.verdict is Verdict.no_known_threat_detected
    assert d.risk_level is RiskLevel.low
    assert "not proof of safety" in text(d)


def test_ml_legitimate_with_no_findings_is_no_known_threat():
    assert run(ml="legitimate").verdict is Verdict.no_known_threat_detected


def test_weak_findings_below_threshold_do_not_trigger_a_warning():
    d = run(rules=20, rep="no_match")  # 20 == weak-indicator cap
    assert d.verdict is Verdict.no_known_threat_detected


# --- Combining signals ------------------------------------------------------
def test_rules_alone_cap_at_suspicious_medium():
    d = run(rules=40)
    assert d.verdict is Verdict.suspicious and d.risk_level is RiskLevel.medium


def test_very_high_rule_score_alone_is_suspicious_but_high_risk():
    d = run(rules=70)
    assert d.verdict is Verdict.suspicious and d.risk_level is RiskLevel.high


def test_ml_phishing_alone_is_suspicious_medium():
    d = run(rules=0, ml="phishing")
    assert d.verdict is Verdict.suspicious and d.risk_level is RiskLevel.medium
    assert "do not corroborate" in text(d)


def test_ml_and_rules_agreeing_is_likely_phishing():
    d = run(rules=40, ml="phishing")
    assert d.verdict is Verdict.likely_phishing and d.risk_level is RiskLevel.high


# --- Conflicting evidence ---------------------------------------------------
def test_rules_flag_but_model_disagrees_is_reported():
    d = run(rules=45, ml="legitimate")
    assert d.verdict is Verdict.suspicious
    assert "disagree" in text(d)


def test_model_flags_but_reputation_has_no_record_does_not_clear_it():
    d = run(ml="phishing", rep="no_match")
    assert d.verdict is Verdict.suspicious
    assert "does not clear it" in text(d)


# --- Component descriptions ---------------------------------------------------
def test_components_describe_all_three_sources_and_their_limits():
    d = run(rep="no_match", ml="legitimate")
    assert [c.name for c in d.components] == ["rules", "ml", "reputation"]
    by_name = {c.name: c for c in d.components}
    assert by_name["reputation"].status == "no_match"
    assert "not that it is safe" in by_name["reputation"].limitation
    assert "not a probability" in by_name["ml"].limitation
