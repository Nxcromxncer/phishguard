"""ML inference: missing model, artifact validation, tampering, honest reporting.

The artifacts come from the real prepare -> train pipeline run on SYNTHETIC data
(see conftest.pipeline). Nothing here says anything about real-world accuracy.
"""

from __future__ import annotations

import json
import os

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier

from app import model as model_module
from app.config import Settings
from app.features import FEATURE_NAMES, parse_url
from app.main import create_app
from app.model import ModelService, write_model_artifact

from .conftest import copy_dir, edit_meta

PHISHY = "http://paypal-login-ab12c.xyz/login/x1y2z3w4.php?id=4821"
BENIGN = "https://www.gardenriver.com/about"


def load(directory, **kwargs):
    kwargs.setdefault("allow_synthetic", True)
    return ModelService.from_directory(directory, **kwargs)


def not_synthetic(meta):
    meta["dataset"]["synthetic"] = False


# --------------------------------------------------------------------------
# No model
# --------------------------------------------------------------------------
def test_empty_directory_reports_not_configured(tmp_path):
    service = ModelService.from_directory(tmp_path)
    assert service.predict(parse_url("https://example.com/")).status == "not_configured"
    assert service.info().status == "not_configured"


def test_missing_model_never_fabricates_a_prediction(tmp_path):
    result = ModelService.from_directory(tmp_path / "does-not-exist").predict(parse_url("http://8.8.8.8/login"))
    assert result.prediction is None and result.score is None and result.model_version is None


def test_default_constructor_is_not_configured():
    assert ModelService().predict(parse_url("https://example.com/")).status == "not_configured"


# --------------------------------------------------------------------------
# A valid artifact
# --------------------------------------------------------------------------
def test_valid_artifact_predicts_with_metadata(pipeline):
    service = load(pipeline.rf)
    result = service.predict(parse_url(PHISHY))
    assert result.status == "predicted"
    assert result.prediction in {"phishing", "legitimate"}
    assert 0.0 <= result.score <= 1.0
    assert result.model_version.startswith("rf-")
    assert result.metadata["model_type"] == "rf"
    assert result.metadata["calibrated"] == "false"
    assert result.metadata["threshold"] == "0.500"
    assert result.metadata["split_method"] == "group"


def test_score_is_described_as_uncalibrated_not_a_probability(pipeline):
    detail = load(pipeline.rf).predict(parse_url(PHISHY)).detail.lower()
    assert "uncalibrated" in detail and "not a probability" in detail
    assert "provisional" in detail


def test_prediction_is_deterministic(pipeline):
    service = load(pipeline.rf)
    first, second = (service.predict(parse_url(PHISHY)) for _ in range(2))
    assert first.score == second.score


def test_phishing_like_scores_higher_than_benign_like(pipeline):
    service = load(pipeline.rf)
    assert service.predict(parse_url(PHISHY)).score > service.predict(parse_url(BENIGN)).score


def test_both_model_kinds_load(pipeline):
    for directory, kind in ((pipeline.rf, "rf"), (pipeline.hgb, "hgb")):
        assert load(directory).predict(parse_url(PHISHY)).metadata["model_type"] == kind


@pytest.mark.parametrize(
    ("threshold", "expected"),
    [(0.5, "phishing"), (0.6, "phishing"), (0.61, "legitimate"), (0.9, "legitimate")],
)
def test_threshold_decides_the_label_but_not_the_score(pipeline, threshold, expected):
    """Fixed-score stub: the label is `score >= threshold` (boundary inclusive)."""

    class FixedScore:
        def predict_proba(self, _x):
            return [[0.4, 0.6]]

    service = load(pipeline.rf)
    meta = {**service._loaded.meta, "threshold": threshold}
    service._loaded = type(service._loaded)(estimator=FixedScore(), meta=meta)
    result = service.predict(parse_url(BENIGN))
    assert result.score == 0.6 and result.prediction == expected


def test_prediction_failure_is_unavailable_not_clean(pipeline):
    service = load(pipeline.rf)

    class Exploding:
        def predict_proba(self, _x):
            raise RuntimeError("boom")

    service._loaded = type(service._loaded)(estimator=Exploding(), meta=service._loaded.meta)
    result = service.predict(parse_url(PHISHY))
    assert result.status == "unavailable" and result.prediction is None and result.score is None


# --------------------------------------------------------------------------
# Synthetic-data guard
# --------------------------------------------------------------------------
def test_synthetic_model_refused_by_default(pipeline):
    service = ModelService.from_directory(pipeline.rf)  # allow_synthetic defaults to False
    result = service.predict(parse_url(PHISHY))
    assert result.status == "unavailable" and "synthetic" in result.detail.lower()
    assert service.info().status == "unavailable"


def test_synthetic_model_when_allowed_is_labelled_as_demo(pipeline):
    service = load(pipeline.rf, allow_synthetic=True)
    assert "DEMO MODEL" in service.predict(parse_url(PHISHY)).detail
    assert "DEMO" in service.info().limitation
    assert service.predict(parse_url(PHISHY)).metadata["trained_on_synthetic"] == "true"


# --------------------------------------------------------------------------
# Compatibility and tampering: every one of these must end in "unavailable"
# --------------------------------------------------------------------------
def assert_rejected(service, fragment):
    result = service.predict(parse_url(PHISHY))
    assert result.status == "unavailable", result
    assert result.prediction is None and result.score is None
    assert fragment.lower() in result.detail.lower(), result.detail


def test_feature_schema_version_mismatch(pipeline, tmp_path):
    directory = copy_dir(pipeline.rf, tmp_path)
    edit_meta(directory, lambda m: m.update(feature_schema_version=999))
    assert_rejected(load(directory), "schema mismatch")


def test_feature_name_or_order_mismatch(pipeline, tmp_path):
    directory = copy_dir(pipeline.rf, tmp_path)
    edit_meta(directory, lambda m: m.update(feature_names=list(reversed(FEATURE_NAMES))))
    assert_rejected(load(directory), "feature names")


def test_wrong_number_of_features_in_the_model(tmp_path):
    estimator = RandomForestClassifier(n_estimators=3, random_state=0).fit([[0, 1, 2], [2, 1, 0]], [0, 1])
    write_model_artifact(
        estimator,
        {"model_type": "rf", "threshold": 0.5, "dataset": {"rows": 2, "synthetic": False}},
        tmp_path,
    )
    assert_rejected(load(tmp_path), "different number of features")


def test_tampered_model_file_is_rejected_before_unpickling(pipeline, tmp_path, monkeypatch):
    directory = copy_dir(pipeline.rf, tmp_path)
    with open(directory / "model.joblib", "ab") as handle:
        handle.write(b"tampered")
    unpickled: list[bool] = []
    real_load = joblib.load
    monkeypatch.setattr(model_module.joblib, "load", lambda *a, **k: unpickled.append(True) or real_load(*a, **k))
    assert_rejected(load(directory), "checksum")
    assert unpickled == [], "the file was unpickled before its checksum was verified"


def test_pinned_hash_must_match(pipeline):
    good = json.loads((pipeline.rf / "model_meta.json").read_text())["model_sha256"]
    assert load(pipeline.rf, expected_sha256=good).info().status == "available"
    assert_rejected(load(pipeline.rf, expected_sha256="0" * 64), "pinned")


def test_swapped_model_with_consistent_metadata_is_caught_by_the_pin(pipeline, tmp_path):
    """Metadata alone proves nothing: an attacker who replaces both files passes the
    internal check, but not a hash pinned out of band."""
    pinned = json.loads((pipeline.rf / "model_meta.json").read_text())["model_sha256"]
    directory = copy_dir(pipeline.hgb, tmp_path)  # a different, self-consistent artifact
    assert load(directory).info().status == "available"  # internal check passes...
    assert_rejected(load(directory, expected_sha256=pinned), "pinned")  # ...the pin does not


def test_scikit_learn_version_mismatch(pipeline, tmp_path):
    directory = copy_dir(pipeline.rf, tmp_path)
    edit_meta(directory, lambda m: m.update(sklearn_version="0.1.0"))
    assert_rejected(load(directory), "scikit-learn")


def test_symlinked_model_file_is_refused(pipeline, tmp_path):
    directory = copy_dir(pipeline.rf, tmp_path)
    os.rename(directory / "model.joblib", directory / "real.bin")
    os.symlink(directory / "real.bin", directory / "model.joblib")
    assert_rejected(load(directory), "symbolic link")


@pytest.mark.parametrize("content", ["{not json", "[]", '{"model_version": "x"}'])
def test_corrupt_or_incomplete_metadata(pipeline, tmp_path, content):
    directory = copy_dir(pipeline.rf, tmp_path)
    (directory / "model_meta.json").write_text(content)
    assert load(directory).info().status == "unavailable"


@pytest.mark.parametrize("threshold", [0, 1, -0.2, 1.5, "0.5", True])
def test_invalid_threshold_is_rejected(pipeline, tmp_path, threshold):
    directory = copy_dir(pipeline.rf, tmp_path)
    edit_meta(directory, lambda m: m.update(threshold=threshold))
    assert_rejected(load(directory), "threshold")


def test_metadata_without_model_file_is_unavailable_not_absent(pipeline, tmp_path):
    directory = copy_dir(pipeline.rf, tmp_path)
    (directory / "model.joblib").unlink()
    assert load(directory).info().status == "unavailable"


def test_garbage_model_file_with_matching_hash_is_unavailable(pipeline, tmp_path):
    import hashlib

    directory = copy_dir(pipeline.rf, tmp_path)
    payload = b"this is not a pickle"
    (directory / "model.joblib").write_bytes(payload)
    edit_meta(directory, lambda m: m.update(model_sha256=hashlib.sha256(payload).hexdigest()))
    assert_rejected(load(directory), "could not be loaded")


# --------------------------------------------------------------------------
# Through the HTTP API
# --------------------------------------------------------------------------
def api_client(model_dir, **overrides):
    cfg = Settings(rate_limit_per_minute=1000, model_dir=str(model_dir), **overrides)
    return TestClient(create_app(cfg))


def test_api_uses_the_trained_model_and_combines_it_with_rules(pipeline):
    with api_client(pipeline.rf, allow_synthetic_model=True) as client:
        phishy = client.post("/scan", json={"url": PHISHY}).json()
        benign = client.post("/scan", json={"url": BENIGN}).json()
        health = client.get("/health").json()
    assert phishy["ml"]["status"] == "predicted" and phishy["ml"]["prediction"] == "phishing"
    assert phishy["verdict"] in {"suspicious", "likely_phishing"}
    assert benign["ml"]["prediction"] == "legitimate"
    assert benign["verdict"] == "no_known_threat_detected"  # model ran and nothing flagged
    assert "not a guarantee" in benign["disclaimer"]
    by_name = {c["name"]: c for c in health["components"]}
    assert by_name["ml"]["status"] == "available" and "DEMO" in by_name["ml"]["limitation"]


def test_api_reports_rejected_model_as_unavailable_and_still_works(pipeline):
    with api_client(pipeline.rf) as client:  # synthetic model, not allowed
        body = client.post("/scan", json={"url": BENIGN}).json()
        health = client.get("/health").json()
    assert body["ml"]["status"] == "unavailable"
    assert body["verdict"] != "no_known_threat_detected"  # a missing component is never a clean result
    assert {c["name"]: c["status"] for c in health["components"]}["ml"] == "unavailable"
