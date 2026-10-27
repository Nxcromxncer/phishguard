"""Shared, deterministic fixtures. No network, no credentials, no real dataset.

IMPORTANT: every Settings used in tests points ``model_dir`` at an empty temporary
directory, so a model you trained into backend/models can never change test results.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.schemas import MLResult, ReputationResult, RuleResult

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))  # lets tests import the training scripts


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    # High rate limit so ordinary tests never trip it; empty model dir => no model.
    return Settings(rate_limit_per_minute=1000, model_dir=str(tmp_path / "no-model-here"))


@pytest.fixture
def client(settings: Settings):
    with TestClient(create_app(settings)) as test_client:
        yield test_client


# --------------------------------------------------------------------------
# A small real pipeline run (synthetic data!) shared by model and training tests
# --------------------------------------------------------------------------
@pytest.fixture(scope="session")
def pipeline(tmp_path_factory) -> SimpleNamespace:
    import make_synthetic_sample
    import prepare_dataset
    import train

    base = tmp_path_factory.mktemp("pipeline")
    raw, data, rf, hgb = base / "syn.csv", base / "data", base / "rf", base / "hgb"
    make_synthetic_sample.main(["--n-per-class", "200", "--out", str(raw)])
    prepare_dataset.main(["--input", str(raw), "--out-dir", str(data), "--provenance-note", "unit test"])
    train.main(["--model", "rf", "--data-dir", str(data), "--out-dir", str(rf)])
    train.main(["--model", "hgb", "--data-dir", str(data), "--out-dir", str(hgb)])
    return SimpleNamespace(base=base, raw=raw, data=data, rf=rf, hgb=hgb)


def copy_dir(src: Path, tmp_path: Path, name: str = "copy") -> Path:
    """Copy a directory so a test can tamper with it without affecting others."""
    dest = tmp_path / name
    shutil.copytree(src, dest)
    return dest


def edit_meta(model_dir: Path, mutate) -> None:
    path = model_dir / "model_meta.json"
    meta = json.loads(path.read_text())
    mutate(meta)
    path.write_text(json.dumps(meta))


# Helpers for building component results in decision tests -------------------
def rules_result(score: int = 0) -> RuleResult:
    return RuleResult(score=score, findings=[])


def ml_result(kind: str = "not_configured") -> MLResult:
    if kind == "phishing":
        return MLResult(status="predicted", prediction="phishing", score=0.9, detail="test")
    if kind == "legitimate":
        return MLResult(status="predicted", prediction="legitimate", score=0.1, detail="test")
    return MLResult(status=kind, detail="test")  # type: ignore[arg-type]


def rep_result(status: str = "not_configured") -> ReputationResult:
    provider = "test-provider" if status in {"malicious", "no_match"} else None
    return ReputationResult(status=status, provider=provider, detail="test")  # type: ignore[arg-type]
