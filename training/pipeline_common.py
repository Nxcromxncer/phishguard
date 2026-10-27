"""Helpers shared by prepare_dataset.py, train.py, and evaluate.py.

Added beyond the required file list so that feature building, hashing, and metric
definitions exist exactly once. Most importantly, features are built with the
backend's own ``parse_url`` / ``extract_features`` so training and inference can
never drift apart.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.features import (  # noqa: E402
    FEATURE_NAMES,
    URLValidationError,
    extract_features,
    features_to_vector,
    parse_url,
)

DEFAULT_DATA_DIR = BACKEND / "data"
DEFAULT_MODEL_DIR = BACKEND / "models"
PREPARED_NAME = "prepared.csv"
REPORT_NAME = "prepared_report.json"
SPLIT_NAME = "split.json"
LABEL_NAMES = {0: "legitimate", 1: "phishing"}
LOW_SUPPORT_ROWS = 30
LOW_SUPPORT_PER_CLASS = 10


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fail(message: str) -> "SystemExit":
    """Exit with a readable message (usable as ``raise fail(...)``)."""
    return SystemExit(f"error: {message}")


def load_prepared(path: Path) -> pd.DataFrame:
    """Read prepared.csv and validate its columns and labels."""
    if not Path(path).is_file():
        raise fail(f"{path} not found. Run training/prepare_dataset.py first.")
    df = pd.read_csv(path, dtype={"url": str, "source": str, "registered_domain": str})
    for column in ("url", "label", "source", "registered_domain"):
        if column not in df.columns:
            raise fail(f"{path} has no '{column}' column; re-run prepare_dataset.py.")
    if not set(df["label"].unique()) <= {0, 1}:
        raise fail("labels in prepared dataset must be 0 or 1.")
    return df


def build_matrix(df: pd.DataFrame) -> np.ndarray:
    """Feature matrix in FEATURE_NAMES order, built with the backend's pipeline."""
    rows: list[list[float]] = []
    for url in df["url"]:
        try:
            rows.append(features_to_vector(extract_features(parse_url(url))))
        except URLValidationError as exc:
            # The dataset was validated with the same parser, so this means the
            # parser changed since preparation. Refuse rather than train on a subset.
            raise fail(f"a prepared URL no longer parses ({exc.code}); re-run prepare_dataset.py.") from None
    return np.asarray(rows, dtype=float)


def metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Confusion-matrix based metrics. Undefined ratios are None, never silently 0."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    def ratio(num: int, den: int) -> float | None:
        return round(num / den, 4) if den else None

    precision, recall = ratio(tp, tp + fp), ratio(tp, tp + fn)
    f1 = round(2 * precision * recall / (precision + recall), 4) if precision and recall else (0.0 if precision is not None and recall is not None else None)
    return {
        "n": int(len(y_true)),
        "n_phishing": int((y_true == 1).sum()),
        "n_legitimate": int((y_true == 0).sum()),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": ratio(fp, fp + tn),
        "false_negatives": fn,
        "low_support": bool(
            len(y_true) < LOW_SUPPORT_ROWS
            or min(int((y_true == 1).sum()), int((y_true == 0).sum())) < LOW_SUPPORT_PER_CLASS
        ),
    }
