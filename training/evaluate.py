"""Evaluate a trained model on its held-out TEST set (never used for fitting).

    python training/evaluate.py --model-dir backend/models

Reports precision, recall, F1, confusion matrix, false-positive rate, and the
false-negative count, plus a breakdown by URL category and a comparison against
the rule engine alone. Writes ``evaluation.json`` into the model directory.

Read the numbers honestly
-------------------------
* A high score proves little by itself. It depends on the dataset (how it was
  collected, whether classes differ in artifacts like paths or schemes), on how
  the split was made (a random split would let a model memorise domains), and on
  how old the data is. Phishing changes quickly; today's test set is not
  tomorrow's traffic.
* Categories with few examples are flagged ``low_support``: do not trust them.
* The threshold used here is provisional (0.5 unless stated), not tuned.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from pipeline_common import (
    DEFAULT_DATA_DIR,
    DEFAULT_MODEL_DIR,
    LABEL_NAMES,
    PREPARED_NAME,
    SPLIT_NAME,
    build_matrix,
    fail,
    load_prepared,
    metrics_from_predictions,
    parse_url,
    sha256_file,
)

from app.config import Settings  # noqa: E402
from app.features import FEATURE_NAMES  # noqa: E402
from app.model import META_FILENAME, MODEL_FILENAME  # noqa: E402
from app.rules import evaluate_rules  # noqa: E402

SECURITY_NOTES = """\
What the errors mean for security
  False positive (legitimate flagged): users see warnings on safe sites. Too many and
    people learn to ignore PhishGuard (alert fatigue), which also trains them to click
    through the one warning that is real; a false positive on a business-critical site
    has a direct cost.
  False negative (phishing missed): the user receives no warning and may enter
    credentials. This is the costly error for the person, but it is invisible: a
    quiet tool looks identical to a safe one. Never present 'no warning' as 'safe'.
  The balance is set by the threshold. Lowering it finds more phishing and flags more
  legitimate sites; there is no setting without both kinds of error."""


def category_masks(df: pd.DataFrame, X: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    col = {name: X[:, i] for i, name in enumerate(FEATURE_NAMES)}
    length = col["url_length"]
    has_path = (col["path_length"] > 1) | (col["query_length"] > 0)
    cats: dict[str, dict[str, np.ndarray]] = {
        "scheme": {"https": col["is_https"] == 1, "http": col["is_https"] == 0},
        "host_type": {"ip_address": col["host_is_ip"] == 1, "domain_name": col["host_is_ip"] == 0},
        "has_path_or_query": {"yes": has_path, "no": ~has_path},
        "internationalized_host": {"yes": col["has_punycode"] == 1, "no": col["has_punycode"] == 0},
        "url_length": {
            "<40": length < 40,
            "40-79": (length >= 40) & (length < 80),
            "80-149": (length >= 80) & (length < 150),
            ">=150": length >= 150,
        },
    }
    sources = df["source"].to_numpy()
    if len(set(sources)) > 1:
        cats["source"] = {str(s): sources == s for s in sorted(set(sources))}
    return cats


def fmt(value: Any) -> str:
    return "n/a" if value is None else (f"{value:.4f}" if isinstance(value, float) else str(value))


def print_metrics(title: str, m: dict[str, Any]) -> None:
    print(f"\n{title}")
    print(f"  n={m['n']} (phishing={m['n_phishing']}, legitimate={m['n_legitimate']})")
    print(f"  precision={fmt(m['precision'])}  recall={fmt(m['recall'])}  f1={fmt(m['f1'])}")
    print(f"  false-positive rate={fmt(m['false_positive_rate'])}  false negatives={m['false_negatives']}")
    print(f"  confusion matrix  [rows = actual, cols = predicted]")
    print(f"                  pred legit   pred phish")
    print(f"    actual legit  {m['tn']:>10}   {m['fp']:>10}")
    print(f"    actual phish  {m['fn']:>10}   {m['tp']:>10}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate a PhishGuard model on its held-out test set.")
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--threshold", type=float, default=None, help="Override the model's stored threshold.")
    p.add_argument("--report", type=Path, default=None, help="Output JSON (default: <model-dir>/evaluation.json).")
    args = p.parse_args(argv)

    meta_path, model_path, split_path = (args.model_dir / n for n in (META_FILENAME, MODEL_FILENAME, SPLIT_NAME))
    for path in (meta_path, model_path, split_path):
        if not path.is_file():
            raise fail(f"{path} not found. Train a model first (training/train.py).")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))

    # The files were written by train.py on this machine; verify integrity before unpickling anyway.
    if sha256_file(model_path) != meta["model_sha256"]:
        raise fail("model file checksum does not match model_meta.json.")
    prepared_path = args.data_dir / PREPARED_NAME
    if sha256_file(prepared_path) != split["prepared_sha256"]:
        raise fail("prepared.csv differs from the one this model was trained on; evaluating would mix datasets.")

    df = load_prepared(prepared_path)
    test_idx = np.asarray(split["test"], dtype=int)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    X = build_matrix(test_df)
    y = test_df["label"].to_numpy(dtype=int)

    model = joblib.load(model_path)
    threshold = float(args.threshold if args.threshold is not None else meta["threshold"])
    scores = model.predict_proba(X)[:, 1]
    pred = (scores >= threshold).astype(int)

    overall = metrics_from_predictions(y, pred)
    overall["roc_auc"] = round(float(roc_auc_score(y, scores)), 4) if len(set(y)) == 2 else None

    rules_pred = np.array(
        [1 if evaluate_rules(parse_url(u)).score >= Settings().suspicious_threshold else 0 for u in test_df["url"]]
    )
    rules_overall = metrics_from_predictions(y, rules_pred)

    categories: dict[str, dict[str, Any]] = {}
    for cat, groups in category_masks(test_df, X).items():
        categories[cat] = {}
        for name, mask in groups.items():
            if mask.sum():
                categories[cat][name] = metrics_from_predictions(y[mask], pred[mask])

    print(f"Model {meta['model_version']}  | threshold {threshold} (provisional)  | split: {meta['split']['method']}")
    ds = meta["dataset"]
    if ds.get("synthetic"):
        print("\n*** SYNTHETIC DATA: these numbers measure how well the model separates INVENTED templates. ***")
        print("*** They say NOTHING about real-world phishing detection. ***")
    for warning in ds.get("warnings", []):
        if not warning.startswith("SYNTHETIC"):
            print(f"\nDATASET WARNING: {warning}")
    print_metrics("ML model on the held-out TEST set", overall)
    print(f"  roc_auc={fmt(overall['roc_auc'])}")
    print_metrics(f"Rule engine alone (score >= {Settings().suspicious_threshold}) on the same test set", rules_overall)

    print("\nBy category (ML model)  [low-support groups marked ~, do not trust them]")
    for cat, groups in categories.items():
        for name, m in groups.items():
            flag = "~" if m["low_support"] else " "
            print(f" {flag} {cat}={name:<12} n={m['n']:<6} P={fmt(m['precision']):<7} R={fmt(m['recall']):<7} FPR={fmt(m['false_positive_rate']):<7} FN={m['false_negatives']}")
    print("\n" + SECURITY_NOTES)
    print("\nCaveat: scores on a held-out split of ONE dataset do not predict accuracy on live traffic.")

    out = args.report or (args.model_dir / "evaluation.json")
    out.write_text(
        json.dumps(
            {
                "model_version": meta["model_version"],
                "threshold": threshold,
                "threshold_status": "provisional",
                "split_method": meta["split"]["method"],
                "test_rows": overall["n"],
                "synthetic_data": bool(ds.get("synthetic")),
                "dataset_warnings": ds.get("warnings", []),
                "ml": overall,
                "rules_baseline": rules_overall,
                "by_category": categories,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
