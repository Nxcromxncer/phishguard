"""Train a phishing-URL classifier on prepared.csv and write a model artifact.

    python training/train.py --model rf            # Random Forest baseline (default)
    python training/train.py --model hgb           # HistGradientBoosting comparison
    python training/train.py --model rf --out-dir /tmp/rf && python training/train.py --model hgb --out-dir /tmp/hgb

Design decisions
----------------
* Features come from the backend's own pipeline (pipeline_common.build_matrix), so
  training and inference cannot disagree on feature names, order, or math.
* Splits are grouped by REGISTERED DOMAIN (``group``, default) so URLs from one
  site never appear in both train and test. A plain random split lets the model
  memorise domains and inflates scores. ``--split time`` orders by the ``date``
  column and tests on the newest rows (requires a date on every row); any domain
  seen in validation/test is removed from training.
* Three sets: train (fit), validation (provisional threshold / model choice,
  Phase 5), test (touched only by evaluate.py).
* Imbalance: class weights (``balanced``), not resampling. No data-dependent
  preprocessing exists (trees need no scaling), so nothing can leak from the
  validation/test sets into preprocessing.
* Reproducible: fixed ``--seed`` everywhere. Same data + seed + library versions
  gives the same split and model.
* The decision threshold is stored in the metadata as 0.5, a provisional value
  that has NOT been tuned. Scores are uncalibrated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from pipeline_common import (
    DEFAULT_DATA_DIR,
    DEFAULT_MODEL_DIR,
    FEATURE_NAMES,
    PREPARED_NAME,
    REPORT_NAME,
    SPLIT_NAME,
    build_matrix,
    fail,
    load_prepared,
    metrics_from_predictions,
    sha256_file,
)

from app.model import write_model_artifact  # noqa: E402  (backend path added by pipeline_common)

MODEL_KINDS = ("rf", "hgb")
DEFAULT_THRESHOLD = 0.5  # provisional; tuned on validation data in Phase 5
TEST_FRACTION = 0.15
VAL_FRACTION = 0.15


def make_model(kind: str, seed: int):
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample", random_state=seed, n_jobs=-1
        )
    if kind == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, class_weight="balanced", early_stopping=False, random_state=seed
        )
    raise fail(f"unknown model kind {kind!r}; choose from {MODEL_KINDS}.")


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------
def group_split(groups: np.ndarray, y: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """Train/val/test split in which no registered domain spans two sets."""
    idx = np.arange(len(groups))
    outer = GroupShuffleSplit(n_splits=1, test_size=TEST_FRACTION, random_state=seed)
    trainval, test = next(outer.split(idx, y, groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=VAL_FRACTION / (1 - TEST_FRACTION), random_state=seed)
    tr, va = next(inner.split(trainval, y[trainval], groups[trainval]))
    return {"train": trainval[tr], "val": trainval[va], "test": test}


def time_split(dates: pd.Series, groups: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Newest rows -> test, next newest -> validation; domains never cross sets."""
    parsed = pd.to_datetime(dates.replace("", pd.NA), errors="coerce", utc=True)
    if parsed.isna().any():
        raise fail(
            f"--split time needs a valid date on every row; {int(parsed.isna().sum())} rows have none "
            "(typical when legitimate URLs come from an undated ranking list). Use --split group."
        )
    order = np.argsort(parsed.to_numpy(), kind="stable")
    n = len(order)
    n_test, n_val = int(n * TEST_FRACTION), int(n * VAL_FRACTION)
    test, val, train = order[n - n_test:], order[n - n_test - n_val: n - n_test], order[: n - n_test - n_val]

    test_groups = set(groups[test])
    val_keep = np.array([i for i in val if groups[i] not in test_groups], dtype=int)
    held_out = test_groups | set(groups[val_keep])
    train_keep = np.array([i for i in train if groups[i] not in held_out], dtype=int)
    dropped = {"validation": int(len(val) - len(val_keep)), "train": int(len(train) - len(train_keep))}
    return {"train": train_keep, "val": val_keep, "test": test}, dropped


def check_split(split: dict[str, np.ndarray], groups: np.ndarray, y: np.ndarray) -> None:
    names = list(split)
    for i, a in enumerate(names):
        if len(split[a]) == 0:
            raise fail(f"the {a} set is empty; the dataset is too small for this split.")
        if len(set(y[split[a]])) < 2:
            raise fail(f"the {a} set contains only one class; need more data or a different seed/split.")
        for b in names[i + 1:]:
            if set(groups[split[a]]) & set(groups[split[b]]):
                raise AssertionError(f"domain leakage between {a} and {b}")  # defensive: must never happen


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train a PhishGuard model (no external credentials needed).")
    p.add_argument("--model", choices=MODEL_KINDS, default="rf", help="rf = Random Forest baseline (default).")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory with prepared.csv and its report.")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--split", choices=["group", "time"], default="group")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    prepared_path = args.data_dir / PREPARED_NAME
    df = load_prepared(prepared_path)
    report_path = args.data_dir / REPORT_NAME
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    prepared_sha = sha256_file(prepared_path)
    if report and report.get("output", {}).get("sha256") != prepared_sha:
        raise fail("prepared.csv does not match prepared_report.json (file changed after preparation); re-run prepare_dataset.py.")

    X = build_matrix(df)
    y = df["label"].to_numpy(dtype=int)
    groups = df["registered_domain"].to_numpy()

    if args.split == "time":
        if "date" not in df.columns:
            raise fail("--split time needs a 'date' column in prepared.csv.")
        split, dropped = time_split(df["date"].fillna(""), groups)
    else:
        split, dropped = group_split(groups, y, args.seed), {}
    check_split(split, groups, y)

    model = make_model(args.model, args.seed)
    model.fit(X[split["train"]], y[split["train"]])  # fit on TRAIN only

    val_scores = model.predict_proba(X[split["val"]])[:, 1]
    val_pred = (val_scores >= DEFAULT_THRESHOLD).astype(int)
    val_metrics = metrics_from_predictions(y[split["val"]], val_pred)
    val_metrics["roc_auc"] = round(float(roc_auc_score(y[split["val"]], val_scores)), 4)

    dataset_meta = {
        "prepared_sha256": prepared_sha,
        "rows": int(len(df)),
        "class_distribution": report.get("class_distribution", {"phishing": int((y == 1).sum()), "legitimate": int((y == 0).sum())}),
        "sources": report.get("sources", sorted(df["source"].unique().tolist())),
        "synthetic": bool(report.get("synthetic", df["source"].str.lower().str.startswith("synthetic").any())),
        "host_only": bool(report.get("options", {}).get("host_only", False)),
        "provenance_note": report.get("provenance_note"),
        "warnings": report.get("warnings", []),
    }
    meta = write_model_artifact(
        model,
        {
            "model_type": args.model,
            "threshold": DEFAULT_THRESHOLD,
            "threshold_status": "provisional: not tuned on held-out data",
            "calibrated": False,
            "seed": args.seed,
            "hyperparameters": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in model.get_params().items()},
            "dataset": dataset_meta,
            "split": {
                "method": args.split,
                "test_fraction": TEST_FRACTION,
                "validation_fraction": VAL_FRACTION,
                "sizes": {name: int(len(idx)) for name, idx in split.items()},
                "rows_dropped_for_domain_leakage": dropped,
            },
            "validation_metrics": val_metrics,
        },
        args.out_dir,
    )

    # Row indices into prepared.csv, so evaluate.py scores exactly the held-out test set.
    (args.out_dir / SPLIT_NAME).write_text(
        json.dumps(
            {"prepared_sha256": prepared_sha, "method": args.split, "seed": args.seed,
             **{name: [int(i) for i in idx] for name, idx in split.items()}}
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Trained {args.model} -> {args.out_dir}  (version {meta['model_version']})")
    print(f"  split={args.split} sizes={meta['split']['sizes']}")
    print(f"  validation @ threshold {DEFAULT_THRESHOLD}: precision={val_metrics['precision']} recall={val_metrics['recall']} "
          f"f1={val_metrics['f1']} FPR={val_metrics['false_positive_rate']} AUC={val_metrics['roc_auc']}")
    if dataset_meta["synthetic"]:
        print("WARNING: trained on SYNTHETIC data - a demo only. The backend will refuse to load it unless "
              "PHISHGUARD_ALLOW_SYNTHETIC_MODEL=1.")
    for warning in dataset_meta["warnings"]:
        if not warning.startswith("SYNTHETIC"):
            print(f"WARNING (dataset): {warning}")
    print("Next: python training/evaluate.py --model-dir", args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
