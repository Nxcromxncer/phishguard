"""Prepare a labelled URL dataset for training.

Reads URL lists from disk (it NEVER visits, downloads, or opens any URL), validates
every URL with the backend's own parser, removes invalid and duplicate records,
and writes ``prepared.csv`` plus ``prepared_report.json`` (provenance, statistics,
and warnings about leakage and collection artifacts).

Input formats (pick ONE of the two modes)
-----------------------------------------
1. Combined CSV:      --input data.csv
     required columns: ``url``, ``label``      optional: ``source``, ``date``
     labels are mapped EXPLICITLY (see DEFAULT_LABEL_MAP or --label-map);
     any other value is rejected, never guessed as "legitimate".

2. Separate files:    --phishing P --legit L
     --phishing-format phishtank   CSV with a ``url`` column (PhishTank
                                   verified_online.csv style). Optional columns:
                                   ``submission_time`` (-> date), ``verified``
                                   (rows not equal to "yes" are dropped).
     --phishing-format urls        text file, one URL per line ('#' comments ok).
     --legit-format tranco         headerless CSV ``rank,domain`` (Tranco list).
                                   Each domain becomes ``https://<domain>/``.
     --legit-format urls           text file, one URL per line.

Why the report warns about paths and schemes
--------------------------------------------
Phishing feeds contain full URLs with paths; ranking lists contain bare domains.
A model can then "learn" that *having a path* means phishing - a collection
artifact that scores brilliantly and fails in the real world. The report measures
this and warns. ``--host-only`` removes the shortcut by reducing every URL to
``scheme://host/`` (at the cost of discarding path information).

Licensing: you are responsible for the license/terms of the data you supply. Record
them with --provenance-note; the report stores it next to the file hashes.

Usage::

    python training/prepare_dataset.py --input my.csv
    python training/prepare_dataset.py --phishing verified_online.csv --legit top-1m.csv --legit-limit 50000
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline_common import (
    DEFAULT_DATA_DIR,
    PREPARED_NAME,
    REPORT_NAME,
    URLValidationError,
    fail,
    parse_url,
    sha256_file,
)

DEFAULT_LABEL_MAP = {
    "phishing": 1, "phish": 1, "malicious": 1, "bad": 1, "1": 1,
    "legitimate": 0, "legit": 0, "benign": 0, "good": 0, "0": 0,
}
MAX_URL_LENGTH = 2048
DISPARITY_WARNING = 0.30
MIN_CLASS_ROWS_WARNING = 100


# --------------------------------------------------------------------------
# Readers: each returns a DataFrame with columns url, label_raw, source, date
# --------------------------------------------------------------------------
def _frame(urls: list[str], label: str, source: str, dates: list[str | None] | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "url": urls,
            "label_raw": label,
            "source": source,
            "date": dates if dates is not None else [None] * len(urls),
        }
    )


def read_url_lines(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


def read_combined(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for column in ("url", "label"):
        if column not in df.columns:
            raise fail(f"{path.name}: required column '{column}' not found (columns: {list(df.columns)}).")
    out = pd.DataFrame(
        {
            "url": df["url"],
            "label_raw": df["label"],
            "source": df["source"] if "source" in df.columns else f"file:{path.name}",
            "date": df["date"] if "date" in df.columns else None,
        }
    )
    return out


def read_phishing(path: Path, fmt: str, stats: dict[str, Any]) -> pd.DataFrame:
    if fmt == "urls":
        return _frame(read_url_lines(path), "phishing", f"phishing:{path.name}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "url" not in df.columns:
        raise fail(f"{path.name}: a 'url' column is required for --phishing-format phishtank.")
    if "verified" in df.columns:
        keep = df["verified"].str.strip().str.lower() == "yes"
        stats["unverified_phishing_dropped"] = int((~keep).sum())
        df = df[keep]
    dates = df["submission_time"].tolist() if "submission_time" in df.columns else None
    return _frame(df["url"].tolist(), "phishing", f"phishing:{path.name}", dates)


def read_legit(path: Path, fmt: str, limit: int | None) -> pd.DataFrame:
    source = f"legit:{path.name}"
    if fmt == "urls":
        urls = read_url_lines(path)
        return _frame(urls[:limit] if limit else urls, "legitimate", source)
    df = pd.read_csv(path, header=None, names=["rank", "domain"], dtype=str, keep_default_na=False, usecols=[0, 1])
    df["rank_num"] = pd.to_numeric(df["rank"], errors="coerce")
    df = df.dropna(subset=["rank_num"]).sort_values("rank_num")
    if limit:
        df = df.head(limit)
    return _frame([f"https://{d.strip()}/" for d in df["domain"]], "legitimate", source)


def parse_label_map(text: str | None) -> dict[str, int]:
    if not text:
        return dict(DEFAULT_LABEL_MAP)
    mapping: dict[str, int] = {}
    for pair in text.split(","):
        key, sep, value = pair.partition("=")
        if not sep or value.strip() not in {"0", "1"} or not key.strip():
            raise fail(f"invalid --label-map entry {pair!r}; use name=1 (phishing) or name=0 (legitimate).")
        mapping[key.strip().lower()] = int(value)
    if set(mapping.values()) != {0, 1}:
        raise fail("--label-map must define at least one phishing (=1) and one legitimate (=0) label.")
    return mapping


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
def clean(df: pd.DataFrame, label_map: dict[str, int], host_only: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Map labels, validate URLs, drop invalid/duplicate/conflicting records."""
    rejected: Counter[str] = Counter()
    records: list[dict[str, Any]] = []

    for url, label_raw, source, date in zip(df["url"], df["label_raw"], df["source"], df["date"]):
        label = label_map.get(str(label_raw).strip().lower())
        if label is None:
            rejected["unknown_label"] += 1  # never guessed as legitimate
            continue
        try:
            parsed = parse_url(str(url), max_length=MAX_URL_LENGTH)
        except URLValidationError as exc:
            rejected[f"invalid_url:{exc.code}"] += 1
            continue
        normalized = f"{parsed.scheme}://{parsed.host}/" if host_only else parsed.normalized
        records.append(
            {
                "url": normalized,
                "label": label,
                "source": str(source),
                "registered_domain": parsed.registered_domain,
                "date": date,
            }
        )

    out = pd.DataFrame(records, columns=["url", "label", "source", "registered_domain", "date"])
    stats: dict[str, Any] = {"rejected": dict(sorted(rejected.items())), "valid_before_dedup": len(out)}

    # Same URL with both labels: the label is unreliable, so drop every copy.
    label_counts = out.groupby("url")["label"].nunique()
    conflicting = set(label_counts[label_counts > 1].index)
    stats["conflicting_urls_removed"] = len(conflicting)
    out = out[~out["url"].isin(conflicting)]

    before = len(out)
    out = out.drop_duplicates(subset="url", keep="first")
    stats["duplicates_removed"] = before - len(out)

    # Dates: normalise to ISO; keep the column only if at least one date parsed.
    parsed_dates = pd.to_datetime(out["date"], errors="coerce", utc=True)
    out = out.drop(columns=["date"])
    if parsed_dates.notna().any():
        out["date"] = parsed_dates.dt.strftime("%Y-%m-%dT%H:%M:%SZ").fillna("")
    stats["rows_with_date"] = int(parsed_dates.notna().sum())

    out = out.sort_values("url", kind="stable").reset_index(drop=True)  # deterministic output
    return out, stats


def _has_path(url: str) -> bool:
    parsed = parse_url(url)
    return parsed.path not in ("", "/") or bool(parsed.query)


def diagnostics(df: pd.DataFrame, host_only: bool) -> tuple[dict[str, Any], list[str]]:
    """Class balance and collection-artifact checks. Returns (stats, warnings)."""
    warnings: list[str] = []
    counts = {int(k): int(v) for k, v in df["label"].value_counts().items()}
    phish, legit = counts.get(1, 0), counts.get(0, 0)

    domains_by_class = df.groupby("label")["registered_domain"].agg(set)
    both = (
        len(domains_by_class.get(0, set()) & domains_by_class.get(1, set()))
        if {0, 1} <= set(domains_by_class.index)
        else 0
    )

    path_rate, https_rate = {}, {}
    for label, name in ((1, "phishing"), (0, "legitimate")):
        part = df[df["label"] == label]
        if len(part):
            path_rate[name] = round(float(part["url"].map(_has_path).mean()), 4)
            https_rate[name] = round(float(part["url"].str.startswith("https://").mean()), 4)

    if len(path_rate) == 2 and not host_only and abs(path_rate["phishing"] - path_rate["legitimate"]) > DISPARITY_WARNING:
        warnings.append(
            f"PATH SHORTCUT RISK: {path_rate['phishing']:.0%} of phishing URLs have a path/query vs "
            f"{path_rate['legitimate']:.0%} of legitimate ones. A model can score well merely by detecting "
            "'has a path'. Evaluation numbers will overstate real-world performance. Consider --host-only "
            "or a legitimate-URL source that includes realistic paths."
        )
    if len(https_rate) == 2 and abs(https_rate["phishing"] - https_rate["legitimate"]) > DISPARITY_WARNING:
        warnings.append(
            f"SCHEME SKEW: {https_rate['phishing']:.0%} of phishing URLs are https vs "
            f"{https_rate['legitimate']:.0%} of legitimate ones. Most real phishing uses https today; "
            "a model leaning on the scheme will not generalize."
        )
    if min(phish, legit) < MIN_CLASS_ROWS_WARNING:
        warnings.append(f"A class has fewer than {MIN_CLASS_ROWS_WARNING} rows; results will be unreliable.")
    elif min(phish, legit) / max(phish, legit) < 0.05:
        warnings.append("Severe class imbalance (<5% minority). Train with class weights and judge by recall/FPR, not accuracy.")
    if both:
        warnings.append(f"{both} registered domain(s) appear in BOTH classes (e.g. abused hosting). Group-aware splits keep them together.")

    stats = {
        "class_distribution": {"phishing": phish, "legitimate": legit},
        "unique_registered_domains": int(df["registered_domain"].nunique()),
        "registered_domains_in_both_classes": both,
        "path_or_query_rate_by_class": path_rate,
        "https_rate_by_class": https_rate,
    }
    return stats, warnings


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare a labelled URL dataset (reads files only; never visits URLs).")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--input", type=Path, help="Combined CSV with 'url' and 'label' columns.")
    mode.add_argument("--phishing", type=Path, help="Phishing URL file (use with --legit).")
    p.add_argument("--legit", type=Path, help="Legitimate URL/domain file (use with --phishing).")
    p.add_argument("--phishing-format", choices=["phishtank", "urls"], default="phishtank")
    p.add_argument("--legit-format", choices=["tranco", "urls"], default="tranco")
    p.add_argument("--legit-limit", type=int, default=None, help="Keep only the top N legitimate entries.")
    p.add_argument("--label-map", default=None, help="Explicit label mapping, e.g. 'spam=1,ham=0'.")
    p.add_argument("--host-only", action="store_true", help="Reduce every URL to scheme://host/ (removes the path shortcut).")
    p.add_argument("--provenance-note", default=None, help="Free text: where the data came from and its license/terms.")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_DATA_DIR)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.phishing and not args.legit:
        raise fail("--legit is required together with --phishing.")
    if args.legit_limit is not None and args.legit_limit < 1:
        raise fail("--legit-limit must be >= 1.")

    label_map = parse_label_map(args.label_map)
    reader_stats: dict[str, Any] = {}
    inputs: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []

    if args.input:
        sources = [(args.input, "combined")]
        frames.append(read_combined(args.input))
    else:
        sources = [(args.phishing, "phishing"), (args.legit, "legitimate")]
        frames.append(read_phishing(args.phishing, args.phishing_format, reader_stats))
        frames.append(read_legit(args.legit, args.legit_format, args.legit_limit))
    for (path, role), frame in zip(sources, frames):
        if not Path(path).is_file():
            raise fail(f"input file not found: {path}")
        inputs.append({"file": Path(path).name, "role": role, "sha256": sha256_file(path), "rows_read": int(len(frame))})

    raw = pd.concat(frames, ignore_index=True)
    df, clean_stats = clean(raw, label_map, args.host_only)

    if df.empty or df["label"].nunique() < 2:
        raise fail("after cleaning, the dataset is empty or contains only one class. See the rejection counts above.")

    diag, warnings = diagnostics(df, args.host_only)
    synthetic = bool(df["source"].str.lower().str.startswith("synthetic").any())
    if synthetic:
        warnings.insert(0, "SYNTHETIC DATA: for pipeline testing only. Any model trained on it is a demo, not a detector.")
    if not args.provenance_note:
        warnings.append("No --provenance-note given: record where the data came from and its license/terms.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / PREPARED_NAME
    df.to_csv(out_path, index=False, lineterminator="\n")

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": inputs,
        "provenance_note": args.provenance_note,
        "options": {
            "host_only": args.host_only,
            "legit_limit": args.legit_limit,
            "label_map": label_map,
            "max_url_length": MAX_URL_LENGTH,
            "phishing_format": args.phishing_format if args.phishing else None,
            "legit_format": args.legit_format if args.phishing else None,
        },
        "rows_read": int(len(raw)),
        **reader_stats,
        **clean_stats,
        "final_rows": int(len(df)),
        "sources": sorted(df["source"].unique().tolist()),
        "synthetic": synthetic,
        "has_dates": "date" in df.columns,
        **diag,
        "warnings": warnings,
        "output": {"file": out_path.name, "sha256": sha256_file(out_path)},
    }
    (args.out_dir / REPORT_NAME).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Read {report['rows_read']} rows -> kept {report['final_rows']}")
    print(f"  rejected: {report['rejected'] or 'none'}")
    print(f"  duplicates removed: {report['duplicates_removed']}, conflicting URLs removed: {report['conflicting_urls_removed']}")
    print(f"  classes: {diag['class_distribution']}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    print(f"Wrote {out_path} and {args.out_dir / REPORT_NAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
