"""Dataset preparation, training, and evaluation scripts (training/*.py).

Added beyond the required test list because the pipeline's correctness (labels,
leakage, reproducibility) matters as much as the code that consumes its output.
All data here is tiny, invented, or synthetic.
"""

from __future__ import annotations

import csv
import json
import socket

import joblib
import numpy as np
import pandas as pd
import pytest

import evaluate
import make_synthetic_sample
import pipeline_common
import prepare_dataset
import train
from app.model import ModelService

from .conftest import copy_dir


def write_csv(path, header, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    return path


def prepare(tmp_path, rows, *extra, header=("url", "label")):
    src = write_csv(tmp_path / "in.csv", header, rows)
    out = tmp_path / "out"
    code = prepare_dataset.main(["--input", str(src), "--out-dir", str(out), "--provenance-note", "t", *extra])
    assert code == 0
    return pd.read_csv(out / "prepared.csv"), json.loads((out / "prepared_report.json").read_text()), out


GOOD = [("https://example.com/a", "phishing"), ("https://example.org/b", "legitimate")]


# --------------------------------------------------------------------------
# prepare_dataset: labels and validation
# --------------------------------------------------------------------------
def test_unknown_labels_are_rejected_never_assumed_legitimate(tmp_path):
    rows = GOOD + [("https://third.example.net/", "spam"), ("https://fourth.example.net/", "?")]
    df, report, _ = prepare(tmp_path, rows)
    assert report["rejected"] == {"unknown_label": 2}
    assert len(df) == 2


def test_invalid_urls_are_dropped_with_reasons(tmp_path):
    rows = GOOD + [("ftp://x.example.com/", "phishing"), ("not a url", "legitimate"), ("javascript:alert(1)", "phishing")]
    df, report, _ = prepare(tmp_path, rows)
    assert len(df) == 2
    assert report["rejected"] == {
        "invalid_url:invalid_character": 1,
        "invalid_url:unsupported_scheme": 2,
    }


def test_duplicates_and_normalization_variants_collapse(tmp_path):
    rows = GOOD + [("https://example.com/a", "phishing"), ("HTTPS://EXAMPLE.COM/a", "phishing")]
    df, report, _ = prepare(tmp_path, rows)
    assert len(df) == 2 and report["duplicates_removed"] == 2


def test_url_with_conflicting_labels_is_removed_entirely(tmp_path):
    rows = GOOD + [("https://both.example.com/x", "phishing"), ("https://both.example.com/x", "legitimate")]
    df, report, _ = prepare(tmp_path, rows)
    assert "both.example.com" not in " ".join(df["url"]) and report["conflicting_urls_removed"] == 1


def test_embedded_password_never_reaches_the_prepared_file(tmp_path):
    _, _, out = prepare(tmp_path, GOOD + [("http://user:SECRET99@creds.example.com/", "phishing")])
    assert "SECRET99" not in (out / "prepared.csv").read_text()


def test_output_is_deterministic_byte_for_byte(tmp_path):
    src = write_csv(tmp_path / "in.csv", ("url", "label"), GOOD + [("https://zzz.example.com/", "phishing")])
    for name in ("one", "two"):
        prepare_dataset.main(["--input", str(src), "--out-dir", str(tmp_path / name), "--provenance-note", "t"])
    assert (tmp_path / "one" / "prepared.csv").read_bytes() == (tmp_path / "two" / "prepared.csv").read_bytes()


def test_custom_label_map(tmp_path):
    df, _, _ = prepare(tmp_path, [("https://a.example.com/", "spam"), ("https://b.example.com/", "ham")], "--label-map", "spam=1,ham=0")
    assert sorted(df["label"]) == [0, 1]


@pytest.mark.parametrize("bad", ["spam=2", "spam", "spam=1", "=1,x=0"])
def test_invalid_label_map_is_refused(tmp_path, bad):
    src = write_csv(tmp_path / "in.csv", ("url", "label"), GOOD)
    with pytest.raises(SystemExit):
        prepare_dataset.main(["--input", str(src), "--out-dir", str(tmp_path / "o"), "--label-map", bad])


def test_single_class_result_is_refused(tmp_path):
    src = write_csv(tmp_path / "in.csv", ("url", "label"), [("https://a.example.com/", "phishing"), ("https://b.example.com/", "phishing")])
    with pytest.raises(SystemExit):
        prepare_dataset.main(["--input", str(src), "--out-dir", str(tmp_path / "o")])


def test_missing_required_column_is_refused(tmp_path):
    src = write_csv(tmp_path / "in.csv", ("link", "label"), GOOD)
    with pytest.raises(SystemExit):
        prepare_dataset.main(["--input", str(src), "--out-dir", str(tmp_path / "o")])


def test_preparation_makes_no_network_connections(tmp_path, monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("network access during dataset preparation")

    monkeypatch.setattr(socket.socket, "connect", boom)
    prepare(tmp_path, GOOD)


# --------------------------------------------------------------------------
# prepare_dataset: separate-file mode (PhishTank / Tranco style)
# --------------------------------------------------------------------------
def feed_files(tmp_path):
    phish = write_csv(
        tmp_path / "phish.csv",
        ("phish_id", "url", "submission_time", "verified", "online"),
        [
            (1, "http://bad-one.example.xyz/login/a.php?x=1", "2026-01-01T00:00:00+00:00", "yes", "yes"),
            (2, "http://bad-two.example.xyz/verify/b.php", "2026-01-02T00:00:00+00:00", "yes", "yes"),
            (3, "http://bad-three.example.xyz/c/d/e.html", "2026-01-03T00:00:00+00:00", "yes", "yes"),
            (4, "http://unverified.example.xyz/z", "2026-01-04T00:00:00+00:00", "no", "yes"),
        ],
    )
    legit = tmp_path / "tranco.csv"
    legit.write_text("1,example.com\n2,wikipedia.org\n3,github.com\n")
    return phish, legit


def test_phishtank_and_tranco_mode(tmp_path):
    phish, legit = feed_files(tmp_path)
    out = tmp_path / "out"
    prepare_dataset.main(["--phishing", str(phish), "--legit", str(legit), "--legit-limit", "2", "--out-dir", str(out), "--provenance-note", "t"])
    df = pd.read_csv(out / "prepared.csv")
    report = json.loads((out / "prepared_report.json").read_text())
    assert report["class_distribution"] == {"phishing": 3, "legitimate": 2}
    assert report["unverified_phishing_dropped"] == 1
    assert "unverified.example.xyz" not in " ".join(df["url"])
    assert set(df[df.label == 0]["url"]) == {"https://example.com/", "https://wikipedia.org/"}
    assert all(len(i["sha256"]) == 64 for i in report["inputs"])
    assert report["has_dates"] is True


def test_path_shortcut_is_warned_about_and_host_only_removes_it(tmp_path):
    phish, legit = feed_files(tmp_path)
    common = ["--phishing", str(phish), "--legit", str(legit), "--provenance-note", "t"]
    prepare_dataset.main([*common, "--out-dir", str(tmp_path / "a")])
    warned = json.loads((tmp_path / "a" / "prepared_report.json").read_text())["warnings"]
    assert any("PATH SHORTCUT" in w for w in warned)

    prepare_dataset.main([*common, "--host-only", "--out-dir", str(tmp_path / "b")])
    report = json.loads((tmp_path / "b" / "prepared_report.json").read_text())
    assert not any("PATH SHORTCUT" in w for w in report["warnings"])
    assert all(url.endswith("/") and url.count("/") == 3 for url in pd.read_csv(tmp_path / "b" / "prepared.csv")["url"])


def test_missing_provenance_note_is_flagged(tmp_path):
    src = write_csv(tmp_path / "in.csv", ("url", "label"), GOOD)
    prepare_dataset.main(["--input", str(src), "--out-dir", str(tmp_path / "o")])
    warnings = json.loads((tmp_path / "o" / "prepared_report.json").read_text())["warnings"]
    assert any("provenance-note" in w for w in warnings)


def test_synthetic_source_is_flagged_in_the_report(pipeline):
    report = json.loads((pipeline.data / "prepared_report.json").read_text())
    assert report["synthetic"] is True and report["warnings"][0].startswith("SYNTHETIC")


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def test_metrics_known_values():
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 0, 1, 0, 0, 0, 0, 0])
    m = pipeline_common.metrics_from_predictions(y_true, y_pred)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (3, 1, 5, 1)
    assert m["precision"] == 0.75 and m["recall"] == 0.75 and m["f1"] == 0.75
    assert m["false_positive_rate"] == round(1 / 6, 4) and m["false_negatives"] == 1
    assert m["low_support"] is True  # only 10 rows: below the 30-row support threshold

    big = pipeline_common.metrics_from_predictions(np.tile(y_true, 5), np.tile(y_pred, 5))
    assert big["n"] == 50 and big["low_support"] is False and big["precision"] == 0.75


def test_undefined_metrics_are_none_not_zero():
    m = pipeline_common.metrics_from_predictions(np.array([0, 0, 0]), np.array([0, 0, 0]))
    assert m["precision"] is None and m["recall"] is None and m["f1"] is None
    assert m["false_positive_rate"] == 0.0
    assert m["low_support"] is True


# --------------------------------------------------------------------------
# train.py
# --------------------------------------------------------------------------
def split_of(model_dir):
    return json.loads((model_dir / "split.json").read_text())


def test_group_split_never_shares_a_registered_domain(pipeline):
    df = pd.read_csv(pipeline.data / "prepared.csv")
    split = split_of(pipeline.rf)
    domains = {name: set(df.iloc[split[name]]["registered_domain"]) for name in ("train", "val", "test")}
    assert not (domains["train"] & domains["val"]) and not (domains["train"] & domains["test"]) and not (domains["val"] & domains["test"])
    assert sum(len(split[n]) for n in ("train", "val", "test")) == len(df)  # every row used exactly once


def test_every_split_contains_both_classes(pipeline):
    df = pd.read_csv(pipeline.data / "prepared.csv")
    split = split_of(pipeline.rf)
    for name in ("train", "val", "test"):
        assert set(df.iloc[split[name]]["label"]) == {0, 1}


def test_metadata_records_provenance_and_honest_status(pipeline):
    meta = json.loads((pipeline.rf / "model_meta.json").read_text())
    report = json.loads((pipeline.data / "prepared_report.json").read_text())
    assert meta["dataset"]["prepared_sha256"] == report["output"]["sha256"]
    assert meta["calibrated"] is False and meta["threshold"] == 0.5 and "provisional" in meta["threshold_status"]
    assert meta["model_type"] == "rf" and meta["split"]["method"] == "group"
    assert meta["feature_names"] and meta["validation_metrics"]["n"] == meta["split"]["sizes"]["val"]
    assert meta["dataset"]["synthetic"] is True


def test_training_is_reproducible(pipeline, tmp_path):
    again = tmp_path / "again"
    train.main(["--model", "rf", "--data-dir", str(pipeline.data), "--out-dir", str(again)])
    assert split_of(again) == split_of(pipeline.rf)
    X = pipeline_common.build_matrix(pd.read_csv(pipeline.data / "prepared.csv").head(50))
    a, b = joblib.load(pipeline.rf / "model.joblib"), joblib.load(again / "model.joblib")
    assert np.array_equal(a.predict_proba(X), b.predict_proba(X))


def test_a_different_seed_gives_a_different_split(pipeline, tmp_path):
    other = tmp_path / "other"
    train.main(["--model", "rf", "--seed", "7", "--data-dir", str(pipeline.data), "--out-dir", str(other)])
    assert split_of(other)["test"] != split_of(pipeline.rf)["test"]


def test_trained_artifact_loads_in_the_backend(pipeline):
    assert ModelService.from_directory(pipeline.rf, allow_synthetic=True).info().status == "available"
    assert ModelService.from_directory(pipeline.hgb, allow_synthetic=True).info().status == "available"


def test_train_refuses_a_prepared_file_changed_after_preparation(pipeline, tmp_path):
    data = copy_dir(pipeline.data, tmp_path, "data")
    with open(data / "prepared.csv", "a") as handle:
        handle.write("https://sneaky.example.com/,1,synthetic,example.com\n")
    with pytest.raises(SystemExit):
        train.main(["--data-dir", str(data), "--out-dir", str(tmp_path / "m")])


def test_unknown_model_kind_is_refused(pipeline, tmp_path):
    with pytest.raises(SystemExit):
        train.make_model("nope", 1)


def test_time_split_without_dates_is_refused(pipeline, tmp_path):
    with pytest.raises(SystemExit):
        train.main(["--split", "time", "--data-dir", str(pipeline.data), "--out-dir", str(tmp_path / "m")])


def test_time_split_orders_by_date_and_keeps_domains_apart(tmp_path):
    raw = tmp_path / "syn.csv"
    make_synthetic_sample.main(["--n-per-class", "200", "--out", str(raw)])
    df = pd.read_csv(raw)
    rng = np.random.default_rng(0)
    df["date"] = pd.to_datetime("2026-01-01") + pd.to_timedelta(rng.permutation(len(df)), unit="h")
    dated = tmp_path / "dated.csv"
    df.to_csv(dated, index=False)

    data, out = tmp_path / "data", tmp_path / "m"
    prepare_dataset.main(["--input", str(dated), "--out-dir", str(data), "--provenance-note", "t"])
    train.main(["--split", "time", "--data-dir", str(data), "--out-dir", str(out)])

    prepared = pd.read_csv(data / "prepared.csv")
    dates = pd.to_datetime(prepared["date"])
    split = split_of(out)
    assert dates.iloc[split["train"]].max() <= dates.iloc[split["val"]].min()
    assert dates.iloc[split["val"]].max() <= dates.iloc[split["test"]].min()
    domains = {n: set(prepared.iloc[split[n]]["registered_domain"]) for n in ("train", "val", "test")}
    assert not (domains["train"] & domains["test"]) and not (domains["val"] & domains["test"]) and not (domains["train"] & domains["val"])


# --------------------------------------------------------------------------
# evaluate.py
# --------------------------------------------------------------------------
def run_evaluate(model_dir, data_dir, report):
    return evaluate.main(["--model-dir", str(model_dir), "--data-dir", str(data_dir), "--report", str(report)])


def test_evaluate_scores_only_the_held_out_test_set(pipeline, tmp_path, capsys):
    report_path = tmp_path / "eval.json"
    assert run_evaluate(pipeline.rf, pipeline.data, report_path) == 0
    report = json.loads(report_path.read_text())
    assert report["test_rows"] == len(split_of(pipeline.rf)["test"])
    for key in ("precision", "recall", "f1", "tp", "fp", "tn", "fn", "false_positive_rate", "false_negatives", "roc_auc"):
        assert key in report["ml"]
    assert "rules_baseline" in report and "scheme" in report["by_category"]
    assert report["synthetic_data"] is True and report["threshold_status"] == "provisional"
    out = capsys.readouterr().out
    assert "SYNTHETIC DATA" in out and "False negative" in out and "confusion matrix" in out


def test_evaluate_refuses_a_dataset_different_from_training(pipeline, tmp_path):
    data = copy_dir(pipeline.data, tmp_path, "data")
    with open(data / "prepared.csv", "a") as handle:
        handle.write("https://sneaky.example.com/,1,synthetic,example.com\n")
    with pytest.raises(SystemExit):
        run_evaluate(pipeline.rf, data, tmp_path / "e.json")


def test_evaluate_refuses_a_tampered_model(pipeline, tmp_path):
    model = copy_dir(pipeline.rf, tmp_path, "model")
    with open(model / "model.joblib", "ab") as handle:
        handle.write(b"x")
    with pytest.raises(SystemExit):
        run_evaluate(model, pipeline.data, tmp_path / "e.json")


def test_evaluate_without_a_model_gives_a_clear_error(pipeline, tmp_path):
    with pytest.raises(SystemExit):
        run_evaluate(tmp_path / "nothing", pipeline.data, tmp_path / "e.json")
