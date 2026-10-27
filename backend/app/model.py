"""ML inference: safe loading of a trained model artifact and prediction.

Artifact layout (written by ``write_model_artifact``, used by training/ and tests)::

    <model_dir>/model.joblib       the fitted scikit-learn estimator
    <model_dir>/model_meta.json    version, feature schema, dataset provenance, threshold, SHA-256

Trust model (read this before changing anything)
------------------------------------------------
``joblib``/``pickle`` files can execute arbitrary code when loaded. Therefore:

* The artifact is loaded ONLY from a project-controlled directory (default
  ``backend/models``), never from a path supplied in an API request. Symlinks,
  non-regular files, and oversized files are refused.
* The SHA-256 recorded in ``model_meta.json`` is checked BEFORE unpickling. This
  detects corruption and accidental file swaps. It does NOT prove authenticity,
  because anyone who can replace the model can replace the metadata too. For real
  protection, pin the hash out of band with ``PHISHGUARD_MODEL_SHA256``; then a
  swapped model and metadata both fail the check.
* Only the file's owner should be able to write the model directory.

Honesty rules
-------------
* No model on disk -> ``not_configured``. A model that exists but fails any check
  -> ``unavailable`` with the reason. A prediction is never fabricated.
* The score is a raw, UNCALIBRATED classifier output, not a probability of
  phishing. ``prediction`` is simply ``score >= threshold``; the threshold is
  provisional until tuned on held-out data (Phase 5).
* A model trained on synthetic data is refused unless explicitly allowed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from .features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION, ParsedURL, extract_features, features_to_vector
from .schemas import ComponentInfo, MLResult

logger = logging.getLogger("phishguard.model")

MODEL_FILENAME = "model.joblib"
META_FILENAME = "model_meta.json"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MAX_MODEL_BYTES = 256 * 1024 * 1024
MAX_META_BYTES = 1024 * 1024
REQUIRED_META_KEYS = (
    "model_version", "model_type", "feature_schema_version", "feature_names",
    "sklearn_version", "model_sha256", "threshold", "dataset",
)


class ModelLoadError(Exception):
    """The artifact exists but must not be used. ``str(exc)`` is a safe, short reason."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _minor(version: str) -> str:
    return ".".join(version.split(".")[:2])


# --------------------------------------------------------------------------
# Writing (used by training/train.py and by tests: one definition of the format)
# --------------------------------------------------------------------------
def write_model_artifact(model: Any, metadata: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """Serialize ``model`` and write metadata. Returns the final metadata dict.

    Fills in: model_sha256, model_version, feature_schema_version, feature_names,
    sklearn_version, created_at, calibrated (False unless the caller says otherwise).
    """
    import sklearn

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / MODEL_FILENAME

    fd, tmp_name = tempfile.mkstemp(dir=out_dir, suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        joblib.dump(model, tmp, compress=3)
        os.replace(tmp, model_path)
    finally:
        tmp.unlink(missing_ok=True)

    digest = sha256_file(model_path)
    created = datetime.now(timezone.utc)
    meta = dict(metadata)
    meta.setdefault("model_type", "unknown")
    meta.setdefault("calibrated", False)
    meta.update(
        {
            "created_at": created.isoformat(timespec="seconds"),
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_names": list(FEATURE_NAMES),
            "sklearn_version": sklearn.__version__,
            "model_sha256": digest,
            "model_version": f"{meta['model_type']}-{created:%Y%m%d}-{digest[:8]}",
        }
    )
    (out_dir / META_FILENAME).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return meta


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class LoadedModel:
    estimator: Any
    meta: dict[str, Any]


def _regular_file(path: Path, max_bytes: int, label: str) -> None:
    if path.is_symlink():
        raise ModelLoadError(f"{label} is a symbolic link; refusing to follow it.")
    if not path.is_file():
        raise ModelLoadError(f"{label} is missing or not a regular file.")
    if path.stat().st_size > max_bytes:
        raise ModelLoadError(f"{label} is larger than the allowed maximum.")


def load_model_artifact(
    model_dir: Path, *, allow_synthetic: bool = False, expected_sha256: str | None = None
) -> LoadedModel:
    """Validate and load an artifact; raise ``ModelLoadError`` on any problem."""
    import sklearn

    model_path = model_dir / MODEL_FILENAME
    meta_path = model_dir / META_FILENAME
    _regular_file(meta_path, MAX_META_BYTES, META_FILENAME)
    _regular_file(model_path, MAX_MODEL_BYTES, MODEL_FILENAME)

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ModelLoadError("model_meta.json is not valid JSON.") from None
    if not isinstance(meta, dict):
        raise ModelLoadError("model_meta.json has an unexpected structure.")
    missing = [key for key in REQUIRED_META_KEYS if key not in meta]
    if missing:
        raise ModelLoadError(f"model_meta.json is missing required fields: {', '.join(missing)}.")

    threshold = meta["threshold"]
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 < threshold < 1:
        raise ModelLoadError("model_meta.json has an invalid decision threshold.")
    if not isinstance(meta["dataset"], dict):
        raise ModelLoadError("model_meta.json has invalid dataset information.")

    # Feature compatibility: refuse rather than silently mis-order columns.
    if meta["feature_schema_version"] != FEATURE_SCHEMA_VERSION:
        raise ModelLoadError(
            f"Feature schema mismatch: model uses version {meta['feature_schema_version']}, "
            f"this backend uses {FEATURE_SCHEMA_VERSION}. Retrain the model."
        )
    if list(meta["feature_names"]) != list(FEATURE_NAMES):
        raise ModelLoadError("Feature names or order differ from this backend's. Retrain the model.")

    if meta["dataset"].get("synthetic") and not allow_synthetic:
        raise ModelLoadError(
            "The model was trained on synthetic data and is not allowed for real assessments "
            "(set PHISHGUARD_ALLOW_SYNTHETIC_MODEL=1 for demos only)."
        )

    if _minor(str(meta["sklearn_version"])) != _minor(sklearn.__version__):
        raise ModelLoadError(
            f"Model was trained with scikit-learn {meta['sklearn_version']}; installed is "
            f"{sklearn.__version__}. Retrain the model."
        )

    actual = sha256_file(model_path)
    if actual != meta["model_sha256"]:
        raise ModelLoadError("Model file checksum does not match model_meta.json.")
    if expected_sha256 is not None and actual != expected_sha256:
        raise ModelLoadError("Model file checksum does not match the pinned PHISHGUARD_MODEL_SHA256.")

    # Only now - after every check above - is the file unpickled.
    try:
        estimator = joblib.load(model_path)
    except Exception as exc:  # noqa: BLE001 - corrupt/incompatible pickle
        raise ModelLoadError(f"Model file could not be loaded ({type(exc).__name__}).") from None

    if not hasattr(estimator, "predict_proba"):
        raise ModelLoadError("Loaded object is not a probabilistic classifier.")
    if getattr(estimator, "n_features_in_", None) != len(FEATURE_NAMES):
        raise ModelLoadError("Model expects a different number of features than this backend produces.")
    if list(getattr(estimator, "classes_", [])) != [0, 1]:
        raise ModelLoadError("Model classes are not [0, 1] (legitimate, phishing).")
    if hasattr(estimator, "set_params") and "n_jobs" in estimator.get_params():
        estimator.set_params(n_jobs=1)  # parallel overhead dominates for one URL

    return LoadedModel(estimator=estimator, meta=meta)


# --------------------------------------------------------------------------
# Service used by the API
# --------------------------------------------------------------------------
class ModelService:
    """Holds the loaded model (or the reason there is none) and predicts."""

    def __init__(
        self,
        loaded: LoadedModel | None = None,
        *,
        state: str = "not_configured",
        detail: str = "No trained model is loaded. Train one with training/train.py.",
    ) -> None:
        self._loaded = loaded
        self._state = "available" if loaded else state
        self._detail = detail

    @classmethod
    def from_directory(
        cls,
        model_dir: str | os.PathLike[str] | None = None,
        *,
        allow_synthetic: bool = False,
        expected_sha256: str | None = None,
    ) -> "ModelService":
        """Never raises: problems become ``not_configured`` / ``unavailable`` states."""
        directory = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        model_path, meta_path = directory / MODEL_FILENAME, directory / META_FILENAME
        if not model_path.exists() and not meta_path.exists() and not model_path.is_symlink():
            return cls(state="not_configured")
        try:
            loaded = load_model_artifact(
                directory, allow_synthetic=allow_synthetic, expected_sha256=expected_sha256
            )
        except ModelLoadError as exc:
            logger.warning("model artifact rejected: %s", exc)
            return cls(state="unavailable", detail=f"A model artifact exists but was not loaded: {exc}")
        logger.info("model loaded version=%s", loaded.meta["model_version"])
        return cls(loaded)

    # -- helpers ------------------------------------------------------------
    def _metadata(self) -> dict[str, str]:
        assert self._loaded is not None
        meta = self._loaded.meta
        dataset = meta["dataset"]
        sources = dataset.get("sources")
        return {
            "model_type": str(meta["model_type"]),
            "created_at": str(meta.get("created_at", "unknown")),
            "feature_schema_version": str(meta["feature_schema_version"]),
            "threshold": f"{float(meta['threshold']):.3f}",
            "calibrated": str(bool(meta.get("calibrated", False))).lower(),
            "training_rows": str(dataset.get("rows", "unknown")),
            "training_sources": ", ".join(sources) if isinstance(sources, list) else "unknown",
            "split_method": str(meta.get("split", {}).get("method", "unknown")) if isinstance(meta.get("split"), dict) else "unknown",
            "trained_on_synthetic": str(bool(dataset.get("synthetic", False))).lower(),
        }

    def predict(self, parsed: ParsedURL) -> MLResult:
        if self._loaded is None:
            return MLResult(status=self._state, detail=self._detail)  # type: ignore[arg-type]
        meta = self._loaded.meta
        try:
            vector = features_to_vector(extract_features(parsed))
            score = float(self._loaded.estimator.predict_proba([vector])[0][1])
        except Exception as exc:  # noqa: BLE001 - never take the API down
            logger.warning("prediction failed type=%s", type(exc).__name__)
            return MLResult(status="unavailable", detail="The model failed to produce a prediction.")

        threshold = float(meta["threshold"])
        prediction = "phishing" if score >= threshold else "legitimate"
        detail = (
            f"Raw model score {score:.3f} against a decision threshold of {threshold:.2f}. "
            "The score is an uncalibrated model output, not a probability of phishing; "
            "the threshold is provisional."
        )
        if meta["dataset"].get("synthetic"):
            detail = "DEMO MODEL TRAINED ON SYNTHETIC DATA - NOT A REAL ASSESSMENT. " + detail
        return MLResult(
            status="predicted",
            prediction=prediction,
            score=min(1.0, max(0.0, score)),
            model_version=str(meta["model_version"]),
            metadata=self._metadata(),
            detail=detail,
        )

    def info(self) -> ComponentInfo:
        if self._loaded is None:
            limits = {
                "not_configured": "No trained model is loaded; no ML assessment is made.",
                "unavailable": self._detail,
            }
            return ComponentInfo(name="ml", status=self._state, limitation=limits.get(self._state, self._detail))
        meta = self._loaded.meta
        note = (
            "DEMO model trained on synthetic data. " if meta["dataset"].get("synthetic") else ""
        )
        return ComponentInfo(
            name="ml",
            status="available",
            limitation=(
                f"{note}Model {meta['model_version']}. Scores are uncalibrated and the threshold is "
                "provisional; accuracy on real-world URLs has not been established."
            ),
        )
