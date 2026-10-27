"""ML inference interface.

*** PHASE 1 PLACEHOLDER ***
No model exists yet. Training, artifact loading, and feature-schema validation are
implemented in Phase 3. Until then this service reports ``not_configured`` and
NEVER invents a prediction.

The interface (``ModelService.predict`` / ``ModelService.info``) is what main.py
depends on, so Phase 3 can replace the body without touching the API layer.
"""

from __future__ import annotations

from .features import ParsedURL
from .schemas import ComponentInfo, MLResult


class ModelService:
    """Phase 1: always reports that no model is configured."""

    def predict(self, parsed: ParsedURL) -> MLResult:
        return MLResult(
            status="not_configured",
            detail="No trained model is loaded. ML inference is implemented in Phase 3.",
        )

    def info(self) -> ComponentInfo:
        return ComponentInfo(
            name="ml",
            status="not_configured",
            limitation="No trained model is loaded; no ML assessment is made.",
        )
