"""FastAPI application: HTTP layer only.

This module wires the pipeline together and owns cross-cutting concerns (limits,
errors, logging). It contains NO detection logic:

    parse_url -> evaluate_rules -> ModelService.predict -> ReputationService.check -> decide

Security notes
--------------
* The submitted URL is only ever *parsed as text*. The backend never connects to it.
* /scan is POST with a JSON body so URLs never appear in the request line, and
  therefore never in server access logs.
* Our own log lines record a short SHA-256 reference, never the URL. (The hash
  lets you correlate log lines; it is NOT anonymization, since common URLs can
  be guessed and hashed.)
* Error responses never echo the submitted URL.
* Bind to 127.0.0.1 (the documented default). Do not expose this API publicly.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import __version__
from .config import Settings
from .decision import decide
from .features import ParsedURL, URLValidationError, parse_url
from .middleware import BodySizeLimitMiddleware
from .model import ModelService
from .ratelimit import SlidingWindowRateLimiter
from .reputation import ReputationService
from .rules import evaluate_rules
from .schemas import (
    ApiInfo,
    ComponentInfo,
    ErrorBody,
    ErrorResponse,
    HealthResponse,
    MLResult,
    ReputationResult,
    ScanRequest,
    ScanResponse,
)

logger = logging.getLogger("phishguard")

_STATUS_CODES = {
    400: "bad_request",
    404: "not_found",
    405: "method_not_allowed",
    413: "payload_too_large",
    422: "validation_error",
    429: "rate_limited",
}


def _configure_logging(level: str) -> None:
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)


def _error(
    status: int,
    code: str,
    message: str,
    details: list[dict[str, str]] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    body = ErrorResponse(error=ErrorBody(code=code, message=message, details=details))
    return JSONResponse(status_code=status, content=body.model_dump(exclude_none=True), headers=headers)


def _display_url(parsed: ParsedURL) -> str:
    """Scheme + ASCII host (+ port). Path, query, fragment, and userinfo are dropped."""
    host = f"[{parsed.host}]" if parsed.is_ip and ":" in parsed.host else parsed.host
    port = f":{parsed.port}" if parsed.port is not None else ""
    has_more = parsed.path not in ("", "/") or parsed.query or parsed.fragment
    return f"{parsed.scheme}://{host}{port}{'/...' if has_more else ''}"


def _url_ref(parsed: ParsedURL) -> str:
    return hashlib.sha256(parsed.normalized.encode()).hexdigest()[:12]


def _enforce_rate_limit(request: Request) -> None:
    client = request.client.host if request.client else "unknown"
    allowed, retry_after = request.app.state.rate_limiter.check(client)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please slow down.",
            headers={"Retry-After": str(retry_after)},
        )


def create_app(
    settings: Settings | None = None,
    ml_service: ModelService | None = None,
    reputation_service: ReputationService | None = None,
) -> FastAPI:
    """Application factory. Injecting services keeps the app easy to test."""
    settings = settings or Settings.from_env()
    _configure_logging(settings.log_level)

    app = FastAPI(
        title="PhishGuard API",
        version=__version__,
        description=(
            "Hybrid phishing URL assessment (rules + ML + threat intelligence). "
            "Returns a risk assessment, never a guarantee of safety. "
            "The backend never visits submitted URLs."
        ),
    )
    app.state.settings = settings
    app.state.ml_service = ml_service or ModelService()
    app.state.reputation_service = reputation_service or ReputationService()
    app.state.rate_limiter = SlidingWindowRateLimiter(
        settings.rate_limit_per_minute, 60.0, settings.rate_limit_max_clients
    )

    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_body_bytes)
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )

    # ---- Error handlers: one consistent shape for every failure ----------
    @app.exception_handler(URLValidationError)
    async def _invalid_url(_: Request, exc: URLValidationError) -> JSONResponse:
        logger.info("scan rejected reason=%s", exc.code)
        return _error(422, "invalid_url", exc.message, details=[{"reason": exc.code}])

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        # Deliberately omit err["input"] so the submitted value is never echoed.
        details = [
            {
                "field": ".".join(str(part) for part in err["loc"] if part != "body") or "body",
                "message": str(err["msg"]),
            }
            for err in exc.errors()
        ]
        return _error(422, "validation_error", "Request validation failed.", details=details)

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = _STATUS_CODES.get(exc.status_code, "http_error")
        return _error(exc.status_code, code, str(exc.detail), headers=getattr(exc, "headers", None))

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception) -> JSONResponse:
        logger.error("unhandled error type=%s", type(exc).__name__)  # no message: may contain input
        return _error(500, "internal_error", "An internal error occurred.")

    # ---- Endpoints ------------------------------------------------------
    @app.get("/", response_model=ApiInfo, summary="API information")
    def root() -> ApiInfo:
        return ApiInfo(
            name="PhishGuard",
            version=__version__,
            docs_url="/docs",
            endpoints=["GET /", "GET /health", "POST /scan"],
        )

    @app.get("/health", response_model=HealthResponse, summary="Health and readiness")
    def health(request: Request) -> HealthResponse:
        state = request.app.state
        components: list[ComponentInfo] = [
            ComponentInfo(
                name="rules",
                status="available",
                limitation="Heuristics over URL text only.",
            ),
            state.ml_service.info(),
            state.reputation_service.info(),
        ]
        return HealthResponse(status="ok", version=__version__, components=components)

    @app.post(
        "/scan",
        response_model=ScanResponse,
        summary="Assess a URL",
        dependencies=[Depends(_enforce_rate_limit)],
    )
    def scan(payload: ScanRequest, request: Request) -> ScanResponse:
        state = request.app.state
        cfg: Settings = state.settings

        parsed = parse_url(payload.url, max_length=cfg.max_url_length)  # may raise URLValidationError
        rules_result = evaluate_rules(parsed)

        # A failing component is reported as "unavailable" - never as a clean result.
        ml_result = _run_component(
            lambda: state.ml_service.predict(parsed),
            MLResult(status="unavailable", detail="The ML component failed; no prediction was made."),
        )
        rep_result = _run_component(
            lambda: state.reputation_service.check(parsed),
            ReputationResult(
                status="unavailable",
                detail="The reputation lookup failed; whether the URL is listed is unknown.",
            ),
        )

        decision = decide(
            rules_result,
            ml_result,
            rep_result,
            suspicious_threshold=cfg.suspicious_threshold,
            high_risk_threshold=cfg.high_risk_threshold,
        )

        logger.info(
            "scan ref=%s verdict=%s risk=%s rule_score=%d rules=%s ml=%s rep=%s",
            _url_ref(parsed),
            decision.verdict.value,
            decision.risk_level.value,
            rules_result.score,
            ",".join(f.rule_id for f in rules_result.findings) or "-",
            ml_result.status,
            rep_result.status,
        )

        return ScanResponse(
            url_display=_display_url(parsed),
            host=parsed.host,
            host_unicode=parsed.host_unicode if parsed.host_unicode != parsed.host else None,
            verdict=decision.verdict,
            risk_level=decision.risk_level,
            rules=rules_result,
            ml=ml_result,
            reputation=rep_result,
            explanation=decision.explanation,
            components=decision.components,
        )

    return app


def _run_component(call: Any, fallback: Any) -> Any:
    """Run an optional detection component; on any error return ``fallback``."""
    try:
        return call()
    except Exception as exc:  # noqa: BLE001 - a component must never take the API down
        logger.warning("component failed type=%s", type(exc).__name__)
        return fallback


app = create_app()
