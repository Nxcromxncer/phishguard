# PhishGuard

A hybrid phishing-URL assessment system: **rule-based heuristics + a machine-learning classifier + threat-intelligence lookup**, combined by an interpretable decision engine, served by a FastAPI backend and (planned) a Chromium Manifest V3 extension.

> **Status: Phase 1 of 6 complete.** The backend, URL parser, rule engine, decision engine and API work and are tested. The extension, ML pipeline and threat-intelligence provider are **not built yet**; the API reports them honestly as `not_configured`. See [Verified functionality](#verified-functionality-phase-1) and [Roadmap](#roadmap).

PhishGuard returns a **risk assessment, never a guarantee of safety.** It has no verdict that means "safe".

---

## Architecture

```
POST /scan {"url": "..."}
      │
      ▼
 ┌─────────────┐   body-size limit (413) · rate limit (429) · schema validation (422)
 │   main.py   │
 └─────┬───────┘
       ▼
 features.py ── parse_url(): strict validation, IDNA/punycode, public-suffix split   (no network)
       │
       ├──► rules.py       weak / moderate / strong findings with evidence           [Phase 1 ✔]
       ├──► model.py       Random Forest inference                                   [Phase 3 – placeholder]
       └──► reputation.py  Safe Browsing-style lookup                                [Phase 4 – placeholder]
                    │
                    ▼
              decision.py   verdict + risk level + explanation + per-component limits [Phase 1 ✔, tuned in Phase 5]
```

The submitted URL is only ever **parsed as text**. The backend never connects to it.

## Directory structure

```
phishguard/
├── backend/
│   ├── app/
│   │   ├── main.py         FastAPI app: endpoints, error shape, logging, wiring
│   │   ├── schemas.py      Pydantic request/response contracts and enums
│   │   ├── features.py     Safe URL parsing + deterministic feature extraction
│   │   ├── rules.py        Explainable rule engine
│   │   ├── model.py        ML interface (Phase 1: reports not_configured)
│   │   ├── reputation.py   Threat-intel interface (Phase 1: reports not_configured)
│   │   ├── decision.py     Hybrid decision logic
│   │   ├── config.py       (added) Settings from PHISHGUARD_* env vars, validated
│   │   ├── ratelimit.py    (added) In-memory sliding-window rate limiter
│   │   └── middleware.py   (added) Request body size limit (incl. chunked bodies)
│   ├── tests/              pytest suites (one per module) + conftest.py
│   ├── data/  models/      (empty; filled in Phase 3, git-ignored)
│   ├── pytest.ini
│   └── requirements.txt
├── extension/              (Phase 2)
├── training/               (Phase 3)
├── .env.example
├── .gitignore
└── README.md
```

`config.py`, `ratelimit.py` and `middleware.py` were added because limits/thresholds must not be magic numbers, and FastAPI has no built-in rate limiter or streaming-aware body cap. `tests/__init__.py` and `conftest.py` support the shared test fixtures.

## Prerequisites and setup (Linux)

Python 3.11+ (developed on 3.13).

```bash
cd phishguard
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

Optional configuration (all values have safe defaults):

```bash
cp .env.example .env        # .env is git-ignored
# edit .env, then:
set -a; . ./.env; set +a
```

## Run the backend

```bash
cd backend
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Bind to `127.0.0.1` only. Interactive docs: <http://127.0.0.1:8000/docs>.

## Run the tests

```bash
cd backend
python -m pytest
```

Success means every test passes (exit code 0). The suite is fully offline and deterministic: no credentials, no network, no model file. Several tests monkeypatch `socket.connect` to prove the backend never opens a connection while analyzing a URL.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | API info |
| GET | `/health` | Readiness and per-component status |
| POST | `/scan` | Assess a URL |

Example (abridged from a real run):

```bash
curl -s -X POST http://127.0.0.1:8000/scan \
  -H 'Content-Type: application/json' \
  -d '{"url":"http://paypal.com@8.8.8.8:8080/login/verify"}'
```

```json
{
  "url_display": "http://8.8.8.8:8080/...",
  "host": "8.8.8.8",
  "verdict": "suspicious",
  "risk_level": "high",
  "rules": { "score": 100, "findings": [ { "rule_id": "ip_host", "severity": "strong", "evidence": ["host: 8.8.8.8"], "...": "..." } ] },
  "ml":         { "status": "not_configured", "prediction": null },
  "reputation": { "status": "not_configured", "provider": null },
  "explanation": [
    "The URL-text rules found suspicious characteristics (rule score 100/100) without an independent confirmation.",
    "No threat-intelligence provider is configured, so known-bad listings were not checked."
  ]
}
```

Notice that rules alone cap at `suspicious` even at score 100: URL-text heuristics need an independent second signal before PhishGuard says `likely_phishing`.

**Verdicts:** `likely_phishing`, `suspicious`, `no_known_threat_detected`, `unknown`. **Risk levels:** `low`, `medium`, `high`, `unknown`.
With only the rules available and no findings, the answer is `unknown`/`unknown`, deliberately *not* "low". Missing information is never reported as a clean result.

**Errors** always use one shape: `{"error": {"code": "...", "message": "...", "details": [...]}}`. Codes: `invalid_url` (422), `validation_error` (422), `payload_too_large` (413), `rate_limited` (429, with `Retry-After`), `not_found`, `method_not_allowed`, `internal_error` (500). Errors never echo the submitted URL.

## Key design decisions (what to learn from each)

- **Strict parsing before analysis.** Only `http`/`https`; no whitespace, control, or invisible characters (bidi overrides disguise URLs); no backslashes (browsers and URL libraries disagree about them, a classic *parser differential*); valid port and hostname. Failures produce a stable `reason` code.
- **Public-suffix-aware domains.** `evil.example.co.uk` has registered domain `example.co.uk`, not `co.uk`. Without the Public Suffix List, "registered domain" logic is wrong for every multi-label TLD. `tldextract` runs on its bundled snapshot with network and cache disabled (guarded by a test).
- **IDN handling.** Hosts are converted to punycode for analysis and display; mixed-script labels (Latin + Cyrillic/Greek/Armenian) are a strong signal. The UI should show `host`, not `host_unicode`.
- **Weak indicators are capped** (total ≤ 20 < suspicious threshold 30), so a long URL with many subdomains and the word "account" can never be flagged by weak signals alone. Moderate/strong signals add up normally.
- **HTTPS is never rewarded.** Most phishing sites use it.
- **POST + hashed log references.** URLs in query strings end up in access logs; a JSON body does not. Our own logs record a 12-char SHA-256 reference. That is for correlation, not anonymization: common URLs can be guessed and hashed.
- **Component failure ≠ clean.** If ML or reputation raises an exception, the API reports `unavailable` and the decision engine says so.

## Privacy and security (implemented so far)

Request/body size limits · in-memory rate limiting · consistent non-leaking errors · no URL in logs · URL never fetched · password portion of `user:pass@host` discarded · no CORS headers unless explicitly configured · no secrets in code (`.env` is git-ignored).

## Limitations (known, current)

- **Only heuristics are active.** Without ML or threat intelligence, PhishGuard cannot say a URL is safe; the API says `unknown`.
- **Rule thresholds are provisional.** Scores and thresholds were chosen by judgment, not tuned on data. Expect both false positives and false negatives. Phase 5 tunes them on held-out data.
- **Brand list is a small illustrative sample** (`rules.py:BRAND_DOMAINS`). Real brands own many more domains (so legitimate ones may be flagged), and attackers target many more brands.
- **Whole-script homographs** (an all-Cyrillic lookalike) are only a weak IDN signal. IDNA 2003 normalization (Python's `idna` codec) differs slightly from the browser standard (UTS #46).
- **Rate limiter is per-process and per client address.** Behind a reverse proxy, all clients share one address. Fine for localhost use, not DDoS protection.
- **Rejects URLs containing zero-width joiners**, which a few legitimate scripts use.
- The `/docs` and `/openapi.json` endpoints are enabled (useful for learning); disable them for any non-local deployment.
- `starlette.testclient` emits a deprecation warning about `httpx`; harmless for now.

## Verified functionality (Phase 1)

Verified by running them in this session:

- 129 pytest tests pass (`cd backend && python -m pytest`, ~0.5 s).
- Real `uvicorn` server started; `/health`, `/scan` (suspicious, ordinary, invalid-scheme) answered correctly via `curl`; server log lines contained no URLs.
- Backend makes no network connections during URL analysis (test + `socket.connect` patched to fail).

Not yet built or verified: browser extension, dataset pipeline, model training, threat-intelligence integration, tuned thresholds, CORS/extension connectivity.

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 | Foundation: schemas, parsing, rules, API, tests | ✔ done |
| 2 | Chromium MV3 extension (popup, service worker, localhost connectivity) | next |
| 3 | Dataset prep, Random Forest training/evaluation, model inference | |
| 4 | Reputation provider abstraction + Google Safe Browsing integration | |
| 5 | Full hybrid decision tuning on held-out data | |
| 6 | Hardening, coverage, final docs | |

## Responsible use

PhishGuard is for authorized, defensive analysis. It deliberately never fetches or opens submitted URLs. Do not use it to build or refine phishing campaigns, and do not treat its output as a guarantee: a warning-free result does not make a link safe.
