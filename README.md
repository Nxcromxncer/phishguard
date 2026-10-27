# PhishGuard

A hybrid phishing-URL assessment system: **rule-based heuristics + a machine-learning classifier + threat-intelligence lookup**, combined by an interpretable decision engine, served by a FastAPI backend and (planned) a Chromium Manifest V3 extension.

> **Status: Phase 3 of 6 complete.** The backend, a working **manual-scan** Chromium extension, and the **ML pipeline** (dataset preparation, Random Forest / gradient-boosting training, evaluation, safe model loading) are built and tested. **No real dataset has been used and no real-world accuracy has been measured:** the pipeline was exercised only on clearly-labelled synthetic data. The threat-intelligence provider is **not built yet** (reported as `not_configured`). See [Verified functionality](#verified-functionality) and [Roadmap](#roadmap).

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
       ├──► model.py       verified artifact load + inference (RF baseline / HGB)    [Phase 3 ✔; needs YOUR trained model]
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
│   │   ├── model.py        Model artifact writing/loading (integrity checks) + inference
│   │   ├── reputation.py   Threat-intel interface (Phase 1: reports not_configured)
│   │   ├── decision.py     Hybrid decision logic
│   │   ├── config.py       (added) Settings from PHISHGUARD_* env vars, validated
│   │   ├── ratelimit.py    (added) In-memory sliding-window rate limiter
│   │   └── middleware.py   (added) Request body size limit (incl. chunked bodies)
│   ├── tests/              pytest suites (one per module) + conftest.py
│   ├── data/  models/      (empty in the repo; prepared data and model artifacts land here, git-ignored)
│   ├── pytest.ini
│   └── requirements.txt
├── extension/
│   ├── manifest.json       MV3 manifest: minimal permissions, strict CSP
│   ├── service-worker.js   Only component that talks to the backend
│   ├── popup.html / .css / .js   UI; renders everything with textContent
│   └── tests/e2e_popup.py  (added) Real-Chromium end-to-end test (Playwright)
├── training/
│   ├── prepare_dataset.py  Validate/clean/dedupe URL data; provenance + leakage report
│   ├── train.py            Train rf (baseline) or hgb; domain-grouped or time split
│   ├── evaluate.py         Held-out test metrics, per-category breakdown, rules baseline
│   ├── pipeline_common.py  (added) Shared feature building, hashing, metric definitions
│   └── make_synthetic_sample.py  (added) SYNTHETIC data to exercise the pipeline offline
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

## Machine learning: data, training, evaluation

The classifier is trained **offline** by you; no API credentials are needed. Until you train and place a model, the API reports ML as `not_configured`.

### 1. Get data (you do this; PhishGuard never downloads or visits URLs)

You need labelled **phishing** URLs and **legitimate** URLs from sources whose license/terms you accept. Typical choices (check each one's current terms before use):

- Phishing: a [PhishTank](https://phishtank.org/) `verified_online.csv` export, or another documented research dataset.
- Legitimate: the [Tranco](https://tranco-list.eu/) top-sites list (`rank,domain`), or your own list of known-good URLs.

Download them in your browser or with your own tools, then:

```bash
source .venv/bin/activate
python training/prepare_dataset.py \
  --phishing verified_online.csv --legit top-1m.csv --legit-limit 50000 \
  --provenance-note "PhishTank export 2026-10-25 (terms: ...); Tranco list ID ... (terms: ...)"
```

Accepted formats (details in `training/prepare_dataset.py`):

| Mode | Flags | File format |
|---|---|---|
| Separate files | `--phishing F --phishing-format phishtank` (default) | CSV with a `url` column; optional `submission_time` (used as date) and `verified` (only `yes` rows kept) |
| | `--phishing-format urls` | one URL per line |
| | `--legit F --legit-format tranco` (default) | headerless `rank,domain`; each becomes `https://<domain>/` |
| | `--legit-format urls` | one URL per line |
| Combined CSV | `--input F` | columns `url`, `label` (+ optional `source`, `date`). Labels are mapped explicitly (`phishing`/`malicious`/`1` -> 1, `legitimate`/`benign`/`0` -> 0, or `--label-map "spam=1,ham=0"`). Any other label is rejected, never guessed as legitimate. |

It validates each URL with the **same parser the API uses**, drops invalid, duplicate and conflicting-label URLs, strips embedded passwords, and writes `backend/data/prepared.csv` plus `prepared_report.json` (input file SHA-256s, row counts, rejection reasons, class distribution, your provenance note, and **warnings**). Read the warnings:

- **PATH SHORTCUT RISK** - phishing feeds have paths, ranking lists are bare domains, so a model can score well just by detecting "has a path". Use `--host-only` or a legitimate source with realistic paths.
- **SCHEME SKEW** - e.g. legitimate URLs all `https`, phishing mixed.
- **Domains in both classes** (abused hosting), **tiny or severely imbalanced classes**, missing provenance note.

### 2. Train

```bash
python training/train.py --model rf          # Random Forest baseline (default)
python training/train.py --model hgb --out-dir /tmp/hgb   # gradient boosting, to compare
```

- `rf` = RandomForest (300 trees, balanced class weights); `hgb` = scikit-learn HistGradientBoosting. **LightGBM / XGBoost are not included**: add them only if they beat the baseline on the grouped split (they would need the extra dependency and a new `--model` entry).
- **Split:** `--split group` (default) puts every *registered domain* in exactly one of train / validation / test (70/15/15), so the model cannot memorise sites. `--split time` tests on the newest rows but needs a date on **every** row, which the PhishTank + Tranco combination usually lacks (the script refuses rather than guess).
- Class imbalance uses class weights. There is no data-dependent preprocessing (trees need no scaling), so nothing can leak from validation/test into fitting.
- Output in `backend/models/`: `model.joblib`, `model_meta.json` (version, feature schema, dataset hash and provenance, hyperparameters, split, validation metrics, SHA-256, threshold), and `split.json` (row indices, so evaluation scores exactly the held-out rows).
- The decision **threshold is 0.5 and provisional**; scores are **uncalibrated** (not probabilities). Both are recorded in the metadata and shown by the API.

### 3. Evaluate

```bash
python training/evaluate.py --model-dir backend/models
```

Reports precision, recall, F1, the confusion matrix, false-positive rate and false-negative count on the **test** set only, a breakdown by scheme / IP host / path / IDN / URL length / source (groups with too few examples are marked `~` low-support), and the **rule engine alone on the same rows** as a baseline the model must beat to justify itself. It refuses to run if `prepared.csv` or the model changed since training. It also prints what false positives and false negatives mean for users.

> A high score on a held-out split of one dataset is **not** real-world accuracy. Datasets age fast, have collection artifacts, and may not resemble the URLs you will see.

### 4. Use the model in the backend

The backend loads `backend/models/` at start-up (restart it after training). Settings (all optional, see `.env.example`):

| Variable | Purpose |
|---|---|
| `PHISHGUARD_MODEL_DIR` | Use another directory than `backend/models` |
| `PHISHGUARD_MODEL_SHA256` | **Pin** the expected model-file hash (see trust model) |
| `PHISHGUARD_ALLOW_SYNTHETIC_MODEL=1` | Demos only: allow a model trained on synthetic data |

A model trained on synthetic data is **refused by default** (ML shows `unavailable` with the reason), so a demo model can never silently influence a verdict.

### Try the pipeline offline with synthetic data (demo only)

```bash
python training/make_synthetic_sample.py --out /tmp/syn.csv
python training/prepare_dataset.py --input /tmp/syn.csv --out-dir /tmp/pg-data
python training/train.py --data-dir /tmp/pg-data --out-dir /tmp/pg-model
python training/evaluate.py --data-dir /tmp/pg-data --model-dir /tmp/pg-model
PHISHGUARD_MODEL_DIR=/tmp/pg-model PHISHGUARD_ALLOW_SYNTHETIC_MODEL=1 \
  uvicorn app.main:app --host 127.0.0.1 --port 8000    # run from backend/
```

The synthetic URLs are invented templates that are easy to separate, so scores near 100% are expected and **mean nothing**. Every output labels them as such.

### Model-file trust model

`joblib`/`pickle` files can run arbitrary code when loaded, so `app/model.py`:

1. loads only from the configured project directory, never from a request, and refuses symlinks, non-regular files and oversized files;
2. checks the SHA-256 in `model_meta.json`, the feature schema version/order, the scikit-learn `major.minor`, class labels and feature count **before** trusting the model, and verifies the hash **before** unpickling;
3. turns any failure into `unavailable` (never a crash, never a fake prediction).

The hash in `model_meta.json` detects corruption and accidental swaps but is **not authentication**: anyone who can replace the model can replace the metadata. To get real protection, copy the printed `model_sha256` into `PHISHGUARD_MODEL_SHA256` (kept outside the model directory) and make `backend/models/` writable only by you.

## Install the browser extension

1. Start the backend (above) and leave it running on `127.0.0.1:8000`.
2. In Chrome/Chromium/Brave/Edge open `chrome://extensions` (Edge: `edge://extensions`).
3. Turn on **Developer mode** (top right).
4. Click **Load unpacked** and select the `extension/` folder.
5. Pin PhishGuard from the puzzle-piece menu, click its icon, paste a URL, and press **Scan**.
   "Use current tab" fills in the URL of the tab you are on (works only for `http(s)` pages).

If you edit extension files, press the reload arrow on its card in `chrome://extensions`.

### Why the extension works with the local backend (permissions explained)

- **Manifest V3** replaces the persistent background page with a **service worker** that the browser suspends when idle. So `service-worker.js` keeps no state in globals and is a stateless request handler.
- **`host_permissions: ["http://127.0.0.1:8000/*"]`** is the only network access. It lets extension pages and the service worker call the backend **without CORS**: requests from an extension to an origin it has host permission for bypass CORS checks, which is why the backend sends no CORS headers (a test asserts that). Any other origin is blocked. Using the literal `127.0.0.1` instead of `localhost` avoids DNS/IPv6 ambiguity and a broader match.
- **`activeTab`** grants access to the current tab's URL *only after you click the extension icon*, and only for that tab. This is why there is no broad `tabs` permission and no `<all_urls>` host permission. The extension cannot see your browsing history or other tabs.
- **No content scripts, no `storage`, no `webNavigation`, no `webRequest`.** Nothing is injected into pages; nothing is persisted.
- **CSP** (`script-src 'self'`; no inline scripts, no `eval`) is set explicitly. In testing, Chromium refused a string-`eval` in the popup, confirming it is enforced.
- **Plain HTTP to localhost** is allowed for loopback and is acceptable for a local developer tool. It is *not* acceptable for a remote backend. If you ever host the API elsewhere, use HTTPS and change `API_BASE` in `service-worker.js` **and** `host_permissions` in the manifest (the manifest is what actually grants access).
- **Localhost exposure:** any local process (and any web page's JavaScript, since the backend does not use CORS to block simple requests) could also call the backend. It only returns assessments of URLs it is given and never fetches them, but do not bind it to `0.0.0.0`.
- **Not implemented:** automatic navigation monitoring and blocking. PhishGuard scans only when you click Scan. It does not warn you or stop you when you visit a page.

## Manual browser test checklist

Automated coverage is in `extension/tests/e2e_popup.py` (below). Do these by hand once, in a real browser, especially the ones marked ★ that automation cannot reproduce.

| # | Scenario | Do | Expect |
|---|---|---|---|
| 1 | Successful scan | Backend running; scan `https://www.example.com/` | Verdict "Unknown - not enough evidence", risk **unknown**, three components listed |
| 2 | Suspicious URL | Scan `http://paypal.com@8.8.8.8:8080/login/verify` | "Suspicious", risk high, expandable findings with evidence |
| 3 | No known threat found | Needs a trained model (or Phase 4). With the synthetic demo model (see ML section) scan `https://www.gardenriver.com/about` | "No known threat detected", risk low, with the demo-model warning. Without a model a clean-looking URL is `unknown` by design |
| 4 | Threat intel unavailable | Shown now as "not configured"; "unavailable" arrives with Phase 4 | Row says *not configured* with its limitation |
| 5 | Invalid URL | Scan `example.com`, `ftp://x.com`, `javascript:alert(1)` | Red inline message under the field, no request sent |
| 6 | Server-side rejection | Scan `http://example.com\@evil.com/` | "URL rejected" banner, URL not echoed |
| 7 | Backend offline | Stop uvicorn; scan | "Backend not reachable" banner; buttons re-enable |
| 8 | Server error / empty response | Not reproducible against the real backend; covered by automated stub tests | n/a |
| 9 | Long URL | Paste a URL over 2048 characters | "URL is too long" message |
| 10 | ★ Use current tab | Open any https site, click the PhishGuard toolbar icon, press **Use current tab** | Field fills with that tab's URL |
| 11 | ★ Keyboard only | Tab through the popup; press Enter in the field | All controls reachable with a visible focus ring; Enter scans |
| 12 | ★ Permissions prompt | On `chrome://extensions` > Details | Site access shows only `127.0.0.1:8000` |

## Test the extension automatically

Needs Playwright and a Chromium binary (not part of `requirements.txt`; the backend does not need them):

```bash
source .venv/bin/activate
pip install playwright
# Playwright's own browser:   playwright install chromium
# ...or point at one you have:
export CHROMIUM_PATH=/usr/bin/chromium      # Arch: sudo pacman -S chromium
python extension/tests/e2e_popup.py         # expect: 40/40 checks passed, exit code 0
```

It loads the unpacked extension in headless Chromium, starts the real backend for the success paths and small stub servers on the same port for failure paths (offline, empty body, non-JSON body, 500, 429), and checks that hostile API content (`<img onerror>`, `<script>`) renders as inert text. It must have port 8000 free.

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
- **Split by domain, not by row.** Many URLs share one site. A random split puts the same domain in train and test, so the model memorises sites and the score is inflated. Splitting by registered domain measures generalization to *unseen* sites.
- **Compare against a baseline.** `evaluate.py` scores the rule engine on the same test rows. If the model cannot clearly beat simple rules on real data, it adds complexity without value.
- **Tie everything to hashes.** The dataset hash is in the model metadata; the model hash is in the metadata and (optionally) pinned in the environment; `split.json` records the dataset it belongs to. Mixing files from different runs is detected, not silently accepted.

## Privacy and security (implemented so far)

Request/body size limits · in-memory rate limiting · consistent non-leaking errors · no URL in logs · URL never fetched · password portion of `user:pass@host` discarded · no CORS headers unless explicitly configured · no secrets in code (`.env` is git-ignored).

## Limitations (known, current)

- **No real model ships with the project, and none has been trained on real data.** Until you prepare a dataset and train, only the rules are active and the API says `unknown` for URLs the rules do not flag. Without threat intelligence (Phase 4) PhishGuard still cannot confirm a URL is safe.
- **ML scores are uncalibrated and the 0.5 threshold is untuned.** Tree-ensemble scores are not probabilities. Phase 5 tunes the threshold on the validation split and may add calibration.
- **A model alone can currently produce "no known threat detected / low".** If a trained model says "legitimate" and no rule fires, the verdict is `no_known_threat_detected` (with the standing caveat that this is not proof of safety). With a weak model this is optimistic; Phase 5 revisits how much a model-only "clean" result should count.
- **The model sees URL text only.** It cannot know a page's content, age, hosting, or certificate, and attackers can craft URLs that look ordinary.
- **Datasets age and carry artifacts.** Phishing tactics change within weeks; the report warns about path/scheme shortcuts but cannot detect every collection artifact. Retrain regularly and re-evaluate. Models are tied to the scikit-learn `major.minor` they were trained with.
- **Time-aware evaluation** needs real dates on every row; the usual PhishTank + Tranco combination only supports the domain-grouped split.
- **LightGBM/XGBoost are not implemented** (only `rf` and `hgb`).
- **Rule thresholds are provisional.** Scores and thresholds were chosen by judgment, not tuned on data. Expect both false positives and false negatives. Phase 5 tunes them on held-out data.
- **Brand list is a small illustrative sample** (`rules.py:BRAND_DOMAINS`). Real brands own many more domains (so legitimate ones may be flagged), and attackers target many more brands.
- **Whole-script homographs** (an all-Cyrillic lookalike) are only a weak IDN signal. IDNA 2003 normalization (Python's `idna` codec) differs slightly from the browser standard (UTS #46).
- **Rate limiter is per-process and per client address.** Behind a reverse proxy, all clients share one address. Fine for localhost use, not DDoS protection.
- **Rejects URLs containing zero-width joiners**, which a few legitimate scripts use.
- The `/docs` and `/openapi.json` endpoints are enabled (useful for learning); disable them for any non-local deployment.
- `starlette.testclient` emits a deprecation warning about `httpx`; harmless for now.
- **Extension:** manual scanning only (no automatic protection); backend address is fixed at `127.0.0.1:8000` (edit the manifest *and* `service-worker.js` to change it); no toolbar icons are bundled, so Chrome shows a default letter icon; the popup's 2048-character limit duplicates the backend default and will not follow a changed `PHISHGUARD_MAX_URL_LENGTH`.

## Verified functionality

Verified by running them in this session:

- **Backend and pipeline:** 196 pytest tests pass (`cd backend && python -m pytest`, ~8 s). This includes the dataset, training and evaluation scripts run on a small synthetic dataset, model-loading failure modes (schema/version/checksum mismatch, tampering, symlinks, bad metadata, synthetic-model guard, pinned hash), reproducibility, and domain-disjoint splits. A real `uvicorn` server answered `/health` and `/scan` via `curl` with no model, a refused synthetic model, an allowed demo model, and a wrong pinned hash; its log lines contained no URLs. The backend makes no network connections during URL analysis or dataset preparation (tested).
- **Training on synthetic data only:** both `rf` and `hgb` trained and evaluated end to end. Scores of ~99-100% were observed and are **meaningless** (invented, easily separable templates). The rule engine alone caught about 60% of the synthetic phishing; this too says nothing about real data.
- **Extension:** 40/40 checks pass in `extension/tests/e2e_popup.py`, which loads the unpacked extension in headless Chromium (Playwright) against the real backend. Covers successful scan, suspicious URL, homograph host, local and server-side invalid-URL handling, long URLs, backend offline, empty / non-JSON / 500 / 429 responses, XSS-style payloads rendered as text, and no uncaught JavaScript errors. This also confirms extension-to-`127.0.0.1` requests work with no CORS headers on the backend.

Not verified (needs a person at a real browser): the **"Use current tab"** success path (Chrome grants `activeTab` only on a real toolbar click, which automation cannot reproduce), visual appearance and contrast, screen-reader behavior, and non-Chromium browsers. Automation ran on a Chromium build, not on branded Chrome.

Not verified at all: **any behavior on real phishing/legitimate data** (no real dataset was available in the build environment: it cannot download them, and PhishGuard deliberately never fetches data itself), and the `PhishTank`/`Tranco` file-format readers against **current real files** (they were tested on small hand-made files that follow the documented columns; if a feed changes its format, `prepare_dataset.py` will fail loudly rather than guess).

Not yet built: threat-intelligence integration, tuned thresholds and calibration, automatic navigation monitoring.

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 | Foundation: schemas, parsing, rules, API, tests | ✔ done |
| 2 | Chromium MV3 extension (popup, service worker, localhost connectivity), manual scan only | ✔ done |
| 3 | Dataset prep, Random Forest (+ gradient boosting) training/evaluation, model inference | ✔ done (pipeline built; **needs your real dataset to produce a usable model**) |
| 4 | Reputation provider abstraction + Google Safe Browsing integration | |
| 5 | Full hybrid decision tuning on held-out data | |
| 6 | Hardening, coverage, final docs | |

## Responsible use

PhishGuard is for authorized, defensive analysis. It deliberately never fetches or opens submitted URLs. Do not use it to build or refine phishing campaigns, and do not treat its output as a guarantee: a warning-free result does not make a link safe.
