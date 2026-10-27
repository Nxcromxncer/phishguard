/**
 * PhishGuard popup UI.
 *
 * Responsibilities: read input, validate it locally, ask the service worker to
 * scan, and render the answer. It never talks to the backend directly.
 *
 * SECURITY: every value that came from the user, the page, or the API is
 * rendered with `textContent` / `createTextNode` via the `el()` helper below.
 * There is NO innerHTML, insertAdjacentHTML, or document.write in this file.
 */

"use strict";

const MAX_URL_LENGTH = 2048;

const form = document.getElementById("scan-form");
const input = document.getElementById("url-input");
const inputError = document.getElementById("input-error");
const scanBtn = document.getElementById("scan-btn");
const tabBtn = document.getElementById("tab-btn");
const statusBox = document.getElementById("status");
const resultBox = document.getElementById("result");

const VERDICT_TEXT = {
  likely_phishing: "Likely phishing",
  suspicious: "Suspicious",
  no_known_threat_detected: "No known threat detected",
  unknown: "Unknown - not enough evidence",
};

/** Create an element with optional class and text (text is never parsed as HTML). */
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

// ---------------------------------------------------------------- validation
/** Returns an error message, or null when the URL looks acceptable. */
function validateUrl(value) {
  if (!value) return "Enter a URL to assess.";
  if (value.length > MAX_URL_LENGTH) return `URL is too long (maximum ${MAX_URL_LENGTH} characters).`;
  if (/\s/.test(value)) return "URL must not contain spaces or line breaks.";
  let parsed;
  try {
    parsed = new URL(value);
  } catch {
    return "That does not look like a valid URL. Include http:// or https://.";
  }
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    return "Only http:// and https:// URLs can be assessed.";
  }
  return null;
}

function showFieldError(message) {
  if (message) {
    inputError.textContent = message;
    inputError.hidden = false;
    input.setAttribute("aria-invalid", "true");
  } else {
    inputError.hidden = true;
    inputError.textContent = "";
    input.removeAttribute("aria-invalid");
  }
}

// -------------------------------------------------------------------- states
function setBusy(busy) {
  scanBtn.disabled = busy;
  tabBtn.disabled = busy;
  scanBtn.textContent = busy ? "Scanning..." : "Scan";
}

function showLoading() {
  clear(statusBox);
  statusBox.className = "status";
  statusBox.append(el("div", "spinner"), el("div", "msg", "Assessing URL. The URL is only analysed as text; it is never opened."));
  statusBox.hidden = false;
  resultBox.hidden = true;
}

function showError(title, message) {
  clear(statusBox);
  statusBox.className = "status error";
  const body = el("div");
  body.append(el("div", "title", title), el("div", "msg", message));
  statusBox.append(body);
  statusBox.hidden = false;
  resultBox.hidden = true;
}

function hideStatus() {
  statusBox.hidden = true;
  clear(statusBox);
}

const ERROR_TITLES = {
  backend_unreachable: "Backend not reachable",
  timeout: "Backend timed out",
  bad_response: "Unexpected response",
  rate_limited: "Too many requests",
  payload_too_large: "Request too large",
  invalid_url: "URL rejected",
  url_too_long: "URL too long",
};

// ----------------------------------------------------------------- rendering
function section(title) {
  const wrap = el("div");
  wrap.append(el("h2", null, title));
  return wrap;
}

function renderVerdict(data) {
  const risk = typeof data.risk_level === "string" ? data.risk_level : "unknown";
  const box = el("div", "verdict");
  box.dataset.risk = ["low", "medium", "high", "unknown"].includes(risk) ? risk : "unknown";
  box.append(el("p", "headline", VERDICT_TEXT[data.verdict] || "Unrecognised verdict"));
  const riskLine = el("p", "risk", "Risk level: ");
  riskLine.append(el("strong", null, risk));
  box.append(riskLine);
  return box;
}

function renderTarget(data) {
  const dl = el("dl", "target");
  dl.append(el("dt", null, "Scanned"), el("dd", null, data.url_display));
  dl.append(el("dt", null, "Host (ASCII form)"), el("dd", null, data.host));
  if (data.host_unicode) {
    dl.append(el("dt", null, "Host as displayed by Unicode"), el("dd", null, data.host_unicode));
  }
  return dl;
}

function renderExplanation(data) {
  const wrap = section("Assessment");
  const list = el("ul", "plain");
  for (const line of Array.isArray(data.explanation) ? data.explanation : []) {
    list.append(el("li", null, line));
  }
  wrap.append(list);
  return wrap;
}

function renderFindings(data) {
  const findings = data.rules && Array.isArray(data.rules.findings) ? data.rules.findings : [];
  const wrap = section(`Rule findings (score ${data.rules ? data.rules.score : "?"}/100)`);
  if (findings.length === 0) {
    wrap.append(el("p", "limit", "No rule triggered. This is not evidence that the URL is safe."));
    return wrap;
  }
  for (const f of findings) {
    const details = el("details", "finding");
    const summary = el("summary");
    summary.append(el("span", `sev ${["weak", "moderate", "strong"].includes(f.severity) ? f.severity : "weak"}`, f.severity), el("span", null, f.title));
    const body = el("div", "body");
    body.append(el("div", null, f.explanation));
    if (Array.isArray(f.evidence) && f.evidence.length) {
      const ev = el("ul", "evidence");
      for (const item of f.evidence) ev.append(el("li", null, item));
      body.append(ev);
    }
    details.append(summary, body);
    wrap.append(details);
  }
  return wrap;
}

function renderComponents(data) {
  const wrap = section("Detection components");
  const table = el("table", "components");
  const labels = { rules: "Rules", ml: "Machine learning", reputation: "Threat intelligence" };
  for (const c of Array.isArray(data.components) ? data.components : []) {
    const row = el("tr");
    row.append(el("th", null, labels[c.name] || c.name));
    const cell = el("td");
    cell.append(el("span", "pill", String(c.status).replace(/_/g, " ")), el("span", "limit", c.limitation));
    row.append(cell);
    table.append(row);
  }
  wrap.append(table);
  return wrap;
}

function renderResult(data) {
  clear(resultBox);
  resultBox.append(
    renderVerdict(data),
    renderTarget(data),
    renderExplanation(data),
    renderFindings(data),
    renderComponents(data)
  );
  resultBox.hidden = false;
}

// -------------------------------------------------------------------- actions
function sendToWorker(message) {
  return new Promise((resolve) => {
    try {
      chrome.runtime.sendMessage(message, (response) => {
        if (chrome.runtime.lastError || !response) {
          resolve({ ok: false, error: { code: "bad_response", message: "No response from the extension background worker." } });
        } else {
          resolve(response);
        }
      });
    } catch {
      resolve({ ok: false, error: { code: "bad_response", message: "Could not contact the extension background worker." } });
    }
  });
}

async function runScan(rawValue) {
  const value = rawValue.trim();
  const problem = validateUrl(value);
  showFieldError(problem);
  if (problem) {
    hideStatus();
    resultBox.hidden = true;
    return;
  }

  setBusy(true);
  showLoading();
  const response = await sendToWorker({ type: "scan", url: value });
  setBusy(false);

  if (!response.ok) {
    const { code, message } = response.error || {};
    showError(ERROR_TITLES[code] || "Scan failed", message || "An unknown error occurred.");
    return;
  }
  hideStatus();
  renderResult(response.data);
}

async function useCurrentTab() {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    const url = tab && tab.url;
    if (!url) {
      showFieldError("The current tab's URL is not available.");
      return;
    }
    if (!/^https?:\/\//i.test(url)) {
      showFieldError("The current tab is not an http(s) page, so it cannot be assessed.");
      return;
    }
    input.value = url;
    showFieldError(null);
    input.focus();
  } catch {
    showFieldError("Could not read the current tab.");
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  runScan(input.value);
});
tabBtn.addEventListener("click", useCurrentTab);
input.addEventListener("input", () => showFieldError(null));
input.focus();
