/**
 * PhishGuard service worker (background context).
 *
 * Role in the architecture
 * ------------------------
 * The popup is short-lived: it is destroyed when it closes. The service worker is
 * the single place that talks to the backend, so that:
 *   - API origin, timeout, and error normalisation live in ONE file;
 *   - future features (automatic navigation monitoring, badge updates, caching)
 *     can reuse the same `scanUrl()` without touching the UI;
 *   - the popup never needs to know how the backend is reached.
 *
 * What it does NOT do (documented limitation)
 * -------------------------------------------
 * There is NO automatic navigation monitoring and NO blocking. The extension
 * only scans when you click "Scan". It does not read page content, does not
 * inject content scripts, and does not stop you from visiting any page.
 *
 * Connectivity
 * ------------
 * A request from an extension context to an origin listed in `host_permissions`
 * is not subject to CORS, so the backend needs no CORS headers. Anything NOT
 * listed in host_permissions (any other origin) is blocked by the browser.
 * MV3 service workers are suspended when idle: keep no state in globals.
 */

// Must match host_permissions in manifest.json. Changing the backend address
// means editing BOTH places (the manifest is what actually grants access).
const API_BASE = "http://127.0.0.1:8000";
const REQUEST_TIMEOUT_MS = 10000;
const MAX_URL_LENGTH = 2048; // mirrors the backend default PHISHGUARD_MAX_URL_LENGTH

/** Build a normalised failure object the popup can render without guessing. */
function failure(code, message, status) {
  return { ok: false, error: { code, message, status: status ?? null } };
}

/**
 * Scan one URL via POST /scan.
 * Always resolves (never throws) to {ok:true,data} or {ok:false,error}.
 */
async function scanUrl(url) {
  if (typeof url !== "string" || url.length === 0) {
    return failure("invalid_input", "No URL was provided.");
  }
  if (url.length > MAX_URL_LENGTH) {
    return failure("url_too_long", `URL is longer than ${MAX_URL_LENGTH} characters.`);
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const response = await fetch(`${API_BASE}/scan`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
      signal: controller.signal,
      credentials: "omit", // never send cookies to the backend
      cache: "no-store",
    });

    let body = null;
    try {
      body = await response.json();
    } catch {
      body = null; // empty or non-JSON body
    }

    if (response.ok) {
      if (body && typeof body === "object" && typeof body.verdict === "string") {
        return { ok: true, data: body };
      }
      return failure("bad_response", "The backend returned an empty or unexpected response.", response.status);
    }

    // Backend errors use {error:{code,message}}; anything else is a generic failure.
    const apiError = body && typeof body === "object" ? body.error : null;
    if (apiError && typeof apiError.message === "string") {
      return failure(String(apiError.code || "http_error"), apiError.message, response.status);
    }
    return failure("http_error", `The backend responded with HTTP ${response.status}.`, response.status);
  } catch (err) {
    if (err && err.name === "AbortError") {
      return failure("timeout", "The backend did not respond in time.");
    }
    return failure(
      "backend_unreachable",
      `Could not reach the PhishGuard backend at ${API_BASE}. Is it running?`
    );
  } finally {
    clearTimeout(timer);
  }
}

// Message API used by the popup:  { type: "scan", url } -> { ok, data | error }
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  // Only accept messages from this extension's own pages.
  if (sender.id !== chrome.runtime.id) {
    return false;
  }
  if (message && message.type === "scan") {
    scanUrl(message.url).then(sendResponse);
    return true; // keep the channel open for the async response
  }
  return false;
});
