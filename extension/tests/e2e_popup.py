"""End-to-end test: load the unpacked extension into real Chromium and drive the popup.

Needs: `pip install playwright` and a Chromium binary (set CHROMIUM_PATH, or it
uses Playwright's own). Run from the repo root:

    python extension/tests/e2e_popup.py

Scenarios: successful scan, suspicious URL, invalid URL (client + server side),
long URL, backend offline, empty response, server error (500), 429, and XSS-style
content rendered as text. It starts the real backend for the first group and tiny
stub servers on the same port for the failure cases.
"""

from __future__ import annotations

import http.server
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / "extension"
BACKEND = ROOT / "backend"
PORT = 8000  # must match host_permissions in manifest.json

results: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, bool(condition), detail))
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f"  -> {detail}" if detail and not condition else ""))


def wait_port(open_expected: bool, timeout: float = 15) -> None:
    end = time.time() + timeout
    while time.time() < end:
        with socket.socket() as s:
            s.settimeout(0.3)
            is_open = s.connect_ex(("127.0.0.1", PORT)) == 0
        if is_open == open_expected:
            return
        time.sleep(0.2)
    raise RuntimeError(f"port {PORT} did not become {'open' if open_expected else 'closed'}")


class Stub:
    """Tiny HTTP server returning a fixed status/body for POST /scan."""

    def __init__(self, status: int, body: bytes, content_type: str = "application/json"):
        stub_status, stub_body, stub_ct = status, body, content_type

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers.get("content-length", 0)))
                self.send_response(stub_status)
                self.send_header("Content-Type", stub_ct)
                self.send_header("Content-Length", str(len(stub_body)))
                self.end_headers()
                self.wfile.write(stub_body)

            def log_message(self, *a):
                pass

        http.server.ThreadingHTTPServer.allow_reuse_address = True
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def start_backend(env_extra: dict[str, str] | None = None) -> subprocess.Popen:
    env = {**os.environ, "PHISHGUARD_RATE_LIMIT_PER_MINUTE": "1000", **(env_extra or {})}
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(PORT)],
        cwd=BACKEND, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    wait_port(True)
    return proc


def stop(proc: subprocess.Popen) -> None:
    proc.terminate()
    proc.wait(timeout=10)
    wait_port(False)


def scan(page, value: str) -> None:
    page.fill("#url-input", value)
    page.click("#scan-btn")


def main() -> int:
    user_dir = tempfile.mkdtemp(prefix="pg-profile-")
    launch_kwargs = dict(
        headless=True,
        args=[f"--disable-extensions-except={EXT}", f"--load-extension={EXT}", "--headless=new"],
    )
    if os.environ.get("CHROMIUM_PATH"):
        launch_kwargs["executable_path"] = os.environ["CHROMIUM_PATH"]

    with sync_playwright() as pw:
        ctx = pw.chromium.launch_persistent_context(user_dir, **launch_kwargs)
        try:
            sw = ctx.service_workers[0] if ctx.service_workers else ctx.wait_for_event("serviceworker", timeout=15000)
            ext_id = sw.url.split("/")[2]
            check("service worker registered", sw.url.endswith("service-worker.js"), sw.url)

            def open_popup():
                page = ctx.new_page()
                page.goto(f"chrome-extension://{ext_id}/popup.html")
                return page

            page_errors: list[str] = []

            # ---------------- Real backend ----------------
            backend = start_backend()
            try:
                page = open_popup()
                page.on("pageerror", lambda e: page_errors.append(str(e)))

                scan(page, "https://www.example.com/")
                page.wait_for_selector("#result:not([hidden])")
                check("success: verdict shown", "Unknown" in page.inner_text(".verdict .headline"))
                check("success: risk level unknown (not 'low')", page.get_attribute(".verdict", "data-risk") == "unknown")
                check("success: components table lists 3 rows", page.locator("table.components tr").count() == 3)
                check("success: disclaimer visible", "not a guarantee" in page.inner_text(".disclaimer"))
                check("success: loading state cleared", page.is_hidden("#status"))

                scan(page, "http://paypal.com@8.8.8.8:8080/login/verify")
                # NOTE: locator waits, not wait_for_function(str): the extension CSP forbids eval.
                page.locator(".verdict .headline", has_text="Suspicious").wait_for()
                check("suspicious: risk high", page.get_attribute(".verdict", "data-risk") == "high")
                check("suspicious: findings rendered", page.locator("details.finding").count() >= 2)
                page.locator("details.finding summary").first.click()
                check("suspicious: evidence expandable", "8.8.8.8" in page.inner_text("details.finding[open] .evidence"))
                check("suspicious: URL path not echoed in display", "login" not in page.inner_text(".target dd:first-of-type"))

                scan(page, "http://аpple.com/")
                page.locator(".target", has_text="xn--").wait_for()
                check("homograph: punycode host shown", "xn--" in page.inner_text(".target"))

                # Client-side validation: no request needed.
                for bad, why in [("", "empty"), ("example.com", "no scheme"), ("ftp://example.com", "ftp"), ("javascript:alert(1)", "javascript"), ("http://a b.com", "space")]:
                    scan(page, bad)
                    check(f"invalid input rejected locally ({why})", page.is_visible("#input-error") and page.get_attribute("#url-input", "aria-invalid") == "true")
                check("invalid input: no result shown", page.is_hidden("#result"))

                # Long URL: over the limit is rejected locally; just under works.
                scan(page, "http://example.com/" + "a" * 2100)
                check("long URL: rejected locally with message", "too long" in page.inner_text("#input-error"))
                scan(page, "http://example.com/" + "a" * 1900)
                page.wait_for_selector("#result:not([hidden])")
                check("long URL under limit: scanned", "long_url" in page.inner_html("#result") or "Very long URL" in page.inner_text("#result"))

                # Server-side rejection that passes the browser's URL parser (backslash).
                scan(page, "http://example.com\\@evil.com/")
                page.wait_for_selector("#status.error")
                check("server rejection: error shown", "URL rejected" in page.inner_text("#status"))
                check("server rejection: input value not echoed", "evil.com" not in page.inner_text("#status"))

                # "Use current tab": opened as a normal tab (the only way to automate it) the popup has
                # no activeTab grant, so Chrome hides the URL. The popup must refuse, not crash or fill
                # in junk. The SUCCESS path needs a real toolbar click - see the manual checklist.
                page.click("#tab-btn")
                msg = page.inner_text("#input-error")
                check("use current tab: refused without activeTab grant",
                      "not available" in msg or "not an http" in msg, msg)
                check("use current tab: input left untouched", page.input_value("#url-input") == "http://example.com\\@evil.com/")
                page.close()
            finally:
                stop(backend)

            # ---------------- Backend offline ----------------
            page = open_popup()
            scan(page, "https://www.example.com/")
            page.wait_for_selector("#status.error")
            check("offline: clear message", "not reachable" in page.inner_text("#status"), page.inner_text("#status"))
            check("offline: no result shown", page.is_hidden("#result"))
            check("offline: buttons re-enabled", page.is_enabled("#scan-btn"))
            page.close()

            # ---------------- Empty response / server error / 429 / XSS ----------------
            for label, status, body, ctype, expect in [
                ("empty 200 body", 200, b"", "application/json", "Unexpected response"),
                ("non-JSON 200 body", 200, b"<html>oops</html>", "text/html", "Unexpected response"),
                ("500 with no body", 500, b"", "application/json", "Scan failed"),
                ("500 generic error", 500, b'{"error":{"code":"internal_error","message":"An internal error occurred."}}', "application/json", "internal error"),
                ("429 rate limited", 429, b'{"error":{"code":"rate_limited","message":"Too many requests. Please slow down."}}', "application/json", "Too many requests"),
            ]:
                stub = Stub(status, body, ctype)
                try:
                    page = open_popup()
                    scan(page, "https://www.example.com/")
                    page.wait_for_selector("#status.error")
                    text = page.inner_text("#status")
                    check(f"{label}: handled", expect.lower() in text.lower(), text)
                    check(f"{label}: no result rendered", page.is_hidden("#result"))
                    page.close()
                finally:
                    stub.close()

            xss_payload = (
                b'{"verdict":"suspicious","risk_level":"high","url_display":"<img src=x onerror=window.__pwned=1>",'
                b'"host":"<script>window.__pwned=1</script>","host_unicode":null,'
                b'"rules":{"score":10,"findings":[{"rule_id":"x","severity":"weak","score":10,"title":"<b>bold</b>",'
                b'"explanation":"<img src=x onerror=window.__pwned=1>","evidence":["<script>window.__pwned=1</script>"]}]},'
                b'"explanation":["<img src=x onerror=window.__pwned=1>"],"components":[]}'
            )
            stub = Stub(200, xss_payload)
            try:
                page = open_popup()
                scan(page, "https://www.example.com/")
                page.wait_for_selector("#result:not([hidden])")
                page.wait_for_timeout(300)
                check("xss: payload rendered as text", "<img src=x" in page.inner_text("#result"))
                check("xss: no injected elements", page.locator("#result img, #result script, #result b").count() == 0)
                check("xss: no script executed", page.evaluate("() => window.__pwned") is None)
                page.close()
            finally:
                stub.close()

            check("no uncaught JS errors in popup", not page_errors, "; ".join(page_errors))
        finally:
            ctx.close()

    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
