#!/usr/bin/env python3
# Cotton Candy Process Registry — POST to "/" with value=<uuid> (and optional info=<string>)
# GET "/" renders a list (no input form).
# Also supports JSON: {"value":"<uuid>", "info":"..."} or {"uuid":"<uuid>", "info":"..."}.

from bottle import Bottle, request, response, HTTPError
import os, time, yaml, tempfile

app = Bottle()

# --- Settings ---
HOST = "0.0.0.0"
PORT = 7206
REG_PATH = "/home/nicolas/Cotton-Candy-Digital-Twin/registry.yaml"
MAX_ENTRIES = 2000
CORS_ALLOW = "*"
CPEE_LOG_BASE = "https://cpee.org/logs"   # log_url = <base>/<uuid>.xes.yaml

# --- Helpers ---
def now_ts() -> int:
    return int(time.time())

def load_registry():
    if not os.path.exists(REG_PATH):
        return []
    try:
        with open(REG_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or []
        return data if isinstance(data, list) else []
    except Exception:
        return []

def atomic_dump(entries):
    d = os.path.dirname(REG_PATH)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".registry.", dir=d, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.safe_dump(entries[-MAX_ENTRIES:], f, sort_keys=False, allow_unicode=True)
        os.replace(tmp, REG_PATH)
    except Exception:
        try: os.unlink(tmp)
        except Exception: pass
        raise

def _set_common_headers():
    if response.content_type is None:
        response.content_type = "application/json; charset=utf-8"
    response.set_header("Access-Control-Allow-Origin", CORS_ALLOW)
    response.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
    response.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

@app.hook('after_request')
def _after(): _set_common_headers()

@app.route("/<:re:.*>", method=["OPTIONS"])
def _options(): return {}

# --- Core: POST to "/" ---
@app.post("/")
def root_post():
    """
    Accepts:
      - form: value=<uuid> and optionally info=<string>
      - JSON: {"value":"<uuid>","info":"..."} or {"uuid":"<uuid>","info":"..."}
    """
    body = request.json or {}
    uuid = (request.forms.get("value") or body.get("uuid") or body.get("value") or "").strip()
    if not uuid:
        raise HTTPError(400, "uuid required as form field 'value' or JSON 'uuid'/'value'")

    info = request.forms.get("info")
    if info is None:
        info = body.get("info")  # may be None — that's fine

    ts = now_ts()
    entry = {
        "uuid": uuid,
        "ts": ts,
        "iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)),
        "log_url": f"{CPEE_LOG_BASE}/{uuid}.xes.yaml",
        "source": request.remote_addr,
    }
    if isinstance(info, str) and info.strip() != "":
        entry["info"] = info.strip()

    entries = load_registry()
    entries.append(entry)
    atomic_dump(entries)
    return {"ok": True, "count": len(entries[-MAX_ENTRIES:])}

# keep /register for backward compatibility (same behavior as "/")
@app.post("/register")
def register_post():
    return root_post()

# --- HTML view at "/" (GET) — no input form, now includes an 'Info' column ---
@app.get("/")
def index():
    response.content_type = "text/html; charset=utf-8"
    entries = list(reversed(load_registry()))
    def esc(s):
        # very small escape to avoid breaking HTML when info has symbols
        return (str(s)
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
                .replace("'", "&#39;"))
    def row(e):
        u   = esc(e.get("uuid",""))
        log = e.get("log_url","")
        iso = esc(e.get("iso",""))
        src = esc(e.get("source",""))
        inf = esc(e.get("info","")) if e.get("info") is not None else ""
        link = f'<a href="{log}" target="_blank">log</a>' if log else ""
        return f"<tr><td><code>{u}</code></td><td>{iso}</td><td>{inf}</td><td>{link}</td><td>{src}</td></tr>"

    html = f"""<!doctype html>
<meta charset="utf-8"><title>Registered Processes</title>
<style>
body{{font:14px system-ui,Arial;margin:20px;color:#111}}
table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #e5e5e5;padding:6px 8px;vertical-align:top}}
th{{background:#f7f7f7;text-align:left}}
code{{font-family:ui-monospace,Menlo,Monaco,monospace}}
a{{text-decoration:none}}
</style>
<h1>Registered Processes</h1>
<p>Total: {len(entries)}</p>
<table>
  <thead>
    <tr><th>UUID</th><th>Log</th><th>Timestamp</th><th>Info</th><th>Source</th></tr>
  </thead>
  <tbody>
    {''.join(row(e) for e in entries) if entries else '<tr><td colspan="5">No entries yet</td></tr>'}
  </tbody>
</table>
"""
    return html

if __name__ == "__main__":
    app.run(host=HOST, port=PORT, server="paste")