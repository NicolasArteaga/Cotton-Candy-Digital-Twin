#!/usr/bin/env python3
# Minimal CPEE run registry (Bottle) on :7206
# POST /register {"uuid": "...", "ts": <optional>, "log_url": <optional>, "meta": {...}}
# GET  /        -> simple HTML table
# GET  /api/list, /api/by/<uuid>, /health

from bottle import Bottle, request, response, HTTPError
import os, time, yaml, tempfile

app = Bottle()

PORT = 7206
HOST = "0.0.0.0"  # keep local; the reverse tunnel exposes it remotely
REG_PATH = "/home/nicolas/Cotton-Candy-Digital-Twin/registry.yaml"
MAX_ENTRIES = 2000
CORS_ALLOW = "*"  # adjust if you want to lock down

def now_ts(): return int(time.time())

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

@app.hook('after_request')
def set_headers():
    if response.content_type is None:
        response.content_type = "application/json; charset=utf-8"
    response.set_header("Access-Control-Allow-Origin", CORS_ALLOW)
    response.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
    response.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

@app.route("/<:re:.*>", method=["OPTIONS"])
def _options():
    return {}

@app.post("/register")
def register():
    body = request.json or {}
    uuid = (body.get("uuid") or body.get("instance_id") or "").strip()
    if not uuid:
        raise HTTPError(400, "uuid required")

    # timestamp: from body or server time
    try:
        ts = int(body.get("ts")) if body.get("ts") is not None else now_ts()
    except Exception:
        ts = now_ts()

    entry = {
        "uuid": uuid,
        "ts": ts,
        "iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)),
    }
    if body.get("log_url"): entry["log_url"] = body["log_url"]
    if isinstance(body.get("meta"), dict): entry["meta"] = body["meta"]
    entry["source"] = request.get("REMOTE_ADDR") or request.environ.get("REMOTE_ADDR")

    entries = load_registry()
    entries.append(entry)
    atomic_dump(entries)
    return {"ok": True, "count": len(entries[-MAX_ENTRIES:])}

@app.get("/api/list")
def api_list():
    return {"entries": load_registry()[-MAX_ENTRIES:]}

@app.get("/api/by/<uuid>")
def api_by(uuid):
    rows = [e for e in load_registry() if e.get("uuid") == uuid]
    if not rows: raise HTTPError(404, "uuid not found")
    return {"entries": rows}

@app.get("/health")
def health():
    return {"ok": True, "count": len(load_registry())}

@app.get("/")
def index():
    response.content_type = "text/html; charset=utf-8"
    entries = list(reversed(load_registry()))
    def row(e):
        u = e.get("uuid",""); iso=e.get("iso",""); log=e.get("log_url",""); src=e.get("source","")
        link = f'<a href="{log}" target="_blank">log</a>' if log else ""
        return f"<tr><td><code>{u}</code></td><td>{iso}</td><td>{link}</td><td>{src}</td></tr>"
    html = f"""<!doctype html>
<meta charset="utf-8"><title>Process Registry</title>
<style>
body{{font:14px system-ui,Arial;margin:20px;color:#111}}
table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #e5e5e5;padding:6px 8px}}
th{{background:#f7f7f7;text-align:left}}
code{{font-family:ui-monospace,Menlo,Monaco,monospace}}
a{{text-decoration:none}}
</style>
<h1>Registered Processes</h1>
<p>Total: {len(entries)}</p>
<table>
<thead><tr><th>UUID</th><th>Timestamp</th><th>Log</th><th>Source</th></tr></thead>
<tbody>{''.join(row(e) for e in entries) if entries else '<tr><td colspan="4">No entries yet</td></tr>'}</tbody>
</table>
<p><a href="/api/list">JSON API</a></p>
"""
    return html

if __name__ == "__main__":
    # Use paste server (same as env service style)
    app.run(host=HOST, port=PORT, server="paste")