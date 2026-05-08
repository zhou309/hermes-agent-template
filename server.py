"""
Hermes Agent — Railway admin server.

Responsibilities:
  - Reverse proxy at / and /* → native Hermes dashboard (hermes_cli/web_server, on 127.0.0.1:9119)
  - Managed subprocesses: `hermes gateway` (agent) and `hermes dashboard` (native UI)
  - Cookie-based session auth at /login (HMAC-signed, 7-day expiry, httponly)

Auth model: Basic Auth was dropped in favor of cookies because the Hermes React
SPA's plain fetch() calls do not reliably include basic-auth creds across browsers.
Cookies auto-include on every same-origin request. The cookie signing secret is
regenerated on every process start, so any ADMIN_PASSWORD change on Railway
(which triggers a redeploy) invalidates all existing sessions.

Configuration is provided exclusively via Railway environment variables —
LLM_MODEL, the provider API keys, and channel tokens. There is no in-app
setup wizard; the gateway autostarts on boot if config is complete.
"""

import asyncio
import os
import re
import secrets
import signal
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import websockets
import websockets.exceptions
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
)
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect, WebSocketState

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")

HERMES_HOME = os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))
ENV_FILE = Path(HERMES_HOME) / ".env"

# Native Hermes dashboard — runs on loopback, fronted by our reverse proxy.
HERMES_DASHBOARD_HOST = "127.0.0.1"
HERMES_DASHBOARD_PORT = int(os.environ.get("HERMES_DASHBOARD_PORT", "9119"))
HERMES_DASHBOARD_URL = f"http://{HERMES_DASHBOARD_HOST}:{HERMES_DASHBOARD_PORT}"

# Mirror dashboard-ref-only/auth_proxy.py: strip only `host` (httpx sets it)
# and `transfer-encoding` (httpx recomputes it from the body). Keep everything
# else — notably `authorization`, because the SPA uses Bearer tokens against
# hermes's own /api/env/reveal and OAuth endpoints, and keep `cookie` since
# some hermes endpoints read it. Aggressive stripping was masking requests in
# ways that produced spurious 401s.
HOP_BY_HOP = {"host", "transfer-encoding"}

ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")
if not ADMIN_PASSWORD:
    ADMIN_PASSWORD = secrets.token_urlsafe(16)
    print(f"[server] Admin credentials — username: {ADMIN_USERNAME}  password: {ADMIN_PASSWORD}", flush=True)
else:
    print(f"[server] Admin username: {ADMIN_USERNAME}", flush=True)

# Provider API-key env vars — gateway needs at least one of these set.
PROVIDER_KEYS = [
    "OPENROUTER_API_KEY",
    "DEEPSEEK_API_KEY",
    "DASHSCOPE_API_KEY",
    "GLM_API_KEY",
    "KIMI_API_KEY",
    "MINIMAX_API_KEY",
    "HF_TOKEN",
    "NVIDIA_API_KEY",
    "ARCEE_API_KEY",
    "STEPFUN_API_KEY",
    "AI_GATEWAY_API_KEY",
    "GEMINI_API_KEY",
]


# ── .env helpers ──────────────────────────────────────────────────────────────
def read_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ('"', "'"):
            v = v[1:-1]
        out[k.strip()] = v
    return out


def write_config_yaml(data: dict[str, str]) -> None:
    """Write a minimal config.yaml so hermes picks up the model and provider."""
    model = data.get("LLM_MODEL", "")
    config_path = Path(HERMES_HOME) / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(f"""\
model:
  default: "{model}"
  provider: "auto"

terminal:
  backend: "local"
  timeout: 60
  cwd: "/tmp"

agent:
  max_iterations: 50

data_dir: "{HERMES_HOME}"
""")


def is_config_complete(data: dict[str, str] | None = None) -> bool:
    """Return True when LLM_MODEL plus at least one provider key are set.

    Reads from .env first, then falls back to the process environment so
    Railway service variables count even before any .env is written.
    """
    if data is None:
        data = read_env(ENV_FILE)
    has_model = bool(data.get("LLM_MODEL") or os.environ.get("LLM_MODEL"))
    has_provider = any(data.get(k) or os.environ.get(k) for k in PROVIDER_KEYS)
    return has_model and has_provider


# ── Auth (cookie-based) ───────────────────────────────────────────────────────
# We use HMAC-signed cookies instead of HTTP Basic Auth because:
#   1. Browser behavior for sending Basic auth on XHR/fetch is inconsistent;
#      the Hermes React SPA's plain fetch() calls don't reliably include it,
#      causing every proxied API call to 401.
# Cookies are auto-included on every same-origin request (navigation + XHR)
# so the proxied Hermes dashboard works with one login.
#
# The SECRET is regenerated on every process start. That means any ADMIN_PASSWORD
# change via Railway → redeploy → all existing cookies invalidate → users re-login.
import hashlib as _hashlib
import hmac as _hmac
from urllib.parse import quote as _url_quote, urlparse as _urlparse

COOKIE_NAME = "hermes_auth"
COOKIE_MAX_AGE = 7 * 86400  # 7 days
COOKIE_SECRET = secrets.token_bytes(32)

# Public paths — no auth required. Everything else is behind the cookie gate.
PUBLIC_PATHS = {"/health", "/login", "/logout"}


def _make_auth_token() -> str:
    """Build a cookie value: `<expires>.<hmac-sha256>`."""
    expires = str(int(time.time()) + COOKIE_MAX_AGE)
    sig = _hmac.new(COOKIE_SECRET, expires.encode(), _hashlib.sha256).hexdigest()
    return f"{expires}.{sig}"


def _verify_auth_token(token: str) -> bool:
    try:
        expires_s, sig = token.rsplit(".", 1)
        if int(expires_s) < time.time():
            return False
        expected = _hmac.new(COOKIE_SECRET, expires_s.encode(), _hashlib.sha256).hexdigest()
        return _hmac.compare_digest(sig, expected)
    except Exception:
        return False


def _is_authenticated(request: Request) -> bool:
    return _verify_auth_token(request.cookies.get(COOKIE_NAME, ""))


def _safe_return_to(value: str) -> str:
    """Reject open-redirect attempts — only allow same-origin relative paths."""
    if not value or not value.startswith("/") or value.startswith("//"):
        return "/"
    p = _urlparse(value)
    if p.scheme or p.netloc:
        return "/"
    return value


def guard(request: Request) -> Response | None:
    """Enforce auth on protected routes.

    - HTML navigation: 302 to /login?returnTo=<path>
    - API / XHR: 401 JSON (so the SPA's fetch() can surface it cleanly)
    """
    if _is_authenticated(request):
        return None
    accept = request.headers.get("accept", "").lower()
    wants_html = "text/html" in accept
    if wants_html:
        rt = request.url.path
        if request.url.query:
            rt = f"{rt}?{request.url.query}"
        return RedirectResponse(f"/login?returnTo={_url_quote(rt)}", status_code=302)
    return JSONResponse({"error": "Unauthorized"}, status_code=401)


LOGIN_PAGE_HTML = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hermes Agent — Sign in</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0d0f14;color:#c9d1d9;font-family:'IBM Plex Sans',sans-serif;
  min-height:100vh;display:flex;align-items:center;justify-content:center;padding:20px}
.card{background:#14181f;border:1px solid #252d3d;border-radius:12px;padding:36px 32px;width:100%;max-width:380px;
  box-shadow:0 20px 40px rgba(0,0,0,0.4)}
.brand{text-align:center;margin-bottom:28px}
.brand-logo{display:inline-flex;align-items:center;gap:10px;font-family:'IBM Plex Mono',monospace;font-weight:600;font-size:18px;color:#6272ff}
.brand-logo span{color:#6b7688;font-weight:400}
.brand-sub{font-family:'IBM Plex Mono',monospace;font-size:11px;color:#6b7688;margin-top:8px;letter-spacing:1.5px;text-transform:uppercase}
label{display:block;font-family:'IBM Plex Mono',monospace;font-size:11px;color:#6b7688;
  letter-spacing:0.05em;text-transform:uppercase;margin-bottom:6px;margin-top:16px}
input{width:100%;background:#0d0f14;border:1px solid #252d3d;border-radius:6px;color:#c9d1d9;
  font-family:'IBM Plex Mono',monospace;font-size:13px;padding:9px 11px;outline:none;transition:border-color .15s}
input:focus{border-color:#6272ff}
button{width:100%;margin-top:24px;background:#6272ff;border:1px solid #6272ff;border-radius:6px;color:#fff;
  font-family:'IBM Plex Mono',monospace;font-size:13px;font-weight:500;padding:10px;cursor:pointer;
  transition:background .15s,border-color .15s}
button:hover{background:#7b8fff;border-color:#7b8fff}
.err{background:rgba(248,81,73,0.08);border:1px solid rgba(248,81,73,0.3);border-radius:6px;
  color:#f85149;font-family:'IBM Plex Mono',monospace;font-size:12px;padding:8px 12px;margin-bottom:14px;text-align:center}
.footnote{margin-top:18px;font-family:'IBM Plex Mono',monospace;font-size:10px;color:#6b7688;text-align:center;line-height:1.6}
</style></head>
<body>
<div class="card">
  <div class="brand">
    <div class="brand-logo">hermes<span>/admin</span></div>
    <div class="brand-sub">Sign in to continue</div>
  </div>
  __ERROR__
  <form method="POST" action="/login">
    <input type="hidden" name="returnTo" value="__RETURN_TO__">
    <label for="username">Username</label>
    <input id="username" name="username" type="text" autocomplete="username" autofocus required>
    <label for="password">Password</label>
    <input id="password" name="password" type="password" autocomplete="current-password" required>
    <button type="submit">Sign in</button>
  </form>
  <p class="footnote">Credentials are the <code>ADMIN_USERNAME</code> and <code>ADMIN_PASSWORD</code><br>Railway service variables.</p>
</div>
</body></html>"""


MISSION_CONTROL_HTML = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mission Control — Hermes</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&family=Press+Start+2P&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:
  radial-gradient(circle at 50% 0%, rgba(124,140,255,.16), transparent 30%),
  radial-gradient(circle at 15% 22%, rgba(135,255,196,.08), transparent 18%),
  linear-gradient(180deg,#0a1020 0%, #0c1430 45%, #09101d 100%);
  color:#e5ebff;font-family:'IBM Plex Sans',sans-serif;min-height:100vh;position:relative;overflow-x:hidden}
body::before{content:"";position:fixed;inset:0;pointer-events:none;background:
  repeating-linear-gradient(180deg, rgba(255,255,255,.03) 0 1px, transparent 1px 4px),
  repeating-linear-gradient(90deg, rgba(255,255,255,.02) 0 1px, transparent 1px 52px);
  opacity:.12;mix-blend-mode:screen}
body::after{content:"";position:fixed;left:0;right:0;bottom:0;height:260px;pointer-events:none;background:
  radial-gradient(circle at 18% 28%, rgba(126,255,188,.10), transparent 18%),
  radial-gradient(circle at 78% 22%, rgba(124,140,255,.10), transparent 20%),
  radial-gradient(circle at 50% 72%, rgba(255,210,122,.08), transparent 18%);}
.wrap{max-width:1240px;margin:0 auto;padding:28px 20px 40px}
.top{display:flex;justify-content:space-between;align-items:center;gap:16px;flex-wrap:wrap;margin-bottom:22px}
.brand{font-family:'Press Start 2P','IBM Plex Mono',monospace;font-size:11px;line-height:1.55;letter-spacing:.08em;text-transform:uppercase;color:#8ea0ff}
.brand b{color:#fff}
.pill{border:1px solid rgba(142,160,255,.28);background:rgba(142,160,255,.08);padding:8px 12px;border-radius:999px;font-family:'Press Start 2P','IBM Plex Mono',monospace;font-size:10px;line-height:1.45;color:#cdd6ff}
.hero{display:grid;grid-template-columns:1.12fr .88fr;gap:18px;margin-bottom:18px}
.card{background:linear-gradient(180deg,rgba(17,23,44,.98),rgba(10,14,26,.98));border:1px solid rgba(142,160,255,.18);border-radius:18px;box-shadow:0 24px 60px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.03)}
.hero-left{padding:30px}
.eyebrow{font-family:'Press Start 2P','IBM Plex Mono',monospace;font-size:10px;line-height:1.55;color:#97a8ff;letter-spacing:.12em;text-transform:uppercase;margin-bottom:16px}
h1{font-size:44px;line-height:1.03;margin-bottom:12px;letter-spacing:-.04em}
.subtitle{font-size:16px;line-height:1.75;color:#b8c4f5;max-width:58ch;margin-bottom:20px}
.actions{display:flex;gap:12px;flex-wrap:wrap}
.btn{display:inline-flex;align-items:center;gap:8px;padding:11px 14px;border-radius:12px;text-decoration:none;font-family:'Press Start 2P','IBM Plex Mono',monospace;font-size:10px;line-height:1.4;font-weight:600;letter-spacing:.06em;text-transform:uppercase}
.btn.primary{background:linear-gradient(180deg,#9fb0ff,#7488ff);color:#081225}
.btn.secondary{border:1px solid rgba(142,160,255,.3);background:rgba(255,255,255,.02);color:#e5ebff}
.btn.secondary:hover{background:rgba(124,140,255,.12)}
.hero-right{padding:22px}
.stat{padding:16px;border-radius:14px;background:linear-gradient(180deg,rgba(255,255,255,.05),rgba(255,255,255,.02));border:1px solid rgba(142,160,255,.12);margin-bottom:12px;box-shadow:inset 0 0 0 1px rgba(255,255,255,.03)}
.stat-label{font-family:'Press Start 2P','IBM Plex Mono',monospace;font-size:10px;line-height:1.55;letter-spacing:.08em;text-transform:uppercase;color:#92a3ff;margin-bottom:8px}
.stat-value{font-size:20px;font-weight:700;line-height:1.4}
.stat-sub{font-size:13px;color:#b8c4f5;line-height:1.6;margin-top:6px}
.grid{display:grid;grid-template-columns:1.18fr .82fr;gap:18px}
.section{padding:20px}
.section h2{font-size:18px;margin-bottom:14px}
.hub{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}
.node{padding:14px;border-radius:14px;background:linear-gradient(180deg,rgba(255,255,255,.05),rgba(255,255,255,.02));border:1px solid rgba(142,160,255,.12);position:relative;overflow:hidden}
.node::before{content:'';position:absolute;inset:0;background:linear-gradient(135deg,transparent 0 48%, rgba(255,255,255,.06) 48% 52%, transparent 52% 100%);opacity:.65}
.node strong{display:block;font-size:15px;margin-bottom:4px;position:relative;z-index:1}
.node p{font-size:13px;color:#b8c4f5;line-height:1.55;position:relative;z-index:1}
.room-list{display:flex;flex-direction:column;gap:10px}
.room{display:flex;justify-content:space-between;gap:12px;align-items:center;padding:14px;border-radius:16px;background:linear-gradient(180deg,rgba(255,255,255,.05),rgba(255,255,255,.02));border:1px solid rgba(142,160,255,.12);text-decoration:none;color:inherit;box-shadow:inset 0 1px 0 rgba(255,255,255,.03)}
.room span{font-size:13px;color:#b8c4f5}
.room code{font-family:'Press Start 2P','IBM Plex Mono',monospace;color:#97a8ff;font-size:9px;line-height:1.4}
.hub-map{margin-top:18px;padding:18px;border-radius:18px;background:
  radial-gradient(circle at center, rgba(124,140,255,.12), transparent 55%),
  linear-gradient(180deg, rgba(245,220,158,.88), rgba(210,180,120,.88));
  border:1px solid rgba(142,160,255,.12);position:relative;overflow:hidden}
.hub-map::before{content:'';position:absolute;inset:18px;border-radius:16px;background:
  linear-gradient(90deg, transparent 0 16.5%, rgba(94,71,34,.26) 16.5% 22%, transparent 22% 44%, rgba(94,71,34,.26) 44% 56%, transparent 56% 78%, rgba(94,71,34,.26) 78% 83.5%, transparent 83.5% 100%),
  linear-gradient(180deg, transparent 0 16.5%, rgba(94,71,34,.26) 16.5% 22%, transparent 22% 44%, rgba(94,71,34,.26) 44% 56%, transparent 56% 78%, rgba(94,71,34,.26) 78% 83.5%, transparent 83.5% 100%),
  linear-gradient(180deg, rgba(94,71,34,.08), rgba(255,255,255,.04));
  pointer-events:none;opacity:.92}
.hub-map::after{content:'';position:absolute;inset:0;background:radial-gradient(circle at 50% 50%, rgba(255,255,255,.10), transparent 26%);pointer-events:none}
.hub-roads{position:absolute;inset:18px;pointer-events:none;z-index:0}
.route{position:absolute;background:linear-gradient(180deg,#9f7e46,#6f5021);box-shadow:0 0 0 3px rgba(255,248,210,.16);border-radius:999px;overflow:hidden;animation:roadPulse 3.8s ease-in-out infinite}
.route::after{content:'';position:absolute;inset:4px;background:repeating-linear-gradient(90deg, rgba(255,255,255,.25) 0 8px, transparent 8px 18px);border-radius:inherit;opacity:.46}
.route .walker{position:absolute;width:12px;height:12px;border-radius:4px;background:linear-gradient(180deg,#fff8cb,#f7c948);border:2px solid #48370e;box-shadow:0 0 0 3px rgba(255,248,210,.12), 0 0 12px rgba(255,255,255,.18)}
.route.up .walker,.route.down .walker{left:50%;transform:translateX(-50%);animation:walkVertical 2.8s linear infinite}
.route.left .walker,.route.right .walker{top:50%;transform:translateY(-50%);animation:walkHorizontal 3s linear infinite}
.route.up .walker{top:-8px}
.route.down .walker{bottom:-8px;animation-direction:reverse}
.route.left .walker{left:-8px}
.route.right .walker{right:-8px;animation-direction:reverse}
.route.up{left:50%;top:14%;transform:translateX(-50%);width:18px;height:26%}
.route.down{left:50%;bottom:14%;transform:translateX(-50%);width:18px;height:26%}
.route.left{top:50%;left:14%;transform:translateY(-50%);width:26%;height:18px}
.route.right{top:50%;right:14%;transform:translateY(-50%);width:26%;height:18px}
.hub-grid{position:relative;z-index:1;display:grid;grid-template-columns:repeat(3,1fr);gap:12px;align-items:stretch}
.tile{min-height:112px;border-radius:16px;border:2px solid;padding:12px;display:flex;flex-direction:column;justify-content:space-between;background:rgba(255,255,255,.06);box-shadow:0 8px 18px rgba(0,0,0,.12);position:relative;overflow:hidden}
.tile::before{content:'';position:absolute;left:12px;right:12px;top:10px;height:10px;border-radius:999px;background:rgba(255,255,255,.16)}
.tile::after{content:'';position:absolute;inset:8px;border-radius:12px;border:1px solid rgba(255,255,255,.18);pointer-events:none}
.tile .name{font-size:15px;font-weight:700;letter-spacing:.04em;position:relative;z-index:1}
.tile .sub{font-size:11px;font-family:'Press Start 2P','IBM Plex Mono',monospace;letter-spacing:.08em;text-transform:uppercase;opacity:.84;line-height:1.5;position:relative;z-index:1}
.sprite{position:absolute;right:10px;top:10px;z-index:2;width:28px;height:28px;border-radius:8px;border:2px solid rgba(20,20,30,.78);box-shadow:0 6px 14px rgba(0,0,0,.18), inset 0 0 0 1px rgba(255,255,255,.2);display:flex;align-items:center;justify-content:center;font-family:'Press Start 2P','IBM Plex Mono',monospace;font-size:8px;line-height:1;background:#fff;image-rendering:pixelated;animation:spriteBob 1.8s ease-in-out infinite}
.sprite::before{content:'';position:absolute;inset:4px;border-radius:5px;background:repeating-linear-gradient(90deg, rgba(255,255,255,.22) 0 3px, transparent 3px 6px), linear-gradient(180deg, rgba(0,0,0,.12), rgba(255,255,255,.18));opacity:.8}
.sprite span{position:relative;z-index:1;color:#142031;text-shadow:0 1px 0 rgba(255,255,255,.3)}
.tile.empty{border-style:dashed;background:rgba(255,255,255,.03);color:#95a0bc}
.tile.hq{grid-column:2;grid-row:2;min-height:164px;background:linear-gradient(180deg,#efe6ff,#d8c8ff);color:#1b1230;border-color:#7757b7;animation:hqPulse 2.8s ease-in-out infinite}
.tile.h2{grid-column:2;grid-row:1;background:linear-gradient(180deg,#d8f1d2,#bfe3b5);color:#17311a;border-color:#4f7f4d;animation:floatTile 6s ease-in-out infinite}
.tile.pro{grid-column:1;grid-row:2;background:linear-gradient(180deg,#dbeaf8,#bcd7ef);color:#173044;border-color:#4c6f93;animation:floatTile 6.4s ease-in-out infinite}
.tile.terra{grid-column:3;grid-row:2;background:linear-gradient(180deg,#f8e1be,#f1cc9d);color:#4a2c08;border-color:#a56a2a;animation:floatTile 6.8s ease-in-out infinite}
.tile.nw{grid-column:1;grid-row:1}
.tile.ne{grid-column:3;grid-row:1}
.tile.sw{grid-column:1;grid-row:3}
.tile.s{grid-column:2;grid-row:3}
.tile.se{grid-column:3;grid-row:3}
.tile .agents{display:flex;flex-wrap:wrap;gap:6px;position:relative;z-index:1}
.chip{display:inline-flex;align-items:center;justify-content:center;min-width:72px;padding:7px 10px;border-radius:999px;font-family:'IBM Plex Mono',monospace;font-size:11px;font-weight:700;background:rgba(255,255,255,.84);border:1px solid rgba(0,0,0,.12);box-shadow:0 3px 8px rgba(0,0,0,.08)}
.tile.empty .chip{background:rgba(255,255,255,.45)}
@keyframes hqPulse{0%,100%{box-shadow:0 0 0 0 rgba(125,90,210,.18), 0 12px 20px rgba(0,0,0,.18)}50%{box-shadow:0 0 0 10px rgba(125,90,210,.06), 0 16px 26px rgba(0,0,0,.22)}}
@keyframes floatTile{0%,100%{transform:translateY(0)}50%{transform:translateY(-1px)}}
@keyframes roadPulse{0%,100%{opacity:.80}50%{opacity:1}}
@keyframes spriteBob{0%,100%{transform:translateY(0)}50%{transform:translateY(-2px)}}
@keyframes walkVertical{0%{top:-8px}100%{top:calc(100% - 4px)}}
@keyframes walkHorizontal{0%{left:-8px}100%{left:calc(100% - 4px)}}
@media (max-width: 960px){.hero,.grid{grid-template-columns:1fr}h1{font-size:34px}.hub-grid{grid-template-columns:1fr 1fr}.tile.hq{grid-column:1 / -1;grid-row:auto}.tile.h2,.tile.pro,.tile.terra,.tile.nw,.tile.ne,.tile.sw,.tile.s,.tile.se{grid-column:auto;grid-row:auto}.route{display:none}}
</style></head>
<body>
<div class="wrap">
  <div class="top">
    <div class="brand"><b>Mission Control</b> / Hermes</div>
    <div class="pill">Pokémon hub map · HQ center · company pods around it</div>
  </div>

  <div class="hero">
    <div class="card hero-left">
      <div class="eyebrow">Portfolio command center</div>
      <h1>Run the shared agents from one hub.</h1>
      <p class="subtitle">Mission Control is the top-level entry point. HQ sits in the middle. The three company rooms orbit it like a Pokémon hub town, and the remaining slots stay open for future companies.</p>
      <div class="actions">
        <a class="btn primary" href="/native">Open native dashboard</a>
        <a class="btn secondary" href="/rooms">View rooms</a>
      </div>
    </div>
    <div class="card hero-right">
      <div class="stat">
        <div class="stat-label">Shared agents</div>
        <div class="stat-value">CEO · CFO · CTO</div>
        <div class="stat-sub">These live above the pods and coordinate the portfolio.</div>
      </div>
      <div class="stat">
        <div class="stat-label">Operating rule</div>
        <div class="stat-value">One room per company</div>
        <div class="stat-sub">Keep each brand isolated, with HQ as the shared center.</div>
      </div>
    </div>
  </div>

  <div class="grid">
    <div class="card section">
      <h2>Shared hub</h2>
      <div class="hub">
        <div class="node"><strong>CFO</strong><p>Cash flow, finance, and portfolio visibility.</p></div>
        <div class="node"><strong>CTO</strong><p>Infrastructure, tooling, and systems architecture.</p></div>
        <div class="node"><strong>Operator</strong><p>Execution, follow-through, and project coordination.</p></div>
        <div class="node"><strong>Research</strong><p>Discovery, content, and opportunity scouting.</p></div>
      </div>
    </div>
    <div class="card section">
      <h2>Rooms</h2>
      <div class="room-list">
        <a class="room" href="/rooms/h2waders"><div><strong>H2 Waders</strong><span>Waitlist-phase brand room</span></div><code>/rooms/h2waders</code></a>
        <a class="room" href="/rooms/pro-fulfill"><div><strong>Pro Fulfill</strong><span>Company-specific workspace</span></div><code>/rooms/pro-fulfill</code></a>
        <a class="room" href="/rooms/terache-tires"><div><strong>Terache Tires</strong><span>Company-specific workspace</span></div><code>/rooms/terache-tires</code></a>
      </div>
    </div>
  </div>

  <div class="card section hub-map">
    <h2>Top-down hub map</h2>
    <div class="hub-roads">
      <span class="route up"><span class="walker"></span></span>
      <span class="route left"><span class="walker"></span></span>
      <span class="route right"><span class="walker"></span></span>
      <span class="route down"><span class="walker"></span></span>
    </div>
    <div class="hub-grid">
      <div class="tile empty nw"><div><div class="name">EMPTY SLOT</div><div class="sub">future company</div></div><div class="agents"><span class="chip">LOCKED</span><span class="chip">FUTURE</span></div></div>
      <div class="tile h2"><div class="sprite"><span>H2</span></div><div><div class="name">H2 WADERS</div><div class="sub">swamp zone</div></div><div class="agents"><span class="chip">pod</span><span class="chip">route → HQ</span></div></div>
      <div class="tile empty ne"><div><div class="name">EMPTY SLOT</div><div class="sub">future company</div></div><div class="agents"><span class="chip">LOCKED</span><span class="chip">FUTURE</span></div></div>
      <div class="tile pro"><div class="sprite"><span>PF</span></div><div><div class="name">PRO FULFILL</div><div class="sub">warehouse zone</div></div><div class="agents"><span class="chip">pod</span><span class="chip">route → HQ</span></div></div>
      <div class="tile hq"><div class="sprite"><span>HQ</span></div><div><div class="name">HQ ROOM</div><div class="sub">all agents report here</div></div><div class="agents"><span class="chip">CEO</span><span class="chip">CTO</span><span class="chip">CFO</span><span class="chip">OPS</span><span class="chip">CONTENT</span><span class="chip">RESEARCH</span><span class="chip">SALES</span><span class="chip">SUPPORT</span></div></div>
      <div class="tile terra"><div class="sprite"><span>TT</span></div><div><div class="name">TERACHE TIRES</div><div class="sub">garage zone</div></div><div class="agents"><span class="chip">pod</span><span class="chip">route → HQ</span></div></div>
      <div class="tile empty sw"><div><div class="name">EMPTY SLOT</div><div class="sub">future company</div></div><div class="agents"><span class="chip">LOCKED</span><span class="chip">FUTURE</span></div></div>
      <div class="tile empty s"><div><div class="name">EMPTY SLOT</div><div class="sub">future company</div></div><div class="agents"><span class="chip">LOCKED</span><span class="chip">FUTURE</span></div></div>
      <div class="tile empty se"><div><div class="name">EMPTY SLOT</div><div class="sub">future company</div></div><div class="agents"><span class="chip">LOCKED</span><span class="chip">FUTURE</span></div></div>
    </div>
  </div>
</div>
</body></html>"""


def _room_meta(room_key: str) -> tuple[str, str, str]:
    rooms = {
        "h2waders": ("H2 Waders", "Waitlist-phase brand room", "Top-down company pod with a muddy swamp feel. All routes point back to HQ."),
        "pro-fulfill": ("Pro Fulfill", "Operations room", "Top-down company pod for logistics and execution. All routes point back to HQ."),
        "terache-tires": ("Terache Tires", "Commerce room", "Top-down company pod for tire and auto work. All routes point back to HQ."),
    }
    return rooms.get(room_key, (room_key.replace("-", " ").title(), "Company room", "Isolated workspace"))


def _room_html(room_key: str) -> str:
    title, subtitle, detail = _room_meta(room_key)
    board = {
        "h2waders": ["CEO", "Ops", "Research", "Content"],
        "pro-fulfill": ["Ops", "Inventory", "Support", "Sales"],
        "terache-tires": ["Sales", "Ops", "Support", "Growth"],
    }.get(room_key, ["Agent 1", "Agent 2", "Agent 3", "Agent 4"])
    chips = "".join(f'<span class="chip">{_html_escape(agent)}</span>' for agent in board)
    return f"""<!DOCTYPE html>
<html lang=\"en\"><head>
<meta charset=\"UTF-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>{_html_escape(title)} — Mission Control</title>
<link rel=\"preconnect\" href=\"https://fonts.googleapis.com\">
<link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&family=Press+Start+2P&display=swap" rel="stylesheet">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:
  radial-gradient(circle at 50% 0%, rgba(124,140,255,.16), transparent 30%),
  radial-gradient(circle at 15% 22%, rgba(135,255,196,.08), transparent 18%),
  linear-gradient(180deg,#0a1020 0%, #0c1430 45%, #09101d 100%);
  color:#e5ebff;font-family:'IBM Plex Sans',sans-serif;min-height:100vh;position:relative;overflow-x:hidden}}
body::before{{content:'';position:fixed;inset:0;pointer-events:none;background:
  repeating-linear-gradient(180deg, rgba(255,255,255,.03) 0 1px, transparent 1px 4px),
  repeating-linear-gradient(90deg, rgba(255,255,255,.02) 0 1px, transparent 1px 52px);
  opacity:.12;mix-blend-mode:screen}}
.wrap{{max-width:960px;margin:0 auto;padding:28px 20px 40px}}
.card{{background:linear-gradient(180deg,rgba(17,23,44,.98),rgba(10,14,26,.98));border:1px solid rgba(142,160,255,.18);border-radius:18px;padding:24px;box-shadow:0 24px 60px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.03)}}
.top{{display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap;align-items:center;margin-bottom:18px}} .brand{{font-family:'Press Start 2P','IBM Plex Mono',monospace;font-size:10px;line-height:1.5;letter-spacing:.08em;text-transform:uppercase;color:#8ea0ff}}
.back{{font-family:'Press Start 2P','IBM Plex Mono',monospace;color:#97a8ff;text-decoration:none;font-size:9px;line-height:1.4}}
h1{{font-size:36px;line-height:1.05;margin-bottom:10px;letter-spacing:-.03em}} p{{color:#b8c4f5;line-height:1.7;margin-bottom:14px}} .meta{{display:flex;gap:10px;flex-wrap:wrap;margin:18px 0}} .pill{{padding:8px 12px;border-radius:999px;background:rgba(124,140,255,.12);border:1px solid rgba(124,140,255,.18);font-family:'Press Start 2P','IBM Plex Mono',monospace;font-size:9px;line-height:1.4}}
.actions{{display:flex;gap:12px;flex-wrap:wrap;margin-top:18px}} a{{color:inherit;text-decoration:none}} .btn{{padding:11px 14px;border-radius:12px;font-family:'Press Start 2P','IBM Plex Mono',monospace;font-size:9px;font-weight:600;letter-spacing:.06em;text-transform:uppercase}} .primary{{background:linear-gradient(180deg,#9fb0ff,#7488ff);color:#07101f}} .secondary{{border:1px solid rgba(142,160,255,.25);background:rgba(255,255,255,.02)}}
.room-map{{margin-top:18px;padding:18px;border-radius:18px;background:
  radial-gradient(circle at center, rgba(124,140,255,.10), transparent 52%),
  linear-gradient(180deg, rgba(245,220,158,.88), rgba(210,180,120,.88));
  border:1px solid rgba(142,160,255,.12);position:relative;overflow:hidden}}
.room-map::before{{content:'';position:absolute;inset:18px;border-radius:16px;background:
  linear-gradient(90deg, transparent 0 16.5%, rgba(94,71,34,.26) 16.5% 22%, transparent 22% 44%, rgba(94,71,34,.26) 44% 56%, transparent 56% 78%, rgba(94,71,34,.26) 78% 83.5%, transparent 83.5% 100%),
  linear-gradient(180deg, transparent 0 16.5%, rgba(94,71,34,.26) 16.5% 22%, transparent 22% 44%, rgba(94,71,34,.26) 44% 56%, transparent 56% 78%, rgba(94,71,34,.26) 78% 83.5%, transparent 83.5% 100%),
  linear-gradient(180deg, rgba(94,71,34,.08), rgba(255,255,255,.04));
  pointer-events:none;opacity:.92}}
.room-map::after{{content:'';position:absolute;inset:0;background:radial-gradient(circle at 50% 50%, rgba(255,255,255,.10), transparent 26%);pointer-events:none}}
.room-roads{{position:absolute;inset:18px;pointer-events:none;z-index:0}}
.route{{position:absolute;background:linear-gradient(180deg,#9f7e46,#6f5021);box-shadow:0 0 0 3px rgba(255,248,210,.16);border-radius:999px;overflow:hidden;animation:roadPulse 3.8s ease-in-out infinite}}
.route::after{{content:'';position:absolute;inset:4px;background:repeating-linear-gradient(90deg, rgba(255,255,255,.25) 0 8px, transparent 8px 18px);border-radius:inherit;opacity:.46}}
.route.up{{left:50%;top:14%;transform:translateX(-50%);width:18px;height:26%}}
.route.down{{left:50%;bottom:14%;transform:translateX(-50%);width:18px;height:26%}}
.route.left{{top:50%;left:14%;transform:translateY(-50%);width:26%;height:18px}}
.route.right{{top:50%;right:14%;transform:translateY(-50%);width:26%;height:18px}}
.room-grid{{position:relative;z-index:1;display:grid;grid-template-columns:repeat(3,1fr);gap:12px;align-items:stretch}}
.tile{{min-height:120px;border-radius:16px;border:2px solid;padding:12px;display:flex;flex-direction:column;justify-content:space-between;background:rgba(255,255,255,.06);box-shadow:0 8px 18px rgba(0,0,0,.12);position:relative;overflow:hidden}}
.tile::before{{content:'';position:absolute;left:12px;right:12px;top:10px;height:10px;border-radius:999px;background:rgba(255,255,255,.16)}}
.tile::after{{content:'';position:absolute;inset:8px;border-radius:12px;border:1px solid rgba(255,255,255,.18);pointer-events:none}}
.tile .name{{font-size:15px;font-weight:700;letter-spacing:.04em;position:relative;z-index:1}} .tile .sub{{font-size:11px;font-family:'Press Start 2P','IBM Plex Mono',monospace;letter-spacing:.08em;text-transform:uppercase;opacity:.84;line-height:1.5;position:relative;z-index:1}}
.tile.hub{{grid-column:2;grid-row:2;min-height:170px;background:linear-gradient(180deg,#efe6ff,#d8c8ff);color:#1b1230;border-color:#7757b7;animation:hqPulse 2.8s ease-in-out infinite}}
.tile.empty{{border-style:dashed;background:rgba(255,255,255,.03);color:#95a0bc}}
.tile.h2{{grid-column:2;grid-row:1;background:linear-gradient(180deg,#d8f1d2,#bfe3b5);color:#17311a;border-color:#4f7f4d;animation:floatTile 6s ease-in-out infinite}}
.tile.pro{{grid-column:1;grid-row:2;background:linear-gradient(180deg,#dbeaf8,#bcd7ef);color:#173044;border-color:#4c6f93;animation:floatTile 6.4s ease-in-out infinite}}
.tile.terra{{grid-column:3;grid-row:2;background:linear-gradient(180deg,#f8e1be,#f1cc9d);color:#4a2c08;border-color:#a56a2a;animation:floatTile 6.8s ease-in-out infinite}}
.tile.nw{{grid-column:1;grid-row:1}} .tile.ne{{grid-column:3;grid-row:1}} .tile.sw{{grid-column:1;grid-row:3}} .tile.s{{grid-column:2;grid-row:3}} .tile.se{{grid-column:3;grid-row:3}}
.tile .agents{{display:flex;flex-wrap:wrap;gap:6px;position:relative;z-index:1}}
.chip{{display:inline-flex;align-items:center;justify-content:center;min-width:72px;padding:7px 10px;border-radius:999px;font-family:'IBM Plex Mono',monospace;font-size:11px;font-weight:700;background:rgba(255,255,255,.84);border:1px solid rgba(0,0,0,.12);box-shadow:0 3px 8px rgba(0,0,0,.08)}}
.tile.empty .chip{{background:rgba(255,255,255,.45)}}
@keyframes hqPulse{{0%,100%{{box-shadow:0 0 0 0 rgba(125,90,210,.18), 0 12px 20px rgba(0,0,0,.18)}}50%{{box-shadow:0 0 0 10px rgba(125,90,210,.06), 0 16px 26px rgba(0,0,0,.22)}}}}
@keyframes floatTile{{0%,100%{{transform:translateY(0)}}50%{{transform:translateY(-1px)}}}}
@keyframes roadPulse{{0%,100%{{opacity:.80}}50%{{opacity:1}}}}
@media (max-width: 960px){{.room-grid{{grid-template-columns:1fr 1fr}}.tile.hub{{grid-column:1 / -1;grid-row:auto}}.tile.h2,.tile.pro,.tile.terra,.tile.nw,.tile.ne,.tile.sw,.tile.s,.tile.se{{grid-column:auto;grid-row:auto}}.route{{display:none}}}}
</style></head><body>
<div class=\"wrap\">
  <div class=\"top\"><div class=\"brand\">Mission Control / Room</div><a class=\"back\" href=\"/\">← Back to Mission Control</a></div>
  <div class=\"card\">
    <h1>{_html_escape(title)}</h1>
    <p>{_html_escape(subtitle)}</p>
    <p>{_html_escape(detail)}</p>
    <div class=\"meta\"><div class=\"pill\">Room key: {_html_escape(room_key)}</div><div class=\"pill\">Isolation: enabled</div><div class=\"pill\">Native dashboard: /native</div></div>
    <div class=\"actions\"><a class=\"btn primary\" href=\"/native\">Open native dashboard</a><a class=\"btn secondary\" href=\"/rooms\">All rooms</a></div>
    <div class=\"room-map\">
      <h2 style=\"margin-bottom:14px;font-size:18px;\">Top-down room view</h2>
      <div class=\"room-roads\">
        <span class=\"route up\"><span class=\"walker\"></span></span>
        <span class=\"route left\"><span class=\"walker\"></span></span>
        <span class=\"route right\"><span class=\"walker\"></span></span>
        <span class=\"route down\"><span class=\"walker\"></span></span>
      </div>
      <div class=\"room-grid\">
        <div class=\"tile empty nw\"><div><div class=\"name\">EMPTY SLOT</div><div class=\"sub\">future company</div></div><div class=\"agents\"><span class=\"chip\">LOCKED</span></div></div>
        <div class=\"tile h2\"><div class=\"sprite\"><span>{_html_escape(title[:2].upper())}</span></div><div><div class=\"name\">{_html_escape(title)}</div><div class=\"sub\">{_html_escape(subtitle)}</div></div><div class=\"agents\">{chips}</div></div>
        <div class=\"tile empty ne\"><div><div class=\"name\">EMPTY SLOT</div><div class=\"sub\">future company</div></div><div class=\"agents\"><span class=\"chip\">LOCKED</span></div></div>
        <div class=\"tile pro\"><div class=\"sprite\"><span>PF</span></div><div><div class=\"name\">PRO FULFILL</div><div class=\"sub\">warehouse zone</div></div><div class=\"agents\"><span class=\"chip\">route → HQ</span></div></div>
        <div class=\"tile hub\"><div class=\"sprite\"><span>HQ</span></div><div><div class=\"name\">HQ ROOM</div><div class=\"sub\">all agents report here</div></div><div class=\"agents\"><span class=\"chip\">CEO</span><span class=\"chip\">CTO</span><span class=\"chip\">CFO</span><span class=\"chip\">OPS</span><span class=\"chip\">CONTENT</span><span class=\"chip\">RESEARCH</span><span class=\"chip\">SALES</span><span class=\"chip\">SUPPORT</span></div></div>
        <div class=\"tile terra\"><div class=\"sprite\"><span>TT</span></div><div><div class=\"name\">TERACHE TIRES</div><div class=\"sub\">garage zone</div></div><div class=\"agents\"><span class=\"chip\">route → HQ</span></div></div>
        <div class=\"tile empty sw\"><div><div class=\"name\">EMPTY SLOT</div><div class=\"sub\">future company</div></div><div class=\"agents\"><span class=\"chip\">LOCKED</span></div></div>
        <div class=\"tile empty s\"><div><div class=\"name\">EMPTY SLOT</div><div class=\"sub\">future company</div></div><div class=\"agents\"><span class=\"chip\">LOCKED</span></div></div>
        <div class=\"tile empty se\"><div><div class=\"name\">EMPTY SLOT</div><div class=\"sub\">future company</div></div><div class=\"agents\"><span class=\"chip\">LOCKED</span></div></div>
      </div>
    </div>
  </div>
</div></body></html>"""

ROOMS_INDEX_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rooms — Mission Control</title>
<style>
body{background:#0b1020;color:#e5ebff;font-family:system-ui,sans-serif;min-height:100vh;margin:0;padding:28px}
.wrap{max-width:920px;margin:0 auto}
.card{background:#12182a;border:1px solid rgba(142,160,255,.18);border-radius:18px;padding:22px;margin-bottom:14px}
a{color:inherit;text-decoration:none}.room{display:block;padding:14px;border:1px solid rgba(142,160,255,.14);border-radius:14px;margin-top:10px;background:rgba(255,255,255,.03)}
.muted{color:#b8c4f5;line-height:1.6}
.back{display:inline-block;margin-top:8px;font-family:ui-monospace,monospace;color:#97a8ff}
</style></head><body><div class="wrap"><div class="card"><h1>Rooms</h1><p class="muted">Each room is its own top-down company pod. HQ is the shared center. The room list starts with three companies and leaves space for more later.</p><a class="back" href="/">← Back to Mission Control</a></div><a class="room" href="/rooms/h2waders"><strong>H2 Waders</strong><div class="muted">Waitlist-phase brand room</div></a><a class="room" href="/rooms/pro-fulfill"><strong>Pro Fulfill</strong><div class="muted">Operations room</div></a><a class="room" href="/rooms/terache-tires"><strong>Terache Tires</strong><div class="muted">Commerce room</div></a></div></body></html>"""


MISSION_CONTROL_OVERLAY_SCRIPT = """<script>
(function(){
  try {
    if (document.getElementById('mission-control-link')) return;
    var btn = document.createElement('a');
    btn.id = 'mission-control-link';
    btn.href = '/';
    btn.textContent = 'Mission Control';
    btn.style.cssText = 'position:fixed;right:18px;bottom:18px;z-index:2147483647;padding:12px 16px;border-radius:999px;background:#7c8cff;color:#07101f;font:600 12px/1 "IBM Plex Mono",monospace;text-decoration:none;box-shadow:0 12px 30px rgba(0,0,0,.35);border:1px solid rgba(255,255,255,.18)';
    document.body.appendChild(btn);
  } catch (e) {}
})();
</script>"""


def _inject_mission_control_link(html: str) -> str:
    if 'id="mission-control-link"' in html:
        return html
    if re.search(r'<body\b[^>]*>', html, flags=re.I):
        return re.sub(r'(<body\b[^>]*>)', r'\1' + MISSION_CONTROL_OVERLAY_SCRIPT, html, count=1, flags=re.I)
    return html + MISSION_CONTROL_OVERLAY_SCRIPT


def _html_escape(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace('"', "&quot;").replace("'", "&#39;"))


async def page_login(request: Request) -> Response:
    """GET /login — render the sign-in form."""
    if _is_authenticated(request):
        return RedirectResponse(_safe_return_to(request.query_params.get("returnTo", "/")), status_code=302)
    rt = _safe_return_to(request.query_params.get("returnTo", "/"))
    error_html = ('<div class="err">Invalid username or password</div>'
                  if request.query_params.get("error") else "")
    html = (LOGIN_PAGE_HTML
            .replace("__ERROR__", error_html)
            .replace("__RETURN_TO__", _html_escape(rt)))
    return HTMLResponse(html)


async def login_post(request: Request) -> Response:
    """POST /login — validate creds and set the auth cookie."""
    form = await request.form()
    username = str(form.get("username", ""))
    password = str(form.get("password", ""))
    return_to = _safe_return_to(str(form.get("returnTo", "/")))

    valid_user = _hmac.compare_digest(username, ADMIN_USERNAME)
    valid_pw = _hmac.compare_digest(password, ADMIN_PASSWORD)
    if valid_user and valid_pw:
        resp = RedirectResponse(return_to, status_code=302)
        resp.set_cookie(
            COOKIE_NAME,
            _make_auth_token(),
            max_age=COOKIE_MAX_AGE,
            httponly=True,
            samesite="lax",
            path="/",
        )
        return resp
    return RedirectResponse(f"/login?returnTo={_url_quote(return_to)}&error=1", status_code=302)


async def logout(request: Request) -> Response:
    """GET /logout — clear cookie and bounce to login."""
    resp = RedirectResponse("/login", status_code=302)
    resp.delete_cookie(COOKIE_NAME, path="/")
    return resp


# ── Gateway manager ───────────────────────────────────────────────────────────
class Gateway:
    def __init__(self):
        self.proc: asyncio.subprocess.Process | None = None
        self.state = "stopped"
        self.logs: deque[str] = deque(maxlen=500)
        self.started_at: float | None = None
        self.restarts = 0

    async def start(self):
        if self.proc and self.proc.returncode is None:
            return
        self.state = "starting"
        try:
            # .env values take priority over Railway env vars.
            # We build the env this way so hermes's own dotenv loading
            # (which reads the same file) doesn't shadow our values.
            env = {**os.environ, "HERMES_HOME": HERMES_HOME}
            env.update(read_env(ENV_FILE))
            model = env.get("LLM_MODEL", "")
            provider_key = next((env.get(k, "") for k in PROVIDER_KEYS if env.get(k)), "")
            print(f"[gateway] model={model or '⚠ NOT SET'} | provider_key={'set' if provider_key else '⚠ NOT SET'}", flush=True)
            write_config_yaml(read_env(ENV_FILE))
            self.proc = await asyncio.create_subprocess_exec(
                "hermes", "gateway",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
            )
            self.state = "running"
            self.started_at = time.time()
            asyncio.create_task(self._drain())
        except Exception as e:
            self.state = "error"
            self.logs.append(f"[error] Failed to start: {e}")

    async def stop(self):
        if not self.proc or self.proc.returncode is not None:
            self.state = "stopped"
            return
        self.state = "stopping"
        self.proc.terminate()
        try:
            await asyncio.wait_for(self.proc.wait(), timeout=10)
        except asyncio.TimeoutError:
            self.proc.kill()
            await self.proc.wait()
        self.state = "stopped"
        self.started_at = None

    async def _drain(self):
        assert self.proc and self.proc.stdout
        async for raw in self.proc.stdout:
            line = ANSI_ESCAPE.sub("", raw.decode(errors="replace").rstrip())
            self.logs.append(line)
        if self.state == "running":
            self.state = "error"
            self.logs.append(f"[error] Gateway exited (code {self.proc.returncode})")

    def status(self) -> dict:
        uptime = int(time.time() - self.started_at) if self.started_at and self.state == "running" else None
        return {
            "state":    self.state,
            "pid":      self.proc.pid if self.proc and self.proc.returncode is None else None,
            "uptime":   uptime,
            "restarts": self.restarts,
        }


gw = Gateway()


# ── Hermes dashboard subprocess ───────────────────────────────────────────────
class Dashboard:
    """Manages the `hermes dashboard` subprocess (native Hermes web UI).

    Bound to loopback only — we expose it to the public internet through our
    reverse proxy on $PORT, where edge cookie auth guards every request.
    The dashboard is independent of the gateway: it reads config files
    directly and tolerates a stopped gateway.

    All subprocess output is streamed to our stdout (→ Railway logs) with a
    `[dashboard]` prefix AND retained in a ring buffer for diagnostics.
    Unexpected exits are explicitly logged with their return code.
    """

    def __init__(self):
        self.proc: asyncio.subprocess.Process | None = None
        self.logs: deque[str] = deque(maxlen=300)
        self._drain_task: asyncio.Task | None = None

    async def start(self):
        if self.proc and self.proc.returncode is None:
            return
        try:
            self.proc = await asyncio.create_subprocess_exec(
                "hermes", "dashboard",
                "--host", HERMES_DASHBOARD_HOST,
                "--port", str(HERMES_DASHBOARD_PORT),
                "--no-open",
                # --tui exposes /api/pty + /api/ws + /api/events so the
                # dashboard's embedded Chat tab works end-to-end. Requires
                # hermes >= v2026.4.23 — older releases exit immediately
                # with "unrecognized arguments: --tui". The Dockerfile
                # pre-builds ui-tui/dist/ so PTY spawn is instant.
                "--tui",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            print(f"[dashboard] spawned pid={self.proc.pid} → {HERMES_DASHBOARD_URL}", flush=True)
            self._drain_task = asyncio.create_task(self._drain())
        except Exception as e:
            print(f"[dashboard] FAILED to spawn: {e!r}", flush=True)

    async def _drain(self):
        """Stream subprocess output to Railway logs (prefixed) and a ring buffer."""
        assert self.proc and self.proc.stdout
        try:
            async for raw in self.proc.stdout:
                line = ANSI_ESCAPE.sub("", raw.decode(errors="replace").rstrip())
                self.logs.append(line)
                print(f"[dashboard] {line}", flush=True)
        except Exception as e:
            print(f"[dashboard] drain error: {e!r}", flush=True)
        finally:
            rc = self.proc.returncode if self.proc else None
            if rc is not None and rc != 0:
                print(f"[dashboard] EXITED with code {rc} — reverse proxy will return 503 until restart", flush=True)
            elif rc == 0:
                print(f"[dashboard] exited cleanly (code 0)", flush=True)

    async def stop(self):
        if not self.proc or self.proc.returncode is not None:
            return
        self.proc.terminate()
        try:
            await asyncio.wait_for(self.proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            self.proc.kill()
            await self.proc.wait()


dash = Dashboard()

# Shared async HTTP client for the reverse proxy. Created lazily so we pick up
# the running event loop, torn down in lifespan.
_http_client: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=5.0),
            follow_redirects=False,
        )
    return _http_client


# ── Route handlers ────────────────────────────────────────────────────────────
async def route_health(request: Request):
    return JSONResponse({"status": "ok", "gateway": gw.state})


async def route_mission_control(request: Request) -> Response:
    """GET /: render the Mission Control landing page."""
    if err := guard(request):
        return err
    return HTMLResponse(MISSION_CONTROL_HTML)


async def route_rooms_index(request: Request) -> Response:
    """GET /rooms: render the room index."""
    if err := guard(request):
        return err
    return HTMLResponse(ROOMS_INDEX_HTML)


async def route_room(request: Request) -> Response:
    """GET /rooms/{room_key}: render a company room view."""
    if err := guard(request):
        return err
    room_key = request.path_params.get("room_key", "")
    return HTMLResponse(_room_html(room_key))


async def route_native(request: Request) -> Response:
    """GET /native and /native/*: proxy to the native Hermes dashboard."""
    if err := guard(request):
        return err
    native_path = request.url.path.removeprefix("/native") or "/"
    proxied = request.scope.copy()
    proxied["path"] = native_path
    proxied["raw_path"] = native_path.encode()
    request = Request(proxied, receive=request.receive)
    return await _proxy_to_dashboard(request)


# ── Reverse proxy → Hermes dashboard ──────────────────────────────────────────
DASHBOARD_UNAVAILABLE_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>Dashboard starting…</title>
<style>body{background:#0d0f14;color:#c9d1d9;font-family:ui-monospace,Menlo,monospace;
display:flex;align-items:center;justify-content:center;min-height:100vh;margin:0}
.card{max-width:480px;padding:32px;border:1px solid #252d3d;border-radius:12px;
background:#14181f;text-align:center}
h1{font-size:16px;color:#d29922;margin:0 0 12px;font-weight:600}
p{font-size:13px;color:#6b7688;line-height:1.6;margin:0 0 16px}</style></head>
<body><div class="card">
<h1>⚠ Hermes dashboard unavailable</h1>
<p>The native Hermes dashboard is not responding on port %d.<br>
It may still be starting up, or it may have crashed.</p>
<p>This page will refresh automatically.</p>
</div>
<script>setTimeout(()=>location.reload(),4000);</script>
</body></html>""" % HERMES_DASHBOARD_PORT


async def _proxy_to_dashboard(request: Request) -> Response:
    """Forward an authenticated request to the Hermes dashboard subprocess.

    Assumes edge auth (cookie middleware) has already validated the caller.
    """
    client = get_http_client()
    target = f"{HERMES_DASHBOARD_URL}{request.url.path}"
    if request.url.query:
        target = f"{target}?{request.url.query}"

    req_headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in HOP_BY_HOP
    }
    body = await request.body()

    try:
        upstream = await client.request(
            request.method,
            target,
            headers=req_headers,
            content=body,
        )
    except (httpx.ConnectError, httpx.ConnectTimeout):
        return HTMLResponse(DASHBOARD_UNAVAILABLE_HTML, status_code=503)
    except httpx.RequestError as e:
        print(f"[proxy] upstream error for {request.method} {request.url.path}: {e}", flush=True)
        return HTMLResponse(DASHBOARD_UNAVAILABLE_HTML, status_code=502)

    # Surface non-2xx responses from hermes into Railway logs so we can
    # diagnose 401/500s without needing browser DevTools access.
    if upstream.status_code >= 400:
        body_snip = upstream.content[:200].decode("utf-8", errors="replace")
        print(
            f"[proxy] {request.method} {request.url.path} -> {upstream.status_code} "
            f"body={body_snip!r}",
            flush=True,
        )

    # Strip hop-by-hop and length/encoding headers — Starlette recomputes them.
    resp_headers = {
        k: v for k, v in upstream.headers.items()
        if k.lower() not in HOP_BY_HOP
        and k.lower() not in ("content-encoding", "content-length")
    }

    content = upstream.content
    content_type = upstream.headers.get("content-type", "")
    if "text/html" in content_type.lower():
        try:
            html = content.decode("utf-8", errors="replace")
            html = _inject_mission_control_link(html)
            content = html.encode("utf-8")
        except Exception:
            pass

    return Response(
        content=content,
        status_code=upstream.status_code,
        headers=resp_headers,
    )


async def route_root(request: Request) -> Response:
    """GET /: proxy to the Hermes dashboard."""
    if err := guard(request): return err
    return await _proxy_to_dashboard(request)


async def route_proxy(request: Request) -> Response:
    """Catch-all: forward any unmatched path to the Hermes dashboard."""
    if err := guard(request): return err
    return await _proxy_to_dashboard(request)


# ── App lifecycle ─────────────────────────────────────────────────────────────
async def auto_start():
    if is_config_complete():
        asyncio.create_task(gw.start())
    else:
        print("[server] Config incomplete — gateway not started. Set LLM_MODEL and a provider key in Railway.", flush=True)


@asynccontextmanager
async def lifespan(app):
    # Dashboard runs always — it's the user-facing UI, independent of gateway state.
    asyncio.create_task(dash.start())
    await auto_start()
    try:
        yield
    finally:
        await asyncio.gather(
            gw.stop(),
            dash.stop(),
            return_exceptions=True,
        )
        global _http_client
        if _http_client is not None:
            await _http_client.aclose()
            _http_client = None


# ── WebSocket reverse proxy ──────────────────────────────────────────────────
# The hermes dashboard exposes 4 WebSocket endpoints when started with --tui.
# Three are opened by the browser SPA and need to flow through our reverse
# proxy; the fourth (/api/pub) is opened only by the PTY child against
# loopback and is intentionally NOT proxied — exposing it would let an
# authed user spam events into channels.
#
#   /api/pty     binary stream — embedded TUI keystrokes/output
#   /api/ws      JSON-RPC      — gateway sidecar driving Chat metadata
#   /api/events  text frames   — dashboard subscriber for /api/pub fan-out
#
# Auth model (matches the HTTP proxy):
#   * Edge: our HMAC cookie via _is_authenticated. WebSocket inherits .cookies
#     from starlette HTTPConnection so the same helper works unchanged.
#   * Upstream: hermes's own ?token=<_SESSION_TOKEN> query param. The SPA
#     fetches that token via /api/auth/session-token and includes it in the
#     WS URL, so we just forward path + query verbatim.
PROXIED_WS_PATHS = ("/api/pty", "/api/ws", "/api/events")


async def _ws_pump_client_to_upstream(
    client: WebSocket,
    upstream: websockets.WebSocketClientProtocol,
) -> None:
    """Forward client → upstream until the client side disconnects."""
    try:
        while True:
            msg = await client.receive()
            if msg.get("type") == "websocket.disconnect":
                return
            data = msg.get("bytes")
            if data is not None:
                await upstream.send(data)
                continue
            text = msg.get("text")
            if text is not None:
                await upstream.send(text)
    except (WebSocketDisconnect, websockets.exceptions.ConnectionClosed):
        return
    except Exception as e:
        print(f"[ws-proxy] client→upstream error on {client.url.path}: {e!r}", flush=True)
        return


async def _ws_pump_upstream_to_client(
    upstream: websockets.WebSocketClientProtocol,
    client: WebSocket,
) -> None:
    """Forward upstream → client until upstream closes."""
    try:
        async for msg in upstream:
            if isinstance(msg, bytes):
                await client.send_bytes(msg)
            else:
                await client.send_text(msg)
    except (websockets.exceptions.ConnectionClosed, WebSocketDisconnect):
        return
    except Exception as e:
        print(f"[ws-proxy] upstream→client error on {client.url.path}: {e!r}", flush=True)
        return


async def ws_proxy(websocket: WebSocket) -> None:
    """Reverse-proxy a single WebSocket from browser → hermes dashboard."""
    if not _is_authenticated(websocket):
        await websocket.close(code=4401)
        return

    path = websocket.url.path
    qs = websocket.url.query
    upstream_url = f"ws://{HERMES_DASHBOARD_HOST}:{HERMES_DASHBOARD_PORT}{path}"
    if qs:
        upstream_url = f"{upstream_url}?{qs}"

    try:
        upstream = await websockets.connect(
            upstream_url,
            open_timeout=5,
        )
    except (asyncio.TimeoutError, OSError, websockets.exceptions.WebSocketException) as e:
        print(f"[ws-proxy] upstream connect failed for {path}: {e!r}", flush=True)
        await websocket.close(code=1011)
        return

    await websocket.accept()

    pump_in = asyncio.create_task(_ws_pump_client_to_upstream(websocket, upstream))
    pump_out = asyncio.create_task(_ws_pump_upstream_to_client(upstream, websocket))

    try:
        done, pending = await asyncio.wait(
            (pump_in, pump_out),
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
    finally:
        try:
            await upstream.close()
        except Exception:
            pass
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.close()
            except Exception:
                pass


ANY_METHOD = ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]

routes = [
    # Public — no auth required.
    Route("/health",                            route_health),
    Route("/login",                             page_login,          methods=["GET"]),
    Route("/login",                             login_post,          methods=["POST"]),
    Route("/logout",                            logout),

    # Reverse-proxy hermes's dashboard WebSockets (Chat tab + sidecar).
    WebSocketRoute("/api/pty",                  ws_proxy),
    WebSocketRoute("/api/ws",                   ws_proxy),
    WebSocketRoute("/api/events",               ws_proxy),

    # Mission Control and room views.
    Route("/",                                  route_mission_control, methods=ANY_METHOD),
    Route("/rooms",                             route_rooms_index,   methods=ANY_METHOD),
    Route("/rooms/{room_key}",                  route_room,          methods=ANY_METHOD),
    Route("/native",                            route_native,        methods=ANY_METHOD),
    Route("/native/{path:path}",                route_native,        methods=ANY_METHOD),

    # Catch-all proxy to the native Hermes dashboard.
    Route("/{path:path}",                       route_proxy,         methods=ANY_METHOD),
]

# No middleware — auth is enforced per-handler via guard(). This keeps /health
# and /login truly unauthenticated without middleware gymnastics.
app = Starlette(routes=routes, lifespan=lifespan)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8080"))
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info", loop="asyncio")
    server = uvicorn.Server(config)

    def _shutdown():
        loop.create_task(gw.stop())
        loop.create_task(dash.stop())
        server.should_exit = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _shutdown)

    loop.run_until_complete(server.serve())
