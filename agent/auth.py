"""Password login for the web UI: one shared password (BUIA_PASSWORD in .env) and a signed cookie. No user accounts.

The cookie is "<expiry>.<HMAC>", keyed by a hash of the password, so it needs no server-side storage, survives a restart,
and every session ends when the password is changed. Everything under /api except OPEN needs it (see require_login).
"""
import asyncio
import hashlib
import hmac
import os
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

PASSWORD = os.getenv("BUIA_PASSWORD", "")
COOKIE = "buia_session"
SESSION_SECONDS = 30 * 86400
OPEN = {"/api/login", "/api/health"}  # reachable without a session: the login itself and the installer's readiness check
KEY = hashlib.sha256(b"buia-session\0" + PASSWORD.encode()).digest()
MAX_FAILS, LOCK_SECONDS = 5, 60  # after 5 wrong passwords in a row, every attempt is refused for a minute
fails, locked_until = 0, 0.0


def sign(expiry: str) -> str:
    return hmac.new(KEY, expiry.encode(), "sha256").hexdigest()


def valid(token: str | None) -> bool:
    if not PASSWORD:  # no password set: KEY would be derivable by anyone, so no cookie counts
        return False
    expiry, _, mac = (token or "").partition(".")
    return expiry.isascii() and expiry.isdigit() and int(expiry) > time.time() and hmac.compare_digest(mac, sign(expiry))


async def require_login(request: Request) -> None:
    """App-wide dependency (agent.py): refuse every API call that has no valid session. Static files stay public: they hold no data."""
    if request.url.path not in OPEN and not valid(request.cookies.get(COOKIE)):
        raise HTTPException(401, "login required")


router = APIRouter(prefix="/api")


class Login(BaseModel):
    password: str


@router.post("/login")
async def login(body: Login, request: Request):
    global fails, locked_until
    if not PASSWORD:
        raise HTTPException(503, "BUIA_PASSWORD is not set in .env. Run .\\install.cmd (it creates one), then reload this page.")
    if time.time() < locked_until:
        raise HTTPException(429, "too many wrong passwords; wait a minute")
    if not hmac.compare_digest(body.password.encode(), PASSWORD.encode()):
        fails += 1
        if fails >= MAX_FAILS:
            fails, locked_until = 0, time.time() + LOCK_SECONDS
        await asyncio.sleep(1)  # slows guessing down
        raise HTTPException(401, "wrong password")
    fails = 0
    resp = JSONResponse({"ok": True})
    # SameSite=Strict also keeps other sites from making the browser send this cookie, so no separate CSRF token is needed.
    # Secure only over https (Tailscale Serve sets X-Forwarded-Proto): on http://127.0.0.1 some browsers would drop a Secure cookie.
    expiry = str(int(time.time()) + SESSION_SECONDS)
    resp.set_cookie(COOKIE, f"{expiry}.{sign(expiry)}", max_age=SESSION_SECONDS, httponly=True, samesite="strict",
                    secure=request.headers.get("x-forwarded-proto") == "https")
    return resp


@router.post("/logout")
async def logout():
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(COOKIE)
    return resp


@router.get("/session")
async def session():
    return {"ok": True}  # only reachable with a valid session (require_login): the page uses it to decide whether to show the login form
