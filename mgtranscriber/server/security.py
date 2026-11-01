"""Request guard (pure ASGI middleware, so streaming responses are untouched).

1. Host header allow-list: stops DNS rebinding, where a foreign web page makes the
   browser talk to this server under the attacker's domain name.
2. Origin check for state-changing requests (POST/PUT/PATCH/DELETE): a browser page from
   another origin cannot queue jobs or rename speakers. Clients that send no Origin
   (curl, SDKs, MCP clients) are not affected.
3. Optional API key (setting ``api_key``) for /api, /v1 and /mcp: ``Authorization: Bearer
   <key>``, ``X-API-Key: <key>`` or the web UI's session cookie (set by
   ``POST /api/auth/login``; a random token checked against ``Sessions``). The static
   frontend and /api/auth/* stay public so the UI can show its login form. Wrong keys (headers
   or login form) are counted per client address and answered with 429 + Retry-After after a few
   misses, with a growing pause (``AuthThrottle``).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import secrets
import socket
import threading
import time
from http.cookies import CookieError, SimpleCookie
from pathlib import Path
from urllib.parse import urlsplit

from .. import APP_NAME
from ..config import Settings, is_loopback_host
from ..fsutil import replace_file
from .errors import error_body

log = logging.getLogger(__name__)

SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
WILDCARD_HOSTS = {"", "0.0.0.0", "::"}
SESSION_COOKIE = f"{APP_NAME}_session"
SESSION_TTL = 30 * 24 * 3600  # seconds a web UI login lasts
PROTECTED_PREFIXES = ("/api/", "/v1/", "/mcp")
PUBLIC_PREFIXES = ("/api/auth/",)
# Swagger UI (/api/docs) is a page of its own that loads its script and style from a CDN and runs inline code
DOCS_PATHS = ("/api/docs", "/api/docs/oauth2-redirect")
# The web UI: one same-origin module script, inline style attributes, blob/data media (recordings, icons).
CSP = ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; "
       "media-src 'self' blob:; font-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'self'; "
       "form-action 'self'; frame-ancestors 'none'")
SECURITY_HEADERS = [
    (b"x-frame-options", b"DENY"),
    (b"x-content-type-options", b"nosniff"),
    (b"referrer-policy", b"no-referrer"),
]


class Sessions:
    """Web UI logins: random tokens in the cookie, checked against a server-side list with expiry.

    The list (``data/sessions.json``) holds an HMAC of each token keyed with the API key, never the
    token itself, so a copy of the file is useless, and changing the key ends every session. Logout
    removes the session, so an old copy of the cookie stops working too.
    """

    def __init__(self, api_key: str | None, path: Path):
        self.api_key = api_key or ""
        self.path = Path(path)
        self.lock = threading.Lock()
        self.expiry: dict[str, float] = {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.expiry = {str(k): float(v) for k, v in data.get("sessions", {}).items()}
        except (OSError, ValueError, AttributeError, TypeError):
            pass

    def _tag(self, token: str) -> str:
        return hmac.new(self.api_key.encode(), token.encode(), hashlib.sha256).hexdigest()

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"sessions": self.expiry}), encoding="utf-8")
            replace_file(tmp, self.path)
        except OSError:
            log.exception("could not save %s", self.path)  # sessions still work until a restart

    def create(self) -> str:
        token = secrets.token_urlsafe(32)
        now = time.time()
        with self.lock:
            self.expiry = {t: e for t, e in self.expiry.items() if e > now}
            self.expiry[self._tag(token)] = now + SESSION_TTL
            self._save()
        return token

    def valid(self, token: str | None) -> bool:
        if not token or not self.api_key:
            return False
        expires = self.expiry.get(self._tag(token))
        return expires is not None and expires > time.time()

    def revoke(self, token: str | None) -> None:
        if not token or not self.api_key:
            return
        with self.lock:
            if self.expiry.pop(self._tag(token), None) is not None:
                self._save()


def needs_auth(path: str) -> bool:
    return (path.startswith(PROTECTED_PREFIXES) or path in ("/v1", "/api")) and not path.startswith(PUBLIC_PREFIXES)


def cookie_value(headers: dict[str, str], name: str) -> str | None:
    try:
        jar = SimpleCookie(headers.get("cookie", ""))
    except CookieError:
        return None
    return jar[name].value if name in jar else None


class AuthThrottle:
    """Failed key attempts per client address: from the third miss on the address is refused (429)
    for 2, 4, 8 ... up to 300 seconds. Nothing is held while waiting (no lock, no sleeping request),
    and one address cannot slow down the others. Memory is bounded: idle entries expire."""

    FREE_MISSES = 2
    MAX_PAUSE = 300.0
    FORGET_AFTER = 900.0
    MAX_ENTRIES = 10_000

    def __init__(self):
        self.entries: dict[str, list[float]] = {}  # address -> [misses, blocked until, last miss]

    def retry_after(self, ip: str) -> int:
        """Seconds until ``ip`` may try again; 0 when it may try now."""
        entry = self.entries.get(ip)
        return max(0, math.ceil(entry[1] - time.monotonic())) if entry else 0

    def failed(self, ip: str) -> None:
        now = time.monotonic()
        if len(self.entries) >= self.MAX_ENTRIES:
            self.entries = {k: v for k, v in self.entries.items() if now - v[2] < self.FORGET_AFTER}
            if len(self.entries) >= self.MAX_ENTRIES:
                self.entries.clear()
        entry = self.entries.get(ip)
        if entry is None or now - entry[2] > self.FORGET_AFTER:
            entry = self.entries[ip] = [0, 0.0, now]
        entry[0] += 1
        entry[2] = now
        if entry[0] > self.FREE_MISSES:
            entry[1] = now + min(self.MAX_PAUSE, 2.0 ** (entry[0] - self.FREE_MISSES))

    def succeeded(self, ip: str) -> None:
        self.entries.pop(ip, None)


def presented_keys(headers: dict[str, str]) -> list[str]:
    auth = headers.get("authorization", "")
    keys = [auth[7:].strip() if auth[:7].lower() == "bearer " else "", headers.get("x-api-key", "")]
    return [k for k in keys if k]


def check_key(headers: dict[str, str], api_key: str | None, sessions: Sessions | None = None,
              throttle: AuthThrottle | None = None, ip: str = "") -> tuple[bool, int]:
    """(authenticated, seconds to wait). A valid session cookie always passes; a key in a header is
    checked unless ``ip`` is paused, and a wrong one counts as a miss. A request without any key
    (the UI before it logs in) is just unauthenticated and costs nothing."""
    if not api_key:
        return True, 0
    if sessions is not None and sessions.valid(cookie_value(headers, SESSION_COOKIE)):
        return True, 0
    keys = presented_keys(headers)
    if not keys:
        return False, 0
    wait = throttle.retry_after(ip) if throttle else 0
    if wait:
        return False, wait
    if any(hmac.compare_digest(k.encode(), api_key.encode()) for k in keys):
        if throttle:
            throttle.succeeded(ip)
        return True, 0
    if throttle:
        throttle.failed(ip)
    return False, 0


def authenticated(headers: dict[str, str], api_key: str | None, sessions: Sessions | None = None,
                  throttle: AuthThrottle | None = None, ip: str = "") -> bool:
    """True when no key is configured or the request carries it (header or a valid session cookie)."""
    return check_key(headers, api_key, sessions, throttle, ip)[0]


def client_ip(scope) -> str:
    client = scope.get("client")
    return client[0] if client else ""


def split_host(value: str) -> str:
    """Host header -> host name without port, lower case ("[::1]:8000" -> "::1")."""
    value = value.strip().lower()
    if value.startswith("["):
        return value[1:value.find("]")] if "]" in value else value
    return value.rsplit(":", 1)[0] if value.count(":") == 1 else value


def allowed_hosts(settings: Settings) -> set[str]:
    """Host names the server answers to; contains "*" when any host is allowed."""
    configured = {split_host(h) if h.strip() != "*" else "*" for h in settings.allowed_hosts.split(",") if h.strip()}
    if configured:
        return configured
    hosts = {"localhost", "127.0.0.1", "::1"}
    if settings.host not in WILDCARD_HOSTS:
        hosts.add(split_host(settings.host))
    elif not settings.is_loopback:  # listening on every interface: this computer's names and addresses
        name = socket.gethostname()
        hosts |= {name.lower(), socket.getfqdn().lower()}
        try:
            hosts |= {info[4][0].lower() for info in socket.getaddrinfo(name, None)}
        except OSError:
            pass
    return hosts


def host_allowed(host: str, hosts: set[str]) -> bool:
    # loopback addresses cannot be pointed at this server by a foreign page, so they are always fine;
    # browsers resolve *.localhost to loopback themselves (not through DNS)
    return "*" in hosts or host in hosts or is_loopback_host(host) or host.lower().endswith(".localhost")


async def send_json(send, path: str, status: int, code: str, message: str,
                    retry_after: int = 0) -> None:
    body = json.dumps(error_body(path, status, code, message)).encode()
    headers = [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]
    if status == 401:
        headers.append((b"www-authenticate", b"Bearer"))
    if retry_after:
        headers.append((b"retry-after", str(retry_after).encode()))
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


class SecurityMiddleware:
    def __init__(self, app, settings: Settings, sessions: Sessions | None = None,
                 throttle: AuthThrottle | None = None):
        self.app = app
        self.throttle = throttle
        self.hosts = allowed_hosts(settings)
        self.api_key = settings.api_key or None
        self.sessions = sessions

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)
        headers = {k.decode("latin-1"): v.decode("latin-1") for k, v in scope["headers"]}
        path = scope.get("path", "")
        send = self.with_security_headers(send, path)
        host = split_host(headers.get("host", ""))
        if not host_allowed(host, self.hosts):
            return await send_json(send, path, 400, "invalid_host",
                                   f"Host '{host}' is not allowed (setting allowed_hosts).")
        method = scope.get("method", "GET")
        if method not in SAFE_METHODS and not self.same_origin(headers):
            return await send_json(send, path, 403, "cross_origin", "Cross-origin request rejected.")
        if method in ("GET", "HEAD") and (path.startswith(PROTECTED_PREFIXES) or path in ("/v1", "/api"))                 and (not self.same_origin(headers) or headers.get("sec-fetch-site") == "same-site"):
            # same-site = another port or sub-domain of this host (a page on localhost:3000)  # a foreign page's <img src="/api/..."> must not trigger work
            return await send_json(send, path, 403, "cross_origin", "Cross-origin request rejected.")
        if needs_auth(path):
            ok, wait = check_key(headers, self.api_key, self.sessions, self.throttle, client_ip(scope))
            if wait:
                return await send_json(send, path, 429, "too_many_attempts",
                                       f"Too many wrong keys. Try again in {wait} s.", wait)
            if not ok:
                return await send_json(send, path, 401, "auth_required", "Missing or invalid API key.")
        await self.app(scope, receive, send)

    @staticmethod
    def with_security_headers(send, path: str):
        """``send`` that adds anti-framing, nosniff, referrer and CSP headers to every response."""
        extra = list(SECURITY_HEADERS)
        extra.append((b"content-security-policy", b"frame-ancestors 'none'" if path in DOCS_PATHS else CSP.encode()))

        async def wrapped(message):
            if message["type"] == "http.response.start":
                present = {k.lower() for k, _ in message.get("headers", [])}
                message = {**message, "headers": [*message.get("headers", []),
                                                  *(h for h in extra if h[0] not in present)]}
            await send(message)
        return wrapped

    @staticmethod
    def same_origin(headers: dict[str, str]) -> bool:
        if headers.get("sec-fetch-site") == "cross-site":
            return False
        origin = headers.get("origin")
        if origin is None:
            return True  # not a browser page (or a same-origin GET-like navigation)
        parts = urlsplit(origin)
        return origin != "null" and bool(parts.netloc) and parts.netloc.lower() == headers.get("host", "").lower()
