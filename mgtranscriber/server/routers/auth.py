"""Web UI login for servers with an API key: exchanges the key for an HttpOnly session cookie."""

import hmac

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel, Field

from ..deps import SettingsDep
from ..errors import ApiError
from ..security import SESSION_COOKIE, SESSION_TTL, Sessions, check_key, cookie_value

router = APIRouter(prefix="/api/auth", tags=["auth"])


class Login(BaseModel):
    key: str = Field(min_length=1, max_length=500)


def sessions(request: Request) -> Sessions:
    return request.app.state.sessions


def headers_of(request: Request) -> dict[str, str]:
    return {k.lower(): v for k, v in request.headers.items()}


def client_of(request: Request) -> str:
    return request.client.host if request.client else ""


@router.get("/status")
def status(request: Request, settings: SettingsDep):
    """Whether the server requires a key and whether this request is authenticated."""
    return {"required": bool(settings.api_key),
            "authenticated": check_key(headers_of(request), settings.api_key, sessions(request),
                                       request.app.state.auth_throttle, client_of(request))[0]}


@router.post("/login")
async def login(body: Login, request: Request, response: Response, settings: SettingsDep):
    if not settings.api_key:
        return {"required": False, "authenticated": True}
    # wrong keys are counted per client address; once paused the address gets 429 until the pause ends
    throttle, ip = request.app.state.auth_throttle, client_of(request)
    wait = throttle.retry_after(ip)
    if wait:
        raise ApiError(429, "too_many_attempts", f"Too many wrong keys. Try again in {wait} s.")
    if not hmac.compare_digest(body.key.strip().encode(), settings.api_key.encode()):
        throttle.failed(ip)
        raise ApiError(401, "invalid_key", "Invalid API key.")
    throttle.succeeded(ip)
    # Secure follows the scheme uvicorn reports; behind a TLS proxy that needs X-Forwarded-Proto (see docs/API.md)
    response.set_cookie(SESSION_COOKIE, sessions(request).create(), max_age=SESSION_TTL, httponly=True,
                        samesite="strict", secure=request.url.scheme == "https", path="/")
    return {"required": True, "authenticated": True}


@router.post("/logout")
def logout(request: Request, response: Response, settings: SettingsDep):
    sessions(request).revoke(cookie_value(headers_of(request), SESSION_COOKIE))
    response.delete_cookie(SESSION_COOKIE, path="/")
    return {"required": bool(settings.api_key), "authenticated": not settings.api_key}
