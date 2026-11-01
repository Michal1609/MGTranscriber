"""Request body size limit (setting ``max_upload_mb``), so an upload cannot fill the disk.

Uploads are spooled to a temporary file before an endpoint runs (and /v1 copies them into
``data/uploads``), so the limit has to apply while the body arrives: ``Content-Length`` is
checked first, and bodies without it (chunked) are counted as they come in. Either way the
client gets 413 in the error format of the path. JSON bodies (the editor saves a whole transcript)
have their own, smaller limit ``JSON_MAX_BYTES``: they are parsed in memory.
"""

from __future__ import annotations

from .security import send_json

JSON_MAX_BYTES = 50 * 1024 * 1024


class BodyTooLarge(Exception):
    pass


class BodyLimitMiddleware:
    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    def message(self, limit: int) -> str:
        if limit != self.max_bytes:
            return f"The JSON request is larger than the limit of {limit // (1024 * 1024)} MB."
        return f"The request is larger than the limit of {limit // (1024 * 1024)} MB (setting max_upload_mb)."

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        limit = self.max_bytes
        if headers.get(b"content-type", b"").split(b";")[0].strip().lower() == b"application/json":
            limit = min(limit, JSON_MAX_BYTES) if limit > 0 else JSON_MAX_BYTES
        if limit <= 0:
            return await self.app(scope, receive, send)
        path = scope.get("path", "")
        length = headers.get(b"content-length", b"")
        if length.isdigit() and int(length) > limit:
            return await send_json(send, path, 413, "request_too_large", self.message(limit))

        received = 0
        too_large = answered = False

        async def counting_receive():
            nonlocal received, too_large
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    too_large = True
                    raise BodyTooLarge()  # stops the body parser; FastAPI may turn it into a 400 we replace
            return message

        async def guarded_send(message):
            nonlocal answered
            if not too_large:
                return await send(message)
            if message["type"] == "http.response.start" and not answered:  # replace the app's error response
                answered = True
                await send_json(send, path, 413, "request_too_large", self.message(limit))

        try:
            await self.app(scope, counting_receive, guarded_send)
        except BodyTooLarge:
            if not answered:
                await send_json(send, path, 413, "request_too_large", self.message(limit))
