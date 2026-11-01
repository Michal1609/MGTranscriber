class ApiError(Exception):
    """An error with a stable ``code`` the frontend can translate; ``message`` is the English fallback.

    Under ``/v1`` (OpenAI-compatible API) it is rendered in OpenAI's error format, using
    ``param`` and ``kind`` (the OpenAI error ``type``).
    """

    def __init__(self, status: int, code: str, message: str, param: str | None = None,
                 kind: str = "invalid_request_error"):
        super().__init__(message)
        self.status, self.code, self.message, self.param, self.kind = status, code, message, param, kind


def is_openai_path(path: str) -> bool:
    return path == "/v1" or path.startswith("/v1/")


def error_body(path: str, status: int, code: str, message: str, param: str | None = None,
               kind: str | None = None) -> dict:
    """``{"detail", "code"}`` for our API, ``{"error": {...}}`` under /v1."""
    if is_openai_path(path):
        kind = kind or ("authentication_error" if status == 401 else
                        "server_error" if status >= 500 else "invalid_request_error")
        return {"error": {"message": message, "type": kind, "param": param, "code": code}}
    return {"detail": message, "code": code}
