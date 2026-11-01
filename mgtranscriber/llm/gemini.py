"""Gemini provider (google-genai SDK). The API key comes from the environment only."""

from __future__ import annotations

from .base import LLMError, LLMProvider, MissingApiKey, ProviderUnavailable, parse_json_object

DEFAULT_MODEL = "gemini-3.5-flash-lite"
EMBED_MODEL = "gemini-embedding-001"
KEY_ENV = "GEMINI_API_KEY"


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, model: str | None = None, api_key: str | None = None, *, client=None):
        super().__init__()
        self.model = model or DEFAULT_MODEL
        self._api_key = api_key
        self._client = client
        self.max_context = 200_000  # the models take more; this keeps one question affordable
        self.chunk_tokens = 150_000  # a typical transcript goes in one request
        self.max_output = 16_000

    def with_model(self, model):
        return GeminiProvider(model, self._api_key, client=self._client)

    @property
    def client(self):
        if self._client is None:
            if not self._api_key:
                raise MissingApiKey(f"Set the {KEY_ENV} environment variable (or .env) with your Gemini API key.")
            try:
                from google import genai
            except ImportError as exc:
                raise LLMError("The google-genai package is not installed (uv add google-genai).",
                               "llm_sdk_missing") from exc
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def _call(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except LLMError:
            raise
        except Exception as exc:  # SDK errors: map the ones the user can fix
            status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
            text = getattr(exc, "message", None) or str(exc)
            if status in (401, 403) or "API key" in text:
                raise MissingApiKey(f"Gemini rejected the API key: {text}", "llm_bad_key") from exc
            if status == 404:
                raise LLMError(f"Gemini model '{self.model}' was not found: {text}", "llm_model_missing") from exc
            if status == 429:
                raise LLMError(f"Gemini rate limit or quota reached: {text}", "llm_rate_limited") from exc
            if type(exc).__name__ in {"ConnectError", "ConnectTimeout", "ReadTimeout", "ConnectionError"}:
                raise ProviderUnavailable(f"Cannot reach the Gemini API: {text}") from exc
            raise LLMError(f"Gemini error: {text}") from exc

    def check(self) -> None:
        _ = self.client  # raises MissingApiKey / missing SDK

    def generate_json(self, system, user, schema, *, max_tokens=2048, temperature=0.0):
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system, response_mime_type="application/json", response_json_schema=schema,
            temperature=temperature, max_output_tokens=max(max_tokens, 4096))
        reply = self._call(self.client.models.generate_content, model=self.model, contents=user, config=config)
        usage = getattr(reply, "usage_metadata", None)
        self.add_usage(getattr(usage, "prompt_token_count", 0), getattr(usage, "candidates_token_count", 0))
        return parse_json_object(reply.text or "")

    def embed(self, texts):
        out: list[list[float]] = []
        for i in range(0, len(texts), 50):
            reply = self._call(self.client.models.embed_content, model=EMBED_MODEL, contents=texts[i:i + 50])
            out += [list(e.values) for e in reply.embeddings]
        return out
