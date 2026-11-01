"""Anthropic (Claude) provider (module named claude to avoid shadowing the SDK). The API key comes from the environment only."""

from __future__ import annotations

from .base import LLMError, LLMProvider, MissingApiKey, ProviderUnavailable, parse_json_object, strict_schema

DEFAULT_MODEL = "claude-haiku-5-5"
KEY_ENV = "ANTHROPIC_API_KEY"


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, model: str | None = None, api_key: str | None = None, *, client=None):
        super().__init__()
        self.model = model or DEFAULT_MODEL
        self._api_key = api_key
        self._client = client
        self.max_context = 200_000  # the models take 1M; this keeps one question affordable
        self.chunk_tokens = 150_000
        self.max_output = 16_000

    def with_model(self, model):
        return AnthropicProvider(model, self._api_key, client=self._client)

    @property
    def client(self):
        if self._client is None:
            if not self._api_key:
                raise MissingApiKey(f"Set the {KEY_ENV} environment variable (or .env) with your Anthropic API key.")
            try:
                import anthropic
            except ImportError as exc:
                raise LLMError("The anthropic package is not installed (uv add anthropic).",
                               "llm_sdk_missing") from exc
            self._client = anthropic.Anthropic(api_key=self._api_key, max_retries=3)
        return self._client

    def check(self) -> None:
        _ = self.client  # raises MissingApiKey / missing SDK

    def generate_json(self, system, user, schema, *, max_tokens=2048, temperature=0.0):
        import anthropic

        try:
            reply = self.client.messages.create(
                model=self.model, max_tokens=max(max_tokens, 4096), system=system,
                messages=[{"role": "user", "content": user}],
                # low effort: extraction does not need deep thinking, and it keeps the call fast and cheap
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": strict_schema(schema)}},
            )
        except anthropic.AuthenticationError as exc:
            raise MissingApiKey(f"Anthropic rejected the API key: {exc.message}", "llm_bad_key") from exc
        except anthropic.NotFoundError as exc:
            raise LLMError(f"Anthropic model '{self.model}' was not found: {exc.message}", "llm_model_missing") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError(f"Anthropic rate limit reached: {exc.message}", "llm_rate_limited") from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderUnavailable("Cannot reach the Anthropic API. Check the internet connection.") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Anthropic error {exc.status_code}: {exc.message}") from exc
        usage = getattr(reply, "usage", None)
        self.add_usage(getattr(usage, "input_tokens", 0), getattr(usage, "output_tokens", 0))
        if getattr(reply, "stop_reason", None) == "refusal":
            raise LLMError("The model declined to process this transcript.", "llm_refused")
        text = next((b.text for b in reply.content if getattr(b, "type", "") == "text"), "")
        return parse_json_object(text)
