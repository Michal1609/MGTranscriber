"""Ollama provider: local models over the REST API (no extra dependencies)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from .base import LLMError, LLMProvider, ModelMissing, ProviderUnavailable, parse_json_object

DEFAULT_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "gemma3:4b"
DEFAULT_EMBED_MODEL = "bge-m3"


def same_model(wanted: str, installed: str) -> bool:
    """``gemma3:4b`` matches ``gemma3:4b``; a name without tag matches ``name:latest``."""
    if wanted == installed:
        return True
    return ":" not in wanted and installed == wanted + ":latest"


class OllamaProvider(LLMProvider):
    name = "ollama"
    local = True

    def __init__(self, model: str | None = None, base_url: str | None = None, *, num_ctx: int = 8192,
                 embed_model: str | None = None, timeout: float = 1800.0):
        super().__init__()
        self.model = model or DEFAULT_MODEL
        self.base_url = (base_url or DEFAULT_URL).rstrip("/")
        self.embed_model = embed_model or DEFAULT_EMBED_MODEL
        self.num_ctx = num_ctx  # always sent: Ollama's own default (4096) silently truncates long prompts
        self.max_context = num_ctx
        self.chunk_tokens = max(1500, int(num_ctx * 0.55))
        self.timeout = timeout

    def with_model(self, model: str) -> "OllamaProvider":
        return OllamaProvider(model, self.base_url, num_ctx=self.num_ctx, embed_model=self.embed_model,
                              timeout=self.timeout)

    # --- HTTP --------------------------------------------------------------------------

    def _request(self, path: str, body: dict | None = None, timeout: float | None = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base_url + path, data, {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = json.load(exc).get("error", "")
            except (ValueError, OSError):
                pass
            if exc.code == 404 and "not found" in detail.lower():
                model = (body or {}).get("model", self.model)
                raise ModelMissing(
                    f"Ollama does not have the model '{model}'. Install it with: ollama pull {model}") from exc
            raise LLMError(f"Ollama error {exc.code}: {detail or exc.reason}") from exc
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
            raise ProviderUnavailable(
                f"Cannot reach Ollama at {self.base_url}. Start it (run 'ollama serve' or open the Ollama app) "
                f"or change the base URL.") from exc

    def installed_models(self) -> list[str]:
        return [m["name"] for m in self._request("/api/tags", timeout=5).get("models", [])]

    def check(self) -> None:
        installed = self.installed_models()
        if not any(same_model(self.model, m) for m in installed):
            raise ModelMissing(
                f"Ollama does not have the model '{self.model}'. Install it with: ollama pull {self.model}")

    # --- API ---------------------------------------------------------------------------

    def generate_json(self, system, user, schema, *, max_tokens=2048, temperature=0.0):
        body = {"model": self.model, "stream": False, "keep_alive": "10m", "format": schema,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "options": {"num_ctx": self.num_ctx, "temperature": temperature, "num_predict": max_tokens, "seed": 1}}
        reply = self._request("/api/chat", body)
        self.add_usage(reply.get("prompt_eval_count"), reply.get("eval_count"))
        return parse_json_object(reply.get("message", {}).get("content", ""))

    def embed(self, texts):
        out: list[list[float]] = []
        for i in range(0, len(texts), 16):
            reply = self._request("/api/embed", {"model": self.embed_model, "input": texts[i:i + 16]}, timeout=600)
            out += reply["embeddings"]
        return out

    def info(self):
        return {**super().info(), "embed_model": self.embed_model, "num_ctx": self.num_ctx}
