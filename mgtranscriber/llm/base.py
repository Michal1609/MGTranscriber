"""Provider interface and errors for the LLM layer.

A provider turns (system prompt, user prompt, JSON schema) into a parsed JSON object, and
optionally text into embedding vectors. Everything above it (chunking, map-reduce, time and
quote validation) is provider independent.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any


class LLMError(Exception):
    """A problem the user can act on. ``code`` is stable (translated by the UI), ``message`` is English."""

    code = "llm_error"
    status = 502

    def __init__(self, message: str, code: str | None = None):
        super().__init__(message)
        self.message = message
        if code:
            self.code = code


class EmptyTranscript(LLMError):
    """The transcript has no text (no speech was found): there is nothing to translate or analyse."""

    code = "empty_transcript"
    status = 409

    def __init__(self, message: str = "The transcript is empty (no speech was found)."):
        super().__init__(message)


def ensure_has_text(library, transcript_id: str) -> None:
    """Raise EmptyTranscript when the transcript has no words; used before any follow-up work is started."""
    if library.summary(transcript_id).get("no_speech"):
        raise EmptyTranscript()


class ProviderUnavailable(LLMError):
    code = "llm_unreachable"
    status = 503


class ModelMissing(LLMError):
    code = "llm_model_missing"
    status = 409


class MissingApiKey(LLMError):
    code = "llm_missing_key"
    status = 409


class AnalysisBusy(LLMError):
    code = "analysis_running"
    status = 409


class BadModelOutput(LLMError):
    code = "llm_bad_output"


class Cancelled(Exception):
    """Raised inside an analysis when the user cancelled it."""


class LLMProvider(ABC):
    """Common interface of ollama / gemini / anthropic."""

    name: str = ""
    model: str = ""
    #: tokens the model can take per request; the analysis keeps chunks below ~half of this
    max_context: int = 8192
    #: tokens of transcript per map chunk (a single chunk when the whole transcript fits)
    chunk_tokens: int = 4500
    #: tokens a model may need for one structured answer
    max_output: int = 3000
    local: bool = False

    def __init__(self) -> None:
        self.usage = {"calls": 0, "prompt_tokens": 0, "output_tokens": 0}

    def add_usage(self, prompt: int = 0, output: int = 0) -> None:
        self.usage["calls"] += 1
        self.usage["prompt_tokens"] += prompt or 0
        self.usage["output_tokens"] += output or 0

    @abstractmethod
    def check(self) -> None:
        """Fail fast with a helpful LLMError when the provider cannot be used."""

    @abstractmethod
    def generate_json(self, system: str, user: str, schema: dict, *, max_tokens: int = 2048,
                      temperature: float = 0.0) -> dict:
        """One structured completion; returns the parsed object."""

    def embed(self, texts: list[str]) -> list[list[float]] | None:
        """Embedding vectors, or None when the provider has none (callers then send the whole text)."""
        return None

    def with_model(self, model: str) -> "LLMProvider":
        """A sibling provider using another model (e.g. a stronger one for the final steps)."""
        raise NotImplementedError

    def info(self) -> dict:
        return {"provider": self.name, "model": self.model}


# --- helpers shared by providers ---------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Rough token count (English/Czech mix); only used for chunk sizing."""
    return int(len(text) / 3.6) + 1


def repair_json(text: str) -> str:
    """Close a JSON document that was cut off by the output limit."""
    text = text.rstrip().rstrip(",")
    stack: list[str] = []
    in_string = escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
    return text + ('"' if in_string else "") + "".join(reversed(stack))


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):  # some models fence their JSON despite the schema
        text = text.strip("`")
        text = text[text.find("{"):] if "{" in text else text
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            data = json.loads(repair_json(text))
        except json.JSONDecodeError as exc:
            raise BadModelOutput(f"The model did not return valid JSON ({exc.msg}).") from exc
    if not isinstance(data, dict):
        raise BadModelOutput("The model returned JSON that is not an object.")
    return data


def strict_schema(schema: Any) -> Any:
    """Copy of a JSON schema with ``additionalProperties: false`` on every object (needed by Anthropic)."""
    if isinstance(schema, dict):
        out = {k: strict_schema(v) for k, v in schema.items()}
        if out.get("type") == "object":
            out.setdefault("additionalProperties", False)
        return out
    if isinstance(schema, list):
        return [strict_schema(v) for v in schema]
    return schema
