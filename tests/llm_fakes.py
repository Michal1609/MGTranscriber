"""A scripted LLM provider and a synthetic one-hour transcript for the LLM tests."""

import re
import zlib

from mgtranscriber.llm.base import LLMProvider
from mgtranscriber.models import Segment

TOPICS = ["gardening", "football", "cooking", "astronomy", "finance", "music"]
STEP = 75.0  # seconds between turns; a topic lasts 8 turns = 10 minutes


def make_segments(turns: int = 48) -> list[Segment]:
    segs = []
    for i in range(turns):
        text = f" Sentence number {i} talks about {TOPICS[(i // 8) % len(TOPICS)]} in detail and more."
        segs.append(Segment(i * STEP, i * STEP + 60, text, f"SPEAKER_0{i % 2}"))
    return segs


def vec(text: str, dim: int = 512) -> list[float]:
    out = [0.0] * dim
    for word in re.findall(r"\w+", text.lower()):
        out[zlib.crc32(word.encode()) % dim] += 1.0
    return out


class FakeProvider(LLMProvider):
    """Answers each kind of request (found by the schema's property names) from ``script`` callables."""

    name = "fake"
    model = "fake-1"

    def __init__(self, *, chunk_tokens: int = 300, max_context: int = 1500, embeddings: bool = True, **script):
        super().__init__()
        self.chunk_tokens, self.max_context = chunk_tokens, max_context
        self.embeddings = embeddings
        self.script = script
        self.calls: list[tuple[str, str, str]] = []  # (kind, system, user)

    def check(self):
        pass

    def kind(self, schema: dict) -> str:
        props = set(schema.get("properties", {}))
        if "topics" in props:
            return "map"
        if "chapters" in props:
            return "chapters"
        if "found" in props:
            return "answer"
        if "items" in props:
            return "translate"
        return "reduce"

    def generate_json(self, system, user, schema, *, max_tokens=2048, temperature=0.0):
        kind = self.kind(schema)
        self.calls.append((kind, system, user))
        self.add_usage(10, 5)
        return self.script[kind](user)

    def embed(self, texts):
        return [vec(t) for t in texts] if self.embeddings else None

    def count(self, kind: str) -> int:
        return sum(1 for c in self.calls if c[0] == kind)
