import json

from ..models import Segment
from . import Exporter, register


def render(segments: list[Segment], names: dict[str, str]) -> str:
    # Same shape as prepis.json; speaker ids stay raw so the file can be loaded back.
    return json.dumps([s.to_dict() for s in segments], ensure_ascii=False, indent=2)


register(Exporter("json", "JSON", "json", "application/json", render, applies_names=False))
