import json

import pytest

from mgtranscriber.exporters import get_exporter, list_exporters
from mgtranscriber.models import Segment
from tests.conftest import LEGACY_SEGMENTS

SEGMENTS = [Segment.from_dict(s) for s in LEGACY_SEGMENTS]


def test_registry_has_builtin_formats():
    assert {"txt", "srt", "json"} <= {e.name for e in list_exporters()}
    with pytest.raises(KeyError):
        get_exporter("nope")


def test_txt_merges_speaker_turns_and_applies_names():
    out = get_exporter("txt").render(SEGMENTS, {"SPEAKER_01": "Bob"})
    assert out == "[00:00:00] SPEAKER_00:\nAhoj.\n\n[00:00:02] Bob:\nNazdar, jak se máš?\n\n"


def test_txt_without_speakers():
    plain = [Segment(s.start, s.end, s.text) for s in SEGMENTS[:1]]
    assert get_exporter("txt").render(plain, {}) == "[00:00:00]\nAhoj.\n\n"


def test_srt():
    out = get_exporter("srt").render(SEGMENTS, {"SPEAKER_00": "Alice"})
    assert out.startswith("1\n00:00:00,000 --> 00:00:01,500\n[Alice] Ahoj.\n\n2\n00:00:02,000 --> 00:00:03,000\n[SPEAKER_01] Nazdar,\n")


def test_json_roundtrip_keeps_raw_speakers():
    out = json.loads(get_exporter("json").render(SEGMENTS, {"SPEAKER_00": "Alice"}))
    assert out[0]["speaker"] == "SPEAKER_00"
    assert [Segment.from_dict(s) for s in out] == SEGMENTS
