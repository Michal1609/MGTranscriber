"""A fake engine for tests. The input file name controls its behavior: '*fail*' raises, '*slow*' takes ~10 s,
'*silent*' sleeps 60 s without any output (like a long model load)."""

import time

from mgtranscriber.engines import Engine
from mgtranscriber.models import Segment, TranscriptionResult, Word
from mgtranscriber.progress import Phase, Progress

SEGMENTS = [
    Segment(0.5, 2.0, " Hello there.", "SPEAKER_00", [Word("Hello", 0.5, 1.0, 0.9, "SPEAKER_00"), Word("there.", 1.1, 2.0, 0.9, "SPEAKER_00")]),
    Segment(2.5, 4.0, " Hi, how are you?", "SPEAKER_01", [Word("Hi,", 2.5, 2.8, 0.9, "SPEAKER_01")]),
    Segment(4.2, 5.0, " Fine.", "SPEAKER_01", []),
]


class MockEngine(Engine):
    name = "mock"
    supports_diarization = True

    def transcribe(self, media, options, progress):
        name = media.audio_path.name
        if "fail" in name:
            raise RuntimeError("mock failure")
        if "silent" in name:
            time.sleep(60)
        steps = 100 if "slow" in name else 3
        progress(Progress(Phase.TRANSCRIBE, 0, "Mock transcribing..."))
        for i in range(steps):
            if "slow" in name:
                time.sleep(0.1)
            progress(Progress(Phase.TRANSCRIBE, 100 * (i + 1) / steps))
        if "nospeech" in name:  # music only: the engine finds nothing
            return TranscriptionResult([], language=options.language or "en", duration=5.0, model="mock-1")
        segments = SEGMENTS if options.diarize else [Segment(s.start, s.end, s.text) for s in SEGMENTS]
        return TranscriptionResult(segments, language=options.language or "en", duration=5.0, model="mock-1")


def mock_task(request, settings, progress, library):
    """A registered job task (see mgtranscriber.tasks): marks an existing transcript."""
    progress(Progress("mock-task", 50, f"Mock task on {request.source}"))
    meta = library.metadata(request.source)
    meta.extra["mock_task"] = True
    library.save_metadata(request.source, meta)
    return request.source
