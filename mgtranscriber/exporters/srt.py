from ..models import Segment
from ..timefmt import srt_time
from . import Exporter, register, speaker_label


def render(segments: list[Segment], names: dict[str, str]) -> str:
    out = []
    for i, seg in enumerate(segments, 1):
        text = seg.text.strip()
        if seg.speaker:
            text = f"[{speaker_label(seg.speaker, names)}] {text}"
        out.append(f"{i}\n{srt_time(seg.start)} --> {srt_time(seg.end)}\n{text}\n\n")
    return "".join(out)


register(Exporter("srt", "SRT", "srt", "application/x-subrip; charset=utf-8", render, stored_as="titulky.srt"))
