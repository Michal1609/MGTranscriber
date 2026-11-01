from ..models import Segment
from ..timefmt import vtt_time
from . import Exporter, register, speaker_label


def render(segments: list[Segment], names: dict[str, str]) -> str:
    out = ["WEBVTT\n\n"]
    for seg in segments:
        text = seg.text.strip().replace("-->", "->")
        if seg.speaker:  # WebVTT voice span: players can show or style the speaker
            label = speaker_label(seg.speaker, names).replace(">", "")
            text = f"<v {label}>{text}"
        out.append(f"{vtt_time(seg.start)} --> {vtt_time(seg.end)}\n{text}\n\n")
    return "".join(out)


register(Exporter("vtt", "VTT", "vtt", "text/vtt; charset=utf-8", render))
