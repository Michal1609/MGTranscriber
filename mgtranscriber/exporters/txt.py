from ..models import UNKNOWN_SPEAKER, Segment, speaker_turns
from ..timefmt import clock
from . import Exporter, register, speaker_label


def render(segments: list[Segment], names: dict[str, str]) -> str:
    out = []
    for turn in speaker_turns(segments):
        head = f"[{clock(turn['start'])}]"
        if turn["speaker"] != UNKNOWN_SPEAKER:
            head += f" {speaker_label(turn['speaker'], names)}:"
        out.append(f"{head}\n{turn['text']}\n\n")
    return "".join(out)


register(Exporter("txt", "TXT", "txt", "text/plain; charset=utf-8", render, stored_as="prepis.txt"))
