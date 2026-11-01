"""Word document (python-docx): a heading with title, source and date, then speaker turns with the
speaker in bold and the start time; ``?layout=table`` puts the turns in a three-column table
(time, speaker, text) instead. Labels follow the transcript language (Czech or English)."""

from __future__ import annotations

import io
from datetime import datetime

from ..models import UNKNOWN_SPEAKER, Segment, speaker_turns
from ..timefmt import clock
from . import ExportContext, Exporter, register, speaker_label

LABELS = {
    "en": {"source": "Source", "date": "Transcribed", "duration": "Duration", "time": "Time", "speaker": "Speaker",
           "text": "Text", "unknown": "Unknown speaker", "redacted": "Redacted version: marked passages are removed."},
    "cs": {"source": "Zdroj", "date": "Přepsáno", "duration": "Délka", "time": "Čas", "speaker": "Mluvčí",
           "text": "Text", "unknown": "Neznámý mluvčí", "redacted": "Redigovaná verze: označené pasáže jsou odstraněny."},
}


def _labels(language: str | None) -> dict:
    return LABELS.get((language or "en").split("-")[0].lower(), LABELS["en"])


def _date(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return value


def render(segments: list[Segment], names: dict[str, str], context: ExportContext) -> bytes:
    from docx import Document
    from docx.shared import Pt, RGBColor

    labels = _labels(context.language)
    doc = Document()
    doc.core_properties.title = context.title or "Transcript"
    doc.add_heading(context.title or "Transcript", level=1)
    facts = [(labels["source"], context.source), (labels["date"], _date(context.date)),
             (labels["duration"], clock(context.duration) if context.duration else None)]
    for label, value in facts:
        if value:
            p = doc.add_paragraph()
            p.add_run(f"{label}: ").bold = True
            p.add_run(str(value))
            p.paragraph_format.space_after = Pt(0)
    if context.redacted:
        doc.add_paragraph().add_run(labels["redacted"]).italic = True

    def who(speaker: str) -> str:
        return labels["unknown"] if speaker == UNKNOWN_SPEAKER and not names.get(speaker) else speaker_label(speaker, names)

    turns = speaker_turns(segments)
    if context.options.get("layout") == "table":
        table = doc.add_table(rows=1, cols=3)
        table.style = "Table Grid"
        for cell, text in zip(table.rows[0].cells, (labels["time"], labels["speaker"], labels["text"])):
            cell.paragraphs[0].add_run(text).bold = True
        for turn in turns:
            row = table.add_row().cells
            row[0].text = clock(turn["start"])
            row[1].paragraphs[0].add_run(who(turn["speaker"])).bold = True
            row[2].text = turn["text"]
    else:
        doc.add_paragraph()
        for turn in turns:
            head = doc.add_paragraph()
            time_run = head.add_run(f"[{clock(turn['start'])}] ")
            time_run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)
            head.add_run(who(turn["speaker"])).bold = True
            head.paragraph_format.space_after = Pt(0)
            head.paragraph_format.keep_with_next = True
            doc.add_paragraph(turn["text"])
    out = io.BytesIO()
    doc.save(out)
    return out.getvalue()


register(Exporter("docx", "DOCX", "docx",
                  "application/vnd.openxmlformats-officedocument.wordprocessingml.document", render,
                  takes_context=True))
