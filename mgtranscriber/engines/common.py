"""Building blocks shared by engines: secrets, audio loading, word alignment and pyannote diarization.

Heavy imports (torch, whisperx) stay inside the functions so importing this module is cheap.
"""

from __future__ import annotations

import gc
import os
import re
from typing import TYPE_CHECKING

from dotenv import dotenv_values

from ..models import Segment, Word
from ..progress import Phase, Progress, ProgressCallback
from .base import EngineError

if TYPE_CHECKING:
    import numpy as np

    from ..config import Settings

DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
SAMPLE_RATE = 16000
SENTENCE_END = re.compile(r"[.!?…]['\"”»)]*$")
MAX_WORD_S = 1.5  # an estimated word end never reaches further than this past the word's start
CUT_WORD_S = 0.5  # the same word twice within this time at a VAD cut is one word
MIN_TURN_S = 1.0  # a shorter run of another speaker inside a segment is boundary noise, not a turn


def env_value(settings: "Settings", name: str) -> str | None:
    """A secret such as an API key: process environment first, then ``.env`` in the project root."""
    value = os.environ.get(name)
    if not value:
        env_file = settings.root / ".env"
        value = dotenv_values(env_file).get(name) if env_file.is_file() else None
    return value or None


def device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def free_gpu() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_audio(path) -> "np.ndarray":
    """Mono float32 at 16 kHz (decoded by ffmpeg)."""
    import whisperx

    return whisperx.load_audio(str(path))


def load_audio_pair(path, level: bool = True) -> tuple["np.ndarray", "np.ndarray"]:
    """``(audio, original)``: ``audio`` is for the VAD, recognition and alignment and, with ``level`` (setting
    ``level_audio``), has its quiet passages lifted so that the VAD does not skip them. ``original`` is for the
    diarization: leveling evens out the loudness of the speakers, which pyannote uses as a cue (AMI: speaker
    confusion rose from 2 to 5-9 % in two meetings). Same length, so word times fit both; the same array when
    nothing was lifted."""
    original = load_audio(path)
    if not level:
        return original, original
    from ..leveling import level as level_audio

    return level_audio(original), original


def check_diarization_access(settings: "Settings") -> None:
    token = settings.hf_token
    if not token:
        raise EngineError("HF_TOKEN is missing in .env (see .env.example). Disable diarization to run without it.")
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    try:
        hf_hub_download(DIARIZATION_MODEL, "config.yaml", token=token)
    except GatedRepoError:
        raise EngineError(f"No access to the diarization model. Accept its terms at https://huggingface.co/{DIARIZATION_MODEL}")
    except HfHubHTTPError as e:
        raise EngineError(f"HuggingFace rejected the token or is unreachable: {e}")


def align(segments: list[dict], audio: "np.ndarray", language: str, progress: ProgressCallback) -> list[dict]:
    """Word timestamps with wav2vec2 (WhisperX). Falls back to spreading words over each segment
    when there is no alignment model for the language. Segments are dicts with start, end, text."""
    import whisperx

    progress(Progress(Phase.ALIGN, 0, f"Aligning words (language: {language})..."))
    dev = device()
    try:
        model, metadata = whisperx.load_align_model(language_code=language, device=dev)
    except ValueError:
        progress(Progress(Phase.ALIGN, 100, f"No alignment model for '{language}', word times are estimated"))
        return [spread_words(s) for s in segments]
    result = whisperx.align(segments, model, metadata, audio, dev, return_char_alignments=False,
                            progress_callback=lambda pct: progress(Progress(Phase.ALIGN, pct)))
    del model
    free_gpu()
    return result["segments"]


def spread_words(segment: dict) -> dict:
    """Estimate word times by character count, for text that has only segment times."""
    words = segment["text"].split()
    start, end = segment["start"], segment["end"]
    total = sum(len(w) + 1 for w in words) or 1
    out, t = [], start
    for w in words:
        dt = (end - start) * (len(w) + 1) / total
        out.append({"word": w, "start": round(t, 3), "end": round(t + dt, 3)})
        t += dt
    return {**segment, "words": out}


def diarize(segments: list[dict], audio: "np.ndarray", settings: "Settings", speakers: int | None,
            progress: ProgressCallback, voices: dict | None = None) -> list[dict]:
    """Run pyannote on ``audio`` and give every word and segment a speaker (WhisperX assignment).
    ``voices``, if given, is filled with the raw speaker embeddings ``{label: [256 floats]}``."""
    import whisperx
    from whisperx.diarize import DiarizationPipeline

    progress(Progress(Phase.DIARIZE, 0, "Identifying speakers..."))
    diarizer = DiarizationPipeline(model_name=DIARIZATION_MODEL, token=settings.hf_token, device=device())
    diarization = diarizer(audio, num_speakers=speakers, return_embeddings=voices is not None,
                           progress_callback=lambda pct: progress(Progress(Phase.DIARIZE, pct)))
    if isinstance(diarization, tuple):
        diarization, embeddings = diarization
        voices.update(embeddings or {})
    result = whisperx.assign_word_speakers(diarization, {"segments": segments})
    del diarizer
    free_gpu()
    return result["segments"]


def set_word_ends(words: list[Word], last_end: float | None = None) -> list[Word]:
    """Give words without an end (or one running into the next word) an end: the next word's start,
    but at most ``MAX_WORD_S`` after their own start, so a word before a pause doesn't swallow the pause.
    The last word ends at ``last_end`` (same cap) or half a second after its start."""
    for this, nxt in zip(words, words[1:]):
        if this.end is None or this.end > nxt.start:
            this.end = round(max(this.start, min(nxt.start, this.start + MAX_WORD_S)), 3)
    if words and words[-1].end is None:
        last = words[-1]
        last.end = round(max(last.start, min(last_end, last.start + MAX_WORD_S)) if last_end is not None
                         else last.start + 0.5, 3)
    return words


def _key(word: str) -> str:
    return "".join(c for c in word.lower() if c.isalnum())


def drop_cut_duplicate(before: list[Word], after: list[Word]) -> list[Word]:
    """``after`` without its first word if it repeats the last word of ``before``. Two VAD pieces
    that cut through continuous speech overlap a little, so the word at the cut can come out twice,
    once whole and once cut off; the longer form is kept (in ``before``)."""
    if not before or not after or abs(after[0].start - before[-1].start) > CUT_WORD_S:
        return after
    a, b = _key(before[-1].word), _key(after[0].word)
    if not a or not b or not (a.startswith(b) or b.startswith(a)):
        return after
    if len(b) > len(a):
        before[-1].word = after[0].word
    before[-1].end = max(before[-1].end or 0, after[0].end or 0) or None
    return after[1:]


def drop_cut_duplicate_text(before: str, after: str) -> str:
    """The same for pieces with text only: drop the first word of ``after`` if it repeats the last of ``before``."""
    last, first = before.split()[-1:], after.split()[:1]
    if last and first and _key(last[0]) and _key(last[0]) == _key(first[0]):
        return after.split(None, 1)[1] if len(after.split()) > 1 else ""
    return after


def _run_seconds(words: list[dict]) -> float:
    times = [t for w in words for t in (w.get("start"), w.get("end")) if t is not None]
    return max(times) - min(times) if times else 0.0


def _coalesce(runs: list[list]) -> list[list]:
    out: list[list] = []
    for speaker, words in runs:
        if out and out[-1][0] == speaker:
            out[-1][1] = out[-1][1] + words
        else:
            out.append([speaker, words])
    return out


def split_by_speaker(segments: list[dict]) -> list[dict]:
    """Cut segments where the speaker of the words changes (after diarization). Segments are built
    before the speakers are known, so one can span a change; pyannote then gives the whole segment
    its majority speaker, and the other speaker's words would be attributed wrongly in the text.
    Word speakers flicker at turn boundaries, so a run shorter than ``MIN_TURN_S`` is not cut off:
    it joins the run before it (the one after it, at the start). Words without a speaker stay with
    the run before them."""
    out: list[dict] = []
    for seg in segments:
        words = seg.get("words") or []
        if len({w.get("speaker") for w in words} - {None}) < 2:
            out.append(seg)
            continue
        runs: list[list] = []  # [speaker, words]
        for word in words:
            if runs and word.get("speaker") in (None, runs[-1][0]):
                runs[-1][1].append(word)
            else:
                runs.append([word.get("speaker"), [word]])
        while len(runs) > 1:
            i = next((i for i, (_, r) in enumerate(runs) if _run_seconds(r) < MIN_TURN_S), None)
            if i is None:
                break
            host = i - 1 if i > 0 else 1
            lo = min(i, host)
            runs[lo:lo + 2] = [[runs[host][0], runs[lo][1] + runs[lo + 1][1]]]
            runs = _coalesce(runs)
        if len(runs) == 1:
            out.append({**seg, "speaker": runs[0][0]})
            continue
        joiner = " " if " " in seg.get("text", "").strip() else ""  # languages written without spaces
        for i, (speaker, run) in enumerate(runs):
            starts = [w["start"] for w in run if w.get("start") is not None]
            ends = [w["end"] for w in run if w.get("end") is not None]
            out.append({**seg, "start": starts[0] if starts else seg["start"] if i == 0 else out[-1]["end"],
                        "end": ends[-1] if ends else seg["end"],
                        "text": " " + joiner.join(w["word"] for w in run).strip(),
                        "speaker": speaker, "words": run})
    return out


# --- language check ------------------------------------------------------------------------
# Parakeet has no language input and can drift into a neighbouring language (Czech audio written
# partly in Slovak). These cheap checks catch the common cases and turn them into a warning.

SK_LETTERS, CS_LETTERS = set("äľĺŕô"), set("ěřů")
SK_WORDS = {"sa", "ako", "alebo", "ktorý", "ktorá", "ktoré", "ktorí", "som", "sme", "ste", "sú", "nie", "pre", "aj",
            "len", "keď", "ešte", "môže", "veľmi", "tiež", "tohto", "ich", "bol", "bola", "bolo", "boli", "teda", "iba"}
CS_WORDS = {"se", "jako", "nebo", "který", "která", "které", "kteří", "jsem", "jsme", "jste", "jsou", "ne", "pro",
            "jen", "když", "ještě", "může", "velmi", "taky", "tohoto", "jejich", "byl", "byla", "bylo", "byli", "tedy", "pouze"}
PL_LETTERS = set("łąęśźćńż")  # not used in Czech or Slovak
CYRILLIC_LANGUAGES = {"ru", "uk", "bg", "be", "sr", "mk", "kk"}


def _slavic_votes(text: str) -> tuple[int, int]:
    """(Czech, Slovak) votes: words with letters or forms that exist in only one of the two."""
    cs = sk = 0
    for word in (w for w in (_key(t) for t in text.split()) if w):
        if word in CS_WORDS and word not in SK_WORDS or CS_LETTERS & set(word):
            cs += 1
        elif word in SK_WORDS and word not in CS_WORDS or SK_LETTERS & set(word):
            sk += 1
    return cs, sk


def _cyrillic_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum("Ѐ" <= c <= "ӿ" for c in letters) / max(len(letters), 1)


def _drifted(piece: str, expected: str) -> str | None:
    """The language ``piece`` seems to be in instead of ``expected``, or None."""
    cyrillic = _cyrillic_share(piece)
    if expected not in CYRILLIC_LANGUAGES and cyrillic > 0.3:
        return "Cyrillic"
    if expected in CYRILLIC_LANGUAGES and cyrillic < 0.2 and sum(c.isalpha() for c in piece) >= 10:
        return "Latin script"
    if expected in ("cs", "sk"):
        if sum(1 for w in piece.split() if PL_LETTERS & set(w.lower())) >= 2:
            return "Polish"
        cs, sk = _slavic_votes(piece)
        other, mine = (sk, cs) if expected == "cs" else (cs, sk)
        if other >= 2 and other >= mine:
            return "Slovak" if expected == "cs" else "Czech"
    return None


def language_warning(pieces: list[str], expected: str | None) -> str | None:
    """A warning when the transcribed ``pieces`` (one per VAD segment) do not look like ``expected``,
    else None. Judged piece by piece, because a model drifts for a few sentences, not the whole
    recording: Czech/Slovak mix-ups, Polish letters in Czech or Slovak, the wrong script."""
    pieces = [p for p in pieces if p.strip()]
    if not expected or not pieces:
        return None
    found = [d for d in (_drifted(p, expected) for p in pieces) if d]
    if len(found) >= max(2, 0.05 * len(pieces)):
        kinds = ", ".join(sorted(set(found)))
        return (f"{len(found)} of {len(pieces)} pieces do not look like '{expected}' ({kinds}): "
                f"the model may have switched language there")
    return None


def split_sentences(words: list[Word], max_seconds: float = 20.0) -> list[Segment]:
    """Group timed words into sentence-like segments: split after . ! ?, when the speaker changes
    or when a segment gets too long. The segment's speaker is that of its words."""
    segments: list[Segment] = []
    current: list[Word] = []

    def flush() -> None:
        if current:
            text = " ".join(w.word for w in current)
            segments.append(Segment(current[0].start, current[-1].end, " " + text, current[0].speaker, list(current)))
            current.clear()

    for word in words:
        if current and word.speaker != current[-1].speaker:
            flush()
        current.append(word)
        too_long = current[-1].end - current[0].start >= max_seconds
        if SENTENCE_END.search(word.word) or too_long:
            flush()
    flush()
    return segments
