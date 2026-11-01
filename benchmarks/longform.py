"""Long-form Czech speech with speakers: ParCzech4Speech test split put together into sessions.

ParCzech4Speech (Stankov, Kopp, Bojar; LINDAT 11234/1-5946, CC-BY) is the Czech Chamber of Deputies:
single-speaker clips of official transcripts aligned to the recordings. There are no absolute times
and no continuous recordings, so a session here is some clips of one sitting
(token id "ps2013-054") in the order of the sitting with 0.4 s of silence between them. Only part of
the clips of a sitting is downloaded, so neighbours are not adjacent in the original and the speeches
come from several agenda items. Speaker turns are known exactly, but there is no overlap, no
back-channel and no real pauses.

Only a few ranges of the two test shards are downloaded (the tar is read from a byte offset on
a 512-byte block boundary), not the 3 GB.

    uv run python -m benchmarks.longform download     # -> benchmarks/data/parczech/
    uv run python -m benchmarks.longform ami          # -> benchmarks/data/ami/

AMI (English meetings, CC-BY-4.0): four meetings of the test set (Mix-Headset = the headset microphones
summed, 16 kHz), reference words and speakers from the AMI manual annotations 1.6.2 (words.xml).
The reference text of a 60 s window is the words that start in it; overlapping speech is in the text
of both speakers in time order.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "parczech"
BASE_URL = "https://huggingface.co/datasets/ufal/parczech4speech-unsegmented/resolve/v1.0-full/data/"
SHARDS = {"test-0000.tar": 1_458_677_760, "test-0001.tar": 1_522_688_000}
CHUNKS_PER_SHARD = 8
CHUNK_BYTES = 36 * 1024 * 1024
SAMPLE_RATE = 16000
CLIP_GAP_S = 0.4
SESSION_TARGET_S = 600.0
SESSION_MAX_S = 840.0
SESSION_MIN_S = 300.0
TOTAL_TARGET_S = 80 * 60.0
BLOCK = 512


# --- tar ranges -------------------------------------------------------------------------

def read_members(blob: bytes) -> list[tuple[str, bytes]]:
    """Members of a tar that starts at a header but may end in the middle of a file.
    Members cut off at the end are dropped."""
    members, pos = [], 0
    while pos + BLOCK <= len(blob):
        header = blob[pos:pos + BLOCK]
        if not header.strip(b"\0"):
            break
        name = header[:100].split(b"\0")[0].decode("utf-8", "replace")
        try:
            size = int(header[124:136].split(b"\0")[0].strip() or b"0", 8)
        except ValueError:
            break
        start = pos + BLOCK
        if start + size > len(blob):
            break
        members.append((name, blob[start:start + size]))
        pos = start + (size + BLOCK - 1) // BLOCK * BLOCK
    return members


def first_header(blob: bytes) -> int:
    """Offset of the first 512-byte block that is a tar header of a .json member (chunks start anywhere)."""
    for pos in range(0, len(blob) - BLOCK, BLOCK):
        if blob[pos + 257:pos + 262] == b"ustar" and blob[pos:pos + 100].split(b"\0")[0].endswith(b".json"):
            try:
                int(blob[pos + 124:pos + 136].split(b"\0")[0].strip(), 8)
            except ValueError:
                continue
            return pos
    raise ValueError("no tar header in the chunk")


def clips_from_blob(blob: bytes) -> list[dict]:
    """Clips (metadata + FLAC bytes) of a chunk, in tar order."""
    members = read_members(blob[first_header(blob):])
    audio = {n[:-5]: d for n, d in members if n.endswith(".flac")}
    clips = []
    for name, data in members:
        if name.endswith(".json") and name[:-5] in audio:
            meta = json.loads(data)
            clips.append({"vert": meta["vert"], "speaker": meta["speaker"], "text": meta["true_text"],
                          "dur": float(meta["dur"]), "start_token": meta["start_token_id"], "flac": audio[name[:-5]]})
    return clips


def fetch_range(url: str, start: int, length: int) -> bytes:
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{start + length - 1}"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


# --- sessions ---------------------------------------------------------------------------

def token_key(token: str) -> tuple:
    """Order of ParCzech token ids ("ps2013-004-01-002-020.u8.p1.s3.w1") = order in the sitting."""
    return tuple(int(x) for x in re.findall(r"\d+", token))


def sitting(clip: dict) -> str:
    return "-".join(clip["start_token"].split("-")[:2])


def make_sessions(clips: list[dict]) -> list[list[dict]]:
    """One session per sitting: its clips in the order of the sitting, up to ``SESSION_MAX_S``.
    (A ``vert`` file is one speech of one speaker, the sitting is the first two parts of the token id,
    "ps2013-054".) The tar files are not in sitting order, so the clips of one sitting come from all
    over the chunks and are not adjacent: a session is a mosaic of speeches of several agenda items.
    A sitting with less than ``SESSION_MIN_S`` of speech or one speaker is dropped."""
    by_sitting: dict[str, list[dict]] = {}
    for clip in clips:
        by_sitting.setdefault(sitting(clip), []).append(clip)
    sessions = []
    for key in sorted(by_sitting):
        total, taken = 0.0, []
        for clip in sorted(by_sitting[key], key=lambda c: token_key(c["start_token"])):
            if total + clip["dur"] > SESSION_MAX_S or total >= SESSION_TARGET_S:
                break
            taken.append(clip)
            total += clip["dur"] + CLIP_GAP_S
        if total >= SESSION_MIN_S and len({c["speaker"] for c in taken}) >= 2:
            sessions.append(taken)
    return sessions


def choose(sessions: list[list[dict]], target_s: float = TOTAL_TARGET_S) -> list[list[dict]]:
    """The sessions with the most speakers first (then the longest), until ``target_s`` of audio is reached."""
    ranked = sorted(sessions, key=lambda s: (-len({c["speaker"] for c in s}), -sum(c["dur"] for c in s)))
    chosen, total = [], 0.0
    for session in ranked:
        chosen.append(session)
        total += sum(c["dur"] + CLIP_GAP_S for c in session)
        if total >= target_s:
            break
    return sorted(chosen, key=lambda s: token_key(s[0]["start_token"]))


def write_session(session: list[dict], index: int, out: Path, decode) -> dict:
    """Write the session as a 16-bit WAV and return its description (clip spans, speakers, reference).
    ``decode(flac_bytes) -> float32 mono 16 kHz`` is passed in so tests need no ffmpeg."""
    gap = np.zeros(int(CLIP_GAP_S * SAMPLE_RATE), dtype=np.float32)
    parts, clips, t = [], [], 0.0
    for i, clip in enumerate(session):
        audio = decode(clip["flac"])
        duration = len(audio) / SAMPLE_RATE
        parts += [audio, gap]
        clips.append({"id": f"s{index:02d}c{i:03d}", "speaker": clip["speaker"], "start": round(t, 3),
                      "end": round(t + duration, 3), "text": clip["text"], "token": clip["start_token"]})
        t += duration + CLIP_GAP_S
    name = f"session{index:02d}.wav"
    data = np.concatenate(parts)
    with wave.open(str(out / name), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes((np.clip(data, -1, 1) * 32767).astype("<i2").tobytes())
    return {"id": f"session{index:02d}", "audio": name, "sitting": sitting(session[0]), "seconds": round(t, 1),
            "speakers": sorted({c["speaker"] for c in session}), "clips": clips}


def download(out: Path = DATA) -> list[dict]:
    from mgtranscriber.engines.common import load_audio

    out.mkdir(parents=True, exist_ok=True)
    index = out / "sessions.json"
    if index.is_file():
        return json.loads(index.read_text(encoding="utf-8"))
    clips: list[dict] = []
    for shard, size in SHARDS.items():
        for k in range(CHUNKS_PER_SHARD):
            # spread over the shard; offsets are multiples of 512 so a header can be found
            offset = (size * (2 * k + 1) // (2 * CHUNKS_PER_SHARD)) // BLOCK * BLOCK
            print(f"{shard} @ {offset // 2**20} MB", file=sys.stderr)
            raw = out / "raw" / f"{shard}.{offset}"  # kept, so a failed build does not download again
            if not raw.is_file():
                raw.parent.mkdir(exist_ok=True)
                raw.write_bytes(fetch_range(BASE_URL + shard, offset, CHUNK_BYTES))
            clips += clips_from_blob(raw.read_bytes())
    sessions = make_sessions(clips)
    tmp = out / "tmp.flac"

    def decode(flac: bytes) -> np.ndarray:
        tmp.write_bytes(flac)
        return load_audio(tmp)

    described = [write_session(s, i, out, decode) for i, s in enumerate(choose(sessions))]
    tmp.unlink(missing_ok=True)
    index.write_text(json.dumps(described, ensure_ascii=False, indent=1), encoding="utf-8")
    return described


# --- diarization ------------------------------------------------------------------------

PAUSE_S = 1.0  # same-speaker spans closer than this are one turn (in the reference and the hypothesis)


def reference_annotation(turns: list[dict]):
    """Reference turns ({"start", "end", "speaker"}, may overlap) as an Annotation; same-speaker turns
    closer than ``PAUSE_S`` are merged."""
    from pyannote.core import Annotation, Segment

    ref = Annotation()
    for i, turn in enumerate(turns):
        ref[Segment(turn["start"], turn["end"]), i] = turn["speaker"]
    return ref.support(PAUSE_S)


def hypothesis_annotation(segments) -> "Annotation":  # noqa: F821  (lazy pyannote import)
    """Speaker turns of an engine result from its words (the segment's speaker where a word has none)."""
    from pyannote.core import Annotation, Segment

    hyp = Annotation()
    for seg in segments:
        words = seg.words or []
        spans = [(w.start, w.end, w.speaker or seg.speaker) for w in words if w.start is not None and w.end is not None]
        for start, end, speaker in spans or [(seg.start, seg.end, seg.speaker)]:
            if speaker is not None and end > start:
                hyp[Segment(start, end), f"{len(hyp)}"] = speaker
    return hyp.support(PAUSE_S)


def diarization_error(ref, hyp, collar: float = 0.25) -> dict:
    """DER with ``collar`` seconds around reference turn boundaries (0.25 = 0.25 s each side)."""
    from pyannote.metrics.diarization import DiarizationErrorRate

    from pyannote.core import Timeline

    metric = DiarizationErrorRate(collar=collar, skip_overlap=False)
    extent = ref.get_timeline().extent() | hyp.get_timeline().extent()  # the whole session is scored
    detail = metric(ref, hyp, uem=Timeline([extent]), detailed=True)
    no_overlap = DiarizationErrorRate(collar=collar, skip_overlap=True)(ref, hyp, uem=Timeline([extent]))
    total = detail["total"] or 1.0
    return {"der": round(100 * detail["diarization error rate"], 2), "der_no_overlap": round(100 * no_overlap, 2),
            "miss": round(100 * detail["missed detection"] / total, 2),
            "false_alarm": round(100 * detail["false alarm"] / total, 2),
            "confusion": round(100 * detail["confusion"] / total, 2),
            "ref_speech_s": round(total, 1),
            "speakers_ref": len(ref.labels()), "speakers_hyp": len(hyp.labels())}


def pooled_der(parts: list[dict]) -> float:
    """DER over all sessions: summed error time / summed reference speech (not a mean of rates)."""
    total = sum(p["ref_speech_s"] for p in parts) or 1.0
    error = sum((p["miss"] + p["false_alarm"] + p["confusion"]) * p["ref_speech_s"] / 100 for p in parts)
    return round(100 * error / total, 2)


# --- AMI --------------------------------------------------------------------------------

AMI_DATA = HERE / "data" / "ami"
AMI_AUDIO_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{0}/audio/{0}.Mix-Headset.wav"
AMI_ANNOTATIONS_URL = "http://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
AMI_MEETINGS = ["ES2004a", "IS1009a", "TS3003a", "EN2002b"]  # scenario + non-scenario meetings of the test set
AMI_WINDOW_S = 60.0


def ami_words(xml: bytes, speaker: str) -> list[dict]:
    """Spoken words of one AMI words.xml: no punctuation, no truncated words ("tha-"), no word without time."""
    import xml.etree.ElementTree as ET

    out = []
    for el in ET.fromstring(xml).iter("w"):
        if el.get("punc") == "true" or el.get("trunc") == "true" or not (el.text or "").strip():
            continue
        try:
            start, end = float(el.get("starttime")), float(el.get("endtime"))
        except (TypeError, ValueError):
            continue
        out.append({"word": el.text.strip(), "start": start, "end": end, "speaker": speaker})
    return out


def ami_session(meeting: str, words: list[dict], seconds: float, window_s: float = AMI_WINDOW_S) -> dict:
    """Reference of one meeting: ``window_s`` windows (the words that start in them, in time order) for the
    word error counts, and one turn per word (merged later) for the diarization."""
    words = sorted(words, key=lambda w: (w["start"], w["end"]))
    n = max(1, int(np.ceil(seconds / window_s)))
    texts: list[list[str]] = [[] for _ in range(n)]
    for w in words:
        texts[min(int(w["start"] // window_s), n - 1)].append(w["word"])
    clips = [{"id": f"{meeting}-w{i:02d}", "speaker": None, "start": round(i * window_s, 3),
              "end": round(min((i + 1) * window_s, seconds), 3), "text": " ".join(t)} for i, t in enumerate(texts)]
    return {"id": meeting, "audio": f"{meeting}.wav", "seconds": round(seconds, 1), "gap": 0.0,
            "speakers": sorted({w["speaker"] for w in words}), "clips": clips,
            "turns": [{"start": w["start"], "end": w["end"], "speaker": w["speaker"]} for w in words]}


def download_ami(out: Path = AMI_DATA) -> list[dict]:
    import zipfile

    out.mkdir(parents=True, exist_ok=True)
    index = out / "sessions.json"
    if index.is_file():
        return json.loads(index.read_text(encoding="utf-8"))
    archive = out / "ami_public_manual_1.6.2.zip"
    if not archive.is_file():
        urllib.request.urlretrieve(AMI_ANNOTATIONS_URL, archive)
    sessions = []
    with zipfile.ZipFile(archive) as z:
        for meeting in AMI_MEETINGS:
            wav = out / f"{meeting}.wav"
            if not wav.is_file():
                print(f"AMI {meeting}", file=sys.stderr)
                urllib.request.urlretrieve(AMI_AUDIO_URL.format(meeting), wav)
            with wave.open(str(wav)) as f:
                assert f.getframerate() == SAMPLE_RATE and f.getnchannels() == 1, "expected 16 kHz mono"
                seconds = f.getnframes() / f.getframerate()
            words = []
            for name in sorted(n for n in z.namelist() if n.startswith(f"words/{meeting}.") and n.endswith(".words.xml")):
                words += ami_words(z.read(name), name.split(".")[1])
            sessions.append(ami_session(meeting, words, seconds))
    index.write_text(json.dumps(sessions, ensure_ascii=False, indent=1), encoding="utf-8")
    return sessions


if __name__ == "__main__":
    if sys.argv[1:2] == ["ami"]:
        done = download_ami()
        print(f"{len(done)} meetings, {sum(s['seconds'] for s in done) / 60:.0f} min")
    elif sys.argv[1:2] == ["download"]:
        done = download()
        print(f"{len(done)} sessions, {sum(s['seconds'] for s in done) / 60:.0f} min")
    else:
        print(__doc__)
