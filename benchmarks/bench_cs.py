"""Speech recognition benchmark: WER, CER and speed on FLEURS, LibriSpeech, Common Voice cs and long real speech.

    uv run python -m benchmarks.bench_cs -n 300                   # every model, model call per utterance
    uv run python -m benchmarks.bench_cs -n 300 -m canary         # one model
    uv run python -m benchmarks.bench_cs -n 300 --app parakeet    # through the app's engine (see below)
    uv run python -m benchmarks.bench_cs report benchmarks/results/*.json   # Markdown table from saved runs

    --dataset fleurs      (default) Czech read sentences (FLEURS cs_cz), CC-BY-4.0, downloaded automatically
    --dataset fleurs-en   English FLEURS en_us; librispeech-clean / librispeech-other: LibriSpeech test sets
                          (CC-BY-4.0, openslr.org/12), a fixed random subset of -n utterances.
                          English text is normalized with Whisper's EnglishTextNormalizer (see normalize()).
    --dataset ami         AMI meeting corpus (CC-BY-4.0): 4 test meetings, Mix-Headset, English, with
                          speakers; through the app like parczech (WER in 60 s windows and whole, DER, time).
    --dataset cv          Common Voice cs test split (CC0). The download needs an account and consent at
                          Mozilla Data Collective: put the unpacked archive (test.tsv + clips/) into
                          benchmarks/data/cv_cs/. Reference sentences are NOT written to the result file.
    All datasets: reference/hypothesis text is left out of result files unless --keep-text is given (never for cv).
    --dataset parczech    ParCzech4Speech test split put together into 5-15 min sessions with >= 2 speakers
                          (benchmarks/longform.py; CC-BY). Always through the app (--app) with diarization:
                          WER per clip (words assigned to clips by time) and per whole session, DER with a
                          0.25 s collar, estimated vs. true number of speakers, transcription time.

The data (~450 MB) is downloaded on the first run into benchmarks/data/ (git-ignored; FLEURS is
CC-BY-4.0). Two ways to run a model, each in its own process:

* **model** (default): the model is loaded once and transcribes every utterance separately, without
  VAD or diarization. RTF = processing time / audio duration, model loading excluded.
* **app** (``--app ENGINE[:MODEL]``): the utterances are joined into one recording (1.5 s of silence
  between them) and transcribed by ``Engine.transcribe`` exactly as the app does it (VAD pieces, word
  timestamps, alignment; no diarization). Words are assigned back to the utterances by their start
  time. RTF here includes model loading and alignment, so it is the time a user waits.

Text normalization (same for reference and hypothesis): lowercase, punctuation and symbols removed,
whitespace collapsed. Numbers are not normalized ("2027" vs "dva tisíce dvacet sedm" are different
words), so every table also has the WER on the utterances whose reference contains no digit.
Error rates are corpus rates (errors / reference words) with a 95 % bootstrap interval over
utterances; differences between models are paired bootstraps on the same utterances.
Every hypothesis is saved in benchmarks/results/<date>-<run>-n<N>.json.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import io
import json
import platform
import re
import subprocess
import sys
import tarfile
import time
import unicodedata
import urllib.request
import wave
from datetime import date
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "fleurs_cs"
CV_DATA = HERE / "data" / "cv_cs"
CV_PER_SPEAKER = 3  # at most this many clips of one client, so the set has many speakers
DATASETS = ["fleurs", "fleurs-en", "librispeech-clean", "librispeech-other", "cv", "parczech", "ami"]
LANGUAGE = {"fleurs-en": "en", "librispeech-clean": "en", "librispeech-other": "en", "ami": "en"}  # default "cs"
LIBRISPEECH_URL = "https://www.openslr.org/resources/12/test-{}.tar.gz"
NORMALIZER_URL = "https://huggingface.co/openai/whisper-large-v3/resolve/main/normalizer.json"
LIBRISPEECH_PER_SPEAKER = 10  # test-other has 33 speakers, so 330 utterances are available
RESULTS = HERE / "results"
BASE_URL = "https://huggingface.co/datasets/google/fleurs/resolve/main/data/cs_cz/"
SAMPLE_RATE = 16000
MODELS = ["whisper-large-v3", "whisper-large-v3-turbo", "parakeet", "canary"]
GAP_S = 1.5  # silence between utterances in app mode
BOOTSTRAP = 2000
DIGIT = re.compile(r"\d")


# --- data -------------------------------------------------------------------------------

def fleurs_dir(dataset: str) -> Path:
    return HERE / "data" / ("fleurs_en" if dataset == "fleurs-en" else "fleurs_cs")


def download(dataset: str = "fleurs") -> None:
    folder = fleurs_dir(dataset)
    base = BASE_URL.replace("cs_cz", "en_us") if dataset == "fleurs-en" else BASE_URL
    folder.mkdir(parents=True, exist_ok=True)
    tsv = folder / "test.tsv"
    if not tsv.is_file():
        urllib.request.urlretrieve(base + "test.tsv", tsv)
    audio_dir = folder / "test"
    if not audio_dir.is_dir() or not any(audio_dir.iterdir()):
        print("Downloading FLEURS test audio (~300-450 MB)...", file=sys.stderr)
        archive = folder / "test.tar.gz"
        urllib.request.urlretrieve(base + "audio/test.tar.gz", archive)
        with tarfile.open(archive) as tar:
            tar.extractall(folder, filter="data")
        archive.unlink()


def download_librispeech(subset: str) -> Path:
    """Extract ``test-clean`` or ``test-other`` (CC-BY-4.0) into benchmarks/data/librispeech/."""
    folder = HERE / "data" / "librispeech"
    if not (folder / "LibriSpeech" / f"test-{subset}").is_dir():
        folder.mkdir(parents=True, exist_ok=True)
        print(f"Downloading LibriSpeech test-{subset} (~340 MB)...", file=sys.stderr)
        archive = folder / f"test-{subset}.tar.gz"
        urllib.request.urlretrieve(LIBRISPEECH_URL.format(subset), archive)
        with tarfile.open(archive) as tar:
            tar.extractall(folder, filter="data")
        archive.unlink()
    return folder / "LibriSpeech" / f"test-{subset}"


def read_wav(path: Path) -> np.ndarray:
    """FLEURS WAVs are 32-bit float; ffmpeg (via WhisperX) decodes them to mono 16 kHz."""
    from mgtranscriber.engines.common import load_audio

    return load_audio(path)


def utterances(n: int, dataset: str = "fleurs") -> list[dict]:
    if dataset == "cv":
        return cv_utterances(n)
    if dataset.startswith("librispeech"):
        return librispeech_utterances(n, dataset.split("-")[1])
    folder = fleurs_dir(dataset)
    rows = list(csv.reader(io.open(folder / "test.tsv", encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE))
    if dataset == "fleurs-en":
        # en_us has several recordings of the same sentence in a row; joined into one recording they make
        # the decoder skip or merge repeats, which is not an error of the model. Every sentence once.
        seen: set[str] = set()
        rows = [r for r in rows if not (r[2] in seen or seen.add(r[2]))]
    return [{"id": row[1], "audio": read_wav(folder / "test" / row[1]), "reference": row[2]} for row in rows[:n]]


def librispeech_select(rows: list[dict], n: int, per_speaker: int = LIBRISPEECH_PER_SPEAKER, seed: int = 0) -> list[dict]:
    """``n`` of the rows ({"id", "text", "path"}) in a fixed random order, at most ``per_speaker`` per speaker
    (the id starts with the speaker number), so the subset covers many voices and chapters."""
    order = np.random.default_rng(seed).permutation(len(rows))
    taken, speakers = [], {}
    for i in order:
        speaker = rows[i]["id"].split("-")[0]
        if speakers.get(speaker, 0) >= per_speaker:
            continue
        speakers[speaker] = speakers.get(speaker, 0) + 1
        taken.append(rows[i])
        if len(taken) == n:
            break
    return sorted(taken, key=lambda r: r["id"])


def librispeech_utterances(n: int, subset: str) -> list[dict]:
    root = download_librispeech(subset)
    rows = []
    for trans in sorted(root.rglob("*.trans.txt")):
        for line in trans.read_text(encoding="utf-8").splitlines():
            utt, _, text = line.partition(" ")
            rows.append({"id": utt, "text": text, "path": trans.parent / f"{utt}.flac"})
    return [{"id": r["id"], "audio": read_wav(r["path"]), "reference": r["text"]} for r in librispeech_select(rows, n)]


def cv_select(rows: list[dict], n: int, per_speaker: int = CV_PER_SPEAKER, seed: int = 0) -> list[dict]:
    """``n`` rows in a fixed random order, at most ``per_speaker`` per client and each sentence once."""
    order = np.random.default_rng(seed).permutation(len(rows))
    taken, clients, sentences = [], {}, set()
    for i in order:
        row = rows[i]
        if clients.get(row["client_id"], 0) >= per_speaker or row["sentence"] in sentences:
            continue
        clients[row["client_id"]] = clients.get(row["client_id"], 0) + 1
        sentences.add(row["sentence"])
        taken.append(row)
        if len(taken) == n:
            break
    return taken


def cv_utterances(n: int) -> list[dict]:
    tsvs = sorted(CV_DATA.rglob("test.tsv"))
    if not tsvs:
        raise SystemExit(f"Common Voice not found: put the unpacked archive (test.tsv + clips/) into {CV_DATA}.\n"
                         "Download needs an account and consent at Mozilla Data Collective (CC0, cs, v25.0 and v27.0 have the same columns).")
    tsv = tsvs[0]
    with open(tsv, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    return [{"id": r["path"], "audio": read_wav(tsv.parent / "clips" / r["path"]), "reference": r["sentence"]}
            for r in cv_select(rows, n)]


# --- metrics ----------------------------------------------------------------------------

_english = None
LANGUAGE_NOW = "cs"


def set_language(language: str) -> None:
    """``en`` switches normalize() to Whisper's EnglishTextNormalizer (digits and number words become the
    same, "don't" stays one word, fillers like "uh"/"mm" and bracketed notes are dropped, British spellings
    become American). Both sides of the comparison are normalized the same way, so a model is not
    penalized for how it formats numbers. Czech keeps the plain normalization below."""
    global _english, LANGUAGE_NOW
    LANGUAGE_NOW = language
    if language == "en" and _english is None:
        from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

        path = HERE / "data" / "english_spelling.json"
        if not path.is_file():
            path.parent.mkdir(exist_ok=True)
            urllib.request.urlretrieve(NORMALIZER_URL, path)
        _english = EnglishTextNormalizer(json.loads(path.read_text(encoding="utf-8")))


def normalize(text: str) -> str:
    if LANGUAGE_NOW == "en" and _english is not None:
        return " ".join(_english(text).split())
    text = unicodedata.normalize("NFC", text).lower()
    text = "".join(" " if unicodedata.category(c).startswith(("P", "S")) else c for c in text)
    return " ".join(text.split())


def edit_distance(ref: list, hyp: list) -> int:
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        current = [i]
        for j, h in enumerate(hyp, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (r != h)))
        previous = current
    return previous[-1]


def counts(ref: str, hyp: str) -> tuple[int, int, int, int]:
    """(word errors, reference words, character errors, reference characters) of one utterance."""
    ref, hyp = normalize(ref), normalize(hyp)
    chars_r, chars_h = ref.replace(" ", ""), hyp.replace(" ", "")
    return (edit_distance(ref.split(), hyp.split()), len(ref.split()),
            edit_distance(list(chars_r), list(chars_h)), len(chars_r))


def word_counts(ref: str, hyp: str) -> tuple[int, int]:
    """(word errors, reference words); no character distance, which is too slow on whole long recordings."""
    ref, hyp = normalize(ref).split(), normalize(hyp).split()
    return edit_distance(ref, hyp), len(ref)


def error_rates(pairs: list[tuple[str, str]]) -> tuple[float, float]:
    """Corpus WER and CER in percent over (reference, hypothesis) pairs."""
    c = np.array([counts(r, h) for r, h in pairs]).reshape(-1, 4).sum(axis=0)
    return 100 * c[0] / max(c[1], 1), 100 * c[2] / max(c[3], 1)


def rate_with_interval(errors: np.ndarray, total: np.ndarray, seed: int = 0) -> tuple[float, float, float]:
    """Corpus rate in percent and its 95 % bootstrap interval (resampling utterances)."""
    rate = 100 * errors.sum() / max(total.sum(), 1)
    idx = np.random.default_rng(seed).integers(0, len(errors), (BOOTSTRAP, len(errors)))
    samples = 100 * errors[idx].sum(axis=1) / np.maximum(total[idx].sum(axis=1), 1)
    return rate, float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def paired_difference(a: dict, b: dict, seed: int = 0) -> tuple[float, float, float]:
    """WER of ``a`` minus WER of ``b`` (points) with a 95 % paired bootstrap interval."""
    ea, eb, n = np.array(a["word_errors"]), np.array(b["word_errors"]), np.array(a["ref_words"])
    idx = np.random.default_rng(seed).integers(0, len(n), (BOOTSTRAP, len(n)))
    diff = 100 * (ea[idx].sum(axis=1) - eb[idx].sum(axis=1)) / n[idx].sum(axis=1)
    return 100 * (ea.sum() - eb.sum()) / n.sum(), float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))


def summarize(items: list[dict]) -> dict:
    """Per-utterance counts plus corpus rates with intervals, on all utterances and on those without digits."""
    c = np.array([counts(i["ref"], i["hyp"]) for i in items])
    plain = np.array([not DIGIT.search(i["ref"]) for i in items])
    wer = rate_with_interval(c[:, 0], c[:, 1])
    cer = rate_with_interval(c[:, 2], c[:, 3])
    wer_plain = rate_with_interval(c[plain, 0], c[plain, 1])
    return {"wer": round(wer[0], 2), "wer_ci": [round(wer[1], 2), round(wer[2], 2)],
            "cer": round(cer[0], 2), "cer_ci": [round(cer[1], 2), round(cer[2], 2)],
            "word_errors_total": int(c[:, 0].sum()), "ref_words_total": int(c[:, 1].sum()),
            "wer_no_digits": round(wer_plain[0], 2), "wer_no_digits_ci": [round(wer_plain[1], 2), round(wer_plain[2], 2)],
            "utterances_no_digits": int(plain.sum()),
            "word_errors": c[:, 0].tolist(), "ref_words": c[:, 1].tolist()}


# --- models (model mode) ----------------------------------------------------------------

def whisper_runner(name: str, language: str = "cs"):
    import torch
    import whisperx

    model = whisperx.load_model(name, "cuda" if torch.cuda.is_available() else "cpu",
                                compute_type="float16" if torch.cuda.is_available() else "int8", language=language)

    def run(audio: np.ndarray) -> str:
        result = model.transcribe(audio, batch_size=8, language=language)
        return " ".join(s["text"].strip() for s in result["segments"])

    return run, "CUDA" if torch.cuda.is_available() else "CPU"


def onnx_runner(name: str, language: str = "cs"):
    from mgtranscriber.config import load_settings
    from mgtranscriber.engines.onnx_asr_local import load

    model, on_gpu = load(name, load_settings(), lambda p: print(p.message or "", file=sys.stderr) if p.message else None)
    # Canary is told the language; Parakeet has no language input at all (it detects it itself)
    kwargs = {"language": language, "pnc": True} if "canary" in name else {}

    def run(audio: np.ndarray) -> str:
        return model.recognize(audio, sample_rate=SAMPLE_RATE, **kwargs)

    return run, "CUDAExecutionProvider" if on_gpu else "CPUExecutionProvider"


RUNNERS = {
    "whisper-large-v3": lambda lang: whisper_runner("large-v3", lang),
    "whisper-large-v3-turbo": lambda lang: whisper_runner("large-v3-turbo", lang),
    "parakeet": lambda lang: onnx_runner("nemo-parakeet-tdt-0.6b-v3", lang),
    "canary": lambda lang: onnx_runner("nemo-canary-1b-v2", lang),
}


def gpu_name() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except OSError:
        return "CPU"


def environment() -> dict:
    return {"date": date.today().isoformat(), "gpu": gpu_name(), "python": platform.python_version()}


def bench(name: str, data: list[dict], warmup: int = 3, dataset: str = "fleurs") -> dict:
    started = time.perf_counter()
    run, device = RUNNERS[name](LANGUAGE.get(dataset, "cs"))
    load_s = time.perf_counter() - started
    for item in data[:warmup]:  # first calls initialize CUDA kernels; not counted
        run(item["audio"])
    items, seconds = [], 0.0
    started = time.perf_counter()
    for item in data:
        items.append({"id": item["id"], "ref": item["reference"], "hyp": run(item["audio"])})
        seconds += len(item["audio"]) / SAMPLE_RATE
    elapsed = time.perf_counter() - started
    return {"run": name, "dataset": dataset, "mode": "model", "device": device, "utterances": len(data), "audio_seconds": round(seconds, 1),
            "rtf": round(elapsed / seconds, 4), "load_seconds": round(load_s, 1), **environment(),
            **summarize(items), "items": items}


# --- app mode ---------------------------------------------------------------------------

def peak_normalize(audio: np.ndarray, peak: float = 0.7) -> np.ndarray:
    top = float(np.abs(audio).max()) if len(audio) else 0.0
    return audio if top == 0.0 else (audio * (peak / top)).astype(np.float32)


def joined_recording(data: list[dict], path: Path) -> list[tuple[float, float]]:
    """Write the utterances into one 16-bit WAV with ``GAP_S`` of silence between them; returns their spans."""
    gap = np.zeros(int(GAP_S * SAMPLE_RATE), dtype=np.float32)
    parts, spans, t = [], [], 0.0
    for item in data:
        parts += [item["audio"], gap]
        duration = len(item["audio"]) / SAMPLE_RATE
        spans.append((t, t + duration))
        t += duration + GAP_S
    audio = np.concatenate(parts)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    return spans


def assign_words(segments, spans: list[tuple[float, float]], gap: float = GAP_S) -> list[str]:
    """Hypothesis text per utterance: every word goes to the utterance whose span (extended to the
    middle of the surrounding gaps) contains the word's start. Words without a time follow the previous word."""
    borders = [end + gap / 2 for _, end in spans[:-1]]
    texts: list[list[str]] = [[] for _ in spans]
    current = 0
    for seg in segments:
        if not seg.words:
            texts[bisect.bisect_right(borders, seg.start)] += seg.text.split()
            continue
        for word in seg.words:
            if word.start is not None:
                current = bisect.bisect_right(borders, word.start)
            texts[current].append(word.word)
    return [" ".join(t) for t in texts]


def bench_app(spec: str, data: list[dict], dataset: str = "fleurs", peak: bool = True) -> dict:
    from mgtranscriber.config import load_settings
    from mgtranscriber.engines import TranscribeOptions, get_engine
    from mgtranscriber.sources import Media

    engine_name, _, model = spec.partition(":")
    if dataset.startswith("librispeech"):
        folder = HERE / "data" / "librispeech"
    else:
        folder = CV_DATA if dataset == "cv" else fleurs_dir(dataset)
    if dataset == "fleurs-en" and peak:
        # some FLEURS en recordings are 50 dB quieter than the rest (peak 0.002); without the app's own leveling
        # (mgtranscriber/leveling.py) its VAD dropped them between loud ones. --no-peak-normalize skips this.
        data = [{**d, "audio": peak_normalize(d["audio"])} for d in data]
    path = folder / f"joined-{dataset}-n{len(data)}.wav"
    spans = joined_recording(data, path)
    engine = get_engine(engine_name, load_settings())
    log: list[str] = []

    def progress(p) -> None:
        if p.message:
            log.append(p.message)
            print(p.message, file=sys.stderr)

    options = TranscribeOptions(language=LANGUAGE.get(dataset, "cs"), model=model or None, diarize=False)
    engine.check(options)
    started = time.perf_counter()
    result = engine.transcribe(Media(str(path), "file", path.stem, audio_path=path), options, progress)
    elapsed = time.perf_counter() - started
    hyps = assign_words(result.segments, spans)
    items = [{"id": d["id"], "ref": d["reference"], "hyp": h} for d, h in zip(data, hyps)]
    seconds = spans[-1][1] + GAP_S
    device = next((m.split(" runs on the ")[1] for m in log if " runs on the " in m), "")
    return {"run": f"app-{spec.replace(':', '-')}", "dataset": dataset, "mode": "app", "engine": engine_name, "model": result.model,
            "device": device, "utterances": len(data), "audio_seconds": round(seconds, 1),
            "rtf": round(elapsed / seconds, 4), "seconds_total": round(elapsed, 1), **environment(),
            "warnings": [m for m in log if m.startswith("Warning")], **summarize(items), "items": items}


# --- long sessions with speakers (parczech) -----------------------------------------------

def bench_sessions(spec: str, sessions: list[dict], root: Path, dataset: str = "parczech") -> dict:
    """Every session through ``Engine.transcribe`` with diarization (number of speakers unknown to the app)."""
    from benchmarks import longform
    from mgtranscriber.config import load_settings
    from mgtranscriber.engines import TranscribeOptions, get_engine
    from mgtranscriber.sources import Media

    engine_name, _, model = spec.partition(":")
    engine = get_engine(engine_name, load_settings())
    options = TranscribeOptions(language=LANGUAGE.get(dataset, "cs"), model=model or None, diarize=True)
    engine.check(options)
    items, details, warnings, device, model_name = [], [], [], "", ""
    for session in sessions:
        path = root / session["audio"]
        log: list[str] = []

        def progress(p, log=log) -> None:
            if p.message:
                log.append(p.message)
                print(p.message, file=sys.stderr)

        started = time.perf_counter()
        result = engine.transcribe(Media(str(path), "file", path.stem, audio_path=path), options, progress)
        elapsed = time.perf_counter() - started
        model_name = result.model
        device = device or next((m.split(" runs on the ")[1] for m in log if " runs on the " in m), "")
        warnings += [m for m in log if m.startswith("Warning")]
        spans = [(c["start"], c["end"]) for c in session["clips"]]
        hyps = assign_words(result.segments, spans, session.get("gap", longform.CLIP_GAP_S))
        items += [{"id": c["id"], "ref": c["text"], "hyp": h, "speaker": c.get("speaker")}
                  for c, h in zip(session["clips"], hyps)]
        der = longform.diarization_error(longform.reference_annotation(session.get("turns") or session["clips"]),
                                         longform.hypothesis_annotation(result.segments))
        werr, words = word_counts(" ".join(c["text"] for c in session["clips"]), " ".join(hyps))
        details.append({"id": session["id"], "seconds": session["seconds"], "transcribe_seconds": round(elapsed, 1),
                        "rtf": round(elapsed / session["seconds"], 4), "word_errors": werr, "ref_words": words,
                        "wer_whole": round(100 * werr / max(words, 1), 2), **der})
    seconds = sum(s["seconds"] for s in sessions)
    elapsed = sum(d["transcribe_seconds"] for d in details)
    werr, words = sum(d["word_errors"] for d in details), sum(d["ref_words"] for d in details)
    return {"run": f"app-{spec.replace(':', '-')}", "dataset": dataset, "mode": "app", "engine": engine_name,
            "model": model_name, "device": device, "utterances": len(items), "sessions": details,
            "audio_seconds": round(seconds, 1), "rtf": round(elapsed / seconds, 4), "seconds_total": round(elapsed, 1),
            "wer_whole": round(100 * werr / max(words, 1), 2), "der": longform.pooled_der(details),
            "speakers_ref": sum(d["speakers_ref"] for d in details),
            "speakers_hyp": sum(d["speakers_hyp"] for d in details),
            **environment(), "warnings": warnings, **summarize(items), "items": items}


# --- report -----------------------------------------------------------------------------

def table(results: list[dict]) -> str:
    best = min(results, key=lambda r: r["wer"])
    lines = ["| run | WER % (95% CI) | errors / words | WER % without digits | CER % | Δ WER vs best | RTF | ×realtime |",
             "|---|---|---|---|---|---|---|---|"]
    for r in results:
        if r is best or [i["id"] for i in r["items"]] != [i["id"] for i in best["items"]]:
            delta = "–"
        else:
            d, lo, hi = paired_difference(r, best)
            delta = f"+{d:.2f} ({lo:+.2f}…{hi:+.2f})"
        lines.append(f"| {r['run']} | {r['wer']:.2f} ({r['wer_ci'][0]:.1f}–{r['wer_ci'][1]:.1f}) | "
                     f"{r['word_errors_total']} / {r['ref_words_total']} | {r['wer_no_digits']:.2f} | {r['cer']:.2f} | "
                     f"{delta} | {r['rtf']:.3f} | {1 / r['rtf']:.0f}× |")
    return "\n".join(lines)


def session_table(results: list[dict]) -> str:
    lines = ["| run | WER % per clip (95% CI) | WER % whole sessions | DER % | miss / FA / confusion % | speakers est. / true | time | ×realtime |",
             "|---|---|---|---|---|---|---|---|"]
    for r in results:
        parts = r["sessions"]
        total = sum(p["ref_speech_s"] for p in parts) or 1.0
        mfc = [sum(p[k] * p["ref_speech_s"] for p in parts) / total for k in ("miss", "false_alarm", "confusion")]
        lines.append(f"| {r['run']} | {r['wer']:.2f} ({r['wer_ci'][0]:.1f}–{r['wer_ci'][1]:.1f}) | {r['wer_whole']:.2f} | "
                     f"{r['der']:.2f} | {mfc[0]:.1f} / {mfc[1]:.1f} / {mfc[2]:.1f} | "
                     f"{r['speakers_hyp']} / {r['speakers_ref']} | {r['seconds_total'] / 60:.1f} min | {1 / r['rtf']:.1f}× |")
    return "\n".join(lines)


def report(results: list[dict]) -> str:
    plain = [r for r in results if "sessions" not in r]
    long = [r for r in results if "sessions" in r]
    return "\n\n".join(x for x in [table(plain) if plain else "", session_table(long) if long else ""] if x)


def run_name(dataset: str, mode: str, name: str, n: int, peak: bool = True) -> str:
    prefix = "" if dataset == "fleurs" else f"{dataset}-"
    tail = "" if dataset in ("parczech", "ami") else f"-n{n}"
    return f"{date.today().isoformat()}-{prefix}{'app-' if mode == 'app' else ''}{name.replace(':', '-')}{'' if peak else '-nopeak'}{tail}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-n", type=int, default=200, help="number of utterances (default 200)")
    parser.add_argument("-m", "--model", action="append", choices=MODELS, help="model(s) to run per utterance")
    parser.add_argument("--app", action="append", metavar="ENGINE[:MODEL]", help="run through Engine.transcribe")
    parser.add_argument("--dataset", choices=DATASETS, default="fleurs")
    parser.add_argument("--no-peak-normalize", action="store_true",
                        help="app mode, fleurs-en: do not peak-normalize the utterances (the app's own leveling is measured)")
    parser.add_argument("--keep-text", action="store_true",
                        help="keep reference and hypothesis text in the result files (local analysis; do not commit; never for cv)")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # the tables contain "Δ" and "×"; the Windows console is cp1250

    if args.dataset in ("fleurs", "fleurs-en"):
        download(args.dataset)
    if args.dataset in ("parczech", "ami"):
        if args.model or not args.app:
            parser.error(f"--dataset {args.dataset} runs long recordings: use --app ENGINE[:MODEL]")
        from benchmarks import longform
        longform.download_ami() if args.dataset == "ami" else longform.download()
    RESULTS.mkdir(exist_ok=True)
    runs = [("model", m) for m in args.model or ([] if args.app else MODELS)] + [("app", a) for a in args.app or []]
    results = []
    for mode, name in runs:
        # each run in its own process, so GPU memory of the previous one is really freed
        out = RESULTS / (run_name(args.dataset, mode, name, args.n, not args.no_peak_normalize) + ".json")
        subprocess.run([sys.executable, "-m", "benchmarks.bench_cs", "--one", mode, name, "-n", str(args.n), "--out", str(out),
                        "--dataset", args.dataset] + (["--no-peak-normalize"] if args.no_peak_normalize else [])
                       + (["--keep-text"] if args.keep_text else []), check=True, cwd=HERE.parent)
        results.append(json.loads(out.read_text(encoding="utf-8")))
        print(f"{name}: WER {results[-1]['wer']} %, RTF {results[-1]['rtf']}", file=sys.stderr)
    print(report(results))


if __name__ == "__main__":
    if "--one" in sys.argv:
        one = argparse.ArgumentParser()
        one.add_argument("--one", nargs=2)
        one.add_argument("-n", type=int)
        one.add_argument("--out", type=Path)
        one.add_argument("--dataset", default="fleurs")
        one.add_argument("--no-peak-normalize", action="store_true")
        one.add_argument("--keep-text", action="store_true")
        a = one.parse_args()
        mode, name = a.one
        set_language(LANGUAGE.get(a.dataset, "cs"))
        if a.dataset in ("parczech", "ami"):
            from benchmarks import longform
            if a.dataset == "ami":
                result = bench_sessions(name, longform.download_ami(), longform.AMI_DATA, "ami")
            else:
                result = bench_sessions(name, longform.download(), longform.DATA)
        else:
            data = utterances(a.n, a.dataset)
            result = bench_app(name, data, a.dataset, not a.no_peak_normalize) if mode == "app" else bench(name, data, dataset=a.dataset)
        # Reference and hypothesis text stays out of the result files (Common Voice: Mozilla's terms; the rest: no
        # need to republish corpus text). Only --keep-text keeps it, and never for Common Voice.
        if a.dataset == "cv" or not a.keep_text:
            result["items"] = [{k: v for k, v in i.items() if k not in ("ref", "hyp")} for i in result["items"]]
            result["references_published"] = result["hypotheses_published"] = False
        a.out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    elif sys.argv[1:2] == ["report"]:
        sys.stdout.reconfigure(encoding="utf-8")
        print(report([json.loads(Path(p).read_text(encoding="utf-8")) for p in sys.argv[2:]]))
    else:
        main()
