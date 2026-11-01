"""NVIDIA Parakeet-TDT-0.6B-v3 and Canary-1B-v2 on the local GPU through onnx-asr (ONNX Runtime, CUDA).

Long audio is cut into speech segments of at most ``MAX_SPEECH_S`` seconds by Silero VAD, which
keeps memory flat (both models handle 20-30 s at a time). Parakeet returns token timestamps, so
words are timed without an alignment pass; Canary returns text only, so its segments go through
the same wav2vec2 alignment as WhisperX. Speakers come from the same pyannote diarization.

The ONNX exports are downloaded into ``<data_dir>/models/<model>`` rather than the HuggingFace
cache: ONNX Runtime refuses external weight files that live outside the model's own folder,
which is how the cache stores them on Windows.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import ClassVar

from ..models import Segment, TranscriptionResult, Word
from ..progress import Phase, Progress, ProgressCallback
from ..sources import Media
from . import common
from .base import Engine, EngineError, TranscribeOptions

log = logging.getLogger(__name__)

MAX_SPEECH_S = 20
# onnx-asr cuts at every pause over 100 ms by default. Such short pieces give Parakeet too little
# context to recognize the language (Czech came out partly Ukrainian, Russian or Polish on FLEURS),
# so pauses up to this long stay inside a piece of up to MAX_SPEECH_S seconds.
MIN_SILENCE_MS = 2000
VAD_REPO = "silero"
CANARY_LANGUAGES = {"bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "hr", "hu", "it", "lt", "lv", "mt",
                    "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "uk"}  # Parakeet v3 covers the same 25


def models_dir(settings) -> Path:
    return Path(settings.extra.get("models_dir") or settings.data_dir / "models")


def _download(name: str, folder: Path, progress: ProgressCallback) -> None:
    """Fetch the fp32 ONNX export once; a marker file tells a finished download from an interrupted one."""
    marker = folder / ".complete"
    if marker.is_file():
        return
    from huggingface_hub import snapshot_download
    from onnx_asr.resolver import model_repos

    progress(Progress(Phase.LOADING, message=f"Downloading {name} (first run only, several GB)..."))
    snapshot_download(model_repos[name], local_dir=folder,
                      allow_patterns=["config.json", "vocab.txt", "*.onnx", "*.onnx.data"],
                      ignore_patterns=["*int8*", "*fp16*"])
    shutil.rmtree(folder / ".cache", ignore_errors=True)
    marker.touch()


def _providers() -> list:
    import torch  # noqa: F401  (loads the CUDA/cuDNN DLLs that onnxruntime-gpu then reuses)
    import onnxruntime as rt

    if "CUDAExecutionProvider" not in rt.get_available_providers() or not torch.cuda.is_available():
        return ["CPUExecutionProvider"]
    # HEURISTIC: the default exhaustive cuDNN search re-runs for every new segment length (minutes per file).
    return [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}), "CPUExecutionProvider"]


def tokens_to_words(tokens: list[str], times: list[float], offset: float, end: float) -> list[Word]:
    """Join subword tokens into words; a token starting with a space starts a new word. A token that
    is only a space also does: Parakeet writes digits without the word-start mark, after a separate
    space token (" rok", " ", "2", "0" is "rok 20", not "rok20"). A word lasts until the next one
    starts, at most ``common.MAX_WORD_S``."""
    words: list[Word] = []
    new_word = True
    for token, t in zip(tokens, times):
        if token.strip() == "":
            new_word = True
            continue
        if new_word or token.startswith(" "):
            words.append(Word(token.strip(), round(offset + t, 3)))
        else:
            words[-1].word += token
        new_word = False
    return common.set_word_ends(words, end)


def enable_itn(model) -> None:
    """Make Canary write numbers as digits ("2027", not "twenty twenty seven"). onnx-asr hard-codes
    the ``<|noitn|>`` prompt token at position 7 of the decoder prompt; swap it when the layout matches."""
    asr = getattr(model, "asr", None)
    prompt, tokens = getattr(asr, "_transcribe_input", None), getattr(asr, "_tokens", None)
    if prompt is not None and tokens and "<|itn|>" in tokens and prompt[0, 7] == tokens.get("<|noitn|>"):
        prompt[0, 7] = tokens["<|itn|>"]


def load(name: str, settings, progress: ProgressCallback):
    """Download (once) and load an onnx-asr model on the GPU if possible. Returns ``(model, on_gpu)``."""
    import onnx_asr

    folder = models_dir(settings) / name
    _download(name, folder, progress)
    progress(Progress(Phase.LOADING, message=f"Loading {name}..."))
    model = onnx_asr.load_model(name, folder, providers=_providers())
    if "canary" in name:
        enable_itn(model)
    # ONNX Runtime silently falls back to the CPU when the CUDA libraries don't match; say so.
    encoder = getattr(getattr(model, "asr", None), "_encoder", None)
    on_gpu = bool(encoder) and encoder.get_providers()[0] == "CUDAExecutionProvider"
    progress(Progress(Phase.LOADING, message=f"{name} runs on the {'GPU' if on_gpu else 'CPU (slow)'}"))
    return model, on_gpu


class OnnxAsrEngine(Engine):
    supports_diarization = True
    onnx_model: ClassVar[str]
    timestamps: ClassVar[bool]  # True: the model times words itself, no alignment pass

    def check(self, options: TranscribeOptions) -> None:
        try:
            import onnx_asr  # noqa: F401
        except ImportError:
            raise EngineError("The onnx-asr package is missing. Install it with: uv sync")
        if options.language and options.language not in CANARY_LANGUAGES:
            raise EngineError(f"{self.name} does not support language '{options.language}'. "
                              f"Supported: {', '.join(sorted(CANARY_LANGUAGES))}")
        if options.diarize:
            common.check_diarization_access(self.settings)

    def _recognize_kwargs(self, options: TranscribeOptions) -> dict:
        return {}

    def transcribe(self, media: Media, options: TranscribeOptions, progress: ProgressCallback) -> TranscriptionResult:
        import onnx_asr

        model, _ = load(self.onnx_model, self.settings, progress)
        vad = onnx_asr.load_vad(VAD_REPO, providers=["CPUExecutionProvider"])
        recognizer = model.with_vad(vad, max_speech_duration_s=MAX_SPEECH_S, min_silence_duration_ms=MIN_SILENCE_MS,
                                    batch_size=max(1, min(options.batch_size, 4)))
        if self.timestamps:
            recognizer = recognizer.with_timestamps()

        audio, original = common.load_audio_pair(media.audio_path, self.settings.level_audio)
        duration = len(audio) / common.SAMPLE_RATE
        progress(Progress(Phase.TRANSCRIBE, 0, f"Transcribing ({self.onnx_model})..."))
        raw: list[dict] = []
        words: list[Word] = []
        pieces: list[str] = []  # text per VAD segment, for the language check
        previous_end = -1.0
        for seg in recognizer.recognize(audio, sample_rate=common.SAMPLE_RATE, **self._recognize_kwargs(options)):
            cut = seg.start < previous_end  # VAD cut through continuous speech: the pieces overlap
            previous_end = seg.end
            if not seg.text.strip():
                continue
            pieces.append(seg.text)
            if self.timestamps:
                new = tokens_to_words(seg.tokens or [], seg.timestamps or [], seg.start, seg.end)
                words += common.drop_cut_duplicate(words, new) if cut else new
            else:
                text = common.drop_cut_duplicate_text(raw[-1]["text"], seg.text.strip()) if cut and raw else seg.text.strip()
                if text:
                    raw.append({"start": round(seg.start, 3), "end": round(seg.end, 3), "text": " " + text})
            progress(Progress(Phase.TRANSCRIBE, min(100.0, 100 * seg.end / duration) if duration else None))
        del recognizer, model, vad
        common.free_gpu()

        if warning := common.language_warning(pieces, options.language):
            log.warning("%s: %s", self.name, warning)
            progress(Progress(Phase.TRANSCRIBE, 100, f"Warning: {warning}"))

        if self.timestamps:
            segments = [s.to_dict() for s in common.split_sentences(words)]
        else:
            segments = common.align(raw, audio, options.language or "en", progress) if raw else []
        voices: dict = {}
        if options.diarize and segments:
            segments = common.split_by_speaker(
                common.diarize(segments, original, self.settings, options.speakers, progress, voices))
        return TranscriptionResult([Segment.from_dict(s) for s in segments], language=options.language,
                                   duration=duration, model=self.onnx_model, speaker_embeddings=voices or None)


class ParakeetEngine(OnnxAsrEngine):
    name = "parakeet"
    default_model = "parakeet-tdt-0.6b-v3"
    onnx_model = "nemo-parakeet-tdt-0.6b-v3"
    # Parakeet has no language input: it detects the language itself and cannot be forced to one.
    # The chosen language is only used to check the result (``common.language_warning``).
    timestamps = True


class CanaryEngine(OnnxAsrEngine):
    name = "canary"
    default_model = "canary-1b-v2"
    onnx_model = "nemo-canary-1b-v2"
    timestamps = False

    def check(self, options: TranscribeOptions) -> None:
        if not options.language:
            raise EngineError("Canary cannot detect the language. Choose the language of the recording.")
        super().check(options)

    def _recognize_kwargs(self, options: TranscribeOptions) -> dict:
        return {"language": options.language, "pnc": True}
