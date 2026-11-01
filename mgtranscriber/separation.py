"""Speech separation: one 16 kHz mono track per speaker, so that overlapping speech can be transcribed.

Whisper and Parakeet write down only the dominant voice when people talk over each other. PixIT
(``pyannote/speech-separation-ami-1.0``, joint diarization + separation) splits the recording into one source per
speaker and says when each speaker is active, so the tracks and the turns are consistent.

The pyannote pipeline keeps the separated windows of the whole file in memory (90 % window overlap, roughly ten
times the audio), so a long recording is processed in chunks of ``CHUNK_S`` seconds with ``PAD_S`` of context on
both sides. The speakers of different chunks are tied together by their embedding centroids. Tracks are streamed
to disk chunk by chunk, so the pipeline's working memory is bounded by the chunk size; the input itself is
decoded once as float32 (about 230 MB per hour of audio).

Heavy imports (torch, pyannote) stay inside the functions so importing this module is cheap.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .engines.base import EngineError
from .engines.common import SAMPLE_RATE, device, env_value, free_gpu, load_audio
from .progress import Progress, ProgressCallback, no_progress

if TYPE_CHECKING:
    import numpy as np

    from .config import Settings

SEPARATION_MODEL = "pyannote/speech-separation-ami-1.0"
SEPARATION_MODEL_URL = f"https://huggingface.co/{SEPARATION_MODEL}"
SEGMENTATION_MODEL = "pyannote/separation-ami-1.0"  # the network inside the pipeline, gated on its own
PHASE = "separate"
CHUNK_S = 600.0  # audio processed by one pipeline run; keep >= 300: in 100 s chunks two speakers of a dialogue
#                  collapsed into one cluster, 300 s chunks separated them cleanly (clustering needs the context)
MIN_SPEAKER_S = 3.0  # a global speaker with less speech overall is a segmentation/embedding ghost: no track
PAD_S = 15.0  # context around a chunk that is run but not kept (the pipeline is weakest at its edges)
MATCH_COSINE = 0.5  # centroids of one speaker in two chunks are at least this similar
MIN_TURN_S = 0.0  # 0.1 would cut the measured speech activity by ~25 % (flickering fragments are chained into turns)
EMBED_MAX_S = 30.0  # speech of one speaker in a chunk used for its embedding
EMBED_MIN_S = 0.5


@dataclass
class SpeakerTrack:
    speaker: str
    path: Path  # 16 kHz mono wav of this speaker only
    # Always None for now: the separation pipeline embeds speakers with speechbrain/spkrec-ecapa-voxceleb
    # (192-dim), voice memory stores community-1's WeSpeaker embedding (256-dim); the two spaces cannot be
    # compared, so these vectors must not be matched against stored voices.
    embedding: list[float] | None = None


@dataclass
class SeparationResult:
    tracks: list[SpeakerTrack]
    turns: list[tuple[float, float, str]]  # (start, end, speaker); overlaps allowed


def _token(settings: "Settings") -> str | None:
    return settings.hf_token or env_value(settings, "HF_TOKEN")


def is_available(settings: "Settings") -> tuple[bool, str | None]:
    """``(True, None)`` or ``(False, reason)``. Does not load the model; asks HuggingFace only when a model is
    not cached yet. The pipeline needs two gated repos (the pipeline and its segmentation/separation network)
    and the speechbrain package for its speaker embedding."""
    try:
        import pyannote.audio  # noqa: F401
        import scipy  # noqa: F401
        import torch  # noqa: F401
    except ImportError as e:
        return False, f"Speech separation needs pyannote-audio and torch ({e})."
    try:
        import speechbrain  # noqa: F401
    except ImportError:
        return False, "Speech separation needs the 'speechbrain' package (speaker embedding of the separation model)."
    from huggingface_hub import HfApi, try_to_load_from_cache
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    token = _token(settings)
    for repo, cached_file in ((SEPARATION_MODEL, "config.yaml"), (SEGMENTATION_MODEL, "pytorch_model.bin")):
        if isinstance(try_to_load_from_cache(repo, cached_file), (str, Path)):
            continue
        if not token:
            return False, "HF_TOKEN is missing: speech separation needs a HuggingFace token (free 'Read' token)."
        try:
            HfApi().auth_check(repo, token=token)
        except GatedRepoError:
            return False, f"No access to the separation model. Accept its terms at https://huggingface.co/{repo}"
        except HfHubHTTPError as e:
            return False, f"HuggingFace rejected the token or is unreachable: {e}"
        except OSError as e:  # offline
            return False, f"The separation model is not downloaded and HuggingFace is unreachable: {e}"
    return True, None


def _patch_speechbrain(models_dir: Path) -> None:
    """pyannote-audio 4.0.7 calls ``EncoderClassifier.from_hparams(token=, huggingface_cache_dir=, revision=,
    savedir=f"{cache_dir}/speechbrain")``, which speechbrain 1.1.1 (the newest) does not know: the keywords
    would reach the model's constructor, and its default link strategy is a symlink, which needs Developer
    Mode on Windows (WinError 1314); the device arrives as a torch.device, not a string. Translate the call:
    public model, so the HF token is not needed; copy files instead of linking; the savedir goes under
    ``models_dir``."""
    from pyannote.audio.pipelines import speaker_verification as sv
    from speechbrain.utils.fetching import FetchConfig, LocalStrategy

    base = sv.SpeechBrain_EncoderClassifier
    if getattr(base, "_mgtranscriber_patched", False):
        return

    class CopyingClassifier(base):
        _mgtranscriber_patched = True

        @classmethod
        def from_hparams(cls, source, hparams_file="hyperparams.yaml", token=None, revision=None,
                         huggingface_cache_dir=None, savedir=None, **kwargs):
            if savedir is None or str(savedir).startswith("None"):
                savedir = models_dir / "speechbrain" / str(source).replace("/", "--")
            Path(savedir).mkdir(parents=True, exist_ok=True)
            if kwargs.get("run_opts"):  # speechbrain wants the device as a string, pyannote hands a torch.device
                kwargs["run_opts"] = {k: str(v).replace("cuda", "cuda:0") if k == "device" and str(v) == "cuda" else
                                      str(v) if k == "device" else v for k, v in kwargs["run_opts"].items()}
            kwargs.setdefault("local_strategy", LocalStrategy.COPY)
            kwargs.setdefault("fetch_config", FetchConfig(revision=revision, huggingface_cache_dir=huggingface_cache_dir))
            return super().from_hparams(source, hparams_file, savedir=str(savedir), **kwargs)

    sv.SpeechBrain_EncoderClassifier = CopyingClassifier


def install_asteroid_shim() -> None:
    """ToTaToNet needs ``DPRNN`` and ``pad_x_to_y`` from asteroid, whose package pins an old pytorch-lightning.
    pyannote decides at import time whether asteroid exists (``ASTEROID_IS_AVAILABLE``); when it does not, or the
    installed one is broken, the vendored subset (``_vendor/asteroid_min``) is set on pyannote's ToTaToNet module
    directly. Nothing is registered in ``sys.modules``. A working installed asteroid is left alone."""
    import importlib

    module = importlib.import_module("pyannote.audio.models.separation.ToTaToNet")  # the module, not the class
    if getattr(module, "ASTEROID_IS_AVAILABLE", False):
        return
    try:
        from asteroid.masknn import DPRNN
        from asteroid.utils.torch_utils import pad_x_to_y
    except Exception:  # missing, or installed but broken
        from ._vendor.asteroid_min import DPRNN, pad_x_to_y
    module.DPRNN, module.pad_x_to_y, module.ASTEROID_IS_AVAILABLE = DPRNN, pad_x_to_y, True


def _load_pipeline(settings: "Settings"):
    import torch

    ok, reason = is_available(settings)
    if not ok:
        raise EngineError(reason or "Speech separation is not available.")
    try:
        from pyannote.audio import Pipeline

        install_asteroid_shim()
        _patch_speechbrain(Path(settings.extra.get("models_dir") or settings.data_dir / "models"))
        pipeline = Pipeline.from_pretrained(SEPARATION_MODEL, token=_token(settings))
    except EngineError:
        raise
    except Exception as e:  # gated repo, offline, incompatible package versions...
        raise EngineError(f"The speech separation model could not be loaded: {type(e).__name__}: {e}") from e
    if pipeline is None:
        raise EngineError(f"The separation model could not be loaded. Accept its terms at {SEPARATION_MODEL_URL}")
    target = torch.device(device())
    pipeline.to(target)
    embedder = getattr(pipeline, "_embedding", None)  # a plain attribute of the pipeline: not moved by .to()
    if embedder is not None and hasattr(embedder, "to"):
        embedder.to(target)
    return pipeline


class _Tracks:
    """The growing global speaker set: centroid, wav writer and number of samples written per speaker."""

    def __init__(self, workdir: Path, num_speakers: int | None):
        self.workdir = workdir
        self.limit = num_speakers
        self.centroids: list["np.ndarray"] = []  # running weighted mean of unit vectors
        self.weights: list[float] = []
        self.paths: list[Path] = []
        self.writers: list[wave.Wave_write] = []
        self.written: list[int] = []

    def _new(self, centroid) -> int:
        index = len(self.centroids)
        path = self.workdir / f"SPEAKER_{index:02d}.wav"
        writer = wave.open(str(path), "wb")
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(SAMPLE_RATE)
        self.centroids.append(centroid)
        self.weights.append(0.0)
        self.paths.append(path)
        self.writers.append(writer)
        self.written.append(0)
        return index

    def match(self, vectors: list["np.ndarray | None"], weights: list[float]) -> list[int | None]:
        """Global index for every local speaker of a chunk, ``None`` for one that has no usable vector (missing,
        or of another dimension than the known centroids: it cannot be told whom it is, the caller decides).
        Speakers with a vector are assigned one-to-one to the known ones by cosine similarity; the rest become
        new speakers (with a speaker limit they join the closest known one). Matched centroids become a running
        mean weighted by ``weights`` (speech seconds)."""
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        def unit(v):
            return v / (np.linalg.norm(v) + 1e-9)

        dim = self.centroids[0].shape if self.centroids else None
        local: list["np.ndarray | None"] = []
        for v in vectors:
            v = None if v is None else np.nan_to_num(np.asarray(v, dtype=np.float64)).reshape(-1)
            if v is not None:  # the pipeline pads zero centroids for speakers it has none for
                v = unit(v) if np.linalg.norm(v) > 1e-6 else None
            local.append(v if v is not None and (dim is None or v.shape == dim) else None)
        result: list[int | None] = [None] * len(local)
        have = [r for r, v in enumerate(local) if v is not None]
        full = bool(self.limit) and len(self.centroids) >= self.limit
        if self.centroids and have:
            known = np.stack([unit(c) for c in self.centroids])
            similarity = np.stack([local[r] for r in have]) @ known.T
            for r, c in zip(*linear_sum_assignment(-similarity)):
                if similarity[r, c] >= MATCH_COSINE or full:
                    result[have[r]] = int(c)
        for r in have:
            if result[r] is not None:
                continue
            if full and self.centroids:
                known = np.stack([unit(c) for c in self.centroids])
                result[r] = int(np.argmax(known @ local[r]))
            else:
                result[r] = self._new(np.zeros_like(local[r]))
        for r in have:
            g, w = result[r], max(float(weights[r]), 1e-6)
            self.centroids[g] = (self.centroids[g] * self.weights[g] + local[r] * w) / (self.weights[g] + w)
            self.weights[g] += w
        return result

    def write(self, chunk_start: int, length: int, sources: dict[int, "np.ndarray"]) -> None:
        """Append ``length`` samples (a chunk core) to every track; speakers silent in it get zeros. A new track
        is filled with zeros up to ``chunk_start`` first."""
        import numpy as np

        for index, writer in enumerate(self.writers):
            gap = chunk_start - self.written[index]
            if gap > 0:
                writer.writeframes(bytes(2 * gap))
                self.written[index] += gap
            data = sources.get(index)
            if data is None:
                data = np.zeros(length, dtype=np.float32)
            pcm = (np.clip(data[:length], -1.0, 1.0) * 32767.0).astype("<i2")
            writer.writeframes(pcm.tobytes())
            self.written[index] += len(pcm)

    def close(self, total: int) -> None:
        for index, writer in enumerate(self.writers):
            if self.written[index] < total:
                writer.writeframes(bytes(2 * (total - self.written[index])))
            writer.close()


def _merge_turns(turns: list[tuple[float, float, str]], gap: float = 0.3) -> list[tuple[float, float, str]]:
    """Joins neighbouring turns of one speaker (chunk borders cut turns in two)."""
    out: list[tuple[float, float, str]] = []
    last: dict[str, int] = {}
    for start, end, speaker in sorted(turns):
        i = last.get(speaker)
        if i is not None and start - out[i][1] <= gap:
            out[i] = (out[i][0], max(out[i][1], end), speaker)
        else:
            last[speaker] = len(out)
            out.append((start, end, speaker))
    return sorted(out)


def _core_turns(diarization, run_start: int, core_start: int, core_end: int) -> dict[str, list[tuple[float, float]]]:
    """Activity of every label clipped to the chunk core, in seconds of the recording; labels with none left
    (active only in the context pad) are not in the result."""
    t0, lo, hi = run_start / SAMPLE_RATE, core_start / SAMPLE_RATE, core_end / SAMPLE_RATE
    out: dict[str, list[tuple[float, float]]] = {}
    for segment, _, label in diarization.itertracks(yield_label=True):
        start, end = max(lo, t0 + segment.start), min(hi, t0 + segment.end)
        if end - start > MIN_TURN_S:
            out.setdefault(label, []).append((round(start, 3), round(end, 3)))
    return out


def _overlapping_speaker(diarization, label: str, run_start: int, turns: list[tuple[float, float, str]]) -> int | None:
    """Global speaker that was active during most of this label's activity in the context pad before the core
    (the turns kept so far end where the core begins), or ``None``."""
    t0 = run_start / SAMPLE_RATE
    own = [(t0 + seg.start, t0 + seg.end) for seg, _, name in diarization.itertracks(yield_label=True) if name == label]
    overlap: dict[str, float] = {}
    for start, end in own:
        for other_start, other_end, speaker in turns:
            shared = min(end, other_end) - max(start, other_start)
            if shared > 0:
                overlap[speaker] = overlap.get(speaker, 0.0) + shared
    if not overlap:
        return None
    return int(max(overlap, key=overlap.get).split("_")[1])


def _embed(pipeline, track: "np.ndarray", spans: list[tuple[float, float]], run_start: int) -> "np.ndarray | None":
    """Embedding of one separated source from its own speech (at most ``EMBED_MAX_S``), in the space the
    pipeline clusters in. ``None`` when there is too little speech or the model refuses."""
    import numpy as np
    import torch

    pieces, taken = [], 0.0
    for start, end in spans:
        a = max(0, int(start * SAMPLE_RATE) - run_start)
        b = min(len(track), int(end * SAMPLE_RATE) - run_start)
        if b > a:
            pieces.append(track[a:b])
            taken += (b - a) / SAMPLE_RATE
        if taken >= EMBED_MAX_S:
            break
    if taken < EMBED_MIN_S:
        return None
    audio = np.concatenate(pieces)[: int(EMBED_MAX_S * SAMPLE_RATE)]
    try:
        with torch.inference_mode():
            vector = pipeline._embedding(torch.from_numpy(audio).float()[None, None].to(pipeline._embedding.device))
        vector = np.asarray(vector, dtype=np.float64).reshape(-1)
        return vector if np.isfinite(vector).all() and np.linalg.norm(vector) > 0 else None
    except Exception:  # an optional cue: without it the chunk falls back to the pipeline's own centroids
        return None


def _drop_ghosts(paths: list[Path], turns: list[tuple[float, float, str]]) -> tuple[list[Path], list[tuple[float, float, str]]]:
    """Removes the tracks (files and turns) of speakers with under ``MIN_SPEAKER_S`` of speech (at most 2 % of all
    speech) and renumbers the rest so that the labels stay SPEAKER_00, SPEAKER_01... without gaps."""
    speech: dict[str, float] = {}
    for start, end, speaker in turns:
        speech[speaker] = speech.get(speaker, 0.0) + end - start
    # relative for short clips: somebody who only says "Yes. No." in 8 s of speech is a speaker, not a ghost
    minimum = min(MIN_SPEAKER_S, 0.02 * sum(speech.values()))
    keep = [i for i in range(len(paths)) if speech.get(f"SPEAKER_{i:02d}", 0.0) >= minimum]
    new = {f"SPEAKER_{old:02d}": f"SPEAKER_{n:02d}" for n, old in enumerate(keep)}
    for i, path in enumerate(paths):
        if i not in keep:
            path.unlink(missing_ok=True)
    kept: list[Path] = []
    for n, old in enumerate(keep):  # ascending: the target name is never a file still to be moved
        target = paths[old].with_name(f"SPEAKER_{n:02d}.wav")
        if target != paths[old]:
            paths[old].replace(target)
        kept.append(target)
    return kept, [(s, e, new[sp]) for s, e, sp in turns if sp in new]


def separate(audio_path: Path, workdir: Path, settings: "Settings", progress: ProgressCallback = no_progress,
             num_speakers: int | None = None) -> SeparationResult:
    """Splits ``audio_path`` into one wav per speaker in ``workdir`` (``SPEAKER_00.wav``...) and returns the
    tracks and the activity of every speaker. Raises ``EngineError`` when the model is not available.
    ``num_speakers`` is an upper bound per chunk (somebody may be silent for ten minutes) and the global limit."""
    import numpy as np
    import torch

    workdir.mkdir(parents=True, exist_ok=True)
    progress(Progress(PHASE, 0, "Loading the speech separation model..."))
    pipeline = _load_pipeline(settings)
    tracks = _Tracks(workdir, num_speakers)
    total = 0
    turns: list[tuple[float, float, str]] = []
    try:
        audio = load_audio(audio_path)
        total = len(audio)
        chunk = int(CHUNK_S * SAMPLE_RATE)
        pad = int(PAD_S * SAMPLE_RATE)
        starts = list(range(0, total, chunk))  # empty audio: no chunks, no tracks
        # a short remainder is joined to the previous chunk instead of being run on its own
        if len(starts) > 1 and total - starts[-1] < chunk // 4:
            starts.pop()
        try:
            for n, core_start in enumerate(starts):
                core_end = starts[n + 1] if n + 1 < len(starts) else total
                length = core_end - core_start
                run_start, run_end = max(0, core_start - pad), min(total, core_end + pad)
                progress(Progress(PHASE, 100 * n / len(starts), f"Separating speakers ({n + 1}/{len(starts)})..."))
                wave_in = torch.from_numpy(np.ascontiguousarray(audio[run_start:run_end]))[None]
                with torch.inference_mode():
                    out = pipeline({"waveform": wave_in, "sample_rate": SAMPLE_RATE, "uri": audio_path.stem},
                                   max_speakers=num_speakers, return_embeddings=True)
                diarization, sources, centroids = out
                labels = diarization.labels() if sources is not None else []
                active = _core_turns(diarization, run_start, core_start, core_end) if labels else {}
                if not active:
                    tracks.write(core_start, length, {})
                    continue
                data = np.asarray(sources.data, dtype=np.float32)
                columns = {label: data[:, column] for column, label in enumerate(labels) if label in active}
                # the pipeline's centroids are indexed by cluster, not by the (filtered) labels: usable only
                # when nobody was dropped, otherwise the speakers are embedded from their own sources
                usable = centroids is not None and len(centroids) == len(labels)
                names = list(active)
                vectors = []
                for label in names:
                    v = _embed(pipeline, columns[label], active[label], run_start)
                    if v is None and usable:
                        v = np.asarray(centroids[labels.index(label)])
                    vectors.append(v)
                weights = [sum(e - s for s, e in active[label]) for label in names]
                mapping = dict(zip(names, tracks.match(vectors, weights)))
                for label in names:
                    if mapping[label] is None:  # no vector: the speaker it shared time with before the core
                        mapping[label] = _overlapping_speaker(diarization, label, run_start, turns)
                offset = core_start - run_start
                per_track: dict[int, "np.ndarray"] = {}
                for label, column in columns.items():
                    piece = column[offset:offset + length]
                    if len(piece) < length:
                        piece = np.pad(piece, (0, length - len(piece)))
                    g = mapping[label]
                    if g is None:  # nobody to attribute it to (first chunk, or no overlap): left out
                        continue
                    per_track[g] = per_track[g] + piece if g in per_track else piece
                tracks.write(core_start, length, per_track)
                for label, spans in active.items():
                    if mapping[label] is None:
                        continue
                    turns.extend((s, e, f"SPEAKER_{mapping[label]:02d}") for s, e in spans)
                del out, diarization, sources, data, columns, per_track
                free_gpu()
        finally:
            tracks.close(total)
    finally:
        del pipeline
        free_gpu()
    turns = _merge_turns(turns)
    paths, turns = _drop_ghosts(tracks.paths, turns)
    progress(Progress(PHASE, 100, f"Separated {len(paths)} speakers"))
    return SeparationResult(tracks=[SpeakerTrack(f"SPEAKER_{i:02d}", p) for i, p in enumerate(paths)], turns=turns)
