"""Speech separation without the model: speaker matching, track writing, chunking with a fake pipeline.
The real model is exercised by a manual run and by the gpu-marked test at the bottom."""

import dataclasses
import wave
from types import SimpleNamespace

import numpy as np
import pytest

from mgtranscriber import separation
from mgtranscriber.config import load_settings
from mgtranscriber.separation import SpeakerTrack, SeparationResult, _Tracks, _merge_turns, separate

SR = 16000


def _read(path):
    with wave.open(str(path)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, SR)
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")


def test_merge_turns_joins_chunk_borders_per_speaker():
    turns = [(0, 10, "SPEAKER_00"), (10.1, 12, "SPEAKER_00"), (5, 11, "SPEAKER_01"), (20, 21, "SPEAKER_00")]
    assert _merge_turns(turns) == [(0, 12, "SPEAKER_00"), (5, 11, "SPEAKER_01"), (20, 21, "SPEAKER_00")]


def test_match_keeps_speakers_across_chunks_and_adds_new(tmp_path):
    t = _Tracks(tmp_path, None)
    a, b, c = np.eye(3, 8)
    assert t.match([a, b], [1] * 2) == [0, 1]
    assert t.match([b + 0.05 * a, a], [1] * 2) == [1, 0]  # order inside a chunk is arbitrary
    assert t.match([c, a], [1] * 2) == [2, 0]  # an unseen voice is a new speaker
    t.close(0)


def test_match_respects_speaker_limit(tmp_path):
    t = _Tracks(tmp_path, 2)
    a, b, c = np.eye(3, 8)
    t.match([a, b], [1] * 2)
    got = t.match([a, c], [1] * 2)
    assert got[0] == 0 and got[1] in (0, 1) and len(t.paths) == 2
    t.close(0)


def test_late_speaker_track_is_padded_and_all_have_equal_length(tmp_path):
    t = _Tracks(tmp_path, None)
    t.match([np.eye(2, 4)[0]], [1])
    t.write(0, 100, {0: np.full(100, 0.5, dtype=np.float32)})
    t.match([np.eye(2, 4)[1]], [1])
    t.write(100, 100, {1: np.full(100, -0.5, dtype=np.float32)})
    t.close(250)
    first, second = (_read(p) for p in t.paths)
    assert len(first) == len(second) == 250
    assert first[0] > 0 and first[150] == 0 and second[50] == 0 and second[150] < 0


class FakePipeline:
    """Two speakers; speaker order flips in every second run like the real clustering may do."""

    def __init__(self):
        self.runs = 0

    def __call__(self, file, max_speakers=None, return_embeddings=False):
        from pyannote.core import Annotation, Segment

        n = file["waveform"].shape[1]
        self.runs += 1
        flip = self.runs % 2 == 0
        seconds = n / SR
        ann = Annotation()
        ann[Segment(0, seconds)] = "SPEAKER_00"
        ann[Segment(seconds / 2, seconds)] = "SPEAKER_01"
        data = np.zeros((n, 2), dtype=np.float32)
        data[:, 0] = 0.3
        data[n // 2:, 1] = -0.3
        centroids = np.eye(2, 8)
        if flip:  # the same voices under the other local labels
            ann = ann.rename_labels({"SPEAKER_00": "SPEAKER_01", "SPEAKER_01": "SPEAKER_00"})
            data = data[:, ::-1]
            centroids = centroids[::-1]
        return ann, SimpleNamespace(data=data), centroids


def test_separate_chunks_a_long_recording(tmp_path, monkeypatch):
    monkeypatch.setattr(separation, "CHUNK_S", 10.0)
    monkeypatch.setattr(separation, "PAD_S", 1.0)
    monkeypatch.setattr(separation, "_load_pipeline", lambda settings: FakePipeline())
    monkeypatch.setattr(separation, "load_audio", lambda path: np.zeros(35 * SR, dtype=np.float32))
    monkeypatch.setattr(separation, "free_gpu", lambda: None)
    seen = []
    result = separate(tmp_path / "in.wav", tmp_path / "out", load_settings(), seen.append)
    assert isinstance(result, SeparationResult)
    assert [t.speaker for t in result.tracks] == ["SPEAKER_00", "SPEAKER_01"]
    assert all(isinstance(t, SpeakerTrack) and t.path.is_file() for t in result.tracks)
    assert {len(_read(t.path)) for t in result.tracks} == {35 * SR}
    # 35 s -> chunks 0-10, 10-20, 20-35 (a remainder under a quarter chunk would be joined): speaker 0 talks all the time
    s0 = [t for t in result.turns if t[2] == "SPEAKER_00"]
    assert s0 == [(0.0, 35.0, "SPEAKER_00")]
    assert all(t[2] in ("SPEAKER_00", "SPEAKER_01") for t in result.turns)
    assert seen[-1].percent == 100


class PadGhostPipeline:
    """A label that speaks only in the context pad behind the core of the first chunk (10-12 s)."""

    def __init__(self):
        self.runs = 0

    def __call__(self, file, max_speakers=None, return_embeddings=False):
        from pyannote.core import Annotation, Segment

        n = file["waveform"].shape[1]
        self.runs += 1
        ann = Annotation()
        ann[Segment(0, n / SR)] = "SPEAKER_00"
        if self.runs == 1:
            ann[Segment(11, 12)] = "SPEAKER_01"
        return ann, SimpleNamespace(data=np.full((n, 2), 0.2, dtype=np.float32)), np.eye(2, 8)


def _patch(monkeypatch, pipeline, seconds):
    monkeypatch.setattr(separation, "_load_pipeline", lambda settings: pipeline)
    monkeypatch.setattr(separation, "load_audio", lambda path: np.zeros(int(seconds * SR), dtype=np.float32))
    monkeypatch.setattr(separation, "free_gpu", lambda: None)


def test_label_active_only_in_the_pad_makes_no_phantom_speaker(tmp_path, monkeypatch):
    monkeypatch.setattr(separation, "CHUNK_S", 10.0)
    monkeypatch.setattr(separation, "PAD_S", 2.0)
    _patch(monkeypatch, PadGhostPipeline(), 20)
    result = separate(tmp_path / "in.wav", tmp_path / "out", load_settings())
    assert [t.speaker for t in result.tracks] == ["SPEAKER_00"]
    assert {s for _, _, s in result.turns} == {"SPEAKER_00"}


def test_empty_audio_gives_no_tracks(tmp_path, monkeypatch):
    _patch(monkeypatch, FakePipeline(), 0)
    result = separate(tmp_path / "in.wav", tmp_path / "out", load_settings())
    assert result.tracks == [] and result.turns == []


def test_centroid_is_a_running_mean_weighted_by_speech(tmp_path):
    t = _Tracks(tmp_path, None)
    a = np.array([1.0, 0.0])
    b = np.array([0.8, 0.6])
    t.match([a], [3.0])
    t.match([b], [1.0])
    expected = (a * 3 + b) / 4
    assert np.allclose(t.centroids[0], expected) and t.weights[0] == 4.0
    t.close(0)


def test_label_without_vector_never_breaks_matching(tmp_path):
    t = _Tracks(tmp_path, None)
    v4 = np.eye(4)[0]
    assert t.match([None, v4], [0.3, 5]) == [None, 0]  # no vector: the caller decides
    assert t.match([v4], [5]) == [0]
    assert t.match([np.ones(7), v4], [1, 1]) == [None, 0]  # another dimension is not comparable either
    assert t.match([np.zeros(4), v4], [1, 1]) == [None, 0]  # a zero vector is no vector
    assert len(t.paths) == 1
    t.close(0)


def test_label_without_vector_joins_the_speaker_it_overlapped(tmp_path, monkeypatch):
    from pyannote.core import Annotation, Segment

    ann = Annotation()
    ann[Segment(0, 3)] = "A"  # in a run that starts at 5 s of the recording: active 5-8 s
    turns = [(4.0, 9.0, "SPEAKER_01"), (0.0, 6.0, "SPEAKER_00")]
    assert separation._overlapping_speaker(ann, "A", 5 * SR, turns) == 1  # 3 s against 1 s
    turns = [(4.0, 5.5, "SPEAKER_01"), (0.0, 20.0, "SPEAKER_00")]
    assert separation._overlapping_speaker(ann, "A", 5 * SR, turns) == 0
    assert separation._overlapping_speaker(ann, "A", 5 * SR, [(50.0, 60.0, "SPEAKER_00")]) is None


def test_ghost_speakers_get_no_track_and_labels_stay_contiguous(tmp_path):
    paths = []
    for i in range(3):
        p = tmp_path / f"SPEAKER_{i:02d}.wav"
        p.write_bytes(bytes([i]))
        paths.append(p)
    turns = [(0.0, 300.0, "SPEAKER_00"), (2.0, 3.0, "SPEAKER_01"), (5.0, 105.0, "SPEAKER_02")]
    kept, out = separation._drop_ghosts(paths, turns)
    assert [p.name for p in kept] == ["SPEAKER_00.wav", "SPEAKER_01.wav"]
    assert kept[1].read_bytes() == bytes([2])  # the third track became the second
    assert out == [(0.0, 300.0, "SPEAKER_00"), (5.0, 105.0, "SPEAKER_01")]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["SPEAKER_00.wav", "SPEAKER_01.wav"]


@pytest.fixture
def totatonet(monkeypatch):
    """pyannote's ToTaToNet module as it is when asteroid is missing: restored after the test."""
    import importlib

    module = importlib.import_module("pyannote.audio.models.separation.ToTaToNet")
    monkeypatch.setattr(module, "ASTEROID_IS_AVAILABLE", False)
    monkeypatch.setattr(module, "DPRNN", None, raising=False)
    monkeypatch.setattr(module, "pad_x_to_y", None, raising=False)
    return module


def test_vendored_dprnn_is_used_when_asteroid_is_missing(monkeypatch, totatonet):
    import sys

    monkeypatch.setitem(sys.modules, "asteroid", None)  # `import asteroid` raises ImportError
    before = {k for k in sys.modules if k == "asteroid" or k.startswith("asteroid.")}
    separation.install_asteroid_shim()
    from mgtranscriber._vendor import asteroid_min

    assert totatonet.ASTEROID_IS_AVAILABLE and totatonet.DPRNN is asteroid_min.DPRNN
    assert totatonet.pad_x_to_y is asteroid_min.pad_x_to_y
    assert {k for k in sys.modules if k == "asteroid" or k.startswith("asteroid.")} == before  # no stubs added


def test_broken_asteroid_falls_back_to_the_vendored_copy(monkeypatch, totatonet):
    import sys

    class Broken:
        def find_spec(self, name, path=None, target=None):
            if name == "asteroid" or name.startswith("asteroid."):
                raise RuntimeError("broken install")

    for name in [k for k in sys.modules if k == "asteroid" or k.startswith("asteroid.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, "meta_path", [Broken(), *sys.meta_path])
    separation.install_asteroid_shim()
    from mgtranscriber._vendor import asteroid_min

    assert totatonet.DPRNN is asteroid_min.DPRNN


def test_short_clip_keeps_a_speaker_with_few_words(tmp_path):
    paths = []
    for i in range(2):
        p = tmp_path / f"SPEAKER_{i:02d}.wav"
        p.write_bytes(b"x")
        paths.append(p)
    turns = [(0.0, 9.0, "SPEAKER_00"), (9.0, 10.0, "SPEAKER_01")]  # 1 s of 10 s: under MIN_SPEAKER_S, still a speaker
    kept, out = separation._drop_ghosts(paths, turns)
    assert len(kept) == 2 and len(out) == 2
    # ...but a 16 ms ghost in a long recording goes
    turns = [(0.0, 300.0, "SPEAKER_00"), (301.0, 301.016, "SPEAKER_01")]
    kept, out = separation._drop_ghosts(kept, turns)
    assert [p.name for p in kept] == ["SPEAKER_00.wav"] and out == [(0.0, 300.0, "SPEAKER_00")]


def test_vendored_dprnn_shapes():
    import torch

    from mgtranscriber._vendor.asteroid_min import DPRNN, pad_x_to_y

    net = DPRNN(in_chan=20, out_chan=16, n_src=3, n_repeats=2, bn_chan=8, hid_size=12, chunk_size=10).eval()
    assert net(torch.randn(2, 20, 41)).shape == (2, 3, 16, 41)
    assert pad_x_to_y(torch.ones(1, 5), torch.ones(1, 8)).shape == (1, 8)
    assert pad_x_to_y(torch.ones(1, 9), torch.ones(1, 8)).shape == (1, 8)


def test_is_available_without_token_or_deps(monkeypatch):
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *a, **k: None)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *a, **k: pytest.fail("must not ask HF without a token"))
    settings = dataclasses.replace(load_settings(), hf_token=None)
    monkeypatch.setattr(separation, "env_value", lambda s, n: None)
    ok, reason = separation.is_available(settings)
    assert not ok and "HF_TOKEN" in reason


@pytest.mark.gpu
def test_real_model_separates_two_voices(tmp_path):
    pytest.importorskip("torch")
    settings = load_settings()
    ok, reason = separation.is_available(settings)
    if not ok:
        pytest.skip(reason)
    from tests.make_clip import ensure_clip

    result = separate(ensure_clip(), tmp_path, settings)
    assert result.tracks and result.turns
