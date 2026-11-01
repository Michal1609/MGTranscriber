"""Local loudness leveling (mgtranscriber/leveling.py) on synthetic signals."""
import numpy as np

from mgtranscriber.config import Settings, load_settings
from mgtranscriber.leveling import SAMPLE_RATE, level


def speechlike(seconds: float, amp: float, seed: int = 0) -> np.ndarray:
    """Noise with a syllable-rate envelope (4 Hz, with pauses) at peak ~``amp``."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    env = np.clip(np.sin(2 * np.pi * 4 * t) * 1.5, 0, 1)
    x = rng.standard_normal(n) * env
    return (x / np.abs(x).max() * amp).astype(np.float32)


def rms_db(x: np.ndarray) -> float:
    return 20 * np.log10(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12)


def test_quiet_passage_between_loud_ones_is_lifted():
    loud1, quiet, loud2 = speechlike(8, 0.7, 1), speechlike(8, 0.002, 2), speechlike(8, 0.7, 3)
    audio = np.concatenate([loud1, quiet, loud2])
    out = level(audio)
    n = len(loud1)
    # the quiet middle (without the transition seconds) is now near the loud ones
    mid = out[n + 3 * SAMPLE_RATE: 2 * n - 3 * SAMPLE_RATE]
    assert rms_db(mid) > rms_db(loud1) - 12
    assert rms_db(mid) > rms_db(quiet) + 30
    # loud parts are untouched and nothing clips
    assert np.allclose(out[: n - 2 * SAMPLE_RATE], audio[: n - 2 * SAMPLE_RATE])
    assert np.abs(out).max() <= 1.0
    assert out.dtype == np.float32 and len(out) == len(audio)


def test_loud_audio_passes_unchanged():
    audio = speechlike(10, 0.5)
    assert level(audio) is audio


def test_silence_and_stationary_noise_are_not_amplified():
    assert np.all(level(np.zeros(10 * SAMPLE_RATE, dtype=np.float32)) == 0)
    noise = (np.random.default_rng(0).standard_normal(10 * SAMPLE_RATE) * 0.001).astype(np.float32)
    assert level(noise) is noise


def test_gain_is_limited():
    audio = speechlike(10, 1e-5)  # 90 dB below full scale: lifted by at most MAX_GAIN_DB
    out = level(audio)
    assert np.abs(out).max() <= np.abs(audio).max() * 10 ** (48.5 / 20)


def test_short_audio_is_left_alone():
    audio = np.ones(100, dtype=np.float32) * 0.001
    assert level(audio) is audio


def test_setting_defaults_on_and_can_be_switched_off(tmp_path):
    assert Settings().level_audio is True
    off = load_settings(tmp_path, env={"MGTRANSCRIBER_LEVEL_AUDIO": "false"})
    assert off.level_audio is False


def test_load_audio_pair_keeps_the_original_for_diarization(monkeypatch):
    from mgtranscriber.engines import common

    source = np.concatenate([speechlike(8, 0.7, 1), speechlike(8, 0.002, 2)])
    monkeypatch.setattr(common, "load_audio", lambda path: source)
    audio, original = common.load_audio_pair("x.wav", True)
    assert original is source and len(audio) == len(source)
    assert rms_db(audio[-6 * SAMPLE_RATE:]) > rms_db(source[-6 * SAMPLE_RATE:]) + 30
    audio, original = common.load_audio_pair("x.wav", False)
    assert audio is original is source
