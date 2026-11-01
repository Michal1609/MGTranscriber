"""Local loudness leveling: lifts quiet passages so that the VAD and the recognizers do not skip them.

A recording that is quiet in places (a distant speaker, a different microphone, a spliced-in clip 50 dB below
the rest) loses those places to the VAD, because its thresholds are absolute. A global peak normalization cannot
help when the quiet passage sits between loud ones, so the gain follows the signal: speech level per window
(a high percentile of 25 ms frame RMS), gain = distance to ``TARGET_DB`` and only upwards, interpolated between
windows. Windows without speech (silence, stationary noise: the frame levels hardly vary) get no gain of their
own and take the gain of the nearest speech, so noise is not pumped up on its own. The gain is limited
(``MAX_GAIN_DB``) and never lets a window clip. Audio that is already loud enough passes unchanged (``DEADBAND_DB``).
"""
from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000
FRAME = 400  # 25 ms
WINDOW_FRAMES = 80  # 2 s
HOP_FRAMES = 20  # 0.5 s
TARGET_DB = -23.0  # level of speech (90th percentile of the frame RMS) after leveling
DEADBAND_DB = 12.0  # a window at most this far below the target is left as it is
MAX_GAIN_DB = 48.0
FLOOR_DB = -80.0  # below this a window is silence whatever its modulation
MODULATION_DB = 6.0  # speech: 90th minus 30th percentile of the frame levels; noise stays below
PEAK_LIMIT = 0.97


def _db(x):
    return 20.0 * np.log10(np.maximum(x, 1e-9))


def level(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Returns ``audio`` (mono float32) with quiet speech lifted towards ``TARGET_DB``; the same array when
    nothing needs lifting."""
    n = len(audio) // FRAME
    if n < WINDOW_FRAMES // 4:
        return audio
    frames = audio[: n * FRAME].reshape(n, FRAME)
    rms_db = _db(np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1)))
    peak = np.abs(frames).max(axis=1)
    centers, gains = [], []
    for start in range(0, max(n - WINDOW_FRAMES, 0) + 1, HOP_FRAMES):
        sl = slice(start, start + WINDOW_FRAMES)
        high, low = np.percentile(rms_db[sl], [90, 30])
        if high < FLOOR_DB or high - low < MODULATION_DB:
            continue
        need = TARGET_DB - high
        gain = 0.0 if need < DEADBAND_DB else min(need, MAX_GAIN_DB)
        top = float(peak[sl].max())
        if top > 0:
            gain = min(gain, max(_db(PEAK_LIMIT / top), 0.0))
        centers.append((start + min(WINDOW_FRAMES, n - start) / 2) * FRAME)
        gains.append(gain)
    if not gains or max(gains) <= 0.0:
        return audio
    gain_db = np.interp(np.arange(len(audio)), centers, gains)
    out = audio * (10.0 ** (gain_db / 20.0)).astype(np.float32)
    return np.clip(out, -1.0, 1.0).astype(np.float32)
