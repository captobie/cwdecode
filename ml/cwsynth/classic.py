"""A plain threshold decoder for clean, machine-sent CW, used to label real recordings.

It is not meant for noisy or hand-sent signals. On clean audio such as W1AW's code practice it
is exact, and every character comes with its time span, which is what segmentation needs.
Speed is estimated from the recording: the dit length from the marks, and the gap unit
separately from the gaps, so Farnsworth spacing works without being told.
"""
from dataclasses import dataclass

import numpy as np
from scipy import signal

from cwsynth.alphabet import PATTERNS, SPACE

TOKEN_FOR = {pattern: token for token, pattern in PATTERNS.items()}


@dataclass(frozen=True)
class Decoded:
    tokens: list[tuple[str, float, float]]   # (token, start_s, end_s); spaces span the word gap
    tone_hz: float
    dot_s: float
    gap_unit_s: float
    carrier_amplitude: float                 # tone amplitude while the key is down

    @property
    def text(self) -> str:
        return "".join(t for t, _, _ in self.tokens)


def find_tone(audio: np.ndarray, sample_rate: int, low: float = 300.0, high: float = 1200.0) -> float:
    freqs, power = signal.welch(audio, sample_rate, nperseg=min(len(audio), 8192))
    band = (freqs >= low) & (freqs <= high)
    return float(freqs[band][np.argmax(power[band])])


def envelope(audio: np.ndarray, sample_rate: int, tone_hz: float, width_hz: float = 120.0) -> np.ndarray:
    """Amplitude of the tone, zero-phase filtered so edges stay where they are."""
    sos = signal.butter(4, [tone_hz - width_hz / 2, tone_hz + width_hz / 2], "bandpass",
                        fs=sample_rate, output="sos")
    env = np.abs(signal.hilbert(signal.sosfiltfilt(sos, audio)))
    smooth = max(1, int(0.002 * sample_rate))
    return np.convolve(env, np.ones(smooth) / smooth, mode="same")


def unknown_token(pattern: str) -> str:
    """A pattern that isn't in the vocabulary. Never equal to any transcript token."""
    return f"[{pattern}]"


def decode(audio: np.ndarray, sample_rate: int, tone_hz: float | None = None) -> Decoded:
    tone = tone_hz or find_tone(audio, sample_rate)
    env = envelope(audio, sample_rate, tone)
    floor, peak = np.percentile(env, 10), np.percentile(env, 99)
    keyed = env > floor + 0.5 * (peak - floor)

    edges = np.flatnonzero(np.diff(keyed.astype(np.int8))) + 1
    if keyed[0]:
        edges = edges[1:]
    starts, ends = edges[::2], edges[1::2]
    starts = starts[:len(ends)]
    # Drop glitches shorter than 5 ms (clicks, MP3 artifacts).
    keep = (ends - starts) >= 0.005 * sample_rate
    starts, ends = starts[keep] / sample_rate, ends[keep] / sample_rate
    if len(starts) == 0:
        return Decoded([], tone, 0.0, 0.0, float(peak))

    marks = ends - starts
    lo, hi = np.percentile(marks, 10), np.percentile(marks, 90)
    dits = marks[marks < (lo + hi) / 2] if hi > 1.8 * lo else marks
    dot = float(np.median(dits))
    gaps = starts[1:] - ends[:-1]
    long_gaps = gaps[gaps > 2 * dot]
    # Letter gaps outnumber word gaps several to one, so the median long gap is a letter gap.
    unit = float(np.median(long_gaps)) / 3 if len(long_gaps) else dot

    tokens: list[tuple[str, float, float]] = []
    symbols, char_start = "", starts[0]
    for i, (start, end) in enumerate(zip(starts, ends)):
        if not symbols:
            char_start = start
        symbols += "." if end - start < 2 * dot else "-"
        gap = gaps[i] if i < len(gaps) else np.inf
        if gap > 2 * dot:
            tokens.append((TOKEN_FOR.get(symbols, unknown_token(symbols)), float(char_start), float(end)))
            symbols = ""
            if gap > 5 * unit and np.isfinite(gap):
                tokens.append((SPACE, float(end), float(starts[i + 1])))

    on = env[keyed]
    return Decoded(tokens, tone, dot, unit, float(np.median(on)) if len(on) else float(peak))
