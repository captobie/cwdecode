"""Envelope + frequency track → audio: the carrier, its drift, and keying chirp."""
from typing import Literal

import numpy as np

DriftShape = Literal["linear", "wander"]

# How fast a chirping transmitter settles back to its frequency after key-down.
CHIRP_SETTLE_S = 0.02
# Control rate for slowly varying random processes (drift, fading); interpolated up.
_CONTROL_RATE = 100.0


def smooth_noise(n: int, sample_rate: int, cutoff_hz: float, rng: np.random.Generator) -> np.ndarray:
    """Zero-mean, unit-variance Gaussian noise band-limited to `cutoff_hz`, `n` samples long.

    Generated at a low control rate over a buffer several periods long (so short clips still
    see a slow process rather than a stretched-out fast one), then interpolated up."""
    duration = n / sample_rate
    rate = max(_CONTROL_RATE, 8 * cutoff_hz)
    needed = int(np.ceil(duration * rate)) + 2
    length = max(needed, int(np.ceil(4 * rate / cutoff_hz)))
    spectrum = np.fft.rfft(rng.standard_normal(length))
    freqs = np.fft.rfftfreq(length, 1 / rate)
    spectrum *= np.exp(-0.5 * (freqs / cutoff_hz) ** 2)
    process = np.fft.irfft(spectrum, n=length)
    process = (process - process.mean()) / (process.std() + 1e-12)
    start = int(rng.integers(0, length - needed + 1))
    control = process[start:start + needed]
    return np.interp(np.arange(n) / sample_rate, np.arange(needed) / rate, control)


def frequency_track(n: int, sample_rate: int, tone_hz: float, drift_hz: float,
                    shape: DriftShape, rng: np.random.Generator) -> np.ndarray:
    """Instantaneous frequency in Hz. `linear` goes from `tone_hz` to `tone_hz + drift_hz`;
    `wander` moves slowly and irregularly over a span of `|drift_hz|` starting at `tone_hz`."""
    track = np.full(n, tone_hz, dtype=np.float64)
    if not drift_hz or n < 2:
        return track
    if shape == "linear":
        return track + drift_hz * np.linspace(0.0, 1.0, n)
    wander = smooth_noise(n, sample_rate, cutoff_hz=0.3, rng=rng)
    span = wander.max() - wander.min()
    wander = (wander - wander[0]) / (span + 1e-12)
    return track + abs(drift_hz) * wander


def chirp_track(marks: list[tuple[int, int]], n: int, sample_rate: int, chirp_hz: float) -> np.ndarray:
    """Frequency offset that jumps by `chirp_hz` at each key-down and settles exponentially,
    like a transmitter whose oscillator is pulled by keying."""
    offset = np.zeros(n, dtype=np.float64)
    if not chirp_hz:
        return offset
    tau = CHIRP_SETTLE_S * sample_rate
    for start, end in marks:
        offset[start:end] = chirp_hz * np.exp(-np.arange(end - start) / tau)
    return offset


def render_tone(envelope: np.ndarray, frequency: np.ndarray, sample_rate: int,
                rng: np.random.Generator) -> np.ndarray:
    """Amplitude-1 carrier following `frequency`, keyed by `envelope`."""
    phase = rng.uniform(0, 2 * np.pi) + 2 * np.pi * np.cumsum(frequency) / sample_rate
    return envelope.astype(np.float64) * np.sin(phase)
