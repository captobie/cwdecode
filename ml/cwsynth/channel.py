"""Propagation and receiver impairments: noise at a defined SNR, fading, static, filtering.

SNR definition: the power of the (unfaded) carrier while the key is down, relative to the
noise power in a 500 Hz reference bandwidth, measured before any receiver filter. A carrier of
amplitude 1 has power 1/2. So the number doesn't depend on duty cycle, sample rate or which
filter the receiver uses, and it converts to other bandwidths by 10·log10(500 / B):
+7 dB lower in 2500 Hz, +4.7 dB higher in the app's ~170 Hz detector bandwidth.
"""
import numpy as np
from scipy import signal

from cwsynth.tone import smooth_noise

REFERENCE_BANDWIDTH_HZ = 500.0
CARRIER_POWER = 0.5


def snr_in_bandwidth(snr_db: float, bandwidth_hz: float) -> float:
    """The same signal and noise expressed as an SNR in another noise bandwidth."""
    return snr_db + 10 * np.log10(REFERENCE_BANDWIDTH_HZ / bandwidth_hz)


def noise_sigma(snr_db: float, sample_rate: int) -> float:
    """Standard deviation of white noise giving `snr_db` against an amplitude-1 carrier."""
    noise_in_band = CARRIER_POWER / 10 ** (snr_db / 10)
    # White noise spreads its variance evenly over 0 .. sample_rate / 2.
    return float(np.sqrt(noise_in_band * (sample_rate / 2) / REFERENCE_BANDWIDTH_HZ))


def white_noise(n: int, snr_db: float, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0.0, noise_sigma(snr_db, sample_rate), size=n)


def qsb_gain(n: int, sample_rate: int, rate_hz: float, depth_db: float,
             rng: np.random.Generator) -> np.ndarray:
    """Amplitude gain for slow fading: an irregular cycle averaging `rate_hz`, dipping as far
    as `depth_db` below full strength. The rate itself wanders between about half and double,
    so fades aren't a clean sinusoid."""
    wobble = np.exp(0.5 * smooth_noise(n, sample_rate, cutoff_hz=max(rate_hz, 0.05), rng=rng))
    phase = rng.uniform(0, 2 * np.pi) + 2 * np.pi * np.cumsum(rate_hz * wobble) / sample_rate
    fade = 0.5 - 0.5 * np.cos(phase)  # 0 = full strength, 1 = deepest
    return 10 ** (-depth_db * fade / 20)


def static_crashes(n: int, sample_rate: int, rate_per_s: float, level_db: float,
                   rng: np.random.Generator) -> np.ndarray:
    """QRN: Poisson-timed noise bursts with fast attack and exponential decay. `level_db` is a
    typical crash's peak power relative to the carrier's."""
    out = np.zeros(n, dtype=np.float64)
    count = rng.poisson(rate_per_s * n / sample_rate)
    for _ in range(int(count)):
        start = int(rng.integers(0, n))
        tau = rng.uniform(0.002, 0.03) * sample_rate
        length = min(int(6 * tau) + 1, n - start)
        amplitude = 10 ** ((level_db + rng.normal(0.0, 4.0)) / 20)
        decay = np.exp(-np.arange(length) / tau)
        out[start:start + length] += amplitude * decay * rng.standard_normal(length)
    return out


def receiver_filter(audio: np.ndarray, sample_rate: int, low_hz: float, high_hz: float,
                    order: int) -> np.ndarray:
    """A causal Butterworth band-pass, like a receiver's IF/audio filter. Narrow CW filters
    ring and delay the keying by a few ms, which is part of what real audio sounds like."""
    nyquist = sample_rate / 2
    low = max(low_hz, 20.0) / nyquist
    high = min(high_hz, nyquist * 0.95) / nyquist
    sos = signal.butter(order, [low, high], btype="bandpass", output="sos")
    return signal.sosfilt(sos, audio)
