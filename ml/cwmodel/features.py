"""Audio → the model's input: a band-limited log-power spectrogram, normalized per clip.

Kept deliberately simple so the Swift side can reproduce it exactly with Accelerate:

- 8 kHz mono float audio
- 256-point periodic Hann window, hop 64 (8 ms), zero-padded by 128 samples at each end
  (torch.stft with center=True, pad_mode="constant"), so frame count = 1 + samples // 64
- power spectrum bins 8…40 (250–1250 Hz at 31.25 Hz per bin): 33 rows
- log10(power + 1e-10), floored at 60 dB below the clip's loudest bin, then minus the clip's
  mean and divided by its standard deviation

The 60 dB floor matters: without it, a clip's normalization depended on how silent its silence
was (digital zero sits 130 dB down, MP3 hiss 85 dB, real band noise 20–40 dB), which says nothing
about the Morse. Real signals never span more than ~40 dB, so the floor only flattens silence.
"""
import numpy as np
import torch

SAMPLE_RATE = 8000
N_FFT = 256
HOP = 64
BAND_HZ = (250.0, 1250.0)
_HZ_PER_BIN = SAMPLE_RATE / N_FFT
BIN_LO = int(np.ceil(BAND_HZ[0] / _HZ_PER_BIN))
BIN_HI = int(np.floor(BAND_HZ[1] / _HZ_PER_BIN)) + 1   # exclusive
N_BINS = BIN_HI - BIN_LO                               # 33
FRAME_S = HOP / SAMPLE_RATE
DYNAMIC_RANGE_DB = 60.0

_window = torch.hann_window(N_FFT)


def n_frames(n_samples: int) -> int:
    return 1 + n_samples // HOP


def spectrogram(audio: np.ndarray | torch.Tensor) -> torch.Tensor:
    """[samples] → [N_BINS, frames] float32."""
    x = torch.as_tensor(audio, dtype=torch.float32)
    stft = torch.stft(x, N_FFT, HOP, window=_window, center=True, pad_mode="constant",
                      return_complex=True)
    power = stft[BIN_LO:BIN_HI].abs() ** 2
    spec = torch.log10(power + 1e-10)
    spec = torch.maximum(spec, spec.max() - DYNAMIC_RANGE_DB / 10)
    return ((spec - spec.mean()) / (spec.std() + 1e-5)).float()
