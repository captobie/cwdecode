"""SampleSpec → audio + label + per-character alignment."""
from dataclasses import dataclass

import numpy as np

from cwsynth import channel
from cwsynth.alphabet import tokenize
from cwsynth.keying import key_text, render_envelope
from cwsynth.spec import ConditionRanges, SampleSpec, sample_spec
from cwsynth.tone import chirp_track, frequency_track, render_tone

# Independent random streams per component, so changing one condition (say, SNR) leaves
# everything else about a sample identical.
_KEYING, _PHASE, _NOISE, _FADING, _STATIC, _DRIFT, _QRM = range(7)


@dataclass
class Sample:
    audio: np.ndarray                           # float32 mono, peak = spec.gain_peak
    text: str                                   # the CTC target; "" for noise-only
    alignment: list[tuple[str, float, float]]   # (token, start_s, end_s) of each keyed token
    spec: SampleSpec
    gain: float                                 # applied to reach gain_peak (for measuring SNR)
    source: str = ""                            # which corpus source the text came from

    @property
    def duration_s(self) -> float:
        return len(self.audio) / self.spec.sample_rate


def _stream(spec: SampleSpec, *key: int) -> np.random.Generator:
    return np.random.default_rng([spec.seed, *key])


def render(spec: SampleSpec) -> Sample:
    sr = spec.sample_rate
    tokens = tokenize(spec.text)

    if tokens:
        keying = key_text(tokens, spec.wpm, spec.fist, _stream(spec, _KEYING), spec.farnsworth_wpm)
        n = round((spec.lead_s + keying.duration + spec.tail_s) * sr)
        envelope = render_envelope(keying, tokens, sr, n, spec.lead_s, spec.rise_ms)
        env, marks, alignment = envelope.values, envelope.marks, envelope.alignment
    else:
        n = round((spec.lead_s + spec.tail_s) * sr)
        env, marks, alignment = np.zeros(n, dtype=np.float32), [], []

    frequency = frequency_track(n, sr, spec.tone_hz, spec.drift_hz, spec.drift_shape,
                                _stream(spec, _DRIFT))
    frequency += chirp_track(marks, n, sr, spec.chirp_hz)
    mix = render_tone(env, frequency, sr, _stream(spec, _PHASE))
    if spec.qsb is not None:
        mix *= channel.qsb_gain(n, sr, spec.qsb.rate_hz, spec.qsb.depth_db, _stream(spec, _FADING))

    for i, q in enumerate(spec.qrm):
        rng = _stream(spec, _QRM, i)
        q_tokens = tokenize(q.text)
        q_keying = key_text(q_tokens, q.wpm, q.fist, rng)
        q_env = render_envelope(q_keying, q_tokens, sr, n, q.start_s, q.rise_ms)
        # Elements before t=0 (a station already sending) are simply not in the buffer.
        q_tone = render_tone(q_env.values, np.full(n, q.tone_hz), sr, rng)
        mix += 10 ** (q.level_db / 20) * q_tone

    if spec.carrier_db is not None:
        mix += 10 ** (spec.carrier_db / 20) * render_tone(np.ones(n), frequency, sr, _stream(spec, _PHASE, 1))

    if spec.snr_db is not None:
        mix += channel.white_noise(n, spec.snr_db, sr, _stream(spec, _NOISE))
    if spec.qrn is not None:
        mix += channel.static_crashes(n, sr, spec.qrn.rate_per_s, spec.qrn.level_db, _stream(spec, _STATIC))
    if spec.rx_filter is not None:
        f = spec.rx_filter
        mix = channel.receiver_filter(mix, sr, f.low_hz, f.high_hz, f.order)

    peak = float(np.max(np.abs(mix))) if n else 0.0
    gain = spec.gain_peak / peak if peak > 0 else 1.0
    audio = (mix * gain).astype(np.float32)
    return Sample(audio=audio, text=spec.text, alignment=alignment, spec=spec, gain=gain)


def generate(seed: int, ranges: ConditionRanges | None = None, corpus=None,
             split: str = "train") -> Sample:
    """One random sample, fully determined by `seed`: text from `corpus`, conditions from
    `ranges`. This is also what an on-the-fly training dataset would call."""
    from cwsynth.corpus import CorpusMix

    ranges = ranges or ConditionRanges()
    corpus = corpus or CorpusMix()
    rng = np.random.default_rng(seed)
    source, text = corpus.sample(rng, split)
    spec = sample_spec(rng, text, ranges, qrm_text=lambda r: corpus.sample_text(r, split))
    sample = render(spec)
    sample.source = source
    return sample
