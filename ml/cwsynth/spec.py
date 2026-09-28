"""What one sample is (`SampleSpec`) and the distributions samples are drawn from
(`ConditionRanges`).

A `SampleSpec` holds every condition as a concrete value, plus a seed for the randomness left
in rendering (per-element jitter, noise). `render(spec)` is deterministic, and specs round-trip
through `to_dict`/`from_dict`, so every manifest row can be re-rendered exactly and error rates
can be broken down by any condition.
"""
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from typing import Any

import numpy as np

from cwsynth.alphabet import SPACE, tokenize
from cwsynth.keying import FIST_STYLES, FistParams, FistStyle, estimate_duration
from cwsynth.tone import DriftShape


@dataclass(frozen=True)
class QsbParams:
    rate_hz: float
    depth_db: float


@dataclass(frozen=True)
class QrnParams:
    rate_per_s: float
    level_db: float     # typical crash peak relative to the carrier


@dataclass(frozen=True)
class RxFilterParams:
    low_hz: float
    high_hz: float
    order: int


@dataclass(frozen=True)
class QrmSignal:
    """A weaker interfering CW signal. The label never includes it: the model learns to copy
    the strongest signal, the one the operator has tuned in."""
    text: str
    wpm: float
    tone_hz: float
    level_db: float     # relative to the target signal, always negative
    start_s: float      # may be negative: already sending when the clip starts
    fist: FistParams
    rise_ms: float


@dataclass(frozen=True)
class SampleSpec:
    text: str                        # canonical label; "" for a noise-only sample
    seed: int
    sample_rate: int = 8000
    lead_s: float = 0.5              # silence before the first element
    tail_s: float = 0.5              # silence after the last (the whole clip if text is "")
    wpm: float = 20.0
    farnsworth_wpm: float | None = None
    fist: FistParams = field(default_factory=FistParams)
    rise_ms: float = 5.0
    tone_hz: float = 600.0
    drift_hz: float = 0.0
    drift_shape: DriftShape = "linear"
    chirp_hz: float = 0.0
    snr_db: float | None = None      # None = no noise
    qsb: QsbParams | None = None
    qrn: QrnParams | None = None
    qrm: tuple[QrmSignal, ...] = ()
    rx_filter: RxFilterParams | None = None
    carrier_db: float | None = None  # unkeyed carrier; only in noise-only samples
    gain_peak: float = 0.5           # final peak level of the clip

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SampleSpec":
        d = dict(d)
        d["fist"] = FistParams(**d["fist"])
        for key, kind in (("qsb", QsbParams), ("qrn", QrnParams), ("rx_filter", RxFilterParams)):
            if d.get(key) is not None:
                d[key] = kind(**d[key])
        d["qrm"] = tuple(
            QrmSignal(**{**q, "fist": FistParams(**q["fist"])}) for q in d.get("qrm", ())
        )
        return cls(**d)


@dataclass
class ConditionRanges:
    """Distributions for every condition. Ranges are (low, high), uniform unless noted."""
    sample_rate: int = 8000

    wpm: tuple[float, float] = (8.0, 40.0)                 # log-uniform
    farnsworth_prob: float = 0.10
    farnsworth_ratio: tuple[float, float] = (0.3, 0.85)    # overall / character speed; W1AW 5 WPM is 0.33
    fist_styles: dict[str, float] = field(default_factory=lambda: {
        "machine": 0.15, "paddle": 0.35, "bug": 0.15, "straight": 0.35})
    rise_ms: tuple[float, float] = (2.0, 8.0)
    max_signal_s: float = 15.0                             # text is trimmed at word breaks to fit
    lead_s: tuple[float, float] = (0.2, 1.5)
    tail_s: tuple[float, float] = (0.2, 1.5)

    tone_hz: tuple[float, float] = (400.0, 900.0)
    drift_prob: float = 0.30
    drift_hz: tuple[float, float] = (2.0, 25.0)            # total drift magnitude
    wander_share: float = 0.5                              # of drifting samples, share that wander
    chirp_prob: float = 0.05
    chirp_hz: tuple[float, float] = (5.0, 40.0)

    clean_prob: float = 0.05                               # no noise at all
    snr_db: tuple[float, float] = (-10.0, 25.0)            # in 500 Hz, see channel.py
    snr_skew: float = 1.5                                  # >1 weights toward the low end

    qsb_prob: float = 0.50
    qsb_rate_hz: tuple[float, float] = (0.05, 1.0)         # log-uniform
    qsb_depth_db: tuple[float, float] = (3.0, 25.0)

    qrn_prob: float = 0.30
    qrn_rate_per_s: tuple[float, float] = (0.5, 8.0)
    qrn_level_db: tuple[float, float] = (-6.0, 12.0)

    qrm_prob: float = 0.30
    qrm_max: int = 2
    qrm_level_db: tuple[float, float] = (-20.0, -3.0)
    qrm_offset_hz: tuple[float, float] = (80.0, 400.0)     # either side of the target

    rx_filter_prob: float = 0.50
    cw_filter_share: float = 0.70                          # the rest are SSB-width filters
    cw_filter_width_hz: tuple[float, float] = (150.0, 600.0)
    cw_filter_offset_hz: float = 50.0                      # target is not always centered

    empty_duration_s: tuple[float, float] = (1.0, 15.0)
    empty_carrier_prob: float = 0.30
    empty_carrier_db: tuple[float, float] = (-12.0, 0.0)

    gain_peak: tuple[float, float] = (0.25, 0.9)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ConditionRanges":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown condition settings: {sorted(unknown)}")
        converted = {k: tuple(v) if isinstance(v, list) else v for k, v in d.items()}
        return cls(**converted)


def _uniform(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    return float(rng.uniform(*bounds))


def _log_uniform(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    return float(np.exp(rng.uniform(np.log(bounds[0]), np.log(bounds[1]))))


def sample_fist(rng: np.random.Generator, style: FistStyle) -> FistParams:
    """One operator's fist: constants for this sample (their bias), plus the jitter scale
    that per-element randomness is drawn with at render time."""
    u = lambda lo, hi: float(rng.uniform(lo, hi))  # noqa: E731
    if style == "machine":
        return FistParams("machine", weight=u(-0.05, 0.10))
    human_gaps = dict(
        gap_jitter=u(0.05, 0.25), char_gap_scale=u(0.75, 1.4), word_gap_scale=u(0.75, 1.6),
        hesitation=u(0.0, 0.6),
    )
    if style == "paddle":
        return FistParams("paddle", weight=u(-0.10, 0.15), dah_ratio=u(2.8, 3.4), **human_gaps)
    if style == "bug":
        return FistParams("bug", element_jitter=u(0.05, 0.20), weight=u(-0.15, 0.15),
                          dah_ratio=u(2.8, 4.5), speed_drift=u(-0.10, 0.10), **human_gaps)
    return FistParams("straight", element_jitter=u(0.05, 0.20), weight=u(-0.15, 0.25),
                      dah_ratio=u(2.4, 3.8), speed_drift=u(-0.15, 0.15),
                      **{**human_gaps, "gap_jitter": u(0.08, 0.30), "hesitation": u(0.0, 0.8)})


def fit_text(text: str, max_s: float, wpm: float, fist: FistParams,
             farnsworth_wpm: float | None = None) -> str:
    """Drop words from the end (or characters, for a single long word) until the text's
    nominal keyed duration is at most `max_s`. Never returns a partial character."""
    tokens = tokenize(text)
    while tokens and estimate_duration(tokens, wpm, fist, farnsworth_wpm) > max_s:
        if SPACE in tokens:
            tokens = tokens[:len(tokens) - 1 - tokens[::-1].index(SPACE)]
        else:
            tokens = tokens[:-1]
    return "".join(tokens)


def _choose_style(rng: np.random.Generator, weights: dict[str, float]) -> FistStyle:
    styles = [s for s in weights if s in FIST_STYLES]
    if len(styles) != len(weights):
        raise ValueError(f"fist styles must be among {FIST_STYLES}")
    p = np.array([weights[s] for s in styles], dtype=float)
    return styles[int(rng.choice(len(styles), p=p / p.sum()))]  # type: ignore[return-value]


def sample_spec(rng: np.random.Generator, text: str, ranges: ConditionRanges,
                qrm_text: Callable[[np.random.Generator], str] | None = None) -> SampleSpec:
    """Draw every condition for one sample. `text` is trimmed to fit `max_signal_s` at the
    drawn speed; an empty `text` makes a noise-only sample. `qrm_text` supplies what
    interfering stations send (no QRM if None)."""
    r = ranges
    wpm = _log_uniform(rng, r.wpm)
    farnsworth = wpm * _uniform(rng, r.farnsworth_ratio) if rng.random() < r.farnsworth_prob else None
    fist = sample_fist(rng, _choose_style(rng, r.fist_styles))
    text = fit_text(text, r.max_signal_s, wpm, fist, farnsworth) if text else ""
    empty = not text

    tone = _uniform(rng, r.tone_hz)
    drift, shape = 0.0, "linear"
    if rng.random() < r.drift_prob:
        drift = _uniform(rng, r.drift_hz) * (1 if rng.random() < 0.5 else -1)
        shape = "wander" if rng.random() < r.wander_share else "linear"
    chirp = _uniform(rng, r.chirp_hz) * (1 if rng.random() < 0.5 else -1) \
        if not empty and rng.random() < r.chirp_prob else 0.0

    # A noise-only sample always has noise; otherwise there'd be nothing to learn from it.
    snr = None
    if empty or rng.random() >= r.clean_prob:
        snr = r.snr_db[0] + (r.snr_db[1] - r.snr_db[0]) * float(rng.random()) ** r.snr_skew

    qsb = QsbParams(_log_uniform(rng, r.qsb_rate_hz), _uniform(rng, r.qsb_depth_db)) \
        if not empty and rng.random() < r.qsb_prob else None
    qrn = QrnParams(_uniform(rng, r.qrn_rate_per_s), _uniform(rng, r.qrn_level_db)) \
        if rng.random() < r.qrn_prob else None

    if empty:
        lead, tail = 0.0, _uniform(rng, r.empty_duration_s)
    else:
        lead, tail = _uniform(rng, r.lead_s), _uniform(rng, r.tail_s)

    qrm: list[QrmSignal] = []
    if qrm_text is not None and not empty and rng.random() < r.qrm_prob:
        clip_s = lead + tail + r.max_signal_s
        for _ in range(int(rng.integers(1, r.qrm_max + 1))):
            q_wpm = _log_uniform(rng, r.wpm)
            q_fist = sample_fist(rng, _choose_style(rng, r.fist_styles))
            offset = _uniform(rng, r.qrm_offset_hz) * (1 if rng.random() < 0.5 else -1)
            q_text = fit_text(qrm_text(rng), clip_s, q_wpm, q_fist)
            if not q_text:
                continue
            qrm.append(QrmSignal(
                text=q_text, wpm=q_wpm, tone_hz=float(np.clip(tone + offset, 200.0, 1500.0)),
                level_db=_uniform(rng, r.qrm_level_db), start_s=float(rng.uniform(-3.0, clip_s * 0.7)),
                fist=q_fist, rise_ms=_uniform(rng, r.rise_ms)))

    rx_filter = None
    if rng.random() < r.rx_filter_prob:
        nyquist = r.sample_rate / 2
        if rng.random() < r.cw_filter_share:
            width = _uniform(rng, r.cw_filter_width_hz)
            center = tone + _uniform(rng, (-r.cw_filter_offset_hz, r.cw_filter_offset_hz))
            low, high = max(center - width / 2, 50.0), center + width / 2
            order = int(rng.integers(2, 5))
        else:
            low, high = _uniform(rng, (200.0, 400.0)), min(_uniform(rng, (2400.0, 3000.0)), nyquist * 0.9)
            order = int(rng.integers(2, 4))
        rx_filter = RxFilterParams(low, high, order)

    carrier = _uniform(rng, r.empty_carrier_db) if empty and rng.random() < r.empty_carrier_prob else None

    return SampleSpec(
        text=text, seed=int(rng.integers(0, 2**63 - 1)), sample_rate=r.sample_rate,
        lead_s=lead, tail_s=tail, wpm=wpm, farnsworth_wpm=farnsworth, fist=fist,
        rise_ms=_uniform(rng, r.rise_ms), tone_hz=tone, drift_hz=drift, drift_shape=shape,
        chirp_hz=chirp, snr_db=snr, qsb=qsb, qrn=qrn, qrm=tuple(qrm), rx_filter=rx_filter,
        carrier_db=carrier, gain_peak=_uniform(rng, r.gain_peak),
    )
