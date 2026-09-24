"""Tokens → key-down/key-up durations with a human "fist" → smooth on/off envelope.

Human timing is modeled per keying style:

- machine:  computer-sent; every element exact (only transmitter weighting varies).
- paddle:   an electronic keyer times dits, dahs and the gaps inside a character exactly;
            the operator times the gaps between characters and words.
- bug:      a semi-automatic key makes dits (and the gaps after them) automatically; dahs
            are held by hand, often long ("swing").
- straight: every element and gap is timed by hand.

After jitter, every duration is clamped to a band around its nominal length so the three gap
classes never overlap: a gap inside a character stays under 1.6 dots, a gap between
characters stays between ~2 and 4.25 gap units, and a gap between words stays above 5.75 gap
units. That keeps every label consistent with its audio: a space in the label is always a gap
a careful listener would hear as a word break.
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np

from cwsynth.alphabet import PATTERNS, SPACE

FistStyle = Literal["machine", "paddle", "bug", "straight"]
FIST_STYLES: tuple[FistStyle, ...] = ("machine", "paddle", "bug", "straight")

ElementKind = Literal["dit", "dah", "element_gap", "char_gap", "word_gap"]

# Clamp bands. Marks and the gap inside a character are in dots; gaps between characters
# and words are in gap units (a dot, or longer with Farnsworth spacing).
DIT_RANGE = (0.5, 1.6)
DAH_RANGE = (2.2, 5.0)
ELEMENT_GAP_RANGE = (0.4, 1.6)
CHAR_GAP_MIN_DOTS = 2.0
CHAR_GAP_RANGE = (1.8, 4.25)
WORD_GAP_RANGE = (5.75, 20.0)


@dataclass(frozen=True)
class FistParams:
    style: FistStyle = "machine"
    element_jitter: float = 0.0   # std dev of the relative error on hand-timed marks
    gap_jitter: float = 0.0       # std dev of the relative error on hand-timed gaps
    weight: float = 0.0           # dots added to every mark and taken from every gap
    dah_ratio: float = 3.0        # dah length in dots
    char_gap_scale: float = 1.0   # this operator's letter spacing relative to nominal
    word_gap_scale: float = 1.0   # this operator's word spacing relative to nominal
    hesitation: float = 0.0       # max extra fraction added to a gap between characters or words
    speed_drift: float = 0.0      # relative tempo change from the start to the end of the text


@dataclass(frozen=True)
class Timing:
    dot: float        # seconds
    gap_unit: float   # seconds; equals `dot` unless Farnsworth spacing stretches the gaps

    @property
    def char_gap(self) -> float:
        return 3 * self.gap_unit

    @property
    def word_gap(self) -> float:
        return 7 * self.gap_unit


def nominal_timing(wpm: float, farnsworth_wpm: float | None = None) -> Timing:
    """PARIS timing; with `farnsworth_wpm` below `wpm`, characters are sent at `wpm` and the
    gaps between them are stretched to an overall speed of `farnsworth_wpm` (ARRL formula)."""
    if wpm <= 0:
        raise ValueError("wpm must be positive")
    dot = 1.2 / wpm
    if farnsworth_wpm is None or farnsworth_wpm >= wpm:
        return Timing(dot=dot, gap_unit=dot)
    if farnsworth_wpm <= 0:
        raise ValueError("farnsworth_wpm must be positive")
    total_gap = (60 * wpm - 37.2 * farnsworth_wpm) / (farnsworth_wpm * wpm)
    return Timing(dot=dot, gap_unit=total_gap / 19)


@dataclass(frozen=True)
class Element:
    on: bool
    kind: ElementKind
    duration: float    # seconds
    token_index: int   # index into the token list; gaps belong to the token before them
    tempo: float       # the speed-drift factor applied to this element


@dataclass(frozen=True)
class Keying:
    elements: list[Element]
    timing: Timing

    @property
    def duration(self) -> float:
        return sum(e.duration for e in self.elements)


def ideal_elements(tokens: list[str]) -> list[tuple[ElementKind, int]]:
    """The element sequence for `tokens`, without durations."""
    out: list[tuple[ElementKind, int]] = []
    for i, token in enumerate(tokens):
        if token == SPACE:
            # A space replaces the letter gap that would otherwise follow the previous token.
            if out and out[-1][0] == "char_gap":
                out.pop()
            out.append(("word_gap", i))
            continue
        pattern = PATTERNS[token]
        for j, symbol in enumerate(pattern):
            out.append(("dit" if symbol == "." else "dah", i))
            if j < len(pattern) - 1:
                out.append(("element_gap", i))
        if i + 1 < len(tokens) and tokens[i + 1] != SPACE:
            out.append(("char_gap", i))
    return out


def _hand_timed(style: FistStyle, kind: ElementKind) -> float:
    """How much of the fist's jitter applies to this kind of element (0 = exact)."""
    if style == "machine":
        return 0.0
    if kind in ("char_gap", "word_gap"):
        return 1.0
    if style == "straight":
        return 1.0
    if style == "bug":
        return {"dah": 1.0, "element_gap": 0.5}.get(kind, 0.0)
    return 0.0  # paddle: the keyer times everything inside a character


def key_text(tokens: list[str], wpm: float, fist: FistParams, rng: np.random.Generator,
             farnsworth_wpm: float | None = None) -> Keying:
    timing = nominal_timing(wpm, farnsworth_wpm)
    dot, unit = timing.dot, timing.gap_unit
    ideal = ideal_elements(tokens)

    durations: list[float] = []
    for kind, _ in ideal:
        if kind == "dit":
            base, sigma = dot, fist.element_jitter
        elif kind == "dah":
            base, sigma = fist.dah_ratio * dot, fist.element_jitter
        elif kind == "element_gap":
            base, sigma = dot, fist.gap_jitter
        elif kind == "char_gap":
            base, sigma = timing.char_gap * fist.char_gap_scale, fist.gap_jitter
        else:
            base, sigma = timing.word_gap * fist.word_gap_scale, fist.gap_jitter

        share = _hand_timed(fist.style, kind)
        factor = 1.0 + (rng.normal(0.0, sigma * share) if sigma and share else 0.0)
        duration = base * factor
        if kind in ("char_gap", "word_gap") and fist.hesitation:
            # Mostly small, occasionally a real pause.
            duration *= 1.0 + fist.hesitation * rng.random() ** 2
        duration += fist.weight * dot if kind in ("dit", "dah") else -fist.weight * dot

        if kind == "dit":
            lo, hi = DIT_RANGE[0] * dot, DIT_RANGE[1] * dot
        elif kind == "dah":
            lo, hi = DAH_RANGE[0] * dot, DAH_RANGE[1] * dot
        elif kind == "element_gap":
            lo, hi = ELEMENT_GAP_RANGE[0] * dot, ELEMENT_GAP_RANGE[1] * dot
        elif kind == "char_gap":
            lo = max(CHAR_GAP_MIN_DOTS * dot, CHAR_GAP_RANGE[0] * unit)
            hi = CHAR_GAP_RANGE[1] * unit
        else:
            lo, hi = WORD_GAP_RANGE[0] * unit, WORD_GAP_RANGE[1] * unit
        durations.append(float(np.clip(duration, lo, hi)))

    # Speed drift: a smooth tempo ramp over the text. Applied after clamping, so the
    # local ratios between element classes are unchanged.
    total = sum(durations)
    elements: list[Element] = []
    elapsed = 0.0
    for (kind, index), duration in zip(ideal, durations):
        position = (elapsed + duration / 2) / total if total else 0.0
        tempo = 1.0 / (1.0 + fist.speed_drift * (position - 0.5))
        elements.append(Element(kind in ("dit", "dah"), kind, duration * tempo, index, tempo))
        elapsed += duration
    return Keying(elements, timing)


def estimate_duration(tokens: list[str], wpm: float, fist: FistParams,
                      farnsworth_wpm: float | None = None) -> float:
    """Keyed duration without jitter; used to fit text to a maximum length."""
    timing = nominal_timing(wpm, farnsworth_wpm)
    lengths = {
        "dit": timing.dot, "dah": fist.dah_ratio * timing.dot, "element_gap": timing.dot,
        "char_gap": timing.char_gap * fist.char_gap_scale,
        "word_gap": timing.word_gap * fist.word_gap_scale,
    }
    return sum(lengths[kind] for kind, _ in ideal_elements(tokens))


@dataclass(frozen=True)
class Envelope:
    values: np.ndarray                          # float32 in [0, 1]
    marks: list[tuple[int, int]]                # key-down [start, end) in samples
    alignment: list[tuple[str, float, float]]   # (token, start_s, end_s) for keyed tokens


def render_envelope(keying: Keying, tokens: list[str], sample_rate: int, n_samples: int,
                    offset_s: float, rise_ms: float) -> Envelope:
    """Key-down intervals placed `offset_s` into an `n_samples` buffer, with edges shaped by
    a raised cosine that goes from 0 to 1 in `rise_ms`. Edges are centered on the nominal
    transitions, so a mark's length at half amplitude is its keyed duration."""
    keyed = np.zeros(n_samples, dtype=np.float32)
    marks: list[tuple[int, int]] = []
    spans: dict[int, list[float]] = {}
    t = offset_s
    for element in keying.elements:
        start_t, t = t, t + element.duration
        if not element.on:
            continue
        # Round cumulative times, not per-element lengths, so rounding never accumulates.
        start = min(round(start_t * sample_rate), n_samples)
        end = min(round(t * sample_rate), n_samples)
        if end > start:
            keyed[start:end] = 1.0
            marks.append((start, end))
        span = spans.setdefault(element.token_index, [start_t, t])
        span[1] = t

    ramp = max(1, round(rise_ms / 1000 * sample_rate))
    if ramp > 1:
        kernel = np.hanning(ramp + 2)[1:-1]
        values = np.convolve(keyed, kernel / kernel.sum(), mode="same").astype(np.float32)
    else:
        values = keyed

    alignment = [(tokens[i], start, end) for i, (start, end) in sorted(spans.items())]
    return Envelope(np.clip(values, 0.0, 1.0), marks, alignment)
