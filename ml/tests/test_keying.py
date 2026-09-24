import numpy as np
import pytest
from scipy import signal

from cwsynth.alphabet import PATTERNS, tokenize
from cwsynth.keying import FIST_STYLES, FistParams, key_text, nominal_timing, render_envelope
from cwsynth.render import render
from cwsynth.spec import SampleSpec, sample_fist

TOKEN_FOR = {pattern: token for token, pattern in PATTERNS.items()}
TEXTS = ["CQ CQ DE K1ABC K", "5NN 14 TU", "VP2E/K1ABC <BT> QTH BOSTON <AR>",
         "EEEE TTTT IIII 5H5H", "<SK> <KN> ? . , / - @ '", "PARIS PARIS"]


def decode_elements(keying) -> str:
    """Decode with nothing but the clamp boundaries: if this reproduces the label, every
    label is determined by its timing."""
    dot, unit = keying.timing.dot, keying.timing.gap_unit
    out, symbols = [], ""
    for e in keying.elements:
        d = e.duration / e.tempo  # undo speed drift; the class bands are local
        if e.on:
            symbols += "." if d < 1.9 * dot else "-"
        elif d >= 1.8 * dot:
            out.append(TOKEN_FOR[symbols])
            symbols = ""
            if d > 5.0 * unit:
                out.append(" ")
    out.append(TOKEN_FOR[symbols])
    return "".join(out)


@pytest.mark.parametrize("style", FIST_STYLES)
def test_every_fist_keeps_labels_recoverable(style):
    rng = np.random.default_rng(1)
    for trial in range(300):
        text = TEXTS[trial % len(TEXTS)]
        fist = sample_fist(rng, style)
        # Exaggerate hesitation and jitter beyond the default ranges to lean on the clamps.
        fist = FistParams(**{**fist.__dict__, "hesitation": fist.hesitation * 3,
                             "gap_jitter": fist.gap_jitter * 1.5})
        wpm = float(rng.uniform(8, 40))
        farnsworth = wpm * 0.5 if trial % 5 == 0 else None
        keying = key_text(tokenize(text), wpm, fist, rng, farnsworth)
        assert decode_elements(keying) == "".join(tokenize(text))


def test_machine_timing_is_exact_paris():
    keying = key_text(tokenize("PARIS PARIS"), 20, FistParams(), np.random.default_rng(0))
    # PARIS is 50 dot units including one word gap; two words = 43 + 7 + 43 (no trailing gap).
    assert keying.duration == pytest.approx(93 * 1.2 / 20)


def test_farnsworth_keeps_character_speed():
    t = nominal_timing(20, farnsworth_wpm=10)
    assert t.dot == pytest.approx(0.06)
    assert t.gap_unit > t.dot
    # One PARIS word takes 60/10 seconds at 10 WPM overall.
    assert 31 * t.dot + 19 * t.gap_unit == pytest.approx(6.0)


def test_paddle_times_characters_exactly():
    fist = sample_fist(np.random.default_rng(3), "paddle")
    keying = key_text(tokenize("HHHH"), 25, fist, np.random.default_rng(4))
    marks = {round(e.duration, 9) for e in keying.elements if e.on}
    assert len(marks) == 1  # every dit identical


def test_audio_round_trip_and_alignment():
    spec = SampleSpec(text="CQ DE K1ABC", seed=5, wpm=18, tone_hz=650, lead_s=0.4, tail_s=0.3)
    sample = render(spec)
    sr = spec.sample_rate
    envelope = np.abs(signal.hilbert(sample.audio / sample.gain))
    envelope = np.convolve(envelope, np.ones(40) / 40, mode="same")
    keyed = envelope > 0.5
    edges = np.flatnonzero(np.diff(keyed.astype(int))) + 1
    starts, ends = edges[::2] / sr, edges[1::2] / sr
    dot = 1.2 / spec.wpm

    out, symbols = [], ""
    for i, (s, e) in enumerate(zip(starts, ends)):
        symbols += "." if e - s < 2 * dot else "-"
        gap = starts[i + 1] - e if i + 1 < len(starts) else np.inf
        if gap > 2 * dot:
            out.append(TOKEN_FOR[symbols])
            symbols = ""
            if gap > 5 * dot and np.isfinite(gap):
                out.append(" ")
    assert "".join(out) == sample.text

    assert sample.alignment[0][0] == "C"
    assert sample.alignment[0][1] == pytest.approx(spec.lead_s)
    assert starts[0] == pytest.approx(spec.lead_s, abs=0.005)
    assert sample.alignment[-1][2] == pytest.approx(ends[-1], abs=0.005)
    assert [a[0] for a in sample.alignment] == [t for t in tokenize(sample.text) if t != " "]


def test_rise_time_is_click_free():
    keying = key_text(tokenize("E"), 20, FistParams(), np.random.default_rng(0))
    env = render_envelope(keying, ["E"], 8000, 1000, 0.01, rise_ms=5).values
    # A raised-cosine edge over L samples peaks at slope 2/L; a hard key would be 1.
    assert np.max(np.abs(np.diff(env))) <= 2 / 40 * 1.05
