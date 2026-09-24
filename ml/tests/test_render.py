import json

import numpy as np
import pytest

from cwsynth.alphabet import normalize, tokenize
from cwsynth.channel import snr_in_bandwidth
from cwsynth.render import generate, render
from cwsynth.spec import ConditionRanges, QsbParams, SampleSpec, fit_text, sample_fist


def band_power(x: np.ndarray, sr: int, low: float, high: float) -> float:
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    band = (freqs >= low) & (freqs < high)
    return float(2 * np.sum(np.abs(spectrum[band]) ** 2) / len(x) ** 2)


@pytest.mark.parametrize("snr", [-10.0, 0.0, 12.0])
def test_snr_matches_definition(snr):
    base = SampleSpec(text="TEST " * 12, seed=11, wpm=20, tone_hz=700)
    clean = render(base)
    noisy = render(SampleSpec(**{**base.__dict__, "snr_db": snr}))
    signal_part = clean.audio.astype(np.float64) / clean.gain
    noise = noisy.audio.astype(np.float64) / noisy.gain - signal_part

    # Carrier power while the key is down (the plateau, where short-term RMS is ~0.707).
    rms = np.sqrt(np.convolve(signal_part ** 2, np.ones(40) / 40, "same"))
    carrier_power = np.mean(signal_part[rms > 0.69] ** 2)
    assert carrier_power == pytest.approx(0.5, rel=0.05)

    noise_power = band_power(noise, 8000, 700 - 250, 700 + 250)
    assert 10 * np.log10(carrier_power / noise_power) == pytest.approx(snr, abs=0.5)


def test_snr_bandwidth_conversion():
    assert snr_in_bandwidth(0.0, 2500) == pytest.approx(-6.99, abs=0.01)
    assert snr_in_bandwidth(0.0, 170) == pytest.approx(4.69, abs=0.01)


def test_render_is_deterministic_and_components_are_independent():
    spec = SampleSpec(text="CQ DE W1AW", seed=3, snr_db=0, qsb=QsbParams(0.3, 12),
                      fist=sample_fist(np.random.default_rng(0), "straight"))
    a, b = render(spec), render(spec)
    assert np.array_equal(a.audio, b.audio)
    # Changing the SNR leaves keying (and so alignment) untouched.
    c = render(SampleSpec(**{**spec.__dict__, "snr_db": 10}))
    assert c.alignment == a.alignment


def test_spec_round_trips_through_json():
    for seed in range(40):
        sample = generate(seed)
        restored = SampleSpec.from_dict(json.loads(json.dumps(sample.spec.to_dict())))
        assert restored == sample.spec
        assert np.array_equal(render(restored).audio, sample.audio)


def test_generated_samples_are_sane():
    ranges = ConditionRanges()
    for seed in range(300):
        s = generate(seed, ranges)
        assert s.text == normalize(s.text)
        assert np.all(np.isfinite(s.audio))
        assert np.max(np.abs(s.audio)) == pytest.approx(s.spec.gain_peak, rel=1e-3)
        if s.text:
            assert len(s.alignment) == sum(1 for t in tokenize(s.text) if t != " ")
            assert all(0 <= start < end <= s.duration_s + 1e-6 for _, start, end in s.alignment)
            assert all(q.level_db < 0 for q in s.spec.qrm)
        else:
            assert s.alignment == [] and s.spec.snr_db is not None


def test_fit_text_trims_whole_words():
    fist = sample_fist(np.random.default_rng(0), "machine")
    text = fit_text("CQ CQ CQ DE K1ABC K1ABC K1ABC K", 6.0, 20, fist)
    assert text and "CQ CQ CQ DE K1ABC K1ABC K1ABC K".startswith(text)
    assert not text.endswith(" ")
