import numpy as np
import pytest
import torch

from cwmodel.data import collate, crop_at_gaps, ctc_feasible, encode_item
from cwmodel.decode import greedy_decode, token_errors
from cwmodel.features import N_BINS, n_frames, spectrogram
from cwmodel.net import CwNet, output_lengths
from cwsynth import classic
from cwsynth.alphabet import TOKEN_TO_ID, tokenize
from cwsynth.render import render
from cwsynth.spec import SampleSpec


def test_feature_shape_and_normalization():
    audio = np.random.default_rng(0).normal(size=8000 * 3).astype(np.float32)
    spec = spectrogram(audio)
    assert spec.shape == (N_BINS, n_frames(len(audio))) == (33, 376)
    assert abs(float(spec.mean())) < 1e-4 and float(spec.std()) == pytest.approx(1.0, abs=1e-3)


def test_silence_floor_makes_sources_comparable():
    tone = np.sin(2 * np.pi * 750 * np.arange(8000) / 8000).astype(np.float32)
    tone[4000:] = 0.0
    hiss = tone + np.random.default_rng(0).normal(0, 1e-5, 8000).astype(np.float32)  # ~85 dB down
    assert torch.allclose(spectrogram(tone), spectrogram(hiss), atol=1e-3)


@pytest.mark.parametrize("frames", [1, 2, 375, 376, 1001])
def test_output_lengths_match_the_network(frames):
    model = CwNet().eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 1, N_BINS, frames))
    assert out.shape[0] == int(output_lengths(torch.tensor(frames)))


def test_crops_contain_exactly_their_label():
    spec = SampleSpec(text="CQ CQ DE K1ABC K1ABC <AR> K", seed=3, wpm=22, tone_hz=700)
    sample = render(spec)
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(60):
        audio, label = crop_at_gaps(sample.audio, sample.text, sample.alignment, rng)
        seen.add(label)
        # Characters come from the audio itself. (Spacing is checked against the source text:
        # with only a few characters, the threshold decoder can't estimate a word gap.)
        assert classic.decode(audio, 8000, tone_hz=700).text.replace(" ", "") == label.replace(" ", "")
        assert label in sample.text and not label.startswith(" ") and not label.endswith(" ")
    assert len(seen) > 10


def test_ctc_feasibility_counts_repeats():
    assert ctc_feasible(8000, tokenize("EE"))
    # 0.1 s gives 13 frames → 7 output frames: 7 distinct tokens fit; EEEEE needs 5 + 4 = 9.
    assert ctc_feasible(800, tokenize("ABCDEFG"))
    assert not ctc_feasible(800, tokenize("EEEEE"))


def test_greedy_decode_collapses_and_scores_tokens():
    ids = [0, TOKEN_TO_ID["C"], TOKEN_TO_ID["C"], 0, TOKEN_TO_ID["Q"], TOKEN_TO_ID[" "],
           TOKEN_TO_ID["<BT>"], 0, TOKEN_TO_ID["<BT>"]]
    log_probs = torch.full((len(ids), 1, 50), -10.0)
    log_probs[torch.arange(len(ids)), 0, torch.tensor(ids)] = 0.0
    assert greedy_decode(log_probs) == ["CQ <BT><BT>"]
    assert token_errors("CQ <BT>", "CQ <BT>") == (0, 4)
    assert token_errors("CQ <KN>", "CQ <BT>") == (1, 4)   # a prosign is one token


def test_collate_pads_and_keeps_lengths():
    items = [encode_item(np.zeros(8000, np.float32) + 0.01, "E", {}),
             encode_item(np.zeros(16000, np.float32) + 0.01, "TT", {})]
    batch = collate(items)
    assert batch["spec"].shape == (2, 1, N_BINS, 256)   # 251 frames, padded to a multiple of 128
    assert batch["frames"].tolist() == [n_frames(8000), n_frames(16000)]
    assert batch["target_lengths"].tolist() == [1, 2]
