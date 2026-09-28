"""W1AW library logic, tested on synthetic sessions (no ARRL files needed)."""
import numpy as np
import pytest

from cwsynth import classic
from cwsynth.alphabet import SPACE, tokenize
from cwsynth.render import render
from cwsynth.spec import SampleSpec
from cwsynth.w1aw import align, cut_clips, parse_archive_page, transcript_tokens

SESSION = ("= NOW 15 WPM = TEXT IS FROM MAY 2024 QST PAGE 12 = THE STATION WAS ON THE AIR "
           "FOR SIX HOURS AND MADE 250 CONTACTS WITH JW0X AND OTHERS. = END OF 15 WPM TEXT = "
           "QST DE W1AW")


def machine_session(text: str, wpm: float, farnsworth: float | None = None):
    spec = SampleSpec(text=text, seed=1, wpm=wpm, farnsworth_wpm=farnsworth, tone_hz=750,
                      lead_s=0.3, tail_s=0.3, gain_peak=0.9)
    sample = render(spec)
    return sample.audio, spec.sample_rate


def test_transcript_bytes_follow_what_is_keyed():
    raw = b"\r\n\x89 NOW 5 WPM \x89\r\nJW0\x98X JW\xc3\x98X A&B 56=\r\nQST DE W1AW \x83 <\r\n\x1a"
    assert "".join(transcript_tokens(raw)) == "<BT> NOW 5 WPM <BT> JW0X JW0X A<AS>B 56<BT> QST DE W1AW <AS>"
    assert transcript_tokens(b"PRICE: $5")[5] == "{:}"   # unsupported: never matches the audio


@pytest.mark.parametrize("wpm,farnsworth", [(15, 7.5), (20, None), (40, None), (15, 5)])
def test_classic_decoder_is_exact_on_machine_code(wpm, farnsworth):
    char_wpm = wpm
    audio, sr = machine_session(SESSION, char_wpm, farnsworth)
    decoded = classic.decode(audio, sr)
    assert decoded.text == "".join(tokenize(SESSION))
    assert decoded.tone_hz == pytest.approx(750, abs=3)
    assert 1.2 / decoded.dot_s == pytest.approx(char_wpm, rel=0.03)


def test_clips_are_verified_and_skip_disagreements():
    audio, sr = machine_session(SESSION, 20)
    decoded = classic.decode(audio, sr)
    # The transcript disagrees with the audio in one word, as in the 2010 recordings.
    reference = tokenize(SESSION.replace("STATION", "STATTON"))
    mapping = align([t for t, _, _ in decoded.tokens], reference)
    clips = cut_clips(decoded.tokens, reference, mapping, len(audio) / sr,
                      np.random.default_rng(0), min_s=1.0, max_s=6.0)
    assert clips
    texts = ["".join(reference[i0:i1 + 1]) for i0, i1, _, _ in clips]
    assert not any("STATTON" in t for t in texts)
    for (i0, i1, s, e), text in zip(clips, texts):
        assert not text.startswith(" ") and not text.endswith(" ")
        # Each clip, decoded on its own, reads exactly its label.
        piece = classic.decode(audio[round(s * sr):round(e * sr)], sr, tone_hz=750)
        assert piece.text == text
        assert e - s <= 6.0 + 1.0


def test_archive_page_pairs_by_date_across_link_styles():
    html = ('<a href="http://www.arrl.org/files/file/Morse/Archive/25%20WPM/260106_25WPM.mp3">'
            '<a href="https://www.arrl.org/files/file/Morse/Archive/25%20WPM/260106_25.txt">'
            '<a href="/files/file/Morse/Archive/25%20WPM/260203_25WPM.mp3">')
    sessions = parse_archive_page(html, 25.0)
    assert sessions == [{"date": "260106", "wpm": 25.0,
                         "mp3": "/files/file/Morse/Archive/25%20WPM/260106_25WPM.mp3",
                         "txt": "/files/file/Morse/Archive/25%20WPM/260106_25.txt"}]
    assert SPACE not in sessions[0]["mp3"]
