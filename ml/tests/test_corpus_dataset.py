import json

import numpy as np
import pytest

from cwsynth.alphabet import PATTERNS, decode, encode, normalize, tokenize
from cwsynth.corpus import SPLITS, CorpusMix, callsign, callsign_split, clean_text
from cwsynth.dataset import read_wav, write_dataset
from cwsynth.spec import ConditionRanges


def test_aliases_normalize_to_prosigns():
    assert normalize(" cq  de k1abc + ") == "CQ DE K1ABC <AR>"
    assert normalize("= ( <bt>") == "<BT> <KN> <BT>"
    assert decode(encode("K1ABC <SK>")) == "K1ABC <SK>"
    with pytest.raises(ValueError):
        tokenize("HELLO!")
    with pytest.raises(ValueError):
        tokenize("<XX>")


def test_vocabulary_patterns_are_unique():
    assert len(set(PATTERNS.values())) == len(PATTERNS)


def test_every_source_produces_valid_text():
    corpus = CorpusMix()
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(3000):
        source, text = corpus.sample(rng, "train")
        seen.add(source)
        assert text == normalize(text)
        assert (text == "") == (source == "noise")
    assert seen == {"callsign", "qso", "abbreviation", "english", "random", "noise"}


def test_callsigns_stay_in_their_split():
    rng = np.random.default_rng(1)
    for split in SPLITS:
        calls = {callsign(rng, split) for _ in range(300)}
        assert all(callsign_split(c) == split for c in calls)


def test_text_file_source(tmp_path):
    path = tmp_path / "book.txt"
    path.write_text("It was a dark & stormy night; the rain fell (in torrents)!\n")
    corpus = CorpusMix({"english": 1.0}, text_file=path)
    text = corpus.sample(np.random.default_rng(0))[1]
    assert text and text == normalize(text)
    assert clean_text("a (b) c!") == "A B C."


def _snapshot(root):
    manifest = (root / "train.jsonl").read_text()
    audio = {row["id"]: read_wav(root / row["audio"])[0].tobytes()
             for row in map(json.loads, manifest.splitlines())}
    return manifest, audio


def test_dataset_is_identical_for_any_worker_count(tmp_path):
    ranges = ConditionRanges(max_signal_s=4.0)
    write_dataset(tmp_path / "a", 24, ranges=ranges, seed=7, workers=1, progress=False)
    write_dataset(tmp_path / "b", 24, ranges=ranges, seed=7, workers=3, progress=False)
    # Extending a split in two steps gives the same result as one step.
    write_dataset(tmp_path / "c", 10, ranges=ranges, seed=7, workers=1, progress=False)
    write_dataset(tmp_path / "c", 14, ranges=ranges, seed=7, start=10, workers=2, progress=False)
    a = _snapshot(tmp_path / "a")
    assert a == _snapshot(tmp_path / "b") == _snapshot(tmp_path / "c")

    info = json.loads((tmp_path / "c" / "dataset.json").read_text())
    assert info["splits"]["train"]["count"] == 24
    assert info["vocabulary"][0] == "<blank>"

    row = json.loads(a[0].splitlines()[0])
    assert {"id", "audio", "text", "duration_s", "params", "alignment", "snr_db_2500hz"} <= set(row)
    with pytest.raises(FileExistsError):
        write_dataset(tmp_path / "a", 2, seed=7, workers=1, progress=False)
