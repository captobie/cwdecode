import io
import json

import numpy as np
import pytest

from cwsynth import job
from cwsynth.corpus import CorpusMix
from cwsynth.dataset import read_wav, write_dataset
from cwsynth.spec import ConditionRanges

FIXED = {"wpm": [20.0, 20.0], "tone_hz": [650.0, 650.0], "snr_db": [6.0, 6.0], "clean_prob": 0.0,
         "farnsworth_prob": 0.0, "fist_styles": {"paddle": 1.0}, "qsb_prob": 0.0, "qrn_prob": 0.0,
         "qrm_prob": 0.0, "rx_filter_prob": 0.0, "drift_prob": 0.0, "chirp_prob": 0.0}


def run_job(request: dict, capsys) -> tuple[int, list[dict]]:
    status = job.run(request)
    return status, [json.loads(line) for line in capsys.readouterr().out.splitlines()]


def test_custom_lines_cycle_by_index():
    corpus = CorpusMix(lines=["cq de k1abc +", "test"])
    rng = np.random.default_rng(0)
    assert [corpus.sample_at(i, rng)[1] for i in range(3)] == ["CQ DE K1ABC <AR>", "TEST", "CQ DE K1ABC <AR>"]
    with pytest.raises(ValueError):
        CorpusMix(lines=["HELLO!"])


def test_dataset_with_lines_and_fixed_conditions(tmp_path):
    write_dataset(tmp_path, 4, ranges=ConditionRanges.from_dict(FIXED),
                  corpus=CorpusMix(lines=["A B", "73"]), workers=1, progress=False)
    rows = [json.loads(line) for line in (tmp_path / "train.jsonl").read_text().splitlines()]
    assert [r["text"] for r in rows] == ["A B", "73", "A B", "73"]
    assert {r["source"] for r in rows} == {"custom"}
    for r in rows:
        assert r["params"]["wpm"] == 20.0 and r["params"]["tone_hz"] == 650.0
        assert r["snr_db"] == 6.0 and r["params"]["fist"]["style"] == "paddle"
    info = json.loads((tmp_path / "dataset.json").read_text())
    assert info["splits"]["train"]["corpus"]["lines"] == ["A B", "73"]


def test_preview_is_the_first_dataset_sample(tmp_path, capsys):
    request = {"conditions": FIXED, "corpus": {"lines": ["CQ TEST"]}, "seed": 3}
    status, events = run_job({**request, "action": "preview", "wav": str(tmp_path / "p.wav")}, capsys)
    assert status == 0 and events[0]["event"] == "preview"
    assert events[0]["text"] == "CQ TEST" and events[0]["snr_db_app"] == pytest.approx(10.7)
    status, events = run_job({**request, "action": "dataset", "out": str(tmp_path / "ds"), "n": 2,
                              "workers": 1}, capsys)
    assert status == 0 and events[-1] == {"event": "done", "manifest": str(tmp_path / "ds" / "train.jsonl"),
                                          "count": 2}
    assert events[-2] == {"event": "progress", "done": 2, "total": 2}
    preview, _ = read_wav(tmp_path / "p.wav")
    first, _ = read_wav(tmp_path / "ds" / "audio" / "train" / "0000" / "train-0000000.wav")
    assert np.array_equal(preview, first)


def test_existing_split_needs_overwrite_or_start(tmp_path, capsys):
    request = {"action": "dataset", "conditions": FIXED, "out": str(tmp_path), "n": 2, "workers": 1}
    run_job(request, capsys)
    status, events = run_job(request, capsys)
    assert status == 1 and events == [{"event": "exists", "count": 2,
                                       "manifest": str(tmp_path / "train.jsonl")}]
    status, events = run_job({**request, "start": 2}, capsys)
    assert status == 0 and events[-1]["count"] == 4
    assert len((tmp_path / "train.jsonl").read_text().splitlines()) == 4


def test_check_reports_bad_and_long_lines(capsys):
    lines = ["cq de k1abc", "HELLO!", "", "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG"]
    status, events = run_job({"action": "check", "conditions": FIXED, "corpus": {"lines": lines}}, capsys)
    results = events[0]["lines"]
    assert status == 0 and [r["line"] for r in results] == [0, 1, 3]
    assert results[0]["text"] == "CQ DE K1ABC" and not results[0]["too_long"]
    assert "!" in results[1]["error"]
    assert results[2]["too_long"]


def test_bad_request_is_an_error_event(capsys, monkeypatch):
    request = json.dumps({"action": "dataset", "conditions": {"bogus": 1}})
    monkeypatch.setattr("sys.stdin", io.StringIO(request))
    with pytest.raises(SystemExit) as exit_info:
        job.main()
    assert exit_info.value.code == 1
    event = json.loads(capsys.readouterr().out)
    assert event["event"] == "error" and "bogus" in event["message"]
