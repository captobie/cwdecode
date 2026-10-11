"""JSON job interface for front ends (the CWSynth app): one request on stdin, events on stdout.

    echo '{"action": "preview", ...}' | python -m cwsynth job

Request fields (all optional except `action`):

    action       "check" | "preview" | "dataset"
    conditions   ConditionRanges fields, as in a config's [conditions]; a fixed value is a
                 range with equal ends, an impairment is switched off with its _prob at 0
    corpus       {"weights": {...}, "text_file": path, "lines": [...]}; `lines` replaces the
                 weighted mix with one custom text per sample, cycling
    split, seed  as for `cwsynth dataset` (defaults "train", 0)
    index        preview: which sample of the split to render (default 0)
    wav          preview: where to write the clip
    out, n       dataset: output folder and number of samples
    start        dataset: first index; >0 extends an existing split
    overwrite    dataset: replace an existing split

Every stdout line is one JSON object with an "event" field: "check", "preview", "progress"
{done, total}, "done", "exists" {count} (the split is already there; resend with overwrite or
start), or "error" {message}. The exit status is 0 unless the last event is "error" or "exists".
"""
import json
import sys
from pathlib import Path
from typing import Any

from cwsynth.alphabet import normalize, tokenize
from cwsynth.channel import snr_in_bandwidth
from cwsynth.corpus import SPLITS, CorpusMix
from cwsynth.dataset import (APP_DETECTOR_BANDWIDTH_HZ, make_sample, sample_seed, write_dataset,
                             write_wav)
from cwsynth.keying import FistParams, estimate_duration
from cwsynth.spec import ConditionRanges


def emit(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, **fields}), flush=True)


def check(request: dict, ranges: ConditionRanges) -> dict:
    """Each custom line normalized, or why it can't be sent; and its keyed length at the
    slowest speed, so the front end can warn before `max_signal_s` trims it."""
    slowest = ranges.wpm[0]
    farnsworth = slowest * ranges.farnsworth_ratio[0] if ranges.farnsworth_prob > 0 else None
    results = []
    for i, line in enumerate(request.get("corpus", {}).get("lines") or []):
        if not line.strip():
            continue
        try:
            text = normalize(line)
        except ValueError as e:
            results.append({"line": i, "error": str(e)})
            continue
        seconds = estimate_duration(tokenize(text), slowest, FistParams(), farnsworth)
        results.append({"line": i, "text": text, "seconds": round(seconds, 2),
                        "too_long": seconds > ranges.max_signal_s})
    return {"lines": results, "max_signal_s": ranges.max_signal_s}


def describe(sample) -> dict:
    spec = sample.spec
    snr = spec.snr_db
    return {
        "text": sample.text, "source": sample.source, "duration_s": round(sample.duration_s, 2),
        "wpm": round(spec.wpm, 1),
        "farnsworth_wpm": None if spec.farnsworth_wpm is None else round(spec.farnsworth_wpm, 1),
        "fist": spec.fist.style, "tone_hz": round(spec.tone_hz),
        "snr_db": None if snr is None else round(snr, 1),
        "snr_db_2500hz": None if snr is None else round(snr_in_bandwidth(snr, 2500), 1),
        "snr_db_app": None if snr is None else round(snr_in_bandwidth(snr, APP_DETECTOR_BANDWIDTH_HZ), 1),
        "qsb": None if spec.qsb is None else f"{spec.qsb.depth_db:.0f} dB at {spec.qsb.rate_hz:.2f} Hz",
        "qrn": None if spec.qrn is None else f"{spec.qrn.rate_per_s:.1f}/s at {spec.qrn.level_db:+.0f} dB",
        "qrm": len(spec.qrm),
        "rx_filter": None if spec.rx_filter is None
        else f"{spec.rx_filter.low_hz:.0f}–{spec.rx_filter.high_hz:.0f} Hz",
    }


def run(request: dict) -> int:
    action = request.get("action")
    ranges = ConditionRanges.from_dict(request.get("conditions", {}))
    corpus_config = request.get("corpus", {})
    split = request.get("split", "train")
    seed = int(request.get("seed", 0))
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")

    if action == "check":
        emit("check", **check(request, ranges))
        return 0

    lines = [line for line in corpus_config.get("lines") or [] if line.strip()] or None
    corpus = CorpusMix(corpus_config.get("weights"), corpus_config.get("text_file"), lines)

    if action == "preview":
        index = int(request.get("index", 0))
        sample = make_sample(index, sample_seed(seed, split, index), ranges, corpus, split)
        wav = Path(request["wav"])
        wav.parent.mkdir(parents=True, exist_ok=True)
        write_wav(wav, sample.audio, sample.spec.sample_rate)
        requested = corpus.lines[index % len(corpus.lines)] if corpus.lines else None
        emit("preview", wav=str(wav), index=index, trimmed=requested is not None and sample.text != requested,
             **describe(sample))
        return 0

    if action == "dataset":
        out, n, start = Path(request["out"]), int(request["n"]), int(request.get("start", 0))
        manifest = out / f"{split}.jsonl"
        if start == 0 and manifest.exists() and not request.get("overwrite"):
            with open(manifest, encoding="utf-8") as f:
                emit("exists", count=sum(1 for _ in f), manifest=str(manifest))
            return 1
        step = max(1, n // 200)

        def progress(done: int, total: int) -> None:
            if done % step == 0 or done == total:
                emit("progress", done=done, total=total)

        write_dataset(out, n, split=split, ranges=ranges, corpus=corpus, seed=seed, start=start,
                      workers=request.get("workers"), overwrite=bool(request.get("overwrite")),
                      progress=False, on_progress=progress)
        emit("done", manifest=str(manifest), count=start + n)
        return 0

    raise ValueError(f"unknown action: {action!r}")


def main() -> None:
    try:
        status = run(json.loads(sys.stdin.read()))
    except Exception as e:  # noqa: BLE001 - every failure goes back to the front end as an event
        emit("error", message=f"{type(e).__name__}: {e}")
        status = 1
    sys.exit(status)
