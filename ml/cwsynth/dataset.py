"""Batch generation: WAV files plus a JSONL manifest per split, deterministic and parallel.

Layout:
    out_dir/dataset.json                     generator version, vocabulary, per-split config
    out_dir/{split}.jsonl                    one line per sample, in index order
    out_dir/audio/{split}/0000/{id}.wav      8 kHz mono 16-bit PCM, 1000 files per folder

Sample `i` of a split is derived from (dataset seed, split, i) alone, so output is identical
for any worker count and a split can be extended later with `start=`.
"""
import json
import os
import subprocess
import sys
import time
import wave
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cwsynth.alphabet import TOKENS, tokenize
from cwsynth.channel import snr_in_bandwidth
from cwsynth.corpus import SPLITS, CorpusMix
from cwsynth.render import Sample, render
from cwsynth.spec import ConditionRanges, sample_spec

GENERATOR_VERSION = "0.1.0"
FILES_PER_FOLDER = 1000
# CWDecode's Goertzel detector bandwidth, for comparing against the app's squelch setting.
APP_DETECTOR_BANDWIDTH_HZ = 170.0


def sample_seed(dataset_seed: int, split: str, index: int) -> int:
    sequence = np.random.SeedSequence([dataset_seed, SPLITS.index(split), index])
    return int(sequence.generate_state(1, np.uint64)[0])


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).round().astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm.tobytes())


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as f:
        pcm = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2")
        return pcm.astype(np.float32) / 32767, f.getframerate()


def manifest_record(sample: Sample, sample_id: str, audio_path: str, split: str) -> dict[str, Any]:
    spec = sample.spec
    return {
        "id": sample_id,
        "audio": audio_path,
        "text": sample.text,
        "n_tokens": len(tokenize(sample.text)),
        "duration_s": round(sample.duration_s, 4),
        "sample_rate": spec.sample_rate,
        "split": split,
        "source": sample.source,
        "snr_db": spec.snr_db,
        "snr_db_2500hz": None if spec.snr_db is None else round(snr_in_bandwidth(spec.snr_db, 2500), 2),
        "snr_db_app": None if spec.snr_db is None
        else round(snr_in_bandwidth(spec.snr_db, APP_DETECTOR_BANDWIDTH_HZ), 2),
        "params": spec.to_dict(),
        "alignment": [[token, round(start, 4), round(end, 4)] for token, start, end in sample.alignment],
    }


# Per-process state, set up once by the pool initializer.
_worker: dict[str, Any] = {}


def _init_worker(ranges: ConditionRanges, weights: dict[str, float], text_file: str | None,
                 out_dir: str, split: str) -> None:
    _worker.update(ranges=ranges, corpus=CorpusMix(weights, text_file),
                   out_dir=Path(out_dir), split=split)


def _make_one(job: tuple[int, int]) -> dict[str, Any]:
    index, seed = job
    split, corpus, out_dir = _worker["split"], _worker["corpus"], _worker["out_dir"]
    rng = np.random.default_rng(seed)
    source, text = corpus.sample(rng, split)
    spec = sample_spec(rng, text, _worker["ranges"], qrm_text=lambda r: corpus.sample_text(r, split))
    sample = render(spec)
    sample.source = source

    sample_id = f"{split}-{index:07d}"
    relative = Path("audio") / split / f"{index // FILES_PER_FOLDER:04d}" / f"{sample_id}.wav"
    (out_dir / relative).parent.mkdir(parents=True, exist_ok=True)
    write_wav(out_dir / relative, sample.audio, spec.sample_rate)
    return manifest_record(sample, sample_id, relative.as_posix(), split)


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_dataset(out_dir: str | Path, n: int, split: str = "train",
                  ranges: ConditionRanges | None = None, corpus: CorpusMix | None = None,
                  seed: int = 0, start: int = 0, workers: int | None = None,
                  overwrite: bool = False, progress: bool = True) -> Path:
    """Write samples `start .. start+n-1` of `split`. With `start > 0` the manifest is
    appended to (extending a split); otherwise an existing manifest needs `overwrite`."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    ranges = ranges or ConditionRanges()
    corpus = corpus or CorpusMix()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / f"{split}.jsonl"
    if start == 0 and manifest.exists() and not overwrite:
        raise FileExistsError(f"{manifest} exists; pass overwrite=True or a start index to extend it")

    jobs = [(i, sample_seed(seed, split, i)) for i in range(start, start + n)]
    workers = workers or os.cpu_count() or 1
    init_args = (ranges, corpus.weights, corpus.text_file, str(out), split)
    began = time.monotonic()

    with open(manifest, "a" if start else "w", encoding="utf-8") as f:
        def emit(k: int, record: dict[str, Any]) -> None:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
            if progress and (k + 1) % 500 == 0:
                rate = (k + 1) / (time.monotonic() - began)
                print(f"  {split}: {k + 1}/{n} ({rate:.0f}/s)", file=sys.stderr)

        if workers == 1:
            _init_worker(*init_args)
            for k, job in enumerate(jobs):
                emit(k, _make_one(job))
        else:
            with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=init_args) as pool:
                for k, record in enumerate(pool.map(_make_one, jobs, chunksize=16)):
                    emit(k, record)

    _update_info(out, split, n, start, seed, ranges, corpus)
    return manifest


def _update_info(out: Path, split: str, n: int, start: int, seed: int,
                 ranges: ConditionRanges, corpus: CorpusMix) -> None:
    path = out / "dataset.json"
    info = json.loads(path.read_text()) if path.exists() else {}
    info.update(generator="cwsynth", generator_version=GENERATOR_VERSION, git_commit=_git_commit(),
                vocabulary=TOKENS, blank_id=0, snr_reference_bandwidth_hz=500.0)
    splits = info.setdefault("splits", {})
    previous = splits.get(split, {}) if start else {}
    splits[split] = {
        "count": start + n if start else n,
        "seed": seed,
        "conditions": asdict(ranges),
        "corpus": {"weights": corpus.weights, "text_file": corpus.text_file},
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if previous and (previous.get("seed") != seed or previous.get("conditions") != asdict(ranges)):
        splits[split]["note"] = "extended with a different seed or conditions than earlier samples"
    path.write_text(json.dumps(info, indent=2) + "\n")
