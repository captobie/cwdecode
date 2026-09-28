"""Training and evaluation data: synthetic samples generated on the fly, and manifests on disk
(cwsynth datasets, the W1AW library)."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from cwmodel.features import SAMPLE_RATE, n_frames, spectrogram
from cwmodel.net import output_lengths
from cwsynth.alphabet import SPACE, TOKEN_TO_ID, tokenize
from cwsynth.corpus import CorpusMix
from cwsynth.dataset import read_wav
from cwsynth.render import generate
from cwsynth.spec import ConditionRanges

# Keep the edge of a crop this far from a character, clear of its keying ramp.
_CROP_MARGIN_S = 0.006


def ctc_feasible(n_samples: int, tokens: list[str]) -> bool:
    """CTC needs a frame per token plus one between each repeated pair."""
    repeats = sum(a == b for a, b in zip(tokens, tokens[1:]))
    return int(output_lengths(torch.tensor(n_frames(n_samples)))) >= len(tokens) + repeats


def crop_at_gaps(audio: np.ndarray, text: str, alignment: list, rng: np.random.Generator,
                 min_s: float = 1.0) -> tuple[np.ndarray, str]:
    """A random stretch of whole characters, cut somewhere inside the gaps around it, like a
    live decoding window that happens to start and end between characters."""
    tokens = tokenize(text)
    positions = [i for i, t in enumerate(tokens) if t != SPACE]
    if len(alignment) < 2:
        return audio, text
    i = int(rng.integers(0, len(alignment)))
    j = int(rng.integers(i, len(alignment)))
    if alignment[j][2] - alignment[i][1] < min_s:
        return audio, text
    lo = alignment[i - 1][2] + _CROP_MARGIN_S if i > 0 else 0.0
    hi = alignment[i][1] - _CROP_MARGIN_S
    start = rng.uniform(lo, max(lo, hi)) if i > 0 else 0.0
    lo = alignment[j][2] + _CROP_MARGIN_S
    hi = alignment[j + 1][1] - _CROP_MARGIN_S if j + 1 < len(alignment) else len(audio) / SAMPLE_RATE
    end = rng.uniform(min(lo, hi), hi) if j + 1 < len(alignment) else hi
    label = "".join(tokens[positions[i]:positions[j] + 1])
    return audio[round(start * SAMPLE_RATE):round(end * SAMPLE_RATE)], label


def encode_item(audio: np.ndarray, text: str, meta: dict) -> dict:
    tokens = tokenize(text)
    return {"spec": spectrogram(audio), "ids": torch.tensor([TOKEN_TO_ID[t] for t in tokens]),
            "text": "".join(tokens), "meta": meta}


class SyntheticStream(IterableDataset):
    """Endless samples from cwsynth.generate. Sample k is fixed by (base_seed, k), and the
    workers take every n-th k, so a run is reproducible for a given worker count."""

    def __init__(self, ranges: ConditionRanges | None = None, corpus_weights: dict | None = None,
                 base_seed: int = 1, crop_prob: float = 0.3):
        self.ranges = ranges or ConditionRanges()
        self.corpus_weights = corpus_weights
        self.base_seed = base_seed
        self.crop_prob = crop_prob

    def __iter__(self):
        info = get_worker_info()
        worker, workers = (info.id, info.num_workers) if info else (0, 1)
        if info:
            # One thread per worker: otherwise each worker's torch/numpy thread pool fights the
            # others for the CPU and the GPU sits waiting for data.
            torch.set_num_threads(1)
        corpus = CorpusMix(self.corpus_weights)
        k = worker
        while True:
            seed = int(np.random.SeedSequence([self.base_seed, k]).generate_state(1, np.uint64)[0])
            k += workers
            sample = generate(seed, self.ranges, corpus)
            audio, text = sample.audio, sample.text
            rng = np.random.default_rng(seed ^ 0x5EED)
            if text and rng.random() < self.crop_prob:
                audio, text = crop_at_gaps(audio, text, sample.alignment, rng)
            if ctc_feasible(len(audio), tokenize(text)):
                yield encode_item(audio, text, {"source": sample.source, "seed": seed})


class BucketedBatches(IterableDataset):
    """Collated batches of similar length. Each worker fills a pool of `pool` batches from
    `stream`, sorts it by length and splits it, so little of each batch is padding. Use with
    DataLoader(batch_size=None)."""

    def __init__(self, stream: SyntheticStream, batch_size: int, pool: int = 8):
        self.stream, self.batch_size, self.pool = stream, batch_size, pool

    def __iter__(self):
        items = iter(self.stream)
        rng = np.random.default_rng()
        while True:
            pool = sorted((next(items) for _ in range(self.batch_size * self.pool)),
                          key=lambda it: it["spec"].shape[1])
            batches = [pool[i:i + self.batch_size] for i in range(0, len(pool), self.batch_size)]
            for i in rng.permutation(len(batches)):
                yield collate(batches[i])


class FixedSamples(Dataset):
    """A fixed list of synthetic samples, for the overfit check."""

    def __init__(self, n: int, ranges: ConditionRanges, base_seed: int = 7):
        corpus = CorpusMix()
        self.items = [encode_item(s.audio, s.text, {"source": s.source})
                      for s in (generate(base_seed * 100_003 + i, ranges, corpus) for i in range(n))]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        return self.items[i]


class ManifestSamples(Dataset):
    """Rows of a cwsynth or W1AW manifest (JSONL); audio paths are relative to its folder."""

    def __init__(self, manifest: str | Path, limit: int | None = None, every: int = 1):
        path = Path(manifest)
        self.root = path.parent
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.rows = rows[::every][:limit]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        row = self.rows[i]
        audio, sr = read_wav(self.root / row["audio"])
        if sr != SAMPLE_RATE:
            raise ValueError(f"{row['audio']}: expected {SAMPLE_RATE} Hz, got {sr}")
        return encode_item(audio, row["text"], row)


# Batches are padded to a multiple of this many frames (~1 s). The GPU (MPS) compiles a new
# graph for every input shape it sees, so free-running lengths made training 6× slower.
PAD_FRAMES_TO = 128


def collate(items: list[dict]) -> dict:
    frames = torch.tensor([it["spec"].shape[1] for it in items])
    width = -(-int(frames.max()) // PAD_FRAMES_TO) * PAD_FRAMES_TO
    batch = torch.zeros(len(items), 1, items[0]["spec"].shape[0], width)
    for b, it in enumerate(items):
        batch[b, 0, :, :it["spec"].shape[1]] = it["spec"]
    return {
        "spec": batch,
        "frames": frames,
        "targets": torch.cat([it["ids"] for it in items]) if items else torch.empty(0, dtype=torch.long),
        "target_lengths": torch.tensor([len(it["ids"]) for it in items]),
        "texts": [it["text"] for it in items],
        "metas": [it["meta"] for it in items],
    }
