"""Streaming decoding with overlapping windows: the reference the app's Swift decoder matches.

Every `hop_s` the last `window_s` of audio is decoded. Each stretch of audio is committed from
the window where it sits with at least `context_s` of audio after it (and, once the stream is
long enough, `window_s - hop_s - context_s` before it). The handoff point between windows is
placed in the widest gap between emitted tokens near the nominal boundary, so a character
emitted a frame or two differently by neighboring windows is still committed exactly once.

A token whose timing depends on context (above all a space, which can only be decided once a
gap has run long enough) can land on different sides of the handoff in neighboring windows.
So the new window's tokens from the last `RECONCILE_S` before the handoff are also aligned with
what was already committed there: tokens the earlier window missed are added, and tokens it
already committed aren't repeated.

Text after the handoff point is tentative: shown, but replaced on the next decode.

    python -m cwmodel.stream --ckpt runs/full/best.pt --sessions data/w1aw/sessions.jsonl
"""
import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from cwmodel.decode import edit_distance
from cwmodel.features import FRAME_S, HOP, SAMPLE_RATE, spectrogram
from cwmodel.net import TIME_STRIDE
from cwsynth.alphabet import SPACE, TOKENS, tokenize

OUTPUT_FRAME_S = FRAME_S * TIME_STRIDE          # 16 ms per model output frame
# The handoff point is searched this far either side of its nominal position.
HANDOFF_SEARCH_S = 0.5
# The exported model takes at least this many spectrogram frames; shorter windows (only ever the
# last one of a very short stream) are padded with silence to reach it.
MIN_FRAMES = 64
MIN_SAMPLES = (MIN_FRAMES - 1) * HOP
# How far back before the handoff a new window's tokens are compared with committed text.
RECONCILE_S = 1.5

Emission = tuple[str, float]                    # (token, absolute time in seconds)
LogProbs = Callable[[torch.Tensor], np.ndarray]  # [bins, frames] → [out frames, classes]


def emissions(log_probs: np.ndarray, start_s: float) -> list[Emission]:
    """Greedy CTC: the best token per frame, runs collapsed, blanks dropped. Each token is
    stamped with the time of its first frame."""
    best = log_probs.argmax(axis=1)
    out, previous = [], 0
    for i, token_id in enumerate(best):
        if token_id != previous and token_id != 0:
            out.append((TOKENS[token_id], start_s + i * OUTPUT_FRAME_S))
        previous = token_id
    return out


def last_match(a: list[str], b: list[str]) -> tuple[int, int] | None:
    """The last pair (i, j) with a[i] == b[j] in a longest common subsequence of a and b,
    backtracking from the end and preferring to drop from `a` on ties (so Swift can match)."""
    n, m = len(a), len(b)
    lengths = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            lengths[i + 1][j + 1] = (lengths[i][j] + 1 if a[i] == b[j]
                                     else max(lengths[i][j + 1], lengths[i + 1][j]))
    i, j = n, m
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1]:
            return i - 1, j - 1
        if lengths[i - 1][j] >= lengths[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return None


def reconcile(previous: list[str], before: list[str], after: list[str]) -> tuple[list[str], int]:
    """Compares tokens committed in the last RECONCILE_S (`previous`) with the new window's
    tokens in that stretch (`before`) and after it (`after`). Returns tokens the earlier window
    missed (to commit first) and how many leading tokens of `after` it already committed."""
    match = last_match(previous, before) if previous and before else None
    if match is None:
        return [], 0
    i, j = match
    if i == len(previous) - 1:
        return before[j + 1:], 0
    already = previous[i + 1:] if j == len(before) - 1 else []
    return [], len(already) if after[:len(already)] == already else 0


def handoff(times: list[float], lo: float, hi: float) -> float:
    """The point in [lo, hi] farthest from any emission time."""
    if hi <= lo:
        return lo
    candidates = np.arange(lo, hi + 1e-9, OUTPUT_FRAME_S)
    if not times:
        return float(candidates[len(candidates) // 2])
    distance = np.min(np.abs(candidates[:, None] - np.array(times)[None, :]), axis=1)
    return float(candidates[int(np.argmax(distance))])


@dataclass
class StreamingDecoder:
    log_probs: LogProbs
    window_s: float = 6.0
    hop_s: float = 1.0
    context_s: float = 2.5

    committed: list[str] = field(default_factory=list)
    tentative: list[str] = field(default_factory=list)
    _recent: list[Emission] = field(default_factory=list)   # committed tokens with their times
    _audio: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    _origin: int = 0              # absolute sample index of _audio[0]
    _decoded_to: int = 0          # absolute sample index where the last window ended
    _committed_until: float = 0.0

    @property
    def text(self) -> str:
        return "".join(self.committed)

    def process(self, samples: np.ndarray) -> str:
        """Feed 8 kHz audio; returns newly committed text."""
        self._audio = np.concatenate([self._audio, samples.astype(np.float32)])
        end = self._origin + len(self._audio)
        new = []
        hop = round(self.hop_s * SAMPLE_RATE)
        while end - self._decoded_to >= hop:
            self._decoded_to += hop
            new += self._decode(self._decoded_to, final=False)
        return "".join(new)

    def finish(self) -> str:
        """Commits everything, including the tentative tail."""
        end = self._origin + len(self._audio)
        new = self._decode(end, final=True) if end > 0 else []
        self._decoded_to = end
        return "".join(new)

    def _decode(self, end: int, final: bool) -> list[str]:
        window = round(self.window_s * SAMPLE_RATE)
        start = max(self._origin, end - window)
        audio = self._audio[start - self._origin:end - self._origin]
        start_s, end_s = start / SAMPLE_RATE, end / SAMPLE_RATE
        if len(audio) < MIN_SAMPLES:
            audio = np.concatenate([audio, np.zeros(MIN_SAMPLES - len(audio), np.float32)])
        tokens = emissions(self.log_probs(spectrogram(audio)), start_s)

        if final:
            cut = np.inf
        else:
            nominal = end_s - self.context_s
            lo = max(self._committed_until, nominal - HANDOFF_SEARCH_S)
            hi = max(lo, nominal + HANDOFF_SEARCH_S)
            cut = handoff([t for _, t in tokens], lo, hi)

        since = self._committed_until - RECONCILE_S
        before = [token for token, t in tokens if since <= t < self._committed_until]
        after = [(token, t) for token, t in tokens if self._committed_until <= t < cut]
        previous = [token for token, t in self._recent if t >= since]
        missed, skip = reconcile(previous, before, [token for token, _ in after])
        # Missed tokens are stamped at the old handoff; the rest keep this window's times.
        new = self._append(missed + [token for token, _ in after[skip:]],
                           [self._committed_until] * len(missed) + [t for _, t in after[skip:]])
        self.tentative = [token for token, t in tokens if t >= cut]
        if not final:
            self._committed_until = cut
            # Keep only what the next window can reach.
            keep_from = max(self._origin, end + round(self.hop_s * SAMPLE_RATE) - window)
            self._audio = self._audio[keep_from - self._origin:]
            self._origin = keep_from
        return new

    def _append(self, tokens: list[str], times: list[float]) -> list[str]:
        """Adds tokens to the committed text without leading or doubled spaces."""
        added = []
        for token, t in zip(tokens, times):
            last = (added or self.committed or [SPACE])[-1]
            if token == SPACE and last == SPACE:
                continue
            added.append(token)
            self._recent.append((token, t))
        self.committed += added
        self._recent = [(token, t) for token, t in self._recent if t >= self._committed_until - 2 * RECONCILE_S]
        return added


def torch_log_probs(model: torch.nn.Module, device: str = "cpu") -> LogProbs:
    @torch.no_grad()
    def run(spec: torch.Tensor) -> np.ndarray:
        return model(spec[None, None].to(device))[:, 0].cpu().numpy()
    return run


def decode_session(decoder: StreamingDecoder, audio: np.ndarray, chunk_s: float = 0.1) -> str:
    chunk = round(chunk_s * SAMPLE_RATE)
    for i in range(0, len(audio), chunk):
        decoder.process(audio[i:i + chunk])
    decoder.finish()
    return decoder.text.strip()


def main() -> None:
    from cwmodel.evaluate import load_model
    from cwsynth import channel, classic
    from cwsynth.dataset import read_wav

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--sessions", default="data/w1aw/sessions.jsonl")
    parser.add_argument("--window", type=float, nargs="+", default=[6.0])
    parser.add_argument("--hop", type=float, default=1.0)
    parser.add_argument("--context", type=float, nargs="+", default=[2.5])
    parser.add_argument("--snr", type=float, help="add white noise at this SNR (500 Hz reference)")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()

    model = load_model(args.ckpt, args.device).eval()
    root = Path(args.sessions).parent
    sessions = [json.loads(line) for line in Path(args.sessions).read_text().splitlines()][:args.limit]
    for window, context in zip(args.window, args.context * len(args.window) if len(args.context) == 1 else args.context):
        errors = length = 0
        for s in sessions:
            audio, _ = read_wav(root / s["audio"])
            if args.snr is not None:
                carrier = classic.decode(audio, SAMPLE_RATE).carrier_amplitude
                rng = np.random.default_rng(0)
                audio = (audio / carrier + channel.white_noise(len(audio), args.snr, SAMPLE_RATE, rng)).astype(np.float32)
            decoder = StreamingDecoder(torch_log_probs(model, args.device), window, args.hop, context)
            hypothesis, reference = tokenize(decode_session(decoder, audio)), tokenize(s["text"])
            e = edit_distance(hypothesis, reference)
            errors, length = errors + e, length + len(reference)
            print(f"  {s['id']:16s} CER {e / len(reference):6.2%}", flush=True)
        print(f"window {window:g} s, hop {args.hop:g} s, context {context:g} s: "
              f"session CER {errors / length:.2%} over {len(sessions)} sessions", flush=True)


if __name__ == "__main__":
    main()
