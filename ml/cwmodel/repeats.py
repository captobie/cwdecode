"""Repeated-character check: does a model output one keyed character twice?

    python -m cwmodel.repeats --ckpt runs/full/best.pt runs/exp/last.pt
    python -m cwmodel.repeats --ckpt runs/exp/last.pt --fail-above 0.05

Greedy CTC decoding only turns one character into two if the model fires twice inside it with
a blank in between. The first model did this with the comma at 18–19 WPM only (`,` → `,,`),
which W1AW's 18 WPM bulletins exposed and nothing else measured. So every keyed token is
rendered between two words, as clean machine-sent code at every speed in the range, and the
check counts how often the output contains that token twice in a row. Whether it fires twice
depends on the neighbouring words (`OK , OK` doubled, `TEST , OK` didn't), so each token is
tried in several contexts, standalone and attached to the word before it (`FRIDAY, AT`).
"""
import argparse
import sys
from collections import defaultdict

import numpy as np
import torch

from cwmodel.data import collate, encode_item
from cwmodel.decode import greedy_decode
from cwmodel.evaluate import load_model
from cwmodel.net import CwNet, output_lengths
from cwsynth.alphabet import KEYED, tokenize
from cwsynth.render import render
from cwsynth.spec import SampleSpec

# Word pairs around the tested token. A pair containing the token is skipped for it, so a
# dropped word space ("OK O OK" → "OK OOK") can't pass for a doubled token.
CONTEXTS = [("OK", "OK"), ("TEST", "OK"), ("FRIDAY", "AT"), ("HW", "UR"), ("DE", "BK"), ("GM", "PSE")]
SNR_DB = 20.0


def variants(token: str) -> list[str]:
    """The texts `token` is tested in: each usable context, standalone and attached."""
    usable = [(a, b) for a, b in CONTEXTS if token not in tokenize(a) + tokenize(b)]
    return [f"{a} {token} {b}" for a, b in usable] + [f"{a}{token} {b}" for a, b in usable]


def doubled(prediction: str, token: str) -> bool:
    """Whether `prediction` contains `token` twice in a row."""
    tokens = tokenize(prediction)
    return any(a == b == token for a, b in zip(tokens, tokens[1:]))


def clips(speeds: list[float], tokens: list[str] = KEYED) -> list[tuple[str, float, dict]]:
    """(token, wpm, encoded item) for every token, speed and context. The tone varies with the
    context; timing is exact machine keying."""
    out = []
    for wpm in speeds:
        for token in tokens:
            for seed, text in enumerate(variants(token)):
                spec = SampleSpec(text=text, seed=seed, wpm=float(wpm),
                                  tone_hz=500.0 + 100.0 * (seed % 5), snr_db=SNR_DB)
                sample = render(spec)
                out.append((token, float(wpm), encode_item(sample.audio, sample.text, {})))
    return out


@torch.no_grad()
def check(model: CwNet, items: list[tuple[str, float, dict]], device: str,
          batch_size: int = 64) -> dict[tuple[str, float], tuple[int, int]]:
    """{(token, wpm): (doubled, rendered)}."""
    model.eval()
    counts: dict[tuple[str, float], list[int]] = defaultdict(lambda: [0, 0])
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        batch = collate([item for _, _, item in chunk])
        predictions = greedy_decode(model(batch["spec"].to(device)), output_lengths(batch["frames"]))
        for (token, wpm, _), prediction in zip(chunk, predictions):
            counts[token, wpm][0] += doubled(prediction, token)
            counts[token, wpm][1] += 1
    return {key: (d, n) for key, (d, n) in counts.items()}


def worst_rate(counts: dict[tuple[str, float], tuple[int, int]]) -> float:
    return max((d / n for d, n in counts.values()), default=0.0)


def report(name: str, counts: dict[tuple[str, float], tuple[int, int]]) -> None:
    total = sum(d for d, _ in counts.values())
    rendered = sum(n for _, n in counts.values())
    print(f"{name}: {total} doubled of {rendered} ({total / max(1, rendered):.2%}), "
          f"worst {worst_rate(counts):.0%}")
    by_token: dict[str, list[tuple[float, int, int]]] = defaultdict(list)
    for (token, wpm), (d, n) in sorted(counts.items(), key=lambda kv: kv[0][1]):
        if d:
            by_token[token].append((wpm, d, n))
    for token, cells in sorted(by_token.items(), key=lambda kv: -sum(d for _, d, _ in kv[1])):
        print(f"  {token!r:7s} " + "  ".join(f"{wpm:g} WPM {d}/{n}" for wpm, d, n in cells))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True, nargs="+")
    parser.add_argument("--speeds", type=float, nargs=2, default=[10.0, 40.0], metavar=("LOW", "HIGH"))
    parser.add_argument("--step", type=float, default=1.0, help="WPM between tested speeds")
    parser.add_argument("--fail-above", type=float,
                        help="exit 1 if any token at any speed doubles more often than this")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()

    speeds = [float(s) for s in np.arange(args.speeds[0], args.speeds[1] + 1e-9, args.step)]
    items = clips(speeds)
    failed = False
    for path in args.ckpt:
        counts = check(load_model(path, args.device), items, args.device)
        report(path, counts)
        failed |= args.fail_above is not None and worst_rate(counts) > args.fail_above
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
