"""Character error rate on a manifest, broken down by condition.

    python -m cwmodel.evaluate --ckpt runs/full/best.pt --manifest data/synth-v1/val.jsonl
    python -m cwmodel.evaluate --ckpt runs/full/best.pt --manifest data/w1aw/clips.jsonl
    python -m cwmodel.evaluate --ckpt runs/full/best.pt --manifest data/w1aw/degraded.jsonl --every 4

Noise-only samples (empty labels) have no error rate; they're reported as the average number
of characters the model invents per sample, which should be close to zero.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cwmodel.data import ManifestSamples, collate
from cwmodel.decode import greedy_decode, token_errors
from cwmodel.net import CwNet, output_lengths
from cwsynth.alphabet import tokenize

SNR_BINS = [-np.inf, -6, -3, 0, 3, 6, 10, 15, np.inf]
WPM_BINS = [0, 12, 16, 20, 25, 30, 35, 100]


def _bin(value: float, edges: list[float], fmt: str) -> str:
    i = int(np.searchsorted(edges, value, side="right")) - 1
    lo, hi = edges[i], edges[i + 1]
    return (f"<{fmt.format(hi)}" if lo == -np.inf else f"≥{fmt.format(lo)}" if hi in (np.inf, 100)
            else f"{fmt.format(lo)}…{fmt.format(hi)}")


def groups_for(meta: dict) -> dict[str, str]:
    params = meta.get("params", {})
    snr = meta.get("snr_db")
    out = {"snr_db": "clean" if snr is None else _bin(snr, SNR_BINS, "{:+.0f}")}
    if "wpm" in params:
        out["wpm"] = f"{params['wpm']:g}" if meta.get("source") == "w1aw" else _bin(params["wpm"], WPM_BINS, "{:g}")
    if "fist" in params:
        out["fist"] = params["fist"]["style"]
    if meta.get("source"):
        out["source"] = meta["source"]
    return out


@torch.no_grad()
def run(model: CwNet, dataset, device: str, batch_size: int = 32, workers: int = 4) -> list[dict]:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate, num_workers=workers)
    results = []
    for batch in loader:
        log_probs = model(batch["spec"].to(device))
        predictions = greedy_decode(log_probs, output_lengths(batch["frames"]))
        for pred, ref, meta in zip(predictions, batch["texts"], batch["metas"]):
            errors, length = token_errors(pred, ref)
            results.append({"id": meta.get("id"), "prediction": pred, "reference": ref,
                            "errors": errors, "length": length, "groups": groups_for(meta)})
    return results


def summarize(results: list[dict]) -> dict:
    labeled = [r for r in results if r["length"]]
    empty = [r for r in results if not r["length"]]
    summary = {
        "cer": sum(r["errors"] for r in labeled) / max(1, sum(r["length"] for r in labeled)),
        "samples": len(results),
        "exact": sum(r["errors"] == 0 for r in labeled) / max(1, len(labeled)),
        "noise_only_false_chars": (sum(len(tokenize(r["prediction"])) for r in empty) / len(empty)
                                   if empty else None),
        "by": {},
    }
    tallies: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    for r in labeled:
        for key, value in r["groups"].items():
            t = tallies[key][value]
            t[0] += r["errors"]
            t[1] += r["length"]
            t[2] += 1
    for key, values in tallies.items():
        summary["by"][key] = {v: {"cer": e / n, "samples": c} for v, (e, n, c) in values.items()}
    return summary


def print_summary(name: str, summary: dict) -> None:
    extra = (f", noise-only: {summary['noise_only_false_chars']:.2f} false chars/sample"
             if summary["noise_only_false_chars"] is not None else "")
    print(f"{name}: CER {summary['cer']:.2%}, exact {summary['exact']:.1%} of {summary['samples']}{extra}")
    for key, values in summary["by"].items():
        if len(values) < 2:
            continue
        order = sorted(values, key=lambda v: (v == "clean", _sort_key(v)))
        print(f"  {key:7s} " + "  ".join(f"{v} {values[v]['cer']:.1%}" for v in order))


def _sort_key(v: str):
    digits = "".join(c for c in v if c.isdigit() or c in "+-.")
    try:
        return (0, float(digits.lstrip("…") or 0) - (0.5 if v.startswith("<") else 0), v)
    except ValueError:
        return (1, 0.0, v)


def load_model(path: str | Path, device: str) -> CwNet:
    checkpoint = torch.load(path, map_location=device)
    model = CwNet(**checkpoint.get("net", {})).to(device)
    model.load_state_dict(checkpoint["model"])
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--manifest", required=True, nargs="+")
    parser.add_argument("--every", type=int, default=1, help="use every n-th row")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", help="write per-sample predictions (JSONL)")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()

    model = load_model(args.ckpt, args.device)
    for manifest in args.manifest:
        results = run(model, ManifestSamples(manifest, args.limit, args.every), args.device)
        print_summary(manifest, summarize(results))
        if args.out:
            with open(args.out, "a") as f:
                for r in results:
                    f.write(json.dumps({**r, "manifest": manifest}) + "\n")


if __name__ == "__main__":
    main()
