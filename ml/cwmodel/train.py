"""Train the CTC model on synthetic CW generated on the fly.

    python -m cwmodel.train --overfit 100 --out runs/overfit          # wiring check: CER → 0
    python -m cwmodel.train --stage clean --steps 6000 --out runs/clean
    python -m cwmodel.train --stage full --steps 30000 --init runs/clean/best.pt --out runs/full

Stages: `clean` has every fist, speed, tone and text but no channel impairments; `full` uses
the default conditions (noise, fading, static, interference, filters). Every `--eval-every`
steps it scores the synthetic val set and a W1AW subset, and keeps the best checkpoint by
synthetic val CER.
"""
import argparse
import dataclasses
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cwmodel import evaluate
from cwmodel.data import BucketedBatches, FixedSamples, ManifestSamples, SyntheticStream, collate
from cwmodel.decode import greedy_decode, token_errors
from cwmodel.net import CwNet, count_parameters, output_lengths
from cwsynth.alphabet import TOKENS
from cwsynth.spec import ConditionRanges

STAGES = {
    "clean": dict(clean_prob=1.0, qsb_prob=0.0, qrn_prob=0.0, qrm_prob=0.0,
                  rx_filter_prob=0.0, drift_prob=0.0, chirp_prob=0.0),
    "full": {},
}


def learning_rate(step: int, total: int, peak: float, warmup: int, floor: float = 1e-5) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * progress))


def ctc_loss(model: CwNet, batch: dict, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    log_probs = model(batch["spec"].to(device))
    lengths = output_lengths(batch["frames"])
    loss = F.ctc_loss(log_probs, batch["targets"].to(device), lengths.to(device),
                      batch["target_lengths"].to(device), blank=0, zero_infinity=True)
    return loss, log_probs


def save(path: Path, model: CwNet, step: int, args: argparse.Namespace, metrics: dict) -> None:
    torch.save({"model": model.state_dict(), "net": {}, "step": step, "vocabulary": TOKENS,
                "args": vars(args), "metrics": metrics}, path)


def overfit(args: argparse.Namespace, device: str) -> None:
    """Memorize a handful of clean samples. If this can't reach ~0 CER, something is wired
    wrong (labels, lengths, features), and there's no point training further."""
    data = FixedSamples(args.overfit, ConditionRanges(**STAGES["clean"]))
    loader = DataLoader(data, batch_size=min(args.batch, len(data)), shuffle=True, collate_fn=collate)
    model = CwNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(args.steps):
        for batch in loader:
            loss, _ = ctc_loss(model, batch, device)
            optimizer.zero_grad()
            loss.backward()
            # Early CTC gradients are huge; without clipping the model stalls on all-blank output.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        if step % 25 == 0 or step == args.steps - 1:
            model.eval()
            errors = length = 0
            with torch.no_grad():
                for batch in DataLoader(data, batch_size=32, collate_fn=collate):
                    lp = model(batch["spec"].to(device))
                    for p, r in zip(greedy_decode(lp, output_lengths(batch["frames"])), batch["texts"]):
                        e, n = token_errors(p, r)
                        errors, length = errors + e, length + n
            model.train()
            print(f"epoch {step:4d}  loss {loss.item():.4f}  CER {errors / max(1, length):.2%}", flush=True)
            if errors == 0:
                print("memorized every sample: the pipeline is wired correctly")
                return


def train(args: argparse.Namespace, device: str) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ranges = ConditionRanges(**STAGES[args.stage])
    stream = SyntheticStream(ranges, base_seed=args.seed, crop_prob=args.crop_prob)
    loader = DataLoader(BucketedBatches(stream, args.batch), batch_size=None, num_workers=args.workers,
                        persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers else None)
    model = CwNet().to(device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device)["model"])
        print(f"initialized from {args.init}")
    decay = [p for p in model.parameters() if p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.ndim <= 1]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": 1e-4},
                                   {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)
    val = ManifestSamples(args.val, limit=args.val_limit) if args.val and Path(args.val).exists() else None
    w1aw = ManifestSamples(args.w1aw, every=args.w1aw_every) if args.w1aw and Path(args.w1aw).exists() else None
    print(f"{count_parameters(model):,} parameters, stage {args.stage}, device {device}, "
          f"val {len(val) if val else 0}, w1aw {len(w1aw) if w1aw else 0}", flush=True)
    (out / "config.json").write_text(json.dumps({**vars(args), "ranges": dataclasses.asdict(ranges)}, indent=2))

    best, began, running = math.inf, time.monotonic(), 0.0
    log = open(out / "log.jsonl", "a")
    model.train()
    for step, batch in enumerate(loader):
        if step >= args.steps:
            break
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, args.steps, args.lr, args.warmup)
        loss, _ = ctc_loss(model, batch, device)
        if not torch.isfinite(loss):
            print(f"step {step}: non-finite loss, skipping batch", flush=True)
            continue
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        running = 0.98 * running + 0.02 * loss.item() if step else loss.item()

        if step % 100 == 0:
            rate = (step + 1) / (time.monotonic() - began)
            print(f"step {step:6d}  loss {running:.3f}  lr {optimizer.param_groups[0]['lr']:.1e}  "
                  f"{rate:.2f} steps/s", flush=True)
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            metrics = {"step": step + 1, "loss": running}
            for name, data in (("val", val), ("w1aw", w1aw)):
                if data is not None:
                    summary = evaluate.summarize(evaluate.run(model, data, device, workers=2))
                    evaluate.print_summary(f"[{step + 1}] {name}", summary)
                    metrics[name] = summary
            model.train()
            log.write(json.dumps(metrics) + "\n")
            log.flush()
            save(out / "last.pt", model, step + 1, args, metrics)
            score = metrics.get("val", {}).get("cer", running)
            if score < best:
                best = score
                save(out / "best.pt", model, step + 1, args, metrics)
                print(f"  new best: {score:.2%}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="runs/exp")
    parser.add_argument("--stage", choices=STAGES, default="full")
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--crop-prob", type=float, default=0.3)
    parser.add_argument("--init", help="start from this checkpoint")
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--val", default="data/synth-v1/val.jsonl")
    parser.add_argument("--val-limit", type=int, default=600)
    parser.add_argument("--w1aw", default="data/w1aw/clips.jsonl")
    parser.add_argument("--w1aw-every", type=int, default=8)
    parser.add_argument("--overfit", type=int, help="memorize this many samples instead of training")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()
    if args.overfit:
        overfit(args, args.device)
    else:
        train(args, args.device)


if __name__ == "__main__":
    main()
