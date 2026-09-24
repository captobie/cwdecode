"""Command line: render one sample, write a dataset, or preview one.

    python -m cwsynth render "CQ DE K1ABC <AR>" --wpm 22 --snr -3 --fist straight -o test.wav
    python -m cwsynth dataset --config configs/default.toml --split train --n 20000 --out data/synth-v1
    python -m cwsynth preview data/synth-v1 --split train --n 20 --png previews/
"""
import argparse
import dataclasses
import json
import sys
import tomllib
from pathlib import Path

import numpy as np

from cwsynth.alphabet import normalize
from cwsynth.corpus import SPLITS, CorpusMix
from cwsynth.dataset import read_wav, write_dataset, write_wav
from cwsynth.keying import FIST_STYLES, FistParams
from cwsynth.render import render
from cwsynth.spec import ConditionRanges, QsbParams, SampleSpec, sample_fist, sample_spec


def load_config(path: str | None) -> tuple[ConditionRanges, dict]:
    if not path:
        return ConditionRanges(), {}
    config = tomllib.loads(Path(path).read_text())
    return ConditionRanges.from_dict(config.get("conditions", {})), config.get("corpus", {})


def cmd_render(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    text = normalize(args.text)
    if args.random:
        ranges, _ = load_config(args.config)
        spec = sample_spec(rng, text, ranges)
        if args.fist:
            spec = dataclasses.replace(spec, fist=sample_fist(rng, args.fist))
    else:
        # Clean, machine-sent code unless a flag says otherwise.
        fist = sample_fist(rng, args.fist) if args.fist and args.fist != "machine" else FistParams()
        spec = SampleSpec(text=text, seed=args.seed, fist=fist)
    overrides = {k: v for k, v in {
        "wpm": args.wpm, "tone_hz": args.tone, "snr_db": args.snr, "drift_hz": args.drift,
        "farnsworth_wpm": args.farnsworth, "sample_rate": args.sample_rate,
    }.items() if v is not None}
    if args.qsb_depth is not None:
        overrides["qsb"] = QsbParams(rate_hz=args.qsb_rate, depth_db=args.qsb_depth)
    spec = dataclasses.replace(spec, **overrides)
    sample = render(spec)
    write_wav(Path(args.output), sample.audio, spec.sample_rate)
    print(f"{args.output}: {sample.duration_s:.2f} s, label {sample.text!r}")
    if args.verbose:
        print(json.dumps(spec.to_dict(), indent=2))


def cmd_dataset(args: argparse.Namespace) -> None:
    ranges, corpus_config = load_config(args.config)
    corpus = CorpusMix(corpus_config.get("weights"), args.text_file or corpus_config.get("text_file"))
    manifest = write_dataset(args.out, args.n, split=args.split, ranges=ranges, corpus=corpus,
                             seed=args.seed, start=args.start, workers=args.workers,
                             overwrite=args.overwrite)
    print(f"wrote {args.n} samples to {manifest}")


def cmd_preview(args: argparse.Namespace) -> None:
    root = Path(args.dataset)
    lines = (root / f"{args.split}.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in lines[args.skip:args.skip + args.n]]
    for row in rows:
        p = row["params"]
        snr = "clean" if row["snr_db"] is None else f"{row['snr_db']:+.1f}dB"
        extras = [name for name in ("qsb", "qrn", "rx_filter") if p[name]]
        extras += [f"qrm×{len(p['qrm'])}"] if p["qrm"] else []
        extras += ["drift"] if p["drift_hz"] else []
        print(f"{row['id']}  {row['duration_s']:5.1f}s  {p['wpm']:4.1f}wpm  {p['fist']['style']:8s} "
              f"{p['tone_hz']:4.0f}Hz  {snr:>8s}  {row['source']:12s} {' '.join(extras):24s} {row['text']!r}")
    if args.png:
        _write_spectrograms(root, rows, Path(args.png))


def _write_spectrograms(root: Path, rows: list[dict], out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy import signal
    except ImportError:
        sys.exit("spectrogram previews need matplotlib: pip install -e '.[preview]'")
    out.mkdir(parents=True, exist_ok=True)
    for row in rows:
        audio, sr = read_wav(root / row["audio"])
        f, t, power = signal.spectrogram(audio, sr, nperseg=256, noverlap=224)
        band = (f >= 200) & (f <= 1300)
        fig, ax = plt.subplots(figsize=(max(6.0, row["duration_s"] * 1.2), 3.0))
        ax.pcolormesh(t, f[band], 10 * np.log10(power[band] + 1e-12), shading="auto", cmap="magma")
        for token, start, end in row["alignment"]:
            ax.axvspan(start, end, ymin=0, ymax=0.04, color="cyan", alpha=0.8)
            ax.text((start + end) / 2, 230, token.strip("<>"), color="cyan", ha="center", fontsize=7)
        p = row["params"]
        snr = "clean" if row["snr_db"] is None else f"{row['snr_db']:+.1f} dB"
        ax.set_title(f"{row['id']}  {row['text']!r}\n{p['wpm']:.1f} WPM {p['fist']['style']}, {snr}",
                     fontsize=8, loc="left")
        ax.set_xlabel("s")
        ax.set_ylabel("Hz")
        fig.tight_layout()
        fig.savefig(out / f"{row['id']}.png", dpi=110)
        plt.close(fig)
    print(f"spectrograms in {out}/")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="cwsynth", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    r = sub.add_parser("render", help="render one text as a WAV")
    r.add_argument("text")
    r.add_argument("-o", "--output", default="cw.wav")
    r.add_argument("--random", action="store_true", help="draw all conditions at random first")
    r.add_argument("--config")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--wpm", type=float)
    r.add_argument("--farnsworth", type=float, help="overall WPM with Farnsworth spacing")
    r.add_argument("--fist", choices=FIST_STYLES)
    r.add_argument("--tone", type=float, help="Hz")
    r.add_argument("--drift", type=float, help="total drift in Hz")
    r.add_argument("--snr", type=float, help="dB in 500 Hz")
    r.add_argument("--qsb-depth", type=float, help="dB")
    r.add_argument("--qsb-rate", type=float, default=0.2, help="Hz")
    r.add_argument("--sample-rate", type=int)
    r.add_argument("-v", "--verbose", action="store_true")
    r.set_defaults(func=cmd_render)

    d = sub.add_parser("dataset", help="write a labeled dataset split")
    d.add_argument("--out", required=True)
    d.add_argument("--n", type=int, required=True)
    d.add_argument("--split", choices=SPLITS, default="train")
    d.add_argument("--config")
    d.add_argument("--text-file", help="plain text to use as the English source")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--start", type=int, default=0, help="first index; >0 extends an existing split")
    d.add_argument("--workers", type=int)
    d.add_argument("--overwrite", action="store_true")
    d.set_defaults(func=cmd_dataset)

    p = sub.add_parser("preview", help="print samples and optionally write spectrogram PNGs")
    p.add_argument("dataset")
    p.add_argument("--split", choices=SPLITS, default="train")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--skip", type=int, default=0)
    p.add_argument("--png", help="folder for spectrogram images")
    p.set_defaults(func=cmd_preview)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
