"""A labeled evaluation library from ARRL's W1AW code-practice recordings.

    python -m cwsynth.w1aw index                    # list every archived session (HTML pages only)
    python -m cwsynth.w1aw fetch --per-speed 3      # download MP3 + transcript pairs
    python -m cwsynth.w1aw build                    # decode, align, cut labeled clips
    python -m cwsynth.w1aw degrade --snr -6 0 6 12  # noisy copies at fixed SNRs

Labels are verified by two independent sources: a clip is kept only if the threshold decoder's
reading of the audio matches ARRL's transcript exactly, token for token, including word gaps.

ARRL prohibits reproducing its material without permission. The files stay under the
git-ignored data folder and are used only for evaluation; don't commit or redistribute them.
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import time
import urllib.request
import zlib
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from cwsynth import channel, classic
from cwsynth.alphabet import ALIASES, PATTERNS, SPACE
from cwsynth.dataset import read_wav, write_wav
from cwsynth.spec import ConditionRanges

BASE_URL = "https://www.arrl.org"
SPEEDS = {"5": 5.0, "7-5": 7.5, "10": 10.0, "13": 13.0, "15": 15.0, "18": 18.0,
          "20": 20.0, "25": 25.0, "30": 30.0, "35": 35.0, "40": 40.0}
USER_AGENT = "cwsynth/0.1 (CWDecode evaluation; personal use)"
# Pages mix relative, http:// and https:// links.
_LINK = re.compile(r'href="(?:https?://www\.arrl\.org)?(/files/file/Morse/Archive/[^"]+?\.(mp3|txt))"',
                   re.IGNORECASE)
_PIECE = re.compile(r"<[A-Z]{2}>|\s+|.")


# ----- Index and download -----------------------------------------------------------------

def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def parse_archive_page(html: str, wpm: float) -> list[dict]:
    """Pair each session's MP3 and transcript by their YYMMDD date prefix."""
    by_date: dict[str, dict] = {}
    for path, kind in _LINK.findall(html):
        date = re.match(r"(\d{6})", path.rsplit("/", 1)[-1])
        if date:
            by_date.setdefault(date.group(1), {})[kind.lower()] = path
    return [{"date": d, "wpm": wpm, "mp3": f["mp3"], "txt": f["txt"]}
            for d, f in sorted(by_date.items()) if "mp3" in f and "txt" in f]


def session_name(entry: dict) -> str:
    return f"{entry['date']}_{entry['wpm']:g}wpm"


def build_index(root: Path) -> list[dict]:
    sessions = []
    for slug, wpm in SPEEDS.items():
        html = _get(f"{BASE_URL}/{slug}-wpm-code-archive").decode("utf-8", errors="replace")
        found = parse_archive_page(html, wpm)
        print(f"{wpm:g} WPM: {len(found)} sessions")
        sessions += found
        time.sleep(1.0)
    root.mkdir(parents=True, exist_ok=True)
    (root / "index.json").write_text(json.dumps(sessions, indent=1) + "\n")
    return sessions


def choose(sessions: list[dict], per_speed: int) -> list[dict]:
    """`per_speed` sessions per speed, evenly spread from oldest to newest."""
    chosen = []
    for wpm in SPEEDS.values():
        pool = sorted((s for s in sessions if s["wpm"] == wpm), key=lambda s: s["date"])
        if pool:
            picks = sorted(set(np.linspace(0, len(pool) - 1, per_speed).round().astype(int)))
            chosen += [pool[i] for i in picks]
    return chosen


def fetch(root: Path, sessions: list[dict]) -> None:
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    log_path = root / "downloads.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {}
    for entry in sessions:
        for kind in ("mp3", "txt"):
            name = f"{session_name(entry)}.{kind}"
            target = raw / name
            downloaded = not (target.exists() and target.stat().st_size > 0)
            if downloaded:
                data = _get(BASE_URL + entry[kind])
                target.with_suffix(".part").write_bytes(data)
                target.with_suffix(".part").rename(target)
                print(f"  {name}  {len(data) / 1e6:.2f} MB")
                time.sleep(1.0)
            elif name in log:
                continue
            else:
                data = target.read_bytes()
            log[name] = {"url": BASE_URL + entry[kind], "bytes": len(data),
                         "sha256": hashlib.sha256(data).hexdigest(), "session": entry}
    log_path.write_text(json.dumps(log, indent=1) + "\n")


# ----- Transcripts, alignment, clips ------------------------------------------------------

# Bytes ARRL's transcripts use for prosigns and symbols, each confirmed against the audio:
# older files key 0x89 as BT, 0x82 as AR and 0x83 as AS; 2026 files write the slashed zero
# in JW0X as UTF-8 "Ø" or as "0" plus a stray 0x98, and key it as a plain 0.
TRANSCRIPT_BYTES = [(b"\xc3\x98", b"0"), (b"\x98", b""), (b"\x89", b"<BT>"),
                    (b"\x82", b"<AR>"), (b"\x83", b"<AS>")]


def transcript_tokens(raw: bytes) -> list[str]:
    """ARRL's transcript as label tokens. `=` is BT, as sent; the end-of-file `<` and control
    characters are not keyed and are dropped. Characters outside the vocabulary become
    `{c}` tokens, which match nothing, so no clip containing one is ever kept."""
    for pattern, replacement in TRANSCRIPT_BYTES:
        raw = raw.replace(pattern, replacement)
    text = raw.decode("latin-1").upper()
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
    text = re.sub(r"<(?![A-Z]{2}>)|(?<!<[A-Z]{2})>", " ", text)
    tokens: list[str] = []
    for piece in _PIECE.findall(text):
        if piece.isspace():
            if tokens and tokens[-1] != SPACE:
                tokens.append(SPACE)
            continue
        token = ALIASES.get(piece, piece)
        tokens.append(token if token in PATTERNS else "{" + piece + "}")
    while tokens and tokens[-1] == SPACE:
        tokens.pop()
    return tokens


def convert_mp3(mp3: Path, wav: Path, sample_rate: int = 8000) -> None:
    wav.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("afconvert"):
        cmd = ["afconvert", "-f", "WAVE", "-d", f"LEI16@{sample_rate}", "-c", "1", str(mp3), str(wav)]
    else:
        cmd = ["ffmpeg", "-loglevel", "error", "-y", "-i", str(mp3), "-ac", "1",
               "-ar", str(sample_rate), "-sample_fmt", "s16", str(wav)]
    subprocess.run(cmd, check=True)


def align(decoded: list[str], reference: list[str]) -> list[int | None]:
    """For each reference token, the index of the decoded token it matches, or None."""
    mapping: list[int | None] = [None] * len(reference)
    matcher = SequenceMatcher(None, decoded, reference, autojunk=False)
    for a, b, size in matcher.get_matching_blocks():
        for k in range(size):
            mapping[b + k] = a + k
    return mapping


def mismatch_rate(decoded: list[str], reference: list[str]) -> float:
    """Approximate token error rate of the decoder against the transcript."""
    matcher = SequenceMatcher(None, decoded, reference, autojunk=False)
    errors = sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in matcher.get_opcodes() if op != "equal")
    return errors / max(1, len(reference))


def cut_clips(tokens: list[tuple[str, float, float]], reference: list[str], mapping: list[int | None],
              duration_s: float, rng: np.random.Generator, min_s: float = 3.0,
              max_s: float = 15.0) -> list[tuple[int, int, float, float]]:
    """Word-aligned clips whose tokens are all verified. Returns (first token, last token,
    start_s, end_s) with some silence either side, never reaching into a neighbor."""
    words, start = [], None
    for i, token in enumerate(reference + [SPACE]):
        if token != SPACE and start is None:
            start = i
        elif token == SPACE and start is not None:
            words.append((start, i - 1))
            start = None

    def verified(i0: int, i1: int) -> bool:
        m = mapping[i0:i1 + 1]
        return all(x is not None for x in m) and m[-1] - m[0] == i1 - i0

    def span(i0: int, i1: int) -> tuple[float, float]:
        return tokens[mapping[i0]][1], tokens[mapping[i1]][2]

    def length(i0: int, i1: int) -> float:
        s, e = span(i0, i1)
        return e - s

    def padded(i0: int, i1: int) -> tuple[float, float]:
        """Up to 0.5 s of silence either side, but at most half the gap to the nearest
        keyed character, so no neighbor's energy gets in."""
        s, e = span(i0, i1)
        k0, k1 = mapping[i0] - 1, mapping[i1] + 1
        while k0 >= 0 and tokens[k0][0] == SPACE:
            k0 -= 1
        while k1 < len(tokens) and tokens[k1][0] == SPACE:
            k1 += 1
        before = s - (tokens[k0][2] if k0 >= 0 else 0.0)
        after = (tokens[k1][1] if k1 < len(tokens) else duration_s) - e
        return max(0.0, s - min(0.5, before / 2)), min(duration_s, e + min(0.5, after / 2))

    clips, w = [], 0
    while w < len(words):
        target, k = rng.uniform(min_s, max_s), w
        while k < len(words):
            i0, i1 = words[w][0], words[k][1]
            if not verified(i0, i1) or length(i0, i1) > max_s - 0.5:
                break
            k += 1
            if length(i0, i1) >= target:
                break
        if k > w:
            s, e = padded(words[w][0], words[k - 1][1])
            if e - s >= min_s:
                clips.append((words[w][0], words[k - 1][1], s, e))
            w = k
        else:
            w += 1
    return clips


def build(root: Path, min_s: float = 3.0, max_s: float = 15.0) -> None:
    log = json.loads((root / "downloads.json").read_text())
    names = sorted({Path(n).stem for n in log})
    clip_rows, session_rows = [], []
    for name in names:
        entry = log[f"{name}.mp3"]["session"]
        wav = root / "wav" / f"{name}.wav"
        if not wav.exists():
            convert_mp3(root / "raw" / f"{name}.mp3", wav)
        audio, sr = read_wav(wav)
        reference = transcript_tokens((root / "raw" / f"{name}.txt").read_bytes())
        dec = classic.decode(audio, sr)
        decoded = [t for t, _, _ in dec.tokens]
        mapping = align(decoded, reference)
        rng = np.random.default_rng(zlib.crc32(name.encode()))
        clips = cut_clips(dec.tokens, reference, mapping, len(audio) / sr, rng, min_s, max_s)

        for n, (i0, i1, s, e) in enumerate(clips):
            clip_id = f"w1aw-{name}-{n:04d}"
            path = Path("clips") / name / f"{clip_id}.wav"
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            write_wav(root / path, audio[round(s * sr):round(e * sr)], sr)
            text = "".join(reference[i0:i1 + 1])
            clip_rows.append({
                "id": clip_id, "audio": path.as_posix(), "text": text,
                "n_tokens": i1 - i0 + 1, "duration_s": round(e - s, 4), "sample_rate": sr,
                "split": "test", "source": "w1aw", "snr_db": None,
                "params": {"session": name, "date": entry["date"], "wpm": entry["wpm"],
                           "char_wpm": round(1.2 / dec.dot_s, 2), "dot_s": round(dec.dot_s, 5),
                           "gap_unit_s": round(dec.gap_unit_s, 5),
                           "farnsworth": dec.gap_unit_s > 1.2 * dec.dot_s,
                           "tone_hz": dec.tone_hz, "offset_s": round(s, 4),
                           "carrier_amplitude": round(dec.carrier_amplitude, 5)},
                "alignment": [[reference[i], round(dec.tokens[mapping[i]][1] - s, 4),
                               round(dec.tokens[mapping[i]][2] - s, 4)]
                              for i in range(i0, i1 + 1) if reference[i] != SPACE],
            })

        words = sum(1 for i, t in enumerate(reference) if t != SPACE and (i == 0 or reference[i - 1] == SPACE))
        covered = sum(r["text"].count(" ") + 1 for r in clip_rows if r["params"]["session"] == name)
        unsupported = sorted({t[1:-1] for t in reference if t.startswith("{")})
        session_rows.append({
            "id": name, "audio": f"wav/{name}.wav", "date": entry["date"], "wpm": entry["wpm"],
            "duration_s": round(len(audio) / sr, 2),
            "text": re.sub(" +", " ", "".join(t for t in reference if not t.startswith("{"))).strip(),
            "unsupported_chars": unsupported, "decoder_mismatch_rate": round(mismatch_rate(decoded, reference), 4),
            "clips": len(clips), "words": words, "words_in_clips": covered,
            "char_wpm": round(1.2 / dec.dot_s, 2), "tone_hz": dec.tone_hz,
        })
        print(f"{name:16s} {len(clips):3d} clips  {covered / max(1, words):5.1%} of {words} words  "
              f"decoder mismatch {session_rows[-1]['decoder_mismatch_rate']:.2%}  "
              f"char {1.2 / dec.dot_s:4.1f} WPM  unsupported {''.join(unsupported) or '-'}")

    _write_jsonl(root / "clips.jsonl", clip_rows)
    _write_jsonl(root / "sessions.jsonl", session_rows)
    hours = sum(r["duration_s"] for r in clip_rows) / 3600
    print(f"{len(clip_rows)} clips ({hours:.2f} h) from {len(session_rows)} sessions")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows))


# ----- Degraded copies ---------------------------------------------------------------------

def degrade(root: Path, snrs: list[float], impair: bool, seed: int = 0) -> None:
    """Each clip at each SNR (500 Hz reference, against the recording's measured carrier),
    optionally with fading, static and receiver filters drawn like cwsynth's."""
    r = ConditionRanges()
    rows = [json.loads(line) for line in (root / "clips.jsonl").read_text().splitlines()]
    out_rows = []
    for row in rows:
        audio, sr = read_wav(root / row["audio"])
        base = audio.astype(np.float64) / row["params"]["carrier_amplitude"]
        for k, snr in enumerate(snrs):
            rng = np.random.default_rng([seed, zlib.crc32(row["id"].encode()), k])
            n, x, extra = len(base), base.copy(), {}
            if impair and rng.random() < r.qsb_prob:
                extra["qsb"] = {"rate_hz": float(np.exp(rng.uniform(*np.log(r.qsb_rate_hz)))),
                                "depth_db": float(rng.uniform(*r.qsb_depth_db))}
                x *= channel.qsb_gain(n, sr, extra["qsb"]["rate_hz"], extra["qsb"]["depth_db"], rng)
            x += channel.white_noise(n, snr, sr, rng)
            if impair and rng.random() < r.qrn_prob:
                extra["qrn"] = {"rate_per_s": float(rng.uniform(*r.qrn_rate_per_s)),
                                "level_db": float(rng.uniform(*r.qrn_level_db))}
                x += channel.static_crashes(n, sr, extra["qrn"]["rate_per_s"], extra["qrn"]["level_db"], rng)
            if impair and rng.random() < r.rx_filter_prob:
                tone = row["params"]["tone_hz"]
                width = float(rng.uniform(*r.cw_filter_width_hz))
                extra["rx_filter"] = {"low_hz": tone - width / 2, "high_hz": tone + width / 2, "order": 3}
                x = channel.receiver_filter(x, sr, tone - width / 2, tone + width / 2, 3)
            x *= float(rng.uniform(*r.gain_peak)) / max(1e-9, float(np.max(np.abs(x))))

            tag = f"snr{snr:+03.0f}" + ("i" if impair else "")
            clip_id = f"{row['id']}-{tag}"
            path = Path("degraded") / tag / row["params"]["session"] / f"{clip_id}.wav"
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            write_wav(root / path, x.astype(np.float32), sr)
            out_rows.append({**row, "id": clip_id, "audio": path.as_posix(), "base_id": row["id"],
                             "snr_db": snr, "snr_db_2500hz": round(channel.snr_in_bandwidth(snr, 2500), 2),
                             "params": {**row["params"], **extra, "impaired": impair}})
    name = "degraded" + ("_impaired" if impair else "") + ".jsonl"
    _write_jsonl(root / name, out_rows)
    print(f"{len(out_rows)} clips in {root / name}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m cwsynth.w1aw", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="data/w1aw")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("index")
    f = sub.add_parser("fetch")
    f.add_argument("--per-speed", type=int, default=3)
    b = sub.add_parser("build")
    b.add_argument("--min-s", type=float, default=3.0)
    b.add_argument("--max-s", type=float, default=15.0)
    d = sub.add_parser("degrade")
    d.add_argument("--snr", type=float, nargs="+", default=[-6.0, 0.0, 6.0, 12.0])
    d.add_argument("--impair", action="store_true", help="also add fading, static and filters")
    d.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    root = Path(args.root)
    if args.command == "index":
        build_index(root)
    elif args.command == "fetch":
        sessions = json.loads((root / "index.json").read_text())
        fetch(root, choose(sessions, args.per_speed))
    elif args.command == "build":
        build(root, args.min_s, args.max_s)
    else:
        degrade(root, args.snr, args.impair, args.seed)


if __name__ == "__main__":
    main()
