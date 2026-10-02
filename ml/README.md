# ml: CWDecode's neural decoder

Everything behind CWDecode's neural decoder: `cwsynth` makes synthetic Morse audio paired with
its exact text, and `cwmodel` trains, evaluates and exports a spectrogram → CNN → CTC model.
The approach follows [DeepFist](https://github.com/n9bc/DeepFist). The code is written from
scratch (DeepFist is GPL-3.0), and the ham vocabulary is ported from HamLexicon on the abandoned
`smart-cleanup` branch.

## Setup

```sh
cd ml
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest
```

## Usage

```sh
# One clip: clean machine code unless flags add impairments (--random draws them all)
.venv/bin/python -m cwsynth render "CQ DE K1ABC <AR>" --wpm 22 --snr -3 --fist straight -o test.wav

# A dataset split (parallel; identical output for any worker count)
.venv/bin/python -m cwsynth dataset --config configs/default.toml --split train --n 20000 --out data/synth-v1
.venv/bin/python -m cwsynth dataset --config configs/default.toml --split val --n 1000 --out data/synth-v1

# Look at it: one line per sample, plus spectrograms with the alignment marked
.venv/bin/python -m cwsynth preview data/synth-v1 --n 20 --png data/synth-v1/png
```

Roughly 560 samples/s on an 8-core Mac and ~185 MB per 1,000 samples (median clip is 11 s).
For training, `cwsynth.generate(seed)` produces the same samples in memory, so an on-the-fly
dataset needs no disk at all.

## Labels

- **Vocabulary:** `A–Z 0–9 . , ? / - @ '`, `<AR> <BT> <KN> <SK> <AS>`, and space. CTC blank
  is ID 0 (`cwsynth.alphabet.TOKENS`). Sounds that are identical share one token: `+ = ( &`
  normalize to `<AR> <BT> <KN> <AS>`. Append new tokens only at the end.
- **Consistency:** hand-sent timing is clamped so a gap inside a character, a gap between
  characters and a gap between words never overlap. A space in a label is always an audible
  word gap (`tests/test_keying.py` checks this for every fist style).
- **Noise-only samples:** 5% of samples have an empty label (noise, static, or an unkeyed
  carrier), so the model learns that noise is not E's and T's.
- **What gets transcribed:** only the strongest signal. Interfering stations are always weaker.

## SNR

Carrier power while the key is down, relative to noise in **500 Hz**, measured before any
receiver filter. Each manifest row also records it in 2,500 Hz (`snr_db_2500hz`, 7 dB lower)
and in the app's ~170 Hz detector bandwidth (`snr_db_app`, 4.7 dB higher).

## Manifest

`{split}.jsonl`, one JSON object per line:

| Field | Meaning |
| --- | --- |
| `id`, `audio` | Sample ID and WAV path relative to the dataset root |
| `text`, `n_tokens` | CTC target (canonical) and its length in tokens |
| `duration_s`, `sample_rate` | Clip length; 8000 Hz by default |
| `source` | Corpus source: callsign, qso, abbreviation, english, random, noise |
| `snr_db`, `snr_db_2500hz`, `snr_db_app` | See above; `null` for clean samples |
| `params` | The full `SampleSpec`: `SampleSpec.from_dict(row["params"])` re-renders the sample exactly |
| `alignment` | `[token, start_s, end_s]` for every keyed token |

`dataset.json` holds the vocabulary, generator version, git commit, and each split's seed,
conditions and corpus weights.

## W1AW evaluation library

ARRL's W1AW code-practice recordings (5–40 WPM, 750 Hz, machine-sent; 15 WPM characters with
Farnsworth spacing at 15 WPM and below) turned into labeled clips for **evaluation only**. ARRL
prohibits reproducing its material without permission: the files stay in the git-ignored
`data/w1aw/`, aren't committed or redistributed, and aren't used for training.

```sh
.venv/bin/python -m cwsynth.w1aw index                   # list archived sessions (HTML only)
.venv/bin/python -m cwsynth.w1aw fetch --per-speed 3     # MP3 + transcript pairs, spread over the years
.venv/bin/python -m cwsynth.w1aw build                   # → clips.jsonl, sessions.jsonl
.venv/bin/python -m cwsynth.w1aw degrade --snr -6 0 6 12 # → degraded.jsonl (add --impair for QSB/QRN/filters)
```

- **Labels are checked two ways.** A threshold decoder (`classic.py`, exact on clean
  machine-sent code) reads the audio, and the result is aligned to ARRL's transcript. A clip is
  kept only if the two agree token for token, word gaps included. Clips are 3–15 s and cut at
  word gaps.
- **Transcript bytes** are mapped to what is actually keyed, each confirmed against the audio:
  `=` and `0x89` → BT, `0x82` → AR, `0x83` → AS, `Ø` or `0` plus a stray `0x98` → 0. The trailing
  `<` and `^Z` aren't keyed.
- **`sessions.jsonl`** keeps each full recording with its whole transcript, for testing
  windowed decoding end to end.
- **`degraded.jsonl`** has every clip at fixed SNRs, measured against the recording's own carrier
  level, so the SNR definition matches cwsynth's.

The first build (3 sessions per speed from 2010/2013, 2018 and 2026): 2,136 clips, 6.0 h,
97–100% of words kept at 10 WPM and up. 5 and 7.5 WPM keep 88–100%, because at those speeds a
single long word can exceed the 15 s clip limit. Disk: about 220 MB of MP3s, 360 MB of WAVs,
340 MB of clips and 1.3 GB of degraded copies (all regenerable).

## The model (`cwmodel`)

Spectrogram → CNN → CTC, trained on cwsynth samples generated on the fly
(`python -m cwmodel.train`; stages `clean` then `full`). `python -m cwmodel.evaluate` reports
error rates by SNR, speed, fist and source; `python -m cwmodel.stream` decodes whole W1AW
sessions the way the app does.

Exporting for the app needs coremltools, whose native parts aren't built for Python 3.14, so it
runs in a separate environment pinned to the torch version coremltools is tested against:

```sh
/opt/homebrew/bin/python3.13 -m venv .venv-export
.venv-export/bin/pip install "torch==2.7.0" coremltools numpy scipy
.venv-export/bin/pip install -e . --no-deps
.venv-export/bin/python -m cwmodel.export --ckpt runs/full/best.pt
```

That writes `Sources/CWKit/Resources/CWNet.mlmodelc`, compiled with Xcode's `coremlcompiler`
(vocabulary and feature settings in its metadata; the app refuses a model that doesn't match its
front end), and the golden files in `Tests/CWKitTests/Resources/Neural/`. Commit a new model only
when it should ship, and tag a CWKit release so apps that depend on it pick it up.

First model (30k steps full stage after 6k clean; 528k parameters):

| Test | CER |
| --- | --- |
| W1AW clips, clean | 0.54% |
| W1AW whole sessions, streamed like the app | 1.19% (0–1% at 13 WPM and up) |
| W1AW sessions at +3 dB | 3.35% |
| W1AW clips with white noise at +12 / +6 / 0 / −6 dB | 0.8% / 1.2% / 14% / 98% |
| Noise-only synthetic samples | 0.00 invented characters per sample |

Known weak spots: hand-sent code (straight key and bug 2.5–3× the machine error rate at good
SNR), SNR below 0 dB, deep fading, and missing word spaces in extreme Farnsworth spacing
(W1AW 5 WPM: 17% session CER, almost all missing spaces).

## Modules

| File | Role |
| --- | --- |
| `alphabet.py` | Vocabulary, normalization, token IDs |
| `keying.py` | Text → element timing with a human fist (machine, paddle, bug, straight key) → envelope |
| `tone.py` | Carrier, drift, chirp |
| `channel.py` | Noise at a defined SNR, QSB, static crashes, receiver filter |
| `spec.py` | `ConditionRanges` (distributions) → `SampleSpec` (one sample's conditions) |
| `corpus.py` | Text sources and their mix; callsigns are split into train/val/test by hash |
| `render.py` | `render(spec)` and `generate(seed)` |
| `dataset.py` | Parallel batch writer and manifest |
| `classic.py` | Threshold decoder for clean machine-sent audio, used to label recordings |
| `w1aw.py` | W1AW index, download, alignment, clips and degraded copies |
| `cwmodel/features.py` | Spectrogram (the Swift `NeuralFeatures` matches it) |
| `cwmodel/net.py` | The network |
| `cwmodel/data.py` | On-the-fly synthetic batches (length-bucketed), manifest readers |
| `cwmodel/train.py`, `evaluate.py` | Training stages and error-rate breakdowns |
| `cwmodel/stream.py` | Windowed streaming decoding (the Swift `StreamingCTCDecoder` matches it) |
| `cwmodel/export.py` | Core ML export and golden test files |
