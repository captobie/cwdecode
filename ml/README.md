# cwsynth: synthetic CW training data

The first piece of CWDecode's neural decoder: synthetic Morse audio paired with its exact text,
for training a spectrogram → CNN → CTC model. The approach follows
[DeepFist](https://github.com/n9bc/DeepFist). The code is written from scratch (DeepFist is
GPL-3.0), and the ham vocabulary is ported from HamLexicon on the abandoned `smart-cleanup` branch.

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

- **Vocabulary:** `A–Z 0–9 . , ? / - @ '`, `<AR> <BT> <KN> <SK>`, and space. CTC blank is
  ID 0 (`cwsynth.alphabet.TOKENS`). Sounds that are identical share one token: `+ = (`
  normalize to `<AR> <BT> <KN>`. Append new tokens only at the end.
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
