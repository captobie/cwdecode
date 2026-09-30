# CWDecode

A macOS app that decodes Morse code (CW) from live audio or recordings.

## Using it

- **Listen** (⌘L) decodes the input chosen in the toolbar's input menu: your radio's USB audio, a virtual device like Loopback, or System Default. The choice is remembered, and you can switch while listening.
- **Open Audio File** (⌘O) decodes a WAV, AIFF, MP3 or M4A recording.
- **Decoder** (toolbar) picks how audio becomes text:
  - **Neural** (default): a small neural network trained on simulated CW with noise, fading, static and interference. It copies weak and hand-sent signals the classic decoder can't, and stays silent on noise instead of printing stray `E`s and `T`s. Text appears about 3½ s behind the audio; the newest few characters are shown dimmed until they're final.
  - **Classic**: tone threshold and timing rules. Text appears as each character ends.
- **Auto-tune** follows the strongest tone between 300 and 1200 Hz. Turn it off to set the tone by hand.
- **Squelch** (Classic only) is the minimum signal-to-noise ratio (in the detector's ~170 Hz bandwidth) before anything decodes. Lower it for weak signals, raise it if noise turns into stray `E`s and `T`s.

Speed (WPM) is detected automatically, including Farnsworth-spaced code. Auto-tune, the tone, SNR and WPM readouts come from the classic detector with either decoder.

## Building

Runs on macOS 15 or later. Built and tested with Xcode 27 (Swift 6). The Xcode project is generated from `project.yml`:

```sh
xcodegen generate
open CWDecode.xcodeproj
```

Run `xcodegen generate` again after adding or removing files. To sign with your own team, set `DEVELOPMENT_TEAM` in `project.yml` (it currently signs to run locally).

Tests: `xcodebuild -project CWDecode.xcodeproj -scheme CWDecode test`

The neural decoder's model (`CWDecode/Model/CWNet.mlpackage`) is committed, so building needs no Python. It's trained and exported from `ml/` (see [`ml/README.md`](ml/README.md)); the export also rewrites the golden files in `CWDecodeTests/Resources/Neural/` that hold the Swift front end, model and streaming decoder to the Python reference.

## How it works

Neural decoder:

```
audio ─▶ AudioResampler ─▶ NeuralFeatures ─▶ CWNet (Core ML) ─▶ StreamingCTCDecoder ─▶ text
          (to 8 kHz)        (spectrogram,      (CNN + CTC,       (6 s windows every 1 s;
                             250–1250 Hz)       0.5M params)      handoffs in gaps between tokens)
```

Classic decoder (also drives the meters with either decoder):

```
audio ─▶ ToneDetector ─▶ adaptive threshold ─▶ MorseDecoder ─▶ text
          (Goertzel,      (noise floor/peak,     (dit/dah and gap
           12 ms window)   hysteresis, debounce)  clustering)
   └──▶ FrequencyTracker (auto-tune, 5 Hz steps every 250 ms)
```

| Path | Role |
| --- | --- |
| `CWDecode/Neural/NeuralPipeline.swift` | Neural decoder: resampling, windowed decoding, meters from the classic detector |
| `CWDecode/Neural/NeuralFeatures.swift` | Spectrogram, identical to `ml/cwmodel/features.py` |
| `CWDecode/Neural/CWNetModel.swift` | Loads the Core ML model, checks it matches the front end, runs a window |
| `CWDecode/Neural/StreamingCTCDecoder.swift` | Overlapping windows → committed and tentative text; port of `ml/cwmodel/stream.py` |
| `CWDecode/Neural/AudioResampler.swift` | Any input rate → 8 kHz |
| `CWDecode/DSP/DecoderPipeline.swift` | Classic decoder: audio → key up/down → text; noise floor, squelch, debounce. Also defines the `DecoderEngine` protocol |
| `CWDecode/DSP/ToneDetector.swift` | Tone power over 12 ms windows every 4 ms |
| `CWDecode/DSP/FrequencyTracker.swift` | Finds the CW tone for auto-tune |
| `CWDecode/Morse/MorseDecoder.swift` | Timing → characters, adaptive speed |
| `CWDecode/DSP/PipelineRunner.swift` | Runs the selected decoder off the main thread |
| `CWDecode/Audio/` | Input device list, audio input, and audio file reading |
| `CWDecode/App/DecoderViewModel.swift` | UI state and actions |
