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

Tests: `swift test` for the decoders (`CWKit`), `xcodebuild -project CWDecode.xcodeproj -scheme CWDecode test` for the app.

The neural decoder's model (`Sources/CWKit/Resources/CWNet.mlmodelc`, compiled, since SwiftPM can't compile an `.mlpackage` resource) is committed, so building needs no Python. It's trained and exported from `ml/` (see [`ml/README.md`](ml/README.md)); the export also rewrites the golden files in `Tests/CWKitTests/Resources/Neural/` that hold the Swift front end, model and streaming decoder to the Python reference.

## CWKit

The decoders live in a Swift package, `CWKit` (`Package.swift` at the repo root), with no app or audio-device code, so other apps can feed it their own audio. The CWDecode app is its first client; [FTX1Remote](https://github.com/captobie/ftx1-remote) uses it for its CW window. Its API is `PipelineRunner` (submit `[Float]` chunks at any sample rate from any thread; text and meters come back as `PipelineOutput` events), `PipelineSettings`, `DecoderKind`, `CWNetModel` and `AudioFileReader`. Apps depend on a tagged version, so tag a release when a decoder change or a new model should reach them.

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
| `Sources/CWKit/Neural/NeuralPipeline.swift` | Neural decoder: resampling, windowed decoding, meters from the classic detector |
| `Sources/CWKit/Neural/NeuralFeatures.swift` | Spectrogram, identical to `ml/cwmodel/features.py` |
| `Sources/CWKit/Neural/CWNetModel.swift` | Loads the Core ML model, checks it matches the front end, runs a window |
| `Sources/CWKit/Neural/StreamingCTCDecoder.swift` | Overlapping windows → committed and tentative text; port of `ml/cwmodel/stream.py` |
| `Sources/CWKit/Neural/AudioResampler.swift` | Any input rate → 8 kHz |
| `Sources/CWKit/DSP/DecoderPipeline.swift` | Classic decoder: audio → key up/down → text; noise floor, squelch, debounce. Also defines the `DecoderEngine` protocol |
| `Sources/CWKit/DSP/ToneDetector.swift` | Tone power over 12 ms windows every 4 ms |
| `Sources/CWKit/DSP/FrequencyTracker.swift` | Finds the CW tone for auto-tune |
| `Sources/CWKit/Morse/MorseDecoder.swift` | Timing → characters, adaptive speed |
| `Sources/CWKit/DSP/PipelineRunner.swift` | Runs the selected decoder off the main thread |
| `Sources/CWKit/Audio/AudioFileReader.swift` | Audio file reading |
| `CWDecode/Audio/` | Input device list and audio input (app only) |
| `CWDecode/App/DecoderViewModel.swift` | UI state and actions |

## License

MIT; see [`LICENSE`](LICENSE).
