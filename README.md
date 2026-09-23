# CWDecode

A macOS app that decodes Morse code (CW) from live audio or recordings.

## Using it

- **Listen** (⌘L) decodes the input chosen in the toolbar's input menu: your radio's USB audio, a virtual device like Loopback, or System Default. The choice is remembered, and you can switch while listening.
- **Open Audio File** (⌘O) decodes a WAV, AIFF, MP3 or M4A recording.
- **Auto-tune** follows the strongest tone between 300 and 1200 Hz. Turn it off to set the tone by hand.
- **Squelch** is the minimum signal-to-noise ratio (in the detector's ~170 Hz bandwidth) before anything decodes. Lower it for weak signals, raise it if noise turns into stray `E`s and `T`s.

Speed (WPM) is detected automatically, including Farnsworth-spaced code.

## Building

Runs on macOS 15 or later. Built and tested with Xcode 27 (Swift 6). The Xcode project is generated from `project.yml`:

```sh
xcodegen generate
open CWDecode.xcodeproj
```

Run `xcodegen generate` again after adding or removing files. To sign with your own team, set `DEVELOPMENT_TEAM` in `project.yml` (it currently signs to run locally).

Tests: `xcodebuild -project CWDecode.xcodeproj -scheme CWDecode test`

## How it works

```
audio ─▶ ToneDetector ─▶ adaptive threshold ─▶ MorseDecoder ─▶ text
          (Goertzel,      (noise floor/peak,     (dit/dah and gap
           12 ms window)   hysteresis, debounce)  clustering)
   └──▶ FrequencyTracker (auto-tune, 5 Hz steps every 250 ms)
```

| Path | Role |
| --- | --- |
| `CWDecode/DSP/DecoderPipeline.swift` | Audio → key up/down → text; noise floor, squelch, debounce |
| `CWDecode/DSP/ToneDetector.swift` | Tone power over 12 ms windows every 4 ms |
| `CWDecode/DSP/FrequencyTracker.swift` | Finds the CW tone for auto-tune |
| `CWDecode/Morse/MorseDecoder.swift` | Timing → characters, adaptive speed |
| `CWDecode/DSP/PipelineRunner.swift` | Runs the pipeline off the main thread |
| `CWDecode/Audio/` | Input device list, audio input, and audio file reading |
| `CWDecode/App/DecoderViewModel.swift` | UI state and actions |
