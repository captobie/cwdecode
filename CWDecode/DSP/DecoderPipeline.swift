import Foundation

struct PipelineSettings: Sendable, Equatable {
    var toneFrequency = 700.0
    var autoTune = true
    /// Minimum signal-to-noise ratio before the key is allowed to close.
    var squelchDB = 12.0
    var initialWPM = 20.0
}

struct PipelineOutput: Sendable {
    var text = ""
    var keyDown = false
    /// Current tone level between the noise floor (0) and the signal peak (1).
    var signalLevel = 0.0
    var snrDB = 0.0
    var toneFrequency = 0.0
    var wpm = 0.0
    var pendingSymbols = ""
}

/// Audio in, text out: tone detection → adaptive keying threshold → Morse timing decoder.
///
/// Not thread-safe; confine each instance to one thread or serial queue.
final class DecoderPipeline {
    let sampleRate: Double
    private(set) var settings: PipelineSettings

    private var detector: ToneDetector
    private var tracker: FrequencyTracker
    private var decoder: MorseDecoder

    private var noisePower: Double?
    private var peakDB = 0.0
    private var snrDB = 0.0
    private var level = 0.0
    private var keyDown = false
    private var pendingHops = 0
    private var pendingStart = 0.0
    private var lastTransition = 0.0
    private var hopCount = 0

    /// A state change must persist this many hops (8 ms) to count, which filters clicks and noise spikes.
    private static let debounceHops = 2

    init(sampleRate: Double, settings: PipelineSettings = PipelineSettings()) {
        self.sampleRate = sampleRate
        self.settings = settings
        detector = ToneDetector(sampleRate: sampleRate, frequency: settings.toneFrequency)
        tracker = FrequencyTracker(sampleRate: sampleRate, frequency: settings.toneFrequency)
        decoder = MorseDecoder(initialWPM: settings.initialWPM)
    }

    func update(_ newSettings: PipelineSettings) {
        if !newSettings.autoTune {
            detector.frequency = newSettings.toneFrequency
        } else if !settings.autoTune {
            tracker.reset(to: detector.frequency)
        }
        settings = newSettings
    }

    func process(_ samples: [Float]) -> PipelineOutput {
        if settings.autoTune, let frequency = tracker.process(samples) {
            detector.frequency = frequency
        }
        var text = ""
        for levelDB in detector.process(samples) {
            text += handleHop(levelDB)
        }
        return output(text: text)
    }

    /// Flushes whatever is still in progress, e.g. at the end of a file or when listening stops.
    func finish() -> PipelineOutput {
        let now = Double(hopCount) * detector.hopDuration
        if keyDown {
            decoder.mark(now - lastTransition)
            keyDown = false
            lastTransition = now
        }
        pendingHops = 0
        return output(text: decoder.space(.infinity))
    }

    private func handleHop(_ levelDB: Double) -> String {
        let now = Double(hopCount) * detector.hopDuration
        hopCount += 1
        let hop = detector.hopDuration

        let power = pow(10, levelDB / 10)
        guard var noise = noisePower else {
            noisePower = power
            peakDB = levelDB
            return ""
        }
        // Noise floor: the mean power of hops that look like noise (~300 ms average).
        // Anything else (marks, their edges, spikes) only nudges it over ~5 s in dB, so the
        // tone can't inflate it, yet a jump in band noise can't hold the key down forever.
        var noiseDB = 10 * log10(noise + 1e-12)
        if !keyDown, pendingHops == 0, levelDB < noiseDB + 6 {
            noise += (power - noise) * min(1, hop / 0.3)
        } else {
            noise *= pow(10, (levelDB - noiseDB) * min(1, hop / 5) / 10)
        }
        noisePower = noise
        noiseDB = 10 * log10(noise + 1e-12)
        // Signal peak: rises within ~20 ms, decays over ~1.5 s so it survives word gaps.
        peakDB += (levelDB - peakDB) * (levelDB > peakDB ? 0.2 : min(1, hop / 1.5))
        peakDB = max(peakDB, noiseDB)

        snrDB = peakDB - noiseDB
        level = snrDB > 0 ? min(max((levelDB - noiseDB) / snrDB, 0), 1) : 0
        // Hysteresis between 60 % and 40 % of the way from noise to peak, but always a fixed
        // margin above the noise so a fading peak can't let noise spikes through.
        let above = levelDB - noiseDB
        let wantsKeyDown = snrDB >= settings.squelchDB
            && (keyDown ? above > max(0.4 * snrDB, 5) : above > max(0.6 * snrDB, 8))

        var text = ""
        if wantsKeyDown != keyDown {
            if pendingHops == 0 { pendingStart = now }
            pendingHops += 1
            if pendingHops >= Self.debounceHops {
                let duration = pendingStart - lastTransition
                if wantsKeyDown {
                    text += decoder.space(duration)
                } else {
                    decoder.mark(duration)
                }
                keyDown = wantsKeyDown
                lastTransition = pendingStart
                pendingHops = 0
            }
        } else {
            pendingHops = 0
        }

        if !keyDown {
            text += decoder.space(now - lastTransition)
        }
        return text
    }

    private func output(text: String) -> PipelineOutput {
        PipelineOutput(
            text: text,
            keyDown: keyDown,
            signalLevel: level,
            snrDB: snrDB,
            toneFrequency: detector.frequency,
            wpm: decoder.estimatedWPM,
            pendingSymbols: decoder.pendingSymbols
        )
    }
}
