import Foundation

/// A condition that is either one value for every sample or drawn from a range per sample.
struct ValueRange: Codable, Equatable {
    var isRange: Bool
    var low: Double
    var high: Double

    /// `[low, high]` as cwsynth expects; equal ends for a fixed value.
    var bounds: [Double] {
        isRange ? [min(low, high), max(low, high)] : [low, low]
    }

    var lowest: Double { bounds[0] }
    var highest: Double { bounds[1] }
}

enum FistStyle: String, Codable, CaseIterable, Identifiable {
    case machine, paddle, bug, straight

    var id: Self { self }

    var title: String {
        switch self {
        case .machine: "Machine"
        case .paddle: "Paddle"
        case .bug: "Bug"
        case .straight: "Straight key"
        }
    }
}

enum DatasetSplit: String, Codable, CaseIterable, Identifiable {
    case train, val, test

    var id: Self { self }
}

/// Everything the window edits. Saved between launches.
struct SynthSettings: Codable, Equatable {
    enum TextSource: String, Codable, CaseIterable, Identifiable {
        /// cwsynth's weighted mix: callsigns, QSOs, abbreviations, English, random groups, noise.
        case mix
        /// One line per sample, cycling through the lines.
        case custom

        var id: Self { self }
    }

    var textSource = TextSource.mix
    var customText = "CQ CQ DE K1ABC K1ABC K\nK1ABC DE W2XYZ UR RST 599 599 <BT> NAME BOB <KN>\n"

    var pitch = ValueRange(isRange: true, low: 400, high: 900)
    var speed = ValueRange(isRange: true, low: 12, high: 30)
    /// Share of samples sent with Farnsworth spacing (characters at the drawn speed, longer gaps).
    var farnsworthShare = 0.0
    var fists: Set<FistStyle> = [.paddle, .bug, .straight]

    var noise = true
    var snr = ValueRange(isRange: true, low: -6, high: 20)

    /// Share of samples with each impairment; cwsynth draws the details per sample.
    var qsbShare = 0.0
    var qrnShare = 0.0
    var qrmShare = 0.0
    var filterShare = 0.0

    var maxClipSeconds = 15.0
    var count = 1000
    var split = DatasetSplit.train
    var seed = 0
    var outputPath: String?

    static let pitchLimits = 200.0...1500.0
    static let speedLimits = 5.0...60.0
    static let snrLimits = -30.0...40.0
    static let clipLimits = 3.0...60.0

    /// The non-empty custom lines, as typed.
    var customLines: [String] {
        customText.split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
    }

    /// Why these settings can't be generated, or nil.
    var problem: String? {
        if fists.isEmpty { return "Choose at least one fist." }
        if textSource == .custom && customLines.isEmpty { return "Enter at least one line of text." }
        for (name, range, limits) in [("Pitch", pitch, Self.pitchLimits), ("Speed", speed, Self.speedLimits),
                                      ("SNR", snr, Self.snrLimits)] {
            if !limits.contains(range.lowest) || !limits.contains(range.highest) {
                return "\(name) must be between \(Int(limits.lowerBound)) and \(Int(limits.upperBound))."
            }
        }
        if !Self.clipLimits.contains(maxClipSeconds) {
            return "Max clip length must be between \(Int(Self.clipLimits.lowerBound)) and \(Int(Self.clipLimits.upperBound)) s."
        }
        if count < 1 { return "Generate at least one sample." }
        return nil
    }

    /// cwsynth `ConditionRanges` for these settings. Everything the window doesn't show keeps
    /// cwsynth's defaults, except drift and chirp, which are off.
    var conditions: JobRequest.Conditions {
        JobRequest.Conditions(
            wpm: speed.bounds,
            farnsworthProb: farnsworthShare,
            fistStyles: Dictionary(uniqueKeysWithValues: fists.map { ($0.rawValue, 1.0) }),
            maxSignalS: maxClipSeconds,
            toneHz: pitch.bounds,
            driftProb: 0,
            chirpProb: 0,
            cleanProb: noise ? 0 : 1,
            snrDb: snr.bounds,
            // Uniform across the range the user typed, rather than cwsynth's lean toward low SNR.
            snrSkew: 1,
            qsbProb: qsbShare,
            qrnProb: qrnShare,
            qrmProb: qrmShare,
            rxFilterProb: filterShare
        )
    }

    var corpus: JobRequest.Corpus {
        JobRequest.Corpus(lines: textSource == .custom ? customLines : nil)
    }

    func request(_ action: JobRequest.Action) -> JobRequest {
        JobRequest(action: action, conditions: conditions, corpus: corpus, split: split, seed: seed)
    }
}

/// The SNR as it would read in other noise bandwidths. cwsynth measures in 500 Hz
/// (`ml/cwsynth/channel.py`); CWDecode's detector is about 170 Hz wide.
enum SNRBandwidth {
    static func convert(_ snr500: Double, toBandwidth hz: Double) -> Double {
        snr500 + 10 * log10(500 / hz)
    }
}
