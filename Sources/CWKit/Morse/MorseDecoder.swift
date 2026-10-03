import Foundation

/// Turns key-down (mark) and key-up (space) durations into text.
///
/// Speed is tracked automatically: recent mark durations are split into a dit
/// cluster and a dah cluster, and the midpoint between them classifies each mark.
/// Marks are only classified when their character ends, so the first character
/// after a speed change benefits from what its own elements taught the decoder.
/// Letter and word gaps are learned the same way, which handles Farnsworth spacing.
struct MorseDecoder {
    private(set) var ditDuration: TimeInterval
    private(set) var dahDuration: TimeInterval

    private var pendingMarks: [TimeInterval] = []
    private var recentMarks: [TimeInterval] = []
    /// Letter and word gaps, in dot units.
    private var recentGaps: [Double] = []
    private var wordGapUnits = 5.0
    private var lastGap = TimeInterval.infinity
    private var characterEnded = true
    private var wordEnded = true

    private static let historyLimit = 16
    private static let maxSymbolsPerCharacter = 10

    init(initialWPM: Double = 20) {
        let dit = 1.2 / initialWPM
        ditDuration = dit
        dahDuration = 3 * dit
    }

    /// One dot unit. A dah is two units longer than a dit, which stays true even when
    /// the tone detector lengthens or shortens every mark by the same amount.
    var unit: TimeInterval { (dahDuration - ditDuration) / 2 }

    /// PARIS-standard words per minute for the current timing estimate.
    var estimatedWPM: Double { 1.2 / unit }

    /// Elements of the character currently being received, as "." and "-".
    var pendingSymbols: String { String(pendingMarks.map(symbol)) }

    /// How much longer marks measure than they should, and so how much shorter the gaps measure.
    private var markBias: TimeInterval {
        min(max(ditDuration - unit, -0.4 * unit), 0.4 * unit)
    }

    /// Records a completed key-down period.
    mutating func mark(_ duration: TimeInterval) {
        guard duration > 0 else { return }
        if lastGap.isFinite {
            learnGap(lastGap + markBias)
        }
        lastGap = .infinity
        learnMark(duration)

        if pendingMarks.count < Self.maxSymbolsPerCharacter {
            pendingMarks.append(duration)
        }
        characterEnded = false
        wordEnded = false
    }

    /// Reports the length of the current key-up period. Safe to call repeatedly
    /// while the gap grows; each character and word break is emitted only once.
    mutating func space(_ duration: TimeInterval) -> String {
        lastGap = duration
        let gap = duration + markBias
        var output = ""
        if !characterEnded, gap >= 2 * unit {
            output += MorseCode.decode(pendingSymbols)
            pendingMarks = []
            characterEnded = true
        }
        if characterEnded, !wordEnded, gap >= wordGapUnits * unit {
            output += " "
            wordEnded = true
        }
        return output
    }

    private func symbol(for duration: TimeInterval) -> Character {
        duration < (ditDuration + dahDuration) / 2 ? "." : "-"
    }

    private mutating func learnMark(_ duration: TimeInterval) {
        // Clamp outliers (a held key, a tuning carrier) so they can't wreck the estimate,
        // while still letting a genuine slowdown pull the estimate upward over time.
        Self.append(min(duration, 8 * ditDuration), to: &recentMarks)
        guard recentMarks.count >= 3 else { return }

        if let clusters = Self.twoClusters(recentMarks, minimumRatio: 2) {
            ditDuration = clusters.short
            dahDuration = clusters.long
        } else {
            // Only one kind of element seen recently: update whichever estimate it's closer to.
            let mean = Self.mean(recentMarks)
            if abs(log(mean / ditDuration)) <= abs(log(mean / dahDuration)) {
                ditDuration = mean
            } else {
                dahDuration = mean
            }
        }
        // A dah far longer than 3 dits means the dit estimate is too short (a noise blip
        // learned as a dit, or a sudden slowdown). Pinning the dah instead would lock in the
        // bad dit, because the 8-dit clamp above keeps real dits out of the history.
        if dahDuration > 4.5 * ditDuration {
            ditDuration = dahDuration / 4.5
        }
        dahDuration = max(dahDuration, 2 * ditDuration)
    }

    private mutating func learnGap(_ gap: TimeInterval) {
        let units = gap / unit
        // Only letter and word gaps; skip element gaps and long pauses between overs.
        guard units >= 2, units < 3 * wordGapUnits else { return }
        Self.append(units, to: &recentGaps)
        guard recentGaps.count >= 2 else { return }

        if let clusters = Self.twoClusters(recentGaps, minimumRatio: 1.6) {
            wordGapUnits = (clusters.short + clusters.long) / 2
        } else {
            // Probably all letter gaps; keep word breaks comfortably above them.
            wordGapUnits = max(5, 1.4 * Self.mean(recentGaps))
        }
    }

    private static func append(_ value: Double, to history: inout [Double]) {
        history.append(value)
        if history.count > historyLimit {
            history.removeFirst()
        }
    }

    private static func mean(_ values: [Double]) -> Double {
        values.reduce(0, +) / Double(values.count)
    }

    /// Two-means clustering in log space. Returns nil if the values don't form two distinct groups.
    private static func twoClusters(_ values: [Double], minimumRatio: Double) -> (short: Double, long: Double)? {
        guard var short = values.min(), var long = values.max(), short > 0 else { return nil }
        for _ in 0..<8 {
            let split = (short * long).squareRoot()
            let shortValues = values.filter { $0 < split }
            let longValues = values.filter { $0 >= split }
            guard !shortValues.isEmpty, !longValues.isEmpty else { return nil }
            short = mean(shortValues)
            long = mean(longValues)
        }
        return long / short >= minimumRatio ? (short, long) : nil
    }
}
