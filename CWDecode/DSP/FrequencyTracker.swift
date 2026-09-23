import Foundation

/// Finds the strongest tone in the CW audio passband so the detector can follow it.
struct FrequencyTracker {
    static let searchRange: ClosedRange<Double> = 300...1200
    private static let step = 5.0
    private static let minimumPeakToMedian: Float = 30  // about 15 dB

    let sampleRate: Double
    private(set) var frequency: Double

    private let windowSize: Int
    private let scanInterval: Int
    private let window: [Float]
    private let bins: [(frequency: Double, coefficient: Float)]
    private var ring: [Float]
    private var writeIndex = 0
    private var filled = 0
    private var samplesSinceScan = 0
    private var candidate: Double?
    private var candidateHits = 0

    init(sampleRate: Double, frequency: Double) {
        self.sampleRate = sampleRate
        self.frequency = frequency
        windowSize = Int(sampleRate * 0.1)
        scanInterval = Int(sampleRate * 0.25)
        window = Goertzel.hannWindow(count: windowSize)
        ring = Array(repeating: 0, count: windowSize)
        bins = stride(from: Self.searchRange.lowerBound, through: Self.searchRange.upperBound, by: Self.step).map {
            ($0, Goertzel.coefficient(frequency: $0, sampleRate: sampleRate))
        }
    }

    /// Feeds audio and returns the new frequency whenever the estimate changes.
    mutating func process(_ samples: [Float]) -> Double? {
        var changed = false
        for sample in samples {
            ring[writeIndex] = sample
            writeIndex = (writeIndex + 1) % windowSize
            filled = min(filled + 1, windowSize)
            samplesSinceScan += 1
            if samplesSinceScan >= scanInterval, filled == windowSize {
                samplesSinceScan = 0
                if scan() { changed = true }
            }
        }
        return changed ? frequency : nil
    }

    mutating func reset(to frequency: Double) {
        self.frequency = frequency
        candidate = nil
        candidateHits = 0
    }

    private mutating func scan() -> Bool {
        var ordered = [Float](repeating: 0, count: windowSize)
        for i in 0..<windowSize {
            ordered[i] = ring[(writeIndex + i) % windowSize] * window[i]
        }
        let powers = ordered.withUnsafeBufferPointer { samples in
            bins.map { Goertzel.power(of: samples, coefficient: $0.coefficient) }
        }

        guard let peak = powers.indices.max(by: { powers[$0] < powers[$1] }),
              peak > 0, peak < powers.count - 1 else { return false }
        let median = powers.sorted()[powers.count / 2]
        guard powers[peak] > median * Self.minimumPeakToMedian else { return false }

        // Parabolic interpolation on the log spectrum for sub-bin accuracy.
        let a = log(Double(powers[peak - 1]) + 1e-20)
        let b = log(Double(powers[peak]) + 1e-20)
        let c = log(Double(powers[peak + 1]) + 1e-20)
        let denominator = a - 2 * b + c
        let offset = denominator == 0 ? 0 : 0.5 * (a - c) / denominator
        let measured = bins[peak].frequency + offset * Self.step

        if abs(measured - frequency) <= 20 {
            frequency += (measured - frequency) * 0.5
            candidate = nil
            return true
        }
        // A new signal must show up in two consecutive scans before we jump to it.
        if let candidate, abs(measured - candidate) <= 20 {
            candidateHits += 1
        } else {
            candidate = measured
            candidateHits = 1
        }
        guard candidateHits >= 2 else { return false }
        frequency = measured
        candidate = nil
        return true
    }
}
