import Foundation

/// Measures the power of a single tone over short overlapping windows.
struct ToneDetector {
    let sampleRate: Double
    let windowSize: Int
    let hopSize: Int

    var frequency: Double {
        didSet { coefficient = Goertzel.coefficient(frequency: frequency, sampleRate: sampleRate) }
    }

    private var coefficient: Float
    private let window: [Float]
    private let normalization: Float
    private var buffer: [Float] = []

    /// A 12 ms window gives roughly 170 Hz of bandwidth, narrow enough to reject
    /// most neighbouring signals but short enough for 40+ WPM code.
    init(sampleRate: Double, frequency: Double, windowDuration: Double = 0.012, hopDuration: Double = 0.004) {
        self.sampleRate = sampleRate
        self.frequency = frequency
        windowSize = max(16, Int(sampleRate * windowDuration))
        hopSize = max(1, Int(sampleRate * hopDuration))
        coefficient = Goertzel.coefficient(frequency: frequency, sampleRate: sampleRate)
        window = Goertzel.hannWindow(count: windowSize)
        // Scale so a full-scale sine of amplitude A reads as A².
        let gain = window.reduce(0, +) / 2
        normalization = 1 / (gain * gain)
    }

    var hopDuration: Double { Double(hopSize) / sampleRate }

    /// Appends samples and returns the tone power, in dBFS, for every hop completed.
    mutating func process(_ samples: [Float]) -> [Double] {
        buffer.append(contentsOf: samples)
        var levels: [Double] = []
        var start = 0
        buffer.withUnsafeBufferPointer { buffer in
            window.withUnsafeBufferPointer { window in
                while buffer.count - start >= windowSize {
                    let slice = UnsafeBufferPointer(rebasing: buffer[start..<(start + windowSize)])
                    let power = Goertzel.power(of: slice, window: window, coefficient: coefficient) * normalization
                    levels.append(10 * log10(Double(power) + 1e-12))
                    start += hopSize
                }
            }
        }
        buffer.removeFirst(start)
        return levels
    }
}
