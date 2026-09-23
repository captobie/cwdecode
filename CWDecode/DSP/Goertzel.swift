import Foundation

/// Single-bin DFT, much cheaper than an FFT when only a few frequencies matter.
enum Goertzel {
    static func coefficient(frequency: Double, sampleRate: Double) -> Float {
        Float(2 * cos(2 * Double.pi * frequency / sampleRate))
    }

    static func hannWindow(count: Int) -> [Float] {
        guard count > 1 else { return [1] }
        return (0..<count).map { i in
            Float(0.5 - 0.5 * cos(2 * Double.pi * Double(i) / Double(count - 1)))
        }
    }

    /// Squared magnitude of `samples` at the frequency `coefficient` was built for.
    /// `window`, when given, must be the same length as `samples`.
    static func power(of samples: UnsafeBufferPointer<Float>, window: UnsafeBufferPointer<Float>? = nil, coefficient: Float) -> Float {
        var s1: Float = 0
        var s2: Float = 0
        if let window {
            for i in 0..<samples.count {
                let s0 = samples[i] * window[i] + coefficient * s1 - s2
                s2 = s1
                s1 = s0
            }
        } else {
            for sample in samples {
                let s0 = sample + coefficient * s1 - s2
                s2 = s1
                s1 = s0
            }
        }
        return s1 * s1 + s2 * s2 - coefficient * s1 * s2
    }
}
