import Foundation
@testable import CWKit

/// Deterministic random numbers so noisy test signals are reproducible.
struct SplitMix64: RandomNumberGenerator {
    var state: UInt64

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    mutating func gaussian() -> Float {
        let u1 = Double.random(in: Double.ulpOfOne..<1, using: &self)
        let u2 = Double.random(in: 0..<1, using: &self)
        return Float((-2 * log(u1)).squareRoot() * cos(2 * .pi * u2))
    }
}

enum MorseSynthesizer {
    struct Element {
        var keyDown: Bool
        var duration: TimeInterval
    }

    /// Ideal keying for `text` at `wpm`, with optional Farnsworth spacing.
    static func elements(for text: String, wpm: Double, farnsworthWPM: Double? = nil) -> [Element] {
        let unit = 1.2 / wpm
        let gapUnit = 1.2 / (farnsworthWPM ?? wpm)
        var result: [Element] = []
        for (wordIndex, word) in text.uppercased().split(separator: " ").enumerated() {
            if wordIndex > 0 { result.append(Element(keyDown: false, duration: 7 * gapUnit)) }
            for (charIndex, character) in word.enumerated() {
                if charIndex > 0 { result.append(Element(keyDown: false, duration: 3 * gapUnit)) }
                guard let pattern = MorseCode.encode(character) else { continue }
                for (symbolIndex, symbol) in pattern.enumerated() {
                    if symbolIndex > 0 { result.append(Element(keyDown: false, duration: unit)) }
                    result.append(Element(keyDown: true, duration: symbol == "." ? unit : 3 * unit))
                }
            }
        }
        return result
    }

    /// Keyed sine wave with 5 ms raised-cosine edges, plus optional white noise.
    static func audio(
        text: String,
        wpm: Double,
        frequency: Double = 700,
        sampleRate: Double = 48_000,
        amplitude: Float = 0.5,
        noise: Float = 0,
        seed: UInt64 = 1,
        trailingSilence: TimeInterval = 1.0
    ) -> [Float] {
        var keying: [Bool] = Array(repeating: false, count: Int(0.5 * sampleRate))
        for element in elements(for: text, wpm: wpm) {
            keying += Array(repeating: element.keyDown, count: Int(element.duration * sampleRate))
        }
        keying += Array(repeating: false, count: Int(trailingSilence * sampleRate))

        let rampSamples = sampleRate * 0.005
        var envelope = 0.0
        var random = SplitMix64(state: seed)
        var samples = [Float](repeating: 0, count: keying.count)
        for (i, keyDown) in keying.enumerated() {
            envelope = min(max(envelope + (keyDown ? 1 : -1) / rampSamples, 0), 1)
            let shaped = 0.5 - 0.5 * cos(.pi * envelope)
            let tone = sin(2 * .pi * frequency * Double(i) / sampleRate) * shaped
            samples[i] = Float(tone) * amplitude + (noise > 0 ? random.gaussian() * noise : 0)
        }
        return samples
    }
}
