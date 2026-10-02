import Foundation
import Testing
@testable import CWKit

/// Holds the Swift neural decoder to the Python reference (`ml/cwmodel`), using golden files
/// written by `python -m cwmodel.export`.
struct NeuralDecoderTests {
    struct Golden: Decodable {
        struct Stream: Decodable {
            var window_s: Double, hop_s: Double, context_s: Double, chunk_s: Double
        }
        struct Spectrogram: Decodable {
            var bins: Int, frames: Int, values: [Float]
        }
        struct Clip: Decodable {
            var samples: Int
            var reference: String
            var streamed_text: String
            var spectrogram: Spectrogram?
            var log_probs_head: [[Float]]?
            var output_frames: Int?
        }
        var pcm_scale: Float
        var stream: Stream
        var clips: [String: Clip]
    }

    private static let bundle = Bundle.module

    private static let golden: Golden = {
        let url = bundle.url(forResource: "golden", withExtension: "json")!
        return try! JSONDecoder().decode(Golden.self, from: Data(contentsOf: url))
    }()

    private static let model = try! CWNetModel()

    private static func audio(_ name: String) -> [Float] {
        let data = try! Data(contentsOf: bundle.url(forResource: name, withExtension: "s16")!)
        return data.withUnsafeBytes { raw in
            raw.bindMemory(to: Int16.self).map { Float(Int16(littleEndian: $0)) / golden.pcm_scale }
        }
    }

    // MARK: - Golden comparisons

    @Test func spectrogramMatchesPython() throws {
        let clip = try #require(Self.golden.clips["features"])
        let expected = try #require(clip.spectrogram)
        let audio = Self.audio("features")
        #expect(NeuralFeatures.frameCount(samples: audio.count) == expected.frames)
        let actual = NeuralFeatures.spectrogram(audio)
        #expect(actual.count == expected.values.count)
        let differences = zip(actual, expected.values).map { abs($0 - $1) }
        let worst = differences.max() ?? 0
        let mean = differences.reduce(0, +) / Float(differences.count)
        #expect(worst < 2e-3, "largest difference \(worst)")
        #expect(mean < 1e-4, "mean difference \(mean)")
    }

    @Test func modelMatchesPython() throws {
        let clip = try #require(Self.golden.clips["features"])
        let head = try #require(clip.log_probs_head)
        let audio = Self.audio("features")
        #expect(Self.model.vocabulary.count == head[0].count)
        let frames = NeuralFeatures.frameCount(samples: audio.count)
        let result = try Self.model.logProbabilities(spectrogram: NeuralFeatures.spectrogram(audio), frames: frames)
        #expect(result.frames == clip.output_frames)
        var worst: Float = 0
        for (t, row) in head.enumerated() {
            for (c, expected) in row.enumerated() {
                worst = max(worst, abs(exp(result.values[t * result.classes + c]) - exp(expected)))
            }
        }
        #expect(worst < 1e-3, "largest probability difference \(worst)")
    }

    @Test(arguments: ["features", "stream_hand", "stream_fast", "stream_noise"])
    func streamingMatchesPython(clip name: String) throws {
        let clip = try #require(Self.golden.clips[name])
        let settings = Self.golden.stream
        let decoder = StreamingCTCDecoder(vocabulary: Self.model.vocabulary, window: settings.window_s,
                                          hop: settings.hop_s, context: settings.context_s) { spectrogram, frames in
            try Self.model.logProbabilities(spectrogram: spectrogram, frames: frames)
        }
        let audio = Self.audio(name)
        let chunk = Int((settings.chunk_s * NeuralFeatures.sampleRate).rounded())
        for start in stride(from: 0, to: audio.count, by: chunk) {
            _ = try decoder.process(Array(audio[start..<min(start + chunk, audio.count)]))
        }
        _ = try decoder.finish()
        #expect(decoder.text.trimmingCharacters(in: .whitespaces) == clip.streamed_text)
    }

    // MARK: - Pieces

    @Test func handoffLandsInTheWidestGap() {
        let frame = StreamingCTCDecoder.outputFrameDuration
        // Emissions at 1.0 and 1.2 s leave the widest gap at the top of the range.
        let cut = StreamingCTCDecoder.handoff(times: [1.0, 1.2], lo: 0.9, hi: 1.5)
        #expect(abs(cut - (0.9 + 37 * frame)) < 1e-9)
        #expect(StreamingCTCDecoder.handoff(times: [], lo: 2, hi: 2) == 2)
    }

    @Test func reconcileAddsMissedTokensAndSkipsRepeats() {
        // The earlier window committed "NOW"; the new one also sees " 5" before the handoff.
        let missed = StreamingCTCDecoder.reconcile(previous: ["N", "O", "W"], before: ["O", "W", " ", "5"], after: [" "])
        #expect(missed.missed == [" ", "5"] && missed.skip == 0)
        // The earlier window committed "W " but the new one emits the space after the handoff.
        let repeated = StreamingCTCDecoder.reconcile(previous: ["O", "W", " "], before: ["O", "W"], after: [" ", "5"])
        #expect(repeated.missed.isEmpty && repeated.skip == 1)
        // Nothing in common: fall back to times alone.
        let unrelated = StreamingCTCDecoder.reconcile(previous: ["A"], before: ["B"], after: ["C"])
        #expect(unrelated.missed.isEmpty && unrelated.skip == 0)
    }

    @Test func resamplerKeepsToneAndLevel() {
        let rate = 48_000.0
        let tone = (0..<Int(rate)).map { Float(sin(2 * Double.pi * 700 * Double($0) / rate)) * 0.5 }
        let resampler = AudioResampler(inputRate: rate)
        var out: [Float] = []
        for start in stride(from: 0, to: tone.count, by: 2048) {
            out += resampler.process(Array(tone[start..<min(start + 2048, tone.count)]))
        }
        out += resampler.flush()
        #expect(abs(out.count - 8000) <= 16, "got \(out.count) samples")
        let steady = Array(out[1000..<7000])
        let rms = (steady.map { $0 * $0 }.reduce(0, +) / Float(steady.count)).squareRoot()
        #expect(abs(rms - 0.5 / Float(2).squareRoot()) < 0.01)
        let crossings = zip(steady, steady.dropFirst()).filter { $0 < 0 && $1 >= 0 }.count
        #expect(abs(Double(crossings) / 0.75 - 700) < 5, "\(crossings) upward crossings in 0.75 s")
    }

    // MARK: - Whole pipeline

    private func decode(_ samples: [Float], sampleRate: Double = 48_000) -> String {
        let pipeline = NeuralPipeline(sampleRate: sampleRate, settings: PipelineSettings(decoder: .neural), model: Self.model)
        var text = ""
        for start in stride(from: 0, to: samples.count, by: 2048) {
            text += pipeline.process(Array(samples[start..<min(start + 2048, samples.count)])).text
        }
        text += pipeline.finish().text
        return text.trimmingCharacters(in: .whitespaces)
    }

    @Test(arguments: [16.0, 28.0])
    func pipelineDecodes48kHzAudio(wpm: Double) {
        let message = "CQ CQ DE W1AW W1AW K"
        let audio = MorseSynthesizer.audio(text: message, wpm: wpm, frequency: 650, noise: 0.02, seed: 3)
        #expect(decode(audio) == message)
    }

    @Test func pipelineStaysSilentOnNoise() {
        var random = SplitMix64(state: 9)
        let noise = (0..<(48_000 * 10)).map { _ in random.gaussian() * 0.1 }
        #expect(decode(noise) == "")
    }
}
