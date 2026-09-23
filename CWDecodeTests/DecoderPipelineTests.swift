import Testing
@testable import CWDecode

struct DecoderPipelineTests {
    private func decode(_ samples: [Float], sampleRate: Double = 48_000, settings: PipelineSettings = PipelineSettings(), chunkSize: Int = 2048) -> (text: String, last: PipelineOutput) {
        let pipeline = DecoderPipeline(sampleRate: sampleRate, settings: settings)
        var text = ""
        var last = PipelineOutput()
        var start = 0
        while start < samples.count {
            let end = min(start + chunkSize, samples.count)
            last = pipeline.process(Array(samples[start..<end]))
            text += last.text
            start = end
        }
        text += pipeline.finish().text
        return (text.trimmingCharacters(in: .whitespaces), last)
    }

    private let message = "CQ CQ DE W1AW W1AW K"

    @Test(arguments: [15.0, 25.0, 35.0])
    func decodesCleanAudio(wpm: Double) {
        let audio = MorseSynthesizer.audio(text: message, wpm: wpm)
        let result = decode(audio, settings: PipelineSettings(autoTune: false))
        #expect(result.text == message, "decoded \(result.text)")
        #expect(abs(result.last.wpm - wpm) / wpm < 0.15, "estimated \(result.last.wpm) WPM")
    }

    /// The noise is white across the whole 24 kHz band. Inside the detector's bandwidth
    /// these amplitudes give roughly 20 dB and 14 dB SNR, the latter just above the default squelch.
    @Test(arguments: [0.1, 0.05] as [Float])
    func decodesNoisyAudio(amplitude: Float) {
        let audio = MorseSynthesizer.audio(text: message, wpm: 22, amplitude: amplitude, noise: 0.1, seed: 7)
        let text = decode(audio, settings: PipelineSettings(autoTune: false)).text
        #expect(text == message, "decoded \(text)")
    }

    @Test func autoTuneFindsAnOffFrequencySignal() {
        let audio = MorseSynthesizer.audio(text: "VVV VVV " + message, wpm: 20, frequency: 860, noise: 0.02)
        let result = decode(audio, settings: PipelineSettings(toneFrequency: 600, autoTune: true))
        #expect(abs(result.last.toneFrequency - 860) < 10)
        #expect(result.text.hasSuffix(message), "decoded \(result.text)")
    }

    @Test func worksAtOtherSampleRatesAndChunkSizes() {
        let audio = MorseSynthesizer.audio(text: message, wpm: 20, sampleRate: 44_100)
        for chunkSize in [256, 1000, 44_100] {
            let result = decode(audio, sampleRate: 44_100, settings: PipelineSettings(autoTune: false), chunkSize: chunkSize)
            #expect(result.text == message)
        }
    }

    @Test func staysQuietOnPureNoise() {
        var random = SplitMix64(state: 42)
        let noise = (0..<(48_000 * 10)).map { _ in random.gaussian() * 0.05 }
        let text = decode(noise).text
        #expect(text.isEmpty, "decoded \(text)")
    }
}
