import Foundation

/// Audio in, text out, with the neural decoder: resample to 8 kHz, then decode overlapping
/// windows with the CTC model.
///
/// The classic pipeline still runs alongside, but only for the signal meter, tone and WPM
/// display; its text is discarded. Not thread-safe; confine each instance to one queue.
final class NeuralPipeline: DecoderEngine {
    let sampleRate: Double
    private let meters: DecoderPipeline
    private let resampler: AudioResampler
    private let decoder: StreamingCTCDecoder
    private var failure: (any Error)?

    init(sampleRate: Double, settings: PipelineSettings, model: CWNetModel) {
        self.sampleRate = sampleRate
        meters = DecoderPipeline(sampleRate: sampleRate, settings: settings)
        resampler = AudioResampler(inputRate: sampleRate)
        decoder = StreamingCTCDecoder(vocabulary: model.vocabulary) { spectrogram, frames in
            try model.logProbabilities(spectrogram: spectrogram, frames: frames)
        }
    }

    func update(_ settings: PipelineSettings) {
        meters.update(settings)
    }

    func process(_ samples: [Float]) -> PipelineOutput {
        output(meters: meters.process(samples)) { try decoder.process(resampler.process(samples)) }
    }

    func finish() -> PipelineOutput {
        output(meters: meters.finish()) {
            let tail = try decoder.process(resampler.flush())
            return try tail + decoder.finish()
        }
    }

    private func output(meters: PipelineOutput, decode: () throws -> String) -> PipelineOutput {
        var out = meters
        out.pendingSymbols = ""
        do {
            out.text = try decode()
        } catch {
            // A model error would repeat on every window; report it once and stop decoding.
            if failure == nil { failure = error; print("Neural decoder failed: \(error)") }
            out.text = ""
        }
        out.tentativeText = decoder.tentativeText
        return out
    }
}
