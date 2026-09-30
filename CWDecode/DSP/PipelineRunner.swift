import Foundation

/// Runs the selected decoder on a private serial queue, recreating it when the sample rate or
/// the decoder kind changes.
final class PipelineRunner: @unchecked Sendable {
    enum Event: Sendable {
        case output(PipelineOutput)
        /// The stream ended and any in-progress character has been flushed.
        case finished(PipelineOutput)
    }

    // Everything below is only touched on `queue`.
    private let queue = DispatchQueue(label: "com.carlobermeier.CWDecode.pipeline", qos: .userInitiated)
    private var pipeline: (any DecoderEngine)?
    private var settings: PipelineSettings
    private let neuralModel: CWNetModel?
    private let emit: @Sendable (Event) -> Void

    /// Without a `neuralModel`, the classic decoder runs whatever `settings.decoder` says.
    init(settings: PipelineSettings, neuralModel: CWNetModel?, emit: @escaping @Sendable (Event) -> Void) {
        self.settings = settings
        self.neuralModel = neuralModel
        self.emit = emit
    }

    /// For live audio: returns immediately.
    func submit(_ samples: [Float], sampleRate: Double) {
        queue.async { self.process(samples, sampleRate: sampleRate) }
    }

    /// For file decoding: blocks the caller so reading can't outrun processing.
    func submitAndWait(_ samples: [Float], sampleRate: Double) {
        queue.sync { self.process(samples, sampleRate: sampleRate) }
    }

    func update(_ settings: PipelineSettings) {
        queue.async {
            let switching = settings.decoder != self.settings.decoder
            self.settings = settings
            if switching, let pipeline = self.pipeline {
                // Flush what the old decoder still holds; the next audio starts the new one.
                self.emit(.output(pipeline.finish()))
                self.pipeline = nil
            } else {
                self.pipeline?.update(settings)
            }
        }
    }

    func finish() {
        queue.async {
            guard let pipeline = self.pipeline else {
                self.emit(.finished(PipelineOutput(toneFrequency: self.settings.toneFrequency)))
                return
            }
            self.emit(.finished(pipeline.finish()))
            self.pipeline = nil
        }
    }

    private func process(_ samples: [Float], sampleRate: Double) {
        if pipeline?.sampleRate != sampleRate {
            if settings.decoder == .neural, let neuralModel {
                pipeline = NeuralPipeline(sampleRate: sampleRate, settings: settings, model: neuralModel)
            } else {
                pipeline = DecoderPipeline(sampleRate: sampleRate, settings: settings)
            }
        }
        guard let pipeline else { return }
        emit(.output(pipeline.process(samples)))
    }
}
