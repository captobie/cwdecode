import Foundation

/// Runs a `DecoderPipeline` on a private serial queue, recreating it when the sample rate changes.
final class PipelineRunner: @unchecked Sendable {
    enum Event: Sendable {
        case output(PipelineOutput)
        /// The stream ended and any in-progress character has been flushed.
        case finished(PipelineOutput)
    }

    // Everything below is only touched on `queue`.
    private let queue = DispatchQueue(label: "com.carlobermeier.CWDecode.pipeline", qos: .userInitiated)
    private var pipeline: DecoderPipeline?
    private var settings: PipelineSettings
    private let emit: @Sendable (Event) -> Void

    init(settings: PipelineSettings, emit: @escaping @Sendable (Event) -> Void) {
        self.settings = settings
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
            self.settings = settings
            self.pipeline?.update(settings)
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
            pipeline = DecoderPipeline(sampleRate: sampleRate, settings: settings)
        }
        guard let pipeline else { return }
        emit(.output(pipeline.process(samples)))
    }
}
