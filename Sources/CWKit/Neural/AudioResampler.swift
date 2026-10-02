@preconcurrency import AVFoundation

/// Streams mono audio from any sample rate to the neural decoder's 8 kHz.
final class AudioResampler {
    let inputRate: Double
    private let converter: AVAudioConverter?
    private let inputFormat: AVAudioFormat
    private let outputFormat: AVAudioFormat

    init(inputRate: Double, outputRate: Double = NeuralFeatures.sampleRate) {
        self.inputRate = inputRate
        inputFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: inputRate, channels: 1, interleaved: false)!
        outputFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: outputRate, channels: 1, interleaved: false)!
        if inputRate == outputRate {
            converter = nil
        } else {
            converter = AVAudioConverter(from: inputFormat, to: outputFormat)
            converter?.sampleRateConverterQuality = AVAudioQuality.max.rawValue
        }
    }

    func process(_ samples: [Float]) -> [Float] {
        guard let converter else { return samples }
        guard !samples.isEmpty, let input = buffer(from: samples) else { return [] }
        var pending: AVAudioPCMBuffer? = input
        return drain(converter) { _, status in
            guard let buffer = pending else {
                status.pointee = .noDataNow
                return nil
            }
            pending = nil
            status.pointee = .haveData
            return buffer
        }
    }

    /// The converter's remaining output, at the end of a stream.
    func flush() -> [Float] {
        guard let converter else { return [] }
        let tail = drain(converter) { _, status in
            status.pointee = .endOfStream
            return nil
        }
        converter.reset()
        return tail
    }

    private func drain(_ converter: AVAudioConverter, input: @escaping AVAudioConverterInputBlock) -> [Float] {
        var output: [Float] = []
        let capacity: AVAudioFrameCount = 4096
        while true {
            guard let buffer = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: capacity) else { break }
            var error: NSError?
            let status = converter.convert(to: buffer, error: &error, withInputFrom: input)
            if buffer.frameLength > 0, let data = buffer.floatChannelData {
                output.append(contentsOf: UnsafeBufferPointer(start: data[0], count: Int(buffer.frameLength)))
            }
            // Keep going only while the output buffer came back full.
            if status != .haveData || buffer.frameLength < capacity { break }
        }
        return output
    }

    private func buffer(from samples: [Float]) -> AVAudioPCMBuffer? {
        guard let buffer = AVAudioPCMBuffer(pcmFormat: inputFormat, frameCapacity: AVAudioFrameCount(samples.count)),
              let data = buffer.floatChannelData else { return nil }
        samples.withUnsafeBufferPointer { data[0].update(from: $0.baseAddress!, count: samples.count) }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        return buffer
    }
}
