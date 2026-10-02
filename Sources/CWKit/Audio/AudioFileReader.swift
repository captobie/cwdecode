import AVFoundation

public enum AudioFileReader {
    /// Reads any format AVFoundation understands (WAV, AIFF, MP3, M4A, …) as mono float chunks.
    public static func read(_ url: URL, chunkFrames: AVAudioFrameCount = 16_384, handler: ([Float], Double) -> Void) throws {
        let file = try AVAudioFile(forReading: url)
        let format = file.processingFormat
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: chunkFrames) else {
            throw CocoaError(.fileReadCorruptFile)
        }
        while file.framePosition < file.length {
            try file.read(into: buffer, frameCount: chunkFrames)
            guard buffer.frameLength > 0 else { break }
            handler(buffer.monoSamples(), format.sampleRate)
        }
    }
}

extension AVAudioPCMBuffer {
    /// Averages all channels into a single array of float samples.
    public func monoSamples() -> [Float] {
        let frames = Int(frameLength)
        guard frames > 0, let channels = floatChannelData else { return [] }
        let channelCount = Int(format.channelCount)
        var mono = Array(UnsafeBufferPointer(start: channels[0], count: frames))
        guard channelCount > 1 else { return mono }
        for channel in 1..<channelCount {
            let data = channels[channel]
            for i in 0..<frames { mono[i] += data[i] }
        }
        let scale = 1 / Float(channelCount)
        for i in 0..<frames { mono[i] *= scale }
        return mono
    }
}
