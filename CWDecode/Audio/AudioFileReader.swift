import AVFoundation

enum AudioFileReader {
    /// Reads any format AVFoundation understands (WAV, AIFF, MP3, M4A, …) as mono float chunks.
    static func read(_ url: URL, chunkFrames: AVAudioFrameCount = 16_384, handler: ([Float], Double) -> Void) throws {
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
