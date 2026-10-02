import AVFoundation
import Testing
@testable import CWKit

struct AudioFileReaderTests {
    @Test func decodesAStereoWAVFile() throws {
        let sampleRate = 44_100.0
        let message = "TEST DE CWDECODE"
        let mono = MorseSynthesizer.audio(text: message, wpm: 20, sampleRate: sampleRate, noise: 0.02)

        let url = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: url) }
        let format = try #require(AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 2))
        do {
            let file = try AVAudioFile(forWriting: url, settings: [
                AVFormatIDKey: kAudioFormatLinearPCM,
                AVSampleRateKey: sampleRate,
                AVNumberOfChannelsKey: 2,
                AVLinearPCMBitDepthKey: 16,
            ])
            let buffer = try #require(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(mono.count)))
            buffer.frameLength = buffer.frameCapacity
            for i in mono.indices {
                buffer.floatChannelData![0][i] = mono[i]
                buffer.floatChannelData![1][i] = 0  // radio audio on the left channel only
            }
            try file.write(from: buffer)
        }

        let pipeline = DecoderPipeline(sampleRate: sampleRate, settings: PipelineSettings(autoTune: false))
        var text = ""
        try AudioFileReader.read(url) { samples, rate in
            #expect(rate == sampleRate)
            text += pipeline.process(samples).text
        }
        text += pipeline.finish().text
        #expect(text.trimmingCharacters(in: .whitespaces) == message)
    }
}
