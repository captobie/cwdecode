import Foundation
import Testing
@testable import CWDecode

struct HubAudioFormatTests {
    @Test func decodesMainAudioAsLittleEndianInt16() throws {
        let message = Data([0x00, 0x00, 0x00, 0xFF, 0x7F, 0x00, 0x80, 0x00, 0x40])
        let samples = try #require(HubAudioFormat.mainAudioSamples(in: message))
        #expect(samples == [0, Float(Int16.max) / 32768, -1, 0.5])
    }

    @Test func ignoresOtherChannelsAndJSON() {
        #expect(HubAudioFormat.mainAudioSamples(in: Data([0x01, 0x00, 0x40])) == nil)
        #expect(HubAudioFormat.mainAudioSamples(in: Data([0x02, 0x00, 0x40])) == nil)
        #expect(HubAudioFormat.mainAudioSamples(in: Data(#"{"type":"state"}"#.utf8)) == nil)
    }

    @Test func ignoresMessagesWithoutAWholeSample() {
        #expect(HubAudioFormat.mainAudioSamples(in: Data()) == nil)
        #expect(HubAudioFormat.mainAudioSamples(in: Data([0x00])) == nil)
        #expect(HubAudioFormat.mainAudioSamples(in: Data([0x00, 0x40])) == nil)
    }

    @Test func dropsATrailingOddByte() throws {
        let samples = try #require(HubAudioFormat.mainAudioSamples(in: Data([0x00, 0x00, 0x40, 0x12])))
        #expect(samples == [0.5])
    }
}
