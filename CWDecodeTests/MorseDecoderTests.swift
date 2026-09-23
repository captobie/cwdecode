import Testing
@testable import CWDecode

struct MorseDecoderTests {
    private func decode(_ elements: [MorseSynthesizer.Element], initialWPM: Double = 20) -> (text: String, decoder: MorseDecoder) {
        var decoder = MorseDecoder(initialWPM: initialWPM)
        var text = ""
        for element in elements {
            if element.keyDown {
                decoder.mark(element.duration)
            } else {
                text += decoder.space(element.duration)
            }
        }
        text += decoder.space(.infinity)
        return (text.trimmingCharacters(in: .whitespaces), decoder)
    }

    @Test func encodesAndDecodesEveryTableEntry() {
        for (pattern, text) in MorseCode.patterns where text.count == 1 {
            #expect(MorseCode.encode(Character(text)) == pattern)
        }
        #expect(MorseCode.decode(".-.-.-.-.-") == MorseCode.unknown)
    }

    @Test func decodesPerfectTimingAtTheInitialSpeed() {
        let text = "CQ CQ DE W1AW K"
        #expect(decode(MorseSynthesizer.elements(for: text, wpm: 20)).text == text)
    }

    @Test(arguments: [8.0, 12.0, 30.0, 45.0])
    func adaptsToOtherSpeeds(wpm: Double) {
        let result = decode(MorseSynthesizer.elements(for: "PARIS PARIS PARIS PARIS PARIS", wpm: wpm))
        #expect(result.text.hasSuffix("PARIS PARIS PARIS"), "decoded \(result.text)")
        #expect(abs(result.decoder.estimatedWPM - wpm) / wpm < 0.1)
    }

    @Test func learnsFarnsworthSpacing() {
        // Letters at 18 WPM, gaps stretched to 10 WPM: letter gaps look like word gaps
        // until the decoder has seen a few of them.
        let elements = MorseSynthesizer.elements(for: "THE QUICK BROWN FOX 73", wpm: 18, farnsworthWPM: 10)
        let decoded = decode(elements, initialWPM: 18).text
        #expect(decoded.hasSuffix("QUICK BROWN FOX 73"), "decoded \(decoded)")
    }

    @Test func emitsEachBreakOnceWhileAGapGrows() {
        var decoder = MorseDecoder(initialWPM: 20)
        decoder.mark(0.06)
        var text = ""
        for step in 1...100 {
            text += decoder.space(Double(step) * 0.01)
        }
        #expect(text == "E ")
        #expect(decoder.pendingSymbols.isEmpty)
    }
}
