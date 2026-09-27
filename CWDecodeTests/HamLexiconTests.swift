import Testing
@testable import CWDecode

struct HamLexiconTests {
    @Test(arguments: ["W1AW", "K1ABC", "VE3XYZ", "DL2ABC", "G4ABC", "2E0ABC", "JA1XYZ", "HB9ABC", "9A1A", "W1AW/P", "DL/W1AW", "S51A"])
    func recognizesCallsigns(callsign: String) {
        #expect(HamLexicon.lookup(callsign)?.kind == .callsign)
    }

    @Test(arguments: ["QTH", "CQ", "73", "5NN", "599", "TEST", "ETE", "W1", "1AW", "WWWW"])
    func rejectsNonCallsigns(word: String) {
        #expect(HamLexicon.callsignDescription(word) == nil)
    }

    @Test func knowsCountries() {
        #expect(HamLexicon.callsignDescription("DL2ABC") == "Germany")
        #expect(HamLexicon.callsignDescription("KH6ABC") == "Hawaii")
        #expect(HamLexicon.callsignDescription("S51A") == "Slovenia")
    }

    @Test func looksUpCodes() {
        #expect(HamLexicon.lookup("qth")?.kind == .qCode)
        #expect(HamLexicon.lookup("QRZ?")?.kind == .qCode)
        #expect(HamLexicon.lookup("KN")?.kind == .prosign)
        #expect(HamLexicon.lookup("<SK>")?.kind == .prosign)
        #expect(HamLexicon.lookup("TU")?.kind == .abbreviation)
        #expect(HamLexicon.lookup("579")?.kind == .signalReport)
        #expect(HamLexicon.lookup("5NN")?.kind == .signalReport)
        #expect(HamLexicon.lookup("699") == nil)
    }

    @Test func suggestsMorseNearMisses() {
        // H (....) read as E+I or S+E: "QTEI" is a split H.
        #expect(HamLexicon.nearMatches(for: "QTEI").contains("QTH"))
        // T (-) and M (--) are one dah apart.
        #expect(HamLexicon.nearMatches(for: "QRT").contains("QRM"))
        #expect(HamLexicon.hints(for: "CQ QTEI DE W1AW").contains { $0.hasPrefix("QTEI may be QTH") })
        #expect(HamLexicon.hints(for: "DE DL2A BC K").contains("DL2A BC may be DL2ABC"))
        #expect(!HamLexicon.hints(for: "DE W1AW E K").contains { $0.contains("W1AWE") })
    }

    @Test func describesForTheTool() {
        #expect(HamLexicon.describe("qsy").contains("valid Q-code"))
        #expect(HamLexicon.describe("QTEI").contains("QTH"))
        #expect(HamLexicon.describe("XYZZY").contains("not a known"))
    }
}

struct MorsePlausibilityTests {
    @Test func gapErrorsAreCheap() {
        #expect(MorsePlausibility.distance("EE", "I") < 1)
        #expect(MorsePlausibility.distance("C Q", "CQ") == 0)
        #expect(MorsePlausibility.distance("QTSE", "QTH") < 1)
    }

    @Test func differentSoundingWordsAreFar() {
        #expect(MorsePlausibility.distance("TU 73", "THANK YOU") > 5)
        #expect(MorsePlausibility.distance("RPRT", "QSY") > 3)
    }
}

@MainActor
struct CorrectionFilterTests {
    private func filter(_ raw: String, _ cleaned: String) -> CorrectionFilter.Outcome {
        CorrectionFilter.apply(raw: raw, cleaned: cleaned, isProtected: CorrectionFilter.isValidWord)
    }

    @Test func keepsNoiseRemovalAndSpacingFixes() {
        #expect(filter("CQ CQ E DE W1AW T K", "CQ CQ DE W1AW K").text == "CQ CQ DE W1AW K")
        #expect(filter("CQ CQ DE W1A W K", "CQ CQ DE W1AW K").text == "CQ CQ DE W1AW K")
        #expect(filter("CQ DE DL2A BC", "CQ DE DL2ABC").text == "CQ DE DL2ABC")
        let outcome = filter("UR RST 5NN QTEI BOSTON", "UR RST 5NN QTH BOSTON")
        #expect(outcome.text == "UR RST 5NN QTH BOSTON")
        #expect(outcome.accepted == [CleanupResult.Correction(received: "QTEI", corrected: "QTH", reason: "correction")])
    }

    @Test func undoesChangesToValidWords() {
        #expect(filter("RIG IS KX3 E ES ANT", "RIG AS KX3 ES ANT").text == "RIG IS KX3 ES ANT")
        #expect(filter("WX IS WARM", "WX IS QRM").text == "WX IS WARM")
        #expect(filter("CQ DE DL2A BC", "CQ DE DL2BK").text == "CQ DE DL2A BC")
    }

    @Test func undoesRewritesButKeepsTheNoiseRemoval() {
        let outcome = filter("PSE E AGN", "PSE")
        #expect(outcome.text == "PSE AGN")
        #expect(outcome.rejected.count == 1)
        #expect(filter("CQ CQ DE W1AW K", "HELLO THERE HOW ARE YOU").text == "CQ CQ DE W1AW K")
        #expect(filter("TU 73", "THANK YOU BEST REGARDS").text == "TU 73")
    }

    @Test func fillsButNeverDropsUndecodableCharacters() {
        #expect(filter("HW C*Y", "HW CPY").text == "HW CPY")
        #expect(filter("DE K1UB*", "DE K1UB").text == "DE K1UB*")
    }

    @Test func mayNotInventText() {
        #expect(filter("CQ CQ", "CQ CQ DE W1AW K").text == "CQ CQ")
    }
}
