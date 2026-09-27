import AppKit

/// Keeps the model's changes that the decoder's errors could explain, and undoes the rest.
///
/// The received and corrected words are aligned, and each run of changed words (a hunk) is
/// judged on its own:
/// - Pure spacing fixes (`W1A W` → `W1AW`) and removing a lone E or T are always fine.
/// - A hunk may not change a word that is already valid: a ham term, a callsign, a number,
///   or a correctly spelled English word. The model likes "fixing" IS to AS.
/// - An undecodable `*` may be filled in but not dropped.
/// - Otherwise it must be close in Morse (`MorsePlausibility`) to what was received.
/// A rejected hunk goes back to what was received, minus any lone E or T the model removed.
enum CorrectionFilter {
    struct Outcome: Equatable {
        var text: String
        var accepted: [CleanupResult.Correction]
        var rejected: [CleanupResult.Correction]
    }

    /// Words the decoder produces from noise alone.
    static let noiseWords: Set<String> = ["E", "T"]

    @MainActor
    static func apply(raw: String, cleaned: String, isProtected: (String) -> Bool) -> Outcome {
        let rawWords = words(raw)
        let cleanedWords = words(cleaned)
        let difference = cleanedWords.difference(from: rawWords)
        var removed = Set<Int>()
        var inserted = Set<Int>()
        for change in difference {
            switch change {
            case .remove(let offset, _, _): removed.insert(offset)
            case .insert(let offset, _, _): inserted.insert(offset)
            }
        }

        var outcome = Outcome(text: "", accepted: [], rejected: [])
        var result: [String] = []
        var i = 0
        var j = 0
        while i < rawWords.count || j < cleanedWords.count {
            if i < rawWords.count, j < cleanedWords.count, !removed.contains(i), !inserted.contains(j) {
                result.append(rawWords[i])
                i += 1
                j += 1
                continue
            }
            var received: [String] = []
            while i < rawWords.count, removed.contains(i) {
                received.append(rawWords[i])
                i += 1
            }
            var corrected: [String] = []
            while j < cleanedWords.count, inserted.contains(j) {
                corrected.append(cleanedWords[j])
                j += 1
            }
            let correction = CleanupResult.Correction(
                received: received.joined(separator: " "),
                corrected: corrected.joined(separator: " "),
                reason: reason(received: received, corrected: corrected)
            )
            if accepts(received: received, corrected: corrected, isProtected: isProtected) {
                result += corrected
                outcome.accepted.append(correction)
            } else {
                // Still drop the noise the model removed along with its implausible change.
                result += received.filter { !noiseWords.contains($0) || corrected.contains($0) }
                outcome.rejected.append(correction)
            }
        }
        outcome.text = result.joined(separator: " ")
        return outcome
    }

    private static func accepts(received: [String], corrected: [String], isProtected: (String) -> Bool) -> Bool {
        if received.joined() == corrected.joined() { return true }
        if corrected.isEmpty, received.allSatisfy(noiseWords.contains) { return true }
        if received.contains(where: isProtected) { return false }
        // An undecodable character (*) may be filled in, but dropping it would hide the gap.
        let receivedText = received.joined(), correctedText = corrected.joined()
        if receivedText.count(where: { $0 == "*" }) > correctedText.count(where: { $0 == "*" }),
           correctedText.count < receivedText.count {
            return false
        }
        let marks = MorsePlausibility.elements(receivedText).filter { $0 != .break }.count
        let allowed = max(1.5, MorsePlausibility.maximumChangeRatio * Double(marks))
        return MorsePlausibility.distance(received.joined(separator: " "), corrected.joined(separator: " ")) <= allowed
    }

    private static func reason(received: [String], corrected: [String]) -> String {
        if received.joined() == corrected.joined() { return "spacing" }
        if corrected.isEmpty { return "noise" }
        return "correction"
    }

    private static func words(_ text: String) -> [String] {
        text.uppercased().split(whereSeparator: \.isWhitespace).map(String.init)
    }

    /// A ham term, callsign, number, or correctly spelled English word of two letters or more.
    @MainActor
    static func isValidWord(_ word: String) -> Bool {
        if HamLexicon.lookup(word) != nil || word.allSatisfy(\.isNumber) { return true }
        guard word.count >= 2, word.allSatisfy(\.isLetter) else { return false }
        let range = NSSpellChecker.shared.checkSpelling(
            of: word.lowercased(), startingAt: 0, language: "en", wrap: false,
            inSpellDocumentWithTag: 0, wordCount: nil
        )
        return range.location == NSNotFound
    }
}
