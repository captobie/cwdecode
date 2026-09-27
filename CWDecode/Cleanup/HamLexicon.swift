import Foundation

/// A small built-in dictionary of amateur radio vocabulary. It hints likely corrections in the
/// prompt, protects valid words in `CorrectionFilter`, and backs the optional lookup tool.
enum HamLexicon {
    enum Kind: String, Sendable {
        case qCode = "Q-code"
        case prosign
        case abbreviation
        case signalReport = "signal report"
        case callsign
    }

    struct Entry: Equatable, Sendable {
        var kind: Kind
        var meaning: String
    }

    static let qCodes: [String: String] = [
        "QRA": "name of station", "QRG": "exact frequency", "QRK": "readability",
        "QRL": "frequency is busy", "QRM": "interference from other stations",
        "QRN": "static or atmospheric noise", "QRO": "increase power", "QRP": "low power",
        "QRQ": "send faster", "QRS": "send slower", "QRT": "stop sending, closing down",
        "QRU": "nothing more for you", "QRV": "ready", "QRX": "wait, stand by",
        "QRZ": "who is calling me?", "QSB": "signal is fading", "QSK": "full break-in",
        "QSL": "acknowledge receipt, confirmation card", "QSO": "contact, conversation",
        "QSP": "relay a message", "QST": "general call to all amateurs",
        "QSX": "listening on another frequency", "QSY": "change frequency",
        "QTC": "messages to send", "QTH": "location", "QTR": "time",
    ]

    static let prosigns: [String: String] = [
        "K": "go ahead, over", "KN": "go ahead, named station only", "AR": "end of message",
        "SK": "end of contact", "<SK>": "end of contact", "BK": "break, back to you",
        "BT": "separator", "AS": "wait", "<KA>": "start of message", "<SN>": "understood",
        "<HH>": "error", "R": "received", "CL": "closing station",
    ]

    static let abbreviations: [String: String] = [
        "CQ": "calling any station", "DE": "from, this is", "TU": "thank you",
        "73": "best regards", "88": "love and kisses", "UR": "your, you are",
        "RST": "readability, strength, tone report", "ES": "and", "FB": "fine business, excellent",
        "OM": "old man, operator", "YL": "young lady, operator", "XYL": "wife",
        "OP": "operator", "NAME": "name", "HR": "here", "WX": "weather", "RIG": "radio",
        "ANT": "antenna", "PWR": "power", "W": "watts", "TNX": "thanks", "TKS": "thanks",
        "PSE": "please", "AGN": "again", "CPY": "copy", "SRI": "sorry", "HW": "how copy?",
        "CUL": "see you later", "CUAGN": "see you again", "GM": "good morning",
        "GA": "good afternoon, go ahead", "GE": "good evening", "GN": "good night",
        "GL": "good luck", "DX": "long distance", "TEST": "contest call", "NR": "number",
        "ABT": "about", "BURO": "QSL bureau", "DR": "dear", "FER": "for", "HPE": "hope",
        "INFO": "information", "MNI": "many", "OB": "old boy", "RPT": "report, repeat",
        "SIG": "signal", "SOLID": "solid copy", "VY": "very", "WID": "with", "WL": "well, will",
        "WKD": "worked", "ENUF": "enough", "HI": "laughter",
        "BCNU": "be seeing you", "AA": "all after", "AB": "all before", "EU": "Europe",
        "NA": "North America", "SA": "South America", "AF": "Africa",
        "OC": "Oceania", "POTA": "Parks on the Air", "SOTA": "Summits on the Air",
    ]

    /// A few common ITU prefixes, enough to tell a plausible callsign from noise.
    static let prefixes: [String: String] = [
        "K": "USA", "W": "USA", "N": "USA", "AA": "USA", "AB": "USA", "AC": "USA", "AD": "USA",
        "AE": "USA", "AF": "USA", "AG": "USA", "AI": "USA", "AJ": "USA", "AK": "USA",
        "KH6": "Hawaii", "KL7": "Alaska", "KP4": "Puerto Rico",
        "VE": "Canada", "VA": "Canada", "VO": "Canada", "VY": "Canada", "XE": "Mexico",
        "G": "England", "M": "England", "2E": "England", "GW": "Wales", "GM": "Scotland",
        "GI": "Northern Ireland", "EI": "Ireland", "F": "France", "DL": "Germany",
        "DK": "Germany", "DJ": "Germany", "DF": "Germany", "DG": "Germany", "DH": "Germany",
        "DB": "Germany", "DC": "Germany", "DD": "Germany", "DM": "Germany", "DO": "Germany",
        "ON": "Belgium", "PA": "Netherlands", "PD": "Netherlands", "PE": "Netherlands",
        "OZ": "Denmark", "SM": "Sweden", "SA": "Sweden", "LA": "Norway", "OH": "Finland",
        "OE": "Austria", "HB": "Switzerland", "HB9": "Switzerland", "I": "Italy", "IK": "Italy",
        "IZ": "Italy", "EA": "Spain", "EB": "Spain", "EC": "Spain", "CT": "Portugal",
        "SP": "Poland", "SQ": "Poland", "OK": "Czech Republic", "OM": "Slovakia",
        "HA": "Hungary", "YO": "Romania", "LZ": "Bulgaria", "SV": "Greece", "9A": "Croatia",
        "S5": "Slovenia", "YU": "Serbia", "UA": "Russia", "RA": "Russia", "R": "Russia",
        "UR": "Ukraine", "UT": "Ukraine", "LY": "Lithuania", "YL": "Latvia", "ES": "Estonia",
        "TA": "Turkey", "4X": "Israel", "4Z": "Israel", "JA": "Japan", "JH": "Japan",
        "JR": "Japan", "JE": "Japan", "JF": "Japan", "JG": "Japan", "JI": "Japan",
        "HL": "South Korea", "BY": "China", "BV": "Taiwan", "VU": "India", "VK": "Australia",
        "ZL": "New Zealand", "ZS": "South Africa", "PY": "Brazil", "PU": "Brazil",
        "LU": "Argentina", "CE": "Chile", "CX": "Uruguay", "HK": "Colombia", "YV": "Venezuela",
        "OA": "Peru", "KP2": "US Virgin Islands", "VP9": "Bermuda", "ZF": "Cayman Islands",
        "EA8": "Canary Islands", "CT3": "Madeira", "TF": "Iceland", "OX": "Greenland",
    ]

    /// ITU structure: a 1–2 character prefix, a digit, and a suffix of up to four characters
    /// ending in a letter; optionally a country prefix before and a /P, /M, /QRP etc. after.
    private nonisolated(unsafe) static let callsignPattern =
        #/(?:[A-Z0-9]{1,3}/)?(?<prefix>[A-Z]{1,2}|[0-9][A-Z]|[A-Z][0-9])(?<digit>[0-9])[A-Z0-9]{0,3}[A-Z](?:/[A-Z0-9]{1,4})?/#

    /// Readability 1–5, strength and tone 1–9, with the cut number N for 9.
    private nonisolated(unsafe) static let reportPattern = #/[1-5][1-9N][1-9N]/#

    static func isSignalReport(_ word: String) -> Bool {
        word.uppercased().wholeMatch(of: reportPattern) != nil
    }

    /// The country for a well-formed callsign, a note when the format is right but the
    /// prefix isn't in the table, or nil when it isn't a callsign at all.
    static func callsignDescription(_ word: String) -> String? {
        let upper = word.uppercased()
        guard let match = upper.wholeMatch(of: callsignPattern) else { return nil }
        let prefix = String(match.prefix)
        let digit = String(match.digit)
        for candidate in [prefix + digit, prefix, String(prefix.prefix(1))] {
            if let country = prefixes[candidate] { return country }
        }
        return "valid format, prefix not in the built-in table"
    }

    static func lookup(_ word: String) -> Entry? {
        let upper = word.uppercased()
        // Q-codes are also sent as questions: QTH?
        let code = upper.hasSuffix("?") ? String(upper.dropLast()) : upper
        if let meaning = qCodes[code] {
            return Entry(kind: .qCode, meaning: meaning)
        }
        if let meaning = prosigns[upper] { return Entry(kind: .prosign, meaning: meaning) }
        if let meaning = abbreviations[upper] { return Entry(kind: .abbreviation, meaning: meaning) }
        if isSignalReport(upper) { return Entry(kind: .signalReport, meaning: "RST report") }
        if let description = callsignDescription(upper) { return Entry(kind: .callsign, meaning: description) }
        return nil
    }

    /// Every dictionary word, for finding near misses.
    private static let vocabulary: [String] = Array(Set(qCodes.keys).union(prosigns.keys).union(abbreviations.keys)).sorted()

    /// Dictionary words that sound (in Morse) almost like `word`, closest first.
    static func nearMatches(for word: String, limit: Int = 3) -> [String] {
        let upper = word.uppercased()
        let elements = MorsePlausibility.elements(upper)
        let markCount = elements.filter { $0 != .break }.count
        guard markCount >= 3 else { return [] }
        let allowed = min(2.0, 0.25 * Double(markCount))
        var matches: [(word: String, distance: Double)] = []
        for candidate in vocabulary where candidate != upper && candidate.count >= 2 {
            let distance = MorsePlausibility.distance(elements, MorsePlausibility.elements(candidate))
            if distance <= allowed { matches.append((candidate, distance)) }
        }
        matches.sort { ($0.distance, $0.word) < ($1.distance, $1.word) }
        return matches.prefix(limit).map(\.word)
    }

    /// Lone letters the decoder makes out of noise; never worth joining onto a word.
    private static let noiseLetters: Set<String> = ["E", "T"]

    /// Plain-text answer for the lookup tool.
    static func describe(_ word: String) -> String {
        let upper = word.trimmingCharacters(in: .whitespacesAndNewlines).uppercased()
        if let entry = lookup(upper) {
            return "\(upper) is a valid \(entry.kind.rawValue): \(entry.meaning)."
        }
        let matches = nearMatches(for: upper)
        if matches.isEmpty {
            return "\(upper) is not a known callsign, Q-code, prosign, abbreviation or signal report."
        }
        return "\(upper) is not a known term. In Morse it is close to: \(matches.joined(separator: ", "))."
    }

    /// Likely corrections for words in a window that aren't recognized, for the prompt.
    static func hints(for text: String, limit: Int = 8) -> [String] {
        var seen = Set<String>()
        var result: [String] = []
        let words = text.split(whereSeparator: \.isWhitespace).map(String.init)
        // A letter gap misread as a word gap splits a callsign or code in two: DL2A BC.
        for (first, second) in zip(words, words.dropFirst()) where result.count < limit {
            let joined = first + second
            guard lookup(second) == nil, !noiseLetters.contains(second), lookup(joined) != nil,
                  seen.insert(joined).inserted else { continue }
            result.append("\(first) \(second) may be \(joined)")
        }
        for word in words {
            guard result.count < limit, seen.insert(word).inserted, lookup(word) == nil,
                  !word.allSatisfy(\.isNumber) else { continue }
            let matches = nearMatches(for: word, limit: 2)
            if !matches.isEmpty {
                result.append("\(word) may be \(matches.joined(separator: " or "))")
            }
        }
        return result
    }
}
