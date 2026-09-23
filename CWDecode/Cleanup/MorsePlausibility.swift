import Foundation

/// Measures how far apart two texts are *as Morse code*, which is how the decoder errs.
///
/// Both texts become a sequence of dits, dahs and breaks. Adding, dropping or flipping a
/// dit or dah costs 1; moving a break (a character or word gap) costs `breakCost`. So
/// "EE" → "I" or "C Q" → "CQ" is nearly free, dropping a stray "E" costs a little over 1,
/// and replacing a word with something that sounds nothing like it costs a lot.
enum MorsePlausibility {
    static let breakCost = 0.4

    /// The most of the received dits and dahs a correction may change.
    static let maximumChangeRatio = 0.35

    static func distance(_ a: String, _ b: String) -> Double {
        distance(elements(a), elements(b))
    }

    enum Element: Equatable {
        case dit, dah, `break`
    }

    /// Characters without a Morse encoding (like `*`) become one unknown element (nil), which
    /// `distance` lets stand for any character.
    static func elements(_ text: String) -> [Element?] {
        var result: [Element?] = []
        for token in tokens(text) {
            if !result.isEmpty, result.last != .break {
                result.append(.break)
            }
            if token == " " {
                continue
            }
            if let pattern = pattern(for: token) {
                result += pattern.map { $0 == "." ? .dit : .dah }
            } else {
                result.append(nil)
            }
        }
        if result.last == .break { result.removeLast() }
        return result
    }

    /// Characters and whole prosigns like `<SK>`; any run of whitespace becomes one " ".
    private static func tokens(_ text: String) -> [String] {
        var result: [String] = []
        var prosign: String?
        for character in text.uppercased() {
            if var current = prosign {
                current.append(character)
                if character == ">" {
                    result.append(current)
                    prosign = nil
                } else {
                    prosign = current
                }
            } else if character == "<" {
                prosign = "<"
            } else if character.isWhitespace {
                if result.last != " " { result.append(" ") }
            } else {
                result.append(String(character))
            }
        }
        if let prosign { result += prosign.map(String.init) }
        return result
    }

    private static let prosignPatterns: [String: String] = {
        var result: [String: String] = [:]
        for (pattern, text) in MorseCode.patterns where text.count > 1 {
            result[text] = pattern
        }
        return result
    }()

    private static func pattern(for token: String) -> String? {
        if token.count == 1, let character = token.first {
            return MorseCode.encode(character)
        }
        return prosignPatterns[token]
    }

    /// Weighted edit distance. An unknown element (nil) matches any single character cheaply.
    static func distance(_ a: [Element?], _ b: [Element?]) -> Double {
        func cost(_ element: Element?) -> Double { element == .break ? breakCost : 1 }
        func substitution(_ x: Element?, _ y: Element?) -> Double {
            if x == y { return 0 }
            if x == .break || y == .break { return cost(x) + cost(y) }
            if x == nil || y == nil { return 0.5 }
            return 1
        }

        var previous = [0.0]
        for element in b { previous.append(previous.last! + cost(element)) }
        var current = previous
        for x in a {
            current[0] = previous[0] + cost(x)
            for (j, y) in b.enumerated() {
                // An unknown character in `a` stands for a whole character, so the rest of its
                // dits and dahs come almost free.
                let insertion = x == nil && y != .break ? 0.1 : cost(y)
                current[j + 1] = min(
                    previous[j + 1] + cost(x),
                    current[j] + insertion,
                    previous[j] + substitution(x, y)
                )
            }
            swap(&previous, &current)
        }
        return previous[b.count]
    }
}
