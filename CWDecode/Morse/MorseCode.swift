import Foundation

/// International Morse code table. Patterns use "." for dit and "-" for dah.
enum MorseCode {
    static let patterns: [String: String] = [
        ".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E", "..-.": "F",
        "--.": "G", "....": "H", "..": "I", ".---": "J", "-.-": "K", ".-..": "L",
        "--": "M", "-.": "N", "---": "O", ".--.": "P", "--.-": "Q", ".-.": "R",
        "...": "S", "-": "T", "..-": "U", "...-": "V", ".--": "W", "-..-": "X",
        "-.--": "Y", "--..": "Z",

        "-----": "0", ".----": "1", "..---": "2", "...--": "3", "....-": "4",
        ".....": "5", "-....": "6", "--...": "7", "---..": "8", "----.": "9",

        ".-.-.-": ".", "--..--": ",", "..--..": "?", ".----.": "'", "-.-.--": "!",
        "-..-.": "/", "-.--.": "(", "-.--.-": ")", ".-...": "&", "---...": ":",
        "-.-.-.": ";", "-...-": "=", ".-.-.": "+", "-....-": "-", "..--.-": "_",
        ".-..-.": "\"", "...-..-": "$", ".--.-.": "@",

        // Prosigns without a single-character equivalent.
        "...-.-": "<SK>", "-.-.-": "<KA>", "...-.": "<SN>", "........": "<HH>",
    ]

    /// Shown for patterns that aren't in the table.
    static let unknown = "*"

    static func decode(_ pattern: String) -> String {
        patterns[pattern] ?? unknown
    }

    private static let encodings: [Character: String] = {
        var result: [Character: String] = [:]
        for (pattern, text) in patterns where text.count == 1 {
            result[Character(text)] = pattern
        }
        return result
    }()

    static func encode(_ character: Character) -> String? {
        encodings[Character(character.uppercased())]
    }
}
