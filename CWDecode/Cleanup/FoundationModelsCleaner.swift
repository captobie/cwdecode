import Foundation
import FoundationModels
import os

/// Corrects decoder output with Apple's on-device language model.
///
/// Every window gets a fresh session, so nothing accumulates in the model's context between
/// requests; the prompt holds the instructions, a little already-cleaned context, dictionary
/// hints and the new text, and is trimmed to leave room for the response.
@available(macOS 26, *)
actor FoundationModelsCleaner: TextCleaner {
    private let model = SystemLanguageModel.default
    /// Off by default: on macOS 27.0 the on-device model never finishes a call to this tool
    /// (it runs until the 4K context overflows, 40–60 s later), so the dictionary is used
    /// through prompt hints and `CorrectionFilter` instead.
    private let tools: [any Tool]
    /// Tokens used by the instructions, tools and output schema, measured once.
    private var fixedTokens: Int?

    /// Held back for the model's answer.
    private static let responseTokens = 500
    private static let logger = Logger(subsystem: "com.carlobermeier.CWDecode", category: "cleanup")

    init(useLookupTool: Bool = false) {
        tools = useLookupTool ? [HamLookupTool()] : []
    }

    nonisolated func availability() -> CleanupAvailability {
        switch model.availability {
        case .available: .available
        case .unavailable(.deviceNotEligible): .deviceNotEligible
        case .unavailable(.appleIntelligenceNotEnabled): .appleIntelligenceNotEnabled
        case .unavailable(.modelNotReady): .modelNotReady
        case .unavailable: .unavailable
        }
    }

    nonisolated func prewarm() {
        makeSession().prewarm()
    }

    func clean(_ request: CleanupRequest) async throws -> CleanupResult {
        // Runs of E and T (TT, TTTET) are noise the model can't read anything into, and greedy
        // decoding on them tends to loop ("MN MN MN …") until it runs out of tokens. The model
        // doesn't see them; CorrectionFilter puts them back, since it compares with the raw text.
        let raw = Self.withoutNoiseRuns(request.raw)
        if raw.isEmpty || Self.isMostlyNoise(request.raw) {
            return CleanupResult(text: request.raw)
        }
        var context = request.context
        var hints = HamLexicon.hints(for: raw)
        while true {
            let prompt = Self.prompt(context: context, hints: hints, raw: raw)
            if try await fits(prompt) {
                do {
                    return try await respond(to: prompt, raw: raw)
                } catch where Self.isContextOverflow(error) && (!context.isEmpty || !hints.isEmpty) {
                    Self.logger.info("Context overflow; retrying with less context")
                }
            }
            // Shed context, then hints. The new text itself is never trimmed.
            if !context.isEmpty {
                context = Self.halved(context)
            } else if !hints.isEmpty {
                hints = []
            } else {
                return try await respond(to: prompt, raw: raw)
            }
        }
    }

    private func respond(to prompt: String, raw: String) async throws -> CleanupResult {
        // A correction is never much longer than its input, so a repetition loop fails fast.
        let maximumTokens = min(Self.responseTokens, 80 + raw.count)
        let response = try await makeSession().respond(
            to: prompt,
            generating: CleanedText.self,
            options: GenerationOptions(samplingMode: .greedy, maximumResponseTokens: maximumTokens)
        )
        return CleanupResult(text: response.content.text, uncertain: response.content.uncertain)
    }

    private nonisolated func makeSession() -> LanguageModelSession {
        LanguageModelSession(model: model, tools: tools, instructions: Self.instructions)
    }

    /// Whether the prompt leaves enough room for the response. Exact on macOS 26.4 and later;
    /// before that, a conservative estimate (decoder output tokenizes poorly, ~3 characters a token).
    private func fits(_ prompt: String) async throws -> Bool {
        let budget = model.contextSize - Self.responseTokens
        if #available(macOS 26.4, *) {
            if fixedTokens == nil {
                fixedTokens = try await model.tokenCount(for: Instructions(Self.instructions))
                    + model.tokenCount(for: tools)
                    + model.tokenCount(for: CleanedText.generationSchema)
            }
            return try await (fixedTokens ?? 0) + model.tokenCount(for: prompt) <= budget
        }
        return (Self.instructions.count + prompt.count) / 3 + 400 <= budget
    }

    /// Whether at least 60 % of the words are made only of E and T.
    static func isMostlyNoise(_ text: String) -> Bool {
        let words = text.split(whereSeparator: \.isWhitespace)
        let noise = words.count(where: isNoiseRun)
        return !words.isEmpty && Double(noise) >= 0.6 * Double(words.count)
    }

    /// Drops words of two or more letters made only of E and T; a lone E or T stays for the model to judge.
    static func withoutNoiseRuns(_ text: String) -> String {
        text.split(whereSeparator: \.isWhitespace)
            .filter { $0.count < 2 || !isNoiseRun($0) }
            .joined(separator: " ")
    }

    private static func isNoiseRun(_ word: Substring) -> Bool {
        word.allSatisfy { $0 == "E" || $0 == "T" }
    }

    private static func halved(_ text: String) -> String {
        let words = text.split(separator: " ")
        return words.suffix(words.count / 2).joined(separator: " ")
    }

    private static func isContextOverflow(_ error: any Error) -> Bool {
        if #available(macOS 27, *), let error = error as? LanguageModelError,
           case .contextSizeExceeded = error {
            return true
        }
        if let error = error as? LanguageModelSession.GenerationError,
           case .exceededContextWindowSize = error {
            return true
        }
        return false
    }

    static func prompt(context: String, hints: [String], raw: String) -> String {
        var prompt = ""
        if !context.isEmpty {
            prompt += "Earlier text, already corrected (context only, do not repeat it):\n\(context)\n\n"
        }
        if !hints.isEmpty {
            prompt += "Dictionary hints (Morse near-misses, not certain):\n"
            prompt += hints.map { "- \($0)" }.joined(separator: "\n") + "\n\n"
        }
        prompt += "Correct this new decoder output:\n\(raw)"
        return prompt
    }

    static let instructions = """
        You fix errors in text from an automatic Morse code (CW) decoder on amateur radio.

        The decoder's errors:
        - A lone E or T between words is noise. Remove it.
        - A misjudged gap splits or joins letters: EE/I, TT/M, ET/A, TE/N, EEE/S, SE/H, "W1A W"/"W1AW".
        - * is a letter the decoder could not read.

        The text is ham radio shorthand, not English sentences: callsigns (W1AW, DL2ABC), \
        Q-codes (QTH, QRZ, QSY, QSL, QRM, QRN, QSB), abbreviations (CQ DE TU UR RST ES FB OM \
        PSE TNX AGN HR NAME RIG ANT WX 73), signal reports (599, 5NN) and prosigns (K, KN, AR, SK, BK).

        Change only what these errors explain. Leave every other word exactly as it is. \
        Never add, reorder or rephrase words. If unsure, keep the word and list it as uncertain.

        Examples:
        Input: CQ CQ E DE W1AW W1AW T K
        Output: CQ CQ DE W1AW W1AW K
        Input: UR RST 5NN T QTSE BOSTON
        Output: UR RST 5NN QTH BOSTON
        Input: NAME IS BOB E HW CPY
        Output: NAME IS BOB HW CPY
        """
}

@available(macOS 26, *)
@Generable
struct CleanedText {
    @Guide(description: "The corrected version of the new decoder output only.")
    var text: String

    @Guide(description: "Words in the corrected text you could not resolve confidently.", .maximumCount(8))
    var uncertain: [String]
}

/// Lets the model check candidates against `HamLexicon` instead of relying on its own priors.
@available(macOS 26, *)
struct HamLookupTool: Tool {
    let name = "lookupHamTerm"
    let description = """
        Checks whether a word is a valid amateur radio callsign, Q-code, prosign, abbreviation \
        or signal report, and lists dictionary words it is close to in Morse.
        """

    @Generable
    struct Arguments {
        @Guide(description: "One word from the text, such as QTH or W1AW")
        var term: String
    }

    func call(arguments: Arguments) async throws -> String {
        HamLexicon.describe(arguments.term)
    }
}
