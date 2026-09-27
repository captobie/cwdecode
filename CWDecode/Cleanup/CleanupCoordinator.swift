import AppKit
import Observation
import os

/// Layer 2: feeds decoder output through a `TextCleaner` in small windows of complete words.
///
/// The raw decoder text is never touched; this keeps its own list of segments, each holding
/// the raw text it covers and, once the model has answered, the corrected text. Anything that
/// goes wrong (no model, an error) leaves that segment showing its raw text, and changes the
/// decoder's errors can't explain are undone by `CorrectionFilter`.
@MainActor
@Observable
final class CleanupCoordinator {
    struct Configuration {
        /// Words to wait for before sending a window while text keeps arriving.
        var minimumWords = 8
        /// The most new words in one window.
        var maximumWords = 30
        /// Already-cleaned words sent along as context.
        var contextWords = 20
        /// Windows' worth of words that may wait while a request runs; older ones are skipped.
        var backlogWindows = 3
        /// After this long without new text, send whatever complete words there are.
        var idleDelay = Duration.milliseconds(1500)
        var availabilityPollInterval = Duration.seconds(30)
        var maximumTextLength = 100_000
        /// Words the model may not change into something else.
        var isProtected: @MainActor (String) -> Bool = CorrectionFilter.isValidWord
    }

    struct Segment: Identifiable, Equatable {
        enum State: Equatable {
            case cleaning
            case cleaned(CleanupResult)
            /// Shown as received.
            case unchanged(Reason)
        }

        enum Reason: Equatable {
            /// Received before cleanup started, or only whitespace.
            case passthrough
            case failed
            /// Text arrived faster than the model could keep up.
            case skipped
        }

        enum Style: Equatable {
            case plain
            /// Not (yet) checked by the model.
            case unverified
            /// Changed by the model.
            case changed
            /// The model wasn't sure about it.
            case uncertain
        }

        struct Piece: Equatable {
            var text: String
            var style: Style
        }

        let id: Int
        /// Decoder output this segment covers, including its trailing space or newline.
        let raw: String
        var state: State

        var text: String { pieces.map(\.text).joined() }

        var pieces: [Piece] {
            guard case .cleaned(let result) = state else {
                return [Piece(text: raw, style: state == .unchanged(.passthrough) ? .plain : .unverified)]
            }
            return Self.cleanedPieces(raw: raw, result: result)
        }

        /// Words the model added or changed are marked, compared with what was received.
        private static func cleanedPieces(raw: String, result: CleanupResult) -> [Piece] {
            let rawWords = raw.split(whereSeparator: \.isWhitespace).map(String.init)
            let words = result.text.split(whereSeparator: \.isWhitespace).map(String.init)
            guard !words.isEmpty else { return [] }
            var inserted = Set<Int>()
            for case .insert(let offset, _, _) in words.difference(from: rawWords) {
                inserted.insert(offset)
            }
            let uncertain = Set(result.uncertain.map { $0.uppercased() })
            let separator = raw.last?.isNewline == true ? "\n" : " "
            return words.enumerated().map { index, word in
                let style: Style = uncertain.contains(word) ? .uncertain : inserted.contains(index) ? .changed : .plain
                return Piece(text: word + (index == words.count - 1 ? separator : " "), style: style)
            }
        }
    }

    private(set) var segments: [Segment] = []
    /// Decoder output not yet in a segment, usually a few words and a partial one.
    private(set) var unsegmented = ""
    private(set) var availability: CleanupAvailability

    var isEnabled: Bool {
        didSet {
            guard isEnabled != oldValue else { return }
            defaults.set(isEnabled, forKey: Self.enabledKey)
            activeChanged(from: oldValue && availability == .available)
        }
    }

    /// Enabled by the user and supported here.
    var isActive: Bool { isEnabled && availability == .available }

    /// A request is running.
    var isCleaning: Bool { inFlight != nil }

    /// Cleaned where available, raw elsewhere; what Copy puts on the pasteboard.
    var text: String { segments.map(\.text).joined() + unsegmented }

    /// Supplies the decoder text received so far, so it can be shown when cleanup starts mid-session.
    @ObservationIgnored var existingText: () -> String = { "" }

    @ObservationIgnored private let cleaner: any TextCleaner
    @ObservationIgnored private let configuration: Configuration
    @ObservationIgnored private let defaults: UserDefaults
    @ObservationIgnored private var inFlight: Task<Void, Never>?
    @ObservationIgnored private var nextID = 0
    @ObservationIgnored private var generation = 0
    @ObservationIgnored private var flushRequested = false
    @ObservationIgnored private var lastAppend = ContinuousClock.now
    @ObservationIgnored private var idleTask: Task<Void, Never>?
    @ObservationIgnored private var pollTask: Task<Void, Never>?
    @ObservationIgnored private var activationObserver: (any NSObjectProtocol)?

    private static let enabledKey = "smartCleanupEnabled"
    private static let logger = Logger(subsystem: "com.carlobermeier.CWDecode", category: "cleanup")

    init(cleaner: any TextCleaner = TextCleaners.makeDefault(),
         configuration: Configuration = Configuration(),
         defaults: UserDefaults = .standard) {
        self.cleaner = cleaner
        self.configuration = configuration
        self.defaults = defaults
        defaults.register(defaults: [Self.enabledKey: true])
        isEnabled = defaults.bool(forKey: Self.enabledKey)
        availability = cleaner.availability()
        Self.logger.info("Smart cleanup availability: \(String(describing: self.availability), privacy: .public)")
        if isActive { cleaner.prewarm() }

        // Apple Intelligence can be turned on, finish downloading, or be turned off at any time.
        activationObserver = NotificationCenter.default.addObserver(
            forName: NSApplication.didBecomeActiveNotification, object: nil, queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated { self?.refreshAvailability() }
        }
        let interval = configuration.availabilityPollInterval
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: interval)
                self?.refreshAvailability()
            }
        }
    }

    isolated deinit {
        pollTask?.cancel()
        idleTask?.cancel()
        inFlight?.cancel()
        if let activationObserver { NotificationCenter.default.removeObserver(activationObserver) }
    }

    // MARK: - Input

    /// New decoder output, exactly as it was appended to the raw text.
    func append(_ text: String) {
        guard isActive, !text.isEmpty else { return }
        unsegmented += text
        lastAppend = .now
        trimToMaximumLength()
        skipBacklog()
        scheduleIdleCheck()
        pump()
    }

    /// The decoder finished; clean whatever is left, including a partial last word.
    func flush() {
        guard isActive else { return }
        flushRequested = true
        pump()
    }

    func reset() {
        generation += 1
        inFlight?.cancel()
        inFlight = nil
        idleTask?.cancel()
        segments = []
        unsegmented = ""
        flushRequested = false
    }

    func refreshAvailability() {
        let newAvailability = cleaner.availability()
        guard newAvailability != availability else { return }
        let wasActive = isActive
        Self.logger.info("Smart cleanup availability: \(String(describing: newAvailability), privacy: .public)")
        availability = newAvailability
        activeChanged(from: wasActive)
    }

    private func activeChanged(from wasActive: Bool) {
        guard isActive != wasActive else { return }
        reset()
        guard isActive else { return }
        cleaner.prewarm()
        let existing = existingText()
        if !existing.isEmpty {
            appendSegment(raw: existing, state: .unchanged(.passthrough))
        }
    }

    // MARK: - Windows

    private func scheduleIdleCheck() {
        idleTask?.cancel()
        let delay = configuration.idleDelay
        idleTask = Task { [weak self] in
            try? await Task.sleep(for: delay)
            guard !Task.isCancelled else { return }
            self?.pump()
        }
    }

    /// Starts the next request if none is running and enough text is waiting.
    private func pump() {
        guard isActive, inFlight == nil else { return }
        let idle = ContinuousClock.now - lastAppend >= configuration.idleDelay
        guard let chunk = Self.nextChunk(of: unsegmented, maximumWords: configuration.maximumWords,
                                         includePartial: flushRequested) else {
            if flushRequested, unsegmented.allSatisfy(\.isWhitespace) {
                flushRequested = false
            }
            return
        }
        guard chunk.words >= configuration.minimumWords || chunk.endsLine || flushRequested || idle else { return }

        let raw = String(unsegmented[..<chunk.end])
        unsegmented.removeSubrange(..<chunk.end)
        let words = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !words.isEmpty else {
            appendSegment(raw: raw, state: .unchanged(.passthrough))
            pump()
            return
        }

        let id = appendSegment(raw: raw, state: .cleaning)
        let request = CleanupRequest(context: contextText(), raw: words)
        let generation = self.generation
        let cleaner = self.cleaner
        inFlight = Task { [weak self] in
            let start = ContinuousClock.now
            let result: Result<CleanupResult, any Error>
            do {
                result = .success(try await cleaner.clean(request))
            } catch {
                result = .failure(error)
            }
            self?.finish(id: id, request: request, result: result,
                         elapsed: ContinuousClock.now - start, generation: generation)
        }
    }

    private func finish(id: Int, request: CleanupRequest, result: Result<CleanupResult, any Error>,
                        elapsed: Duration, generation: Int) {
        guard generation == self.generation else { return }
        inFlight = nil
        defer { pump() }
        guard let index = segments.firstIndex(where: { $0.id == id }) else { return }

        switch result {
        case .success(let answer):
            let outcome = CorrectionFilter.apply(raw: request.raw, cleaned: answer.text,
                                                 isProtected: configuration.isProtected)
            let words = Set(outcome.text.split(separator: " ").map(String.init))
            segments[index].state = .cleaned(CleanupResult(
                text: outcome.text,
                corrections: outcome.accepted,
                uncertain: answer.uncertain.map { $0.uppercased() }.filter(words.contains)
            ))
            Self.logger.debug("""
                Cleaned in \(elapsed, privacy: .public): \(request.raw, privacy: .public) → \
                \(outcome.text, privacy: .public) (model: \(answer.text, privacy: .public))
                """)
            for correction in outcome.rejected {
                Self.logger.info("Undid implausible change \(correction.received, privacy: .public) → \(correction.corrected, privacy: .public)")
            }
        case .failure(let error):
            segments[index].state = .unchanged(.failed)
            Self.logger.error("Cleanup failed for \(request.raw, privacy: .public): \(error, privacy: .public)")
            refreshAvailability()
        }
    }

    /// While a request runs, keeps at most a few windows waiting; older text stays raw.
    private func skipBacklog() {
        guard inFlight != nil else { return }
        let limit = configuration.backlogWindows * configuration.maximumWords
        while Self.completeWordCount(unsegmented) > limit,
              let chunk = Self.nextChunk(of: unsegmented, maximumWords: configuration.maximumWords, includePartial: false) {
            appendSegment(raw: String(unsegmented[..<chunk.end]), state: .unchanged(.skipped))
            unsegmented.removeSubrange(..<chunk.end)
            Self.logger.info("Skipped \(chunk.words) words: cleanup can't keep up")
        }
    }

    @discardableResult
    private func appendSegment(raw: String, state: Segment.State) -> Int {
        nextID += 1
        segments.append(Segment(id: nextID, raw: raw, state: state))
        return nextID
    }

    private func trimToMaximumLength() {
        var length = segments.reduce(unsegmented.count) { $0 + $1.raw.count }
        while length > configuration.maximumTextLength, let first = segments.first {
            length -= first.raw.count
            segments.removeFirst()
        }
    }

    private func contextText() -> String {
        var words: [Substring] = []
        for segment in segments.reversed() where segment.state != .cleaning {
            words = segment.text.split(whereSeparator: \.isWhitespace) + words
            if words.count >= configuration.contextWords { break }
        }
        return words.suffix(configuration.contextWords).joined(separator: " ")
    }

    private static func completeWordCount(_ text: String) -> Int {
        zip(text, text.dropFirst()).count { !$0.isWhitespace && $1.isWhitespace }
    }

    /// The end of the next window: up to `maximumWords` complete words (a word is complete once
    /// whitespace follows it), stopping after a newline. With `includePartial`, a trailing
    /// unfinished word counts too. Nil when there's nothing to take.
    static func nextChunk(of text: String, maximumWords: Int, includePartial: Bool)
        -> (end: String.Index, words: Int, endsLine: Bool)? {
        var words = 0
        var inWord = false
        var end = text.startIndex
        var index = text.startIndex
        while index < text.endIndex, words < maximumWords {
            let character = text[index]
            index = text.index(after: index)
            if character.isWhitespace {
                if inWord { words += 1 }
                inWord = false
                end = index
                if character.isNewline { return (end, words, true) }
            } else {
                inWord = true
            }
        }
        if inWord, includePartial, index == text.endIndex {
            return (text.endIndex, words + 1, false)
        }
        return end > text.startIndex ? (end, words, false) : nil
    }
}
