import Foundation
import Synchronization
import Testing
@testable import CWDecode

/// Stands in for the language model.
final class FakeCleaner: TextCleaner {
    private let state = Mutex<(availability: CleanupAvailability, requests: [CleanupRequest])>((.available, []))
    private let respond: @Sendable (CleanupRequest) async throws -> CleanupResult

    init(respond: @escaping @Sendable (CleanupRequest) async throws -> CleanupResult = { CleanupResult(text: $0.raw) }) {
        self.respond = respond
    }

    var requests: [CleanupRequest] { state.withLock { $0.requests } }

    func setAvailability(_ availability: CleanupAvailability) {
        state.withLock { $0.availability = availability }
    }

    func availability() -> CleanupAvailability { state.withLock { $0.availability } }
    func prewarm() {}

    func clean(_ request: CleanupRequest) async throws -> CleanupResult {
        state.withLock { $0.requests.append(request) }
        return try await respond(request)
    }
}

@MainActor
struct CleanupCoordinatorTests {
    private func makeCoordinator(_ cleaner: FakeCleaner, idleDelay: Duration = .seconds(60)) -> CleanupCoordinator {
        var configuration = CleanupCoordinator.Configuration()
        configuration.minimumWords = 4
        configuration.maximumWords = 6
        configuration.contextWords = 5
        configuration.backlogWindows = 1
        configuration.idleDelay = idleDelay
        return CleanupCoordinator(cleaner: cleaner, configuration: configuration,
                                  defaults: UserDefaults(suiteName: UUID().uuidString)!)
    }

    /// Feeds text a character at a time, the way the decoder produces it.
    private func type(_ text: String, into coordinator: CleanupCoordinator) {
        for character in text { coordinator.append(String(character)) }
    }

    private func settle(_ coordinator: CleanupCoordinator) async {
        for _ in 0..<100 where coordinator.isCleaning {
            try? await Task.sleep(for: .milliseconds(5))
        }
    }

    @Test func waitsForEnoughCompleteWords() async {
        let cleaner = FakeCleaner()
        let coordinator = makeCoordinator(cleaner)
        type("CQ CQ DE W1A", into: coordinator)
        await settle(coordinator)
        #expect(cleaner.requests.isEmpty)

        type("W K ", into: coordinator)
        await settle(coordinator)
        #expect(cleaner.requests.map(\.raw) == ["CQ CQ DE W1AW"])
        #expect(coordinator.text == "CQ CQ DE W1AW K ")
    }

    @Test func flushIncludesThePartialWord() async {
        let cleaner = FakeCleaner()
        let coordinator = makeCoordinator(cleaner)
        type("TU 73 S", into: coordinator)
        coordinator.flush()
        await settle(coordinator)
        #expect(cleaner.requests.map(\.raw) == ["TU 73 S"])
        #expect(coordinator.unsegmented.isEmpty)
    }

    @Test func idleSendsWhatIsComplete() async throws {
        let cleaner = FakeCleaner()
        let coordinator = makeCoordinator(cleaner, idleDelay: .milliseconds(20))
        type("QRZ DE K", into: coordinator)
        try await Task.sleep(for: .milliseconds(100))
        await settle(coordinator)
        #expect(cleaner.requests.map(\.raw) == ["QRZ DE"])
        #expect(coordinator.unsegmented == "K")
    }

    @Test func keepsCleanedTextAndSendsItAsContext() async {
        let cleaner = FakeCleaner { request in
            CleanupResult(text: request.raw.replacingOccurrences(of: " E ", with: " "), uncertain: ["W1AW"])
        }
        let coordinator = makeCoordinator(cleaner)
        type("CQ E CQ W1AW ", into: coordinator)
        await settle(coordinator)
        type("DE E K 73 ", into: coordinator)
        await settle(coordinator)

        #expect(coordinator.text == "CQ CQ W1AW DE K 73 ")
        #expect(cleaner.requests.last?.context == "CQ CQ W1AW")
        let pieces = coordinator.segments[0].pieces
        #expect(pieces.first { $0.text.hasPrefix("W1AW") }?.style == .uncertain)
    }

    @Test func implausibleChangesAreUndone() async {
        let cleaner = FakeCleaner { _ in CleanupResult(text: "HELLO FROM BOSTON MASSACHUSETTS") }
        let coordinator = makeCoordinator(cleaner)
        type("CQ CQ DE W1AW ", into: coordinator)
        await settle(coordinator)
        #expect(coordinator.segments.first?.state == .cleaned(CleanupResult(text: "CQ CQ DE W1AW")))
        #expect(coordinator.text == "CQ CQ DE W1AW ")
    }

    @Test func errorsFallBackToRaw() async {
        struct Failure: Error {}
        let cleaner = FakeCleaner { _ in throw Failure() }
        let coordinator = makeCoordinator(cleaner)
        type("CQ CQ DE W1AW ", into: coordinator)
        await settle(coordinator)
        #expect(coordinator.segments.first?.state == .unchanged(.failed))
        #expect(coordinator.text == "CQ CQ DE W1AW ")
    }

    @Test func doesNothingWhenUnavailable() async {
        let cleaner = FakeCleaner()
        cleaner.setAvailability(.appleIntelligenceNotEnabled)
        let coordinator = makeCoordinator(cleaner)
        #expect(!coordinator.isActive)
        type("CQ CQ DE W1AW K ", into: coordinator)
        coordinator.flush()
        await settle(coordinator)
        #expect(cleaner.requests.isEmpty)
        #expect(coordinator.segments.isEmpty)
    }

    @Test func startsWhenTheModelBecomesAvailable() async {
        let cleaner = FakeCleaner()
        cleaner.setAvailability(.modelNotReady)
        let coordinator = makeCoordinator(cleaner)
        coordinator.existingText = { "EARLIER TEXT " }
        type("IGNORED ", into: coordinator)

        cleaner.setAvailability(.available)
        coordinator.refreshAvailability()
        #expect(coordinator.isActive)
        type("CQ CQ DE W1AW ", into: coordinator)
        await settle(coordinator)
        #expect(coordinator.text == "EARLIER TEXT CQ CQ DE W1AW ")
        #expect(cleaner.requests.map(\.raw) == ["CQ CQ DE W1AW"])
    }

    @Test func skipsBacklogWhileBusy() async {
        let cleaner = FakeCleaner { request in
            try await Task.sleep(for: .milliseconds(100))
            return CleanupResult(text: request.raw)
        }
        let coordinator = makeCoordinator(cleaner)
        // One window in flight, then more than one window (the backlog limit) waiting.
        type("A1 A2 A3 A4 B1 B2 B3 B4 B5 B6 C1 C2 C3 C4 C5 C6 D1 ", into: coordinator)
        #expect(coordinator.segments.contains { $0.state == .unchanged(.skipped) })
        #expect(coordinator.text == "A1 A2 A3 A4 B1 B2 B3 B4 B5 B6 C1 C2 C3 C4 C5 C6 D1 ")
    }

    @Test func resetDiscardsAnInFlightAnswer() async throws {
        let cleaner = FakeCleaner { request in
            try? await Task.sleep(for: .milliseconds(50))
            return CleanupResult(text: request.raw)
        }
        let coordinator = makeCoordinator(cleaner)
        type("CQ CQ DE W1AW ", into: coordinator)
        coordinator.reset()
        try await Task.sleep(for: .milliseconds(100))
        #expect(coordinator.segments.isEmpty)
        #expect(coordinator.text.isEmpty)
    }

    @Test func turningOffClearsTheCleanedText() {
        let coordinator = makeCoordinator(FakeCleaner())
        type("CQ CQ ", into: coordinator)
        coordinator.isEnabled = false
        #expect(!coordinator.isActive)
        #expect(coordinator.text.isEmpty)
    }

    @Test func chunksStopAtNewlines() {
        let text = "ONE TWO\nTHREE "
        let chunk = CleanupCoordinator.nextChunk(of: text, maximumWords: 10, includePartial: false)
        #expect(chunk.map { String(text[..<$0.end]) } == "ONE TWO\n")
        #expect(chunk?.endsLine == true)
        #expect(CleanupCoordinator.nextChunk(of: "PART", maximumWords: 10, includePartial: false) == nil)
    }
}

/// Runs the real on-device model; skipped where Apple Intelligence isn't available.
struct FoundationModelsCleanerTests {
    static var modelAvailable: Bool {
        guard #available(macOS 26, *) else { return false }
        return FoundationModelsCleaner().availability() == .available
    }

    @Test func skipsWindowsOfNoise() {
        guard #available(macOS 26, *) else { return }
        #expect(FoundationModelsCleaner.isMostlyNoise("TTTT TT T TTT TE K1ABC"))
        #expect(!FoundationModelsCleaner.isMostlyNoise("CQ CQ E DE W1AW T K"))
        #expect(FoundationModelsCleaner.withoutNoiseRuns("RST 1 TTTTT E TE 599") == "RST 1 E 599")
    }

    @Test(.enabled(if: modelAvailable), .timeLimit(.minutes(1)))
    @MainActor
    func cleansTypicalNoise() async throws {
        guard #available(macOS 26, *) else { return }
        let raw = "CQ CQ E DE W1AW W1AW T K"
        let result = try await FoundationModelsCleaner().clean(CleanupRequest(context: "", raw: raw))
        let outcome = CorrectionFilter.apply(raw: raw, cleaned: result.text, isProtected: CorrectionFilter.isValidWord)
        print("Model cleaned \"\(raw)\" → \"\(result.text)\", kept \"\(outcome.text)\"")
        #expect(outcome.text.contains("DE W1AW W1AW"))
        #expect(outcome.text.split(separator: " ").count <= raw.split(separator: " ").count)
    }
}
