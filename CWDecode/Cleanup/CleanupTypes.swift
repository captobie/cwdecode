import Foundation

/// Whether smart cleanup can run on this Mac right now.
enum CleanupAvailability: Equatable, Sendable {
    case available
    /// Older than macOS 26.
    case unsupportedOS
    case deviceNotEligible
    case appleIntelligenceNotEnabled
    /// Still downloading or otherwise preparing; usually resolves on its own.
    case modelNotReady
    case unavailable

    /// Why cleanup is off, for a tooltip. Nil when available.
    var explanation: String? {
        switch self {
        case .available: nil
        case .unsupportedOS: "Smart cleanup needs macOS 26 or later."
        case .deviceNotEligible: "Smart cleanup needs a Mac that supports Apple Intelligence."
        case .appleIntelligenceNotEnabled: "Turn on Apple Intelligence in System Settings to use smart cleanup."
        case .modelNotReady: "Apple Intelligence is still getting ready. Smart cleanup will start when it's done."
        case .unavailable: "Apple Intelligence isn't available right now."
        }
    }
}

/// One window of decoder output to correct.
struct CleanupRequest: Sendable, Equatable {
    /// Already-cleaned text just before `raw`, for context only.
    var context: String
    /// New decoder output: complete words, without trailing whitespace.
    var raw: String
}

struct CleanupResult: Sendable, Equatable {
    struct Correction: Sendable, Equatable {
        var received: String
        var corrected: String
        var reason: String
    }

    /// Corrected version of the request's `raw` text.
    var text: String
    var corrections: [Correction] = []
    /// Words the model couldn't resolve confidently.
    var uncertain: [String] = []
}

/// Layer 2: turns a window of noisy decoder output into corrected text.
protocol TextCleaner: Sendable {
    func availability() -> CleanupAvailability
    /// Loads the model ahead of the first request. Optional.
    func prewarm()
    func clean(_ request: CleanupRequest) async throws -> CleanupResult
}

/// Used on systems without Foundation Models.
struct UnavailableCleaner: TextCleaner {
    var reason = CleanupAvailability.unsupportedOS

    func availability() -> CleanupAvailability { reason }
    func prewarm() {}
    func clean(_ request: CleanupRequest) async throws -> CleanupResult {
        throw CancellationError()
    }
}

enum TextCleaners {
    /// The on-device model when this macOS has it; otherwise a cleaner that is always unavailable.
    static func makeDefault() -> any TextCleaner {
        if #available(macOS 26, *) {
            FoundationModelsCleaner()
        } else {
            UnavailableCleaner()
        }
    }
}
