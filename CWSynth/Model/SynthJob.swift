import Foundation

/// One request to `python -m cwsynth job` (see `ml/cwsynth/job.py`).
struct JobRequest: Encodable, Equatable, Sendable {
    enum Action: String, Encodable, Sendable {
        case check, preview, dataset
    }

    struct Conditions: Encodable, Equatable, Sendable {
        var wpm: [Double]
        var farnsworthProb: Double
        var fistStyles: [String: Double]
        var maxSignalS: Double
        var toneHz: [Double]
        var driftProb: Double
        var chirpProb: Double
        var cleanProb: Double
        var snrDb: [Double]
        var snrSkew: Double
        var qsbProb: Double
        var qrnProb: Double
        var qrmProb: Double
        var rxFilterProb: Double

        enum CodingKeys: String, CodingKey {
            case wpm
            case farnsworthProb = "farnsworth_prob"
            case fistStyles = "fist_styles"
            case maxSignalS = "max_signal_s"
            case toneHz = "tone_hz"
            case driftProb = "drift_prob"
            case chirpProb = "chirp_prob"
            case cleanProb = "clean_prob"
            case snrDb = "snr_db"
            case snrSkew = "snr_skew"
            case qsbProb = "qsb_prob"
            case qrnProb = "qrn_prob"
            case qrmProb = "qrm_prob"
            case rxFilterProb = "rx_filter_prob"
        }
    }

    struct Corpus: Encodable, Equatable, Sendable {
        /// nil uses cwsynth's weighted text mix.
        var lines: [String]?
    }

    var action: Action
    var conditions: Conditions
    var corpus: Corpus
    var split: DatasetSplit
    var seed: Int
    var index: Int?
    var wav: String?
    var out: String?
    var n: Int?
    var start: Int?
    var overwrite: Bool?

    func encoded() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        return try encoder.encode(self)
    }
}

/// One line of the job's output. Which fields are set depends on `event`.
struct SynthEvent: Decodable, Sendable {
    struct LineCheck: Decodable, Sendable {
        var line: Int
        var text: String?
        var error: String?
        var seconds: Double?
        var tooLong: Bool?
    }

    var event: String
    var message: String?
    // progress, done, exists
    var done: Int?
    var total: Int?
    var count: Int?
    var manifest: String?
    // check
    var lines: [LineCheck]?
    var maxSignalS: Double?
    // preview
    var wav: String?
    var index: Int?
    var trimmed: Bool?
    var text: String?
    var source: String?
    var durationS: Double?
    var wpm: Double?
    var farnsworthWpm: Double?
    var fist: String?
    var toneHz: Double?
    var snrDb: Double?
    var snrDb2500Hz: Double?
    var snrDbApp: Double?
    var qsb: String?
    var qrn: String?
    var qrm: Int?
    var rxFilter: String?

    static func decode(_ line: String) throws -> SynthEvent {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(SynthEvent.self, from: Data(line.utf8))
    }
}

enum SynthJobError: LocalizedError {
    case pythonMissing(URL)
    case failed(String)

    var errorDescription: String? {
        switch self {
        case .pythonMissing(let url):
            "No Python environment at \(url.path). Set it up in the ml folder: python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'"
        case .failed(let message):
            message
        }
    }
}

/// Runs cwsynth's job mode in the repo's `ml/.venv` and streams its events. Cancelling the
/// consuming task terminates the Python process.
struct SynthJob: Sendable {
    let mlFolder: URL

    var python: URL { mlFolder.appending(path: ".venv/bin/python") }

    func events(for request: JobRequest) -> AsyncThrowingStream<SynthEvent, Error> {
        let python = python
        let mlFolder = mlFolder
        let process = ProcessBox()
        return AsyncThrowingStream { continuation in
            let task = Task.detached {
                do {
                    guard FileManager.default.isExecutableFile(atPath: python.path) else {
                        throw SynthJobError.pythonMissing(python)
                    }
                    let input = Pipe(), output = Pipe(), errors = Pipe()
                    let p = Process()
                    p.executableURL = python
                    p.arguments = ["-m", "cwsynth", "job"]
                    p.currentDirectoryURL = mlFolder
                    p.standardInput = input
                    p.standardOutput = output
                    p.standardError = errors
                    // Not waitUntilExit(): it waits on this thread's run loop, which a
                    // concurrency thread never runs, so it never returns.
                    let (exit, exited) = AsyncStream.makeStream(of: Int32.self)
                    p.terminationHandler = { process in
                        exited.yield(process.terminationStatus)
                        exited.finish()
                    }
                    try process.start(p)
                    let stderr = Task.detached { errors.fileHandleForReading.readDataToEndOfFile() }
                    input.fileHandleForWriting.write(try request.encoded())
                    try input.fileHandleForWriting.close()

                    var answered = false
                    for try await line in output.fileHandleForReading.bytes.lines {
                        let event = try SynthEvent.decode(line)
                        answered = answered || ["error", "exists"].contains(event.event)
                        continuation.yield(event)
                    }
                    var status: Int32 = 0
                    for await code in exit { status = code }
                    let log = String(decoding: await stderr.value, as: UTF8.self)
                    if status != 0 && !answered && !Task.isCancelled {
                        let tail = log.split(separator: "\n").suffix(8).joined(separator: "\n")
                        throw SynthJobError.failed("cwsynth exited with status \(status).\n\(tail)")
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in
                task.cancel()
                process.terminate()
            }
        }
    }

    /// Runs a request and returns its final event (the only one, except for progress).
    func run(_ request: JobRequest) async throws -> SynthEvent {
        var last: SynthEvent?
        for try await event in events(for: request) {
            last = event
        }
        guard let last else { throw SynthJobError.failed("cwsynth returned nothing") }
        if last.event == "error" { throw SynthJobError.failed(last.message ?? "cwsynth failed") }
        return last
    }
}

/// Lets the stream's termination handler stop a process started on another task.
private final class ProcessBox: @unchecked Sendable {
    private let lock = NSLock()
    private var process: Process?
    private var terminated = false

    func start(_ p: Process) throws {
        try lock.withLock {
            guard !terminated else { throw CancellationError() }
            try p.run()
            process = p
        }
    }

    func terminate() {
        lock.withLock {
            terminated = true
            if let process, process.isRunning { process.terminate() }
        }
    }
}
