import XCTest
@testable import CWSynth

/// Runs the real cwsynth job mode in the repo's ml/.venv; skipped when that isn't set up.
final class SynthJobTests: XCTestCase {
    private var job: SynthJob!
    private var folder: URL!

    override func setUpWithError() throws {
        let ml = URL(filePath: #filePath).deletingLastPathComponent().deletingLastPathComponent().appending(path: "ml")
        job = SynthJob(mlFolder: ml)
        try XCTSkipUnless(FileManager.default.isExecutableFile(atPath: job.python.path), "no ml/.venv")
        folder = FileManager.default.temporaryDirectory.appending(path: "cwsynth-tests-\(UUID().uuidString)")
    }

    override func tearDownWithError() throws {
        if let folder { try? FileManager.default.removeItem(at: folder) }
    }

    private func settings() -> SynthSettings {
        var settings = SynthSettings()
        settings.textSource = .custom
        settings.customText = "CQ TEST\nK1ABC +\n"
        settings.pitch = ValueRange(isRange: false, low: 700, high: 700)
        settings.speed = ValueRange(isRange: false, low: 25, high: 25)
        settings.snr = ValueRange(isRange: false, low: 10, high: 10)
        settings.fists = [.machine]
        settings.count = 4
        return settings
    }

    func testPreviewWritesTheRequestedClip() async throws {
        var request = settings().request(.preview)
        request.index = 1
        request.wav = folder.appending(path: "p.wav").path
        let event = try await job.run(request)
        XCTAssertEqual(event.event, "preview")
        XCTAssertEqual(event.text, "K1ABC <AR>")
        XCTAssertEqual(event.toneHz, 700)
        XCTAssertEqual(event.wpm, 25)
        XCTAssertEqual(event.snrDb, 10)
        XCTAssertTrue(FileManager.default.fileExists(atPath: request.wav!))
    }

    func testDatasetStreamsProgressThenAsksBeforeReplacing() async throws {
        var request = settings().request(.dataset)
        request.out = folder.path
        request.n = 4
        var events: [SynthEvent] = []
        for try await event in job.events(for: request) { events.append(event) }
        XCTAssertEqual(events.last?.event, "done")
        XCTAssertEqual(events.last?.count, 4)
        XCTAssertTrue(events.contains { $0.event == "progress" && $0.done == 4 })
        let manifest = try String(contentsOf: folder.appending(path: "train.jsonl"), encoding: .utf8)
        XCTAssertEqual(manifest.split(separator: "\n").count, 4)

        let again = try await job.run(request)
        XCTAssertEqual(again.event, "exists")
        XCTAssertEqual(again.count, 4)
    }

    func testStoppingEndsThePythonProcess() async throws {
        var request = settings().request(.dataset)
        request.out = folder.path
        request.n = 5000
        for try await event in job.events(for: request) where event.event == "progress" {
            break  // ends the stream, which terminates Python
        }
        try await Task.sleep(for: .seconds(1))
        let manifest = try String(contentsOf: folder.appending(path: "train.jsonl"), encoding: .utf8)
        let written = manifest.split(separator: "\n").count
        try await Task.sleep(for: .seconds(1))
        let later = try String(contentsOf: folder.appending(path: "train.jsonl"), encoding: .utf8)
        XCTAssertLessThan(written, 5000)
        XCTAssertEqual(later.split(separator: "\n").count, written, "nothing is written after stopping")
    }

    func testCheckFlagsUnsendableText() async throws {
        var request = settings().request(.check)
        request.corpus.lines = ["HELLO!", "OK"]
        let event = try await job.run(request)
        XCTAssertEqual(event.lines?.first?.error?.contains("!"), true)
        XCTAssertNil(event.lines?.last?.error)
    }

    func testPythonErrorsBecomeThrownErrors() async throws {
        var request = settings().request(.dataset)
        request.out = folder.path  // no n
        do {
            _ = try await job.run(request)
            XCTFail("expected an error")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("n"), error.localizedDescription)
        }
    }
}
