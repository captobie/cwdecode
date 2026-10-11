import XCTest
@testable import CWSynth

final class SynthSettingsTests: XCTestCase {
    func testFixedValuesBecomeEqualBounds() throws {
        var settings = SynthSettings()
        settings.pitch = ValueRange(isRange: false, low: 700, high: 900)
        settings.speed = ValueRange(isRange: true, low: 30, high: 18)
        let conditions = settings.conditions
        XCTAssertEqual(conditions.toneHz, [700, 700])
        XCTAssertEqual(conditions.wpm, [18, 30])
    }

    func testNoiseOffMeansCleanSamples() {
        var settings = SynthSettings()
        XCTAssertEqual(settings.conditions.cleanProb, 0)
        settings.noise = false
        XCTAssertEqual(settings.conditions.cleanProb, 1)
    }

    func testRequestJSONUsesCwsynthNames() throws {
        var settings = SynthSettings()
        settings.textSource = .custom
        settings.customText = "CQ TEST\n\n  73  \n"
        settings.fists = [.straight]
        settings.qsbShare = 0.25
        var request = settings.request(.preview)
        request.index = 2
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: request.encoded()) as? [String: Any])
        XCTAssertEqual(json["action"] as? String, "preview")
        XCTAssertEqual(json["split"] as? String, "train")
        XCTAssertEqual(json["index"] as? Int, 2)
        XCTAssertNil(json["out"], "unset options are left out, not sent as null")
        let corpus = try XCTUnwrap(json["corpus"] as? [String: Any])
        XCTAssertEqual(corpus["lines"] as? [String], ["CQ TEST", "73"])
        let conditions = try XCTUnwrap(json["conditions"] as? [String: Any])
        XCTAssertEqual(conditions["fist_styles"] as? [String: Double], ["straight": 1])
        XCTAssertEqual(conditions["qsb_prob"] as? Double, 0.25)
        XCTAssertEqual(conditions["drift_prob"] as? Double, 0)
        XCTAssertNotNil(conditions["max_signal_s"])
    }

    func testMixSendsNoLines() throws {
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: SynthSettings().request(.check).encoded()) as? [String: Any])
        XCTAssertEqual((json["corpus"] as? [String: Any])?.isEmpty, true)
    }

    func testProblems() {
        var settings = SynthSettings()
        XCTAssertNil(settings.problem)
        settings.fists = []
        XCTAssertNotNil(settings.problem)
        settings = SynthSettings()
        settings.speed = ValueRange(isRange: true, low: 2, high: 20)
        XCTAssertNotNil(settings.problem)
        settings = SynthSettings()
        settings.textSource = .custom
        settings.customText = "  \n"
        XCTAssertNotNil(settings.problem)
    }

    func testSNRBandwidthConversion() {
        XCTAssertEqual(SNRBandwidth.convert(0, toBandwidth: 2500), -6.99, accuracy: 0.01)
        XCTAssertEqual(SNRBandwidth.convert(0, toBandwidth: 170), 4.69, accuracy: 0.01)
    }

    func testDecodesJobEvents() throws {
        let preview = try SynthEvent.decode(#"{"event": "preview", "wav": "/tmp/p.wav", "index": 0, "trimmed": false, "text": "CQ", "source": "custom", "duration_s": 3.2, "wpm": 20.0, "farnsworth_wpm": null, "fist": "paddle", "tone_hz": 600, "snr_db": 3.0, "snr_db_2500hz": -4.0, "snr_db_app": 7.7, "qsb": "19 dB at 0.86 Hz", "qrn": null, "qrm": 0, "rx_filter": "381–795 Hz"}"#)
        XCTAssertEqual(preview.durationS, 3.2)
        XCTAssertEqual(preview.snrDb2500Hz, -4.0)
        XCTAssertEqual(preview.rxFilter, "381–795 Hz")
        XCTAssertNil(preview.farnsworthWpm)

        let check = try SynthEvent.decode(#"{"event": "check", "lines": [{"line": 1, "error": "not in the CW vocabulary: '!'"}, {"line": 3, "text": "THE", "seconds": 40.1, "too_long": true}], "max_signal_s": 15.0}"#)
        XCTAssertEqual(check.lines?.count, 2)
        XCTAssertEqual(check.lines?[1].tooLong, true)
        XCTAssertEqual(check.maxSignalS, 15)
    }
}
