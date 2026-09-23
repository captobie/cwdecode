@preconcurrency import AVFoundation

enum AudioInputError: LocalizedError {
    case permissionDenied
    case noInputDevice
    case deviceUnavailable(name: String)

    var errorDescription: String? {
        switch self {
        case .permissionDenied:
            "CWDecode doesn't have permission to use the microphone. Allow it in System Settings → Privacy & Security → Microphone."
        case .noInputDevice:
            "No audio input device is available. Connect a microphone or your radio's audio interface."
        case .deviceUnavailable(let name):
            "“\(name)” isn't connected. Reconnect it or choose another input."
        }
    }
}

/// Streams mono samples from an audio input device.
@MainActor
final class AudioInputService {
    typealias SampleHandler = @Sendable (_ samples: [Float], _ sampleRate: Double) -> Void

    /// Called when the running device stops delivering audio as configured, e.g. it was
    /// unplugged or its sample rate changed. Restart to continue.
    var onConfigurationChange: (@MainActor () -> Void)?

    private var engine: AVAudioEngine?
    private var startFormat: AVAudioFormat?
    private var configurationObserver: (any NSObjectProtocol)?
    private(set) var deviceID: AudioDeviceID?

    var isRunning: Bool { engine != nil }

    func start(device: AudioInputDevice, handler: @escaping SampleHandler) async throws {
        stop()
        guard await AVAudioApplication.requestRecordPermission() else {
            throw AudioInputError.permissionDenied
        }

        // A fresh engine per start, so nothing is left over from the previous device's format.
        let engine = AVAudioEngine()
        let input = engine.inputNode
        do {
            try input.auAudioUnit.setDeviceID(device.id)
        } catch {
            throw AudioInputError.deviceUnavailable(name: device.name)
        }
        let format = input.inputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0 else {
            throw AudioInputError.deviceUnavailable(name: device.name)
        }

        // A nil format taps the node's own output format, whatever the device negotiated.
        input.installTap(onBus: 0, bufferSize: 2048, format: nil, block: Self.makeTapBlock(handler: handler))
        engine.prepare()
        do {
            try engine.start()
        } catch {
            input.removeTap(onBus: 0)
            throw error
        }

        configurationObserver = NotificationCenter.default.addObserver(
            forName: .AVAudioEngineConfigurationChange,
            object: engine,
            queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated { self?.configurationChanged() }
        }
        self.engine = engine
        startFormat = format
        deviceID = device.id
    }

    func stop() {
        if let configurationObserver {
            NotificationCenter.default.removeObserver(configurationObserver)
        }
        configurationObserver = nil
        guard let engine else { return }
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        self.engine = nil
        startFormat = nil
        deviceID = nil
    }

    private func configurationChanged() {
        guard let engine, let startFormat else { return }
        // Starting on a non-default device makes macOS build an aggregate device, which posts
        // a configuration change while the engine carries on fine. Only a stopped engine or
        // a different format needs a restart; restarting on every notification loops forever.
        let format = engine.inputNode.inputFormat(forBus: 0)
        if engine.isRunning,
           format.sampleRate == startFormat.sampleRate,
           format.channelCount == startFormat.channelCount {
            return
        }
        onConfigurationChange?()
    }

    /// Built outside the main actor: the tap runs on a real-time audio thread.
    private nonisolated static func makeTapBlock(handler: @escaping SampleHandler) -> AVAudioNodeTapBlock {
        { buffer, _ in
            handler(buffer.monoSamples(), buffer.format.sampleRate)
        }
    }
}

extension AVAudioPCMBuffer {
    /// Averages all channels into a single array of float samples.
    func monoSamples() -> [Float] {
        let frames = Int(frameLength)
        guard frames > 0, let channels = floatChannelData else { return [] }
        let channelCount = Int(format.channelCount)
        var mono = Array(UnsafeBufferPointer(start: channels[0], count: frames))
        guard channelCount > 1 else { return mono }
        for channel in 1..<channelCount {
            let data = channels[channel]
            for i in 0..<frames { mono[i] += data[i] }
        }
        let scale = 1 / Float(channelCount)
        for i in 0..<frames { mono[i] *= scale }
        return mono
    }
}
