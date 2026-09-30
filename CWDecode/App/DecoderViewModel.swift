import AppKit
import Observation

@MainActor
@Observable
final class DecoderViewModel {
    private(set) var decodedText = ""
    private(set) var pendingSymbols = ""
    /// Neural decoder: the newest text, not yet committed; replaced on every update.
    private(set) var tentativeText = ""
    private(set) var isListening = false
    private(set) var isDecodingFile = false
    private(set) var keyDown = false
    private(set) var signalLevel = 0.0
    private(set) var snrDB = 0.0
    private(set) var wpm = 0.0
    private(set) var detectedFrequency: Double

    var errorMessage: String?
    var isShowingFileImporter = false

    var toneFrequency: Double { didSet { settingsChanged() } }
    var autoTune: Bool {
        didSet {
            // Keep listening on the tone we found rather than jumping back to the old manual setting.
            if !autoTune { toneFrequency = detectedFrequency }
            settingsChanged()
        }
    }
    var squelchDB: Double { didSet { settingsChanged() } }
    var decoder: DecoderKind { didSet { settingsChanged() } }

    /// Why the neural decoder can't be used, or nil when it can.
    let neuralUnavailableReason: String?

    private(set) var inputDevices: [AudioInputDevice] = []
    private(set) var defaultInputDevice: AudioInputDevice?
    /// The chosen input's UID, or nil to follow the system default input.
    var selectedInputUID: String? {
        didSet {
            guard selectedInputUID != oldValue else { return }
            selectedInputName = inputDevices.first { $0.uid == selectedInputUID }?.name
            UserDefaults.standard.set(selectedInputUID, forKey: DefaultsKey.inputDeviceUID)
            UserDefaults.standard.set(selectedInputName, forKey: DefaultsKey.inputDeviceName)
            if isListening { Task { await restartListening() } }
        }
    }
    /// Remembered so a disconnected device can still be shown by name.
    private var selectedInputName: String?

    /// Name of the chosen input when it isn't currently connected.
    var unavailableInputName: String? {
        guard let selectedInputUID, !inputDevices.contains(where: { $0.uid == selectedInputUID }) else { return nil }
        return selectedInputName ?? selectedInputUID
    }

    @ObservationIgnored private let audio = AudioInputService()
    @ObservationIgnored private var deviceMonitor: AudioDeviceMonitor?
    @ObservationIgnored private let runner: PipelineRunner
    @ObservationIgnored private var isStarting = false
    @ObservationIgnored private var recentRestarts: [Date] = []

    private static let maxTextLength = 100_000

    private enum DefaultsKey {
        static let toneFrequency = "toneFrequency"
        static let autoTune = "autoTune"
        static let squelchDB = "squelchDB"
        static let decoder = "decoder"
        static let inputDeviceUID = "inputDeviceUID"
        static let inputDeviceName = "inputDeviceName"
    }

    init() {
        let neuralModel: CWNetModel?
        do {
            neuralModel = try CWNetModel()
            neuralUnavailableReason = nil
        } catch {
            neuralModel = nil
            neuralUnavailableReason = error.localizedDescription
        }

        let defaults = UserDefaults.standard
        defaults.register(defaults: [
            DefaultsKey.toneFrequency: PipelineSettings().toneFrequency,
            DefaultsKey.autoTune: PipelineSettings().autoTune,
            DefaultsKey.squelchDB: PipelineSettings().squelchDB,
            DefaultsKey.decoder: DecoderKind.neural.rawValue,
        ])
        let preferred = DecoderKind(rawValue: defaults.string(forKey: DefaultsKey.decoder) ?? "") ?? .neural
        let initial = PipelineSettings(
            toneFrequency: defaults.double(forKey: DefaultsKey.toneFrequency),
            autoTune: defaults.bool(forKey: DefaultsKey.autoTune),
            squelchDB: defaults.double(forKey: DefaultsKey.squelchDB),
            decoder: neuralModel == nil ? .classic : preferred
        )
        toneFrequency = initial.toneFrequency
        autoTune = initial.autoTune
        squelchDB = initial.squelchDB
        decoder = initial.decoder
        detectedFrequency = initial.toneFrequency
        selectedInputUID = defaults.string(forKey: DefaultsKey.inputDeviceUID)
        selectedInputName = defaults.string(forKey: DefaultsKey.inputDeviceName)

        // The stream preserves the order of pipeline output, which matters for the text.
        let (events, continuation) = AsyncStream.makeStream(of: PipelineRunner.Event.self)
        runner = PipelineRunner(settings: initial, neuralModel: neuralModel) { continuation.yield($0) }

        Task { [weak self] in
            for await event in events {
                self?.handle(event)
            }
        }

        refreshDevices()
        deviceMonitor = AudioDeviceMonitor { [weak self] in self?.refreshDevices() }
        audio.onConfigurationChange = { [weak self] in
            Task { await self?.restartListening() }
        }
    }

    private var settings: PipelineSettings {
        PipelineSettings(toneFrequency: toneFrequency, autoTune: autoTune, squelchDB: squelchDB, decoder: decoder)
    }

    // MARK: - Actions

    func toggleListening() {
        if isListening {
            stopListening()
        } else {
            Task { await startListening() }
        }
    }

    func startListening() async {
        guard !isListening, !isStarting, !isDecodingFile else { return }
        isStarting = true
        defer { isStarting = false }

        refreshDevices()
        let runner = self.runner
        do {
            try await audio.start(device: currentInputDevice()) { samples, sampleRate in
                runner.submit(samples, sampleRate: sampleRate)
            }
            isListening = true
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func stopListening() {
        guard isListening else { return }
        audio.stop()
        runner.finish()
        isListening = false
    }

    /// Picks up a new device, a new default input, or a changed sample rate.
    private func restartListening() async {
        guard isListening, !isStarting else { return }
        // A device that keeps reconfiguring would otherwise flip Listen/Stop endlessly.
        let now = Date()
        recentRestarts = recentRestarts.filter { now.timeIntervalSince($0) < 10 } + [now]
        stopListening()
        guard recentRestarts.count <= 3 else {
            recentRestarts = []
            errorMessage = "The audio input keeps changing its configuration, so listening stopped. Check the device, then click Listen again."
            return
        }
        await startListening()
    }

    func decodeFile(at url: URL) {
        guard !isDecodingFile else { return }
        stopListening()
        isDecodingFile = true
        if !decodedText.isEmpty, !decodedText.hasSuffix("\n") {
            decodedText += "\n"
        }

        let runner = self.runner
        Task.detached(priority: .userInitiated) { [weak self] in
            let didAccess = url.startAccessingSecurityScopedResource()
            defer { if didAccess { url.stopAccessingSecurityScopedResource() } }
            do {
                try AudioFileReader.read(url) { samples, sampleRate in
                    runner.submitAndWait(samples, sampleRate: sampleRate)
                }
            } catch {
                await self?.reportFileError(error, url: url)
            }
            runner.finish()
        }
    }

    func copyText() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(decodedText, forType: .string)
    }

    func clearText() {
        decodedText = ""
        tentativeText = ""
    }

    // MARK: - Pipeline events

    private func handle(_ event: PipelineRunner.Event) {
        switch event {
        case .output(let output):
            apply(output)
        case .finished(let output):
            apply(output)
            keyDown = false
            signalLevel = 0
            pendingSymbols = ""
            tentativeText = ""
            isDecodingFile = false
        }
    }

    private func apply(_ output: PipelineOutput) {
        if !output.text.isEmpty {
            decodedText += output.text
            if decodedText.count > Self.maxTextLength {
                decodedText = String(decodedText.suffix(Self.maxTextLength))
            }
        }
        keyDown = output.keyDown
        signalLevel = output.signalLevel
        snrDB = output.snrDB
        wpm = output.wpm
        pendingSymbols = output.pendingSymbols
        tentativeText = output.tentativeText
        detectedFrequency = output.toneFrequency
    }

    private func reportFileError(_ error: any Error, url: URL) {
        errorMessage = "Couldn't read “\(url.lastPathComponent)”: \(error.localizedDescription)"
    }

    // MARK: - Input devices

    private func currentInputDevice() throws -> AudioInputDevice {
        if let selectedInputUID {
            guard let device = inputDevices.first(where: { $0.uid == selectedInputUID }) else {
                throw AudioInputError.deviceUnavailable(name: selectedInputName ?? selectedInputUID)
            }
            return device
        }
        guard let defaultInputDevice else { throw AudioInputError.noInputDevice }
        return defaultInputDevice
    }

    private func refreshDevices() {
        inputDevices = AudioDevices.inputDevices()
        defaultInputDevice = AudioDevices.defaultInputDevice()
        if let device = inputDevices.first(where: { $0.uid == selectedInputUID }) {
            selectedInputName = device.name
        }
        // Follow along if the device we'd pick now isn't the one we're listening to,
        // e.g. the system default changed or the chosen device was reconnected.
        if isListening, (try? currentInputDevice())?.id != audio.deviceID {
            Task { await restartListening() }
        }
    }

    private func settingsChanged() {
        let defaults = UserDefaults.standard
        defaults.set(toneFrequency, forKey: DefaultsKey.toneFrequency)
        defaults.set(autoTune, forKey: DefaultsKey.autoTune)
        defaults.set(squelchDB, forKey: DefaultsKey.squelchDB)
        defaults.set(decoder.rawValue, forKey: DefaultsKey.decoder)
        if !autoTune { detectedFrequency = toneFrequency }
        runner.update(settings)
    }
}
