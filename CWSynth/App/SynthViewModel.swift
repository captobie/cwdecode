import AppKit
import AVFoundation
import Observation

@MainActor
@Observable
final class SynthViewModel {
    var settings: SynthSettings {
        didSet {
            guard settings != oldValue else { return }
            save()
            if settings.customText != oldValue.customText || settings.speed != oldValue.speed
                || settings.farnsworthShare != oldValue.farnsworthShare
                || settings.maxClipSeconds != oldValue.maxClipSeconds || settings.textSource != oldValue.textSource {
                scheduleCheck()
            }
        }
    }

    /// The repo's `ml` folder, where `.venv` and `cwsynth` live.
    var mlFolder: URL? {
        didSet {
            UserDefaults.standard.set(mlFolder?.path, forKey: DefaultsKey.mlFolder)
            scheduleCheck()
        }
    }

    /// Problems with the custom lines: unsendable characters, or too long for the max clip length.
    private(set) var lineIssues: [String] = []
    private(set) var preview: SynthEvent?
    private(set) var previewIndex = 0
    private(set) var isPreviewing = false
    private(set) var isPlaying = false
    private(set) var progress: (done: Int, total: Int)?
    private(set) var lastResult: String?
    var errorMessage: String?
    /// Set when the chosen split already exists; the window asks whether to replace or extend it.
    var existingSplit: (count: Int, manifest: String)?

    var isGenerating: Bool { progress != nil }

    var outputFolder: URL? {
        settings.outputPath.map { URL(filePath: $0, directoryHint: .isDirectory) }
    }

    var canGenerate: Bool {
        settings.problem == nil && outputFolder != nil && mlFolder != nil && !isGenerating
    }

    @ObservationIgnored private var checkTask: Task<Void, Never>?
    @ObservationIgnored private var generateTask: Task<Void, Never>?
    @ObservationIgnored private var player: AVAudioPlayer?
    @ObservationIgnored private var playerDelegate = PlayerDelegate()

    private enum DefaultsKey {
        static let settings = "settings"
        static let mlFolder = "mlFolder"
    }

    init() {
        let defaults = UserDefaults.standard
        settings = defaults.data(forKey: DefaultsKey.settings)
            .flatMap { try? JSONDecoder().decode(SynthSettings.self, from: $0) } ?? SynthSettings()
        mlFolder = defaults.string(forKey: DefaultsKey.mlFolder).map { URL(filePath: $0, directoryHint: .isDirectory) }
            ?? Self.builtInMLFolder
        playerDelegate.onFinish = { [weak self] in self?.isPlaying = false }
        scheduleCheck()
    }

    /// `ml` in the checkout this app was built from (Info.plist `CWSynthMLFolder`), if it's there.
    private static var builtInMLFolder: URL? {
        guard let path = Bundle.main.object(forInfoDictionaryKey: "CWSynthMLFolder") as? String,
              FileManager.default.fileExists(atPath: path) else { return nil }
        return URL(filePath: path, directoryHint: .isDirectory)
    }

    private func save() {
        if let data = try? JSONEncoder().encode(settings) {
            UserDefaults.standard.set(data, forKey: DefaultsKey.settings)
        }
    }

    // MARK: Checking custom text

    private func scheduleCheck() {
        checkTask?.cancel()
        guard settings.textSource == .custom, let mlFolder, !settings.customLines.isEmpty else {
            lineIssues = []
            return
        }
        // Lines are numbered as the user sees them, blank lines included.
        let typed = settings.customText.split(omittingEmptySubsequences: false, whereSeparator: \.isNewline)
        var request = settings.request(.check)
        request.corpus.lines = typed.map(String.init)
        let slowest = settings.speed.lowest
        checkTask = Task {
            try? await Task.sleep(for: .milliseconds(400))
            guard !Task.isCancelled else { return }
            do {
                let result = try await SynthJob(mlFolder: mlFolder).run(request)
                guard !Task.isCancelled else { return }
                lineIssues = (result.lines ?? []).compactMap { line in
                    if let error = line.error {
                        return "Line \(line.line + 1): \(error.replacingOccurrences(of: "not in the CW vocabulary", with: "can't be sent"))"
                    }
                    if line.tooLong == true, let seconds = line.seconds {
                        return "Line \(line.line + 1): about \(Int(seconds.rounded())) s at \(Int(slowest)) WPM, so slower samples are cut at a word to \(Int(settings.maxClipSeconds)) s"
                    }
                    return nil
                }
            } catch {
                guard !Task.isCancelled else { return }
                lineIssues = [error.localizedDescription]
            }
        }
    }

    // MARK: Preview

    /// Renders sample `index` of the split exactly as Generate would, then plays it.
    func renderPreview(index: Int) {
        guard let mlFolder, settings.problem == nil else {
            errorMessage = settings.problem ?? "Choose the ml folder first."
            return
        }
        var request = settings.request(.preview)
        request.index = max(0, index)
        request.wav = FileManager.default.temporaryDirectory.appending(path: "cwsynth-preview.wav").path
        isPreviewing = true
        Task {
            defer { isPreviewing = false }
            do {
                let result = try await SynthJob(mlFolder: mlFolder).run(request)
                preview = result
                previewIndex = result.index ?? 0
                play()
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }

    func play() {
        guard let path = preview?.wav else { return }
        do {
            player?.stop()
            player = try AVAudioPlayer(contentsOf: URL(filePath: path))
            player?.delegate = playerDelegate
            isPlaying = player?.play() ?? false
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func stopPlaying() {
        player?.stop()
        isPlaying = false
    }

    // MARK: Generate

    func chooseOutputFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.prompt = "Choose"
        panel.message = "Choose or create a folder for the dataset. Each split is written as train.jsonl, val.jsonl or test.jsonl with its audio."
        panel.directoryURL = outputFolder ?? mlFolder?.appending(path: "data")
        if panel.runModal() == .OK, let url = panel.url {
            settings.outputPath = url.path
        }
    }

    func chooseMLFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.prompt = "Choose"
        panel.message = "Choose the ml folder of your cwdecode checkout (the one with cwsynth and .venv)."
        if panel.runModal() == .OK, let url = panel.url {
            mlFolder = url
        }
    }

    /// Writes the split; if it already exists, asks first (see `existingSplit`).
    func generate(start: Int = 0, overwrite: Bool = false) {
        guard canGenerate, let mlFolder, let outputFolder else { return }
        var request = settings.request(.dataset)
        request.out = outputFolder.path
        request.n = settings.count
        request.start = start
        request.overwrite = overwrite
        progress = (0, settings.count)
        lastResult = nil
        generateTask = Task {
            defer { progress = nil }
            do {
                for try await event in SynthJob(mlFolder: mlFolder).events(for: request) {
                    switch event.event {
                    case "progress":
                        progress = (event.done ?? 0, event.total ?? settings.count)
                    case "exists":
                        existingSplit = (event.count ?? 0, event.manifest ?? "")
                    case "done":
                        lastResult = "Wrote \(settings.count) samples (\(event.count ?? settings.count) in the \(settings.split.rawValue) split)."
                    case "error":
                        errorMessage = event.message
                    default:
                        break
                    }
                }
                if Task.isCancelled {
                    lastResult = "Stopped. The samples written so far are in \(settings.split.rawValue).jsonl; dataset.json wasn't updated."
                }
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }

    func stopGenerating() {
        generateTask?.cancel()
    }

    func revealOutput() {
        guard let outputFolder else { return }
        NSWorkspace.shared.activateFileViewerSelecting([outputFolder])
    }
}

private final class PlayerDelegate: NSObject, AVAudioPlayerDelegate {
    var onFinish: (@MainActor () -> Void)?

    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        let onFinish = onFinish
        Task { @MainActor in onFinish?() }
    }
}
