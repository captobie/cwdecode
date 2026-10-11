import Foundation
import Network

enum HubAudioError: LocalizedError {
    case invalidHost(String)
    case unreachable(host: String, reason: String)
    case noAudio(host: String)
    case disconnected(host: String)

    var errorDescription: String? {
        switch self {
        case .invalidHost(let host):
            "“\(host)” isn't a valid host name for the FTX-1 Remote hub."
        case .unreachable(let host, let reason):
            "Couldn't reach the FTX-1 Remote hub at \(host):\(HubAudioFormat.port). Check that FTX-1 Remote is running on that Mac. (\(reason))"
        case .noAudio(let host):
            "No audio arrived from the FTX-1 Remote hub at \(host) for \(Int(HubAudioFormat.audioTimeout)) seconds, so listening stopped. Check that FTX-1 Remote's audio input is running."
        case .disconnected(let host):
            "The FTX-1 Remote hub at \(host) closed the connection."
        }
    }
}

/// The audio relay FTX-1 Remote's Mac app sends its iPad clients (`AudioStreamFormat` and
/// `RigWebSocketServer` in ftx1-remote): binary WebSocket messages of one tag byte followed by
/// mono Int16 little-endian PCM at a fixed 8 kHz.
enum HubAudioFormat {
    static let port: UInt16 = 8765
    static let sampleRate = 8000.0
    /// Main receiver; 0x01 is Sub, 0x02 transmit audio.
    static let mainAudioTag: UInt8 = 0x00
    static let audioTimeout: TimeInterval = 10

    /// The samples of a Main-channel audio message, or nil for anything else.
    static func mainAudioSamples(in message: Data) -> [Float]? {
        guard message.first == mainAudioTag else { return nil }
        let payload = message.dropFirst()
        let count = payload.count / MemoryLayout<Int16>.size
        guard count > 0 else { return nil }
        return payload.withUnsafeBytes { raw in
            (0..<count).map { index in
                let value = Int16(littleEndian: raw.loadUnaligned(fromByteOffset: index * 2, as: Int16.self))
                return Float(value) / 32768
            }
        }
    }
}

/// Streams the Main receive audio from an FTX-1 Remote hub, for testing CWDecode against it.
/// Read-only: it never sends anything, since the hub treats incoming JSON as rig commands and
/// tagged binary messages as transmit audio.
@MainActor
final class HubAudioService {
    /// Called when the connection fails or audio stops arriving. Start again to continue.
    var onFailure: (@MainActor (any Error) -> Void)?

    private var session: HubSession?

    var isRunning: Bool { session != nil }

    func start(host: String, handler: @escaping AudioInputService.SampleHandler) throws {
        stop()
        let host = host.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !host.isEmpty, !host.contains(where: \.isWhitespace),
              let url = URL(string: "ws://\(host):\(HubAudioFormat.port)"), url.host() != nil else {
            throw HubAudioError.invalidHost(host)
        }

        let session = HubSession(url: url, host: host, handler: handler) { [weak self] error, failed in
            Task { @MainActor in
                // A late report from a session already stopped or replaced is stale.
                guard let self, self.session === failed else { return }
                self.stop()
                self.onFailure?(error)
            }
        }
        session.start()
        self.session = session
    }

    func stop() {
        session?.cancel()
        session = nil
    }
}

/// One connection. Everything below runs on its own serial queue.
private final class HubSession: @unchecked Sendable {
    typealias FailureHandler = @Sendable (_ error: HubAudioError, _ session: HubSession) -> Void

    private let queue = DispatchQueue(label: "CWDecode.HubAudio")
    private let connection: NWConnection
    private let host: String
    private let handler: AudioInputService.SampleHandler
    private let onFailure: FailureHandler

    private var isReady = false
    private var isFinished = false
    private var lastAudio = Date()
    private var watchdog: DispatchSourceTimer?

    init(url: URL, host: String, handler: @escaping AudioInputService.SampleHandler, onFailure: @escaping FailureHandler) {
        let webSocket = NWProtocolWebSocket.Options()
        webSocket.autoReplyPing = true
        let parameters = NWParameters.tcp
        parameters.defaultProtocolStack.applicationProtocols.insert(webSocket, at: 0)
        connection = NWConnection(to: .url(url), using: parameters)
        self.host = host
        self.handler = handler
        self.onFailure = onFailure
    }

    func start() {
        connection.stateUpdateHandler = { [weak self] state in self?.stateChanged(state) }
        connection.start(queue: queue)
        queue.async { [self] in
            lastAudio = Date()
            startWatchdog()
            receive()
        }
    }

    func cancel() {
        queue.async { [self] in finish() }
    }

    private func stateChanged(_ state: NWConnection.State) {
        switch state {
        case .ready:
            isReady = true
        case .waiting(let error), .failed(let error):
            // Waiting means it can't connect for now; for a test input, report that rather than wait.
            fail(isReady ? .disconnected(host: host) : .unreachable(host: host, reason: error.localizedDescription))
        default:
            break
        }
    }

    private func receive() {
        connection.receiveMessage { [weak self] content, context, _, error in
            guard let self, !isFinished else { return }
            if let error {
                fail(isReady ? .disconnected(host: host) : .unreachable(host: host, reason: error.localizedDescription))
                return
            }
            let metadata = context?.protocolMetadata(definition: NWProtocolWebSocket.definition) as? NWProtocolWebSocket.Metadata
            if metadata?.opcode == .close || (content == nil && metadata == nil) {
                fail(.disconnected(host: host))
                return
            }
            if metadata?.opcode == .binary, let content, let samples = HubAudioFormat.mainAudioSamples(in: content) {
                lastAudio = Date()
                handler(samples, HubAudioFormat.sampleRate)
            }
            receive()
        }
    }

    /// A dead link (e.g. Tailscale dropping) can leave the connection looking open with nothing
    /// arriving, and a hub whose own audio input is off sends no audio at all.
    private func startWatchdog() {
        let timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(deadline: .now() + 1, repeating: 1)
        timer.setEventHandler { [weak self] in
            guard let self, Date().timeIntervalSince(lastAudio) > HubAudioFormat.audioTimeout else { return }
            fail(.noAudio(host: host))
        }
        timer.resume()
        watchdog = timer
    }

    private func fail(_ error: HubAudioError) {
        guard !isFinished else { return }
        finish()
        onFailure(error, self)
    }

    private func finish() {
        isFinished = true
        watchdog?.cancel()
        watchdog = nil
        connection.stateUpdateHandler = nil
        connection.cancel()
    }
}
