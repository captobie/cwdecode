import Foundation

/// Decodes a stream in overlapping windows; a line-for-line port of `ml/cwmodel/stream.py`,
/// which golden tests hold it to.
///
/// Every `hop` seconds the last `window` seconds are decoded. Each stretch of audio is committed
/// from the window where it has at least `context` seconds of audio after it, and the handoff
/// between windows is placed in the widest gap between emitted tokens near the nominal
/// boundary, so each character is committed exactly once. A token whose timing depends on
/// context (above all a space) can still land on different sides of the handoff in neighboring
/// windows, so the new window's tokens from just before the handoff are aligned with what was
/// committed there: missed tokens are added, duplicates skipped. Text after the handoff is tentative.
final class StreamingCTCDecoder {
    typealias Model = (_ spectrogram: [Float], _ frames: Int) throws -> LogProbabilities

    static let timeStride = 2
    static let outputFrameDuration = Double(NeuralFeatures.hop) / NeuralFeatures.sampleRate * Double(timeStride)
    static let handoffSearch = 0.5
    static let minimumSamples = (CWNetModel.minimumFrames - 1) * NeuralFeatures.hop
    /// How far back before the handoff a new window's tokens are compared with committed text.
    static let reconcileWindow = 1.5

    let window: Double
    let hop: Double
    let context: Double
    private let vocabulary: [String]
    private let model: Model

    private(set) var committed: [String] = []
    private(set) var tentative: [String] = []
    private var recent: [(token: String, time: Double)] = []   // committed tokens with their times
    private var audio: [Float] = []
    private var origin = 0            // absolute sample index of audio[0]
    private var decodedTo = 0         // absolute sample index where the last window ended
    private var committedUntil = 0.0

    init(vocabulary: [String], window: Double = 6, hop: Double = 1, context: Double = 2.5, model: @escaping Model) {
        self.vocabulary = vocabulary
        self.window = window
        self.hop = hop
        self.context = context
        self.model = model
    }

    var text: String { committed.joined() }
    var tentativeText: String { tentative.joined() }

    /// Feeds 8 kHz audio; returns newly committed text.
    func process(_ samples: [Float]) throws -> String {
        audio += samples
        let end = origin + audio.count
        let hopSamples = Int((hop * NeuralFeatures.sampleRate).rounded())
        var new: [String] = []
        while end - decodedTo >= hopSamples {
            decodedTo += hopSamples
            new += try decode(until: decodedTo, final: false)
        }
        return new.joined()
    }

    /// Commits everything that's left, including the tentative tail.
    func finish() throws -> String {
        let end = origin + audio.count
        let new = end > 0 ? try decode(until: end, final: true) : []
        decodedTo = end
        return new.joined()
    }

    private func decode(until end: Int, final: Bool) throws -> [String] {
        let windowSamples = Int((window * NeuralFeatures.sampleRate).rounded())
        let start = max(origin, end - windowSamples)
        var chunk = Array(audio[(start - origin)..<(end - origin)])
        let startTime = Double(start) / NeuralFeatures.sampleRate
        let endTime = Double(end) / NeuralFeatures.sampleRate
        if chunk.count < Self.minimumSamples {
            chunk += [Float](repeating: 0, count: Self.minimumSamples - chunk.count)
        }
        let frames = NeuralFeatures.frameCount(samples: chunk.count)
        let logProbs = try model(NeuralFeatures.spectrogram(chunk), frames)
        let tokens = emissions(logProbs, startTime: startTime)

        let cut: Double
        if final {
            cut = .infinity
        } else {
            let nominal = endTime - context
            let lo = max(committedUntil, nominal - Self.handoffSearch)
            let hi = max(lo, nominal + Self.handoffSearch)
            cut = Self.handoff(times: tokens.map(\.time), lo: lo, hi: hi)
        }

        let since = committedUntil - Self.reconcileWindow
        let before = tokens.filter { since <= $0.time && $0.time < committedUntil }.map(\.token)
        let after = tokens.filter { committedUntil <= $0.time && $0.time < cut }
        let previous = recent.filter { $0.time >= since }.map(\.token)
        let (missed, skip) = Self.reconcile(previous: previous, before: before, after: after.map(\.token))
        // Missed tokens are stamped at the old handoff; the rest keep this window's times.
        let kept = after.dropFirst(skip)
        let new = append(missed + kept.map(\.token),
                         times: Array(repeating: committedUntil, count: missed.count) + kept.map(\.time))
        tentative = tokens.filter { $0.time >= cut }.map(\.token)
        if !final {
            committedUntil = cut
            // Keep only what the next window can reach.
            let keepFrom = max(origin, end + Int((hop * NeuralFeatures.sampleRate).rounded()) - windowSamples)
            audio.removeFirst(keepFrom - origin)
            origin = keepFrom
        }
        return new
    }

    /// Greedy CTC: best token per frame, runs collapsed, blanks dropped, each stamped with the
    /// time of its first frame.
    private func emissions(_ logProbs: LogProbabilities, startTime: Double) -> [(token: String, time: Double)] {
        var out: [(String, Double)] = []
        var previous = 0
        for frame in 0..<logProbs.frames {
            let row = logProbs.row(frame)
            var best = 0
            var bestValue = -Float.infinity
            for (c, value) in zip(0..., row) where value > bestValue {
                best = c
                bestValue = value
            }
            if best != previous && best != 0 {
                out.append((vocabulary[best], startTime + Double(frame) * Self.outputFrameDuration))
            }
            previous = best
        }
        return out
    }

    /// The point in [lo, hi], on the model's frame grid, farthest from any emission.
    static func handoff(times: [Double], lo: Double, hi: Double) -> Double {
        guard hi > lo else { return lo }
        let count = Int(((hi + 1e-9 - lo) / outputFrameDuration).rounded(.up))
        let candidates = (0..<count).map { lo + Double($0) * outputFrameDuration }
        guard !times.isEmpty else { return candidates[candidates.count / 2] }
        var best = candidates[0]
        var bestDistance = -1.0
        for candidate in candidates {
            let distance = times.map { abs(candidate - $0) }.min()!
            if distance > bestDistance {
                best = candidate
                bestDistance = distance
            }
        }
        return best
    }

    /// The last pair (i, j) with a[i] == b[j] in a longest common subsequence of a and b,
    /// backtracking from the end and preferring to drop from `a` on ties, as the Python does.
    static func lastMatch(_ a: [String], _ b: [String]) -> (Int, Int)? {
        let n = a.count, m = b.count
        var lengths = [[Int]](repeating: [Int](repeating: 0, count: m + 1), count: n + 1)
        for i in 0..<n {
            for j in 0..<m {
                lengths[i + 1][j + 1] = a[i] == b[j] ? lengths[i][j] + 1 : max(lengths[i][j + 1], lengths[i + 1][j])
            }
        }
        var i = n, j = m
        while i > 0 && j > 0 {
            if a[i - 1] == b[j - 1] { return (i - 1, j - 1) }
            if lengths[i - 1][j] >= lengths[i][j - 1] { i -= 1 } else { j -= 1 }
        }
        return nil
    }

    /// Compares tokens committed in the last `reconcileWindow` (`previous`) with the new window's
    /// tokens in that stretch (`before`) and after it (`after`). Returns tokens the earlier window
    /// missed, to commit first, and how many leading tokens of `after` it already committed.
    static func reconcile(previous: [String], before: [String], after: [String]) -> (missed: [String], skip: Int) {
        guard !previous.isEmpty, !before.isEmpty, let (i, j) = lastMatch(previous, before) else { return ([], 0) }
        if i == previous.count - 1 {
            return (Array(before[(j + 1)...]), 0)
        }
        let already = j == before.count - 1 ? Array(previous[(i + 1)...]) : []
        return ([], Array(after.prefix(already.count)) == already ? already.count : 0)
    }

    /// Adds tokens to the committed text without leading or doubled spaces.
    private func append(_ tokens: [String], times: [Double]) -> [String] {
        var added: [String] = []
        for (token, time) in zip(tokens, times) {
            let last = added.last ?? committed.last ?? " "
            if token == " " && last == " " { continue }
            added.append(token)
            recent.append((token, time))
        }
        committed += added
        recent.removeAll { $0.time < committedUntil - 2 * Self.reconcileWindow }
        return added
    }
}
