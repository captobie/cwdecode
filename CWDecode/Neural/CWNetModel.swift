import CoreML
import Foundation

enum CWNetModelError: LocalizedError {
    case missing
    case mismatch(String)

    var errorDescription: String? {
        switch self {
        case .missing:
            "The neural decoder's model isn't in the app bundle."
        case .mismatch(let detail):
            "The neural decoder's model doesn't match this version of the app (\(detail))."
        }
    }
}

/// Per-frame token log-probabilities for one window: `frames` rows of `classes` values.
struct LogProbabilities: Sendable {
    var frames: Int
    var classes: Int
    var values: [Float]

    func row(_ frame: Int) -> ArraySlice<Float> {
        values[(frame * classes)..<((frame + 1) * classes)]
    }
}

/// The exported network (`ml/cwmodel/export.py`), run on the CPU so results are deterministic
/// and match the Python reference; a window takes milliseconds.
final class CWNetModel: @unchecked Sendable {
    let vocabulary: [String]
    private let model: MLModel

    /// The window size the model accepts, in spectrogram frames.
    static let minimumFrames = 64

    init(bundle: Bundle = Bundle(for: CWNetModel.self)) throws {
        guard let url = bundle.url(forResource: "CWNet", withExtension: "mlmodelc") else {
            throw CWNetModelError.missing
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuOnly
        model = try MLModel(contentsOf: url, configuration: configuration)

        let metadata = model.modelDescription.metadata[.creatorDefinedKey] as? [String: String] ?? [:]
        guard let vocabularyJSON = metadata["vocabulary"]?.data(using: .utf8),
              let vocabulary = try? JSONDecoder().decode([String].self, from: vocabularyJSON),
              vocabulary.first == "<blank>" else {
            throw CWNetModelError.mismatch("no vocabulary")
        }
        self.vocabulary = vocabulary
        try Self.checkFeatures(metadata["features"])
    }

    private static func checkFeatures(_ json: String?) throws {
        struct Features: Decodable {
            var sample_rate: Double, n_fft: Int, hop: Int, bin_lo: Int, bin_hi: Int
            var dynamic_range_db: Float, time_stride: Int
        }
        guard let data = json?.data(using: .utf8), let f = try? JSONDecoder().decode(Features.self, from: data) else {
            throw CWNetModelError.mismatch("no feature settings")
        }
        let matches = f.sample_rate == NeuralFeatures.sampleRate && f.n_fft == NeuralFeatures.fftSize
            && f.hop == NeuralFeatures.hop && f.bin_lo == NeuralFeatures.bins.lowerBound
            && f.bin_hi == NeuralFeatures.bins.upperBound && f.dynamic_range_db == NeuralFeatures.dynamicRangeDB
            && f.time_stride == StreamingCTCDecoder.timeStride
        guard matches else {
            throw CWNetModelError.mismatch("feature settings differ")
        }
    }

    /// `spectrogram` is bin-major [binCount × frames]; frames must be at least `minimumFrames`.
    func logProbabilities(spectrogram: [Float], frames: Int) throws -> LogProbabilities {
        let input = try MLMultiArray(shape: [1, 1, NSNumber(value: NeuralFeatures.binCount), NSNumber(value: frames)],
                                     dataType: .float32)
        input.withUnsafeMutableBufferPointer(ofType: Float.self) { buffer, strides in
            // Strides may pad rows; copy one bin (row) at a time.
            for b in 0..<NeuralFeatures.binCount {
                let row = b * strides[2]
                for t in 0..<frames {
                    buffer[row + t * strides[3]] = spectrogram[b * frames + t]
                }
            }
        }
        let features = try MLDictionaryFeatureProvider(dictionary: ["spectrogram": MLFeatureValue(multiArray: input)])
        let result = try model.prediction(from: features)
        guard let output = result.featureValue(for: "log_probs")?.multiArrayValue else {
            throw CWNetModelError.mismatch("no log_probs output")
        }
        let outFrames = output.shape[1].intValue
        let classes = output.shape[2].intValue
        var values = [Float](repeating: 0, count: outFrames * classes)
        output.withUnsafeBufferPointer(ofType: Float.self) { buffer in
            let strides = output.strides.map(\.intValue)
            for t in 0..<outFrames {
                for c in 0..<classes {
                    values[t * classes + c] = buffer[t * strides[1] + c * strides[2]]
                }
            }
        }
        return LogProbabilities(frames: outFrames, classes: classes, values: values)
    }
}
