import Accelerate

/// Audio → the neural decoder's input, exactly as `ml/cwmodel/features.py` computes it:
/// 8 kHz audio, a 256-point periodic Hann window every 64 samples (8 ms) with 128 zeros of
/// padding at each end, power in bins 8…40 (250–1250 Hz), log10, floored 60 dB below the
/// loudest bin, then standardized over the whole window. Golden tests pin it to the Python code.
enum NeuralFeatures {
    static let sampleRate = 8000.0
    static let fftSize = 256
    static let hop = 64
    static let bins = 8..<41
    static let binCount = 33
    static let dynamicRangeDB: Float = 60

    static func frameCount(samples: Int) -> Int { 1 + samples / hop }

    /// Periodic Hann window, as torch.hann_window(256).
    private static let window: [Float] = (0..<fftSize).map {
        Float(0.5 - 0.5 * cos(2 * Double.pi * Double($0) / Double(fftSize)))
    }

    /// DFT basis for just the bins we keep, [fftSize × binCount] row-major, unnormalized like torch.stft.
    private static let (cosBasis, sinBasis): ([Float], [Float]) = {
        var c = [Float](repeating: 0, count: fftSize * binCount)
        var s = c
        for n in 0..<fftSize {
            for (j, k) in bins.enumerated() {
                let angle = 2 * Double.pi * Double(k * n) / Double(fftSize)
                c[n * binCount + j] = Float(cos(angle))
                s[n * binCount + j] = Float(sin(angle))
            }
        }
        return (c, s)
    }()

    /// [binCount × frames], bin-major (the layout of the model's [1, 1, 33, frames] input).
    static func spectrogram(_ audio: [Float]) -> [Float] {
        let frames = frameCount(samples: audio.count)
        let pad = fftSize / 2
        var padded = [Float](repeating: 0, count: audio.count + 2 * pad)
        padded.replaceSubrange(pad..<(pad + audio.count), with: audio)

        var windowed = [Float](repeating: 0, count: frames * fftSize)
        padded.withUnsafeBufferPointer { src in
            windowed.withUnsafeMutableBufferPointer { dst in
                for t in 0..<frames {
                    vDSP_vmul(src.baseAddress! + t * hop, 1, window, 1,
                              dst.baseAddress! + t * fftSize, 1, vDSP_Length(fftSize))
                }
            }
        }

        // [frames × fftSize] · [fftSize × binCount] → real and imaginary parts per frame and bin.
        var re = [Float](repeating: 0, count: frames * binCount)
        var im = re
        vDSP_mmul(windowed, 1, cosBasis, 1, &re, 1, vDSP_Length(frames), vDSP_Length(binCount), vDSP_Length(fftSize))
        vDSP_mmul(windowed, 1, sinBasis, 1, &im, 1, vDSP_Length(frames), vDSP_Length(binCount), vDSP_Length(fftSize))

        // Power, transposed to bin-major.
        var power = [Float](repeating: 0, count: frames * binCount)
        for t in 0..<frames {
            for b in 0..<binCount {
                let i = t * binCount + b
                power[b * frames + t] = re[i] * re[i] + im[i] * im[i]
            }
        }

        var epsilon: Float = 1e-10
        vDSP_vsadd(power, 1, &epsilon, &power, 1, vDSP_Length(power.count))
        var count = Int32(power.count)
        var logs = [Float](repeating: 0, count: power.count)
        vvlog10f(&logs, power, &count)

        var floor = vDSP.maximum(logs) - dynamicRangeDB / 10
        vDSP_vthr(logs, 1, &floor, &logs, 1, vDSP_Length(logs.count))

        // Standardize with the unbiased standard deviation, as torch.Tensor.std does.
        let mean = vDSP.mean(logs)
        var centered = vDSP.add(-mean, logs)
        let variance = vDSP.sumOfSquares(centered) / Float(max(1, centered.count - 1))
        var scale = 1 / (variance.squareRoot() + 1e-5)
        vDSP_vsmul(centered, 1, &scale, &centered, 1, vDSP_Length(centered.count))
        return centered
    }
}
