import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @Environment(DecoderViewModel.self) private var model

    var body: some View {
        @Bindable var model = model

        VStack(spacing: 0) {
            SignalStatusBar()
            Divider()
            DecodedTextView(text: model.decodedText)
            Divider()
            TuningControls()
        }
        .frame(minWidth: 620, minHeight: 360)
        .toolbar {
            ToolbarItem {
                InputDevicePicker()
            }
            ToolbarItemGroup {
                Button {
                    model.toggleListening()
                } label: {
                    Label(model.isListening ? "Stop" : "Listen",
                          systemImage: model.isListening ? "stop.fill" : "mic.fill")
                }
                .help(model.isListening ? "Stop decoding the audio input" : "Decode the default audio input")
                .disabled(model.isDecodingFile)

                Button {
                    model.isShowingFileImporter = true
                } label: {
                    Label("Open Audio File", systemImage: "waveform")
                }
                .help("Decode a recording")
                .disabled(model.isDecodingFile)

                Button {
                    model.copyText()
                } label: {
                    Label("Copy", systemImage: "doc.on.doc")
                }
                .help("Copy the decoded text")
                .disabled(model.decodedText.isEmpty)

                Button {
                    model.clearText()
                } label: {
                    Label("Clear", systemImage: "trash")
                }
                .help("Clear the decoded text")
                .disabled(model.decodedText.isEmpty)
            }
        }
        .fileImporter(isPresented: $model.isShowingFileImporter, allowedContentTypes: [.audio]) { result in
            switch result {
            case .success(let url):
                model.decodeFile(at: url)
            case .failure(let error):
                model.errorMessage = error.localizedDescription
            }
        }
        .alert("CWDecode", isPresented: Binding(
            get: { model.errorMessage != nil },
            set: { if !$0 { model.errorMessage = nil } }
        )) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(model.errorMessage ?? "")
        }
    }
}

private struct InputDevicePicker: View {
    @Environment(DecoderViewModel.self) private var model

    var body: some View {
        @Bindable var model = model

        Picker("Input", selection: $model.selectedInputUID) {
            Label(defaultTitle, systemImage: "mic")
                .tag(String?.none)
            if !model.inputDevices.isEmpty {
                Divider()
            }
            ForEach(model.inputDevices) { device in
                Text(device.name).tag(Optional(device.uid))
            }
            if let name = model.unavailableInputName {
                Divider()
                Text("\(name) (not connected)").tag(model.selectedInputUID)
            }
        }
        .pickerStyle(.menu)
        .frame(maxWidth: 260)
        .help("Audio input to decode")
    }

    private var defaultTitle: String {
        if let name = model.defaultInputDevice?.name {
            "System Default (\(name))"
        } else {
            "System Default"
        }
    }
}

private struct DecodedTextView: View {
    let text: String

    var body: some View {
        ScrollView {
            Group {
                if text.isEmpty {
                    Text("Click Listen to decode your audio input, or open a recording.")
                        .foregroundStyle(.secondary)
                } else {
                    Text(text)
                        .textSelection(.enabled)
                }
            }
            .font(.system(size: 20, design: .monospaced))
            .frame(maxWidth: .infinity, alignment: .topLeading)
            .padding()
        }
        .defaultScrollAnchor(.bottom, for: .sizeChanges)
        .background(Color(nsColor: .textBackgroundColor))
    }
}

private struct SignalStatusBar: View {
    @Environment(DecoderViewModel.self) private var model

    var body: some View {
        HStack(spacing: 14) {
            Circle()
                .fill(model.keyDown ? Color.green : Color.secondary.opacity(0.25))
                .frame(width: 12, height: 12)
                .help("Key down")

            LevelMeter(level: model.isListening || model.isDecodingFile ? model.signalLevel : 0)
                .frame(width: 120, height: 8)
                .help("Tone level between the noise floor and the signal peak")

            Text(pendingDisplay)
                .font(.system(.title3, design: .monospaced))
                .frame(minWidth: 110, alignment: .leading)
                .help("Elements of the character being received")

            Spacer()

            if model.isDecodingFile {
                ProgressView().controlSize(.small)
                Text("Decoding file…").foregroundStyle(.secondary)
            }

            Group {
                Text("\(Int(model.detectedFrequency.rounded())) Hz")
                Text("SNR \(Int(model.snrDB.rounded())) dB")
                Text(model.wpm > 0 ? "\(Int(model.wpm.rounded())) WPM" : "– WPM")
            }
            .monospacedDigit()
            .foregroundStyle(.secondary)
        }
        .padding(.horizontal)
        .padding(.vertical, 10)
    }

    private var pendingDisplay: String {
        String(model.pendingSymbols.map { $0 == "." ? "·" : "–" })
    }
}

private struct LevelMeter: View {
    let level: Double

    var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .leading) {
                Capsule().fill(Color.secondary.opacity(0.2))
                Capsule()
                    .fill(Color.accentColor)
                    .frame(width: geometry.size.width * min(max(level, 0), 1))
            }
        }
    }
}

private struct TuningControls: View {
    @Environment(DecoderViewModel.self) private var model

    var body: some View {
        @Bindable var model = model

        HStack(spacing: 24) {
            Toggle("Auto-tune", isOn: $model.autoTune)
                .help("Follow the strongest tone between \(Int(FrequencyTracker.searchRange.lowerBound)) and \(Int(FrequencyTracker.searchRange.upperBound)) Hz")

            HStack {
                Text("Tone")
                Slider(value: frequency, in: FrequencyTracker.searchRange, step: 10)
                    .disabled(model.autoTune)
                Text("\(Int(frequency.wrappedValue.rounded())) Hz")
                    .monospacedDigit()
                    .frame(width: 64, alignment: .trailing)
            }

            HStack {
                Text("Squelch")
                Slider(value: $model.squelchDB, in: 3...30, step: 1)
                    .frame(maxWidth: 160)
                Text("\(Int(model.squelchDB)) dB")
                    .monospacedDigit()
                    .frame(width: 44, alignment: .trailing)
            }
            .help("Minimum signal-to-noise ratio needed before anything is decoded")
        }
        .padding(.horizontal)
        .padding(.vertical, 12)
    }

    private var frequency: Binding<Double> {
        Binding(
            get: { model.autoTune ? model.detectedFrequency : model.toneFrequency },
            set: { model.toneFrequency = $0 }
        )
    }
}
