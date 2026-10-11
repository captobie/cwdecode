import SwiftUI

struct ContentView: View {
    @Environment(SynthViewModel.self) private var model

    var body: some View {
        @Bindable var model = model

        VStack(spacing: 0) {
            Form {
                TextSection()
                SignalSection()
                NoiseSection()
                ImpairmentsSection()
                OutputSection()
                PreviewSection()
            }
            .formStyle(.grouped)
            .disabled(model.isGenerating)

            Divider()
            ActionBar()
        }
        .frame(minWidth: 560, minHeight: 600)
        .alert("CWSynth", isPresented: Binding(
            get: { model.errorMessage != nil },
            set: { if !$0 { model.errorMessage = nil } }
        )) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(model.errorMessage ?? "")
        }
        .confirmationDialog(
            "The \(model.settings.split.rawValue) split already has \(model.existingSplit?.count ?? 0) samples.",
            isPresented: Binding(
                get: { model.existingSplit != nil },
                set: { if !$0 { model.existingSplit = nil } }
            )
        ) {
            if let existing = model.existingSplit {
                Button("Add \(model.settings.count) More") {
                    model.generate(start: existing.count)
                }
                Button("Replace", role: .destructive) {
                    model.generate(overwrite: true)
                }
                Button("Cancel", role: .cancel) {}
            }
        } message: {
            Text("Adding continues the numbering, so the new samples differ from the existing ones. Replacing rewrites the manifest; audio files from a larger earlier set stay on disk.")
        }
    }
}

private struct TextSection: View {
    @Environment(SynthViewModel.self) private var model

    var body: some View {
        @Bindable var model = model

        Section {
            Picker("Source", selection: $model.settings.textSource) {
                Text("cwsynth mix").tag(SynthSettings.TextSource.mix)
                Text("My text").tag(SynthSettings.TextSource.custom)
            }
            .pickerStyle(.segmented)

            if model.settings.textSource == .custom {
                TextEditor(text: $model.settings.customText)
                    .font(.system(.body, design: .monospaced))
                    .frame(minHeight: 90, maxHeight: 180)
                ForEach(model.lineIssues, id: \.self) { issue in
                    Label(issue, systemImage: "exclamationmark.triangle")
                        .foregroundStyle(.orange)
                        .font(.callout)
                }
            }
        } header: {
            Text("Text")
        } footer: {
            Text(model.settings.textSource == .mix
                 ? "Callsigns, QSOs, abbreviations, English, random groups and 5% noise-only clips, as in training."
                 : "One clip per line, cycling through the lines. Prosigns in angle brackets, like <BT>; + = ( & are sent as AR BT KN AS. Don't use the same lines in a train set and a val or test set.")
        }
    }
}

private struct SignalSection: View {
    @Environment(SynthViewModel.self) private var model

    var body: some View {
        @Bindable var model = model

        Section("Signal") {
            RangeRow(title: "Pitch", unit: "Hz", range: $model.settings.pitch, step: 10)
            RangeRow(title: "Speed", unit: "WPM", range: $model.settings.speed, step: 1)
            ShareRow(title: "Farnsworth spacing", share: $model.settings.farnsworthShare,
                     help: "Characters at the drawn speed with longer gaps, overall 30–85% of it")
            LabeledContent("Fist") {
                HStack {
                    ForEach(FistStyle.allCases) { style in
                        Toggle(style.title, isOn: Binding(
                            get: { model.settings.fists.contains(style) },
                            set: { on in
                                if on { model.settings.fists.insert(style) } else { model.settings.fists.remove(style) }
                            }
                        ))
                    }
                }
            }
            .help("Each sample uses one of the checked fists, chosen at random. Hand-sent fists get a new operator's timing quirks every sample.")
        }
    }
}

private struct NoiseSection: View {
    @Environment(SynthViewModel.self) private var model

    var body: some View {
        @Bindable var model = model

        Section {
            Toggle("Background noise", isOn: $model.settings.noise)
            if model.settings.noise {
                RangeRow(title: "SNR", unit: "dB", range: $model.settings.snr, step: 1)
            }
        } header: {
            Text("Noise")
        } footer: {
            if model.settings.noise {
                Text("Carrier vs. noise in 500 Hz, before any receiver filter. \(equivalents)")
            }
        }
    }

    private var equivalents: String {
        let snr = model.settings.snr
        func describe(_ hz: Double) -> String {
            let low = SNRBandwidth.convert(snr.lowest, toBandwidth: hz)
            let high = SNRBandwidth.convert(snr.highest, toBandwidth: hz)
            return snr.isRange ? String(format: "%+.0f to %+.0f dB", low, high) : String(format: "%+.1f dB", low)
        }
        return "That's \(describe(2500)) in 2.5 kHz, \(describe(170)) in CWDecode's detector."
    }
}

private struct ImpairmentsSection: View {
    @Environment(SynthViewModel.self) private var model

    var body: some View {
        @Bindable var model = model

        Section {
            ShareRow(title: "Fading (QSB)", share: $model.settings.qsbShare,
                     help: "3–25 dB deep, 0.05–1 Hz")
            ShareRow(title: "Static crashes (QRN)", share: $model.settings.qrnShare,
                     help: "0.5–8 crashes per second")
            ShareRow(title: "Interference (QRM)", share: $model.settings.qrmShare,
                     help: "1–2 weaker CW signals, 80–400 Hz away; never in the label")
            ShareRow(title: "Receiver filter", share: $model.settings.filterShare,
                     help: "Mostly CW filters 150–600 Hz wide, sometimes SSB width")
        } header: {
            Text("Impairments")
        } footer: {
            Text("Share of samples with each one. cwsynth draws the details per sample; Preview shows what it picked.")
        }
    }
}

private struct OutputSection: View {
    @Environment(SynthViewModel.self) private var model

    var body: some View {
        @Bindable var model = model

        Section("Output") {
            LabeledContent("Folder") {
                HStack {
                    Text(model.outputFolder?.path(percentEncoded: false) ?? "None")
                        .lineLimit(1)
                        .truncationMode(.head)
                        .foregroundStyle(model.outputFolder == nil ? .secondary : .primary)
                    Button("Choose…") { model.chooseOutputFolder() }
                }
            }
            Picker("Split", selection: $model.settings.split) {
                ForEach(DatasetSplit.allCases) { split in
                    Text(split.rawValue).tag(split)
                }
            }
            .pickerStyle(.segmented)
            .help("train sets can be mixed into training with --extra-train; val and test are for scoring")
            TextField("Samples", value: $model.settings.count, format: .number)
            TextField("Seed", value: $model.settings.seed, format: .number.grouping(.never))
                .help("The same seed and settings make the same files")
            TextField("Max clip length (s)", value: $model.settings.maxClipSeconds, format: .number)
                .help("Text is cut at a word so the keyed part fits")
            LabeledContent("ml folder") {
                HStack {
                    Text(model.mlFolder?.path(percentEncoded: false) ?? "Not found")
                        .lineLimit(1)
                        .truncationMode(.head)
                        .foregroundStyle(model.mlFolder == nil ? .red : .secondary)
                    Button("Choose…") { model.chooseMLFolder() }
                }
            }
        }
    }
}

private struct PreviewSection: View {
    @Environment(SynthViewModel.self) private var model

    var body: some View {
        if let preview = model.preview {
            Section("Preview: sample \(model.previewIndex)") {
                LabeledContent("Label") {
                    Text(preview.text?.isEmpty == false ? preview.text! : "(noise only)")
                        .font(.system(.body, design: .monospaced))
                        .textSelection(.enabled)
                        .multilineTextAlignment(.trailing)
                }
                if preview.trimmed == true {
                    Label("Cut to fit the max clip length", systemImage: "scissors")
                        .foregroundStyle(.orange)
                }
                LabeledContent("Conditions", value: conditions(preview))
                if !impairments(preview).isEmpty {
                    LabeledContent("Impairments", value: impairments(preview))
                }
            }
        }
    }

    private func conditions(_ p: SynthEvent) -> String {
        var parts = [String(format: "%.1f s", p.durationS ?? 0), String(format: "%.0f Hz", p.toneHz ?? 0)]
        var speed = String(format: "%.1f WPM", p.wpm ?? 0)
        if let f = p.farnsworthWpm { speed += String(format: " (Farnsworth %.1f)", f) }
        parts.append(speed)
        parts.append(p.fist ?? "")
        parts.append(p.snrDb.map { String(format: "%+.1f dB SNR", $0) } ?? "no noise")
        return parts.joined(separator: " · ")
    }

    private func impairments(_ p: SynthEvent) -> String {
        var parts: [String] = []
        if let qsb = p.qsb { parts.append("QSB \(qsb)") }
        if let qrn = p.qrn { parts.append("QRN \(qrn)") }
        if let qrm = p.qrm, qrm > 0 { parts.append("QRM ×\(qrm)") }
        if let filter = p.rxFilter { parts.append("filter \(filter)") }
        return parts.joined(separator: " · ")
    }
}

private struct ActionBar: View {
    @Environment(SynthViewModel.self) private var model

    var body: some View {
        HStack(spacing: 12) {
            Button {
                model.renderPreview(index: 0)
            } label: {
                Label("Preview", systemImage: "waveform")
            }
            .help("Render and play sample 0, exactly as Generate would write it")
            .disabled(model.isPreviewing || model.isGenerating)

            Button {
                model.renderPreview(index: model.previewIndex + 1)
            } label: {
                Label("Next", systemImage: "forward")
            }
            .help("Render and play the next sample")
            .disabled(model.preview == nil || model.isPreviewing || model.isGenerating)

            Button {
                model.isPlaying ? model.stopPlaying() : model.play()
            } label: {
                Label(model.isPlaying ? "Stop" : "Play", systemImage: model.isPlaying ? "stop.fill" : "play.fill")
            }
            .disabled(model.preview == nil)

            if model.isPreviewing {
                ProgressView().controlSize(.small)
            }

            Spacer()

            if let progress = model.progress {
                ProgressView(value: Double(progress.done), total: Double(max(progress.total, 1)))
                    .frame(width: 140)
                Text("\(progress.done)/\(progress.total)")
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
                Button("Stop") { model.stopGenerating() }
            } else {
                if let result = model.lastResult {
                    Button(result) { model.revealOutput() }
                        .buttonStyle(.link)
                        .lineLimit(1)
                        .help("Show the folder in Finder")
                } else if let problem = model.settings.problem {
                    Text(problem).foregroundStyle(.secondary).lineLimit(1)
                }
                Button("Generate") { model.generate() }
                    .keyboardShortcut(.defaultAction)
                    .disabled(!model.canGenerate)
                    .help(model.outputFolder == nil ? "Choose an output folder first" : "Write the dataset split")
            }
        }
        .padding(12)
    }
}

/// A value that is fixed for every sample, or drawn per sample from a range.
private struct RangeRow: View {
    let title: String
    let unit: String
    @Binding var range: ValueRange
    let step: Double

    var body: some View {
        LabeledContent(title) {
            HStack {
                Picker(title, selection: $range.isRange) {
                    Text("Fixed").tag(false)
                    Text("Range").tag(true)
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .fixedSize()

                number($range.low)
                if range.isRange {
                    Text("to")
                    number($range.high)
                }
                Text(unit).foregroundStyle(.secondary)
            }
        }
    }

    private func number(_ value: Binding<Double>) -> some View {
        HStack(spacing: 2) {
            TextField(title, value: value, format: .number.precision(.fractionLength(0...1)))
                .labelsHidden()
                .multilineTextAlignment(.trailing)
                .frame(width: 60)
            Stepper(title, value: value, step: step)
                .labelsHidden()
        }
    }
}

/// The share of samples (0–100%) that get something.
private struct ShareRow: View {
    let title: String
    @Binding var share: Double
    let help: String

    var body: some View {
        LabeledContent {
            HStack {
                Slider(value: $share, in: 0...1, step: 0.05)
                    .frame(width: 180)
                Text(share == 0 ? "Off" : "\(Int((share * 100).rounded()))%")
                    .monospacedDigit()
                    .frame(width: 36, alignment: .trailing)
            }
        } label: {
            Text(title)
            Text(help)
        }
    }
}
