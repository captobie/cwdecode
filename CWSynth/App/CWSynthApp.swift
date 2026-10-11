import SwiftUI

/// Makes labeled CW training and evaluation sets with cwsynth (`ml/cwsynth`), the same
/// generator the neural decoder is trained on. A developer tool; it isn't shipped.
@main
struct CWSynthApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @State private var model = SynthViewModel()

    var body: some Scene {
        Window("CWSynth", id: "main") {
            ContentView()
                .environment(model)
        }
        .defaultSize(width: 640, height: 860)
        .commands {
            CommandGroup(replacing: .newItem) {}
            CommandMenu("Dataset") {
                Button("Preview") { model.renderPreview(index: 0) }
                    .keyboardShortcut("p")
                    .disabled(model.isPreviewing)
                Button("Generate") { model.generate() }
                    .keyboardShortcut(.return)
                    .disabled(!model.canGenerate)
                Divider()
                Button("Choose Output Folder…") { model.chooseOutputFolder() }
                    .keyboardShortcut("o")
                Button("Choose ml Folder…") { model.chooseMLFolder() }
            }
        }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
}
