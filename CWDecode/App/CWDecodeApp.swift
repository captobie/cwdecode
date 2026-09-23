import SwiftUI

@main
struct CWDecodeApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @State private var model = DecoderViewModel()

    var body: some Scene {
        Window("CWDecode", id: "main") {
            ContentView()
                .environment(model)
        }
        .defaultSize(width: 760, height: 520)
        .commands {
            DecoderCommands(model: model)
        }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
}

struct DecoderCommands: Commands {
    let model: DecoderViewModel

    var body: some Commands {
        CommandGroup(replacing: .newItem) {
            Button("Open Audio File…") {
                model.isShowingFileImporter = true
            }
            .keyboardShortcut("o")
            .disabled(model.isDecodingFile)
        }
        CommandMenu("Decoder") {
            Button(model.isListening ? "Stop Listening" : "Start Listening") {
                model.toggleListening()
            }
            .keyboardShortcut("l")
            .disabled(model.isDecodingFile)

            Divider()

            Button("Copy Decoded Text") {
                model.copyText()
            }
            .keyboardShortcut("c", modifiers: [.command, .shift])

            Button("Clear") {
                model.clearText()
            }
            .keyboardShortcut("k")
        }
    }
}
