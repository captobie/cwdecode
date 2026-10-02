// swift-tools-version:6.0
import PackageDescription

// CWKit: the CW decoders (neural and classic) without any app or audio-device code, shared by
// the CWDecode app in this repo and by other apps that bring their own audio.
let package = Package(
    name: "CWKit",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "CWKit", targets: ["CWKit"]),
    ],
    targets: [
        .target(
            name: "CWKit",
            // Precompiled: SwiftPM compiles an .mlpackage's inner model.mlmodel instead of the
            // package and fails, so `ml/cwmodel/export.py` writes the compiled model here.
            resources: [.copy("Resources/CWNet.mlmodelc")]
        ),
        .testTarget(
            name: "CWKitTests",
            dependencies: ["CWKit"],
            resources: [.process("Resources")]
        ),
    ]
)
