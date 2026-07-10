// swift-tools-version: 6.2
import PackageDescription

// mlx-nemotron-stt-swift — the MLXEngine `stt` (speech-to-text) package: NVIDIA Nemotron 3.5
// ASR streaming (cache-aware FastConformer-RNNT, 40 locales) over our NemotronASR core.
// Conforms to the MLXEngine contract (MLXToolKit) and returns the canonical text transcript.
//
// Swift-port naming: `-swift` on the package/repo; module/product stays clean `MLXNemotronSTT`.
// Engine dep is a LOCAL PATH to the feat/stt-capability worktree while the stt capability
// (contract 1.20.0) is unlanded — repoint to the tagged release on Phase-6 promotion.
let package = Package(
    name: "mlx-nemotron-stt-swift",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .library(name: "MLXNemotronSTT", targets: ["MLXNemotronSTT"]),
        .executable(name: "nemotron-stt-gate", targets: ["nemotron-stt-gate"]),
    ],
    dependencies: [
        // Local worktree on feat/stt-capability (contract 1.20.0, stt capability). Repoint to
        // `.package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.28.0")` on promotion.
        .package(path: "../../../MLXEngine/mlx-engine-swift-stt"),
        // The Nemotron ASR core (this repo's sibling WIP package).
        .package(path: "../nemotron-asr-mlx-swift"),
        // HubClient — the native downloader for WeightSourcing auto-materialization.
        .package(url: "https://github.com/huggingface/swift-huggingface.git",
                 .upToNextMajor(from: "0.9.0")),
    ],
    targets: [
        .target(
            name: "MLXNemotronSTT",
            dependencies: [
                .product(name: "MLXToolKit", package: "mlx-engine-swift-stt"),
                .product(name: "NemotronASR", package: "nemotron-asr-mlx-swift"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
            ]
        ),
        .executableTarget(
            name: "nemotron-stt-gate",
            dependencies: [
                "MLXNemotronSTT",
                .product(name: "MLXServeCore", package: "mlx-engine-swift-stt"),
            ]
        ),
        .testTarget(
            name: "MLXNemotronSTTTests",
            dependencies: [
                "MLXNemotronSTT",
                .product(name: "MLXServeCore", package: "mlx-engine-swift-stt"),
                .product(name: "MLXServeConformance", package: "mlx-engine-swift-stt"),
            ]
        ),
    ]
)
