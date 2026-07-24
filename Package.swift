// swift-tools-version: 6.2
import PackageDescription

// mlx-nemotron-stt-swift — the MLXEngine `stt` (speech-to-text) package: NVIDIA Nemotron 3.5
// ASR streaming (cache-aware FastConformer-RNNT, 40 locales) over our NemotronASR core.
// Conforms to the MLXEngine contract (MLXToolKit) and returns the canonical text transcript.
//
// Swift-port naming: `-swift` on the package/repo; module/product stays clean `MLXNemotronSTT`.
// Consumes the published MLXEngine (v0.32.0, engine-executed materialization) + NemotronASR core (v0.2.0).
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
        // MLXEngine contract + runtime. Floor 0.32.0 = contract 1.24.0, where the engine
        // downloads a WeightSourcing config's missing sources itself (this package has no
        // materializer of its own).
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.32.0"),
        // The Nemotron ASR core (v0.2.0 = the live streaming session this package exposes).
        .package(url: "https://github.com/xocialize/nemotron-asr-mlx-swift.git", from: "0.2.0"),
    ],
    targets: [
        .target(
            name: "MLXNemotronSTT",
            dependencies: [
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "NemotronASR", package: "nemotron-asr-mlx-swift"),
            ]
        ),
        .executableTarget(
            name: "nemotron-stt-gate",
            dependencies: [
                "MLXNemotronSTT",
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
                .product(name: "MLXEngineTestKit", package: "mlx-engine-swift"),
            ]
        ),
        .testTarget(
            name: "MLXNemotronSTTTests",
            dependencies: [
                "MLXNemotronSTT",
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
                .product(name: "MLXServeConformance", package: "mlx-engine-swift"),
            ]
        ),
    ]
)
