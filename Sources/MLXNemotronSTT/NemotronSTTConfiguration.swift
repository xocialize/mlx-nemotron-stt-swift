import Foundation
import MLXToolKit

/// Init-time configuration for `NemotronSTTPackage` (C9): which Nemotron ASR repo and the
/// default latency mode. Per-request language rides the `STTRequest`, not here.
public struct NemotronSTTConfiguration: PackageConfiguration, ModelStorable, QuantConfigured {
    public var repo: String
    /// Look-ahead `[left, right]` att-context. `[56, 13]` = best offline accuracy (1.12 s
    /// chunks); `[56, 3]` / `[56, 0]` trade accuracy for latency. nil = the model default.
    public var attContextSize: [Int]?
    /// Explicit snapshot directory (dev escape hatch — never touches the network).
    public var modelDirectory: URL?
    /// Engine-chosen models root (auto-materialization target). Set by the engine from its
    /// `ModelStore`. Excluded from `Codable` (environment-specific, not portable config).
    public var modelsRootDirectory: URL?
    /// The published quantization of `repo`. The bf16 repo has no suffix; the `-8bit` repo is int8.
    public var quant: Quant

    public init(
        repo: String = "mlx-community/nemotron-3.5-asr-streaming-0.6b",
        quant: Quant = .bf16,
        attContextSize: [Int]? = nil,
        modelDirectory: URL? = nil,
        modelsRootDirectory: URL? = nil
    ) {
        self.repo = repo
        self.quant = quant
        self.attContextSize = attContextSize
        self.modelDirectory = modelDirectory
        self.modelsRootDirectory = modelsRootDirectory
    }

    private enum CodingKeys: String, CodingKey {
        case repo, quant, attContextSize
    }
}

// MARK: - Weight sources (auto-materialization, engine MAT gate)

extension NemotronSTTConfiguration: WeightSourcing {
    /// Presence probe: config + the safetensors checkpoint (single-file for both quants).
    var requiredFiles: [String] { ["config.json", "model.safetensors"] }

    public var weightSources: [WeightSource] {
        // config + checkpoint + tokenizer; the repo's docs/*.png diagrams are skipped.
        [WeightSource(role: "main", repo: repo,
                      matching: ["config.json", "*.safetensors", "tokenizer*.json"])]
    }

    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        let fm = FileManager.default
        func has(_ dir: URL) -> Bool {
            requiredFiles.allSatisfy { fm.fileExists(atPath: dir.appending(path: $0).path) }
        }
        // Explicit local directory first (dev escape hatch), then the ModelStore layout.
        if let dir = modelDirectory, has(dir) { return [] }
        if let dir = ModelStore(root: storeRoot).directory(for: repo), has(dir) { return [] }
        return weightSources
    }

    /// The configuration with a nil `modelDirectory` resolved to the store layout — what `load()`
    /// uses AFTER materialization. An explicit directory always wins.
    public func resolved(storeRoot: URL?) -> NemotronSTTConfiguration {
        var cfg = self
        if cfg.modelDirectory == nil {
            cfg.modelDirectory = ModelStore(root: storeRoot).directory(for: repo)
        }
        return cfg
    }
}

// MARK: - Cold-start prewarm

extension NemotronSTTConfiguration: WeightPrewarming {
    public var prewarmPaths: [URL] {
        [resolved(storeRoot: modelsRootDirectory).modelDirectory].compactMap { $0 }
    }
}
