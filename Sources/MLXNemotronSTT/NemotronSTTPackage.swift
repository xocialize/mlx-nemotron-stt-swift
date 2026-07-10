import Foundation
import MLX
import MLXToolKit
import NemotronASR

/// Errors specific to the Nemotron STT package boundary.
public enum NemotronSTTError: Error, Equatable {
    /// Weight sources are missing and there is no store root to materialize into.
    case missingWeights(String)
    /// The resolved model directory could not be determined.
    case unresolvedDirectory
}

/// The MLXEngine `stt` (speech-to-text) package: NVIDIA Nemotron 3.5 ASR streaming exposing the
/// canonical `stt` surface.
///
/// Engine-owned lifecycle (C13): the engine constructs from a `NemotronSTTConfiguration`, pages
/// weights in with `load()` (auto-materialized on first run), drives `run(_:)`, and reclaims with
/// `unload()`. Returns the canonical text transcript (`STTResponse`) with timestamped segments and
/// the detected language — playback/UI stays with the consumer.
///
/// The model's internal cache-aware chunked streaming is package-internal memory discipline
/// (per-chunk `RunProgress` reports + cancellation checkpoints); it is NOT a request surface. Live
/// partial hypotheses are the deferred companion-N2 token-streaming contract.
@InferenceActor
public final class NemotronSTTPackage: ModelPackage {
    public typealias Configuration = NemotronSTTConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // Weights: OpenMDW-1.1 (LF, maximally permissive). Port code (nemotron-asr-mlx-swift,
            // donor Blaizzy/mlx-audio) is MIT.
            license: LicenseDeclaration(weightLicense: .openMDW1_1, portCodeLicense: .mit),
            provenance: Provenance(
                sourceRepo: "mlx-community/nemotron-3.5-asr-streaming-0.6b",
                revision: "main", tier: 1),
            requirements: RequirementsManifest(
                // Split footprint (1.14 efficiency contract), measured on this port:
                //   bf16 weights floor ~1.24 GB resident; peak activation ~0.8 GB over the
                //   bf16 GPU smoke (2.0 GB peak − ~1.24 GB weights). int8 halves the weights floor.
                footprints: [
                    QuantFootprint(
                        quant: .bf16, residentBytes: 1_240_000_000,
                        peakActivationBytes: 820_000_000),
                    QuantFootprint(
                        quant: .int8, residentBytes: 640_000_000,
                        peakActivationBytes: 820_000_000),
                ],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: nil
            ),
            specialties: [],
            surfaces: [
                STTContract.descriptor(
                    name: "nemotron-3.5-asr",
                    summary:
                        "NVIDIA Nemotron 3.5 ASR — cache-aware streaming multilingual "
                        + "speech-to-text (40 locales, native punctuation/capitalization).")
            ]
        )
    }

    private let configuration: Configuration
    private var model: NemotronASRModel?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    public func load() async throws {
        guard model == nil else { return }
        // Auto-materialize missing sources into the engine store (dir-less configs only; explicit
        // directories never touch the network), forwarding progress so the engine surfaces
        // `.downloading`.
        let storeRoot = configuration.modelsRootDirectory
        let missing = configuration.missingWeightSources(storeRoot: storeRoot)
        if !missing.isEmpty {
            guard let storeRoot else {
                throw NemotronSTTError.missingWeights(
                    "no models root set and sources missing: "
                        + missing.map(\.role).joined(separator: ", "))
            }
            try await WeightMaterializer.materialize(missing, into: storeRoot)
        }
        try Task.checkCancellation()
        guard let dir = configuration.resolved(storeRoot: storeRoot).modelDirectory else {
            throw NemotronSTTError.unresolvedDirectory
        }
        // Weight load runs on the default stream; the quantized path is auto-detected from the
        // checkpoint (mlx-community `-8bit` repos). No CPU pin — a quantized forward under a CPU
        // pin grinds indefinitely.
        model = try NemotronLoad.load(directory: dir)
    }

    public func unload() async {
        model = nil
        MLX.GPU.clearCache()
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        // CAN-1: the entry checkpoint is the FIRST act of run() — before notLoaded validation.
        try Task.checkCancellation()
        guard let model else { throw PackageError.notLoaded }
        guard request.capability == .stt, let stt = request as? STTRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }

        // Decode → 16 kHz mono; a lang == nil request rides the model's `auto` prompt.
        let samples = try AudioDecode.monoSamples(from: stt.audio)
        let waveform = MLXArray(samples)

        RunProgress.report(.encode)

        // Cache-aware streaming decode. Each decoded chunk is a natural yield point: report
        // progress and honor cancellation. The core rethrows nothing — cancellation is
        // cooperative via `shouldContinue`; we surface it as a CancellationError below.
        var cancelled = false
        let transcription = model.streamGenerate(
            waveform,
            language: stt.language,
            attContextSize: configuration.attContextSize,
            onPartial: { _ in
                RunProgress.report(.decode)
            },
            shouldContinue: {
                if Task.isCancelled {
                    cancelled = true
                    return false
                }
                return true
            }
        )
        if cancelled { throw CancellationError() }
        // Post-decode checkpoint (CAN): bail before building the response.
        try Task.checkCancellation()

        let result = transcription.result
        let segments = result.sentences.map { sentence in
            STTSegment(
                text: sentence.text,
                start: sentence.start,
                duration: sentence.duration)
        }
        // Auto mode reports the detected locale; an explicit language echoes back.
        let detected = transcription.detectedLanguage ?? explicitLanguage(stt.language)
        return STTResponse(text: result.text, segments: segments, detectedLanguage: detected)
    }

    /// An explicit non-`auto` request language echoes back as the detected language.
    private func explicitLanguage(_ language: String?) -> String? {
        guard let language, language != "auto" else { return nil }
        return language
    }
}

extension NemotronSTTPackage {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration {
        .of(NemotronSTTPackage.self)
    }
}
