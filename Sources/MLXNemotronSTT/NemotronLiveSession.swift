// Live transcription: the package-specific surface for audio that is still arriving.
//
// The engine's `stt` contract is one-shot — a complete utterance in, a transcript out — and
// stays that way. This is the concrete-type escape hatch for a microphone: reach it by
// downcasting the instance `engine.prepare(.stt)` returns. It reuses the model the engine
// already prepared, so there is exactly one residency and the store, progress and eviction
// all still apply.
//
// The threading rule this file exists to enforce: MLX must never run on an audio callback
// thread. `push` only copies samples and hands them to a stream; every MLX operation happens
// in the driver task on `@InferenceActor`, serialized with `run(_:)` like all other engine
// work.

import Foundation
import MLXToolKit
import NemotronASR

/// A live transcription session over a prepared `NemotronSTTPackage`.
///
/// Feed microphone samples with `push`, read cumulative transcripts from `updates`, and end
/// with `finish` (final result) or `cancel` (abandon). Hold on to the session for as long as
/// you are feeding it — dropping it tears the session down.
@InferenceActor
public final class NemotronLiveTranscription {

    /// Per-session knobs. Both default to whatever the package was configured with.
    public struct Options: Sendable {
        /// Prompt key (e.g. `"en-US"`); `nil` rides the model's `auto` prompt.
        public var language: String?
        /// `[left, right]` look-ahead — the latency lever. `[56, 13]` decodes 1.12 s chunks,
        /// `[56, 0]` decodes 80 ms chunks at a higher cost per second of audio.
        public var attContextSize: [Int]?

        public init(language: String? = nil, attContextSize: [Int]? = nil) {
            self.language = language
            self.attContextSize = attContextSize
        }
    }

    /// One cumulative transcript. Every partial supersedes the one before it.
    public struct Partial: Sendable {
        public let result: AlignedResult
        public let detectedLanguage: String?
        /// Seconds of audio transcribed so far.
        public let processedSeconds: Double
        /// True for the last element, produced by `finish`.
        public let isFinal: Bool
    }

    private enum Input: Sendable {
        case samples([Float])
        case finish
    }

    /// Cumulative transcripts as they are decoded. The last element has `isFinal == true`.
    ///
    /// Ends with `CancellationError` if the session is cancelled, or
    /// `NemotronSTTError.sessionInvalidated` if the engine unloads the model underneath it.
    public nonisolated let updates: AsyncThrowingStream<Partial, Error>

    private nonisolated let input: AsyncStream<Input>.Continuation
    private nonisolated let inputs: AsyncStream<Input>
    private nonisolated let cancelled = CancelFlag()
    private nonisolated let updatesContinuation: AsyncThrowingStream<Partial, Error>.Continuation

    private var session: NemotronStreamingSession?
    private var driver: Task<Void, Never>?
    /// True once the session has emitted its last update, by any route.
    private(set) var isDone = false

    init(model: NemotronASRModel, options: Options) {
        let (inputs, input) = AsyncStream<Input>.makeStream()
        self.inputs = inputs
        self.input = input

        let (updates, updatesContinuation) = AsyncThrowingStream<Partial, Error>.makeStream()
        self.updates = updates
        self.updatesContinuation = updatesContinuation

        self.session = NemotronStreamingSession(
            model: model, language: options.language, attContextSize: options.attContextSize)
    }

    /// Start the driver. Separate from `init` so the task never captures a half-built session.
    func start() {
        guard driver == nil else { return }
        driver = Task { @InferenceActor [weak self] in
            await self?.drive()
        }
    }

    // -------------------------------------------------------------------- input side
    // All of these are callable from an audio thread: they copy and enqueue, nothing more.

    /// Feed 16 kHz mono samples. Any buffer size works. Safe to call from an audio callback —
    /// it copies and returns without touching MLX.
    public nonisolated func push(_ samples: [Float]) {
        guard !cancelled.isSet, !samples.isEmpty else { return }
        input.yield(.samples(samples))
    }

    /// Stop feeding audio. The tail is flushed and a final `Partial` is emitted, after which
    /// `updates` ends.
    public nonisolated func finish() {
        guard !cancelled.isSet else { return }
        input.yield(.finish)
        input.finish()
    }

    /// Abandon the session. `updates` ends with `CancellationError`; no final result.
    public nonisolated func cancel() {
        cancelled.set()
        input.finish()
    }

    // ------------------------------------------------------------------- driver side

    private func drive() async {
        loop: for await item in inputs {
            guard !cancelled.isSet, let session else { break loop }
            switch item {
            case .samples(let buffer):
                if let update = session.push(buffer) {
                    updatesContinuation.yield(
                        Partial(
                            result: update.result,
                            detectedLanguage: update.detectedLanguage,
                            processedSeconds: update.processedSeconds,
                            isFinal: false))
                }
            case .finish:
                let final = session.finish()
                updatesContinuation.yield(
                    Partial(
                        result: final.result,
                        detectedLanguage: final.detectedLanguage,
                        processedSeconds: session.processedSeconds,
                        isFinal: true))
                complete(with: nil)
                break loop
            }
        }
        // The loop also ends on cancel, on invalidation, or if the caller drops the session.
        if !isDone {
            complete(with: cancelled.isSet ? CancellationError() : nil)
        }
    }

    /// End `updates` and release the model reference.
    private func complete(with error: Error?) {
        guard !isDone else { return }
        isDone = true
        session?.cancel()
        session = nil  // drop the model ref so an evicting engine actually frees the weights
        if let error {
            updatesContinuation.finish(throwing: error)
        } else {
            updatesContinuation.finish()
        }
    }

    /// Called when the engine unloads the package out from under an open session.
    func invalidate() {
        guard !isDone else { return }
        cancelled.set()
        input.finish()
        driver?.cancel()
        complete(with: NemotronSTTError.sessionInvalidated)
    }
}

/// A cancellation flag readable from the audio thread and the inference actor alike.
private final class CancelFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var value = false

    var isSet: Bool { lock.withLock { value } }
    func set() { lock.withLock { value = true } }
}
