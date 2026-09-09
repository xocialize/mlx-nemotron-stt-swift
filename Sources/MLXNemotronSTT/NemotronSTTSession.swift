// The canonical live-transcription surface (MLXEngine contract 1.39.0, companion N2).
//
// `NemotronLiveTranscription` (this package's own session, shipped 2026-08) stays exactly as it
// is — ML[X] Audio Studio's Dictate (AB-T-0104) and voice-chat-kit both drive it directly, and
// nothing about their call sites changes. This file is the ADAPTER that puts the same machinery
// behind the contract's `LiveTranscribing` / `STTSession`, so a consumer can swap live-STT
// providers instead of downcasting to a concrete package type.
//
// Two things the adapter adds rather than forwards:
//
//   1. BACK-PRESSURE. `NemotronLiveTranscription.push` yields into an unbounded continuation, so
//      a caller outrunning the model on a slow machine shows up as memory growth and no signal.
//      The adapter bounds it — and it can do so honestly with no change to the shipped session,
//      because the backlog is exactly `pushed − processedSeconds`, and `processedSeconds` rides
//      every update.
//   2. THE COMMITMENT WATERMARK. `committedThrough == processedSeconds`, which is the finding
//      that came out of building this: greedy RNN-T over a cache-aware encoder only ever APPENDS
//      to its hypothesis (`decodePromptedChunk` → `state.hypothesis.append`), and the right
//      context is consumed inside a chunk before any of that chunk's tokens are emitted. Nothing
//      Nemotron has emitted is ever revised. `.cumulative` is a DELIVERY rule ("replace what you
//      hold"), not a claim that anything may change — AB-D-0069's prose had those conflated.

import Foundation
import MLXToolKit
import NemotronASR

/// The contract-facing live session over `NemotronLiveTranscription`.
///
/// `.cumulative`: every chunk carries the whole transcript so far, so a consumer REPLACES what
/// it is holding. 16 kHz mono in — the model's native rate, and the rate `MicCapture`-style taps
/// already convert to.
public final class NemotronSTTSession: STTSession, @unchecked Sendable {

    public nonisolated let updates: AsyncThrowingStream<STTStreamChunk, Error>
    public nonisolated let discipline: STTStreamDiscipline = .cumulative
    public nonisolated let maxBufferedSeconds: Double
    public nonisolated let expectedSampleRate = NemotronSTTSession.nativeSampleRate

    /// The one rate the encoder's mel front-end is built for. A caller converts to it; the
    /// session does not, because `push` is copy-only by contract (LIV-3) and a resampler in the
    /// audio path is exactly what an ASR front-end should not be surprised by.
    public static let nativeSampleRate = 16_000

    /// The default input bound. Must exceed one decode chunk — `processedSeconds` trails the
    /// audio pushed by up to one chunk even in perfect health (1.12 s at the default
    /// `attContextSize [56, 13]`), so a bound near the chunk length would report overrun on a
    /// session that is keeping up fine. 8 s is ~7 chunks of slack and ~512 KB of Float samples.
    public static let defaultMaxBufferedSeconds: Double = 8

    /// The input policy — rate gate and back-pressure accounting — as a value, so it is
    /// exercisable without a model behind it and so the lock protects one small thing.
    struct InputBudget {
        let expectedSampleRate: Int
        let maxBufferedSeconds: Double
        private(set) var pushedSeconds: Double = 0
        private(set) var processedSeconds: Double = 0
        private(set) var ended = false

        /// What becomes of `sampleCount` samples offered at `sampleRate`.
        mutating func admit(sampleCount: Int, sampleRate: Int) -> PushOutcome {
            if ended { return .ended }
            guard sampleRate == expectedSampleRate else { return .unsupportedSampleRate }
            guard sampleCount > 0 else { return .accepted }
            let seconds = Double(sampleCount) / Double(expectedSampleRate)
            // The backlog the model has not reached yet — measured, not estimated:
            // `processedSeconds` rides every update the shipped session emits.
            guard (pushedSeconds - processedSeconds) + seconds <= maxBufferedSeconds else {
                return .overrun      // dropped, loudly — the alternative is unbounded growth
            }
            pushedSeconds += seconds
            return .accepted
        }

        mutating func note(processed: Double) { processedSeconds = processed }
        mutating func end() { ended = true }
    }

    private let live: NemotronLiveTranscription
    private let outputs: AsyncThrowingStream<STTStreamChunk, Error>.Continuation
    private let lock = NSLock()
    private var budget: InputBudget
    private var chunkIndex = 0
    private var pump: Task<Void, Never>?

    init(live: NemotronLiveTranscription,
         maxBufferedSeconds: Double = NemotronSTTSession.defaultMaxBufferedSeconds) {
        self.live = live
        self.maxBufferedSeconds = maxBufferedSeconds
        self.budget = InputBudget(expectedSampleRate: Self.nativeSampleRate,
                                  maxBufferedSeconds: maxBufferedSeconds)
        (updates, outputs) = AsyncThrowingStream<STTStreamChunk, Error>.makeStream()
        pump = Task { [weak self] in await self?.forward(from: live) }
    }

    // ------------------------------------------------------------------ input (audio thread)

    /// Copy-only and non-blocking: safe from an `AVAudioEngine` tap or a CoreAudio IOProc
    /// (LIV-3). Everything MLX happens in the shipped session's driver, on `@InferenceActor`.
    public nonisolated func push(_ samples: [Float], sampleRate: Int) -> PushOutcome {
        let outcome = lock.withLock { budget.admit(sampleCount: samples.count,
                                                   sampleRate: sampleRate) }
        guard outcome == .accepted, !samples.isEmpty else { return outcome }
        live.push(samples)
        return .accepted
    }

    public nonisolated func finish() {
        guard !lock.withLock({ budget.ended }) else { return }
        live.finish()
    }

    public nonisolated func cancel() {
        guard !lock.withLock({ budget.ended }) else { return }
        lock.withLock { budget.end() }
        live.cancel()
    }

    // ----------------------------------------------------------------------------- the pump

    /// Map the shipped session's cumulative `Partial`s onto canonical chunks. The terminal
    /// condition is forwarded unchanged — `CancellationError` from `cancel()`,
    /// `NemotronSTTError.sessionInvalidated` when the engine unloads the model underneath.
    private func forward(from live: NemotronLiveTranscription) async {
        do {
            for try await partial in live.updates {
                outputs.yield(Self.chunk(from: partial, index: note(partial)))
            }
            markEnded()
            outputs.finish()
        } catch {
            markEnded()
            outputs.finish(throwing: error)
        }
    }

    /// Record one update's progress and take its ordinal. Synchronous (and so `NSLock`-safe)
    /// because the pump that calls it is not.
    private nonisolated func note(_ partial: NemotronLiveTranscription.Partial) -> Int {
        lock.withLock {
            budget.note(processed: partial.processedSeconds)
            let index = chunkIndex
            chunkIndex += 1
            if partial.isFinal { budget.end() }
            return index
        }
    }

    private nonisolated func markEnded() {
        lock.withLock { budget.end() }
    }

    static func chunk(from partial: NemotronLiveTranscription.Partial,
                      index: Int) -> STTStreamChunk {
        STTStreamChunk(
            text: partial.result.text,
            segments: partial.result.sentences.map {
                // No `speaker`: this surface does not diarize and does not declare that it does
                // (`STTControls.attributesSpeakers` is false), so a nil here says nothing about
                // the audio — which is exactly what 1.38.0 added the declaration to disambiguate.
                STTSegment(text: $0.text, start: $0.start, duration: $0.duration)
            },
            detectedLanguage: partial.detectedLanguage,
            processedSeconds: partial.processedSeconds,
            // Everything decoded is committed. Greedy RNN-T appends and never retracts.
            committedThrough: partial.processedSeconds,
            index: index,
            isFinal: partial.isFinal)
    }
}

// MARK: - The opt-in

extension NemotronSTTPackage: LiveTranscribing {
    /// Open a canonical live session (contract 1.39.0). Reached through
    /// `MLXServeEngine.transcribeLive(_:package:)`, which holds residency for the session and
    /// admits it like any other work.
    ///
    /// The latency lever (`attContextSize`) stays package CONFIGURATION (C9), not a request
    /// field: it is a property of how this model is being run, not of one utterance. An
    /// unset one inherits `NemotronSTTConfiguration.attContextSize`.
    public func startLiveTranscription(_ request: STTSessionRequest) async throws -> any STTSession {
        // The engine refuses an undeclared control before admission, but a direct package caller
        // does not go through the engine — and silently dropping a canonical field is the
        // violation 1.16.0 named. This surface declares no biasing, so it says so.
        if let context = request.context, !context.isEmpty {
            throw PackageError.unsupportedRequestFeature(
                "context — nemotron-3.5-asr declares no recognition-biasing surface "
                    + "(ToolDescriptor.sttControls)")
        }
        let live = try startLiveTranscription(
            NemotronLiveTranscription.Options(language: request.language))
        return NemotronSTTSession(live: live)
    }
}
