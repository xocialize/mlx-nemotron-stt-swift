// nemotron-stt-gate — live engine gate for the Nemotron STT package.
//
// Drives the REAL MLXServeEngine path: register → prepare(.stt) → run(STTRequest) with a
// wav file, printing the transcript, detected language, and timings. This is the wrapper-level
// `--e2e-stt-pkg` gate (exercises license admission, materialization routing, load, and run
// through the engine — not just the core).
//
// Usage: swift run nemotron-stt-gate <path-to.wav> [--language en-US] [--model-dir DIR]
//                                                  [--live] [--cancel-after N]
//
// `--live` runs the contract's **LIV-1..6 gate** (MLXEngine 1.39.0, companion N2) against the
// canonical plane: `MLXServeEngine.transcribeLive` fed the same wav in real-time-shaped buffers.
// LIV-5 (the final transcript equals what `engine.run(STTRequest)` produced) is the one that
// promotes the plane, and the one that catches a live path quietly diverging from the batch one.
// `--cancel-after N` additionally checks the package still serves a normal request afterwards.

import Foundation
import MLXEngineTestKit
import MLXNemotronSTT
import MLXServeConformance
import MLXServeCore
import MLXToolKit
import NemotronASR

@main
struct Main {
    static func main() async {
        var args = Array(CommandLine.arguments.dropFirst())
        func popOption(_ name: String) -> String? {
            guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
            let v = args[i + 1]
            args.removeSubrange(i...(i + 1))
            return v
        }
        func popFlag(_ name: String) -> Bool {
            guard let i = args.firstIndex(of: name) else { return false }
            args.remove(at: i)
            return true
        }
        let language = popOption("--language")
        let modelDir = popOption("--model-dir")
        let live = popFlag("--live")
        let cancelAfter = popOption("--cancel-after").flatMap(Int.init)
        let metrics = popFlag("--metrics")
        let mat = popFlag("--mat")
        guard let wavPath = args.first else {
            FileHandle.standardError.write(Data("usage: nemotron-stt-gate <wav> [--language L] [--model-dir DIR]\n".utf8))
            exit(2)
        }

        do {
            let engine = MLXServeEngine()
            var config = NemotronSTTConfiguration(attContextSize: [56, 3])
            if let modelDir { config.modelDirectory = URL(fileURLWithPath: modelDir) }

            let t0 = Date()
            _ = try await engine.register(NemotronSTTPackage.registration, configuration: config)
            _ = try await engine.prepare(.stt)
            let loadTime = Date().timeIntervalSince(t0)
            print("prepared (.stt) in \(String(format: "%.1f", loadTime))s")

            let wav = try Data(contentsOf: URL(fileURLWithPath: wavPath))
            let request = STTRequest(
                audio: Audio(format: .wav, data: wav), language: language)

            let t1 = Date()
            let response = try await engine.run(request)
            let runTime = Date().timeIntervalSince(t1)

            guard let stt = response as? STTResponse else {
                FileHandle.standardError.write(Data("unexpected response type\n".utf8))
                exit(1)
            }
            print("transcript: \(stt.text)")
            print("segments: \(stt.segments.count)  detected: \(stt.detectedLanguage ?? "?")")
            print("run in \(String(format: "%.2f", runTime))s")

            if live {
                let ok = try await runLiveGate(
                    engine: engine, wav: wav, language: language,
                    oneShot: stt, cancelAfter: cancelAfter)
                guard ok else { exit(1) }
            }
            if metrics {
                try await runMetrics(
                    engine: engine, wav: wav, language: language,
                    audioSeconds: Double(stt.segments.last.map { $0.start + $0.duration } ?? 0),
                    runSeconds: runTime, config: config, includeMaterialization: mat)
            }
        } catch {
            FileHandle.standardError.write(Data("ERROR: \(error)\n".utf8))
            exit(1)
        }
    }

    /// Registry evidence: one-shot throughput, the live surface at every rung of the latency
    /// ladder, and (opt-in) first-run materialization. Grep `[RUN]` / `[LIVE]` / `[MAT]`.
    @MainActor
    static func runMetrics(
        engine: MLXServeEngine, wav: Data, language: String?,
        audioSeconds: Double, runSeconds: Double,
        config: NemotronSTTConfiguration, includeMaterialization: Bool
    ) async throws {
        print("\n--- metrics ---")
        let samples = try AudioDecode.monoSamples(from: Audio(format: .wav, data: wav))
        let seconds = audioSeconds > 0 ? audioSeconds : Double(samples.count) / 16_000.0

        // One-shot: the canonical engine.run path, timed above. Re-run warm so the number is
        // steady-state rather than first-call kernel compilation.
        let warmStart = Date()
        _ = try await engine.run(
            STTRequest(audio: Audio(format: .wav, data: wav), language: language))
        let warm = Date().timeIntervalSince(warmStart)
        let phys = HostMemory.physFootprint() ?? 0
        print(
            String(
                format: "[RUN] pkg=nemotron-stt audio=%.1fs cold=%.2fs warm=%.2fs rtfx=%.1f phys=%.2fGB",
                seconds, runSeconds, warm, seconds / warm, Double(phys) / 1_073_741_824))

        // The latency ladder, fed the way a microphone feeds it (100 ms buffers) but unpaced,
        // so wall time measures compute rather than the length of the clip. rtf < 1 is the bar
        // for keeping up with a live speaker.
        guard let package = try await engine.prepare(.stt) as? NemotronSTTPackage else { return }
        for right in [0, 1, 3, 6, 13] {
            // Deliberately the package-specific surface: the latency ladder is `attContextSize`,
            // which is package CONFIGURATION and has no canonical request field. Naming the type
            // is required since 1.39.0 — `.init(language:)` also matches `STTSessionRequest`.
            let session = try await package.startLiveTranscription(
                NemotronLiveTranscription.Options(language: language,
                                                  attContextSize: [56, right]))
            let sampler = MemorySampler()
            sampler.start(initial: HostMemory.physFootprint() ?? 0)
            let start = Date()

            let collector = Task { () -> (first: Double?, count: Int, gaps: [Double]) in
                var first: Double?
                var count = 0
                var gaps: [Double] = []
                var previous = start
                for try await update in session.updates where !update.isFinal {
                    let now = Date()
                    if first == nil { first = now.timeIntervalSince(start) }
                    gaps.append(now.timeIntervalSince(previous))
                    previous = now
                    count += 1
                }
                return (first, count, gaps)
            }

            var cursor = 0
            while cursor < samples.count {
                let end = min(cursor + 1_600, samples.count)
                session.push(Array(samples[cursor..<end]))
                cursor = end
            }
            session.finish()
            let observed = try await collector.value
            let wall = Date().timeIntervalSince(start)
            sampler.stop()

            let meanGap = observed.gaps.isEmpty
                ? 0 : observed.gaps.reduce(0, +) / Double(observed.gaps.count)
            print(
                String(
                    format:
                        "[LIVE] acs=[56,%d] chunk=%dms ttfp=%.0fms partials=%d cadence=%.0fms "
                        + "wall=%.2fs rtf=%.2f peak=%.2fGB",
                    right, (right + 1) * 80, (observed.first ?? 0) * 1000, observed.count,
                    meanGap * 1000, wall, wall / seconds,
                    Double(sampler.peak) / 1_073_741_824))
        }

        guard includeMaterialization else { return }
        // First-run download, measured the way a fresh machine experiences it: a throwaway
        // engine against an empty store.
        let temp = FileManager.default.temporaryDirectory
            .appendingPathComponent("nemotron-mat-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: temp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: temp) }

        var matConfig = NemotronSTTConfiguration(attContextSize: config.attContextSize)
        matConfig.modelsRootDirectory = temp
        let matEngine = MLXServeEngine()
        await matEngine.useModelStore(ModelStore(root: temp))
        _ = try await matEngine.register(NemotronSTTPackage.registration, configuration: matConfig)
        let run = try await MaterializationBench.run(
            engine: matEngine, capability: .stt, configuration: matConfig,
            sourceRepo: "mlx-community/nemotron-3.5-asr-streaming-0.6b", storeRoot: temp)
        print(run.logLine)
    }

    /// The LIV-1..6 gate (contract 1.39.0) over the canonical live plane.
    ///
    /// Package-level checks (LIV-1's session half, LIV-3, LIV-6) run against sessions the
    /// package hands out directly, because that is what they are about. LIV-2/4/5 run over the
    /// chunks a real `engine.transcribeLive` session delivered.
    static func runLiveGate(
        engine: MLXServeEngine, wav: Data, language: String?,
        oneShot: STTResponse, cancelAfter: Int?
    ) async throws -> Bool {
        print("\n--- live (LIV-1..6) ---")
        guard let package = try await engine.prepare(.stt) as? NemotronSTTPackage else {
            print("  FAIL  prepared package is not a NemotronSTTPackage")
            return false
        }
        let samples = try AudioDecode.monoSamples(from: Audio(format: .wav, data: wav))
        let audioSeconds = Double(samples.count) / 16_000.0
        let bufferSize = 1_600      // 100 ms — the shape a microphone tap delivers
        var ok = true

        func report(_ r: StreamingConformance.Report) {
            for check in r.checks {
                print("  \(check.passed ? "PASS" : "FAIL")  \(check.name) — \(check.note)")
            }
            ok = ok && r.passed
        }

        // LIV-1 — advertisement ⇔ conformance, then session ⇔ declaration.
        report(LiveTranscriptionConformance.checkAdvertisement(NemotronSTTPackage.self))
        do {
            let probe = try await package.startLiveTranscription(STTSessionRequest())
            let drained = Task { for try await _ in probe.updates {} }
            report(LiveTranscriptionConformance.checkSessionMatchesDeclaration(
                NemotronSTTPackage.self, session: probe))
            // LIV-3 — push must not block on the inference actor (the audio-thread rule).
            report(await LiveTranscriptionConformance.checkPushIsNonBlocking(session: probe))
            probe.cancel()
            // `cancel()` returns immediately; the session is not torn down until `updates` ends,
            // and this package admits one session at a time. Draining is what a correct caller
            // does before opening the next one.
            _ = try? await drained.value
        }

        // LIV-2/4/5 — a real session through the engine seam.
        let handle = try await engine.transcribeLive(STTSessionRequest(language: language))
        let collector = Task { () -> ([STTStreamChunk], (any Error)?) in
            var chunks: [STTStreamChunk] = []
            do {
                for try await chunk in handle.updates { chunks.append(chunk) }
                return (chunks, nil)
            } catch {
                return (chunks, error)
            }
        }
        // Feed faster than real time (a file harness, not a microphone) and BACK OFF on
        // `.overrun` instead of dropping. That is the whole point of an explicit outcome: a
        // file-fed caller can wait, a microphone cannot, and only the caller knows which it is.
        // Dropping here instead would cost audio and show up as an LIV-5 failure — which is
        // exactly what it did before this loop backed off.
        var pushed = 0
        var backoffs = 0
        var dropped = 0
        while pushed < samples.count {
            let end = min(pushed + bufferSize, samples.count)
            switch handle.push(Array(samples[pushed..<end]),
                               sampleRate: handle.expectedSampleRate) {
            case .accepted:
                pushed = end
                try await Task.sleep(nanoseconds: 2_000_000)
            case .overrun:
                backoffs += 1
                try await Task.sleep(nanoseconds: 20_000_000)   // let the model catch up, retry
            case .ended:
                dropped += samples.count - pushed
                pushed = samples.count
            default:
                dropped += end - pushed
                pushed = end
            }
        }
        handle.finish()
        let (chunks, streamError) = await collector.value

        if let streamError {
            print("  FAIL  updates ended with \(streamError)")
            return false
        }
        print("  \(dropped == 0 ? "PASS" : "FAIL")  back-pressure — \(backoffs) back-off(s) at "
            + "maxBufferedSeconds=\(handle.maxBufferedSeconds), \(dropped) sample(s) dropped")
        ok = ok && dropped == 0

        report(LiveTranscriptionConformance.checkSequence(chunks))
        report(LiveTranscriptionConformance.checkDiscipline(chunks,
                                                            declared: handle.discipline))
        report(LiveTranscriptionConformance.checkFinalParity(
            chunks: chunks, discipline: handle.discipline, batchTranscript: oneShot.text))

        let processed = chunks.last?.processedSeconds ?? 0
        let sane = abs(processed - audioSeconds) < 1.5
        print("  \(sane ? "PASS" : "FAIL")  processed \(String(format: "%.1f", processed))s of "
            + "\(String(format: "%.1f", audioSeconds))s audio over \(chunks.count) chunk(s)")
        ok = ok && sane

        // LIV-6 — cancel semantics, plus the package must still serve a normal request after.
        // ~5 s of audio, comfortably inside `maxBufferedSeconds`, and enough for a few chunks
        // at every rung of the latency ladder. `cancelAfter` is clamped to what that can
        // actually produce — a check that waits for a chunk the feed cannot deliver hangs.
        let feed = stride(from: 0, to: min(samples.count, bufferSize * 50), by: bufferSize)
            .map { Array(samples[$0..<min($0 + bufferSize, samples.count)]) }
        let cancelProbe = try await package.startLiveTranscription(STTSessionRequest())
        report(await LiveTranscriptionConformance.checkCancelSemantics(
            session: cancelProbe, feed: feed, cancelAfter: min(cancelAfter ?? 0, 2)))

        let after = try await engine.run(
            STTRequest(audio: Audio(format: .wav, data: wav), language: language))
        let recovered = (after as? STTResponse)?.text == oneShot.text
        print("  \(recovered ? "PASS" : "FAIL")  package still runs after a cancelled session")
        ok = ok && recovered

        return ok
    }
}
