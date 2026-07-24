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
// `--live` additionally drives the live transcription surface: the same wav is pushed through
// startLiveTranscription in small real-time-shaped buffers, and the final transcript must
// equal what engine.run(STTRequest) produced (LIV-1), after at least one interim partial
// (LIV-2), with monotonic timestamps and a sane processed duration (LIV-3). `--cancel-after N`
// cancels the session after N partials and checks the package still runs afterwards (LIV-4).

import Foundation
import MLXEngineTestKit
import MLXNemotronSTT
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
            let session = try await package.startLiveTranscription(
                .init(language: language, attContextSize: [56, right]))
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

    /// LIV-1..4 — the live surface against the one-shot result it must agree with.
    static func runLiveGate(
        engine: MLXServeEngine, wav: Data, language: String?,
        oneShot: STTResponse, cancelAfter: Int?
    ) async throws -> Bool {
        print("\n--- live ---")
        guard let package = try await engine.prepare(.stt) as? NemotronSTTPackage else {
            print("  FAIL  prepared package is not a NemotronSTTPackage")
            return false
        }
        let samples = try AudioDecode.monoSamples(from: Audio(format: .wav, data: wav))
        let audioSeconds = Double(samples.count) / 16_000.0
        // 100 ms buffers — the shape a microphone tap delivers.
        let bufferSize = 1_600

        var ok = true
        let session = try await package.startLiveTranscription(.init(language: language))

        // Consume updates concurrently: a real consumer reads while it feeds.
        let collector = Task { () -> (partials: [String], final: String?, seconds: Double, error: Error?) in
            var partials: [String] = []
            var final: String?
            var seconds = 0.0
            do {
                for try await update in session.updates {
                    seconds = update.processedSeconds
                    if update.isFinal { final = update.result.text } else {
                        partials.append(update.result.text)
                    }
                }
            } catch {
                return (partials, final, seconds, error)
            }
            return (partials, final, seconds, nil)
        }

        var pushed = 0
        while pushed < samples.count {
            let end = min(pushed + bufferSize, samples.count)
            session.push(Array(samples[pushed..<end]))
            pushed = end
            // Let the driver keep up rather than queueing the whole file at once.
            try await Task.sleep(nanoseconds: 2_000_000)
        }
        session.finish()
        let collected = await collector.value

        if let error = collected.error {
            print("  FAIL  updates ended with \(error)")
            return false
        }
        // LIV-1: the live transcript must equal the one-shot transcript.
        let same = collected.final == oneShot.text
        print("  \(same ? "PASS" : "FAIL")  LIV-1 live transcript == engine.run transcript")
        if !same {
            print("    one-shot: \(oneShot.text)")
            print("    live    : \(collected.final ?? "<none>")")
        }
        ok = ok && same

        // LIV-2: partial hypotheses arrived before the final one.
        let sawPartials = !collected.partials.isEmpty
        print("  \(sawPartials ? "PASS" : "FAIL")  LIV-2 \(collected.partials.count) interim partials")
        ok = ok && sawPartials

        // LIV-3: the session accounted for roughly the audio it was given.
        let sane = abs(collected.seconds - audioSeconds) < 1.5
        print(
            "  \(sane ? "PASS" : "FAIL")  LIV-3 processed \(String(format: "%.1f", collected.seconds))s "
                + "of \(String(format: "%.1f", audioSeconds))s audio")
        ok = ok && sane

        // LIV-4: cancelling mid-stream ends the stream and leaves the package usable.
        if let cancelAfter {
            let cancelled = try await package.startLiveTranscription(.init(language: language))
            let watcher = Task { () -> Error? in
                do {
                    var seen = 0
                    for try await update in cancelled.updates {
                        if !update.isFinal {
                            seen += 1
                            if seen >= cancelAfter { cancelled.cancel() }
                        }
                    }
                    return nil
                } catch {
                    return error
                }
            }
            var sent = 0
            while sent < samples.count {
                let end = min(sent + bufferSize, samples.count)
                cancelled.push(Array(samples[sent..<end]))
                sent = end
                try await Task.sleep(nanoseconds: 2_000_000)
            }
            let cancelError = await watcher.value
            let isCancellation = cancelError is CancellationError
            print("  \(isCancellation ? "PASS" : "FAIL")  LIV-4a cancel ends updates with CancellationError")
            ok = ok && isCancellation

            // The package must still serve a normal request afterwards.
            let after = try await engine.run(
                STTRequest(audio: Audio(format: .wav, data: wav), language: language))
            let recovered = (after as? STTResponse)?.text == oneShot.text
            print("  \(recovered ? "PASS" : "FAIL")  LIV-4b package still runs after a cancelled session")
            ok = ok && recovered
        }

        return ok
    }
}
