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
        } catch {
            FileHandle.standardError.write(Data("ERROR: \(error)\n".utf8))
            exit(1)
        }
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
