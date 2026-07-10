// nemotron-stt-gate — live engine gate for the Nemotron STT package.
//
// Drives the REAL MLXServeEngine path: register → prepare(.stt) → run(STTRequest) with a
// wav file, printing the transcript, detected language, and timings. This is the wrapper-level
// `--e2e-stt-pkg` gate (exercises license admission, materialization routing, load, and run
// through the engine — not just the core).
//
// Usage: swift run nemotron-stt-gate <path-to.wav> [--language en-US] [--model-dir DIR]

import Foundation
import MLXNemotronSTT
import MLXServeCore
import MLXToolKit

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
        let language = popOption("--language")
        let modelDir = popOption("--model-dir")
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
        } catch {
            FileHandle.standardError.write(Data("ERROR: \(error)\n".utf8))
            exit(1)
        }
    }
}
