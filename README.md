# mlx-nemotron-stt-swift

![platform](https://img.shields.io/badge/platform-macOS%2026%2B%20%C2%B7%20Apple%20Silicon-black)
![swift](https://img.shields.io/badge/Swift-6.2-orange)
![engine](https://img.shields.io/badge/MLXEngine-%E2%89%A5%20v0.32.0-purple)
![capability](https://img.shields.io/badge/capability-stt-lightgrey)
![port code](https://img.shields.io/badge/port%20code-MIT-blue)
![weights](https://img.shields.io/badge/weights-OpenMDW--1.1-green)

The **MLXEngine `stt` (speech-to-text) package**: NVIDIA **Nemotron 3.5 ASR streaming** wrapped as
a conformant [`ModelPackage`](https://github.com/xocialize/mlx-engine-swift), returning the
canonical text transcript. Multilingual (40 language-locales), native punctuation and
capitalization, on-device on Apple Silicon.

- **Model:** [`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) — 600M cache-aware FastConformer-RNNT with language-ID prompt conditioning.
- **Core:** [`xocialize/nemotron-asr-mlx-swift`](https://github.com/xocialize/nemotron-asr-mlx-swift) (`NemotronASR`) — the MLX-Swift port, parity-locked bit-for-bit against the Python `Blaizzy/mlx-audio` reference.
- **Weights:** [`mlx-community/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/mlx-community/nemotron-3.5-asr-streaming-0.6b) (bf16, ~1.2 GB) and `-8bit`, downloaded by the engine on first run (19.4 s cold, measured).
- **Capability:** `stt` (contract 1.20.0, engine ≥ v0.32.0). Canonical output: text.
- **License:** weights **OpenMDW-1.1** (permissive); port code **MIT**.

## Install

Add the package to your `Package.swift`:

```swift
.package(url: "https://github.com/xocialize/mlx-nemotron-stt-swift.git", from: "0.3.0"),
```

and the product to your target:

```swift
.product(name: "MLXNemotronSTT", package: "mlx-nemotron-stt-swift"),
```

Requires [`mlx-engine-swift`](https://github.com/xocialize/mlx-engine-swift) **≥ v0.32.0** —
`stt` landed in v0.29.0, but this package carries no downloader of its own and relies on the
engine-executed materialization added in contract 1.24.0 / v0.32.0. The engine + core (`NemotronASR`) resolve transitively. In an Xcode app,
add the package and link the **MLXNemotronSTT** product to your app target.

## Usage (through MLXEngine)

```swift
import MLXNemotronSTT
import MLXServeCore
import MLXToolKit

let engine = MLXServeEngine()
try await engine.register(NemotronSTTPackage.registration,
                          configuration: NemotronSTTConfiguration())
try await engine.prepare(.stt)

// audio: canonical WAV bytes, any sample rate / channels (resampled internally).
let response = try await engine.run(STTRequest(audio: Audio(format: .wav, data: wav))) as? STTResponse
print(response?.text ?? "")            // transcript
print(response?.detectedLanguage ?? "?")  // e.g. "en-US" (auto-detected when language == nil)
```

`STTResponse` also carries `segments` (timestamped spans). Pass `language:` (a BCP-47 locale like
`"en-US"`) to pin recognition, or leave it `nil` to auto-detect.

### Latency modes

The look-ahead context trades accuracy for latency, via `NemotronSTTConfiguration.attContextSize`:

| `attContextSize` | Chunk | Use |
|---|---|---|
| `[56, 13]` (default) | 1.12 s | best offline accuracy |
| `[56, 6]` / `[56, 3]` | 0.56 s / 0.32 s | interactive |
| `[56, 0]` | 80 ms | lowest latency |

## Live transcription (microphone)

`engine.run` needs a complete utterance. For audio that is still arriving, open a live session
on the prepared package — the same model residency, reached by downcasting the instance
`prepare` hands back:

```swift
guard let package = try await engine.prepare(.stt) as? NemotronSTTPackage else { return }
let live = try await package.startLiveTranscription(.init(attContextSize: [56, 3]))

// Read cumulative transcripts; the last one has isFinal == true.
Task { @MainActor in
    for try await update in live.updates {
        transcript = update.result.text
    }
}

// Feed it. push() is safe to call from an audio callback: it copies and returns,
// and never touches MLX — all inference runs on the engine's inference actor.
micTap { buffer in live.push(buffer) }   // 16 kHz mono Float32, any buffer size

live.finish()   // flush the tail and emit the final transcript (or cancel() to abandon)
```

`updates` ends with `CancellationError` if you cancel, or `NemotronSTTError.sessionInvalidated`
if the engine evicts the model underneath the session. One session at a time — a second
`startLiveTranscription` while one is open throws `NemotronSTTError.sessionActive`.

Use `AudioDecode.monoSamples(from:)` to turn a file into the samples a session wants.

## Design notes

- **Two shapes, one model.** `run()` is one-shot and utterance-shaped — a complete utterance in,
  a transcript out — and that is the engine's canonical `stt` contract, unchanged. Live partial
  hypotheses are a *package-specific* surface (`startLiveTranscription`) rather than a request
  type, because the engine's token-streaming contract streams output for a request whose input
  is known up front, which a microphone is not. Both share the core's cache-aware streaming
  session, and the core's `stream-gate` holds ragged live pushes to byte-identical output
  against the whole-buffer pass.
- **MLX never runs on the audio thread.** `push`/`finish`/`cancel` are `nonisolated` and only
  enqueue; the driver task is `@InferenceActor`, so live decoding serializes with `run()` and
  every other package exactly like normal engine work.
- **Born engine-clean.** Split memory footprint (bf16 1.24 GB resident + ~0.82 GB activation;
  int8 0.64 GB), `WeightSourcing` declared for ENGINE-EXECUTED materialization (the package
  has no materializer of its own — `load()` only guards; MAT gate green), and cooperative
  cancellation checkpointed per decoded chunk (CAN gate green). Conforms to the C0–C13 gate.

## Tests & gate

```bash
swift test                                   # C0–C13 + MAT-1..5 + CAN-1..3 + live guards (offline)
swift run nemotron-stt-gate <path.wav>       # register → prepare(.stt) → run via MLXServeEngine
swift run nemotron-stt-gate <path.wav> --live --cancel-after 2
swift run nemotron-stt-gate <path.wav> --metrics [--mat]   # [RUN]/[LIVE]/[MAT] evidence
```

`--live` pushes the same wav through `startLiveTranscription` in 100 ms buffers and requires the
final transcript to equal what `run()` produced (LIV-1), after interim partials (LIV-2), with a
sane processed duration (LIV-3); `--cancel-after` checks that cancelling mid-stream ends the
stream and leaves the package runnable (LIV-4).

`--metrics` measures the latency ladder the way a microphone feeds it (100 ms buffers,
unpaced, so wall time is compute). Every rung runs faster than realtime on an M5 Max —
80 ms at rtf 0.34, 1.12 s at rtf 0.04 — with peak flat at ~2.3 GB. `--mat` adds a cold
first-run download against a throwaway engine and an empty store: **1.28 GB in 19.4 s,
72 progress events**. Since this package has no materializer, that download succeeding is
the proof the engine executed it.

## Related

- [`xocialize/mlx-engine-swift`](https://github.com/xocialize/mlx-engine-swift) — the engine/contract.
- [`xocialize/nemotron-asr-mlx-swift`](https://github.com/xocialize/nemotron-asr-mlx-swift) — the core port + parity gates.
- [`xocialize/mlx-audio-dsp`](https://github.com/xocialize/mlx-audio-dsp) — shared STFT/mel primitives.
