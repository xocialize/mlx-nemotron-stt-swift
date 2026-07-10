# mlx-nemotron-stt-swift

![platform](https://img.shields.io/badge/platform-macOS%2026%2B%20%C2%B7%20Apple%20Silicon-black)
![swift](https://img.shields.io/badge/Swift-6.2-orange)
![engine](https://img.shields.io/badge/MLXEngine-%E2%89%A5%20v0.29.0-purple)
![capability](https://img.shields.io/badge/capability-stt-lightgrey)
![port code](https://img.shields.io/badge/port%20code-MIT-blue)
![weights](https://img.shields.io/badge/weights-OpenMDW--1.1-green)

The **MLXEngine `stt` (speech-to-text) package**: NVIDIA **Nemotron 3.5 ASR streaming** wrapped as
a conformant [`ModelPackage`](https://github.com/xocialize/mlx-engine-swift), returning the
canonical text transcript. Multilingual (40 language-locales), native punctuation and
capitalization, on-device on Apple Silicon.

- **Model:** [`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) — 600M cache-aware FastConformer-RNNT with language-ID prompt conditioning.
- **Core:** [`xocialize/nemotron-asr-mlx-swift`](https://github.com/xocialize/nemotron-asr-mlx-swift) (`NemotronASR`) — the MLX-Swift port, parity-locked bit-for-bit against the Python `Blaizzy/mlx-audio` reference.
- **Weights:** [`mlx-community/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/mlx-community/nemotron-3.5-asr-streaming-0.6b) (bf16, ~1.2 GB) and `-8bit`, auto-materialized on first run.
- **Capability:** `stt` (contract 1.20.0, engine ≥ v0.29.0). Canonical output: text.
- **License:** weights **OpenMDW-1.1** (permissive); port code **MIT**.

## Install

Add the package to your `Package.swift`:

```swift
.package(url: "https://github.com/xocialize/mlx-nemotron-stt-swift.git", from: "0.1.0"),
```

and the product to your target:

```swift
.product(name: "MLXNemotronSTT", package: "mlx-nemotron-stt-swift"),
```

Requires [`mlx-engine-swift`](https://github.com/xocialize/mlx-engine-swift) **≥ v0.29.0** (the
`stt` capability). The engine + core (`NemotronASR`) resolve transitively. In an Xcode app,
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

## Design notes

- **One-shot, utterance-shaped.** Capture a complete utterance → one `run()` → transcript. The
  model's cache-aware chunked streaming is package-internal memory discipline (bounded STFT +
  per-layer attention/conv ring-buffer caches), not a request surface. At ~27× realtime a few
  seconds of speech transcribes in well under a second, so turn-based feels instant. Live partial
  hypotheses are the engine's deferred token-streaming contract, not this surface.
- **Born engine-clean.** Split memory footprint (bf16 1.24 GB resident + ~0.82 GB activation;
  int8 0.64 GB), `WeightSourcing` auto-materialization (MAT gate green), and cooperative
  cancellation checkpointed per decoded chunk (CAN gate green). Conforms to the C0–C13 gate.

## Tests & gate

```bash
swift test                                   # C0–C13 + MAT-1..5 + CAN-1..3 (offline)
swift run nemotron-stt-gate <path.wav>       # live: register → prepare(.stt) → run via MLXServeEngine
```

## Related

- [`xocialize/mlx-engine-swift`](https://github.com/xocialize/mlx-engine-swift) — the engine/contract.
- [`xocialize/nemotron-asr-mlx-swift`](https://github.com/xocialize/nemotron-asr-mlx-swift) — the core port + parity gates.
- [`xocialize/mlx-audio-dsp`](https://github.com/xocialize/mlx-audio-dsp) — shared STFT/mel primitives.
