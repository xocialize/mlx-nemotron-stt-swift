# Publish record — Nemotron STT (Phase 6) — DONE 2026-07-10

Everything below shipped. Kept as the provenance record.

| Piece | Published | Tag |
|---|---|---|
| Engine `stt` capability (contract 1.20.0) | `xocialize/mlx-engine-swift` main | **v0.29.0** |
| Core `NemotronASR` | `xocialize/nemotron-asr-mlx-swift` | **v0.1.0** |
| Kit `MLXNemotronSTT` | `xocialize/mlx-nemotron-stt-swift` | **v0.1.0** |
| Weights (bf16 + 8bit) | `mlx-community/nemotron-3.5-asr-streaming-0.6b` (+ `-8bit`) | pre-existing |
| App voice input | `MLXCompanion` branch `feat/voice-input-stt` (local) | build succeeds |

Notes:
- Engine landed on `main` (fast-forward) + tag v0.29.0. v0.28.0/v0.28.1 were already taken; a
  new capability warranted the minor bump to 0.29.0. Contract 1.19.0 stays reserved for the
  unlanded `feat/meshrig-capability` worktree.
- Core consumes `mlx-audio-dsp` by URL; kit consumes engine v0.29.0 + core v0.1.0 by URL.
- App SPM graph: engine bumped to 0.29.0, `mlx-nemotron-stt-swift` 0.1.0 added and MLXNemotronSTT
  linked to the target. `xcodebuild … build` = **BUILD SUCCEEDED** against the published graph.
- Registry row flipped to `audio/PROD ✅ (v0.1.0)`.

## The one remaining step (needs a human at the mic)
Launch MLXCompanion (branch `feat/voice-input-stt`), tap the mic, speak, and confirm the
transcript → Roxy replies (LLM + TTS); toggle hands-free and confirm auto-endpoint + barge-in.
First run downloads the ~1.2 GB bf16 weights via the settings/progress panel. Then flip the
registry `Val` 🟡 → ✅.
