# Publish / promotion hand-off — Nemotron STT (Phase 6)

Everything below needs **git pushes to your remotes** (and a live Xcode build), so it was left
for you to run/authorize. All the code is built and validated locally; this is the wiring to ship it.

## State (2026-07-10)

| Piece | Location | Status |
|---|---|---|
| Engine `stt` capability (contract 1.20.0) | `~/Development/MLXEngine/mlx-engine-swift-stt` (worktree, branch `feat/stt-capability`) | committed, full suite green |
| Core `NemotronASR` | `mlxengine-audio/WIP/nemotron-asr-mlx-swift` | committed, all parity gates green |
| Kit `MLXNemotronSTT` | `mlxengine-audio/WIP/mlx-nemotron-stt-swift` | committed, C0–C13 + MAT + CAN green, live engine gate green |
| App voice input | `MLXCompanion` branch `feat/voice-input-stt` | committed, code-complete (graph move pending) |

Weights are already on `mlx-community/nemotron-3.5-asr-streaming-0.6b` (bf16) and `-8bit` — no weight publish needed.

## Steps

1. **Land + tag the engine.** Merge `feat/stt-capability` into the engine `main` and push, then
   tag **0.28.0** and push the tag. (1.19.0 is reserved for the unlanded `feat/meshrig-capability`
   branch — reconcile the changelog order when meshrig also lands.)
   ```
   cd ~/Development/MLXEngine/mlx-engine-swift
   git merge feat/stt-capability      # or land via PR
   git tag 0.28.0 && git push origin main --tags
   git worktree remove ../mlx-engine-swift-stt   # cleanup after merge
   ```

2. **Promote + publish the two packages** (WIP → PROD, push to `xocialize`):
   - `nemotron-asr-mlx-swift`: switch its `Package.swift` `mlx-audio-dsp` dep from the local path
     back to `.package(url: "https://github.com/xocialize/mlx-audio-dsp.git", from: "0.1.0")`,
     move to `mlxengine-audio/PROD/`, push, tag `0.1.0`.
   - `mlx-nemotron-stt-swift`: switch its engine dep from the local worktree path to
     `.package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.28.0")` and its core
     dep to the published `nemotron-asr-mlx-swift` URL/tag; move to PROD, push, tag `0.1.0`.

3. **Repoint the app SPM graph + build.** On `MLXCompanion` branch `feat/voice-input-stt`:
   - Bump the app's `mlx-engine-swift` package requirement to **0.28.0** (in Xcode: the
     `XCRemoteSwiftPackageReference "mlx-engine-swift"` version). Every other package
     (LLM/TTS/embed) resolves the same engine, so they move together — resolve and confirm.
   - Add the `mlx-nemotron-stt-swift` package (remote URL, from 0.1.0) and link the
     **MLXNemotronSTT** product to the `MLXCompanion` target (Xcode ▸ target ▸ Frameworks, or the
     package-product picker). The app code already imports it.
   - Build + launch. Smoke test: tap the mic, speak, confirm transcript → Roxy replies (LLM + TTS);
     toggle hands-free and confirm auto-endpointing + barge-in (speaking cuts her off). First run
     downloads the ~1.2 GB bf16 weights via the settings/progress panel.

4. **Registry + collection.**
   - Flip the registry row (`docs/model-registry.md`) `audio/WIP 🧪` → `audio/PROD ✅` and `Val 🟡`
     → `✅` once the in-app validation above passes.
   - Optional: group the bf16 + 8bit mlx-community repos into an HF Collection.

## Notes / gotchas captured during the port
See `~/.claude/skills/mlx-swift-integration/references/integration-lessons.md` ("First `stt`
capability") — streaming-model-≠-streaming-contract, class-level monkeypatch for goldens, the
consumer-app SPM interlock, illegal-key sanitize remaps, AVFoundation input decode.
