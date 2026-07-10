// CancellationTests.swift — Nemotron STT through the engine's CAN gate (offline, no MLX kernels).
// CAN-1/2 drive the real run() pre-cancelled (the entry checkpoint fires before notLoaded
// validation or weights); CAN-3 is the document of record for the checkpoint cadence.
//
// Nemotron ASR is cache-aware STREAMING: run() decodes the utterance chunk by chunk. The real
// seams: the wrapper's entry checkpoint (NemotronSTTPackage.run, first act), the per-chunk
// `shouldContinue` poll threaded into the core's streamGenerate (checked once per decoded chunk,
// which also reports RunProgress .decode), and the wrapper's post-decode checkpoint before the
// response is built. The cadence is per decoded chunk.

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest

@testable import MLXNemotronSTT

final class CancellationTests: XCTestCase {

    // MARK: - CAN-1 / CAN-2 — pre-cancelled run() propagation + classification

    func testCANGatePreCancelledRun() async {
        // Stub config; construction is cheap (C13) and the entry checkpoint throws before
        // validation or weights are touched, so this is offline-safe.
        let package = NemotronSTTPackage(configuration: NemotronSTTConfiguration())
        let report = await CancellationConformance.checkRun(
            package: package,
            request: STTRequest(audio: Audio(format: .wav, data: Data())))
        XCTAssertTrue(report.passed, report.summary)
    }

    // MARK: - CAN-3 — checkpoint-cadence declaration (the document of record)

    func testCANCadenceDeclaration() {
        // stt is a long-run capability (arbitrarily long audio) — no sub-second exemption.
        XCTAssertTrue(CancellationConformance.longRunImplied(by: NemotronSTTPackage.manifest))

        let report = CancellationConformance.checkCadence(
            manifest: NemotronSTTPackage.manifest,
            posture: .cadence([
                // One cancellation check + one RunProgress .decode report per decoded chunk.
                // Seams: wrapper entry (NemotronSTTPackage.run, first act), per-chunk
                // shouldContinue poll inside NemotronASRModel.streamGenerate, wrapper post-decode.
                .init(phase: .decode, unit: .chunk, reportsRunProgress: true)
            ]))
        XCTAssertTrue(report.passed, report.summary)
    }
}
