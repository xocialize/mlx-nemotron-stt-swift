// MaterializationTests.swift — Nemotron STT through the engine's MAT gate (offline, no network):
// the WeightSourcing declaration, fresh-machine honesty, explicit-path satisfaction, and the
// store-layout probe/resolution. Single snapshot (config + checkpoint + tokenizer).

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest

@testable import MLXNemotronSTT

final class MaterializationTests: XCTestCase {

    /// Temp dir holding probe files that make an explicit-dir config read as satisfied.
    private func satisfiedDir() throws -> (dir: URL, cleanup: () -> Void) {
        let dir = FileManager.default.temporaryDirectory
            .appending(path: "nemotron-mat-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        for f in ["config.json", "model.safetensors"] {
            FileManager.default.createFile(atPath: dir.appending(path: f).path, contents: Data([0]))
        }
        return (dir, { try? FileManager.default.removeItem(at: dir) })
    }

    func testMATGate() throws {
        let (dir, cleanup) = try satisfiedDir()
        defer { cleanup() }
        let report = MaterializationConformance.check(
            freshConfiguration: NemotronSTTConfiguration(),
            satisfiedConfiguration: NemotronSTTConfiguration(modelDirectory: dir))
        XCTAssertTrue(report.passed, report.summary)
    }

    func testDeclaresSingleMainSource() {
        let sources = NemotronSTTConfiguration().weightSources
        XCTAssertEqual(sources.map(\.role), ["main"])
        XCTAssertEqual(sources[0].repo, "mlx-community/nemotron-3.5-asr-streaming-0.6b")
        XCTAssertEqual(sources[0].matching, ["config.json", "*.safetensors", "tokenizer*.json"])
    }

    func testFreshMachineReportsMissing() {
        // No explicit dir, no store root → the source is missing (honest fresh-machine posture).
        let missing = NemotronSTTConfiguration().missingWeightSources(storeRoot: nil)
        XCTAssertEqual(missing.map(\.role), ["main"])
    }

    func testExplicitDirSatisfies() throws {
        let (dir, cleanup) = try satisfiedDir()
        defer { cleanup() }
        let missing = NemotronSTTConfiguration(modelDirectory: dir).missingWeightSources(storeRoot: nil)
        XCTAssertTrue(missing.isEmpty)
    }
}
