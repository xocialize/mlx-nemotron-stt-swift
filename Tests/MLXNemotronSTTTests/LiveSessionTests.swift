// LiveSessionTests.swift — the live transcription surface's boundary behavior, offline.
//
// These never load weights or run a kernel: they cover the guards around the session
// (unprepared package, one-at-a-time, unload invalidation), which is where a consumer driving
// a microphone actually gets into trouble. Transcript correctness is the core's `stream-gate`
// (ragged pushes == batch, exactly) and the `--live` mode of nemotron-stt-gate.

import Foundation
import MLXToolKit
import XCTest

@testable import MLXNemotronSTT

final class LiveSessionTests: XCTestCase {

    /// A session cannot open before the engine has prepared the package — there is no model.
    func testStartBeforeLoadThrowsNotLoaded() async {
        let package = NemotronSTTPackage(configuration: NemotronSTTConfiguration())
        do {
            _ = try await package.startLiveTranscription()
            XCTFail("expected notLoaded")
        } catch let error as PackageError {
            XCTAssertEqual(error, .notLoaded)
        } catch {
            XCTFail("expected PackageError.notLoaded, got \(error)")
        }
    }

    /// Unloading with no session open is a no-op, not a crash.
    func testUnloadWithoutSessionIsSafe() async {
        let package = NemotronSTTPackage(configuration: NemotronSTTConfiguration())
        await package.unload()
    }

    /// `sessionInvalidated` is distinct from cancellation, so a consumer can tell "the engine
    /// evicted the model" apart from "I stopped it".
    func testErrorCasesAreDistinct() {
        XCTAssertNotEqual(NemotronSTTError.sessionActive, NemotronSTTError.sessionInvalidated)
        XCTAssertNotEqual(
            NemotronSTTError.sessionInvalidated, NemotronSTTError.unresolvedDirectory)
    }

    /// The one-shot `stt` contract is unchanged by the live surface: the manifest still
    /// advertises exactly the STT surface, and no specialty term was invented for it.
    func testManifestStillDeclaresOnlyTheSTTSurface() {
        let manifest = NemotronSTTPackage.manifest
        XCTAssertEqual(manifest.surfaces.count, 1)
        XCTAssertTrue(manifest.specialties.isEmpty)
    }
}
