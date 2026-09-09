// CanonicalLiveSurfaceTests.swift — the LIV gate's offline half against this package
// (MLXEngine contract 1.39.0, companion N2).
//
// Everything here runs with no weights and no kernel. The parts that need a real decode —
// LIV-2 (observed assembly), LIV-4 (sequence integrity) and LIV-5 (final parity vs
// `run(STTRequest)`) — are the `--live` mode of `nemotron-stt-gate`; LIV-5 is the one that
// promotes the plane, so it is a measured gate, not a unit test.

import Foundation
import MLXServeConformance
import MLXToolKit
import NemotronASR
import XCTest

@testable import MLXNemotronSTT

final class CanonicalLiveSurfaceTests: XCTestCase {

    /// LIV-1 — the descriptor declares a discipline and the type conforms. Both directions.
    func testLIV1AdvertisementCoherence() {
        let report = LiveTranscriptionConformance.checkAdvertisement(NemotronSTTPackage.self)
        XCTAssertTrue(report.passed, report.summary)
    }

    /// The declaration itself, pinned: `.cumulative`, and no claim to diarize or bias.
    func testDeclaredControls() throws {
        let surface = try XCTUnwrap(NemotronSTTPackage.manifest.surfaces.first)
        XCTAssertEqual(surface.sttControls?.liveDiscipline, .cumulative)
        XCTAssertEqual(surface.sttControls?.attributesSpeakers, false)
        XCTAssertEqual(surface.sttControls?.supportsContextBiasing, false)
        // The live declaration is a different ENTRY POINT, not a knob on `run`: it must not add
        // a parameter to the one-shot schema.
        XCTAssertEqual(surface.parameters.map(\.name), ["audio", "language"])
    }

    /// The canonical entry point refuses a field this surface does not declare, even when
    /// reached directly rather than through the engine's pre-flight. Silently dropping it is
    /// the violation 1.16.0 named.
    func testUndeclaredContextIsRefusedByThePackageItself() async {
        let package = NemotronSTTPackage(configuration: NemotronSTTConfiguration())
        do {
            _ = try await package.startLiveTranscription(
                STTSessionRequest(context: ["Nemotron", "MLXEngine"]))
            XCTFail("expected the undeclared context to be refused")
        } catch let error as PackageError {
            guard case .unsupportedRequestFeature(let detail) = error else {
                return XCTFail("expected unsupportedRequestFeature, got \(error)")
            }
            XCTAssertTrue(detail.contains("context"))
        } catch {
            XCTFail("expected PackageError, got \(error)")
        }
        // …and it is refused BEFORE the notLoaded check, so a caller learns about the field
        // rather than about residency.
    }

    /// The canonical surface needs a prepared model, same as the package-specific one.
    func testCanonicalStartBeforeLoadThrowsNotLoaded() async {
        let package = NemotronSTTPackage(configuration: NemotronSTTConfiguration())
        do {
            _ = try await package.startLiveTranscription(STTSessionRequest())
            XCTFail("expected notLoaded")
        } catch let error as PackageError {
            XCTAssertEqual(error, .notLoaded)
        } catch {
            XCTFail("expected PackageError.notLoaded, got \(error)")
        }
    }

    /// The chunk mapping, including the finding this adapter encodes: greedy RNN-T over a
    /// cache-aware encoder APPENDS and never retracts, so everything decoded is committed.
    /// `.cumulative` is a delivery rule, not a claim that anything may change.
    func testChunkMappingCommitsEverythingProcessed() {
        let tokens = [AlignedToken(id: 1, text: "hello", start: 0, duration: 0.5),
                      AlignedToken(id: 2, text: "there", start: 0.5, duration: 0.62)]
        let partial = NemotronLiveTranscription.Partial(
            result: AlignedResult(text: "hello there",
                                  sentences: [AlignedSentence(text: "hello there",
                                                              tokens: tokens)]),
            detectedLanguage: "en-US", processedSeconds: 1.12, isFinal: false)
        let chunk = NemotronSTTSession.chunk(from: partial, index: 3)

        XCTAssertEqual(chunk.text, "hello there")
        XCTAssertEqual(chunk.committedThrough, chunk.processedSeconds)
        XCTAssertEqual(chunk.index, 3)
        XCTAssertFalse(chunk.isFinal)
        XCTAssertEqual(chunk.segments.count, 1)
        // No speaker: this surface does not diarize and does not claim to.
        XCTAssertNil(chunk.segments.first?.speaker)
    }

    /// The rate the caller must convert to is declared, and samples at any other rate are
    /// DROPPED with a distinguishable outcome rather than transcribed as nonsense: 48 kHz
    /// samples read as 16 kHz produce confident garbage that nothing downstream can detect.
    func testSampleRateMismatchIsReportedNotTranscribed() {
        var budget = NemotronSTTSession.InputBudget(expectedSampleRate: 16_000,
                                                    maxBufferedSeconds: 8)
        XCTAssertEqual(budget.admit(sampleCount: 1_600, sampleRate: 48_000),
                       .unsupportedSampleRate)
        XCTAssertEqual(budget.admit(sampleCount: 1_600, sampleRate: 16_000), .accepted)
    }

    /// Back-pressure is the adapter's addition: the shipped session's input continuation is
    /// unbounded, so a caller outrunning the model would grow memory with no signal. The
    /// backlog is measured (`pushed − processedSeconds`), not estimated — so decoding
    /// progress is what makes room, exactly as it should be.
    func testOverrunIsReportedOnceTheBoundIsReachedAndClearsAsAudioIsConsumed() {
        var budget = NemotronSTTSession.InputBudget(expectedSampleRate: 16_000,
                                                    maxBufferedSeconds: 1)
        XCTAssertEqual(budget.admit(sampleCount: 8_000, sampleRate: 16_000), .accepted)
        XCTAssertEqual(budget.admit(sampleCount: 8_000, sampleRate: 16_000), .accepted)
        // Nothing decoded yet, so the backlog is still 1.0 s: the next buffer overruns.
        XCTAssertEqual(budget.admit(sampleCount: 8_000, sampleRate: 16_000), .overrun)

        budget.note(processed: 0.75)   // the model caught up on 0.75 s
        XCTAssertEqual(budget.admit(sampleCount: 8_000, sampleRate: 16_000), .accepted)

        budget.end()
        XCTAssertEqual(budget.admit(sampleCount: 8_000, sampleRate: 16_000), .ended)
        // `.ended` outranks a rate complaint: "stop pushing" is the more useful answer.
        XCTAssertEqual(budget.admit(sampleCount: 8_000, sampleRate: 48_000), .ended)
    }

    /// The default bound must clear one decode chunk with room to spare — `processedSeconds`
    /// trails the audio pushed by up to a chunk even when the model is keeping up, so a bound
    /// near the chunk length would report overrun on a healthy session.
    func testDefaultBufferBoundClearsTheSlowestChunk() {
        XCTAssertGreaterThan(NemotronSTTSession.defaultMaxBufferedSeconds, 1.12 * 2)
    }
}
