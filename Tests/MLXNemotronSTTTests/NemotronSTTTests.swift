import Foundation
import MLXServeCore
import MLXToolKit
import Testing

@testable import MLXNemotronSTT

@Test func manifestExposesSTTAndIsPermissive() {
    let manifest = NemotronSTTPackage.manifest
    #expect(manifest.capabilities == [.stt])
    #expect(LicensePolicy.permissiveOnly.evaluate(manifest.license).isAdmitted)
    #expect(manifest.surfaces.count == 1)
    #expect(manifest.surfaces[0].capability == .stt)
}

@Test func canonicalOutputIsText() {
    #expect(Capability.stt.canonicalOutput == .text)
}

@Test func registrationConstructsPackage() throws {
    // Construct only — load()/run() download weights + run on the GPU (device/integration test).
    let package = try NemotronSTTPackage.registration.makePackage(NemotronSTTConfiguration())
    #expect(package is NemotronSTTPackage)
}

@Test func footprintSplitDeclaresBothQuants() {
    let footprints = NemotronSTTPackage.manifest.requirements.footprints
    #expect(footprints.contains { $0.quant == .bf16 })
    #expect(footprints.contains { $0.quant == .int8 })
    // Split footprint (1.14): activation measured separately, not baked into residency.
    #expect(footprints.allSatisfy { $0.peakActivationBytes > 0 })
    // int8 weights floor is smaller than bf16.
    let bf16 = footprints.first { $0.quant == .bf16 }!
    let int8 = footprints.first { $0.quant == .int8 }!
    #expect(int8.residentBytes < bf16.residentBytes)
}

@Test func admissibleOnModestBudget() async {
    // bf16 weights ~1.24 GB + ~0.8 GB activation — admissible on a 3 GB budget.
    let device = DeviceProfile(
        chipTier: .base, macOS: SemanticVersion(major: 26, minor: 0, patch: 0),
        backends: [.metalGPU], totalMemoryBytes: 16_000_000_000)
    let engine = MLXServeEngine(
        device: device, governor: MemoryGovernor(budgetBytes: 3_000_000_000))
    let verdict = await engine.admissibility(for: NemotronSTTPackage.manifest.requirements)
    #expect(verdict.admissible)
    #expect(verdict.eligibility.isEligible)
}

@Test func quantConfigSelectsRepo() {
    let bf16 = NemotronSTTConfiguration()
    #expect(bf16.quant == .bf16)
    #expect(bf16.repo == "mlx-community/nemotron-3.5-asr-streaming-0.6b")

    let int8 = NemotronSTTConfiguration(
        repo: "mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit", quant: .int8)
    #expect(int8.quant == .int8)
}
