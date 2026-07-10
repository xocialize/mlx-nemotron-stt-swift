// AudioDecode.swift — turn a canonical `Audio` (.wav) artifact into the 16 kHz mono
// Float32 waveform the NemotronASR core expects. Uses AVFoundation for format-agnostic
// decode + sample-rate conversion so callers can pass any-rate/any-channel WAV.

import AVFoundation
import Foundation
import MLXToolKit

enum AudioDecode {
    enum DecodeError: Error, LocalizedError {
        case unreadable
        case emptyAudio
        var errorDescription: String? {
            switch self {
            case .unreadable: return "Could not decode the STT request audio."
            case .emptyAudio: return "The STT request audio was empty."
            }
        }
    }

    /// Decode WAV bytes → [Float] mono at `targetRate` (default 16 kHz).
    static func monoSamples(from audio: Audio, targetRate: Double = 16000) throws -> [Float] {
        // Write to a temp file so AVAudioFile can parse the container (it needs a URL).
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString).appendingPathExtension("wav")
        try audio.data.write(to: tmp)
        defer { try? FileManager.default.removeItem(at: tmp) }

        let file: AVAudioFile
        do {
            file = try AVAudioFile(forReading: tmp)
        } catch {
            throw DecodeError.unreadable
        }

        let outFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: targetRate,
            channels: 1, interleaved: false)!

        guard let converter = AVAudioConverter(from: file.processingFormat, to: outFormat) else {
            throw DecodeError.unreadable
        }

        let inFormat = file.processingFormat
        let inFrames = AVAudioFrameCount(file.length)
        guard inFrames > 0 else { throw DecodeError.emptyAudio }
        guard let inBuffer = AVAudioPCMBuffer(pcmFormat: inFormat, frameCapacity: inFrames) else {
            throw DecodeError.unreadable
        }
        try file.read(into: inBuffer)

        // Output capacity scaled by the resample ratio (+ headroom for filter tails).
        let ratio = targetRate / inFormat.sampleRate
        let outCapacity = AVAudioFrameCount(Double(inFrames) * ratio) + 1024
        guard let outBuffer = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: outCapacity)
        else { throw DecodeError.unreadable }

        var supplied = false
        var conversionError: NSError?
        let status = converter.convert(to: outBuffer, error: &conversionError) { _, outStatus in
            if supplied {
                outStatus.pointee = .noDataNow
                return nil
            }
            supplied = true
            outStatus.pointee = .haveData
            return inBuffer
        }
        if status == .error || conversionError != nil { throw DecodeError.unreadable }

        let count = Int(outBuffer.frameLength)
        guard count > 0, let channel = outBuffer.floatChannelData?[0] else {
            throw DecodeError.emptyAudio
        }
        return Array(UnsafeBufferPointer(start: channel, count: count))
    }
}
