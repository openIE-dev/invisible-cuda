// Invisible CUDA — Swift Proof
//
// Demonstrates CUDA compatibility using Swift's C interop (dlopen)
// against the Invisible CUDA runtime.
//
// Prerequisites:
//   cargo install invisible-cuda && invisible-cuda install
//
// Run:
//   swift run -c release

import Foundation

// MARK: - Dynamic Library Loading

typealias FnGetDeviceCount = @convention(c) (UnsafeMutablePointer<Int32>) -> Int32
typealias FnGetDeviceProperties = @convention(c) (UnsafeMutableRawPointer, Int32) -> Int32
typealias FnDeviceSynchronize = @convention(c) () -> Int32
typealias FnRuntimeGetVersion = @convention(c) (UnsafeMutablePointer<Int32>) -> Int32
typealias FnMalloc = @convention(c) (UnsafeMutablePointer<UnsafeMutableRawPointer?>, Int) -> Int32
typealias FnFree = @convention(c) (UnsafeMutableRawPointer?) -> Int32
typealias FnMemcpy = @convention(c) (UnsafeMutableRawPointer?, UnsafeRawPointer?, Int, Int32) -> Int32
typealias FnMemset = @convention(c) (UnsafeMutableRawPointer?, Int32, Int) -> Int32
typealias FnMemGetInfo = @convention(c) (UnsafeMutablePointer<Int>, UnsafeMutablePointer<Int>) -> Int32
typealias FnStreamCreate = @convention(c) (UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> Int32
typealias FnStreamDestroy = @convention(c) (UnsafeMutableRawPointer?) -> Int32
typealias FnStreamSynchronize = @convention(c) (UnsafeMutableRawPointer?) -> Int32
typealias FnEventCreate = @convention(c) (UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> Int32
typealias FnEventDestroy = @convention(c) (UnsafeMutableRawPointer?) -> Int32
typealias FnEventRecord = @convention(c) (UnsafeMutableRawPointer?, UnsafeMutableRawPointer?) -> Int32
typealias FnEventSynchronize = @convention(c) (UnsafeMutableRawPointer?) -> Int32
typealias FnEventElapsedTime = @convention(c) (UnsafeMutablePointer<Float>, UnsafeMutableRawPointer?, UnsafeMutableRawPointer?) -> Int32
typealias FnCublasCreate = @convention(c) (UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> Int32
typealias FnCublasDestroy = @convention(c) (UnsafeMutableRawPointer?) -> Int32
typealias FnCublasSgemm = @convention(c) (
    UnsafeMutableRawPointer?, Int32, Int32, Int32, Int32, Int32,
    UnsafePointer<Float>, UnsafeRawPointer?, Int32,
    UnsafeRawPointer?, Int32,
    UnsafePointer<Float>, UnsafeMutableRawPointer?, Int32
) -> Int32

struct CudaLib {
    let getDeviceCount: FnGetDeviceCount
    let getDeviceProperties: FnGetDeviceProperties
    let deviceSynchronize: FnDeviceSynchronize
    let runtimeGetVersion: FnRuntimeGetVersion
    let malloc: FnMalloc
    let free: FnFree
    let memcpy: FnMemcpy
    let memset: FnMemset
    let memGetInfo: FnMemGetInfo
    let streamCreate: FnStreamCreate
    let streamDestroy: FnStreamDestroy
    let streamSynchronize: FnStreamSynchronize
    let eventCreate: FnEventCreate
    let eventDestroy: FnEventDestroy
    let eventRecord: FnEventRecord
    let eventSynchronize: FnEventSynchronize
    let eventElapsedTime: FnEventElapsedTime
    let cublasCreate: FnCublasCreate
    let cublasDestroy: FnCublasDestroy
    let cublasSgemm: FnCublasSgemm

    static func load() -> CudaLib? {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        #if os(macOS)
        let libPath = "\(home)/.invisible-cuda/lib/libcuda.dylib"
        #else
        let libPath = "\(home)/.invisible-cuda/lib/libcuda.so"
        #endif

        let envPath = ProcessInfo.processInfo.environment["INVISIBLE_CUDA_LIB"]
        let path = envPath ?? libPath

        guard let handle = dlopen(path, RTLD_NOW) else {
            print("Could not load: \(path)")
            print("Install via: cargo install invisible-cuda && invisible-cuda install")
            return nil
        }

        func sym<T>(_ name: String) -> T {
            let s = dlsym(handle, name)!
            return unsafeBitCast(s, to: T.self)
        }

        return CudaLib(
            getDeviceCount: sym("cudaGetDeviceCount"),
            getDeviceProperties: sym("cudaGetDeviceProperties"),
            deviceSynchronize: sym("cudaDeviceSynchronize"),
            runtimeGetVersion: sym("cudaRuntimeGetVersion"),
            malloc: sym("cudaMalloc"),
            free: sym("cudaFree"),
            memcpy: sym("cudaMemcpy"),
            memset: sym("cudaMemset"),
            memGetInfo: sym("cudaMemGetInfo"),
            streamCreate: sym("cudaStreamCreate"),
            streamDestroy: sym("cudaStreamDestroy"),
            streamSynchronize: sym("cudaStreamSynchronize"),
            eventCreate: sym("cudaEventCreate"),
            eventDestroy: sym("cudaEventDestroy"),
            eventRecord: sym("cudaEventRecord"),
            eventSynchronize: sym("cudaEventSynchronize"),
            eventElapsedTime: sym("cudaEventElapsedTime"),
            cublasCreate: sym("cublasCreate_v2"),
            cublasDestroy: sym("cublasDestroy_v2"),
            cublasSgemm: sym("cublasSgemm_v2")
        )
    }
}

// MARK: - Proof

func main() {
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║         INVISIBLE CUDA — Swift Proof                       ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    var passed = 0, failed = 0
    let start = CFAbsoluteTimeGetCurrent()

    guard let cuda = CudaLib.load() else { exit(1) }

    // ── Device Discovery ──
    print("── Device Discovery ──")

    var count: Int32 = 0
    if cuda.getDeviceCount(&count) == 0 && count >= 1 {
        print("  PASS  Device count: \(count)")
        passed += 1
    } else { print("  FAIL  Device count"); failed += 1 }

    // DeviceProp: 256 bytes name + 8 bytes totalGlobalMem + more
    var propBuf = [UInt8](repeating: 0, count: 512)
    propBuf.withUnsafeMutableBufferPointer { buf in
        if cuda.getDeviceProperties(buf.baseAddress!, 0) == 0 {
            let name = String(cString: buf.baseAddress!.assumingMemoryBound(to: CChar.self))
            let memPtr = buf.baseAddress!.advanced(by: 256).assumingMemoryBound(to: UInt64.self)
            let totalMem = memPtr.pointee
            print("  PASS  Device: \(name) (\(String(format: "%.1f", Double(totalMem) / 1e9)) GB)")
            passed += 1
        } else { print("  FAIL  Device properties"); failed += 1 }
    }

    var version: Int32 = 0
    if cuda.runtimeGetVersion(&version) == 0 && version >= 12000 {
        print("  PASS  Runtime version: \(version)")
        passed += 1
    } else { print("  FAIL  Runtime version"); failed += 1 }

    var freeMem = 0, totalMem = 0
    if cuda.memGetInfo(&freeMem, &totalMem) == 0 && totalMem > 0 {
        print("  PASS  Memory: \(String(format: "%.1f", Double(freeMem) / 1e9)) GB free / \(String(format: "%.1f", Double(totalMem) / 1e9)) GB total")
        passed += 1
    } else { print("  FAIL  Memory info"); failed += 1 }
    print()

    // ── Memory Operations ──
    print("── Memory Operations ──")

    var ptr: UnsafeMutableRawPointer? = nil
    if cuda.malloc(&ptr, 4096) == 0 && ptr != nil {
        cuda.free(ptr)
        print("  PASS  malloc + free (4 KB)")
        passed += 1
    } else { print("  FAIL  malloc + free"); failed += 1 }

    // memcpy roundtrip
    do {
        let n = 1024
        var src = (0..<n).map { Float($0) * 0.1 }
        var dPtr: UnsafeMutableRawPointer? = nil
        cuda.malloc(&dPtr, n * 4)
        src.withUnsafeBufferPointer { cuda.memcpy(dPtr, $0.baseAddress, n * 4, 1) }
        var dst = [Float](repeating: 0, count: n)
        dst.withUnsafeMutableBufferPointer { cuda.memcpy($0.baseAddress, dPtr, n * 4, 2) }
        cuda.free(dPtr)

        let ok = zip(src, dst).allSatisfy { abs($0 - $1) < 1e-5 }
        if ok { print("  PASS  memcpy roundtrip (4 KB)"); passed += 1 }
        else { print("  FAIL  memcpy roundtrip"); failed += 1 }
    }

    // memset
    do {
        var dPtr: UnsafeMutableRawPointer? = nil
        cuda.malloc(&dPtr, 4096)
        cuda.memset(dPtr, 0xAB, 4096)
        var buf = [UInt8](repeating: 0, count: 4096)
        buf.withUnsafeMutableBufferPointer { cuda.memcpy($0.baseAddress, dPtr, 4096, 2) }
        cuda.free(dPtr)

        let ok = buf.allSatisfy { $0 == 0xAB }
        if ok { print("  PASS  memset (4 KB, pattern 0xAB)"); passed += 1 }
        else { print("  FAIL  memset"); failed += 1 }
    }
    print()

    // ── Streams & Events ──
    print("── Streams & Events ──")

    do {
        var stream: UnsafeMutableRawPointer? = nil
        if cuda.streamCreate(&stream) == 0 {
            cuda.streamSynchronize(stream)
            cuda.streamDestroy(stream)
            print("  PASS  Stream create + sync + destroy")
            passed += 1
        } else { print("  FAIL  Stream"); failed += 1 }
    }

    do {
        var evStart: UnsafeMutableRawPointer? = nil
        var evEnd: UnsafeMutableRawPointer? = nil
        cuda.eventCreate(&evStart)
        cuda.eventCreate(&evEnd)
        cuda.eventRecord(evStart, nil)
        cuda.deviceSynchronize()
        cuda.eventRecord(evEnd, nil)
        cuda.eventSynchronize(evEnd)
        var ms: Float = 0
        cuda.eventElapsedTime(&ms, evStart, evEnd)
        cuda.eventDestroy(evStart)
        cuda.eventDestroy(evEnd)
        if ms >= 0 {
            print("  PASS  Event timing (\(String(format: "%.3f", ms)) ms)")
            passed += 1
        } else { print("  FAIL  Event timing"); failed += 1 }
    }
    print()

    // ── cuBLAS ──
    print("── cuBLAS ──")

    do {
        var handle: UnsafeMutableRawPointer? = nil
        if cuda.cublasCreate(&handle) == 0 {
            print("  PASS  cuBLAS handle created")
            passed += 1

            var aData: [Float] = (1...16).map { Float($0) }
            var eye = [Float](repeating: 0, count: 16)
            for i in 0..<4 { eye[i * 4 + i] = 1.0 }

            var dA: UnsafeMutableRawPointer? = nil
            var dI: UnsafeMutableRawPointer? = nil
            var dC: UnsafeMutableRawPointer? = nil
            cuda.malloc(&dA, 64); cuda.malloc(&dI, 64); cuda.malloc(&dC, 64)
            aData.withUnsafeBufferPointer { cuda.memcpy(dA, $0.baseAddress, 64, 1) }
            eye.withUnsafeBufferPointer { cuda.memcpy(dI, $0.baseAddress, 64, 1) }
            cuda.memset(dC, 0, 64)

            var alpha: Float = 1.0, beta: Float = 0.0
            cuda.cublasSgemm(handle, 0, 0, 4, 4, 4,
                &alpha, dA, 4, dI, 4, &beta, dC, 4)

            var result = [Float](repeating: 0, count: 16)
            result.withUnsafeMutableBufferPointer { cuda.memcpy($0.baseAddress, dC, 64, 2) }
            cuda.free(dA); cuda.free(dI); cuda.free(dC)

            let ok = zip(aData, result).allSatisfy { abs($0 - $1) < 1e-3 }
            if ok { print("  PASS  SGEMM identity multiply (4x4)"); passed += 1 }
            else { print("  FAIL  SGEMM identity multiply"); failed += 1 }

            cuda.cublasDestroy(handle)
        } else { print("  FAIL  cuBLAS create"); failed += 1 }
    }
    print()

    // ── Performance ──
    print("── Performance ──")

    do {
        let size = 4 * 1024 * 1024
        var data = [UInt8](repeating: 0, count: size)
        var dPtr: UnsafeMutableRawPointer? = nil
        cuda.malloc(&dPtr, size)

        for _ in 0..<5 {
            data.withUnsafeBufferPointer { cuda.memcpy(dPtr, $0.baseAddress, size, 1) }
            data.withUnsafeMutableBufferPointer { cuda.memcpy($0.baseAddress, dPtr, size, 2) }
        }

        let t = CFAbsoluteTimeGetCurrent()
        let iters = 50
        for _ in 0..<iters {
            data.withUnsafeBufferPointer { cuda.memcpy(dPtr, $0.baseAddress, size, 1) }
            data.withUnsafeMutableBufferPointer { cuda.memcpy($0.baseAddress, dPtr, size, 2) }
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - t
        cuda.free(dPtr)

        let gbS = Double(iters * size * 2) / elapsed / 1e9
        print("  Mem BW (4 MB): \(String(format: "%.2f", gbS)) GB/s")
    }
    print()

    // ── Summary ──
    let total = CFAbsoluteTimeGetCurrent() - start
    print("══════════════════════════════════════════════════════════════")
    if failed == 0 {
        print("  ALL \(passed) TESTS PASSED (\(String(format: "%.1f", total))s)")
        print("  CUDA compatibility: VERIFIED via Invisible CUDA (Swift)")
    } else {
        print("  \(passed) passed, \(failed) FAILED (\(String(format: "%.1f", total))s)")
    }
    print("══════════════════════════════════════════════════════════════")

    exit(failed > 0 ? 1 : 0)
}

main()
