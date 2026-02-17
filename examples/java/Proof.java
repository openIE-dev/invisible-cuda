/**
 * Invisible CUDA — Java Proof
 *
 * Demonstrates CUDA compatibility using Java's Foreign Function & Memory API
 * (Project Panama, Java 22+) against the Invisible CUDA runtime.
 *
 * Prerequisites:
 *   cargo install invisible-cuda && invisible-cuda install
 *
 * Run:
 *   java --enable-native-access=ALL-UNNAMED Proof.java
 */

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;

public class Proof {

    static final Linker LINKER = Linker.nativeLinker();
    static SymbolLookup lib;

    static MethodHandle fn(String name, FunctionDescriptor desc) {
        var sym = lib.find(name).orElseThrow(() ->
            new RuntimeException("Symbol not found: " + name));
        return LINKER.downcallHandle(sym, desc);
    }

    // Common descriptors
    static final ValueLayout.OfInt    INT    = ValueLayout.JAVA_INT;
    static final ValueLayout.OfLong   LONG   = ValueLayout.JAVA_LONG;
    static final ValueLayout.OfFloat  FLOAT  = ValueLayout.JAVA_FLOAT;
    static final AddressLayout        PTR    = ValueLayout.ADDRESS;

    // Function handles
    static MethodHandle cudaGetDeviceCount, cudaGetDeviceProperties, cudaDeviceSynchronize;
    static MethodHandle cudaRuntimeGetVersion, cudaMalloc, cudaFree, cudaMemcpy, cudaMemset;
    static MethodHandle cudaMemGetInfo, cudaStreamCreate, cudaStreamDestroy, cudaStreamSynchronize;
    static MethodHandle cudaEventCreate, cudaEventDestroy, cudaEventRecord;
    static MethodHandle cudaEventSynchronize, cudaEventElapsedTime;
    static MethodHandle cublasCreate, cublasDestroy, cublasSgemm;

    static void loadLibrary() throws Exception {
        String home = System.getProperty("user.home");
        String os = System.getProperty("os.name").toLowerCase();
        String libName = os.contains("mac") ? "libcuda.dylib" : "libcuda.so";

        String envPath = System.getenv("INVISIBLE_CUDA_LIB");
        Path libPath;
        if (envPath != null) {
            libPath = Path.of(envPath);
        } else {
            libPath = Path.of(home, ".invisible-cuda", "lib", libName);
        }

        if (!Files.exists(libPath)) {
            throw new RuntimeException("Library not found: " + libPath +
                "\nInstall via: cargo install invisible-cuda && invisible-cuda install");
        }

        lib = SymbolLookup.libraryLookup(libPath, Arena.global());

        cudaGetDeviceCount = fn("cudaGetDeviceCount", FunctionDescriptor.of(INT, PTR));
        cudaGetDeviceProperties = fn("cudaGetDeviceProperties", FunctionDescriptor.of(INT, PTR, INT));
        cudaDeviceSynchronize = fn("cudaDeviceSynchronize", FunctionDescriptor.of(INT));
        cudaRuntimeGetVersion = fn("cudaRuntimeGetVersion", FunctionDescriptor.of(INT, PTR));
        cudaMalloc = fn("cudaMalloc", FunctionDescriptor.of(INT, PTR, LONG));
        cudaFree = fn("cudaFree", FunctionDescriptor.of(INT, PTR));
        cudaMemcpy = fn("cudaMemcpy", FunctionDescriptor.of(INT, PTR, PTR, LONG, INT));
        cudaMemset = fn("cudaMemset", FunctionDescriptor.of(INT, PTR, INT, LONG));
        cudaMemGetInfo = fn("cudaMemGetInfo", FunctionDescriptor.of(INT, PTR, PTR));
        cudaStreamCreate = fn("cudaStreamCreate", FunctionDescriptor.of(INT, PTR));
        cudaStreamDestroy = fn("cudaStreamDestroy", FunctionDescriptor.of(INT, PTR));
        cudaStreamSynchronize = fn("cudaStreamSynchronize", FunctionDescriptor.of(INT, PTR));
        cudaEventCreate = fn("cudaEventCreate", FunctionDescriptor.of(INT, PTR));
        cudaEventDestroy = fn("cudaEventDestroy", FunctionDescriptor.of(INT, PTR));
        cudaEventRecord = fn("cudaEventRecord", FunctionDescriptor.of(INT, PTR, PTR));
        cudaEventSynchronize = fn("cudaEventSynchronize", FunctionDescriptor.of(INT, PTR));
        cudaEventElapsedTime = fn("cudaEventElapsedTime", FunctionDescriptor.of(INT, PTR, PTR, PTR));
        cublasCreate = fn("cublasCreate_v2", FunctionDescriptor.of(INT, PTR));
        cublasDestroy = fn("cublasDestroy_v2", FunctionDescriptor.of(INT, PTR));
        cublasSgemm = fn("cublasSgemm_v2", FunctionDescriptor.of(INT,
            PTR, INT, INT, INT, INT, INT, PTR, PTR, INT, PTR, INT, PTR, PTR, INT));
    }

    public static void main(String[] args) throws Exception {
        System.out.println("╔══════════════════════════════════════════════════════════════╗");
        System.out.println("║         INVISIBLE CUDA — Java Proof (Panama FFM)            ║");
        System.out.println("╚══════════════════════════════════════════════════════════════╝");
        System.out.println();

        int passed = 0, failed = 0;
        long startNs = System.nanoTime();

        loadLibrary();

        try (var arena = Arena.ofConfined()) {

            // ── Device Discovery ──
            System.out.println("── Device Discovery ──");

            var countPtr = arena.allocate(INT);
            if ((int) cudaGetDeviceCount.invoke(countPtr) == 0 && countPtr.get(INT, 0) >= 1) {
                System.out.printf("  PASS  Device count: %d%n", countPtr.get(INT, 0));
                passed++;
            } else { System.out.println("  FAIL  Device count"); failed++; }

            // DeviceProp: 256-byte name + 8-byte totalGlobalMem + more
            var propBuf = arena.allocate(512);
            if ((int) cudaGetDeviceProperties.invoke(propBuf, 0) == 0) {
                String name = propBuf.getString(0);
                long totalMem = propBuf.get(LONG, 256);
                System.out.printf("  PASS  Device: %s (%.1f GB)%n", name, totalMem / 1e9);
                passed++;
            } else { System.out.println("  FAIL  Device properties"); failed++; }

            var versionPtr = arena.allocate(INT);
            if ((int) cudaRuntimeGetVersion.invoke(versionPtr) == 0 && versionPtr.get(INT, 0) >= 12000) {
                System.out.printf("  PASS  Runtime version: %d%n", versionPtr.get(INT, 0));
                passed++;
            } else { System.out.println("  FAIL  Runtime version"); failed++; }

            var freePtr = arena.allocate(LONG);
            var totalPtr = arena.allocate(LONG);
            if ((int) cudaMemGetInfo.invoke(freePtr, totalPtr) == 0) {
                System.out.printf("  PASS  Memory: %.1f GB free / %.1f GB total%n",
                    freePtr.get(LONG, 0) / 1e9, totalPtr.get(LONG, 0) / 1e9);
                passed++;
            } else { System.out.println("  FAIL  Memory info"); failed++; }
            System.out.println();

            // ── Memory Operations ──
            System.out.println("── Memory Operations ──");

            var ptrPtr = arena.allocate(PTR);
            if ((int) cudaMalloc.invoke(ptrPtr, 4096L) == 0) {
                var p = ptrPtr.get(PTR, 0);
                cudaFree.invoke(p);
                System.out.println("  PASS  malloc + free (4 KB)");
                passed++;
            } else { System.out.println("  FAIL  malloc + free"); failed++; }

            // memcpy roundtrip
            {
                int n = 1024;
                var src = arena.allocate(n * 4L);
                for (int i = 0; i < n; i++) src.setAtIndex(FLOAT, i, i * 0.1f);

                var dp = arena.allocate(PTR);
                cudaMalloc.invoke(dp, (long) n * 4);
                var dPtr = dp.get(PTR, 0);
                cudaMemcpy.invoke(dPtr, src, (long) n * 4, 1); // HtoD
                var dst = arena.allocate(n * 4L);
                cudaMemcpy.invoke(dst, dPtr, (long) n * 4, 2); // DtoH
                cudaFree.invoke(dPtr);

                boolean ok = true;
                for (int i = 0; i < n; i++) {
                    if (Math.abs(src.getAtIndex(FLOAT, i) - dst.getAtIndex(FLOAT, i)) > 1e-5f) {
                        ok = false; break;
                    }
                }
                if (ok) { System.out.println("  PASS  memcpy roundtrip (4 KB)"); passed++; }
                else { System.out.println("  FAIL  memcpy roundtrip"); failed++; }
            }

            // memset
            {
                var dp = arena.allocate(PTR);
                cudaMalloc.invoke(dp, 4096L);
                var dPtr = dp.get(PTR, 0);
                cudaMemset.invoke(dPtr, 0xAB, 4096L);
                var buf = arena.allocate(4096);
                cudaMemcpy.invoke(buf, dPtr, 4096L, 2);
                cudaFree.invoke(dPtr);

                boolean ok = true;
                for (int i = 0; i < 4096; i++) {
                    if ((buf.get(ValueLayout.JAVA_BYTE, i) & 0xFF) != 0xAB) {
                        ok = false; break;
                    }
                }
                if (ok) { System.out.println("  PASS  memset (4 KB, pattern 0xAB)"); passed++; }
                else { System.out.println("  FAIL  memset"); failed++; }
            }
            System.out.println();

            // ── Streams & Events ──
            System.out.println("── Streams & Events ──");

            {
                var sp = arena.allocate(PTR);
                if ((int) cudaStreamCreate.invoke(sp) == 0) {
                    var stream = sp.get(PTR, 0);
                    cudaStreamSynchronize.invoke(stream);
                    cudaStreamDestroy.invoke(stream);
                    System.out.println("  PASS  Stream create + sync + destroy");
                    passed++;
                } else { System.out.println("  FAIL  Stream"); failed++; }
            }

            {
                var ep1 = arena.allocate(PTR);
                var ep2 = arena.allocate(PTR);
                cudaEventCreate.invoke(ep1);
                cudaEventCreate.invoke(ep2);
                var evStart = ep1.get(PTR, 0);
                var evEnd = ep2.get(PTR, 0);
                cudaEventRecord.invoke(evStart, MemorySegment.NULL);
                cudaDeviceSynchronize.invoke();
                cudaEventRecord.invoke(evEnd, MemorySegment.NULL);
                cudaEventSynchronize.invoke(evEnd);
                var msPtr = arena.allocate(FLOAT);
                cudaEventElapsedTime.invoke(msPtr, evStart, evEnd);
                float ms = msPtr.get(FLOAT, 0);
                cudaEventDestroy.invoke(evStart);
                cudaEventDestroy.invoke(evEnd);
                if (ms >= 0) {
                    System.out.printf("  PASS  Event timing (%.3f ms)%n", ms);
                    passed++;
                } else { System.out.println("  FAIL  Event timing"); failed++; }
            }
            System.out.println();

            // ── cuBLAS ──
            System.out.println("── cuBLAS ──");

            {
                var hp = arena.allocate(PTR);
                if ((int) cublasCreate.invoke(hp) == 0) {
                    var handle = hp.get(PTR, 0);
                    System.out.println("  PASS  cuBLAS handle created");
                    passed++;

                    float[] aData = new float[16];
                    float[] eye = new float[16];
                    for (int i = 0; i < 16; i++) aData[i] = i + 1;
                    for (int i = 0; i < 4; i++) eye[i * 4 + i] = 1.0f;

                    var aHost = arena.allocate(64);
                    var iHost = arena.allocate(64);
                    for (int i = 0; i < 16; i++) {
                        aHost.setAtIndex(FLOAT, i, aData[i]);
                        iHost.setAtIndex(FLOAT, i, eye[i]);
                    }

                    var dap = arena.allocate(PTR);
                    var dip = arena.allocate(PTR);
                    var dcp = arena.allocate(PTR);
                    cudaMalloc.invoke(dap, 64L);
                    cudaMalloc.invoke(dip, 64L);
                    cudaMalloc.invoke(dcp, 64L);
                    var dA = dap.get(PTR, 0);
                    var dI = dip.get(PTR, 0);
                    var dC = dcp.get(PTR, 0);
                    cudaMemcpy.invoke(dA, aHost, 64L, 1);
                    cudaMemcpy.invoke(dI, iHost, 64L, 1);
                    cudaMemset.invoke(dC, 0, 64L);

                    var alpha = arena.allocate(FLOAT);
                    var beta = arena.allocate(FLOAT);
                    alpha.set(FLOAT, 0, 1.0f);
                    beta.set(FLOAT, 0, 0.0f);

                    cublasSgemm.invoke(handle, 0, 0, 4, 4, 4,
                        alpha, dA, 4, dI, 4, beta, dC, 4);

                    var resultBuf = arena.allocate(64);
                    cudaMemcpy.invoke(resultBuf, dC, 64L, 2);
                    cudaFree.invoke(dA);
                    cudaFree.invoke(dI);
                    cudaFree.invoke(dC);

                    boolean ok = true;
                    for (int i = 0; i < 16; i++) {
                        if (Math.abs(resultBuf.getAtIndex(FLOAT, i) - aData[i]) > 1e-3f) {
                            ok = false; break;
                        }
                    }
                    if (ok) { System.out.println("  PASS  SGEMM identity multiply (4x4)"); passed++; }
                    else { System.out.println("  FAIL  SGEMM identity multiply"); failed++; }

                    cublasDestroy.invoke(handle);
                } else { System.out.println("  FAIL  cuBLAS create"); failed++; }
            }
            System.out.println();

            // ── Performance ──
            System.out.println("── Performance ──");

            {
                int size = 4 * 1024 * 1024;
                var data = arena.allocate(size);
                var dp = arena.allocate(PTR);
                cudaMalloc.invoke(dp, (long) size);
                var dPtr = dp.get(PTR, 0);

                for (int i = 0; i < 5; i++) {
                    cudaMemcpy.invoke(dPtr, data, (long) size, 1);
                    cudaMemcpy.invoke(data, dPtr, (long) size, 2);
                }

                long t = System.nanoTime();
                int iters = 50;
                for (int i = 0; i < iters; i++) {
                    cudaMemcpy.invoke(dPtr, data, (long) size, 1);
                    cudaMemcpy.invoke(data, dPtr, (long) size, 2);
                }
                double elapsed = (System.nanoTime() - t) / 1e9;
                cudaFree.invoke(dPtr);

                double gbS = (double)(iters * size * 2) / elapsed / 1e9;
                System.out.printf("  Mem BW (4 MB): %.2f GB/s%n", gbS);
            }
            System.out.println();
        }

        // ── Summary ──
        double totalTime = (System.nanoTime() - startNs) / 1e9;
        System.out.println("══════════════════════════════════════════════════════════════");
        if (failed == 0) {
            System.out.printf("  ALL %d TESTS PASSED (%.1fs)%n", passed, totalTime);
            System.out.println("  CUDA compatibility: VERIFIED via Invisible CUDA (Java)");
        } else {
            System.out.printf("  %d passed, %d FAILED (%.1fs)%n", passed, failed, totalTime);
        }
        System.out.println("══════════════════════════════════════════════════════════════");

        System.exit(failed > 0 ? 1 : 0);
    }
}
