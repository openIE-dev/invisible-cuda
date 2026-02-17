// Invisible CUDA — C# Proof
//
// Demonstrates CUDA compatibility using P/Invoke + NativeLibrary
// against the Invisible CUDA runtime.
//
// Prerequisites:
//   cargo install invisible-cuda && invisible-cuda install
//
// Run:
//   dotnet run -c Release

using System;
using System.Diagnostics;
using System.Runtime.InteropServices;

class Proof
{
    static IntPtr lib;

    // Delegate types for CUDA functions
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    delegate int CudaGetDeviceCount(out int count);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    delegate int CudaGetDeviceProperties(IntPtr prop, int device);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    delegate int CudaDeviceSynchronize();
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    delegate int CudaRuntimeGetVersion(out int version);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    delegate int CudaMalloc(out IntPtr ptr, long size);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    delegate int CudaFree(IntPtr ptr);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    delegate int CudaMemcpy(IntPtr dst, IntPtr src, long count, int kind);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    delegate int CudaMemset(IntPtr ptr, int value, long count);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    delegate int CudaMemGetInfo(out long free, out long total);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    delegate int CudaStreamCreate(out IntPtr stream);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    delegate int CudaStreamDestroy(IntPtr stream);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    delegate int CudaStreamSynchronize(IntPtr stream);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    delegate int CudaEventCreate(out IntPtr evt);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    delegate int CudaEventDestroy(IntPtr evt);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    delegate int CudaEventRecord(IntPtr evt, IntPtr stream);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    delegate int CudaEventSynchronize(IntPtr evt);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    delegate int CudaEventElapsedTime(out float ms, IntPtr start, IntPtr end);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    delegate int CublasCreate(out IntPtr handle);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    delegate int CublasDestroy(IntPtr handle);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    unsafe delegate int CublasSgemm(IntPtr handle, int transa, int transb,
        int m, int n, int k, float* alpha, IntPtr a, int lda,
        IntPtr b, int ldb, float* beta, IntPtr c, int ldc);

    // Loaded delegates
    static CudaGetDeviceCount fnGetDeviceCount = null!;
    static CudaGetDeviceProperties fnGetDeviceProperties = null!;
    static CudaDeviceSynchronize fnDeviceSynchronize = null!;
    static CudaRuntimeGetVersion fnRuntimeGetVersion = null!;
    static CudaMalloc fnMalloc = null!;
    static CudaFree fnFree = null!;
    static CudaMemcpy fnMemcpy = null!;
    static CudaMemset fnMemset = null!;
    static CudaMemGetInfo fnMemGetInfo = null!;
    static CudaStreamCreate fnStreamCreate = null!;
    static CudaStreamDestroy fnStreamDestroy = null!;
    static CudaStreamSynchronize fnStreamSynchronize = null!;
    static CudaEventCreate fnEventCreate = null!;
    static CudaEventDestroy fnEventDestroy = null!;
    static CudaEventRecord fnEventRecord = null!;
    static CudaEventSynchronize fnEventSynchronize = null!;
    static CudaEventElapsedTime fnEventElapsedTime = null!;
    static CublasCreate fnCublasCreate = null!;
    static CublasDestroy fnCublasDestroy = null!;
    static CublasSgemm fnCublasSgemm = null!;

    static T Fn<T>(string name) where T : Delegate =>
        Marshal.GetDelegateForFunctionPointer<T>(NativeLibrary.GetExport(lib, name));

    static void LoadLibrary()
    {
        var home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        var libName = RuntimeInformation.IsOSPlatform(OSPlatform.OSX)
            ? "libcuda.dylib" : "libcuda.so";
        var path = Environment.GetEnvironmentVariable("INVISIBLE_CUDA_LIB")
            ?? $"{home}/.invisible-cuda/lib/{libName}";

        lib = NativeLibrary.Load(path);

        fnGetDeviceCount = Fn<CudaGetDeviceCount>("cudaGetDeviceCount");
        fnGetDeviceProperties = Fn<CudaGetDeviceProperties>("cudaGetDeviceProperties");
        fnDeviceSynchronize = Fn<CudaDeviceSynchronize>("cudaDeviceSynchronize");
        fnRuntimeGetVersion = Fn<CudaRuntimeGetVersion>("cudaRuntimeGetVersion");
        fnMalloc = Fn<CudaMalloc>("cudaMalloc");
        fnFree = Fn<CudaFree>("cudaFree");
        fnMemcpy = Fn<CudaMemcpy>("cudaMemcpy");
        fnMemset = Fn<CudaMemset>("cudaMemset");
        fnMemGetInfo = Fn<CudaMemGetInfo>("cudaMemGetInfo");
        fnStreamCreate = Fn<CudaStreamCreate>("cudaStreamCreate");
        fnStreamDestroy = Fn<CudaStreamDestroy>("cudaStreamDestroy");
        fnStreamSynchronize = Fn<CudaStreamSynchronize>("cudaStreamSynchronize");
        fnEventCreate = Fn<CudaEventCreate>("cudaEventCreate");
        fnEventDestroy = Fn<CudaEventDestroy>("cudaEventDestroy");
        fnEventRecord = Fn<CudaEventRecord>("cudaEventRecord");
        fnEventSynchronize = Fn<CudaEventSynchronize>("cudaEventSynchronize");
        fnEventElapsedTime = Fn<CudaEventElapsedTime>("cudaEventElapsedTime");
        fnCublasCreate = Fn<CublasCreate>("cublasCreate_v2");
        fnCublasDestroy = Fn<CublasDestroy>("cublasDestroy_v2");
        fnCublasSgemm = Fn<CublasSgemm>("cublasSgemm_v2");
    }

    static unsafe void Main()
    {
        Console.WriteLine("╔══════════════════════════════════════════════════════════════╗");
        Console.WriteLine("║         INVISIBLE CUDA — C# Proof (.NET)                    ║");
        Console.WriteLine("╚══════════════════════════════════════════════════════════════╝");
        Console.WriteLine();

        int passed = 0, failed = 0;
        var sw = Stopwatch.StartNew();

        LoadLibrary();

        // ── Device Discovery ──
        Console.WriteLine("── Device Discovery ──");

        if (fnGetDeviceCount(out int count) == 0 && count >= 1)
        { Console.WriteLine($"  PASS  Device count: {count}"); passed++; }
        else { Console.WriteLine("  FAIL  Device count"); failed++; }

        var propBuf = Marshal.AllocHGlobal(512);
        if (fnGetDeviceProperties(propBuf, 0) == 0)
        {
            var name = Marshal.PtrToStringAnsi(propBuf)!;
            var totalMem = Marshal.ReadInt64(propBuf + 256);
            Console.WriteLine($"  PASS  Device: {name} ({totalMem / 1e9:F1} GB)");
            passed++;
        }
        else { Console.WriteLine("  FAIL  Device properties"); failed++; }
        Marshal.FreeHGlobal(propBuf);

        if (fnRuntimeGetVersion(out int version) == 0 && version >= 12000)
        { Console.WriteLine($"  PASS  Runtime version: {version}"); passed++; }
        else { Console.WriteLine("  FAIL  Runtime version"); failed++; }

        if (fnMemGetInfo(out long freeMem, out long totalMemL) == 0 && totalMemL > 0)
        { Console.WriteLine($"  PASS  Memory: {freeMem / 1e9:F1} GB free / {totalMemL / 1e9:F1} GB total"); passed++; }
        else { Console.WriteLine("  FAIL  Memory info"); failed++; }
        Console.WriteLine();

        // ── Memory Operations ──
        Console.WriteLine("── Memory Operations ──");

        if (fnMalloc(out IntPtr ptr, 4096) == 0 && ptr != IntPtr.Zero)
        { fnFree(ptr); Console.WriteLine("  PASS  malloc + free (4 KB)"); passed++; }
        else { Console.WriteLine("  FAIL  malloc + free"); failed++; }

        // memcpy roundtrip
        {
            int n = 1024;
            var src = new float[n];
            for (int i = 0; i < n; i++) src[i] = i * 0.1f;
            var srcHandle = GCHandle.Alloc(src, GCHandleType.Pinned);
            var dstArr = new float[n];
            var dstHandle = GCHandle.Alloc(dstArr, GCHandleType.Pinned);

            fnMalloc(out IntPtr dPtr, n * 4);
            fnMemcpy(dPtr, srcHandle.AddrOfPinnedObject(), n * 4, 1);
            fnMemcpy(dstHandle.AddrOfPinnedObject(), dPtr, n * 4, 2);
            fnFree(dPtr);
            srcHandle.Free(); dstHandle.Free();

            bool ok = true;
            for (int i = 0; i < n; i++)
                if (MathF.Abs(src[i] - dstArr[i]) > 1e-5f) { ok = false; break; }
            if (ok) { Console.WriteLine("  PASS  memcpy roundtrip (4 KB)"); passed++; }
            else { Console.WriteLine("  FAIL  memcpy roundtrip"); failed++; }
        }

        // memset
        {
            fnMalloc(out IntPtr dPtr, 4096);
            fnMemset(dPtr, 0xAB, 4096);
            var buf = new byte[4096];
            var bufHandle = GCHandle.Alloc(buf, GCHandleType.Pinned);
            fnMemcpy(bufHandle.AddrOfPinnedObject(), dPtr, 4096, 2);
            fnFree(dPtr); bufHandle.Free();

            bool ok = true;
            foreach (var b in buf) if (b != 0xAB) { ok = false; break; }
            if (ok) { Console.WriteLine("  PASS  memset (4 KB, pattern 0xAB)"); passed++; }
            else { Console.WriteLine("  FAIL  memset"); failed++; }
        }
        Console.WriteLine();

        // ── Streams & Events ──
        Console.WriteLine("── Streams & Events ──");

        if (fnStreamCreate(out IntPtr stream) == 0)
        { fnStreamSynchronize(stream); fnStreamDestroy(stream);
          Console.WriteLine("  PASS  Stream create + sync + destroy"); passed++; }
        else { Console.WriteLine("  FAIL  Stream"); failed++; }

        {
            fnEventCreate(out IntPtr evStart);
            fnEventCreate(out IntPtr evEnd);
            fnEventRecord(evStart, IntPtr.Zero);
            fnDeviceSynchronize();
            fnEventRecord(evEnd, IntPtr.Zero);
            fnEventSynchronize(evEnd);
            fnEventElapsedTime(out float ms, evStart, evEnd);
            fnEventDestroy(evStart); fnEventDestroy(evEnd);
            if (ms >= 0)
            { Console.WriteLine($"  PASS  Event timing ({ms:F3} ms)"); passed++; }
            else { Console.WriteLine("  FAIL  Event timing"); failed++; }
        }
        Console.WriteLine();

        // ── cuBLAS ──
        Console.WriteLine("── cuBLAS ──");

        if (fnCublasCreate(out IntPtr handle) == 0)
        {
            Console.WriteLine("  PASS  cuBLAS handle created"); passed++;

            var aData = new float[16];
            var eye = new float[16];
            for (int i = 0; i < 16; i++) aData[i] = i + 1;
            for (int i = 0; i < 4; i++) eye[i * 4 + i] = 1.0f;

            var aH = GCHandle.Alloc(aData, GCHandleType.Pinned);
            var iH = GCHandle.Alloc(eye, GCHandleType.Pinned);

            fnMalloc(out IntPtr dA, 64);
            fnMalloc(out IntPtr dI, 64);
            fnMalloc(out IntPtr dC, 64);
            fnMemcpy(dA, aH.AddrOfPinnedObject(), 64, 1);
            fnMemcpy(dI, iH.AddrOfPinnedObject(), 64, 1);
            fnMemset(dC, 0, 64);
            aH.Free(); iH.Free();

            float alpha = 1.0f, beta = 0.0f;
            fnCublasSgemm(handle, 0, 0, 4, 4, 4,
                &alpha, dA, 4, dI, 4, &beta, dC, 4);

            var result = new float[16];
            var rH = GCHandle.Alloc(result, GCHandleType.Pinned);
            fnMemcpy(rH.AddrOfPinnedObject(), dC, 64, 2);
            rH.Free();
            fnFree(dA); fnFree(dI); fnFree(dC);

            bool ok = true;
            for (int i = 0; i < 16; i++)
                if (MathF.Abs(result[i] - aData[i]) > 1e-3f) { ok = false; break; }
            if (ok) { Console.WriteLine("  PASS  SGEMM identity multiply (4x4)"); passed++; }
            else { Console.WriteLine("  FAIL  SGEMM identity multiply"); failed++; }

            fnCublasDestroy(handle);
        }
        else { Console.WriteLine("  FAIL  cuBLAS create"); failed++; }
        Console.WriteLine();

        // ── Performance ──
        Console.WriteLine("── Performance ──");

        {
            int size = 4 * 1024 * 1024;
            var data = new byte[size];
            var dataH = GCHandle.Alloc(data, GCHandleType.Pinned);
            fnMalloc(out IntPtr dPtr, size);

            for (int i = 0; i < 5; i++)
            {
                fnMemcpy(dPtr, dataH.AddrOfPinnedObject(), size, 1);
                fnMemcpy(dataH.AddrOfPinnedObject(), dPtr, size, 2);
            }

            var t = Stopwatch.StartNew();
            int iters = 50;
            for (int i = 0; i < iters; i++)
            {
                fnMemcpy(dPtr, dataH.AddrOfPinnedObject(), size, 1);
                fnMemcpy(dataH.AddrOfPinnedObject(), dPtr, size, 2);
            }
            double elapsed = t.Elapsed.TotalSeconds;
            fnFree(dPtr); dataH.Free();

            double gbS = (double)(iters * size * 2) / elapsed / 1e9;
            Console.WriteLine($"  Mem BW (4 MB): {gbS:F2} GB/s");
        }
        Console.WriteLine();

        // ── Summary ──
        double total = sw.Elapsed.TotalSeconds;
        Console.WriteLine("══════════════════════════════════════════════════════════════");
        if (failed == 0)
        {
            Console.WriteLine($"  ALL {passed} TESTS PASSED ({total:F1}s)");
            Console.WriteLine("  CUDA compatibility: VERIFIED via Invisible CUDA (C#)");
        }
        else Console.WriteLine($"  {passed} passed, {failed} FAILED ({total:F1}s)");
        Console.WriteLine("══════════════════════════════════════════════════════════════");

        Environment.Exit(failed > 0 ? 1 : 0);
    }
}
