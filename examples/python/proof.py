#!/usr/bin/env python3
"""Invisible CUDA — Python Proof

Demonstrates CUDA compatibility on any hardware using the Invisible CUDA Python SDK.

Prerequisites:
    pip install invisible-cuda numpy

Run:
    python proof.py
"""

import sys
import time

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║         INVISIBLE CUDA — Python SDK Proof                  ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    passed = 0
    failed = 0
    start = time.time()

    # ── 1. Device Discovery ──────────────────────────────────────────
    print("── Device Discovery ──")

    import invisible_cuda as cuda
    from invisible_cuda.runtime import (
        get_device_count, get_device, set_device, get_device_properties,
        device_synchronize, runtime_get_version, malloc, free,
        memcpy_htod, memcpy_dtoh, memset, mem_get_info,
        Stream, Event,
    )
    from invisible_cuda.cublas import CublasHandle
    from invisible_cuda._types import CublasOperation, MemcpyKind
    from invisible_cuda.info import device_info, print_device_info
    from invisible_cuda.array import GpuArray
    import ctypes

    try:
        count = get_device_count()
        assert count >= 1
        print(f"  PASS  Device count: {count}")
        passed += 1
    except Exception as e:
        print(f"  FAIL  Device count: {e}")
        failed += 1

    try:
        info = device_info(0)
        mem_gb = info["total_memory"] / 1e9
        print(f"  PASS  Device: {info['name']} ({mem_gb:.1f} GB, CUDA {info['cuda_version']})")
        passed += 1
    except Exception as e:
        print(f"  FAIL  Device info: {e}")
        failed += 1

    try:
        version = runtime_get_version()
        assert version >= 12000
        print(f"  PASS  Runtime version: {version}")
        passed += 1
    except Exception as e:
        print(f"  FAIL  Runtime version: {e}")
        failed += 1

    try:
        free_mem, total_mem = mem_get_info()
        assert total_mem > 0 and free_mem > 0 and free_mem <= total_mem
        print(f"  PASS  Memory: {free_mem/1e9:.1f} GB free / {total_mem/1e9:.1f} GB total")
        passed += 1
    except Exception as e:
        print(f"  FAIL  Memory info: {e}")
        failed += 1
    print()

    # ── 2. Memory Operations ─────────────────────────────────────────
    print("── Memory Operations ──")

    # malloc + free
    try:
        ptr = malloc(4096)
        assert ptr != 0
        free(ptr)
        print("  PASS  malloc + free (4 KB)")
        passed += 1
    except Exception as e:
        print(f"  FAIL  malloc + free: {e}")
        failed += 1

    # memcpy roundtrip
    try:
        import struct
        n = 1024
        src = [i * 0.1 for i in range(n)]
        src_bytes = struct.pack(f"{n}f", *src)
        src_buf = (ctypes.c_char * len(src_bytes))(*src_bytes)

        ptr = malloc(n * 4)
        memcpy_htod(ptr, src_buf, n * 4)
        dst_buf = (ctypes.c_char * (n * 4))()
        memcpy_dtoh(dst_buf, ptr, n * 4)
        free(ptr)

        dst = struct.unpack(f"{n}f", bytes(dst_buf))
        for i in range(n):
            assert abs(src[i] - dst[i]) < 1e-5, f"mismatch at {i}"
        print("  PASS  memcpy roundtrip (4 KB)")
        passed += 1
    except Exception as e:
        print(f"  FAIL  memcpy roundtrip: {e}")
        failed += 1

    # memset
    try:
        ptr = malloc(4096)
        memset(ptr, 0xAB, 4096)
        buf = (ctypes.c_char * 4096)()
        memcpy_dtoh(buf, ptr, 4096)
        free(ptr)
        for b in bytes(buf):
            assert b == 0xAB
        print("  PASS  memset (4 KB, pattern 0xAB)")
        passed += 1
    except Exception as e:
        print(f"  FAIL  memset: {e}")
        failed += 1
    print()

    # ── 3. Streams & Events ──────────────────────────────────────────
    print("── Streams & Events ──")

    try:
        with Stream() as s:
            s.synchronize()
            assert s.query()
        print("  PASS  Stream create + sync + query")
        passed += 1
    except Exception as e:
        print(f"  FAIL  Stream: {e}")
        failed += 1

    try:
        with Event() as start_ev, Event() as end_ev:
            start_ev.record()
            device_synchronize()
            end_ev.record()
            end_ev.synchronize()
            ms = Event.elapsed_time(start_ev, end_ev)
            assert ms >= 0.0
        print(f"  PASS  Event timing ({ms:.3f} ms)")
        passed += 1
    except Exception as e:
        print(f"  FAIL  Event: {e}")
        failed += 1
    print()

    # ── 4. cuBLAS ────────────────────────────────────────────────────
    print("── cuBLAS ──")

    try:
        with CublasHandle() as handle:
            import struct
            n = 4
            # A = [[1..16]], I = identity
            a_data = list(range(1, 17))
            eye = [0.0] * 16
            for i in range(4):
                eye[i * 4 + i] = 1.0

            a_bytes = struct.pack("16f", *[float(x) for x in a_data])
            i_bytes = struct.pack("16f", *eye)
            z_bytes = struct.pack("16f", *([0.0] * 16))

            d_a = malloc(64)
            d_i = malloc(64)
            d_c = malloc(64)
            memcpy_htod(d_a, (ctypes.c_char * 64)(*a_bytes), 64)
            memcpy_htod(d_i, (ctypes.c_char * 64)(*i_bytes), 64)
            memset(d_c, 0, 64)

            handle.sgemm(
                CublasOperation.N, CublasOperation.N,
                n, n, n, 1.0,
                d_a, n,
                d_i, n,
                0.0, d_c, n,
            )

            result_buf = (ctypes.c_char * 64)()
            memcpy_dtoh(result_buf, d_c, 64)
            result = struct.unpack("16f", bytes(result_buf))

            ok = all(abs(result[i] - float(a_data[i])) < 1e-3 for i in range(16))
            free(d_a); free(d_i); free(d_c)

            if ok:
                print("  PASS  SGEMM identity multiply (4x4)")
                passed += 1
            else:
                print(f"  FAIL  SGEMM identity multiply: {result}")
                failed += 1
    except Exception as e:
        print(f"  FAIL  cuBLAS: {e}")
        failed += 1
    print()

    # ── 5. GpuArray (high-level API) ─────────────────────────────────
    print("── GpuArray ──")

    try:
        import numpy as np

        # Roundtrip
        arr = GpuArray.from_numpy(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape(2, 2))
        result = arr.to_numpy()
        assert np.allclose(result.flatten(), [1.0, 2.0, 3.0, 4.0])
        print("  PASS  from_numpy + to_numpy roundtrip")
        passed += 1

        # Zeros / ones
        z = GpuArray.zeros((100,))
        assert np.all(z.to_numpy() == 0.0)
        o = GpuArray.ones((100,))
        assert np.all(o.to_numpy() == 1.0)
        print("  PASS  zeros + ones")
        passed += 1

        # Arithmetic
        a = GpuArray.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        b = GpuArray.from_numpy(np.array([4.0, 5.0, 6.0], dtype=np.float32))
        assert np.allclose((a + b).to_numpy(), [5.0, 7.0, 9.0])
        assert np.allclose((a - b).to_numpy(), [-3.0, -3.0, -3.0])
        assert np.allclose((a * b).to_numpy(), [4.0, 10.0, 18.0])
        assert np.allclose((a * 10.0).to_numpy(), [10.0, 20.0, 30.0])
        print("  PASS  element-wise arithmetic (add, sub, mul, mul_scalar)")
        passed += 1

        # Matmul
        dim = 64
        a = GpuArray.from_numpy(np.ones((dim, dim), dtype=np.float32))
        b = GpuArray.from_numpy(np.ones((dim, dim), dtype=np.float32))
        c = a @ b
        result = c.to_numpy()
        assert abs(result[0, 0] - float(dim)) < 0.1
        print(f"  PASS  matmul {dim}x{dim} (result[0,0] = {result[0,0]:.1f})")
        passed += 1

        # Transpose
        a = GpuArray.from_numpy(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
        t = a.T
        assert t.shape == (3, 2)
        assert np.allclose(t.to_numpy().flatten(), [1, 4, 2, 5, 3, 6])
        print("  PASS  transpose (2x3 -> 3x2)")
        passed += 1

    except ImportError:
        print("  SKIP  numpy not available — skipping GpuArray tests")
    except Exception as e:
        print(f"  FAIL  GpuArray: {e}")
        failed += 1
    print()

    # ── 6. Performance Benchmark ─────────────────────────────────────
    print("── Performance ──")

    try:
        import numpy as np

        # Memory bandwidth
        size = 4 * 1024 * 1024
        data = (ctypes.c_char * size)()
        ptr = malloc(size)
        for _ in range(5):  # warmup
            memcpy_htod(ptr, data, size)
            dst = (ctypes.c_char * size)()
            memcpy_dtoh(dst, ptr, size)

        t = time.time()
        iters = 50
        for _ in range(iters):
            memcpy_htod(ptr, data, size)
            dst = (ctypes.c_char * size)()
            memcpy_dtoh(dst, ptr, size)
        elapsed = time.time() - t
        free(ptr)
        gb_s = (iters * size * 2) / elapsed / 1e9
        print(f"  Mem BW (4 MB): {gb_s:.2f} GB/s")

        # SGEMM throughput
        dim = 512
        a = GpuArray.from_numpy(np.full((dim, dim), 0.5, dtype=np.float32))
        b = GpuArray.from_numpy(np.full((dim, dim), 0.5, dtype=np.float32))
        _ = a @ b  # warmup
        t = time.time()
        iters = 10
        for _ in range(iters):
            _ = a @ b
        elapsed = time.time() - t
        flops = 2.0 * dim**3 * iters
        gflops = flops / elapsed / 1e9
        print(f"  SGEMM {dim}x{dim}: {gflops:.2f} GFLOPS")
    except Exception as e:
        print(f"  Benchmark error: {e}")
    print()

    # ── Summary ──────────────────────────────────────────────────────
    total_time = time.time() - start
    print("══════════════════════════════════════════════════════════════")
    if failed == 0:
        print(f"  ALL {passed} TESTS PASSED ({total_time:.1f}s)")
        print("  CUDA compatibility: VERIFIED via Invisible CUDA Python SDK")
    else:
        print(f"  {passed} passed, {failed} FAILED ({total_time:.1f}s)")
    print("══════════════════════════════════════════════════════════════")

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
