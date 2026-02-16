# Raw Proof Results

This directory contains the complete, unmodified output from running the Invisible CUDA proof suite on 33 EC2 instance types. Each file captures the full stdout of three proof binaries (compatibility, limits, coverage) executed on a single instance.

## File Format

Each result file contains three sections, executed sequentially:

### 1. Header

```
===================================================
  Instance: t4g-micro
  Type:     t4g.micro
  Arch:     arm64
  GPU:      none
  Price:    $0.008/hr
  Backend:  CPU
  Proves:   Cheapest ARM
  Mode:     standard
===================================================
```

### 2. Compatibility Proof Output

17 tests across 4 categories:

- **Correctness** (9 tests): Vector Add, Vector Mul, FMA, SAXPY, Memory round-trip, Memset, D2D copy, Multi alloc+free, Multi-kernel dispatch
- **BLAS** (5 tests): SGEMM at 4 sizes, SGEMM with alpha/beta, SAXPY, SDOT, SNRM2
- **Stress** (3 tests): Large allocation (1 GB), Rapid kernels (10K dispatches), 500 concurrent buffers

Plus 14 performance benchmarks (informational, not pass/fail).

Followed by a `--- JSON_START ---` / `--- JSON_END ---` block containing structured results in JSON format.

### 3. Scientific Limits Proof Output

24 tests across 5 categories:

- **Hard Limits** (5): CUDA-exclusive features that cannot work without NVIDIA hardware (tensor cores, texture sampling, surfaces, async copy, mbarrier)
- **Semantic Gaps** (6): Features that work but behave differently (warp shuffle, warp vote, warp identity, barrier+shared memory, control flow, 2D/3D grids)
- **Precision Boundaries** (5): Numerical accuracy differences (transcendentals, FMA, FP16, special floats, accumulated error)
- **Performance Scaling** (4): Throughput measurements (parallelism scaling, memory bandwidth, kernel launch overhead, BLAS SGEMM)
- **Edge Cases** (4): Extreme conditions (max grid size, register pressure, bitwise ops, allocation limits)

Results use three statuses:
- `supported` — Full CUDA compatibility
- `degraded` — Works with documented behavioral differences
- `unsupported` — Requires NVIDIA hardware (explicitly CUDA-exclusive)

### 4. Library Coverage Proof Output

47 CUDA library modules tested across 4 tiers:

- **Tier 1** (22): Core CUDA libraries (cuBLAS, cuDNN, cuFFT, cuSPARSE, cuRAND, cuSOLVER, cuTENSOR, NCCL, NVML, NVRTC, NVENC, NVDEC, nvJPEG, NPP, TensorRT, NVTX, cuFile, etc.)
- **Tier 2** (10): Specialized rendering & vision (nvdiffrast, flash_attn, gaussian_rast, pytorch3d, faiss_gpu, etc.)
- **Tier 3** (4): Scientific computing (molecular_dynamics, gpu_crypto, rapids, audio_ops)
- **Tier 4** (10): Advanced kernels & research (CUTLASS, Triton, apex, tiny_cuda_nn, xformers, etc.)
- **HIP/ROCm** (1): AMD compatibility layer

## JSON Schema (Compatibility Proof)

```json
{
  "format": "invisible-cuda-proof-v3",
  "backend": "CPU",
  "device": "CPU (N threads)",
  "os": "linux",
  "arch": "x86_64 | aarch64",
  "compute_units": 4,
  "memory_bytes": 0,
  "max_threads_per_block": 1024,
  "has_shared_memory": true,
  "has_atomics": true,
  "hw_cpu_model": "Intel(R) Xeon(R) Platinum 8488C",
  "hw_cores_logical": 8,
  "hw_cores_physical": 4,
  "hw_ram_bytes": 16483618816,
  "hw_mem_type": "DDR4",
  "hw_mem_speed_mt": 2933,
  "hw_mem_channels": 8,
  "hw_mem_channels_estimated": true,
  "hw_mem_bandwidth_gbps": 187.7,
  "hw_numa_nodes": 1,
  "hw_cache_l1d": "64 KiB",
  "hw_cache_l2": "2 MiB",
  "hw_cache_l3": "35.8 MiB",
  "hw_isa": "sse4_2 avx avx2 avx512f",
  "hw_kernel": "6.1.161-183.298.amzn2023.x86_64",
  "tests": [
    {"name": "...", "passed": true, "ms": 1.0, "error": null}
  ],
  "benchmarks": [
    {"name": "...", "value": 10.0, "unit": "GB/s", "normalized": 1.36, "normalized_unit": "GB/s @DDR4-3200x1"}
  ]
}
```

## How to Use This Data

Parse the JSON blocks programmatically:

```bash
# Extract JSON from a result file
sed -n '/--- JSON_START ---/,/--- JSON_END ---/p' results/raw/t3-micro.txt | grep -v '---'

# Check if all tests passed
grep "RESULT:" results/raw/*.txt
```
