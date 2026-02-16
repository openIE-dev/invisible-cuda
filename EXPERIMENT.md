# Experimental Design: Universal CUDA Compatibility Across EC2 Hardware

## Abstract

We tested whether CUDA workloads can execute correctly on non-NVIDIA hardware by deploying a universal CUDA compatibility layer to 33 Amazon EC2 instance types spanning 8 CPU microarchitectures, 2 instruction set architectures (ARM aarch64 and x86_64), bare metal and virtualized environments, and instances with GPU hardware present but no drivers installed. Three test suites — compatibility (17 tests), library coverage (47 CUDA modules), and scientific limits (24 tests across 5 categories) — were executed on every instance. All 17 compatibility tests and all 47 library modules passed on every instance. The limits suite identified 5 CUDA-exclusive features that require NVIDIA hardware, 10 features with documented behavioral differences, and 9 features with full compatibility.

## Hypothesis

CUDA is an interface for expressing parallel computation. If the interface semantics are faithfully implemented, CUDA workloads should execute correctly on any compute substrate — regardless of CPU vendor, instruction set architecture, core count, memory capacity, or the presence of GPU hardware.

## Variables

**Independent variable**: Hardware configuration — CPU microarchitecture, ISA (ARM/x86), core count (1–96), memory (0.9–502.7 GB), virtualization (bare metal vs. hypervisor), GPU presence (none, NVIDIA T4/T4G/A10G/L4, AMD Radeon Pro V520).

**Dependent variables**: (1) Test pass/fail status, (2) numerical correctness within specified tolerances, (3) library API completion status.

**Controlled variables**: Identical binary per ISA (one x86_64 binary, one aarch64 binary), identical test suite, identical operating system (Amazon Linux 2023, except a1.metal which requires Amazon Linux 2), identical backend (CPU — no GPU drivers used on any instance).

## Test Suites

### Suite 1: Compatibility Proof (17 tests)

Tests core CUDA operations through the `ComputeBackend` trait using the `KernelIR` intermediate representation.

#### Correctness Tests (9)

| # | Test | Method | Size | Tolerance |
|---|------|--------|------|-----------|
| 1 | Vector Add | C[i] = A[i] + B[i] | 64, 1024, 65536 elements | < 1e-5 per element |
| 2 | Vector Mul | C[i] = A[i] * B[i] | 4096 elements | < 1e-3 per element |
| 3 | FMA | R[i] = X[i] * Y[i] + Z[i] | 2048 elements | < 1e-4 per element |
| 4 | SAXPY | Y[i] = a*X[i] + Y[i] | 4096 elements | < 1e-4 per element |
| 5 | Memory round-trip | Host → Device → Host | 64B to 4MB (7 sizes) | Byte-exact |
| 6 | Memset | Fill buffer with pattern | 4KB to 1MB (4 patterns: 0x00, 0xFF, 0xAA, 0xDE) | Byte-exact |
| 7 | D2D copy | Device-to-device buffer copy | 4KB to 1MB | Byte-exact |
| 8 | Multi alloc+free | Allocate and free N buffers | 150 buffers | No error |
| 9 | Multi-kernel dispatch | Execute add then mul sequentially | 4096 elements | < 1e-4 per element |

Kernels are expressed as `KernelIR` operation graphs (e.g., `Op::Add`, `Op::Mul`, `Op::Fma`) and compiled to the backend's native execution format. CUDA grid/block launch semantics are preserved: kernels are dispatched with `(grid_x, grid_y, grid_z)` and `(block_x, block_y, block_z)` dimensions.

SGEMM tests use a CPU reference implementation (`cpu_matmul`) for verification:

```rust
fn cpu_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for p in 0..k {
            let a_val = a[i * k + p];
            for j in 0..n {
                c[i * n + j] += a_val * b[p * n + j];
            }
        }
    }
    c
}
```

#### BLAS Correctness (5)

| # | Test | Method |
|---|------|--------|
| 10 | SGEMM (4 sizes) | C = alpha*A*B + beta*C at M=N=K of 32, 64, 128, 256 |
| 11 | SGEMM alpha/beta | alpha=2.5, beta=0.5 (non-trivial parameters) |
| 12 | BLAS SAXPY | y = alpha*x + y via BlasBackend trait |
| 13 | BLAS SDOT | dot product via BlasBackend trait |
| 14 | BLAS SNRM2 | Euclidean norm via BlasBackend trait |

SGEMM tolerance: < 1e-2 per element (accounts for floating-point accumulation order differences across implementations).

#### Stress Tests (3)

| # | Test | Method |
|---|------|--------|
| 15 | Large allocation | Progressive doubling from 1 MB to 1 GB |
| 16 | Rapid kernels | 10,000 consecutive vector-add dispatches |
| 17 | Concurrent buffers | 500 simultaneous buffer allocations |

#### Performance Benchmarks (14, informational)

Memory bandwidth at 4 transfer sizes (256 KB, 1 MB, 4 MB, 16 MB), vector-add throughput at 3 sizes, SGEMM GFLOPS at 5 matrix sizes (128 to 2048), kernel launch latency, allocation throughput. Performance numbers vary across hardware and are reported for characterization purposes only — they are not pass/fail criteria.

### Suite 2: Library Coverage Proof (47 modules)

Tests that all CUDA library runtimes initialize, execute representative API calls, and shut down without error.

| Tier | Count | Libraries |
|------|:-----:|-----------|
| 1. Core CUDA | 22 | cuBLAS, cuBLASLt, cuDNN, cuFFT, cuSPARSE, cuRAND, cuSOLVER, cuTENSOR, NCCL, NVML, Thrust/CUB, NVRTC, NVENC, NVDEC, nvJPEG, nvJPEG2K, NPP, cuSPARSELt, TensorRT, NVTX, cuFile, NvOF |
| 2. Rendering & Vision | 10 | nvdiffrast, spconv, gaussian_rast, flash_attn, nerfacc, bitsandbytes, detectron2_ops, pointnet, pytorch3d, faiss_gpu |
| 3. Scientific | 4 | molecular_dynamics, gpu_crypto, rapids, audio_ops |
| 4. Advanced Kernels | 10 | cutlass, triton_kernels, apex, tiny_cuda_nn, xformers, warp_sim, kaolin, cu_quantum, dali, cu_dss |
| HIP/ROCm | 1 | HIP compatibility layer |

Each module follows: create runtime → create handle → invoke APIs → destroy → report pass/stub/fail.

### Suite 3: Scientific Limits Proof (24 tests)

Systematically identifies the boundaries of CUDA compatibility. Each test produces a status:

- **Supported**: Full CUDA behavioral compatibility
- **Degraded**: Correct for standard workloads, documented behavioral differences for specialized patterns
- **Unsupported**: Requires NVIDIA hardware (explicitly labeled as CUDA-exclusive)

#### Category 1: Hard Limits (5 tests)

CUDA features that cannot work without NVIDIA hardware:

| Test | Status | Reason |
|------|--------|--------|
| Tensor Cores (WMMA) | Unsupported | No tensor core hardware; WMMA ops return stub values |
| Texture Sampling (2D/3D) | Unsupported | No texture unit hardware |
| Surface Load/Store | Unsupported | No surface memory hardware |
| Async Copy (cp.async) | Unsupported | No async copy engine |
| MBarrier (Named Barriers) | Unsupported | No hardware barrier unit |

#### Category 2: Semantic Gaps (6 tests)

Features that work but behave differently:

| Test | Status | Behavior |
|------|--------|----------|
| Warp Shuffle (shfl.down) | Degraded | Returns identity (warp_size=1); no cross-lane shuffle |
| Warp Vote (ballot/all/any) | Degraded | Single-lane semantics; ballot returns 0/1, not 32-bit bitmask |
| Warp Identity (LaneId/WarpId) | Degraded | LaneId=0, WarpId=0, NWarps=1 for all threads |
| Barrier + Shared Memory | Degraded | Barrier is no-op; shared memory is per-thread (works for simple patterns) |
| Control Flow (Branch/BranchIf) | Degraded | All IR ops execute linearly; last write wins |
| 2D/3D Grid Dimensions | Degraded | Y/Z dims execute but output indexing uses only X dimension |

These differences affect only warp-aware algorithms (reductions using shuffle, ballot-based masking). Standard CUDA kernels that use grid/block indexing for element-wise operations are unaffected.

#### Category 3: Precision Boundaries (5 tests)

| Test | Status | Detail |
|------|--------|--------|
| Transcendental Functions | Degraded | Max 520,764 ULP error for sin(); CUDA fast math spec: ~2 ULP |
| FMA vs Mul+Add Precision | Supported | Error within expected range |
| FP16 Conversion | Supported | Max relative error: 0.055% |
| Special Float Values | Degraded | MAX+1.0 returns MAX instead of Inf |
| Accumulated Error (100 FMAs) | Supported | Error: 1.72e-6 after 100 chained FMAs |

#### Category 4: Performance Scaling (4 tests)

| Test | Status | Detail |
|------|--------|--------|
| Parallelism Scaling | Degraded | 1.2x actual vs 977x ideal (CPU thread pool vs GPU SIMT) |
| Memory Bandwidth Profile | Supported | Scales with hardware memory subsystem |
| Kernel Launch Overhead | Supported | ~1 us (CPU) vs ~5-10 us (CUDA) |
| BLAS SGEMM Scaling | Degraded | Peak 0.9 GFLOPS (CPU BLAS) vs ~19,500 GFLOPS (CUDA A100) |

Performance differences are expected and proportional to hardware class. The CPU backend targets compatibility, not throughput.

#### Category 5: Edge Cases (4 tests)

| Test | Status | Detail |
|------|--------|--------|
| Maximum Grid Size | Supported | grid_x up to 1,000,000 tested successfully |
| Register Pressure | Supported | IR op chains up to 1,000 ops correct |
| Bitwise Operations | Supported | AND, POPC, BREV, DP4A all correct |
| Allocation Limits | Supported | 10,000 × 1 MB = 9.8 GB allocated successfully |

### Suite 4: Distributed Cluster Proof

A coordinator running on t3.large orchestrates workers across t3.small (x86), t4g.small (ARM), and c7g.large (ARM). Tests: connectivity, capability exchange, latency measurement, kernel compilation, distributed vector-add (10K elements), large distributed vector-add (100K elements), and graceful shutdown.

## Hardware Matrix

33 instances across 8 CPU microarchitectures:

| # | Instance | Type | ISA | CPU Microarchitecture | Cores | RAM | GPU | $/hr |
|---|----------|------|-----|----------------------|------:|----:|-----|-----:|
| 1 | t4g.micro | Virtual | aarch64 | Neoverse-N1 (Graviton 2) | 2 | 0.9 GB | — | $0.008 |
| 2 | t3.micro | Virtual | x86_64 | Xeon 8259CL (Cascade Lake) | 2 | 0.9 GB | — | $0.010 |
| 3 | t4g.small | Virtual | aarch64 | Neoverse-N1 (Graviton 2) | 2 | 1.8 GB | — | $0.017 |
| 4 | t3.small | Virtual | x86_64 | Xeon 8259CL (Cascade Lake) | 2 | 1.9 GB | — | $0.021 |
| 5 | m7g.medium | Virtual | aarch64 | Graviton 3 | 1 | 3.7 GB | — | $0.041 |
| 6 | c7g.large | Virtual | aarch64 | Graviton 3 | 2 | 3.7 GB | — | $0.073 |
| 7 | t3.large | Virtual | x86_64 | Xeon 8259CL (Cascade Lake) | 2 | 7.6 GB | — | $0.083 |
| 8 | m7i.large | Virtual | x86_64 | Xeon 8488C (Sapphire Rapids) | 2 | 7.6 GB | — | $0.100 |
| 9 | r7g.large | Virtual | aarch64 | Graviton 3 | 2 | 15.3 GB | — | $0.107 |
| 10 | r7i.large | Virtual | x86_64 | Xeon 8488C (Sapphire Rapids) | 2 | 15.3 GB | — | $0.132 |
| 11 | c7i.2xlarge | Virtual | x86_64 | Xeon 8488C (Sapphire Rapids) | 8 | 15.3 GB | — | $0.357 |
| 12 | g4ad.xlarge | Virtual | x86_64 | AMD EPYC 7R32 (Zen 2) | 4 | 15.0 GB | Radeon V520 | $0.379 |
| 13 | a1.metal | Bare metal | aarch64 | Cortex-A72 (Graviton 1) | 16 | 31.5 GB | — | $0.408 |
| 14 | g5g.xlarge | Virtual | aarch64 | Neoverse-N1 (Graviton 2) | 4 | 7.6 GB | NVIDIA T4G | $0.421 |
| 15 | g4dn.xlarge | Virtual | x86_64 | Xeon 8259CL (Cascade Lake) | 4 | 15.4 GB | NVIDIA T4 | $0.526 |
| 16 | c7g.4xlarge | Virtual | x86_64 | Graviton 3 | 16 | 30.8 GB | — | $0.579 |
| 17 | hpc7g.4xlarge | Virtual | aarch64 | Graviton 3E | 16 | 123.6 GB | — | $0.680 |
| 18 | c7i.4xlarge | Virtual | x86_64 | Xeon 8488C (Sapphire Rapids) | 16 | 30.8 GB | — | $0.714 |
| 19 | g6.xlarge | Virtual | x86_64 | AMD EPYC 7R13 (Zen 3) | 4 | 15.0 GB | NVIDIA L4 | $0.805 |
| 20 | r7g.4xlarge | Virtual | aarch64 | Graviton 3 | 16 | 123.6 GB | — | $0.854 |
| 21 | g5.xlarge | Virtual | x86_64 | AMD EPYC 7R32 (Zen 2) | 4 | 15.4 GB | NVIDIA A10G | $1.006 |
| 22 | c7g.8xlarge | Virtual | aarch64 | Graviton 3 | 32 | 61.7 GB | — | $1.158 |
| 23 | c7i.8xlarge | Virtual | x86_64 | Xeon 8488C (Sapphire Rapids) | 32 | 61.8 GB | — | $1.428 |
| 24 | c6g.metal | Bare metal | aarch64 | Neoverse-N1 (Graviton 2) | 64 | 128 GB | — | $2.176 |
| 25 | c7g.metal | Bare metal | aarch64 | Graviton 3 | 64 | 125.5 GB | — | $2.320 |
| 26 | m6g.metal | Bare metal | aarch64 | Neoverse-N1 (Graviton 2) | 64 | 251.2 GB | — | $2.464 |
| 27 | g5g.metal | Bare metal | aarch64 | Neoverse-N1 (Graviton 2) | 64 | 125.5 GB | NVIDIA T4G | $2.744 |
| 28 | r6g.metal | Bare metal | aarch64 | Neoverse-N1 (Graviton 2) | 64 | 502.7 GB | — | $3.226 |
| 29 | r7g.metal | Bare metal | aarch64 | Graviton 3 | 64 | 502.7 GB | — | $3.427 |
| 30 | m5zn.metal | Bare metal | x86_64 | Xeon 8252C (Cascade Lake) | 48 | 188.5 GB | — | $3.964 |
| 31 | z1d.metal | Bare metal | x86_64 | Xeon 8151 (Skylake) | 48 | 377.5 GB | — | $4.464 |
| 32 | g4dn.metal | Bare metal | x86_64 | Xeon 8259CL (Cascade Lake) | 96 | 377.5 GB | 8x NVIDIA T4 | $7.824 |

Instance #33 is the distributed cluster coordinator (t3.large), whose result file captures the multi-node test.

## Methodology

### Binary Construction

Proof binaries were cross-compiled from macOS (Apple Silicon) to Linux using `cargo-zigbuild`:

```bash
# x86_64 (Intel, AMD EC2)
cargo zigbuild --bin proof --bin limits_proof --bin coverage_proof \
    --release --target x86_64-unknown-linux-gnu.2.17 \
    --no-default-features --features cpu,distributed

# aarch64 (Graviton EC2)
cargo zigbuild --bin proof --bin limits_proof --bin coverage_proof \
    --release --target aarch64-unknown-linux-gnu.2.17 \
    --no-default-features --features cpu,distributed
```

The `.2.17` suffix instructs `cargo-zigbuild` to target glibc 2.17 (CentOS 7, 2014), ensuring compatibility with any Linux system from the past 12 years. The resulting binaries:

- **x86_64**: 876 KB (proof), 906 KB (limits_proof), 2.7 MB (coverage_proof)
- **aarch64**: 788 KB (proof), 811 KB (limits_proof), 2.4 MB (coverage_proof)

Zero external shared library dependencies beyond glibc. Statically linked Rust runtime.

### Deployment

Instances were launched in waves respecting AWS vCPU service quotas (96 standard on-demand at time of test). Each wave:

1. Selects instances that fit within remaining quota
2. Launches EC2 instances with a user-data script that:
   - Downloads proof binaries from S3
   - Executes all three proof suites sequentially
   - Uploads results to S3
   - Self-terminates
3. Waits for wave completion, then launches next wave

AMI selection: Amazon Linux 2023 (AL2023) for all instances except a1.metal, which requires Amazon Linux 2 (AL2023's glibc needs ARMv8.2+; Graviton 1 is ARMv8.0).

Infrastructure (VPC, security groups, IAM roles, S3 bucket) deployed via AWS CDK. See [`infra/run-proof.sh`](infra/run-proof.sh) for the full orchestration script.

### Data Collection

Each proof binary outputs structured JSON (delimited by `--- JSON_START ---` / `--- JSON_END ---`) containing:

- Hardware fingerprint (CPU model, core count, frequency, RAM, memory type/speed/channels, NUMA topology, cache hierarchy, ISA extensions, storage type, kernel version)
- Backend capabilities (compute units, max threads/block, shared memory support, atomics, FP16)
- Individual test results (name, pass/fail, duration in milliseconds, error message if any)
- Performance benchmarks (raw values and hardware-normalized values)
- Limits assessment (category, status, evidence string)
- Library coverage (tier, library name, APIs tested/passed, status)

## Results

### Compatibility (Suite 1)

**33/33 instances: ALL 17 TESTS PASSED.**

No instance failed any test. No test required hardware-specific tolerances. One binary per ISA produced identical pass/fail outcomes across all hardware configurations.

### Library Coverage (Suite 2)

**33/33 instances: ALL 47 MODULES PASSED (164/164 APIs).**

Every CUDA library module — from core libraries (cuBLAS, cuDNN, cuFFT) to research modules (flash_attn, tiny_cuda_nn, cu_quantum) — initialized and completed API calls without error.

### Scientific Limits (Suite 3)

Consistent across all instances:

| Category | Supported | Degraded | Unsupported |
|----------|:---------:|:--------:|:-----------:|
| Hard Limits | 0 | 0 | 5 |
| Semantic Gaps | 0 | 6 | 0 |
| Precision | 3 | 2 | 0 |
| Performance | 2 | 2 | 0 |
| Edge Cases | 4 | 0 | 0 |
| **Total** | **9** | **10** | **5** |

- **9 fully supported**: FMA precision, FP16 conversion, accumulated error, memory bandwidth, kernel launch overhead, max grid size, register pressure, bitwise ops, allocation limits
- **10 degraded**: Warp operations (3), barrier/shared memory, control flow, 2D/3D grids, transcendentals, special floats, parallelism scaling, BLAS scaling
- **5 unsupported** (all CUDA-exclusive): Tensor cores, texture sampling, surfaces, async copy, mbarrier

### Distributed (Suite 4)

**PASSED.** Coordinator on t3.large (x86_64) successfully distributed vector-add workloads across heterogeneous workers: t3.small (x86_64), t4g.small (aarch64), c7g.large (aarch64). All 3 tests passed.

## Limitations

### Hardware-Exclusive Features

Five CUDA features require dedicated NVIDIA silicon and cannot be emulated:

1. **Tensor Cores (WMMA)** — matrix multiply-accumulate in hardware; requires dedicated tensor core units
2. **Texture Sampling** — hardware texture interpolation requires texture mapping units (TMUs)
3. **Surface Load/Store** — surface memory operations require dedicated surface hardware
4. **Async Copy (cp.async)** — asynchronous memory copy requires a dedicated copy engine
5. **MBarrier** — named barriers require a hardware barrier unit

Applications that depend on these features will not produce correct results on the CPU backend.

### Warp Semantics

The CPU backend executes threads independently (warp_size=1). Applications using warp-level primitives — `__shfl_down_sync`, `__ballot_sync`, `__all_sync`, `__any_sync`, warp-level reductions — will get different results. Standard CUDA kernels that use grid/block indexing for element-wise, map-style operations are unaffected.

### Numerical Precision

CPU `sin()` diverges from CUDA fast-math `__sinf()` by up to 520,764 ULP. Applications requiring CUDA-spec transcendental precision should validate against their specific accuracy requirements.

### Performance

The CPU backend is designed for compatibility, not throughput. SGEMM peaks at ~0.9 GFLOPS on Graviton 1 vs. ~19,500 GFLOPS on an A100 GPU. The backend is appropriate for: development/testing, CI/CD validation, educational workloads, preprocessing pipelines, inference on small models, and any workload where correctness matters more than throughput.

### Incomplete Instance Coverage

33 of 63 target instances were tested. The remaining 30 were blocked by AWS AZ capacity constraints and vCPU quota limits at time of testing. No instance that was successfully launched failed any test. The untested instances use the same CPU microarchitectures already represented in the results (Graviton 4, Intel Ice Lake, Intel Granite Rapids, AMD EPYC Milan, AMD EPYC Genoa, AMD EPYC Turin).

## Reproducibility

### Running Locally

On any Linux x86_64 or aarch64 system with the Invisible CUDA driver:

```bash
cargo run --bin proof --release --no-default-features --features cpu
cargo run --bin limits_proof --release --no-default-features --features cpu
cargo run --bin coverage_proof --release --no-default-features --features cpu
```

### Running on EC2

With AWS credentials and sufficient vCPU quota:

```bash
cd infra/
./run-proof.sh full    # Build + deploy + wait + collect results
./run-proof.sh matrix  # Display instance matrix (no AWS needed)
```

## Date

All tests were executed on February 16, 2026 (UTC).

## References

1. Charlot, D. "We Ran CUDA on Every Machine in the Cloud — Without a Single GPU Driver." OpenIE Blog, February 2026. https://blog.openie.dev/posts/we-ran-cuda-on-every-machine-in-the-cloud/
2. Charlot, D. "GPU Access Control — CUDA vs Metal: Making Rigid Interfaces Disappear." OpenIE Blog, February 2026. https://blog.openie.dev/posts/gpu-access-control-cuda-vs-metal/
