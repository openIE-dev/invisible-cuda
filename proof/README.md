# Proof Source Code

This directory contains the source code for the Invisible CUDA proof suite — the test binaries that were deployed to 33 EC2 instance types to verify CUDA compatibility across hardware.

> **Note:** These sources import from `invisible_cuda::*` (the Invisible CUDA driver). The driver source is not included in this repository. The code is published for transparency — to show exactly what was tested, how correctness was verified, and what tolerances were applied.

## Files

### proof.rs — Compatibility Proof (17 tests)

The primary proof binary. Runs three test categories:

**Correctness Tests (9)**

| Test | What It Verifies | Tolerance |
|------|-----------------|-----------|
| Vector Add (64/1K/64K) | Element-wise addition at three sizes | < 1e-5 per element |
| Vector Mul (4096) | Element-wise multiplication | < 1e-3 per element |
| FMA x*y+z (2048) | Fused multiply-add | < 1e-4 per element |
| SAXPY a*x+y (4096) | BLAS Level 1 via IR kernels | < 1e-4 per element |
| Memory round-trip (64B..4MB) | Host → Device → Host at 7 sizes | Byte-exact |
| Memset (4KB..1MB) | Fill patterns (0x00, 0xFF, 0xAA, 0xDE) | Byte-exact |
| D2D copy (4KB..1MB) | Device-to-device buffer copy | Byte-exact |
| Multi alloc+free (150 buffers) | Allocate and free 150 buffers | No errors |
| Multi-kernel dispatch (add+mul) | Sequential kernel execution | < 1e-4 per element |

**BLAS Correctness (5)**

| Test | What It Verifies |
|------|-----------------|
| SGEMM (32..256, 4 sizes) | Matrix multiply at M=N=K of 32, 64, 128, 256 |
| SGEMM alpha=2.5 beta=0.5 | Non-trivial alpha/beta GEMM parameters |
| BLAS SAXPY | y = alpha*x + y |
| BLAS SDOT | Dot product |
| BLAS SNRM2 | Euclidean norm |

SGEMM verified against a CPU reference implementation with tolerance < 1e-2 per element.

**Stress Tests (3)**

| Test | What It Verifies |
|------|-----------------|
| Large allocation (up to 1GB) | Progressive allocation doubling until failure or 1 GB |
| Rapid kernels (10K dispatches) | 10,000 consecutive kernel launches |
| 500 concurrent buffers | Simultaneous buffer management |

**Performance Benchmarks (14, informational)**

Memory bandwidth (4 sizes), vector add throughput (3 sizes), SGEMM GFLOPS (5 sizes), kernel launch latency, allocation throughput. Performance numbers are reported but are not pass/fail criteria.

### coverage_proof.rs — Library Coverage Proof (47 modules)

Tests that all 47 CUDA library runtimes initialize, execute representative API calls, and shut down cleanly. Each module follows: create runtime → create handle → invoke APIs → destroy → report status.

Modules tested: cuBLAS, cuBLASLt, cuDNN, cuFFT, cuSPARSE, cuRAND, cuSOLVER, cuTENSOR, NCCL, NVML, Thrust/CUB, NVRTC, NVENC, NVDEC, nvJPEG, nvJPEG2K, NPP, cuSPARSELt, TensorRT, NVTX, cuFile, NvOF, nvdiffrast, spconv, gaussian_rast, flash_attn, nerfacc, bitsandbytes, detectron2_ops, pointnet, pytorch3d, faiss_gpu, molecular_dynamics, gpu_crypto, rapids, audio_ops, cutlass, triton_kernels, apex, tiny_cuda_nn, xformers, warp_sim, kaolin, cu_quantum, dali, cu_dss, HIP/ROCm.

### limits_proof.rs — Scientific Limits Proof (24 tests)

Systematically identifies the boundaries of CUDA compatibility. Tests are categorized by status:

- **Supported**: Full CUDA behavior
- **Degraded**: Correct results with documented behavioral differences
- **Unsupported**: Requires NVIDIA hardware (CUDA-exclusive)

Five categories: Hard Limits (5), Semantic Gaps (6), Precision Boundaries (5), Performance Scaling (4), Edge Cases (4).

### dist_proof.rs — Distributed Cluster Proof

7-step orchestration: connect to workers, exchange capabilities, measure latency, compile kernels, distribute vector-add across workers, distribute large vector-add, shutdown. Tests heterogeneous execution across x86 and ARM workers.

### worker.rs — Distributed Worker Daemon

TCP server with a frame-based protocol. Handles: Ping/Pong, GetCapabilities, CompileKernel, Alloc/Free, CopyHtoD/CopyDtoH, Launch, Shutdown. Each worker runs its own local backend.

### hw_fingerprint.rs — Hardware Fingerprint Collection

Collects: CPU model, core count, frequency, RAM, memory type/speed/channels, NUMA topology, cache hierarchy, ISA extensions, storage type, kernel version, microcode revision. Used to document the hardware for each test run.

## Key Abstractions Used

The proof binaries interact with the driver through these interfaces:

```rust
// Backend detection and dispatch
invisible_cuda::compute::detect_backend() -> AnyBackend
invisible_cuda::compute::AnyBackend  // enum dispatch over backends

// ComputeBackend trait (implemented by CPU, Metal, Vulkan, etc.)
trait ComputeBackend {
    fn compile_kernel(&mut self, name: &str, ir: &KernelIR) -> Result<(), String>;
    fn alloc(&mut self, size: usize, shared: bool) -> Result<(BackendBufferId, Option<*mut u8>), String>;
    fn free(&mut self, id: BackendBufferId) -> Result<(), String>;
    fn copy_htod(&mut self, id: BackendBufferId, data: &[u8]) -> Result<(), String>;
    fn copy_dtoh(&self, id: BackendBufferId, size: usize) -> Result<Vec<u8>, String>;
    fn launch(&mut self, name: &str, grid: (u32,u32,u32), block: (u32,u32,u32), params: &[KernelParam]) -> Result<(), String>;
    fn synchronize(&self) -> Result<(), String>;
}

// Kernel intermediate representation
invisible_cuda::ir::KernelIR  // portable kernel graph
invisible_cuda::ir::Op        // operation nodes: Add, Mul, Fma, Input, Output, ...

// BLAS abstraction
invisible_cuda::blas::traits::BlasBackend  // sgemm, saxpy, sdot, snrm2
```

The EC2 experiment used the CPU backend exclusively (`--features cpu`). All 33 instances ran CUDA workloads on their CPUs, regardless of whether GPU hardware was present.
