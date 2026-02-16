# Invisible CUDA

**A universal CUDA compatibility layer that runs CUDA workloads on any hardware.**

We deployed CUDA test suites to 33 EC2 instance types — ARM and x86, bare metal and virtualized, GPU instances with no NVIDIA drivers installed — and every test passed.

## Results

| Instance | Arch | CPU | Cores | RAM | GPU | $/hr | Result |
|----------|------|-----|------:|----:|-----|-----:|--------|
| t4g.micro | ARM | Neoverse-N1 (Graviton 2) | 2 | 0.9 GB | — | $0.008 | 17/17 PASS |
| t3.micro | x86 | Xeon Platinum 8259CL | 2 | 0.9 GB | — | $0.010 | 17/17 PASS |
| t4g.small | ARM | Neoverse-N1 (Graviton 2) | 2 | 1.8 GB | — | $0.017 | 17/17 PASS |
| t3.small | x86 | Xeon Platinum 8259CL | 2 | 1.9 GB | — | $0.021 | 17/17 PASS |
| m7g.medium | ARM | Graviton 3 | 1 | 3.7 GB | — | $0.041 | 17/17 PASS |
| c7g.large | ARM | Graviton 3 | 2 | 3.7 GB | — | $0.073 | 17/17 PASS |
| t3.large | x86 | Xeon Platinum 8259CL | 2 | 7.6 GB | — | $0.083 | 17/17 PASS |
| m7i.large | x86 | Xeon Platinum 8488C | 2 | 7.6 GB | — | $0.100 | 17/17 PASS |
| r7g.large | ARM | Graviton 3 | 2 | 15.3 GB | — | $0.107 | 17/17 PASS |
| r7i.large | x86 | Xeon Platinum 8488C | 2 | 15.3 GB | — | $0.132 | 17/17 PASS |
| c7i.2xlarge | x86 | Xeon Platinum 8488C | 8 | 15.3 GB | — | $0.357 | 17/17 PASS |
| g4ad.xlarge | x86 | AMD EPYC 7R32 | 4 | 15.0 GB | Radeon Pro V520 | $0.379 | 17/17 PASS |
| a1.metal | ARM | Cortex-A72 (Graviton 1) | 16 | 31.5 GB | — | $0.408 | 17/17 PASS |
| g5g.xlarge | ARM | Neoverse-N1 (Graviton 2) | 4 | 7.6 GB | NVIDIA T4G | $0.421 | 17/17 PASS |
| g4dn.xlarge | x86 | Xeon Platinum 8259CL | 4 | 15.4 GB | NVIDIA T4 | $0.526 | 17/17 PASS |
| c7g.4xlarge | ARM | Graviton 3 | 16 | 30.8 GB | — | $0.579 | 17/17 PASS |
| hpc7g.4xlarge | ARM | Graviton 3E | 16 | 123.6 GB | — | $0.680 | 17/17 PASS |
| c7i.4xlarge | x86 | Xeon Platinum 8488C | 16 | 30.8 GB | — | $0.714 | 17/17 PASS |
| g6.xlarge | x86 | AMD EPYC 7R13 | 4 | 15.0 GB | NVIDIA L4 | $0.805 | 17/17 PASS |
| r7g.4xlarge | ARM | Graviton 3 | 16 | 123.6 GB | — | $0.854 | 17/17 PASS |
| g5.xlarge | x86 | AMD EPYC 7R32 | 4 | 15.4 GB | NVIDIA A10G | $1.006 | 17/17 PASS |
| c7g.8xlarge | ARM | Graviton 3 | 32 | 61.7 GB | — | $1.158 | 17/17 PASS |
| c7i.8xlarge | x86 | Xeon Platinum 8488C | 32 | 61.8 GB | — | $1.428 | 17/17 PASS |
| c6g.metal | ARM | Neoverse-N1 (Graviton 2) | 64 | 128 GB | — | $2.176 | 17/17 PASS |
| c7g.metal | ARM | Graviton 3 | 64 | 125.5 GB | — | $2.320 | 17/17 PASS |
| m6g.metal | ARM | Neoverse-N1 (Graviton 2) | 64 | 251.2 GB | — | $2.464 | 17/17 PASS |
| g5g.metal | ARM | Neoverse-N1 (Graviton 2) | 64 | 125.5 GB | NVIDIA T4G | $2.744 | 17/17 PASS |
| r6g.metal | ARM | Neoverse-N1 (Graviton 2) | 64 | 502.7 GB | — | $3.226 | 17/17 PASS |
| r7g.metal | ARM | Graviton 3 | 64 | 502.7 GB | — | $3.427 | 17/17 PASS |
| m5zn.metal | x86 | Xeon Platinum 8252C | 48 | 188.5 GB | — | $3.964 | 17/17 PASS |
| z1d.metal | x86 | Xeon Platinum 8151 | 48 | 377.5 GB | — | $4.464 | 17/17 PASS |
| g4dn.metal | x86 | Xeon Platinum 8259CL | 96 | 377.5 GB | 8x NVIDIA T4 | $7.824 | 17/17 PASS |

**Additionally**: 47/47 CUDA library modules passed on every instance. A distributed cluster test passed across heterogeneous x86 + ARM workers.

## What's in This Repository

| Directory | Contents |
|-----------|----------|
| [`proof/`](proof/) | Source code for all proof binaries (compatibility, coverage, limits, distributed) |
| [`results/raw/`](results/raw/) | Complete, unmodified output from all 33 instance runs |
| [`infra/`](infra/) | EC2 orchestration script (wave-based deployment) |
| [`EXPERIMENT.md`](EXPERIMENT.md) | Full experimental design and methodology |

## What's NOT in This Repository

The Invisible CUDA driver — the compute backend, kernel IR, codegen, BLAS implementation, PTX translation pipeline, Metal shaders, and CUDA library stubs — is not included. This repository contains the **test harness and results**, not the driver itself.

The proof source code is published for transparency: to show exactly what was tested, how correctness was verified, and what tolerances were applied.

## Hardware Coverage

8 CPU microarchitectures across 4 vendors:

| Vendor | Microarchitecture | Instances Tested |
|--------|-------------------|:---:|
| AWS | Cortex-A72 (Graviton 1, 2018) | 1 |
| AWS | Neoverse-N1 (Graviton 2, 2019) | 7 |
| AWS | Graviton 3 / 3E (2022) | 10 |
| Intel | Skylake (Xeon 8151) | 1 |
| Intel | Cascade Lake (Xeon 8259CL, 8252C) | 7 |
| Intel | Sapphire Rapids (Xeon 8488C) | 6 |
| AMD | EPYC 7R32 (Zen 2) | 2 |
| AMD | EPYC 7R13 (Zen 3) | 1 |

Range: 1 core / 0.9 GB / $0.008/hr to 96 cores / 502.7 GB / $7.824/hr.

## Blog Post

[We Ran CUDA on Every Machine in the Cloud — Without a Single GPU Driver](https://blog.openie.dev/posts/we-ran-cuda-on-every-machine-in-the-cloud/)

## License

MIT
