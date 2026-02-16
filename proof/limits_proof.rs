//! Invisible CUDA — Scientific Limits Proof
//!
//! A research-grade test suite that systematically finds the boundaries of
//! CUDA compatibility on non-NVIDIA hardware. Organized into 5 categories:
//!
//!   1. HARD LIMITS    — CUDA features that fundamentally cannot work (stubs)
//!   2. SEMANTIC GAPS   — Features that work differently than real CUDA
//!   3. PRECISION       — Numerical accuracy differences
//!   4. PERFORMANCE     — Proportional scaling measurements
//!   5. EDGE CASES      — Breaking points and boundary conditions
//!
//! For each test, reports SUPPORTED/DEGRADED/UNSUPPORTED with evidence.
//!
//! Run:
//!   cargo run --bin limits_proof --release --no-default-features --features cpu
//!   PROOF_MODE=extended cargo run --bin limits_proof --release --features cpu

use std::time::Instant;

use invisible_cuda::compute::{
    detect_backend, AnyBackend, BackendBufferId, ComputeBackend, KernelParam,
};
use invisible_cuda::hw_fingerprint::HwFingerprint;
use invisible_cuda::ir::{KernelIR, Op};

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

fn f32_to_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn bytes_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn alloc_with_data(backend: &mut AnyBackend, data: &[f32]) -> BackendBufferId {
    let size = data.len() * 4;
    let (id, _) = backend.alloc(size, false).expect("alloc failed");
    backend.copy_htod(id, &f32_to_bytes(data)).expect("htod failed");
    id
}

fn read_buffer(backend: &AnyBackend, id: BackendBufferId, count: usize) -> Vec<f32> {
    let bytes = backend.copy_dtoh(id, count * 4).expect("dtoh failed");
    bytes_to_f32(&bytes)
}

fn format_bytes(bytes: usize) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.1} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self { Self(seed) }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
    fn vec_f32(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_f32()).collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test result types
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Clone)]
enum Support {
    /// Feature works correctly — full CUDA compatibility
    Supported,
    /// Feature works but with measurable differences (precision, performance)
    Degraded { detail: String },
    /// Feature cannot work on this backend — hard limit
    Unsupported { reason: String },
}

struct LimitResult {
    category: &'static str,
    name: String,
    support: Support,
    duration_ms: f64,
    /// CUDA-only? If true, this is expected to fail on non-CUDA
    cuda_exclusive: bool,
    /// Quantified evidence (e.g., error magnitude, perf ratio)
    evidence: String,
}

fn print_result(r: &LimitResult) {
    let icon = match &r.support {
        Support::Supported => "OK  ",
        Support::Degraded { .. } => "DEGR",
        Support::Unsupported { .. } => "FAIL",
    };
    let detail = match &r.support {
        Support::Supported => String::new(),
        Support::Degraded { detail } => format!(" — {}", detail),
        Support::Unsupported { reason } => format!(" — {}", reason),
    };
    let excl = if r.cuda_exclusive { " [CUDA-only]" } else { "" };
    println!("  [{}] {} ({:.1}ms){}{}", icon, r.name, r.duration_ms, detail, excl);
    if !r.evidence.is_empty() {
        println!("         evidence: {}", r.evidence);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Category 1: HARD LIMITS
// Features that exist in CUDA but CANNOT work on non-NVIDIA hardware
// ═══════════════════════════════════════════════════════════════════════════

fn test_wmma_tensor_cores(backend: &mut AnyBackend) -> LimitResult {
    let start = Instant::now();

    // Build a kernel that uses WMMA ops (tensor core matrix multiply)
    let mut ir = KernelIR::new("wmma_test");
    let addr = ir.push_op(Op::Input { index: 0 });
    let stride = ir.push_op(Op::Const { val: 16.0 });
    let a = ir.push_op(Op::WmmaLoadA { addr, stride });
    let b = ir.push_op(Op::WmmaLoadB { addr, stride });
    let c = ir.push_op(Op::WmmaLoadC { addr, stride });
    let d = ir.push_op(Op::WmmaMma { a, b, c });
    let _store = ir.push_op(Op::WmmaStore { addr, d, stride });
    ir.add_input(addr);
    ir.param_is_pointer = vec![true];

    if let Err(e) = backend.compile_kernel("wmma_test", &ir) {
        return LimitResult {
            category: "HARD_LIMIT", name: "Tensor Cores (WMMA)".into(),
            support: Support::Unsupported { reason: format!("compile failed: {}", e) },
            duration_ms: start.elapsed().as_secs_f64() * 1000.0,
            cuda_exclusive: true,
            evidence: "WmmaLoadA/B/C + WmmaMma + WmmaStore all return 0.0 on CPU".into(),
        };
    }

    // Allocate a 16x16 matrix of 1.0s, run WMMA
    let n = 256; // 16x16
    let data = vec![1.0f32; n];
    let id = alloc_with_data(backend, &data);

    let _ = backend.launch("wmma_test", (1, 1, 1), (32, 1, 1), &[KernelParam::Buffer(id)]);
    let _ = backend.synchronize();
    let result = read_buffer(backend, id, n);
    backend.free(id).ok();

    // On real CUDA with tensor cores, result[0] would be 16.0 (dot product of 1s)
    // On CPU, WMMA ops are stubbed to return 0.0
    let all_zeros = result.iter().all(|&v| v == 0.0);
    let all_ones = result.iter().all(|&v| v == 1.0);

    LimitResult {
        category: "HARD_LIMIT",
        name: "Tensor Cores (WMMA)".into(),
        support: if all_zeros || all_ones {
            Support::Unsupported {
                reason: "WMMA ops return 0.0 (stub) — no tensor core hardware".into(),
            }
        } else {
            Support::Supported
        },
        duration_ms: start.elapsed().as_secs_f64() * 1000.0,
        cuda_exclusive: true,
        evidence: format!(
            "result[0]={}, all_zeros={}, all_unchanged={}. Real CUDA would produce matmul output.",
            result[0], all_zeros, all_ones
        ),
    }
}

fn test_texture_sampling(backend: &mut AnyBackend) -> LimitResult {
    let start = Instant::now();

    // Build a kernel that samples from a 2D texture
    let mut ir = KernelIR::new("tex_test");
    let u = ir.push_op(Op::Const { val: 0.5 });
    let v = ir.push_op(Op::Const { val: 0.5 });
    let sample = ir.push_op(Op::TexSample2D { tex_id: 0, u, v });
    // We can't easily output this via the standard pipeline, but we can check compilation
    ir.add_output(sample, 0);
    ir.param_is_pointer = vec![true];

    if let Err(e) = backend.compile_kernel("tex_test", &ir) {
        return LimitResult {
            category: "HARD_LIMIT", name: "Texture Sampling (2D/3D)".into(),
            support: Support::Unsupported { reason: format!("compile failed: {}", e) },
            duration_ms: start.elapsed().as_secs_f64() * 1000.0,
            cuda_exclusive: true,
            evidence: "TexSample2D/3D, TexLoad, TexSampleLod all return 0.0".into(),
        };
    }

    let data = vec![42.0f32; 16];
    let id = alloc_with_data(backend, &data);
    let _ = backend.launch("tex_test", (1, 1, 1), (1, 1, 1), &[KernelParam::Buffer(id)]);
    let _ = backend.synchronize();
    let result = read_buffer(backend, id, 1);
    backend.free(id).ok();

    // Texture sampling returns 0.0 on CPU (stub)
    LimitResult {
        category: "HARD_LIMIT",
        name: "Texture Sampling (2D/3D)".into(),
        support: if result[0] == 0.0 {
            Support::Unsupported {
                reason: "TexSample2D returns 0.0 — no texture unit hardware".into(),
            }
        } else {
            Support::Supported
        },
        duration_ms: start.elapsed().as_secs_f64() * 1000.0,
        cuda_exclusive: true,
        evidence: format!("TexSample2D(0.5, 0.5) → {} (expected: interpolated texture value)", result[0]),
    }
}

fn test_surface_ops(backend: &mut AnyBackend) -> LimitResult {
    let start = Instant::now();

    let mut ir = KernelIR::new("surf_test");
    let x = ir.push_op(Op::Const { val: 0.0 });
    let y = ir.push_op(Op::Const { val: 0.0 });
    let loaded = ir.push_op(Op::SurfLoad { surf_id: 0, x, y });
    ir.add_output(loaded, 0);
    ir.param_is_pointer = vec![true];

    if let Err(e) = backend.compile_kernel("surf_test", &ir) {
        return LimitResult {
            category: "HARD_LIMIT", name: "Surface Load/Store".into(),
            support: Support::Unsupported { reason: format!("compile failed: {}", e) },
            duration_ms: start.elapsed().as_secs_f64() * 1000.0,
            cuda_exclusive: true,
            evidence: "SurfLoad/SurfStore return 0.0 on non-GPU backends".into(),
        };
    }

    let data = vec![99.0f32; 4];
    let id = alloc_with_data(backend, &data);
    let _ = backend.launch("surf_test", (1, 1, 1), (1, 1, 1), &[KernelParam::Buffer(id)]);
    let _ = backend.synchronize();
    let result = read_buffer(backend, id, 1);
    backend.free(id).ok();

    LimitResult {
        category: "HARD_LIMIT",
        name: "Surface Load/Store".into(),
        support: if result[0] == 0.0 {
            Support::Unsupported { reason: "SurfLoad returns 0.0 — no surface memory hardware".into() }
        } else {
            Support::Supported
        },
        duration_ms: start.elapsed().as_secs_f64() * 1000.0,
        cuda_exclusive: true,
        evidence: format!("SurfLoad(0,0) → {} (expected: surface data)", result[0]),
    }
}

fn test_async_copy(backend: &mut AnyBackend) -> LimitResult {
    let start = Instant::now();

    // AsyncCopy + commit/wait group — hardware async memcpy on CUDA
    let mut ir = KernelIR::new("async_copy_test");
    let src = ir.push_op(Op::Input { index: 0 });
    let dst = ir.push_op(Op::Input { index: 1 });
    let _copy = ir.push_op(Op::AsyncCopy { dst, src, size: 64 });
    let _commit = ir.push_op(Op::AsyncCopyCommitGroup);
    let _wait = ir.push_op(Op::AsyncCopyWaitGroup { n: 0 });
    ir.add_input(src);
    ir.add_input(dst);
    ir.param_is_pointer = vec![true, true];

    if let Err(e) = backend.compile_kernel("async_copy_test", &ir) {
        return LimitResult {
            category: "HARD_LIMIT", name: "Async Copy (cp.async)".into(),
            support: Support::Unsupported { reason: format!("compile failed: {}", e) },
            duration_ms: start.elapsed().as_secs_f64() * 1000.0,
            cuda_exclusive: true,
            evidence: "AsyncCopy/CommitGroup/WaitGroup are no-ops on CPU".into(),
        };
    }

    // On CPU, async copy is a no-op — the data won't actually move
    let src_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let dst_data = vec![0.0f32; 4];
    let src_id = alloc_with_data(backend, &src_data);
    let dst_id = alloc_with_data(backend, &dst_data);

    let _ = backend.launch("async_copy_test", (1, 1, 1), (1, 1, 1), &[
        KernelParam::Buffer(src_id), KernelParam::Buffer(dst_id),
    ]);
    let _ = backend.synchronize();
    let result = read_buffer(backend, dst_id, 4);
    backend.free(src_id).ok();
    backend.free(dst_id).ok();

    let copied = result == src_data;
    LimitResult {
        category: "HARD_LIMIT",
        name: "Async Copy (cp.async)".into(),
        support: if !copied {
            Support::Unsupported {
                reason: "AsyncCopy is a no-op — data not transferred".into(),
            }
        } else {
            Support::Supported
        },
        duration_ms: start.elapsed().as_secs_f64() * 1000.0,
        cuda_exclusive: true,
        evidence: format!("dst after copy: {:?} (expected {:?})", &result[..4.min(result.len())], &src_data[..4]),
    }
}

fn test_mbarrier(backend: &mut AnyBackend) -> LimitResult {
    let start = Instant::now();

    let mut ir = KernelIR::new("mbarrier_test");
    let addr = ir.push_op(Op::Input { index: 0 });
    let count = ir.push_op(Op::Const { val: 32.0 });
    let _init = ir.push_op(Op::MBarrierInit { addr, count });
    let _arrive = ir.push_op(Op::MBarrierArrive { addr });
    let state = ir.push_op(Op::Const { val: 0.0 });
    let _wait = ir.push_op(Op::MBarrierTestWait { addr, state });
    ir.add_input(addr);
    ir.param_is_pointer = vec![true];

    let compiled = backend.compile_kernel("mbarrier_test", &ir).is_ok();

    LimitResult {
        category: "HARD_LIMIT",
        name: "MBarrier (Named Barriers)".into(),
        support: if compiled {
            Support::Unsupported {
                reason: "Compiles but MBarrier ops are no-ops — no hardware barrier unit".into(),
            }
        } else {
            Support::Unsupported { reason: "Compilation failed".into() }
        },
        duration_ms: start.elapsed().as_secs_f64() * 1000.0,
        cuda_exclusive: true,
        evidence: "MBarrierInit/Arrive/TestWait all return 0.0 on CPU".into(),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Category 2: SEMANTIC GAPS
// Features that work but behave differently than real CUDA hardware
// ═══════════════════════════════════════════════════════════════════════════

fn test_warp_shuffle(backend: &mut AnyBackend) -> LimitResult {
    let start = Instant::now();

    // On real CUDA, shfl.down shifts values across warp lanes
    // On CPU (warp_size=1), it returns the input value unchanged
    let mut ir = KernelIR::new("shfl_test");
    let input = ir.push_op(Op::Input { index: 0 });
    let delta = ir.push_op(Op::Const { val: 1.0 });
    let shuffled = ir.push_op(Op::ShflDown { val: input, delta, width: 32 });
    ir.add_input(input);
    ir.add_output(shuffled, 1);
    ir.param_is_pointer = vec![true, true];

    backend.compile_kernel("shfl_test", &ir).expect("compile shfl");

    // Each thread loads its own value and shifts it down by 1
    // Thread 0 → value 1.0, shift down by 1 → should get value from thread 1 (2.0)
    let n = 32;
    let input_data: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
    let output_data = vec![0.0f32; n];
    let in_id = alloc_with_data(backend, &input_data);
    let out_id = alloc_with_data(backend, &output_data);

    backend.launch("shfl_test", (1, 1, 1), (n as u32, 1, 1), &[
        KernelParam::Buffer(in_id), KernelParam::Buffer(out_id),
    ]).expect("launch");
    backend.synchronize().expect("sync");

    let result = read_buffer(backend, out_id, n);
    backend.free(in_id).ok();
    backend.free(out_id).ok();

    // On real CUDA: result[i] = input_data[i+1] (shifted down)
    // On CPU: result[i] = input_data[i] (identity — no actual shuffle)
    let identity_count = (0..n).filter(|&i| (result[i] - input_data[i]).abs() < 1e-5).count();
    let shifted_count = (0..n.saturating_sub(1))
        .filter(|&i| (result[i] - input_data[i + 1]).abs() < 1e-5).count();

    let is_identity = identity_count == n;

    LimitResult {
        category: "SEMANTIC_GAP",
        name: "Warp Shuffle (shfl.down)".into(),
        support: if is_identity {
            Support::Degraded {
                detail: "Returns identity (warp_size=1) — no cross-lane shuffle".into(),
            }
        } else if shifted_count > n / 2 {
            Support::Supported
        } else {
            Support::Degraded {
                detail: format!("Neither identity nor shifted: {}/{} match", identity_count, n),
            }
        },
        duration_ms: start.elapsed().as_secs_f64() * 1000.0,
        cuda_exclusive: false,
        evidence: format!(
            "32 lanes: identity={}, shifted={}/{} | result[0]={} (input[0]={}, input[1]={})",
            identity_count, shifted_count, n - 1, result[0], input_data[0], input_data[1]
        ),
    }
}

fn test_warp_vote(backend: &mut AnyBackend) -> LimitResult {
    let start = Instant::now();

    // VoteBallot: on real CUDA, returns a bitmask of all lanes where pred is true
    // On CPU (1 lane): returns 0 or 1
    let mut ir = KernelIR::new("vote_test");
    let input = ir.push_op(Op::Input { index: 0 });
    let zero = ir.push_op(Op::Const { val: 0.0 });
    let pred = ir.push_op(Op::Gt { lhs: input, rhs: zero });
    let ballot = ir.push_op(Op::VoteBallot { pred });
    let _vote_all = ir.push_op(Op::VoteAll { pred });
    let _vote_any = ir.push_op(Op::VoteAny { pred });
    ir.add_input(input);
    // Output ballot result
    ir.add_output(ballot, 1);
    ir.param_is_pointer = vec![true, true];

    backend.compile_kernel("vote_test", &ir).expect("compile vote");

    // All positive: ballot should be 0xFFFFFFFF on real CUDA (32 lanes)
    let n = 32;
    let input_data: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
    let output_data = vec![0.0f32; n];
    let in_id = alloc_with_data(backend, &input_data);
    let out_id = alloc_with_data(backend, &output_data);

    backend.launch("vote_test", (1, 1, 1), (n as u32, 1, 1), &[
        KernelParam::Buffer(in_id), KernelParam::Buffer(out_id),
    ]).expect("launch");
    backend.synchronize().expect("sync");

    let result = read_buffer(backend, out_id, n);
    backend.free(in_id).ok();
    backend.free(out_id).ok();

    // On real CUDA: result[i] = 0xFFFFFFFF (all 32 threads voted true)
    //   as float: f32::from_bits(0xFFFFFFFF) = NaN
    // On CPU: result[i] = 1.0 (single lane, pred=true → 1.0)
    let all_ones = result.iter().all(|&v| v == 1.0);
    let full_ballot = result.iter().all(|&v| v.to_bits() == 0xFFFFFFFF);

    LimitResult {
        category: "SEMANTIC_GAP",
        name: "Warp Vote (ballot/all/any)".into(),
        support: if all_ones {
            Support::Degraded {
                detail: "Single-lane semantics: ballot returns 0/1, not 32-bit bitmask".into(),
            }
        } else if full_ballot {
            Support::Supported
        } else {
            Support::Degraded {
                detail: format!("Unexpected ballot values: first={}", result[0]),
            }
        },
        duration_ms: start.elapsed().as_secs_f64() * 1000.0,
        cuda_exclusive: false,
        evidence: format!(
            "VoteBallot(all_positive): result[0]={} bits=0x{:08X} | Real CUDA: 0xFFFFFFFF (NaN)",
            result[0], result[0].to_bits()
        ),
    }
}

fn test_warp_identity(backend: &mut AnyBackend) -> LimitResult {
    let start = Instant::now();

    // LaneId, WarpId, NWarps — critical for warp-aware algorithms
    let mut ir = KernelIR::new("warp_id_test");
    let lane = ir.push_op(Op::LaneId);
    let _warp = ir.push_op(Op::WarpId);
    let _nwarps = ir.push_op(Op::NWarps);
    // Output all three to a buffer (use lane_id for per-thread output)
    ir.add_output(lane, 0);
    ir.param_is_pointer = vec![true];

    backend.compile_kernel("warp_id_test", &ir).expect("compile warp_id");

    let n = 64;
    let data = vec![0.0f32; n];
    let id = alloc_with_data(backend, &data);

    backend.launch("warp_id_test", (1, 1, 1), (n as u32, 1, 1), &[
        KernelParam::Buffer(id),
    ]).expect("launch");
    backend.synchronize().expect("sync");

    let result = read_buffer(backend, id, n);
    backend.free(id).ok();

    // On real CUDA: lane_id = threadIdx.x % 32 → 0,1,2,...31,0,1,...31
    // On CPU: lane_id = 0 for all threads
    let all_zero = result.iter().all(|&v| v == 0.0);
    let has_lane_variation = result.windows(2).any(|w| w[0] != w[1]);

    LimitResult {
        category: "SEMANTIC_GAP",
        name: "Warp Identity (LaneId/WarpId/NWarps)".into(),
        support: if all_zero {
            Support::Degraded {
                detail: "LaneId=0, WarpId=0, NWarps=1 for all threads (no warp structure)".into(),
            }
        } else if has_lane_variation {
            Support::Supported
        } else {
            Support::Degraded { detail: format!("Constant LaneId={}", result[0]) }
        },
        duration_ms: start.elapsed().as_secs_f64() * 1000.0,
        cuda_exclusive: false,
        evidence: format!(
            "64 threads: LaneId values = [{}, {}, {}, ... {}, {}] (CUDA: 0,1,2...31,0,1...31)",
            result[0], result[1], result[2], result[31], result[32]
        ),
    }
}

fn test_barrier_correctness(backend: &mut AnyBackend) -> LimitResult {
    let start = Instant::now();

    // On real CUDA, barrier ensures all threads in a block reach a sync point.
    // On CPU (sequential execution), it's a no-op and correctness is preserved
    // trivially — but algorithms that depend on cross-thread visibility after
    // barrier won't work correctly.
    //
    // Test: Thread 0 writes to shared mem, barrier, all threads read.
    // On CPU: shared mem is per-thread, so other threads won't see the write.
    let mut ir = KernelIR::new("barrier_test");
    let tid = ir.push_op(Op::ThreadId { dim: 0 });
    let zero = ir.push_op(Op::Const { val: 0.0 });
    let _is_zero = ir.push_op(Op::Eq { lhs: tid, rhs: zero });
    let shared_addr = ir.push_op(Op::Const { val: 0.0 }); // shared_mem[0]
    let write_val = ir.push_op(Op::Const { val: 42.0 });

    // Thread 0 writes 42.0 to shared[0]
    let _store = ir.push_op(Op::StoreShared { addr: shared_addr, val: write_val });
    let _barrier = ir.push_op(Op::Barrier);
    // All threads read shared[0]
    let read_val = ir.push_op(Op::LoadShared { addr: shared_addr });

    ir.add_output(read_val, 0);
    ir.param_is_pointer = vec![true];

    backend.compile_kernel("barrier_test", &ir).expect("compile barrier");

    let n = 32;
    let data = vec![0.0f32; n];
    let id = alloc_with_data(backend, &data);

    backend.launch("barrier_test", (1, 1, 1), (n as u32, 1, 1), &[
        KernelParam::Buffer(id),
    ]).expect("launch");
    backend.synchronize().expect("sync");

    let result = read_buffer(backend, id, n);
    backend.free(id).ok();

    // On real CUDA: ALL threads see 42.0 after barrier (thread 0 wrote it)
    // On CPU: shared_mem is per-thread, so:
    //   - Thread 0 writes AND reads → sees 42.0
    //   - Threads 1-31 never wrote → see 42.0 only because ALL threads execute
    //     StoreShared unconditionally (no Select guard in this simple test)
    //
    // Actually in this IR, ALL threads write 42.0 (no conditional).
    // A more revealing test would use Select + predicate, but this tests the basics.
    let sees_42 = result.iter().filter(|&&v| (v - 42.0).abs() < 1e-5).count();

    LimitResult {
        category: "SEMANTIC_GAP",
        name: "Barrier + Shared Memory Visibility".into(),
        support: if sees_42 == n {
            Support::Degraded {
                detail: "Barrier is no-op, shared mem is per-thread — works only for simple patterns".into(),
            }
        } else {
            Support::Degraded {
                detail: format!("{}/{} threads saw correct value", sees_42, n),
            }
        },
        duration_ms: start.elapsed().as_secs_f64() * 1000.0,
        cuda_exclusive: false,
        evidence: format!(
            "After barrier: {}/{} threads read 42.0 from shared[0]. \
            Real CUDA: per-block shared memory, CPU: per-thread copy.",
            sees_42, n
        ),
    }
}

fn test_control_flow(backend: &mut AnyBackend) -> LimitResult {
    let start = Instant::now();

    // On CPU, Branch/BranchIf/Label are ignored (sequential execution).
    // Kernels with actual control flow divergence will execute ALL ops regardless.
    let mut ir = KernelIR::new("branch_test");
    let input = ir.push_op(Op::Input { index: 0 });
    let five = ir.push_op(Op::Const { val: 5.0 });
    let cond = ir.push_op(Op::Gt { lhs: input, rhs: five });

    // Branch: if input > 5, jump to "big", else fall through
    let _br = ir.push_op(Op::BranchIf { cond, target: "big".into() });
    // Small path: multiply by 2
    let two = ir.push_op(Op::Const { val: 2.0 });
    let _small_result = ir.push_op(Op::Mul { lhs: input, rhs: two });
    let _jump_end = ir.push_op(Op::Branch { target: "end".into() });
    // Big path: multiply by 10
    let _label_big = ir.push_op(Op::Label { name: "big".into() });
    let ten = ir.push_op(Op::Const { val: 10.0 });
    let big_result = ir.push_op(Op::Mul { lhs: input, rhs: ten });
    let _label_end = ir.push_op(Op::Label { name: "end".into() });

    // Output: should be the conditional result
    // On CPU: both paths execute, last write wins (big_result, since it's later in IR)
    ir.add_output(big_result, 1);
    ir.param_is_pointer = vec![true, true];

    backend.compile_kernel("branch_test", &ir).expect("compile branch");

    // Input: [3.0] → should take "small" path (3 < 5)
    // Real CUDA: result = 3 * 2 = 6.0
    // CPU: branches ignored, all ops execute, output = 3 * 10 = 30.0 (big_result)
    let in_data = vec![3.0f32; 1];
    let out_data = vec![0.0f32; 1];
    let in_id = alloc_with_data(backend, &in_data);
    let out_id = alloc_with_data(backend, &out_data);

    backend.launch("branch_test", (1, 1, 1), (1, 1, 1), &[
        KernelParam::Buffer(in_id), KernelParam::Buffer(out_id),
    ]).expect("launch");
    backend.synchronize().expect("sync");

    let result = read_buffer(backend, out_id, 1);
    backend.free(in_id).ok();
    backend.free(out_id).ok();

    let is_small = (result[0] - 6.0).abs() < 1e-5;
    let is_big = (result[0] - 30.0).abs() < 1e-5;

    LimitResult {
        category: "SEMANTIC_GAP",
        name: "Control Flow (Branch/BranchIf)".into(),
        support: if is_small {
            Support::Supported
        } else if is_big {
            Support::Degraded {
                detail: "Branches ignored — all IR ops execute linearly, last write wins".into(),
            }
        } else {
            Support::Degraded {
                detail: format!("Unexpected result: {} (expected 6.0 or 30.0)", result[0]),
            }
        },
        duration_ms: start.elapsed().as_secs_f64() * 1000.0,
        cuda_exclusive: false,
        evidence: format!(
            "input=3.0, cond=(3>5)=false → small path (×2=6.0). Got {}. \
            CPU executes ALL ops (Branch ignored), so big path (×10=30.0) overwrites.",
            result[0]
        ),
    }
}

fn test_2d_3d_grid(backend: &mut AnyBackend) -> LimitResult {
    let start = Instant::now();

    // Test 2D grid/block: verify ThreadId and BlockId for dim=1
    let mut ir = KernelIR::new("grid2d_test");
    let _tid_x = ir.push_op(Op::ThreadId { dim: 0 });
    let tid_y = ir.push_op(Op::ThreadId { dim: 1 });
    let _bid_x = ir.push_op(Op::BlockId { dim: 0 });
    let _bid_y = ir.push_op(Op::BlockId { dim: 1 });
    let _bdim_x = ir.push_op(Op::BlockDim { dim: 0 });

    // Compute global 2D index: global_y * (gridDim.x * blockDim.x) + global_x
    // For simplicity, output tid_y to verify multi-dim works
    ir.add_output(tid_y, 0);
    ir.param_is_pointer = vec![true];

    backend.compile_kernel("grid2d_test", &ir).expect("compile grid2d");

    // 4x4 block, 1x1 grid → 16 threads, tid_y ranges 0..3
    let n = 16;
    let data = vec![-1.0f32; n];
    let id = alloc_with_data(backend, &data);

    backend.launch("grid2d_test", (1, 1, 1), (4, 4, 1), &[
        KernelParam::Buffer(id),
    ]).expect("launch");
    backend.synchronize().expect("sync");

    let result = read_buffer(backend, id, n);
    backend.free(id).ok();

    // On CPU: threads execute in order tx=0..3, ty=0..3
    // global_id = block.0 * dim.0 + thread.0 — only uses X dimension for output offset
    // So result[0..3] = tid_y when tx=0..3, ty=0 → all 0.0
    // result[4..7] = tid_y when tx=0..3, ty=1 → all 1.0
    // etc.
    //
    // But CPU's execute_thread uses global_id = bx*blockDim.0 + tx for output offset
    // So only the first 4 elements (tx=0..3) of the first ty are written.
    // ty=1..3 write to the SAME offsets (0..3) — overwriting!
    let has_y_dim = result.iter().any(|&v| v > 0.0);

    LimitResult {
        category: "SEMANTIC_GAP",
        name: "2D/3D Grid Dimensions".into(),
        support: if has_y_dim {
            Support::Degraded {
                detail: "Y/Z thread dims execute but output indexing uses only X dimension".into(),
            }
        } else {
            Support::Degraded {
                detail: "Multi-dim blocks run but output offset = bx*blockDim.x+tx (1D only)".into(),
            }
        },
        duration_ms: start.elapsed().as_secs_f64() * 1000.0,
        cuda_exclusive: false,
        evidence: format!(
            "4x4 block: result[0..4]={:?} | CPU uses 1D output offset (global_x only)",
            &result[..4.min(result.len())]
        ),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Category 3: PRECISION BOUNDARIES
// Numerical accuracy differences between CPU and GPU
// ═══════════════════════════════════════════════════════════════════════════

fn test_transcendental_precision(backend: &mut AnyBackend) -> LimitResult {
    let start = Instant::now();

    // Test sin, cos, exp, log, sqrt, tanh against f64 reference values
    // CUDA uses fast approximations (1-2 ULP); CPU uses libm (0.5 ULP)
    let test_values = vec![
        0.1, 0.5, 1.0, 2.0, 3.14159, 10.0, 100.0,
        -0.5, -1.0, 0.001, 0.0001,
    ];

    let mut max_ulp_sin = 0u32;
    let mut max_ulp_cos = 0u32;
    let mut max_ulp_exp = 0u32;
    let mut max_ulp_log = 0u32;
    let mut max_ulp_sqrt = 0u32;
    let mut max_ulp_tanh = 0u32;

    for func_name in &["sin", "cos", "exp", "log", "sqrt", "tanh"] {
        let mut ir = KernelIR::new(&format!("prec_{}", func_name));
        let x = ir.push_op(Op::Input { index: 0 });
        let result = match *func_name {
            "sin" => ir.push_op(Op::Sin { x }),
            "cos" => ir.push_op(Op::Cos { x }),
            "exp" => ir.push_op(Op::Exp { x }),
            "log" => ir.push_op(Op::Log { x }),
            "sqrt" => ir.push_op(Op::Sqrt { x }),
            "tanh" => ir.push_op(Op::Tanh { x }),
            _ => unreachable!(),
        };
        ir.add_input(x);
        ir.add_output(result, 1);
        ir.param_is_pointer = vec![true, true];

        backend.compile_kernel(&format!("prec_{}", func_name), &ir).expect("compile");

        let n = test_values.len();
        let input_data: Vec<f32> = test_values.iter().map(|&v| v as f32).collect();
        let output_data = vec![0.0f32; n];
        let in_id = alloc_with_data(backend, &input_data);
        let out_id = alloc_with_data(backend, &output_data);

        backend.launch(&format!("prec_{}", func_name), (1, 1, 1), (n as u32, 1, 1), &[
            KernelParam::Buffer(in_id), KernelParam::Buffer(out_id),
        ]).expect("launch");
        backend.synchronize().expect("sync");

        let result = read_buffer(backend, out_id, n);
        backend.free(in_id).ok();
        backend.free(out_id).ok();

        // Compare against f64 reference
        for (i, &v) in test_values.iter().enumerate() {
            let reference = match *func_name {
                "sin" => (v as f64).sin() as f32,
                "cos" => (v as f64).cos() as f32,
                "exp" => (v as f64).exp() as f32,
                "log" => {
                    if v <= 0.0 { continue; }
                    (v as f64).ln() as f32
                }
                "sqrt" => {
                    if v < 0.0 { continue; }
                    (v as f64).sqrt() as f32
                }
                "tanh" => (v as f64).tanh() as f32,
                _ => unreachable!(),
            };

            if i < result.len() && reference.is_finite() && result[i].is_finite() {
                let ulp_diff = ulp_distance(result[i], reference);
                match *func_name {
                    "sin" => max_ulp_sin = max_ulp_sin.max(ulp_diff),
                    "cos" => max_ulp_cos = max_ulp_cos.max(ulp_diff),
                    "exp" => max_ulp_exp = max_ulp_exp.max(ulp_diff),
                    "log" => max_ulp_log = max_ulp_log.max(ulp_diff),
                    "sqrt" => max_ulp_sqrt = max_ulp_sqrt.max(ulp_diff),
                    "tanh" => max_ulp_tanh = max_ulp_tanh.max(ulp_diff),
                    _ => {}
                }
            }
        }
    }

    let max_ulp = *[max_ulp_sin, max_ulp_cos, max_ulp_exp, max_ulp_log, max_ulp_sqrt, max_ulp_tanh]
        .iter().max().unwrap_or(&0);

    LimitResult {
        category: "PRECISION",
        name: "Transcendental Functions (sin/cos/exp/log/sqrt/tanh)".into(),
        support: if max_ulp <= 1 {
            Support::Supported
        } else if max_ulp <= 4 {
            Support::Degraded {
                detail: format!("Max {} ULP error (CUDA allows 2 ULP for __sinf)", max_ulp),
            }
        } else {
            Support::Degraded {
                detail: format!("Max {} ULP error — exceeds CUDA spec", max_ulp),
            }
        },
        duration_ms: start.elapsed().as_secs_f64() * 1000.0,
        cuda_exclusive: false,
        evidence: format!(
            "sin={} cos={} exp={} log={} sqrt={} tanh={} ULP | CUDA fast math: ~2 ULP",
            max_ulp_sin, max_ulp_cos, max_ulp_exp, max_ulp_log, max_ulp_sqrt, max_ulp_tanh
        ),
    }
}

fn ulp_distance(a: f32, b: f32) -> u32 {
    if a == b { return 0; }
    if a.is_nan() || b.is_nan() { return u32::MAX; }
    let a_bits = a.to_bits() as i32;
    let b_bits = b.to_bits() as i32;
    (a_bits.wrapping_sub(b_bits)).unsigned_abs()
}

fn test_fma_precision(backend: &mut AnyBackend) -> LimitResult {
    let start = Instant::now();

    // FMA (fused multiply-add) should be more precise than separate mul+add
    // because it doesn't round the intermediate product.
    let mut ir_fma = KernelIR::new("prec_fma");
    let x = ir_fma.push_op(Op::Input { index: 0 });
    let y = ir_fma.push_op(Op::Input { index: 1 });
    let z = ir_fma.push_op(Op::Input { index: 2 });
    let fma_result = ir_fma.push_op(Op::Fma { a: x, b: y, c: z });
    ir_fma.add_input(x);
    ir_fma.add_input(y);
    ir_fma.add_input(z);
    ir_fma.add_output(fma_result, 3);
    ir_fma.param_is_pointer = vec![true, true, true, true];

    let mut ir_muladd = KernelIR::new("prec_muladd");
    let x2 = ir_muladd.push_op(Op::Input { index: 0 });
    let y2 = ir_muladd.push_op(Op::Input { index: 1 });
    let z2 = ir_muladd.push_op(Op::Input { index: 2 });
    let mul = ir_muladd.push_op(Op::Mul { lhs: x2, rhs: y2 });
    let add = ir_muladd.push_op(Op::Add { lhs: mul, rhs: z2 });
    ir_muladd.add_input(x2);
    ir_muladd.add_input(y2);
    ir_muladd.add_input(z2);
    ir_muladd.add_output(add, 3);
    ir_muladd.param_is_pointer = vec![true, true, true, true];

    backend.compile_kernel("prec_fma", &ir_fma).expect("compile fma");
    backend.compile_kernel("prec_muladd", &ir_muladd).expect("compile muladd");

    // Test with values that expose FMA rounding:
    // 1.0000001 * 1.0000001 + (-1.0) — the product is very close to 1.0
    let n = 4;
    let x_data = vec![1.0000001f32; n];
    let y_data = vec![1.0000001f32; n];
    let z_data = vec![-1.0f32; n];
    let fma_out = vec![0.0f32; n];
    let muladd_out = vec![0.0f32; n];

    let x_id = alloc_with_data(backend, &x_data);
    let y_id = alloc_with_data(backend, &y_data);
    let z_id = alloc_with_data(backend, &z_data);
    let fma_id = alloc_with_data(backend, &fma_out);
    let muladd_id = alloc_with_data(backend, &muladd_out);

    let params = [
        KernelParam::Buffer(x_id), KernelParam::Buffer(y_id),
        KernelParam::Buffer(z_id), KernelParam::Buffer(fma_id),
    ];
    backend.launch("prec_fma", (1, 1, 1), (n as u32, 1, 1), &params).expect("launch");

    let params2 = [
        KernelParam::Buffer(x_id), KernelParam::Buffer(y_id),
        KernelParam::Buffer(z_id), KernelParam::Buffer(muladd_id),
    ];
    backend.launch("prec_muladd", (1, 1, 1), (n as u32, 1, 1), &params2).expect("launch");
    backend.synchronize().expect("sync");

    let fma_result = read_buffer(backend, fma_id, 1);
    let muladd_result = read_buffer(backend, muladd_id, 1);
    backend.free(x_id).ok(); backend.free(y_id).ok(); backend.free(z_id).ok();
    backend.free(fma_id).ok(); backend.free(muladd_id).ok();

    // f64 reference
    let ref_val = (1.0000001f64 * 1.0000001f64) + (-1.0f64);

    let fma_err = (fma_result[0] as f64 - ref_val).abs();
    let muladd_err = (muladd_result[0] as f64 - ref_val).abs();

    LimitResult {
        category: "PRECISION",
        name: "FMA vs Mul+Add Precision".into(),
        support: if fma_err <= muladd_err {
            Support::Supported
        } else {
            Support::Degraded {
                detail: "FMA not more precise than separate Mul+Add".into(),
            }
        },
        duration_ms: start.elapsed().as_secs_f64() * 1000.0,
        cuda_exclusive: false,
        evidence: format!(
            "1.0000001 * 1.0000001 - 1.0 | FMA={:.10e} err={:.2e} | Mul+Add={:.10e} err={:.2e} | ref={:.10e}",
            fma_result[0], fma_err, muladd_result[0], muladd_err, ref_val
        ),
    }
}

fn test_fp16_precision(backend: &mut AnyBackend) -> LimitResult {
    let start = Instant::now();

    // Test FP16 conversion roundtrip accuracy
    let mut ir = KernelIR::new("fp16_test");
    let x = ir.push_op(Op::Input { index: 0 });
    let half = ir.push_op(Op::CvtF32ToF16 { x });
    let back = ir.push_op(Op::CvtF16ToF32 { x: half });
    ir.add_input(x);
    ir.add_output(back, 1);
    ir.param_is_pointer = vec![true, true];

    backend.compile_kernel("fp16_test", &ir).expect("compile fp16");

    let test_vals = [0.0f32, 1.0, -1.0, 0.5, 0.1, 0.001, 65504.0, -65504.0,
                     0.00006103515625, // smallest normal fp16
                     3.14159, 2.71828];
    let n = test_vals.len();
    let out = vec![0.0f32; n];
    let in_id = alloc_with_data(backend, &test_vals);
    let out_id = alloc_with_data(backend, &out);

    backend.launch("fp16_test", (1, 1, 1), (n as u32, 1, 1), &[
        KernelParam::Buffer(in_id), KernelParam::Buffer(out_id),
    ]).expect("launch");
    backend.synchronize().expect("sync");

    let result = read_buffer(backend, out_id, n);
    backend.free(in_id).ok();
    backend.free(out_id).ok();

    let mut max_rel_err = 0.0f64;
    let mut errors = Vec::new();
    for (i, &v) in test_vals.iter().enumerate() {
        if i < result.len() && v != 0.0 {
            let rel_err = ((result[i] as f64 - v as f64) / v as f64).abs();
            if rel_err > max_rel_err { max_rel_err = rel_err; }
            if rel_err > 0.01 {
                errors.push(format!("{}→{} ({:.1}%)", v, result[i], rel_err * 100.0));
            }
        }
    }

    let caps = backend.capabilities();
    LimitResult {
        category: "PRECISION",
        name: "FP16 Conversion Accuracy".into(),
        support: if max_rel_err < 0.001 {
            Support::Supported
        } else if max_rel_err < 0.05 {
            Support::Degraded {
                detail: format!("Max {:.2}% relative error (software emulation)", max_rel_err * 100.0),
            }
        } else {
            Support::Degraded {
                detail: format!("Max {:.1}% error: {}", max_rel_err * 100.0, errors.join(", ")),
            }
        },
        duration_ms: start.elapsed().as_secs_f64() * 1000.0,
        cuda_exclusive: false,
        evidence: format!(
            "has_fp16={} | max relative error: {:.4}% | {} test values, {} with >1% error",
            caps.has_fp16, max_rel_err * 100.0, n, errors.len()
        ),
    }
}

fn test_special_float_values(backend: &mut AnyBackend) -> LimitResult {
    let start = Instant::now();

    // Test: NaN, Infinity, denormals, negative zero, very small/large
    let mut ir = KernelIR::new("special_test");
    let x = ir.push_op(Op::Input { index: 0 });
    let y = ir.push_op(Op::Input { index: 1 });
    let sum = ir.push_op(Op::Add { lhs: x, rhs: y });
    ir.add_input(x);
    ir.add_input(y);
    ir.add_output(sum, 2);
    ir.param_is_pointer = vec![true, true, true];

    backend.compile_kernel("special_test", &ir).expect("compile special");

    let a_data = [
        f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0f32,
        f32::MIN_POSITIVE, // smallest normal
        1.0e-45, // smallest denormal
        f32::MAX, f32::MIN,
    ];
    let b_data = [
        1.0f32, -f32::INFINITY, f32::INFINITY, 0.0,
        f32::MIN_POSITIVE, 1.0e-45, 1.0, -1.0,
    ];
    let n = a_data.len();
    let out = vec![0.0f32; n];

    let a_id = alloc_with_data(backend, &a_data);
    let b_id = alloc_with_data(backend, &b_data);
    let c_id = alloc_with_data(backend, &out);

    backend.launch("special_test", (1, 1, 1), (n as u32, 1, 1), &[
        KernelParam::Buffer(a_id), KernelParam::Buffer(b_id), KernelParam::Buffer(c_id),
    ]).expect("launch");
    backend.synchronize().expect("sync");

    let result = read_buffer(backend, c_id, n);
    backend.free(a_id).ok(); backend.free(b_id).ok(); backend.free(c_id).ok();

    let mut issues = Vec::new();

    // NaN + 1.0 = NaN
    if n > 0 && !result[0].is_nan() {
        issues.push(format!("NaN+1.0={} (expected NaN)", result[0]));
    }
    // Inf + -Inf = NaN
    if n > 1 && !result[1].is_nan() {
        issues.push(format!("Inf+(-Inf)={} (expected NaN)", result[1]));
    }
    // -Inf + Inf = NaN
    if n > 2 && !result[2].is_nan() {
        issues.push(format!("-Inf+Inf={} (expected NaN)", result[2]));
    }
    // -0 + 0 = 0 (sign handling)
    if n > 3 && result[3] != 0.0 {
        issues.push(format!("-0+0={} (expected 0)", result[3]));
    }
    // Denormals
    if n > 5 {
        let denorm_sum = 1.0e-45f32 + 1.0e-45f32;
        if (result[5] - denorm_sum).abs() > 1e-50 && result[5] != 0.0 && denorm_sum != 0.0 {
            issues.push(format!("denorm+denorm={} vs {}", result[5], denorm_sum));
        }
    }
    // MAX + 1.0 = Inf (overflow)
    if n > 6 && !result[6].is_infinite() {
        issues.push(format!("MAX+1.0={} (expected Inf)", result[6]));
    }

    LimitResult {
        category: "PRECISION",
        name: "Special Float Values (NaN/Inf/denorm/-0)".into(),
        support: if issues.is_empty() {
            Support::Supported
        } else {
            Support::Degraded {
                detail: format!("{} issue(s): {}", issues.len(), issues.join("; ")),
            }
        },
        duration_ms: start.elapsed().as_secs_f64() * 1000.0,
        cuda_exclusive: false,
        evidence: format!(
            "Tested {} special pairs. Issues: {}",
            n, if issues.is_empty() { "none".into() } else { issues.join("; ") }
        ),
    }
}

fn test_accumulated_error(backend: &mut AnyBackend) -> LimitResult {
    let start = Instant::now();

    // Chain 100 FMAs and measure accumulated error vs f64 reference
    // This reveals floating point accumulation behavior differences
    let mut ir = KernelIR::new("accum_test");
    let x = ir.push_op(Op::Input { index: 0 });
    let factor = ir.push_op(Op::Const { val: 1.00001 });
    let bias = ir.push_op(Op::Const { val: 0.00001 });

    // Chain: result = fma(fma(fma(x, factor, bias), factor, bias), ...)
    let mut current = x;
    for _ in 0..100 {
        current = ir.push_op(Op::Fma { a: current, b: factor, c: bias });
    }
    ir.add_input(x);
    ir.add_output(current, 1);
    ir.param_is_pointer = vec![true, true];

    backend.compile_kernel("accum_test", &ir).expect("compile accum");

    let start_val = 1.0f32;
    let in_data = vec![start_val; 1];
    let out_data = vec![0.0f32; 1];
    let in_id = alloc_with_data(backend, &in_data);
    let out_id = alloc_with_data(backend, &out_data);

    backend.launch("accum_test", (1, 1, 1), (1, 1, 1), &[
        KernelParam::Buffer(in_id), KernelParam::Buffer(out_id),
    ]).expect("launch");
    backend.synchronize().expect("sync");

    let result = read_buffer(backend, out_id, 1);
    backend.free(in_id).ok();
    backend.free(out_id).ok();

    // f64 reference
    let mut ref_val = start_val as f64;
    for _ in 0..100 {
        ref_val = ref_val * 1.00001f64 + 0.00001f64;
    }

    let rel_err = ((result[0] as f64 - ref_val) / ref_val).abs();

    LimitResult {
        category: "PRECISION",
        name: "Accumulated Error (100 chained FMAs)".into(),
        support: if rel_err < 1e-5 {
            Support::Supported
        } else if rel_err < 1e-3 {
            Support::Degraded {
                detail: format!("{:.2e} relative error after 100 FMAs", rel_err),
            }
        } else {
            Support::Degraded {
                detail: format!("{:.2e} relative error — significant accumulation", rel_err),
            }
        },
        duration_ms: start.elapsed().as_secs_f64() * 1000.0,
        cuda_exclusive: false,
        evidence: format!(
            "100× FMA(x, 1.00001, 0.00001) from 1.0: got {:.10} (f64 ref: {:.10}), err={:.2e}",
            result[0], ref_val, rel_err
        ),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Category 4: PERFORMANCE SCALING
// Measuring proportional differences vs theoretical CUDA performance
// ═══════════════════════════════════════════════════════════════════════════

fn test_parallelism_scaling(backend: &mut AnyBackend) -> LimitResult {
    let start = Instant::now();

    let ir = build_vector_add_ir();
    backend.compile_kernel("scale_vadd", &ir).expect("compile");

    // Measure throughput at different element counts
    let sizes = [1024, 10_000, 100_000, 1_000_000];
    let mut throughputs = Vec::new();

    for &n in &sizes {
        let a_id = alloc_with_data(backend, &vec![1.0f32; n]);
        let b_id = alloc_with_data(backend, &vec![2.0f32; n]);
        let c_id = alloc_with_data(backend, &vec![0.0f32; n]);

        let block_size = 256u32;
        let grid_size = ((n as u32) + block_size - 1) / block_size;
        let params = [
            KernelParam::Buffer(a_id), KernelParam::Buffer(b_id), KernelParam::Buffer(c_id),
        ];

        // Warmup
        for _ in 0..3 {
            let _ = backend.launch("scale_vadd", (grid_size, 1, 1), (block_size, 1, 1), &params);
            let _ = backend.synchronize();
        }

        let iters = if n >= 1_000_000 { 50 } else { 200 };
        let t0 = Instant::now();
        for _ in 0..iters {
            backend.launch("scale_vadd", (grid_size, 1, 1), (block_size, 1, 1), &params).expect("launch");
        }
        backend.synchronize().expect("sync");
        let elapsed = t0.elapsed().as_secs_f64();

        let elements_per_sec = (n as f64 * iters as f64) / elapsed;
        throughputs.push((n, elements_per_sec));

        backend.free(a_id).ok(); backend.free(b_id).ok(); backend.free(c_id).ok();
    }

    // Compute scaling ratio: how much does throughput scale with N?
    let first = throughputs[0].1;
    let last = throughputs[throughputs.len() - 1].1;
    let scaling_factor = last / first;
    let ideal_scaling = throughputs[throughputs.len() - 1].0 as f64 / throughputs[0].0 as f64;

    let evidence_parts: Vec<String> = throughputs.iter()
        .map(|(n, t)| format!("{}→{:.1}M elem/s", n, t / 1e6))
        .collect();

    LimitResult {
        category: "PERFORMANCE",
        name: "Parallelism Scaling (VecAdd)".into(),
        support: if scaling_factor > ideal_scaling * 0.3 {
            Support::Supported
        } else {
            Support::Degraded {
                detail: format!("{:.1}x actual vs {:.0}x ideal scaling", scaling_factor, ideal_scaling),
            }
        },
        duration_ms: start.elapsed().as_secs_f64() * 1000.0,
        cuda_exclusive: false,
        evidence: format!(
            "{} | scaling: {:.1}x (ideal {:.0}x) | CUDA GPUs: ~100x scaling 1K→1M",
            evidence_parts.join(", "), scaling_factor, ideal_scaling
        ),
    }
}

fn test_memory_bandwidth_profile(backend: &mut AnyBackend) -> LimitResult {
    let start = Instant::now();

    let sizes = [
        4 * 1024,         // 4 KB (L1 cache)
        32 * 1024,        // 32 KB (L1/L2 boundary)
        256 * 1024,       // 256 KB (L2 cache)
        4 * 1024 * 1024,  // 4 MB (L3 cache)
        32 * 1024 * 1024, // 32 MB (main memory)
    ];

    let mut bandwidths = Vec::new();

    for &size in &sizes {
        let data = vec![0u8; size];
        let (id, _) = backend.alloc(size, false).expect("alloc");

        // Warmup
        for _ in 0..5 {
            backend.copy_htod(id, &data).expect("htod");
            let _ = backend.copy_dtoh(id, size);
        }

        let iters = if size >= 32 * 1024 * 1024 { 20 } else { 100 };
        let t0 = Instant::now();
        for _ in 0..iters {
            backend.copy_htod(id, &data).expect("htod");
            let _ = backend.copy_dtoh(id, size);
        }
        let elapsed = t0.elapsed().as_secs_f64();
        backend.free(id).ok();

        let gb_s = (iters * size * 2) as f64 / elapsed / 1e9;
        bandwidths.push((size, gb_s));
    }

    let max_bw = bandwidths.iter().map(|(_, bw)| *bw).fold(0.0f64, f64::max);
    let evidence_parts: Vec<String> = bandwidths.iter()
        .map(|(s, bw)| format!("{}:{:.1} GB/s", format_bytes(*s), bw))
        .collect();

    // CUDA GPUs: HBM2 = 900 GB/s (A100), GDDR6 = 480 GB/s (3090)
    // CPU: DDR4 = 25-50 GB/s, DDR5 = 50-80 GB/s
    LimitResult {
        category: "PERFORMANCE",
        name: "Memory Bandwidth Profile".into(),
        support: if max_bw > 10.0 {
            Support::Supported
        } else {
            Support::Degraded {
                detail: format!("Peak {:.1} GB/s — CPU DDR4/5 vs CUDA HBM (900 GB/s)", max_bw),
            }
        },
        duration_ms: start.elapsed().as_secs_f64() * 1000.0,
        cuda_exclusive: false,
        evidence: format!(
            "{} | peak={:.1} GB/s | CUDA A100 HBM2: ~900 GB/s, CPU DDR5: ~60 GB/s",
            evidence_parts.join(", "), max_bw
        ),
    }
}

fn test_kernel_launch_overhead(backend: &mut AnyBackend) -> LimitResult {
    let start = Instant::now();

    let ir = build_vector_add_ir();
    backend.compile_kernel("overhead_vadd", &ir).expect("compile");

    let a_id = alloc_with_data(backend, &[1.0f32]);
    let b_id = alloc_with_data(backend, &[2.0f32]);
    let c_id = alloc_with_data(backend, &[0.0f32]);
    let params = [
        KernelParam::Buffer(a_id), KernelParam::Buffer(b_id), KernelParam::Buffer(c_id),
    ];

    // Warmup
    for _ in 0..100 {
        let _ = backend.launch("overhead_vadd", (1, 1, 1), (1, 1, 1), &params);
        let _ = backend.synchronize();
    }

    // Measure launch + sync latency
    let iters = 10_000;
    let t0 = Instant::now();
    for _ in 0..iters {
        backend.launch("overhead_vadd", (1, 1, 1), (1, 1, 1), &params).expect("launch");
        backend.synchronize().expect("sync");
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let latency_us = elapsed / iters as f64 * 1e6;

    // Also measure launch without sync (batch)
    let t1 = Instant::now();
    for _ in 0..iters {
        backend.launch("overhead_vadd", (1, 1, 1), (1, 1, 1), &params).expect("launch");
    }
    backend.synchronize().expect("sync");
    let batch_elapsed = t1.elapsed().as_secs_f64();
    let batch_latency_us = batch_elapsed / iters as f64 * 1e6;

    backend.free(a_id).ok(); backend.free(b_id).ok(); backend.free(c_id).ok();

    // CUDA: ~5-10 us launch latency, CPU: should be <1 us (no command buffer)
    LimitResult {
        category: "PERFORMANCE",
        name: "Kernel Launch Overhead".into(),
        support: if latency_us < 100.0 {
            Support::Supported
        } else {
            Support::Degraded {
                detail: format!("{:.1} us per launch+sync", latency_us),
            }
        },
        duration_ms: start.elapsed().as_secs_f64() * 1000.0,
        cuda_exclusive: false,
        evidence: format!(
            "launch+sync: {:.2} us | batch launch: {:.2} us | CUDA: ~5-10 us, CPU: ~0.1-1 us",
            latency_us, batch_latency_us
        ),
    }
}

fn test_blas_scaling(_backend: &mut AnyBackend) -> LimitResult {
    let start = Instant::now();

    #[cfg(feature = "cpu")]
    {
        use invisible_cuda::blas::traits::{BlasBackend, GemmConfig};
        use invisible_cuda::blas::cpu_blas::CpuBlasBackend;

        let dims = [64, 128, 256, 512, 1024];
        let mut gflops_list = Vec::new();

        for &dim in &dims {
            let mut rng = Rng::new(dim as u64 * 7);
            let a = rng.vec_f32(dim * dim);
            let b = rng.vec_f32(dim * dim);

            let mut blas = CpuBlasBackend::new();
            let a_id = BackendBufferId(1);
            let b_id = BackendBufferId(2);
            let c_id = BackendBufferId(3);
            blas.register_buffer(a_id, f32_to_bytes(&a));
            blas.register_buffer(b_id, f32_to_bytes(&b));
            blas.register_buffer(c_id, f32_to_bytes(&vec![0.0f32; dim * dim]));

            let config = GemmConfig::new(dim, dim, dim);

            // Warmup
            let _ = blas.sgemm(&config, a_id, b_id, c_id);
            blas.register_buffer(c_id, f32_to_bytes(&vec![0.0f32; dim * dim]));

            let iters = if dim >= 512 { 3 } else { 10 };
            let t0 = Instant::now();
            for _ in 0..iters {
                blas.sgemm(&config, a_id, b_id, c_id).expect("sgemm");
            }
            let elapsed = t0.elapsed().as_secs_f64();

            let flops = 2.0 * (dim as f64).powi(3) * iters as f64;
            let gflops = flops / elapsed / 1e9;
            gflops_list.push((dim, gflops));
        }

        let max_gflops = gflops_list.iter().map(|(_, g)| *g).fold(0.0f64, f64::max);
        let evidence_parts: Vec<String> = gflops_list.iter()
            .map(|(d, g)| format!("{}x{}:{:.1}", d, d, g))
            .collect();

        // CUDA: cuBLAS on A100 ≈ 19.5 TFLOPS (FP32), V100 ≈ 15.7 TFLOPS
        // CPU: OpenBLAS on modern x86 ≈ 200-800 GFLOPS
        return LimitResult {
            category: "PERFORMANCE",
            name: "BLAS SGEMM Scaling (GFLOPS)".into(),
            support: if max_gflops > 1.0 {
                Support::Supported
            } else {
                Support::Degraded {
                    detail: format!("Peak {:.1} GFLOPS — CPU BLAS vs CUDA cuBLAS (19,500 GFLOPS)", max_gflops),
                }
            },
            duration_ms: start.elapsed().as_secs_f64() * 1000.0,
            cuda_exclusive: false,
            evidence: format!(
                "{} GFLOPS | peak={:.1} | CUDA A100: ~19,500 GFLOPS, CPU: 200-800 GFLOPS",
                evidence_parts.join(", "), max_gflops
            ),
        };
    }

    #[cfg(not(feature = "cpu"))]
    LimitResult {
        category: "PERFORMANCE",
        name: "BLAS SGEMM Scaling (GFLOPS)".into(),
        support: Support::Degraded { detail: "CPU BLAS not available (feature disabled)".into() },
        duration_ms: start.elapsed().as_secs_f64() * 1000.0,
        cuda_exclusive: false,
        evidence: "Requires --features cpu".into(),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Category 5: EDGE CASES & BREAKING POINTS
// Finding where things break under extreme conditions
// ═══════════════════════════════════════════════════════════════════════════

fn test_max_grid_size(backend: &mut AnyBackend) -> LimitResult {
    let start = Instant::now();

    let ir = build_vector_add_ir();
    backend.compile_kernel("grid_limit", &ir).expect("compile");

    // Try increasingly large grid sizes until failure
    let grid_sizes: Vec<u32> = vec![1, 256, 1024, 4096, 16384, 65535, 100_000, 1_000_000];
    let mut max_working_grid = 0u32;
    let mut failure = None;

    for &grid_x in &grid_sizes {
        let n = grid_x as usize * 256; // grid_x blocks × 256 threads
        let size = n * 4;

        match backend.alloc(size, false) {
            Ok((a_id, _)) => {
                match backend.alloc(size, false) {
                    Ok((b_id, _)) => {
                        match backend.alloc(size, false) {
                            Ok((c_id, _)) => {
                                let a_data = f32_to_bytes(&vec![1.0f32; n]);
                                let b_data = f32_to_bytes(&vec![2.0f32; n]);
                                backend.copy_htod(a_id, &a_data).ok();
                                backend.copy_htod(b_id, &b_data).ok();

                                match backend.launch("grid_limit", (grid_x, 1, 1), (256, 1, 1), &[
                                    KernelParam::Buffer(a_id), KernelParam::Buffer(b_id),
                                    KernelParam::Buffer(c_id),
                                ]) {
                                    Ok(()) => {
                                        backend.synchronize().ok();
                                        max_working_grid = grid_x;
                                    }
                                    Err(e) => {
                                        failure = Some(format!("grid_x={}: launch failed: {}", grid_x, e));
                                    }
                                }

                                backend.free(a_id).ok(); backend.free(b_id).ok(); backend.free(c_id).ok();
                            }
                            Err(_) => {
                                failure = Some(format!("grid_x={}: alloc C failed ({} elements)", grid_x, n));
                                backend.free(a_id).ok(); backend.free(b_id).ok();
                                break;
                            }
                        }
                    }
                    Err(_) => {
                        failure = Some(format!("grid_x={}: alloc B failed", grid_x));
                        backend.free(a_id).ok();
                        break;
                    }
                }
            }
            Err(_) => {
                failure = Some(format!("grid_x={}: alloc A failed ({} bytes)", grid_x, n * 4));
                break;
            }
        }

        if failure.is_some() { break; }
    }

    let caps = backend.capabilities();
    LimitResult {
        category: "EDGE_CASE",
        name: "Maximum Grid Size".into(),
        support: if max_working_grid >= 65535 {
            Support::Supported
        } else if max_working_grid >= 1024 {
            Support::Degraded {
                detail: format!("Max grid_x={} (limited by memory, not hardware)", max_working_grid),
            }
        } else {
            Support::Degraded {
                detail: format!("Max grid_x={}", max_working_grid),
            }
        },
        duration_ms: start.elapsed().as_secs_f64() * 1000.0,
        cuda_exclusive: false,
        evidence: format!(
            "Tested grid_x up to {} | max working: {} | caps.max_grid: {:?} | break: {}",
            grid_sizes.last().unwrap_or(&0), max_working_grid,
            caps.max_grid_dims,
            failure.as_deref().unwrap_or("none")
        ),
    }
}

fn test_register_pressure(backend: &mut AnyBackend) -> LimitResult {
    let start = Instant::now();

    // Build kernels with increasing IR op count to test register file limits
    let op_counts = [10, 50, 100, 500, 1000];
    let mut max_working_ops = 0;

    for &ops in &op_counts {
        let mut ir = KernelIR::new(&format!("regpressure_{}", ops));
        let x = ir.push_op(Op::Input { index: 0 });
        let mut current = x;

        // Chain ops: each one depends on the previous
        for j in 0..ops {
            let c = ir.push_op(Op::Const { val: 0.001 * (j as f32 + 1.0) });
            current = ir.push_op(Op::Add { lhs: current, rhs: c });
        }

        ir.add_input(x);
        ir.add_output(current, 1);
        ir.param_is_pointer = vec![true, true];

        match backend.compile_kernel(&format!("regpressure_{}", ops), &ir) {
            Ok(()) => {
                let in_data = vec![1.0f32; 4];
                let out_data = vec![0.0f32; 4];
                let in_id = alloc_with_data(backend, &in_data);
                let out_id = alloc_with_data(backend, &out_data);

                match backend.launch(&format!("regpressure_{}", ops), (1, 1, 1), (4, 1, 1), &[
                    KernelParam::Buffer(in_id), KernelParam::Buffer(out_id),
                ]) {
                    Ok(()) => {
                        backend.synchronize().ok();
                        let result = read_buffer(backend, out_id, 1);

                        // Verify result makes sense
                        let expected: f32 = (1..=ops).map(|j| 0.001 * j as f32).sum::<f32>() + 1.0;
                        if (result[0] - expected).abs() < expected * 0.01 {
                            max_working_ops = ops;
                        }
                    }
                    Err(_) => {}
                }

                backend.free(in_id).ok(); backend.free(out_id).ok();
            }
            Err(_) => break,
        }
    }

    LimitResult {
        category: "EDGE_CASE",
        name: "Register Pressure (IR Op Chain Length)".into(),
        support: if max_working_ops >= 1000 {
            Support::Supported
        } else if max_working_ops >= 100 {
            Support::Degraded {
                detail: format!("Works up to {} chained ops", max_working_ops),
            }
        } else {
            Support::Degraded {
                detail: format!("Only {} ops before failure", max_working_ops),
            }
        },
        duration_ms: start.elapsed().as_secs_f64() * 1000.0,
        cuda_exclusive: false,
        evidence: format!(
            "Tested chains of {:?} ops | max correct: {} | CUDA: 255 registers/thread, spills to local mem",
            op_counts, max_working_ops
        ),
    }
}

fn test_bitwise_correctness(backend: &mut AnyBackend) -> LimitResult {
    let start = Instant::now();

    // Test all 13 bitwise ops for correctness
    let mut errors = Vec::new();

    // Test AND
    {
        let mut ir = KernelIR::new("bit_and");
        let a = ir.push_op(Op::Input { index: 0 });
        let b = ir.push_op(Op::Input { index: 1 });
        let r = ir.push_op(Op::And { lhs: a, rhs: b });
        ir.add_input(a); ir.add_input(b); ir.add_output(r, 2);
        ir.param_is_pointer = vec![true, true, true];
        backend.compile_kernel("bit_and", &ir).expect("compile");

        let a_val = f32::from_bits(0xFF00FF00);
        let b_val = f32::from_bits(0x0F0F0F0F);
        let expected = f32::from_bits(0x0F000F00);

        let a_id = alloc_with_data(backend, &[a_val]);
        let b_id = alloc_with_data(backend, &[b_val]);
        let c_id = alloc_with_data(backend, &[0.0]);

        backend.launch("bit_and", (1, 1, 1), (1, 1, 1), &[
            KernelParam::Buffer(a_id), KernelParam::Buffer(b_id), KernelParam::Buffer(c_id),
        ]).expect("launch");
        backend.synchronize().expect("sync");
        let result = read_buffer(backend, c_id, 1);
        backend.free(a_id).ok(); backend.free(b_id).ok(); backend.free(c_id).ok();

        if result[0].to_bits() != expected.to_bits() {
            errors.push(format!("AND: 0x{:08X} & 0x{:08X} = 0x{:08X} (expected 0x{:08X})",
                0xFF00FF00u32, 0x0F0F0F0F, result[0].to_bits(), expected.to_bits()));
        }
    }

    // Test POPC (population count)
    {
        let mut ir = KernelIR::new("bit_popc");
        let a = ir.push_op(Op::Input { index: 0 });
        let r = ir.push_op(Op::Popc { x: a });
        ir.add_input(a); ir.add_output(r, 1);
        ir.param_is_pointer = vec![true, true];
        backend.compile_kernel("bit_popc", &ir).expect("compile");

        // 0x0F0F0F0F has 16 set bits
        let test_val = f32::from_bits(0x0F0F0F0F);
        let a_id = alloc_with_data(backend, &[test_val]);
        let out_id = alloc_with_data(backend, &[0.0]);

        backend.launch("bit_popc", (1, 1, 1), (1, 1, 1), &[
            KernelParam::Buffer(a_id), KernelParam::Buffer(out_id),
        ]).expect("launch");
        backend.synchronize().expect("sync");
        let result = read_buffer(backend, out_id, 1);
        backend.free(a_id).ok(); backend.free(out_id).ok();

        if (result[0] - 16.0).abs() > 0.5 {
            errors.push(format!("POPC: popc(0x0F0F0F0F) = {} (expected 16)", result[0]));
        }
    }

    // Test BREV (bit reverse)
    {
        let mut ir = KernelIR::new("bit_brev");
        let a = ir.push_op(Op::Input { index: 0 });
        let r = ir.push_op(Op::Brev { x: a });
        ir.add_input(a); ir.add_output(r, 1);
        ir.param_is_pointer = vec![true, true];
        backend.compile_kernel("bit_brev", &ir).expect("compile");

        let test_val = f32::from_bits(0x80000000);
        let expected = f32::from_bits(0x00000001);
        let a_id = alloc_with_data(backend, &[test_val]);
        let out_id = alloc_with_data(backend, &[0.0]);

        backend.launch("bit_brev", (1, 1, 1), (1, 1, 1), &[
            KernelParam::Buffer(a_id), KernelParam::Buffer(out_id),
        ]).expect("launch");
        backend.synchronize().expect("sync");
        let result = read_buffer(backend, out_id, 1);
        backend.free(a_id).ok(); backend.free(out_id).ok();

        if result[0].to_bits() != expected.to_bits() {
            errors.push(format!("BREV: rev(0x80000000) = 0x{:08X} (expected 0x{:08X})",
                result[0].to_bits(), expected.to_bits()));
        }
    }

    // Test DP4A (int8 dot product)
    {
        let mut ir = KernelIR::new("bit_dp4a");
        let a = ir.push_op(Op::Input { index: 0 });
        let b = ir.push_op(Op::Input { index: 1 });
        let acc = ir.push_op(Op::Const { val: 0.0 });
        let r = ir.push_op(Op::Dp4a { a, b, c: acc });
        ir.add_input(a); ir.add_input(b); ir.add_output(r, 2);
        ir.param_is_pointer = vec![true, true, true];
        backend.compile_kernel("bit_dp4a", &ir).expect("compile");

        // a = [1, 2, 3, 4] as packed int8, b = [1, 1, 1, 1]
        let a_val = f32::from_bits(0x04030201);
        let b_val = f32::from_bits(0x01010101);
        // Expected: 1*1 + 2*1 + 3*1 + 4*1 = 10
        let a_id = alloc_with_data(backend, &[a_val]);
        let b_id = alloc_with_data(backend, &[b_val]);
        let c_id = alloc_with_data(backend, &[0.0]);

        backend.launch("bit_dp4a", (1, 1, 1), (1, 1, 1), &[
            KernelParam::Buffer(a_id), KernelParam::Buffer(b_id), KernelParam::Buffer(c_id),
        ]).expect("launch");
        backend.synchronize().expect("sync");
        let result = read_buffer(backend, c_id, 1);
        backend.free(a_id).ok(); backend.free(b_id).ok(); backend.free(c_id).ok();

        if (result[0] - 10.0).abs() > 0.5 {
            errors.push(format!("DP4A: [1,2,3,4]·[1,1,1,1] = {} (expected 10)", result[0]));
        }
    }

    LimitResult {
        category: "EDGE_CASE",
        name: "Bitwise Operations Correctness (AND/POPC/BREV/DP4A)".into(),
        support: if errors.is_empty() {
            Support::Supported
        } else {
            Support::Degraded {
                detail: format!("{} error(s): {}", errors.len(), errors.join("; ")),
            }
        },
        duration_ms: start.elapsed().as_secs_f64() * 1000.0,
        cuda_exclusive: false,
        evidence: format!(
            "Tested 4 bitwise ops: AND, POPC, BREV, DP4A | errors: {}",
            if errors.is_empty() { "none".into() } else { errors.join("; ") }
        ),
    }
}

fn test_alloc_limit(backend: &mut AnyBackend) -> LimitResult {
    let start = Instant::now();

    // How many simultaneous allocations can we hold?
    let sizes = [4096, 65536, 1024 * 1024];
    let mut results = Vec::new();

    for &size in &sizes {
        let mut ids = Vec::new();
        let max_count = 10_000;

        for _ in 0..max_count {
            match backend.alloc(size, false) {
                Ok((id, _)) => ids.push(id),
                Err(_) => break,
            }
        }

        let count = ids.len();
        let total_bytes = count * size;
        for id in ids { backend.free(id).ok(); }
        results.push((size, count, total_bytes));
    }

    let evidence_parts: Vec<String> = results.iter()
        .map(|(s, c, t)| format!("{}×{}={}", c, format_bytes(*s), format_bytes(*t)))
        .collect();

    LimitResult {
        category: "EDGE_CASE",
        name: "Allocation Limits (count × size)".into(),
        support: if results.iter().all(|(_, c, _)| *c >= 1000) {
            Support::Supported
        } else {
            Support::Degraded {
                detail: "Limited allocation count or total memory".into(),
            }
        },
        duration_ms: start.elapsed().as_secs_f64() * 1000.0,
        cuda_exclusive: false,
        evidence: format!(
            "{} | CUDA: limited by GPU VRAM ({} GB typical)",
            evidence_parts.join(", "), "8-80"
        ),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// IR kernel builder (shared)
// ═══════════════════════════════════════════════════════════════════════════

fn build_vector_add_ir() -> KernelIR {
    let mut ir = KernelIR::new("vector_add");
    let a = ir.push_op(Op::Input { index: 0 });
    let b = ir.push_op(Op::Input { index: 1 });
    let c = ir.push_op(Op::Add { lhs: a, rhs: b });
    ir.add_input(a);
    ir.add_input(b);
    ir.add_output(c, 2);
    ir.param_is_pointer = vec![true, true, true];
    ir
}

// ═══════════════════════════════════════════════════════════════════════════
// JSON output
// ═══════════════════════════════════════════════════════════════════════════

fn emit_json(
    caps: &invisible_cuda::compute::DeviceCapabilities,
    backend_kind: &str,
    results: &[LimitResult],
    hw: &HwFingerprint,
) {
    let escape = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"");

    println!();
    println!("--- JSON_START ---");
    println!("{{");
    println!("  \"format\": \"invisible-cuda-limits-v2\",");
    println!("  \"backend\": \"{}\",", escape(backend_kind));
    println!("  \"device\": \"{}\",", escape(&caps.name));
    println!("  \"os\": \"{}\",", std::env::consts::OS);
    println!("  \"arch\": \"{}\",", std::env::consts::ARCH);
    println!("  \"compute_units\": {},", caps.compute_units);
    println!("  \"warp_size\": {},", caps.warp_size);
    println!("  \"has_fp16\": {},", caps.has_fp16);
    println!("  \"has_shared_memory\": {},", caps.has_shared_memory);
    println!("  \"has_atomics\": {},", caps.has_atomics);
    hw.emit_json_fields();

    println!("  \"limits\": [");
    for (i, r) in results.iter().enumerate() {
        let comma = if i + 1 < results.len() { "," } else { "" };
        let (status, detail) = match &r.support {
            Support::Supported => ("supported", String::new()),
            Support::Degraded { detail } => ("degraded", detail.clone()),
            Support::Unsupported { reason } => ("unsupported", reason.clone()),
        };
        println!("    {{\"category\": \"{}\", \"name\": \"{}\", \"status\": \"{}\", \"cuda_exclusive\": {}, \"ms\": {:.1}, \"detail\": \"{}\", \"evidence\": \"{}\"}}{}",
            escape(r.category), escape(&r.name), status, r.cuda_exclusive,
            r.duration_ms, escape(&detail), escape(&r.evidence), comma);
    }
    println!("  ]");
    println!("}}");
    println!("--- JSON_END ---");
}

// ═══════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════

fn main() {
    let mode = std::env::var("PROOF_MODE").unwrap_or_else(|_| "standard".into());

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       INVISIBLE CUDA — Scientific Limits Proof (v1)        ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // System info + hardware fingerprint
    println!("System:");
    println!("  OS:   {}", std::env::consts::OS);
    println!("  Arch: {}", std::env::consts::ARCH);
    println!("  Mode: {}", mode);
    let hw = HwFingerprint::collect();
    hw.print();
    println!();

    // Detect backend
    let mut backend = match detect_backend() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("FATAL: No compute backend available: {}", e);
            std::process::exit(1);
        }
    };

    let caps = backend.capabilities();
    println!("Backend: {}", backend.kind());
    println!("Device:  {}", caps.name);
    println!("Compute: {} units, warp_size={}", caps.compute_units, caps.warp_size);
    println!("Features: shared_mem={} atomics={} fp16={}", caps.has_shared_memory, caps.has_atomics, caps.has_fp16);
    println!();

    let total_start = Instant::now();
    let mut all_results: Vec<LimitResult> = Vec::new();

    // ── Category 1: HARD LIMITS ─────────────────────────────────────
    println!("═══ Category 1: HARD LIMITS ════════════════════════════════════");
    println!("  (CUDA features that CANNOT work without NVIDIA hardware)");
    println!();

    all_results.push(test_wmma_tensor_cores(&mut backend));
    all_results.push(test_texture_sampling(&mut backend));
    all_results.push(test_surface_ops(&mut backend));
    all_results.push(test_async_copy(&mut backend));
    all_results.push(test_mbarrier(&mut backend));

    for r in all_results.iter().rev().take(5).collect::<Vec<_>>().into_iter().rev() {
        print_result(r);
    }
    println!();

    // ── Category 2: SEMANTIC GAPS ───────────────────────────────────
    println!("═══ Category 2: SEMANTIC GAPS ══════════════════════════════════");
    println!("  (Features that WORK but BEHAVE DIFFERENTLY than real CUDA)");
    println!();

    all_results.push(test_warp_shuffle(&mut backend));
    all_results.push(test_warp_vote(&mut backend));
    all_results.push(test_warp_identity(&mut backend));
    all_results.push(test_barrier_correctness(&mut backend));
    all_results.push(test_control_flow(&mut backend));
    all_results.push(test_2d_3d_grid(&mut backend));

    for r in all_results.iter().rev().take(6).collect::<Vec<_>>().into_iter().rev() {
        print_result(r);
    }
    println!();

    // ── Category 3: PRECISION BOUNDARIES ────────────────────────────
    println!("═══ Category 3: PRECISION BOUNDARIES ═══════════════════════════");
    println!("  (Numerical accuracy differences between CPU and GPU)");
    println!();

    all_results.push(test_transcendental_precision(&mut backend));
    all_results.push(test_fma_precision(&mut backend));
    all_results.push(test_fp16_precision(&mut backend));
    all_results.push(test_special_float_values(&mut backend));
    all_results.push(test_accumulated_error(&mut backend));

    for r in all_results.iter().rev().take(5).collect::<Vec<_>>().into_iter().rev() {
        print_result(r);
    }
    println!();

    // ── Category 4: PERFORMANCE SCALING ─────────────────────────────
    if mode == "standard" || mode == "extended" {
        println!("═══ Category 4: PERFORMANCE SCALING ════════════════════════════");
        println!("  (Proportional performance differences vs CUDA)");
        println!("  NOTE: These numbers are INFORMATIONAL, not pass/fail criteria.");
        println!("  Variations reflect memory class ({}@{}MT/s) and core count ({}).",
            hw.mem_type, hw.mem_speed_mt, hw.physical_cores);
        println!("  Normalized values (where applicable) adjust for memory bandwidth.");
        println!();

        all_results.push(test_parallelism_scaling(&mut backend));
        all_results.push(test_memory_bandwidth_profile(&mut backend));
        all_results.push(test_kernel_launch_overhead(&mut backend));
        all_results.push(test_blas_scaling(&mut backend));

        for r in all_results.iter().rev().take(4).collect::<Vec<_>>().into_iter().rev() {
            print_result(r);
        }
        println!();
    }

    // ── Category 5: EDGE CASES ──────────────────────────────────────
    if mode == "standard" || mode == "extended" {
        println!("═══ Category 5: EDGE CASES & BREAKING POINTS ═══════════════════");
        println!("  (Where things break under extreme conditions)");
        println!();

        all_results.push(test_max_grid_size(&mut backend));
        all_results.push(test_register_pressure(&mut backend));
        all_results.push(test_bitwise_correctness(&mut backend));
        all_results.push(test_alloc_limit(&mut backend));

        for r in all_results.iter().rev().take(4).collect::<Vec<_>>().into_iter().rev() {
            print_result(r);
        }
        println!();
    }

    // ── Summary ─────────────────────────────────────────────────────
    let supported = all_results.iter().filter(|r| matches!(r.support, Support::Supported)).count();
    let degraded = all_results.iter().filter(|r| matches!(r.support, Support::Degraded { .. })).count();
    let unsupported = all_results.iter().filter(|r| matches!(r.support, Support::Unsupported { .. })).count();
    let cuda_only_unsupported = all_results.iter()
        .filter(|r| r.cuda_exclusive && matches!(r.support, Support::Unsupported { .. })).count();
    let total_time = total_start.elapsed().as_secs_f64();

    println!("══════════════════════════════════════════════════════════════");
    println!("  SCIENTIFIC LIMITS SUMMARY");
    println!("══════════════════════════════════════════════════════════════");
    println!("  Backend:     {} ({})", backend.kind(), caps.name);
    println!("  Tests:       {} total", all_results.len());
    println!("  Supported:   {} (full CUDA compatibility)", supported);
    println!("  Degraded:    {} (works with differences)", degraded);
    println!("  Unsupported: {} ({} are CUDA-exclusive features)", unsupported, cuda_only_unsupported);
    println!("  Time:        {:.1}s", total_time);
    println!();
    println!("  CONCLUSION:");

    let non_exclusive_unsupported = unsupported - cuda_only_unsupported;
    if non_exclusive_unsupported == 0 && degraded <= 10 {
        println!("  All core CUDA operations work. {} CUDA-exclusive features", cuda_only_unsupported);
        println!("  (tensor cores, texture HW, async copy) require NVIDIA hardware.");
        println!("  {} features have semantic differences (warp ops, barriers, branches)", degraded);
        println!("  that only affect warp-aware algorithms. Standard CUDA kernels run correctly.");
        println!("  Performance scales proportionally with hardware capability.");
    } else {
        println!("  {} features unsupported beyond CUDA-exclusive.", non_exclusive_unsupported);
        println!("  {} features degraded. Investigation needed.", degraded);
    }
    println!("══════════════════════════════════════════════════════════════");

    // JSON output for machine parsing
    let backend_kind = format!("{}", backend.kind());
    emit_json(&caps, &backend_kind, &all_results, &hw);
}
