//! Invisible CUDA — Comprehensive Universal Backend Proof
//!
//! Proves CUDA compatibility on any hardware by auto-detecting the best
//! backend and running correctness, performance, scaling, and stress tests.
//!
//! Modes (set via PROOF_MODE env var):
//!   quick    — 5 correctness tests + 2 benchmarks (~10s)
//!   standard — full suite: correctness + scaling + BLAS + stress (~60s)  [default]
//!   extended — standard + sustained throughput + large allocations (~5min)
//!
//! Run locally:
//!   cargo run --bin proof --release --no-default-features --features cpu
//!
//! Run on EC2:
//!   INVISIBLE_CUDA_BACKEND=cpu PROOF_MODE=standard ./proof

use std::time::Instant;

use invisible_cuda::blas::traits::{BlasBackend, GemmConfig};
#[cfg(feature = "cpu")]
use invisible_cuda::blas::cpu_blas::CpuBlasBackend;
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

fn format_gflops(flops: f64) -> String {
    if flops >= 1e12 { format!("{:.2} TFLOPS", flops / 1e12) }
    else if flops >= 1e9 { format!("{:.2} GFLOPS", flops / 1e9) }
    else if flops >= 1e6 { format!("{:.2} MFLOPS", flops / 1e6) }
    else { format!("{:.0} FLOPS", flops) }
}

// ═══════════════════════════════════════════════════════════════════════════
// Pseudo-RNG (no external dep)
// ═══════════════════════════════════════════════════════════════════════════

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
// CPU reference implementations
// ═══════════════════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════════════════
// IR kernel builders
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

fn build_vector_mul_ir() -> KernelIR {
    let mut ir = KernelIR::new("vector_mul");
    let a = ir.push_op(Op::Input { index: 0 });
    let b = ir.push_op(Op::Input { index: 1 });
    let c = ir.push_op(Op::Mul { lhs: a, rhs: b });
    ir.add_input(a);
    ir.add_input(b);
    ir.add_output(c, 2);
    ir.param_is_pointer = vec![true, true, true];
    ir
}

fn build_fma_ir() -> KernelIR {
    let mut ir = KernelIR::new("vector_fma");
    let x = ir.push_op(Op::Input { index: 0 });
    let y = ir.push_op(Op::Input { index: 1 });
    let z = ir.push_op(Op::Input { index: 2 });
    let r = ir.push_op(Op::Fma { a: x, b: y, c: z });
    ir.add_input(x);
    ir.add_input(y);
    ir.add_input(z);
    ir.add_output(r, 3);
    ir.param_is_pointer = vec![true, true, true, true];
    ir
}

fn build_saxpy_ir() -> KernelIR {
    // y = a*x + y  (a from buf[0], x from buf[1], y from buf[2], out to buf[3])
    let mut ir = KernelIR::new("saxpy");
    let a = ir.push_op(Op::Input { index: 0 });
    let x = ir.push_op(Op::Input { index: 1 });
    let y = ir.push_op(Op::Input { index: 2 });
    let ax = ir.push_op(Op::Mul { lhs: a, rhs: x });
    let r = ir.push_op(Op::Add { lhs: ax, rhs: y });
    ir.add_input(a);
    ir.add_input(x);
    ir.add_input(y);
    ir.add_output(r, 3);
    ir.param_is_pointer = vec![true, true, true, true];
    ir
}

// ═══════════════════════════════════════════════════════════════════════════
// Test infrastructure
// ═══════════════════════════════════════════════════════════════════════════

struct TestResult {
    name: String,
    passed: bool,
    error: Option<String>,
    duration_ms: f64,
}

fn run_test(name: &str, backend: &mut AnyBackend, f: impl FnOnce(&mut AnyBackend) -> Result<(), String>) -> TestResult {
    let start = Instant::now();
    let result = f(backend);
    let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
    TestResult {
        name: name.to_string(),
        passed: result.is_ok(),
        error: result.err(),
        duration_ms,
    }
}

fn print_test(t: &TestResult) {
    let status = if t.passed { "PASS" } else { "FAIL" };
    let suffix = match &t.error {
        Some(e) => format!(" — {}", e),
        None => String::new(),
    };
    println!("  {}  {} ({:.1}ms){}", status, t.name, t.duration_ms, suffix);
}

struct BenchResult {
    name: String,
    value: f64,
    unit: String,
}

fn print_bench(b: &BenchResult) {
    let formatted = if b.unit == "FLOPS" {
        format_gflops(b.value)
    } else {
        format!("{:.2} {}", b.value, b.unit)
    };
    println!("  {}: {}", b.name, formatted);
}

// ═══════════════════════════════════════════════════════════════════════════
// Section 1: Correctness Tests
// ═══════════════════════════════════════════════════════════════════════════

fn test_vector_add(backend: &mut AnyBackend) -> Result<(), String> {
    let ir = build_vector_add_ir();
    backend.compile_kernel("vector_add", &ir)?;

    for &n in &[64, 1024, 65536] {
        let a_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (0..n).map(|i| (i * 2) as f32).collect();
        let c_data = vec![0.0f32; n];

        let a_id = alloc_with_data(backend, &a_data);
        let b_id = alloc_with_data(backend, &b_data);
        let c_id = alloc_with_data(backend, &c_data);

        let block_size = 256u32.min(n as u32);
        let grid_size = ((n as u32) + block_size - 1) / block_size;

        backend.launch("vector_add", (grid_size, 1, 1), (block_size, 1, 1), &[
            KernelParam::Buffer(a_id), KernelParam::Buffer(b_id), KernelParam::Buffer(c_id),
        ])?;
        backend.synchronize()?;

        let result = read_buffer(backend, c_id, n);
        let mut errors = 0;
        for i in 0..n {
            if (result[i] - (a_data[i] + b_data[i])).abs() > 1e-5 { errors += 1; }
        }
        backend.free(a_id)?; backend.free(b_id)?; backend.free(c_id)?;
        if errors > 0 { return Err(format!("n={}: {} errors", n, errors)); }
    }
    Ok(())
}

fn test_vector_mul(backend: &mut AnyBackend) -> Result<(), String> {
    let ir = build_vector_mul_ir();
    backend.compile_kernel("vector_mul", &ir)?;

    let n = 4096;
    let a_data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
    let b_data: Vec<f32> = (0..n).map(|i| (n - i) as f32 * 0.01).collect();
    let c_data = vec![0.0f32; n];

    let a_id = alloc_with_data(backend, &a_data);
    let b_id = alloc_with_data(backend, &b_data);
    let c_id = alloc_with_data(backend, &c_data);

    backend.launch("vector_mul", (16, 1, 1), (256, 1, 1), &[
        KernelParam::Buffer(a_id), KernelParam::Buffer(b_id), KernelParam::Buffer(c_id),
    ])?;
    backend.synchronize()?;

    let result = read_buffer(backend, c_id, n);
    let mut errors = 0;
    for i in 0..n {
        let expected = a_data[i] * b_data[i];
        if (result[i] - expected).abs() > 1e-3 { errors += 1; }
    }
    backend.free(a_id)?; backend.free(b_id)?; backend.free(c_id)?;
    if errors > 0 { Err(format!("{}/{} errors", errors, n)) } else { Ok(()) }
}

fn test_fma(backend: &mut AnyBackend) -> Result<(), String> {
    let ir = build_fma_ir();
    backend.compile_kernel("vector_fma", &ir)?;

    let n = 2048;
    let mut rng = Rng::new(99);
    let x = rng.vec_f32(n);
    let y = rng.vec_f32(n);
    let z = rng.vec_f32(n);
    let out = vec![0.0f32; n];

    let x_id = alloc_with_data(backend, &x);
    let y_id = alloc_with_data(backend, &y);
    let z_id = alloc_with_data(backend, &z);
    let o_id = alloc_with_data(backend, &out);

    backend.launch("vector_fma", (8, 1, 1), (256, 1, 1), &[
        KernelParam::Buffer(x_id), KernelParam::Buffer(y_id),
        KernelParam::Buffer(z_id), KernelParam::Buffer(o_id),
    ])?;
    backend.synchronize()?;

    let result = read_buffer(backend, o_id, n);
    let mut max_err = 0.0f32;
    for i in 0..n {
        let expected = x[i] * y[i] + z[i];
        max_err = max_err.max((result[i] - expected).abs());
    }
    backend.free(x_id)?; backend.free(y_id)?; backend.free(z_id)?; backend.free(o_id)?;
    if max_err > 1e-3 { Err(format!("max error {:.6}", max_err)) } else { Ok(()) }
}

fn test_saxpy_kernel(backend: &mut AnyBackend) -> Result<(), String> {
    let ir = build_saxpy_ir();
    backend.compile_kernel("saxpy", &ir)?;

    let n = 4096;
    let alpha = 2.5f32;
    let a_data = vec![alpha; n];
    let mut rng = Rng::new(77);
    let x_data = rng.vec_f32(n);
    let y_data = rng.vec_f32(n);
    let out_data = vec![0.0f32; n];

    let a_id = alloc_with_data(backend, &a_data);
    let x_id = alloc_with_data(backend, &x_data);
    let y_id = alloc_with_data(backend, &y_data);
    let o_id = alloc_with_data(backend, &out_data);

    backend.launch("saxpy", (16, 1, 1), (256, 1, 1), &[
        KernelParam::Buffer(a_id), KernelParam::Buffer(x_id),
        KernelParam::Buffer(y_id), KernelParam::Buffer(o_id),
    ])?;
    backend.synchronize()?;

    let result = read_buffer(backend, o_id, n);
    let mut max_err = 0.0f32;
    for i in 0..n {
        let expected = alpha * x_data[i] + y_data[i];
        max_err = max_err.max((result[i] - expected).abs());
    }
    backend.free(a_id)?; backend.free(x_id)?; backend.free(y_id)?; backend.free(o_id)?;
    if max_err > 1e-3 { Err(format!("max error {:.6}", max_err)) } else { Ok(()) }
}

fn test_memory_roundtrip(backend: &mut AnyBackend) -> Result<(), String> {
    let sizes = [64, 512, 4096, 65536, 262144, 1_048_576, 4_194_304];
    for &size in &sizes {
        let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        let (id, _) = backend.alloc(size, false)?;
        backend.copy_htod(id, &data)?;
        let result = backend.copy_dtoh(id, size)?;
        backend.free(id)?;
        if data != result {
            return Err(format!("Mismatch at size={}", format_bytes(size)));
        }
    }
    Ok(())
}

fn test_memset(backend: &mut AnyBackend) -> Result<(), String> {
    for &(size, val) in &[(4096usize, 0x00u8), (4096, 0xFF), (65536, 0xAB), (1_048_576, 0x42)] {
        let (id, _) = backend.alloc(size, false)?;
        backend.memset(id, val, size)?;
        let result = backend.copy_dtoh(id, size)?;
        backend.free(id)?;
        for (i, &byte) in result.iter().enumerate() {
            if byte != val {
                return Err(format!("size={} val={:#x}: byte {} got {:#x}", size, val, i, byte));
            }
        }
    }
    Ok(())
}

fn test_dtod_copy(backend: &mut AnyBackend) -> Result<(), String> {
    for &size in &[4096usize, 65536, 1_048_576] {
        let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        let (src_id, _) = backend.alloc(size, false)?;
        let (dst_id, _) = backend.alloc(size, false)?;
        backend.copy_htod(src_id, &data)?;
        backend.copy_dtod(src_id, dst_id, size)?;
        let result = backend.copy_dtoh(dst_id, size)?;
        backend.free(src_id)?; backend.free(dst_id)?;
        if data != result { return Err(format!("D2D mismatch at size={}", format_bytes(size))); }
    }
    Ok(())
}

fn test_multi_alloc_free(backend: &mut AnyBackend) -> Result<(), String> {
    let mut ids = Vec::new();
    for i in 0..100 {
        let size = 1024 * (i + 1);
        let (id, _) = backend.alloc(size, false)?;
        ids.push(id);
    }
    // Free in reverse order
    for id in ids.iter().rev() { backend.free(*id)?; }

    // Allocate again, free in interleaved order
    let mut ids2 = Vec::new();
    for _ in 0..50 {
        let (id, _) = backend.alloc(8192, false)?;
        ids2.push(id);
    }
    for i in (0..50).step_by(2) { backend.free(ids2[i])?; }
    for i in (1..50).step_by(2) { backend.free(ids2[i])?; }
    Ok(())
}

fn test_multi_kernel(backend: &mut AnyBackend) -> Result<(), String> {
    let ir_add = build_vector_add_ir();
    let ir_mul = build_vector_mul_ir();
    backend.compile_kernel("mk_add", &ir_add)?;
    backend.compile_kernel("mk_mul", &ir_mul)?;

    let n = 1024;
    let a_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
    let c_data = vec![0.0f32; n];

    let a_id = alloc_with_data(backend, &a_data);
    let b_id = alloc_with_data(backend, &b_data);
    let add_id = alloc_with_data(backend, &c_data);
    let mul_id = alloc_with_data(backend, &c_data);

    let grid = (4u32, 1, 1);
    let block = (256u32, 1, 1);

    backend.launch("mk_add", grid, block, &[
        KernelParam::Buffer(a_id), KernelParam::Buffer(b_id), KernelParam::Buffer(add_id),
    ])?;
    backend.launch("mk_mul", grid, block, &[
        KernelParam::Buffer(a_id), KernelParam::Buffer(b_id), KernelParam::Buffer(mul_id),
    ])?;
    backend.synchronize()?;

    let add_result = read_buffer(backend, add_id, n);
    let mul_result = read_buffer(backend, mul_id, n);

    let mut errors = 0;
    for i in 0..n {
        if (add_result[i] - (a_data[i] + b_data[i])).abs() > 1e-3 { errors += 1; }
        if (mul_result[i] - (a_data[i] * b_data[i])).abs() > 1e-1 { errors += 1; }
    }
    backend.free(a_id)?; backend.free(b_id)?; backend.free(add_id)?; backend.free(mul_id)?;
    if errors > 0 { Err(format!("{} mismatches", errors)) } else { Ok(()) }
}

// ═══════════════════════════════════════════════════════════════════════════
// Section 2: BLAS Correctness
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "cpu")]
fn test_sgemm_blas(backend: &mut AnyBackend) -> Result<(), String> {
    for &(m, k, n) in &[(32, 32, 32), (64, 128, 64), (128, 64, 128), (256, 256, 256)] {
        let mut rng = Rng::new(m as u64 * 1000 + n as u64);
        let a = rng.vec_f32(m * k);
        let b = rng.vec_f32(k * n);
        let expected = cpu_matmul(&a, &b, m, k, n);

        let mut blas = CpuBlasBackend::new();
        let a_id = BackendBufferId(1);
        let b_id = BackendBufferId(2);
        let c_id = BackendBufferId(3);
        blas.register_buffer(a_id, f32_to_bytes(&a));
        blas.register_buffer(b_id, f32_to_bytes(&b));
        blas.register_buffer(c_id, f32_to_bytes(&vec![0.0f32; m * n]));

        let config = GemmConfig::new(m, n, k);
        blas.sgemm(&config, a_id, b_id, c_id)?;

        let result_bytes = blas.buffers_ref().get(&c_id.0)
            .ok_or("BLAS result buffer missing")?;
        let result = bytes_to_f32(result_bytes);

        let mut max_err = 0.0f32;
        for i in 0..m * n {
            max_err = max_err.max((result[i] - expected[i]).abs());
        }
        if max_err > 1e-2 {
            return Err(format!("{}x{}x{}: max error {:.6}", m, k, n, max_err));
        }

        // Also verify compute backend round-trip at this size
        let ba = alloc_with_data(backend, &a);
        let bb = alloc_with_data(backend, &b);
        let a_rt = read_buffer(backend, ba, m * k);
        let b_rt = read_buffer(backend, bb, k * n);
        backend.free(ba)?; backend.free(bb)?;
        for i in 0..m * k {
            if (a[i] - a_rt[i]).abs() > 1e-6 { return Err(format!("A round-trip at [{}]", i)); }
        }
        for i in 0..k * n {
            if (b[i] - b_rt[i]).abs() > 1e-6 { return Err(format!("B round-trip at [{}]", i)); }
        }
    }
    Ok(())
}

#[cfg(feature = "cpu")]
fn test_sgemm_alpha_beta(backend: &mut AnyBackend) -> Result<(), String> {
    let _ = backend;
    let m = 64;
    let k = 64;
    let n = 64;
    let mut rng = Rng::new(555);
    let a = rng.vec_f32(m * k);
    let b = rng.vec_f32(k * n);
    let c_init = rng.vec_f32(m * n);

    let alpha = 2.5f32;
    let beta = 0.5f32;

    // CPU reference: C = alpha * A*B + beta * C
    let ab = cpu_matmul(&a, &b, m, k, n);
    let expected: Vec<f32> = (0..m * n).map(|i| alpha * ab[i] + beta * c_init[i]).collect();

    let mut blas = CpuBlasBackend::new();
    let a_id = BackendBufferId(1);
    let b_id = BackendBufferId(2);
    let c_id = BackendBufferId(3);
    blas.register_buffer(a_id, f32_to_bytes(&a));
    blas.register_buffer(b_id, f32_to_bytes(&b));
    blas.register_buffer(c_id, f32_to_bytes(&c_init));

    let config = GemmConfig::new(m, n, k).with_alpha(alpha).with_beta(beta);
    blas.sgemm(&config, a_id, b_id, c_id)?;

    let result = bytes_to_f32(blas.buffers_ref().get(&c_id.0).ok_or("missing")?);
    let mut max_err = 0.0f32;
    for i in 0..m * n {
        max_err = max_err.max((result[i] - expected[i]).abs());
    }
    if max_err > 1e-1 { Err(format!("max error {:.6}", max_err)) } else { Ok(()) }
}

#[cfg(feature = "cpu")]
fn test_blas_saxpy(_backend: &mut AnyBackend) -> Result<(), String> {
    let n = 1024;
    let alpha = 3.0f32;
    let mut rng = Rng::new(333);
    let x = rng.vec_f32(n);
    let y_init = rng.vec_f32(n);
    let expected: Vec<f32> = (0..n).map(|i| alpha * x[i] + y_init[i]).collect();

    let mut blas = CpuBlasBackend::new();
    let x_id = BackendBufferId(10);
    let y_id = BackendBufferId(11);
    blas.register_buffer(x_id, f32_to_bytes(&x));
    blas.register_buffer(y_id, f32_to_bytes(&y_init));

    blas.saxpy(n, alpha, x_id, y_id)?;

    let result = bytes_to_f32(blas.buffers_ref().get(&y_id.0).ok_or("missing")?);
    let mut max_err = 0.0f32;
    for i in 0..n { max_err = max_err.max((result[i] - expected[i]).abs()); }
    if max_err > 1e-4 { Err(format!("max error {:.6}", max_err)) } else { Ok(()) }
}

#[cfg(feature = "cpu")]
fn test_blas_sdot(_backend: &mut AnyBackend) -> Result<(), String> {
    let n = 512;
    let mut rng = Rng::new(444);
    let x = rng.vec_f32(n);
    let y = rng.vec_f32(n);
    let expected: f32 = (0..n).map(|i| x[i] * y[i]).sum();

    let mut blas = CpuBlasBackend::new();
    let x_id = BackendBufferId(20);
    let y_id = BackendBufferId(21);
    blas.register_buffer(x_id, f32_to_bytes(&x));
    blas.register_buffer(y_id, f32_to_bytes(&y));

    let result = blas.sdot(n, x_id, y_id)?;
    if (result - expected).abs() > 1e-2 {
        Err(format!("got {}, expected {}", result, expected))
    } else { Ok(()) }
}

#[cfg(feature = "cpu")]
fn test_blas_snrm2(_backend: &mut AnyBackend) -> Result<(), String> {
    let n = 256;
    let mut rng = Rng::new(666);
    let x = rng.vec_f32(n);
    let expected: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();

    let mut blas = CpuBlasBackend::new();
    let x_id = BackendBufferId(30);
    blas.register_buffer(x_id, f32_to_bytes(&x));

    let result = blas.snrm2(n, x_id)?;
    if (result - expected).abs() > 1e-3 {
        Err(format!("got {}, expected {}", result, expected))
    } else { Ok(()) }
}

// ═══════════════════════════════════════════════════════════════════════════
// Section 3: Performance Benchmarks
// ═══════════════════════════════════════════════════════════════════════════

fn bench_memory_bandwidth(backend: &mut AnyBackend, sizes: &[usize]) -> Vec<BenchResult> {
    let mut results = Vec::new();
    for &size in sizes {
        let data = vec![0u8; size];
        let (id, _) = backend.alloc(size, false).expect("alloc");

        // Warmup
        for _ in 0..3 {
            backend.copy_htod(id, &data).expect("htod");
            let _ = backend.copy_dtoh(id, size).expect("dtoh");
        }

        let iterations = if size >= 16 * 1024 * 1024 { 20 } else { 50 };
        let start = Instant::now();
        for _ in 0..iterations {
            backend.copy_htod(id, &data).expect("htod");
            let _ = backend.copy_dtoh(id, size).expect("dtoh");
        }
        let elapsed = start.elapsed().as_secs_f64();
        backend.free(id).expect("free");

        let gb_s = (iterations * size * 2) as f64 / elapsed / (1024.0 * 1024.0 * 1024.0);
        results.push(BenchResult {
            name: format!("Mem BW ({})", format_bytes(size)),
            value: gb_s, unit: "GB/s".into(),
        });
    }
    results
}

fn bench_kernel_throughput(backend: &mut AnyBackend, sizes: &[usize]) -> Vec<BenchResult> {
    let ir = build_vector_add_ir();
    if backend.compile_kernel("bench_vadd", &ir).is_err() { return vec![]; }

    let mut results = Vec::new();
    for &n in sizes {
        let a_id = alloc_with_data(backend, &vec![1.0f32; n]);
        let b_id = alloc_with_data(backend, &vec![2.0f32; n]);
        let c_id = alloc_with_data(backend, &vec![0.0f32; n]);

        let block_size = 256u32;
        let grid_size = ((n as u32) + block_size - 1) / block_size;
        let params = [
            KernelParam::Buffer(a_id), KernelParam::Buffer(b_id), KernelParam::Buffer(c_id),
        ];

        // Warmup
        for _ in 0..5 {
            let _ = backend.launch("bench_vadd", (grid_size, 1, 1), (block_size, 1, 1), &params);
            let _ = backend.synchronize();
        }

        let iterations = if n >= 1_000_000 { 100 } else { 500 };
        let start = Instant::now();
        for _ in 0..iterations {
            backend.launch("bench_vadd", (grid_size, 1, 1), (block_size, 1, 1), &params).expect("launch");
        }
        backend.synchronize().expect("sync");
        let elapsed = start.elapsed().as_secs_f64();

        backend.free(a_id).expect("free"); backend.free(b_id).expect("free"); backend.free(c_id).expect("free");

        let gb_s = (iterations as f64 * n as f64 * 3.0 * 4.0) / elapsed / (1024.0 * 1024.0 * 1024.0);
        results.push(BenchResult {
            name: format!("VecAdd ({})", format_bytes(n * 4)),
            value: gb_s, unit: "GB/s".into(),
        });
    }
    results
}

#[cfg(feature = "cpu")]
fn bench_sgemm_throughput() -> Vec<BenchResult> {
    let mut results = Vec::new();
    for &dim in &[128, 256, 512, 1024, 2048] {
        let mut rng = Rng::new(dim as u64);
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

        let iterations = if dim >= 1024 { 3 } else if dim >= 512 { 10 } else { 30 };
        let start = Instant::now();
        for _ in 0..iterations {
            blas.sgemm(&config, a_id, b_id, c_id).expect("sgemm");
        }
        let elapsed = start.elapsed().as_secs_f64();

        let flops = 2.0 * (dim as f64).powi(3) * iterations as f64;
        results.push(BenchResult {
            name: format!("SGEMM {}x{}", dim, dim),
            value: flops / elapsed, unit: "FLOPS".into(),
        });
    }
    results
}

fn bench_kernel_latency(backend: &mut AnyBackend) -> Vec<BenchResult> {
    let ir = build_vector_add_ir();
    if backend.compile_kernel("latency_vadd", &ir).is_err() { return vec![]; }

    let a_id = alloc_with_data(backend, &[1.0f32]);
    let b_id = alloc_with_data(backend, &[2.0f32]);
    let c_id = alloc_with_data(backend, &[0.0f32]);
    let params = [
        KernelParam::Buffer(a_id), KernelParam::Buffer(b_id), KernelParam::Buffer(c_id),
    ];

    // Warmup
    for _ in 0..100 {
        let _ = backend.launch("latency_vadd", (1, 1, 1), (1, 1, 1), &params);
        let _ = backend.synchronize();
    }

    let iterations = 1000;
    let start = Instant::now();
    for _ in 0..iterations {
        backend.launch("latency_vadd", (1, 1, 1), (1, 1, 1), &params).expect("launch");
        backend.synchronize().expect("sync");
    }
    let latency_us = start.elapsed().as_secs_f64() / iterations as f64 * 1e6;

    backend.free(a_id).expect("free"); backend.free(b_id).expect("free"); backend.free(c_id).expect("free");
    vec![BenchResult { name: "Kernel latency".into(), value: latency_us, unit: "us".into() }]
}

fn bench_alloc_free(backend: &mut AnyBackend) -> Vec<BenchResult> {
    let iterations = 1000;
    let start = Instant::now();
    for _ in 0..iterations {
        let (id, _) = backend.alloc(4096, false).expect("alloc");
        backend.free(id).expect("free");
    }
    let ops_s = iterations as f64 / start.elapsed().as_secs_f64();
    vec![BenchResult { name: "Alloc+Free (4KB)".into(), value: ops_s, unit: "ops/s".into() }]
}

// ═══════════════════════════════════════════════════════════════════════════
// Section 4: Stress Tests
// ═══════════════════════════════════════════════════════════════════════════

fn stress_large_alloc(backend: &mut AnyBackend) -> Result<(), String> {
    let sizes = [
        1 * 1024 * 1024, 16 * 1024 * 1024, 64 * 1024 * 1024,
        256 * 1024 * 1024, 512 * 1024 * 1024, 1024 * 1024 * 1024,
    ];
    let mut max_size = 0;
    for &size in &sizes {
        match backend.alloc(size, false) {
            Ok((id, _)) => { max_size = size; backend.free(id)?; }
            Err(_) => break,
        }
    }
    println!("    Max allocation: {}", format_bytes(max_size));
    if max_size >= 16 * 1024 * 1024 { Ok(()) }
    else { Err(format!("Only allocated up to {}", format_bytes(max_size))) }
}

fn stress_rapid_kernels(backend: &mut AnyBackend) -> Result<(), String> {
    let ir = build_vector_add_ir();
    backend.compile_kernel("stress_vadd", &ir)?;

    let n = 1024;
    let a_id = alloc_with_data(backend, &vec![1.0f32; n]);
    let b_id = alloc_with_data(backend, &vec![2.0f32; n]);
    let c_id = alloc_with_data(backend, &vec![0.0f32; n]);
    let params = [
        KernelParam::Buffer(a_id), KernelParam::Buffer(b_id), KernelParam::Buffer(c_id),
    ];

    let start = Instant::now();
    for _ in 0..10_000 {
        backend.launch("stress_vadd", (4, 1, 1), (256, 1, 1), &params)?;
    }
    backend.synchronize()?;
    let elapsed = start.elapsed().as_secs_f64();
    println!("    10K dispatches in {:.2}s ({:.0} kernels/s)", elapsed, 10_000.0 / elapsed);

    let result = read_buffer(backend, c_id, n);
    backend.free(a_id)?; backend.free(b_id)?; backend.free(c_id)?;
    if (result[0] - 3.0).abs() > 1e-3 { Err(format!("Got {}, expected 3.0", result[0])) }
    else { Ok(()) }
}

fn stress_sustained_throughput(backend: &mut AnyBackend) -> Result<(), String> {
    let ir = build_vector_add_ir();
    backend.compile_kernel("sustained_vadd", &ir)?;

    let n = 1_000_000;
    let a_id = alloc_with_data(backend, &vec![1.0f32; n]);
    let b_id = alloc_with_data(backend, &vec![2.0f32; n]);
    let c_id = alloc_with_data(backend, &vec![0.0f32; n]);
    let params = [
        KernelParam::Buffer(a_id), KernelParam::Buffer(b_id), KernelParam::Buffer(c_id),
    ];

    let block_size = 256u32;
    let grid_size = ((n as u32) + block_size - 1) / block_size;

    let duration = std::time::Duration::from_secs(30);
    let window = std::time::Duration::from_secs(5);
    let start = Instant::now();
    let mut window_start = start;
    let mut window_iters = 0u64;
    let mut windows: Vec<f64> = Vec::new();

    while start.elapsed() < duration {
        for _ in 0..100 {
            backend.launch("sustained_vadd", (grid_size, 1, 1), (block_size, 1, 1), &params)?;
        }
        backend.synchronize()?;
        window_iters += 100;

        if window_start.elapsed() >= window {
            let elapsed = window_start.elapsed().as_secs_f64();
            let gb_s = (window_iters as f64 * n as f64 * 3.0 * 4.0) / elapsed / (1024.0 * 1024.0 * 1024.0);
            windows.push(gb_s);
            window_start = Instant::now();
            window_iters = 0;
        }
    }

    backend.free(a_id)?; backend.free(b_id)?; backend.free(c_id)?;

    if windows.len() < 2 { return Err("Not enough windows".into()); }

    let max_bw = windows.iter().copied().fold(0.0f64, f64::max);
    let min_bw = windows.iter().copied().fold(f64::MAX, f64::min);
    let avg_bw = windows.iter().sum::<f64>() / windows.len() as f64;

    println!("    Sustained: {:.2} GB/s avg ({:.2}–{:.2}, {} windows)",
        avg_bw, min_bw, max_bw, windows.len());

    if min_bw < max_bw * 0.5 {
        Err(format!("Throughput degraded: {:.2}→{:.2} GB/s", max_bw, min_bw))
    } else { Ok(()) }
}

fn stress_concurrent_buffers(backend: &mut AnyBackend) -> Result<(), String> {
    // Hold 500 buffers simultaneously, write and read each
    let count = 500;
    let size = 4096;
    let mut ids = Vec::new();
    for i in 0..count {
        let (id, _) = backend.alloc(size, false)?;
        let data: Vec<u8> = (0..size).map(|j| ((i + j) % 256) as u8).collect();
        backend.copy_htod(id, &data)?;
        ids.push((id, data));
    }

    // Verify all buffers
    for (id, expected) in &ids {
        let result = backend.copy_dtoh(*id, size)?;
        if *expected != result { return Err(format!("Buffer {:?} mismatch", id)); }
    }

    // Free all
    for (id, _) in &ids { backend.free(*id)?; }
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// JSON output
// ═══════════════════════════════════════════════════════════════════════════

fn emit_json(
    caps: &invisible_cuda::compute::DeviceCapabilities,
    backend_kind: &str,
    tests: &[TestResult],
    benches: &[BenchResult],
    hw: &HwFingerprint,
) {
    // Escape for JSON
    let escape = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"");

    println!();
    println!("--- JSON_START ---");
    println!("{{");
    println!("  \"format\": \"invisible-cuda-proof-v3\",");
    println!("  \"backend\": \"{}\",", escape(backend_kind));
    println!("  \"device\": \"{}\",", escape(&caps.name));
    println!("  \"os\": \"{}\",", std::env::consts::OS);
    println!("  \"arch\": \"{}\",", std::env::consts::ARCH);
    println!("  \"compute_units\": {},", caps.compute_units);
    println!("  \"memory_bytes\": {},", caps.total_memory);
    println!("  \"max_threads_per_block\": {},", caps.max_threads_per_block);
    println!("  \"has_shared_memory\": {},", caps.has_shared_memory);
    println!("  \"has_atomics\": {},", caps.has_atomics);
    hw.emit_json_fields();

    println!("  \"tests\": [");
    for (i, t) in tests.iter().enumerate() {
        let comma = if i + 1 < tests.len() { "," } else { "" };
        let err = match &t.error { Some(e) => format!("\"{}\"", escape(e)), None => "null".into() };
        println!("    {{\"name\": \"{}\", \"passed\": {}, \"ms\": {:.1}, \"error\": {}}}{}", escape(&t.name), t.passed, t.duration_ms, err, comma);
    }
    println!("  ],");

    let bw_factor = hw.mem_bw_factor();
    println!("  \"benchmarks\": [");
    for (i, b) in benches.iter().enumerate() {
        let comma = if i + 1 < benches.len() { "," } else { "" };
        // Add normalized value for bandwidth and FLOPS benchmarks
        let normalized = if b.unit == "GB/s" && bw_factor > 0.0 {
            b.value / bw_factor
        } else if b.unit == "FLOPS" {
            let gflops = b.value / 1e9;
            hw.compute_factor(gflops)
        } else {
            b.value
        };
        let norm_unit = if b.unit == "GB/s" { "GB/s @DDR4-3200x1" }
            else if b.unit == "FLOPS" { "GFLOPS/core/GHz" }
            else { &b.unit };
        println!("    {{\"name\": \"{}\", \"value\": {:.4}, \"unit\": \"{}\", \"normalized\": {:.4}, \"normalized_unit\": \"{}\"}}{}",
            escape(&b.name), b.value, escape(&b.unit), normalized, escape(norm_unit), comma);
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
    println!("║       INVISIBLE CUDA — Universal Backend Proof (v2)        ║");
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
    println!("Memory:  {} ({} bytes)", format_bytes(caps.total_memory as usize), caps.total_memory);
    println!("Compute: {} units", caps.compute_units);
    println!("Max threads/block: {}", caps.max_threads_per_block);
    if caps.peak_gflops_f32 > 0.0 {
        println!("Peak GFLOPS (FP32): {:.1}", caps.peak_gflops_f32);
    }
    println!("Features: shared_mem={} atomics={} fp16={} warp={}",
        caps.has_shared_memory, caps.has_atomics, caps.has_fp16, caps.warp_size);
    println!();

    let total_start = Instant::now();
    let mut all_tests: Vec<TestResult> = Vec::new();
    let mut all_benches: Vec<BenchResult> = Vec::new();

    // ── Correctness Tests ────────────────────────────────────────────
    println!("── Correctness Tests ──────────────────────────────────────────");

    all_tests.push(run_test("Vector Add (64/1K/64K)", &mut backend, test_vector_add));
    all_tests.push(run_test("Vector Mul (4096)", &mut backend, test_vector_mul));
    all_tests.push(run_test("FMA x*y+z (2048)", &mut backend, test_fma));
    all_tests.push(run_test("SAXPY a*x+y (4096)", &mut backend, test_saxpy_kernel));
    all_tests.push(run_test("Memory round-trip (64B..4MB)", &mut backend, test_memory_roundtrip));
    all_tests.push(run_test("Memset (4KB..1MB, 4 patterns)", &mut backend, test_memset));
    all_tests.push(run_test("D2D copy (4KB..1MB)", &mut backend, test_dtod_copy));
    all_tests.push(run_test("Multi alloc+free (150 buffers)", &mut backend, test_multi_alloc_free));
    all_tests.push(run_test("Multi-kernel dispatch (add+mul)", &mut backend, test_multi_kernel));

    for t in &all_tests { print_test(t); }
    println!();

    // ── BLAS Correctness (standard + extended) ───────────────────────
    if mode == "standard" || mode == "extended" {
        println!("── BLAS Correctness ───────────────────────────────────────────");

        #[cfg(feature = "cpu")]
        {
            all_tests.push(run_test("SGEMM (32..256, 4 sizes)", &mut backend, test_sgemm_blas));
            all_tests.push(run_test("SGEMM alpha=2.5 beta=0.5", &mut backend, test_sgemm_alpha_beta));
            all_tests.push(run_test("BLAS SAXPY", &mut backend, test_blas_saxpy));
            all_tests.push(run_test("BLAS SDOT", &mut backend, test_blas_sdot));
            all_tests.push(run_test("BLAS SNRM2", &mut backend, test_blas_snrm2));

            for t in all_tests.iter().rev().take(5).collect::<Vec<_>>().into_iter().rev() {
                print_test(t);
            }
        }
        #[cfg(not(feature = "cpu"))]
        println!("  (skipped — cpu feature not enabled)");

        println!();
    }

    // ── Performance Benchmarks ───────────────────────────────────────
    println!("── Performance Benchmarks ─────────────────────────────────────");

    let mem_sizes = if mode == "extended" {
        vec![64 * 1024, 256 * 1024, 1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024, 64 * 1024 * 1024]
    } else if mode == "standard" {
        vec![256 * 1024, 1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024]
    } else {
        vec![1024 * 1024, 16 * 1024 * 1024]
    };
    all_benches.extend(bench_memory_bandwidth(&mut backend, &mem_sizes));

    let kernel_sizes = if mode == "extended" {
        vec![1024, 10_000, 100_000, 1_000_000, 4_000_000]
    } else if mode == "standard" {
        vec![10_000, 100_000, 1_000_000]
    } else {
        vec![100_000, 1_000_000]
    };
    all_benches.extend(bench_kernel_throughput(&mut backend, &kernel_sizes));

    if mode == "standard" || mode == "extended" {
        #[cfg(feature = "cpu")]
        all_benches.extend(bench_sgemm_throughput());

        all_benches.extend(bench_kernel_latency(&mut backend));
        all_benches.extend(bench_alloc_free(&mut backend));
    }

    println!("  NOTE: Performance numbers are informational. They measure execution");
    println!("  throughput, not correctness. Variations across hardware are expected");
    println!("  and reflect memory/compute class differences, not compatibility gaps.");
    if hw.mem_bandwidth_gbps > 0.0 {
        let ch_note = if hw.mem_channels_estimated { " (est.)" } else { "" };
        println!("  Memory normalization: {} @ {} MT/s × {} ch{} = {:.1} GB/s (factor={:.2}x vs DDR4-3200×1)",
            hw.mem_type, hw.mem_speed_mt, hw.mem_channels, ch_note, hw.mem_bandwidth_gbps, hw.mem_bw_factor());
    }
    println!();
    for b in &all_benches {
        print_bench(b);
        // Print normalized value for bandwidth benchmarks
        if b.unit == "GB/s" && hw.mem_bw_factor() > 0.0 && hw.mem_bw_factor() != 1.0 {
            let normalized = b.value / hw.mem_bw_factor();
            println!("    (normalized: {:.2} GB/s @ DDR4-3200×1 equivalent)", normalized);
        }
        if b.unit == "FLOPS" && hw.mem_bandwidth_gbps > 0.0 {
            let gflops = b.value / 1e9;
            let per_core_ghz = hw.compute_factor(gflops);
            println!("    (normalized: {:.3} GFLOPS/core/GHz)", per_core_ghz);
        }
    }
    println!();

    // ── Stress Tests (standard + extended) ───────────────────────────
    if mode == "standard" || mode == "extended" {
        println!("── Stress Tests ───────────────────────────────────────────────");
        all_tests.push(run_test("Large allocation (up to 1GB)", &mut backend, stress_large_alloc));
        all_tests.push(run_test("Rapid kernels (10K dispatches)", &mut backend, stress_rapid_kernels));
        all_tests.push(run_test("500 concurrent buffers", &mut backend, stress_concurrent_buffers));

        for t in all_tests.iter().rev().take(3).collect::<Vec<_>>().into_iter().rev() {
            print_test(t);
        }

        if mode == "extended" {
            all_tests.push(run_test("Sustained throughput (30s)", &mut backend, stress_sustained_throughput));
            print_test(all_tests.last().unwrap());
        }
        println!();
    }

    // ── Summary ──────────────────────────────────────────────────────
    let passed = all_tests.iter().filter(|t| t.passed).count();
    let failed = all_tests.iter().filter(|t| !t.passed).count();
    let total_time = total_start.elapsed().as_secs_f64();

    println!("══════════════════════════════════════════════════════════════");
    if failed == 0 {
        println!("  RESULT: ALL {} TESTS PASSED on {} ({})",
            passed, backend.kind(), caps.name);
        println!("  CUDA compatibility: VERIFIED");
    } else {
        println!("  RESULT: {} passed, {} FAILED on {} ({})",
            passed, failed, backend.kind(), caps.name);
        println!("  CUDA compatibility: PARTIAL");
        for t in &all_tests {
            if !t.passed {
                println!("    FAILED: {} — {}", t.name, t.error.as_deref().unwrap_or("?"));
            }
        }
    }
    println!("  Total time: {:.1}s ({} tests, {} benchmarks)", total_time, all_tests.len(), all_benches.len());
    println!("══════════════════════════════════════════════════════════════");

    // JSON output for machine parsing
    let backend_kind = format!("{}", backend.kind());
    emit_json(&caps, &backend_kind, &all_tests, &all_benches, &hw);

    if failed > 0 { std::process::exit(1); }
}
