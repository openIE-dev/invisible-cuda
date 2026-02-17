//! Invisible CUDA — Rust Proof
//!
//! Demonstrates CUDA compatibility on any hardware using the Invisible CUDA SDK.
//!
//! Prerequisites:
//!   cargo install invisible-cuda && invisible-cuda install
//!
//! Run:
//!   cargo run --release

use invisible_cuda_sdk::{
    self as cuda, CublasHandle, CublasOperation, CuContext, CuModule,
    Dim3, DType, GpuArray, Stream, Event,
};
use std::ffi::c_void;
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         INVISIBLE CUDA — Rust SDK Proof                    ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let mut passed = 0u32;
    let mut failed = 0u32;
    let start = Instant::now();

    // ── 1. Device Discovery ──────────────────────────────────────────
    println!("── Device Discovery ──");

    match cuda::is_available() {
        true => { println!("  PASS  Runtime detected"); passed += 1; }
        false => { println!("  FAIL  Runtime not found"); failed += 1; return; }
    }

    let count = cuda::get_device_count().expect("get_device_count");
    assert!(count >= 1);
    println!("  PASS  Device count: {}", count);
    passed += 1;

    let info = cuda::device_info(0).expect("device_info");
    println!("  PASS  Device: {} ({:.1} GB, CUDA {})",
        info.name,
        info.total_memory as f64 / 1e9,
        info.cuda_version);
    passed += 1;

    let version = cuda::runtime_get_version().expect("runtime_get_version");
    assert!(version >= 12000);
    println!("  PASS  Runtime version: {}", version);
    passed += 1;

    let (free, total) = cuda::mem_get_info().expect("mem_get_info");
    assert!(total > 0 && free > 0 && free <= total);
    println!("  PASS  Memory: {:.1} GB free / {:.1} GB total",
        free as f64 / 1e9, total as f64 / 1e9);
    passed += 1;
    println!();

    // ── 2. Memory Operations ─────────────────────────────────────────
    println!("── Memory Operations ──");

    // malloc + free
    let ptr = cuda::malloc(4096).expect("malloc");
    assert!(!ptr.is_null());
    unsafe { cuda::free(ptr).expect("free") };
    println!("  PASS  malloc + free (4 KB)");
    passed += 1;

    // memcpy roundtrip
    let src: Vec<f32> = (0..1024).map(|i| i as f32 * 0.1).collect();
    let ptr = cuda::malloc(src.len() * 4).expect("malloc");
    unsafe {
        cuda::memcpy_htod(ptr, &src).expect("htod");
        let mut dst = vec![0.0f32; 1024];
        cuda::memcpy_dtoh(&mut dst, ptr as *const _).expect("dtoh");
        cuda::free(ptr).expect("free");
        assert_eq!(src, dst);
    }
    println!("  PASS  memcpy roundtrip (4 KB)");
    passed += 1;

    // Large roundtrip
    let n = 256 * 1024;
    let src: Vec<f32> = (0..n).map(|i| (i % 256) as f32 / 256.0).collect();
    let ptr = cuda::malloc(n * 4).expect("malloc");
    unsafe {
        cuda::memcpy_htod(ptr, &src).expect("htod");
        let mut dst = vec![0.0f32; n];
        cuda::memcpy_dtoh(&mut dst, ptr as *const _).expect("dtoh");
        cuda::free(ptr).expect("free");
        for i in 0..n {
            assert!((src[i] - dst[i]).abs() < 1e-5, "mismatch at {}", i);
        }
    }
    println!("  PASS  memcpy roundtrip (1 MB)");
    passed += 1;

    // memset
    let ptr = cuda::malloc(4096).expect("malloc");
    unsafe {
        cuda::memset(ptr, 0xAB, 4096).expect("memset");
        let mut buf = vec![0u8; 4096];
        cuda::memcpy_dtoh(&mut buf, ptr as *const _).expect("dtoh");
        cuda::free(ptr).expect("free");
        for &b in &buf { assert_eq!(b, 0xAB); }
    }
    println!("  PASS  memset (4 KB, pattern 0xAB)");
    passed += 1;
    println!();

    // ── 3. Streams & Events ──────────────────────────────────────────
    println!("── Streams & Events ──");

    let stream = Stream::new().expect("stream create");
    stream.synchronize().expect("stream sync");
    assert!(stream.query().expect("stream query"));
    println!("  PASS  Stream create + sync + query");
    passed += 1;

    let start_ev = Event::new().expect("event create");
    let end_ev = Event::new().expect("event create");
    start_ev.record(None).expect("record");
    cuda::device_synchronize().expect("sync");
    end_ev.record(None).expect("record");
    end_ev.synchronize().expect("sync");
    let ms = Event::elapsed_time(&start_ev, &end_ev).expect("elapsed");
    assert!(ms >= 0.0);
    println!("  PASS  Event timing ({:.3} ms)", ms);
    passed += 1;
    println!();

    // ── 4. Driver API — PTX Kernel ───────────────────────────────────
    println!("── PTX Kernel (Driver API) ──");

    let _ctx = CuContext::new(0).expect("context");

    let ptx = br#"
.version 7.8
.target sm_86
.address_size 64

.visible .entry vector_add(
    .param .u64 .ptr .global param_a,
    .param .u64 .ptr .global param_b,
    .param .u64 .ptr .global param_c,
    .param .u32 param_n
)
{
    .reg .u32 %r<5>;
    .reg .u64 %rd<7>;
    .reg .f32 %f<3>;
    .reg .pred %p;

    ld.param.u64 %rd0, [param_a];
    ld.param.u64 %rd1, [param_b];
    ld.param.u64 %rd2, [param_c];
    ld.param.u32 %r0, [param_n];

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r4, %r1, %r2, %r3;

    setp.ge.u32 %p, %r4, %r0;
    @%p bra done;

    cvt.u64.u32 %rd3, %r4;
    shl.b64 %rd4, %rd3, 2;
    add.u64 %rd5, %rd0, %rd4;
    add.u64 %rd6, %rd1, %rd4;

    ld.global.f32 %f0, [%rd5];
    ld.global.f32 %f1, [%rd6];
    add.f32 %f2, %f0, %f1;

    add.u64 %rd5, %rd2, %rd4;
    st.global.f32 [%rd5], %f2;

done:
    ret;
}
"#;

    let module = CuModule::load_data(ptx).expect("load PTX");
    let func = module.get_function("vector_add").expect("get function");
    println!("  PASS  PTX module loaded + function resolved");
    passed += 1;

    let n = 1024u32;
    let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();

    let size = n as usize * 4;
    let d_a = cuda::malloc(size).expect("malloc a");
    let d_b = cuda::malloc(size).expect("malloc b");
    let d_c = cuda::malloc(size).expect("malloc c");

    unsafe {
        cuda::memcpy_htod(d_a, &a).expect("htod a");
        cuda::memcpy_htod(d_b, &b).expect("htod b");
        cuda::memset(d_c, 0, size).expect("memset c");

        let mut args: Vec<*mut c_void> = vec![
            &d_a as *const _ as *mut c_void,
            &d_b as *const _ as *mut c_void,
            &d_c as *const _ as *mut c_void,
            &n as *const _ as *mut c_void,
        ];

        cuda::launch_kernel(
            func,
            Dim3::new(4, 1, 1),
            Dim3::new(256, 1, 1),
            &mut args,
            0,
        ).expect("launch");
    }

    cuda::device_synchronize().expect("sync");

    let mut result = vec![0.0f32; n as usize];
    unsafe { cuda::memcpy_dtoh(&mut result, d_c as *const _).expect("dtoh") };

    let mut errors = 0;
    for i in 0..n as usize {
        let expected = a[i] + b[i]; // i + (1024 - i) = 1024
        if (result[i] - expected).abs() > 1e-3 { errors += 1; }
    }
    unsafe { cuda::free(d_a).ok(); cuda::free(d_b).ok(); cuda::free(d_c).ok(); }

    if errors == 0 {
        println!("  PASS  vector_add kernel (1024 elements)");
        passed += 1;
    } else {
        println!("  FAIL  vector_add kernel ({} errors)", errors);
        failed += 1;
    }
    println!();

    // ── 5. cuBLAS ────────────────────────────────────────────────────
    println!("── cuBLAS ──");

    let handle = CublasHandle::new().expect("cublas create");
    println!("  PASS  cuBLAS handle created");
    passed += 1;

    // SGEMM: C = A * I (identity multiply)
    let n = 4i32;
    let a_data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let mut eye = vec![0.0f32; 16];
    for i in 0..4 { eye[i * 4 + i] = 1.0; }

    let size = 16 * 4;
    let d_a = cuda::malloc(size).expect("malloc");
    let d_i = cuda::malloc(size).expect("malloc");
    let d_c = cuda::malloc(size).expect("malloc");

    unsafe {
        cuda::memcpy_htod(d_a, &a_data).expect("htod");
        cuda::memcpy_htod(d_i, &eye).expect("htod");
        cuda::memset(d_c, 0, size).expect("memset");

        handle.sgemm(
            CublasOperation::N, CublasOperation::N,
            n, n, n, 1.0,
            d_a as *const _, n,
            d_i as *const _, n,
            0.0, d_c, n,
        ).expect("sgemm");

        let mut result = vec![0.0f32; 16];
        cuda::memcpy_dtoh(&mut result, d_c as *const _).expect("dtoh");

        let mut ok = true;
        for i in 0..16 {
            if (result[i] - a_data[i]).abs() > 1e-3 { ok = false; break; }
        }
        cuda::free(d_a).ok(); cuda::free(d_i).ok(); cuda::free(d_c).ok();

        if ok {
            println!("  PASS  SGEMM identity multiply (4x4)");
            passed += 1;
        } else {
            println!("  FAIL  SGEMM identity multiply");
            failed += 1;
        }
    }
    drop(handle);
    println!();

    // ── 6. GpuArray (high-level API) ─────────────────────────────────
    println!("── GpuArray ──");

    // Roundtrip
    let arr = GpuArray::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("from_slice");
    assert_eq!(arr.to_vec().expect("to_vec"), vec![1.0, 2.0, 3.0, 4.0]);
    println!("  PASS  from_slice + to_vec roundtrip");
    passed += 1;

    // Zeros / ones
    let z = GpuArray::zeros(&[100], DType::Float32).expect("zeros");
    assert!(z.to_vec().expect("to_vec").iter().all(|&v| v == 0.0));
    let o = GpuArray::ones(&[100]).expect("ones");
    assert!(o.to_vec().expect("to_vec").iter().all(|&v| v == 1.0));
    println!("  PASS  zeros + ones");
    passed += 1;

    // Arithmetic
    let a = GpuArray::from_slice(&[1.0, 2.0, 3.0], &[3]).expect("a");
    let b = GpuArray::from_slice(&[4.0, 5.0, 6.0], &[3]).expect("b");
    assert_eq!(a.add(&b).expect("add").to_vec().expect("vec"), vec![5.0, 7.0, 9.0]);
    assert_eq!(a.sub(&b).expect("sub").to_vec().expect("vec"), vec![-3.0, -3.0, -3.0]);
    assert_eq!(a.mul(&b).expect("mul").to_vec().expect("vec"), vec![4.0, 10.0, 18.0]);
    assert_eq!(a.mul_scalar(10.0).expect("mul_scalar").to_vec().expect("vec"), vec![10.0, 20.0, 30.0]);
    println!("  PASS  element-wise arithmetic (add, sub, mul, mul_scalar)");
    passed += 1;

    // Matmul
    let dim = 64;
    let ones = vec![1.0f32; dim * dim];
    let a = GpuArray::from_slice(&ones, &[dim, dim]).expect("a");
    let b = GpuArray::from_slice(&ones, &[dim, dim]).expect("b");
    let c = a.matmul(&b).expect("matmul");
    let result = c.to_vec().expect("to_vec");
    assert!((result[0] - dim as f32).abs() < 0.1, "expected {}, got {}", dim, result[0]);
    println!("  PASS  matmul {}x{} (result[0] = {:.1})", dim, dim, result[0]);
    passed += 1;

    // Reshape + transpose
    let a = GpuArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).expect("a");
    let t = a.transpose().expect("transpose");
    assert_eq!(t.shape(), &[3, 2]);
    assert_eq!(t.to_vec().expect("vec"), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    println!("  PASS  transpose (2x3 -> 3x2)");
    passed += 1;
    println!();

    // ── 7. Performance Benchmark ─────────────────────────────────────
    println!("── Performance ──");

    // Memory bandwidth
    let size = 4 * 1024 * 1024; // 4 MB
    let data = vec![0u8; size];
    let ptr = cuda::malloc(size).expect("malloc");
    for _ in 0..5 { // warmup
        unsafe { cuda::memcpy_htod(ptr, &data).ok(); }
        unsafe { cuda::memcpy_dtoh(&mut vec![0u8; size], ptr as *const _).ok(); }
    }
    let t = Instant::now();
    let iters = 50;
    for _ in 0..iters {
        unsafe { cuda::memcpy_htod(ptr, &data).expect("htod"); }
        unsafe { cuda::memcpy_dtoh(&mut vec![0u8; size], ptr as *const _).expect("dtoh"); }
    }
    let elapsed = t.elapsed().as_secs_f64();
    unsafe { cuda::free(ptr).ok(); }
    let gb_s = (iters * size * 2) as f64 / elapsed / 1e9;
    println!("  Mem BW (4 MB): {:.2} GB/s", gb_s);

    // SGEMM throughput
    let dim = 512;
    let a = GpuArray::from_slice(&vec![0.5f32; dim * dim], &[dim, dim]).expect("a");
    let b = GpuArray::from_slice(&vec![0.5f32; dim * dim], &[dim, dim]).expect("b");
    let _ = a.matmul(&b).expect("warmup"); // warmup
    let t = Instant::now();
    let iters = 10;
    for _ in 0..iters { let _ = a.matmul(&b).expect("matmul"); }
    let elapsed = t.elapsed().as_secs_f64();
    let flops = 2.0 * (dim as f64).powi(3) * iters as f64;
    let gflops = flops / elapsed / 1e9;
    println!("  SGEMM {}x{}: {:.2} GFLOPS", dim, dim, gflops);
    println!();

    // ── Summary ──────────────────────────────────────────────────────
    let total_time = start.elapsed().as_secs_f64();
    println!("══════════════════════════════════════════════════════════════");
    if failed == 0 {
        println!("  ALL {} TESTS PASSED ({:.1}s)", passed, total_time);
        println!("  CUDA compatibility: VERIFIED via Invisible CUDA SDK");
    } else {
        println!("  {} passed, {} FAILED ({:.1}s)", passed, failed, total_time);
    }
    println!("══════════════════════════════════════════════════════════════");

    if failed > 0 { std::process::exit(1); }
}
