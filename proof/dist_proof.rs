//! Invisible CUDA — Distributed Multi-Node Proof
//!
//! Connects to worker daemons, distributes a compute workload across
//! the fleet, verifies results, and reports throughput.
//!
//! Usage:
//!   dist_proof worker1:9741 worker2:9741 ...
//!
//! The proof:
//! 1. Connects to all workers
//! 2. Queries capabilities
//! 3. Compiles a vector-add kernel on each worker
//! 4. Distributes data, executes partitioned grid, collects results
//! 5. Verifies correctness of merged output
//! 6. Reports aggregate throughput

use std::io::{Read as _, Write as _};
use std::net::TcpStream;
use std::time::Instant;

use invisible_cuda::distributed::protocol::{
    FrameHeader, WireMessage, WireResponse,
    serialize_message, deserialize_response,
};
use invisible_cuda::distributed::transfer::compress_buffer;
use invisible_cuda::distributed::partition::split_grid_x;
use invisible_cuda::ir::{KernelIR, Op};

fn main() {
    let workers: Vec<String> = std::env::args().skip(1).collect();

    if workers.is_empty() {
        eprintln!("Usage: dist_proof <worker1:port> [worker2:port] ...");
        eprintln!("  Example: dist_proof 10.0.1.10:9741 10.0.1.11:9741");
        std::process::exit(1);
    }

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  Invisible CUDA — Distributed Multi-Node Proof          ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    let total_start = Instant::now();
    let mut total_pass = 0;
    let mut total_fail = 0;

    // ── Step 1: Connect to all workers ──────────────────────────
    println!("── Connecting to {} workers ─────────────────────────────", workers.len());
    let mut connections: Vec<(String, TcpStream)> = Vec::new();

    for addr in &workers {
        match TcpStream::connect(addr) {
            Ok(stream) => {
                println!("  OK  {}", addr);
                connections.push((addr.clone(), stream));
            }
            Err(e) => {
                println!("  FAIL  {} — {}", addr, e);
                total_fail += 1;
            }
        }
    }

    if connections.is_empty() {
        println!("\n  ERROR: No workers available. Aborting.");
        std::process::exit(1);
    }

    // ── Step 2: Query capabilities ──────────────────────────────
    println!();
    println!("── Worker Capabilities ─────────────────────────────────");
    let mut total_compute_units: u32 = 0;

    for (addr, stream) in &mut connections {
        match send_recv(stream, &WireMessage::GetCapabilities) {
            Ok(WireResponse::Capabilities { name, backend_kind, compute_units, total_memory, .. }) => {
                println!("  {}  {} ({}) — {} CUs, {} MB",
                    addr, name, backend_kind, compute_units,
                    total_memory / (1024 * 1024));
                total_compute_units += compute_units;
            }
            Ok(other) => println!("  {}  unexpected: {:?}", addr, other),
            Err(e) => println!("  {}  error: {}", addr, e),
        }
    }
    println!("  Total: {} compute units across {} workers", total_compute_units, connections.len());

    // ── Step 3: Ping latency ────────────────────────────────────
    println!();
    println!("── Ping Latency ───────────────────────────────────────");
    for (addr, stream) in &mut connections {
        let start = Instant::now();
        match send_recv(stream, &WireMessage::Ping) {
            Ok(WireResponse::Pong { .. }) => {
                let us = start.elapsed().as_micros();
                println!("  {}  {} us", addr, us);
            }
            _ => println!("  {}  failed", addr),
        }
    }

    // ── Step 4: Compile vector-add kernel on all workers ────────
    println!();
    println!("── Compile Kernel ─────────────────────────────────────");
    let ir = build_vector_add_ir();
    let ir_bytes = bincode::serialize(&ir).expect("Failed to serialize IR");

    let mut compile_ok = 0;
    for (addr, stream) in &mut connections {
        let msg = WireMessage::CompileKernel {
            name: "vector_add".into(),
            ir_bytes: ir_bytes.clone(),
        };
        match send_recv(stream, &msg) {
            Ok(WireResponse::Ok) => {
                println!("  {}  compiled", addr);
                compile_ok += 1;
            }
            Ok(WireResponse::Error { message }) => {
                println!("  {}  FAIL: {}", addr, message);
            }
            other => println!("  {}  unexpected: {:?}", addr, other),
        }
    }
    report_test("Compile kernel on all workers", compile_ok == connections.len(),
                &mut total_pass, &mut total_fail);

    // ── Step 5: Distributed vector add ──────────────────────────
    println!();
    println!("── Distributed Vector Add ──────────────────────────────");

    let n: usize = 65536;
    let n_workers = connections.len();

    // Create test data
    let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();
    let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();

    // Partition the grid
    let block_size = 256u32;
    let total_blocks = ((n as u32) + block_size - 1) / block_size;
    let grid = (total_blocks, 1, 1);
    let block = (block_size, 1, 1);
    let partitions = split_grid_x(grid, block, n_workers);

    println!("  N = {} elements, {} workers, {} blocks/worker",
        n, n_workers, total_blocks / n_workers as u32);

    let dist_start = Instant::now();

    // For each worker, allocate buffers, upload data, launch partitioned kernel
    let mut worker_results: Vec<(usize, usize, Vec<f32>)> = Vec::new(); // (start_elem, count, data)

    for (i, (addr, stream)) in connections.iter_mut().enumerate() {
        let partition = &partitions[i];
        let start_elem = partition.grid_offset.0 as usize * block_size as usize;
        let end_elem = (start_elem + partition.grid.0 as usize * block_size as usize).min(n);
        let count = end_elem - start_elem;

        if count == 0 {
            continue;
        }

        // Allocate 3 buffers on worker (a_slice, b_slice, c_slice)
        let a_id = alloc_on_worker(stream, count * 4);
        let b_id = alloc_on_worker(stream, count * 4);
        let c_id = alloc_on_worker(stream, count * 4);

        if a_id.is_none() || b_id.is_none() || c_id.is_none() {
            println!("  {}  FAIL: buffer allocation", addr);
            total_fail += 1;
            continue;
        }
        let (a_id, b_id, c_id) = (a_id.unwrap(), b_id.unwrap(), c_id.unwrap());

        // Upload A and B slices
        let a_slice: Vec<u8> = a[start_elem..end_elem].iter()
            .flat_map(|f| f.to_le_bytes()).collect();
        let b_slice: Vec<u8> = b[start_elem..end_elem].iter()
            .flat_map(|f| f.to_le_bytes()).collect();

        upload_to_worker(stream, a_id, &a_slice);
        upload_to_worker(stream, b_id, &b_slice);

        // Launch kernel with partition grid (grid is relative to worker's slice)
        let partition_blocks = ((count as u32) + block_size - 1) / block_size;
        let launch_msg = WireMessage::Launch {
            kernel_name: "vector_add".into(),
            grid: (partition_blocks, 1, 1),
            block: (block_size, 1, 1),
            param_buffer_ids: vec![a_id, b_id, c_id],
            scalar_params: vec![(3, (count as u32).to_le_bytes().to_vec())],
        };

        match send_recv(stream, &launch_msg) {
            Ok(WireResponse::Ok) => {}
            Ok(WireResponse::Error { message }) => {
                println!("  {}  launch FAIL: {}", addr, message);
                total_fail += 1;
                continue;
            }
            _ => { continue; }
        }

        // Synchronize
        let _ = send_recv(stream, &WireMessage::Synchronize);

        // Download result
        let c_data = download_from_worker(stream, c_id, count * 4);
        match c_data {
            Some(data) => {
                let result: Vec<f32> = data.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                worker_results.push((start_elem, count, result));
            }
            None => {
                println!("  {}  download FAIL", addr);
                total_fail += 1;
            }
        }

        // Free buffers
        let _ = send_recv(stream, &WireMessage::Free { buffer_id: a_id });
        let _ = send_recv(stream, &WireMessage::Free { buffer_id: b_id });
        let _ = send_recv(stream, &WireMessage::Free { buffer_id: c_id });
    }

    let dist_elapsed = dist_start.elapsed();

    // Merge and verify results
    let mut merged = vec![0.0f32; n];
    for (start, count, data) in &worker_results {
        merged[*start..*start + *count].copy_from_slice(data);
    }

    let correct = merged.iter().zip(expected.iter())
        .all(|(a, b)| (a - b).abs() < 1e-6);

    let elements_processed: usize = worker_results.iter().map(|(_, c, _)| c).sum();
    let throughput_gb = (elements_processed * 4 * 3) as f64 / dist_elapsed.as_secs_f64() / 1e9;

    report_test(
        &format!("Distributed VecAdd {} elements across {} workers", n, n_workers),
        correct && elements_processed == n,
        &mut total_pass, &mut total_fail,
    );
    println!("    Time: {:.1}ms, Throughput: {:.2} GB/s (including network)",
        dist_elapsed.as_secs_f64() * 1000.0, throughput_gb);

    // ── Step 6: Distributed BLAS-style large vector ─────────────
    println!();
    println!("── Large Distributed Vector Add (1M elements) ───────────");

    let large_n: usize = 1_048_576;
    let large_a: Vec<f32> = (0..large_n).map(|i| (i as f32) * 0.001).collect();
    let large_b: Vec<f32> = (0..large_n).map(|i| 1.0 - (i as f32) * 0.001).collect();
    let large_expected: Vec<f32> = large_a.iter().zip(large_b.iter())
        .map(|(x, y)| x + y).collect();

    let large_blocks = ((large_n as u32) + block_size - 1) / block_size;
    let large_partitions = split_grid_x((large_blocks, 1, 1), (block_size, 1, 1), n_workers);

    let large_start = Instant::now();
    let mut large_results: Vec<(usize, usize, Vec<f32>)> = Vec::new();

    for (i, (_addr, stream)) in connections.iter_mut().enumerate() {
        let partition = &large_partitions[i];
        let start_elem = partition.grid_offset.0 as usize * block_size as usize;
        let end_elem = (start_elem + partition.grid.0 as usize * block_size as usize).min(large_n);
        let count = end_elem - start_elem;

        if count == 0 { continue; }

        let a_id = alloc_on_worker(stream, count * 4).unwrap_or(0);
        let b_id = alloc_on_worker(stream, count * 4).unwrap_or(0);
        let c_id = alloc_on_worker(stream, count * 4).unwrap_or(0);

        let a_bytes: Vec<u8> = large_a[start_elem..end_elem].iter()
            .flat_map(|f| f.to_le_bytes()).collect();
        let b_bytes: Vec<u8> = large_b[start_elem..end_elem].iter()
            .flat_map(|f| f.to_le_bytes()).collect();

        upload_to_worker(stream, a_id, &a_bytes);
        upload_to_worker(stream, b_id, &b_bytes);

        let partition_blocks = ((count as u32) + block_size - 1) / block_size;
        let _ = send_recv(stream, &WireMessage::Launch {
            kernel_name: "vector_add".into(),
            grid: (partition_blocks, 1, 1),
            block: (block_size, 1, 1),
            param_buffer_ids: vec![a_id, b_id, c_id],
            scalar_params: vec![(3, (count as u32).to_le_bytes().to_vec())],
        });
        let _ = send_recv(stream, &WireMessage::Synchronize);

        if let Some(data) = download_from_worker(stream, c_id, count * 4) {
            let result: Vec<f32> = data.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            large_results.push((start_elem, count, result));
        }

        let _ = send_recv(stream, &WireMessage::Free { buffer_id: a_id });
        let _ = send_recv(stream, &WireMessage::Free { buffer_id: b_id });
        let _ = send_recv(stream, &WireMessage::Free { buffer_id: c_id });
    }

    let large_elapsed = large_start.elapsed();

    let mut large_merged = vec![0.0f32; large_n];
    for (start, count, data) in &large_results {
        large_merged[*start..*start + *count].copy_from_slice(data);
    }

    let large_correct = large_merged.iter().zip(large_expected.iter())
        .all(|(a, b)| (a - b).abs() < 1e-3);
    let large_processed: usize = large_results.iter().map(|(_, c, _)| c).sum();
    let large_throughput = (large_processed * 4 * 3) as f64 / large_elapsed.as_secs_f64() / 1e9;

    report_test(
        &format!("Large VecAdd {}M elements across {} workers",
            large_n / 1_000_000, n_workers),
        large_correct && large_processed == large_n,
        &mut total_pass, &mut total_fail,
    );
    println!("    Time: {:.1}ms, Throughput: {:.2} GB/s",
        large_elapsed.as_secs_f64() * 1000.0, large_throughput);

    // ── Step 7: Shutdown workers ────────────────────────────────
    println!();
    println!("── Shutting down workers ───────────────────────────────");
    for (addr, stream) in &mut connections {
        match send_recv(stream, &WireMessage::Shutdown) {
            Ok(WireResponse::Ok) => println!("  {}  shutdown OK", addr),
            _ => println!("  {}  shutdown failed (may already be down)", addr),
        }
    }

    // ── Summary ─────────────────────────────────────────────────
    let total_elapsed = total_start.elapsed();
    println!();
    println!("══════════════════════════════════════════════════════════");
    if total_fail == 0 {
        println!("  RESULT: ALL {} TESTS PASSED", total_pass);
        println!("  Distributed compute across {} workers: VERIFIED", n_workers);
    } else {
        println!("  RESULT: {} PASSED, {} FAILED", total_pass, total_fail);
    }
    println!("  Total time: {:.1}s", total_elapsed.as_secs_f64());
    println!("══════════════════════════════════════════════════════════");

    // JSON output for machine parsing
    println!();
    println!("--- JSON_START ---");
    println!("{{");
    println!("  \"format\": \"invisible-cuda-dist-proof\",");
    println!("  \"workers\": {},", n_workers);
    println!("  \"total_compute_units\": {},", total_compute_units);
    println!("  \"tests_passed\": {},", total_pass);
    println!("  \"tests_failed\": {},", total_fail);
    println!("  \"distributed_vecadd_64k_ms\": {:.1},", dist_elapsed.as_secs_f64() * 1000.0);
    println!("  \"distributed_vecadd_64k_gbps\": {:.2},", throughput_gb);
    println!("  \"distributed_vecadd_1m_ms\": {:.1},", large_elapsed.as_secs_f64() * 1000.0);
    println!("  \"distributed_vecadd_1m_gbps\": {:.2},", large_throughput);
    println!("  \"total_time_s\": {:.1}", total_elapsed.as_secs_f64());
    println!("}}");
    println!("--- JSON_END ---");

    std::process::exit(if total_fail > 0 { 1 } else { 0 });
}

// ── Helpers ─────────────────────────────────────────────────────

fn send_recv(stream: &mut TcpStream, msg: &WireMessage) -> Result<WireResponse, String> {
    let payload = serialize_message(msg)?;
    let header = FrameHeader {
        magic: FrameHeader::MAGIC,
        payload_len: payload.len() as u32,
    };

    stream.write_all(&header.to_bytes())
        .map_err(|e| format!("Write failed: {}", e))?;
    stream.write_all(&payload)
        .map_err(|e| format!("Write failed: {}", e))?;
    stream.flush()
        .map_err(|e| format!("Flush failed: {}", e))?;

    let mut resp_header_buf = [0u8; 8];
    stream.read_exact(&mut resp_header_buf)
        .map_err(|e| format!("Read header failed: {}", e))?;
    let resp_header = FrameHeader::from_bytes(&resp_header_buf)?;

    let mut resp_payload = vec![0u8; resp_header.payload_len as usize];
    stream.read_exact(&mut resp_payload)
        .map_err(|e| format!("Read payload failed: {}", e))?;

    deserialize_response(&resp_payload)
}

fn alloc_on_worker(stream: &mut TcpStream, size: usize) -> Option<u64> {
    match send_recv(stream, &WireMessage::Alloc { size, shared: false }) {
        Ok(WireResponse::Allocated { buffer_id }) => Some(buffer_id),
        _ => None,
    }
}

fn upload_to_worker(stream: &mut TcpStream, buffer_id: u64, data: &[u8]) {
    let (compressed_data, was_compressed, uncompressed_size) = compress_buffer(data);
    let _ = send_recv(stream, &WireMessage::CopyHtoD {
        buffer_id,
        data: compressed_data,
        compressed: was_compressed,
        uncompressed_size,
    });
}

fn download_from_worker(stream: &mut TcpStream, buffer_id: u64, size: usize) -> Option<Vec<u8>> {
    match send_recv(stream, &WireMessage::CopyDtoH { buffer_id, size }) {
        Ok(WireResponse::Data { data, compressed, uncompressed_size }) => {
            invisible_cuda::distributed::transfer::decompress_buffer(
                &data, compressed, uncompressed_size
            ).ok()
        }
        _ => None,
    }
}

fn report_test(name: &str, passed: bool, total_pass: &mut u32, total_fail: &mut u32) {
    if passed {
        *total_pass += 1;
        println!("  PASS  {}", name);
    } else {
        *total_fail += 1;
        println!("  FAIL  {}", name);
    }
}

/// Build a vector-add kernel: c[i] = a[i] + b[i]
/// Same as proof.rs build_vector_add_ir — uses push_op pattern.
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
