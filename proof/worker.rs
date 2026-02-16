//! Invisible CUDA — Distributed Worker Daemon
//!
//! Listens on a TCP port for coordinator connections. Receives kernels,
//! executes them on the local CPU backend, and returns results.
//!
//! Usage:
//!   worker                          # Listen on 0.0.0.0:9741
//!   worker --port 9742              # Listen on custom port
//!   worker --bind 127.0.0.1:9741   # Listen on specific address

use std::collections::HashMap;
use std::io::{Read as _, Write as _};
use std::net::{TcpListener, TcpStream};
use std::time::{SystemTime, UNIX_EPOCH};

use invisible_cuda::compute::traits::{BackendBufferId, ComputeBackend, KernelParam};
use invisible_cuda::compute::cpu::CpuComputeBackend;
use invisible_cuda::distributed::protocol::{
    FrameHeader, WireMessage, WireResponse,
    deserialize_message, serialize_response,
};
use invisible_cuda::distributed::transfer::{compress_buffer, decompress_buffer};
use invisible_cuda::ir::KernelIR;

fn main() {
    let bind_addr = parse_args();

    eprintln!("╔══════════════════════════════════════════════════════════╗");
    eprintln!("║     Invisible CUDA — Distributed Worker Daemon          ║");
    eprintln!("╚══════════════════════════════════════════════════════════╝");
    eprintln!();

    // Initialize CPU backend
    let mut backend = CpuComputeBackend::new();
    let caps = backend.capabilities();
    eprintln!("  Backend:  CPU");
    eprintln!("  Device:   {}", caps.name);
    eprintln!("  Compute:  {} units", caps.compute_units);
    eprintln!("  Memory:   system RAM");
    eprintln!("  Binding:  {}", bind_addr);
    eprintln!();

    let listener = TcpListener::bind(&bind_addr).unwrap_or_else(|e| {
        eprintln!("ERROR: Cannot bind to {}: {}", bind_addr, e);
        std::process::exit(1);
    });
    eprintln!("  Listening for coordinator connections...");

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let peer = stream.peer_addr().map(|a| a.to_string()).unwrap_or_default();
                eprintln!("  Connected: {}", peer);
                handle_connection(stream, &mut backend);
                eprintln!("  Disconnected: {}", peer);
            }
            Err(e) => {
                eprintln!("  Accept error: {}", e);
            }
        }
    }
}

fn parse_args() -> String {
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    let mut bind = String::from("0.0.0.0:9741");

    while i < args.len() {
        match args[i].as_str() {
            "--port" => {
                i += 1;
                if i < args.len() {
                    bind = format!("0.0.0.0:{}", args[i]);
                }
            }
            "--bind" => {
                i += 1;
                if i < args.len() {
                    bind = args[i].clone();
                }
            }
            "--help" | "-h" => {
                eprintln!("Usage: worker [--port PORT] [--bind ADDR:PORT]");
                eprintln!("  Default: 0.0.0.0:9741");
                std::process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }
    bind
}

fn handle_connection(mut stream: TcpStream, backend: &mut CpuComputeBackend) {
    // Track kernel names compiled on this backend
    let mut compiled_kernels: HashMap<String, KernelIR> = HashMap::new();
    let mut next_buffer_id: u64 = 1;
    let mut buffer_map: HashMap<u64, BackendBufferId> = HashMap::new();

    loop {
        // Read frame header (8 bytes)
        let mut header_buf = [0u8; 8];
        if stream.read_exact(&mut header_buf).is_err() {
            return; // Connection closed
        }

        let header = match FrameHeader::from_bytes(&header_buf) {
            Ok(h) => h,
            Err(e) => {
                eprintln!("    Frame error: {}", e);
                let _ = send_response(&mut stream, &WireResponse::Error { message: e });
                return;
            }
        };

        // Read payload
        let mut payload = vec![0u8; header.payload_len as usize];
        if stream.read_exact(&mut payload).is_err() {
            return; // Connection closed
        }

        // Deserialize message
        let msg = match deserialize_message(&payload) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("    Deserialize error: {}", e);
                let _ = send_response(&mut stream, &WireResponse::Error { message: e });
                continue;
            }
        };

        // Handle message
        let response = handle_message(
            msg,
            backend,
            &mut compiled_kernels,
            &mut next_buffer_id,
            &mut buffer_map,
        );

        if send_response(&mut stream, &response).is_err() {
            return; // Connection closed
        }
    }
}

fn handle_message(
    msg: WireMessage,
    backend: &mut CpuComputeBackend,
    compiled_kernels: &mut HashMap<String, KernelIR>,
    next_buffer_id: &mut u64,
    buffer_map: &mut HashMap<u64, BackendBufferId>,
) -> WireResponse {
    match msg {
        WireMessage::Ping => {
            let ts = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            WireResponse::Pong { timestamp: ts }
        }

        WireMessage::GetCapabilities => {
            let caps = backend.capabilities();
            WireResponse::Capabilities {
                name: caps.name,
                backend_kind: format!("{}", backend.kind()),
                total_memory: caps.total_memory,
                compute_units: caps.compute_units,
                max_threads_per_block: caps.max_threads_per_block,
            }
        }

        WireMessage::CompileKernel { name, ir_bytes } => {
            // Deserialize KernelIR from bincode
            let ir: KernelIR = match bincode::deserialize(&ir_bytes) {
                Ok(ir) => ir,
                Err(e) => {
                    return WireResponse::Error {
                        message: format!("Failed to deserialize KernelIR: {}", e),
                    };
                }
            };

            // Compile on local backend
            match backend.compile_kernel(&name, &ir) {
                Ok(()) => {
                    compiled_kernels.insert(name, ir);
                    WireResponse::Ok
                }
                Err(e) => WireResponse::Error {
                    message: format!("Compile failed: {}", e),
                },
            }
        }

        WireMessage::Alloc { size, shared } => {
            match backend.alloc(size, shared) {
                Ok((backend_id, _ptr)) => {
                    let wire_id = *next_buffer_id;
                    *next_buffer_id += 1;
                    buffer_map.insert(wire_id, backend_id);
                    WireResponse::Allocated { buffer_id: wire_id }
                }
                Err(e) => WireResponse::Error {
                    message: format!("Alloc failed: {}", e),
                },
            }
        }

        WireMessage::Free { buffer_id } => {
            match buffer_map.remove(&buffer_id) {
                Some(backend_id) => {
                    match backend.free(backend_id) {
                        Ok(()) => WireResponse::Ok,
                        Err(e) => WireResponse::Error {
                            message: format!("Free failed: {}", e),
                        },
                    }
                }
                None => WireResponse::Error {
                    message: format!("Unknown buffer ID: {}", buffer_id),
                },
            }
        }

        WireMessage::CopyHtoD { buffer_id, data, compressed, uncompressed_size } => {
            let backend_id = match buffer_map.get(&buffer_id) {
                Some(id) => *id,
                None => {
                    return WireResponse::Error {
                        message: format!("Unknown buffer ID: {}", buffer_id),
                    };
                }
            };

            // Decompress if needed
            let raw_data = match decompress_buffer(&data, compressed, uncompressed_size) {
                Ok(d) => d,
                Err(e) => {
                    return WireResponse::Error {
                        message: format!("Decompress failed: {}", e),
                    };
                }
            };

            match backend.copy_htod(backend_id, &raw_data) {
                Ok(()) => WireResponse::Ok,
                Err(e) => WireResponse::Error {
                    message: format!("CopyHtoD failed: {}", e),
                },
            }
        }

        WireMessage::CopyDtoH { buffer_id, size } => {
            let backend_id = match buffer_map.get(&buffer_id) {
                Some(id) => *id,
                None => {
                    return WireResponse::Error {
                        message: format!("Unknown buffer ID: {}", buffer_id),
                    };
                }
            };

            match backend.copy_dtoh(backend_id, size) {
                Ok(raw_data) => {
                    let (data, compressed, uncompressed_size) = compress_buffer(&raw_data);
                    WireResponse::Data { data, compressed, uncompressed_size }
                }
                Err(e) => WireResponse::Error {
                    message: format!("CopyDtoH failed: {}", e),
                },
            }
        }

        WireMessage::Launch { kernel_name, grid, block, param_buffer_ids, scalar_params } => {
            // Build KernelParam list
            let mut params: Vec<KernelParam> = param_buffer_ids
                .iter()
                .map(|wire_id| {
                    // Check if this is a scalar param position
                    KernelParam::Buffer(
                        *buffer_map.get(wire_id).unwrap_or(&BackendBufferId(0)),
                    )
                })
                .collect();

            // Override scalar params at their indices
            for (idx, scalar_bytes) in &scalar_params {
                if *idx < params.len() {
                    params[*idx] = KernelParam::Scalar(scalar_bytes.clone());
                }
            }

            match backend.launch(&kernel_name, grid, block, &params) {
                Ok(()) => WireResponse::Ok,
                Err(e) => WireResponse::Error {
                    message: format!("Launch failed: {}", e),
                },
            }
        }

        WireMessage::Synchronize => {
            match backend.synchronize() {
                Ok(()) => WireResponse::Ok,
                Err(e) => WireResponse::Error {
                    message: format!("Synchronize failed: {}", e),
                },
            }
        }

        WireMessage::Shutdown => {
            // Clean up all buffers
            for (_wire_id, backend_id) in buffer_map.drain() {
                let _ = backend.free(backend_id);
            }
            compiled_kernels.clear();
            eprintln!("    Shutdown requested");
            WireResponse::Ok
        }
    }
}

fn send_response(stream: &mut TcpStream, response: &WireResponse) -> Result<(), String> {
    let payload = serialize_response(response)
        .map_err(|e| format!("Serialize response failed: {}", e))?;

    let header = FrameHeader {
        magic: FrameHeader::MAGIC,
        payload_len: payload.len() as u32,
    };

    stream.write_all(&header.to_bytes())
        .map_err(|e| format!("Write header failed: {}", e))?;
    stream.write_all(&payload)
        .map_err(|e| format!("Write payload failed: {}", e))?;
    stream.flush()
        .map_err(|e| format!("Flush failed: {}", e))?;

    Ok(())
}
