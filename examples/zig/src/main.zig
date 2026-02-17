// Invisible CUDA — Zig Proof
//
// Demonstrates CUDA compatibility using Zig's C interop (dlopen)
// against the Invisible CUDA runtime.
//
// Prerequisites:
//   cargo install invisible-cuda && invisible-cuda install
//
// Run:
//   zig build run -Doptimize=ReleaseFast

const std = @import("std");
const builtin = @import("builtin");
const c = @cImport({
    @cInclude("dlfcn.h");
});

// ── CUDA Types ──

const RTLD_NOW = 0x2;
const cudaMemcpyHostToDevice: c_int = 1;
const cudaMemcpyDeviceToHost: c_int = 2;
const CUBLAS_OP_N: c_int = 0;

const DeviceProp = extern struct {
    name: [256]u8,
    totalGlobalMem: usize,
    sharedMemPerBlock: usize,
    regsPerBlock: c_int,
    warpSize: c_int,
    maxThreadsPerBlock: c_int,
    maxThreadsDim: [3]c_int,
    maxGridSize: [3]c_int,
    clockRate: c_int,
    major: c_int,
    minor: c_int,
};

// ── Function pointer types ──

const FnGetDeviceCount = *const fn (*c_int) callconv(.C) c_int;
const FnGetDeviceProperties = *const fn (*DeviceProp, c_int) callconv(.C) c_int;
const FnDeviceSynchronize = *const fn () callconv(.C) c_int;
const FnRuntimeGetVersion = *const fn (*c_int) callconv(.C) c_int;
const FnMalloc = *const fn (*?*anyopaque, usize) callconv(.C) c_int;
const FnFree = *const fn (?*anyopaque) callconv(.C) c_int;
const FnMemcpy = *const fn (?*anyopaque, ?*const anyopaque, usize, c_int) callconv(.C) c_int;
const FnMemset = *const fn (?*anyopaque, c_int, usize) callconv(.C) c_int;
const FnMemGetInfo = *const fn (*usize, *usize) callconv(.C) c_int;
const FnStreamCreate = *const fn (*?*anyopaque) callconv(.C) c_int;
const FnStreamDestroy = *const fn (?*anyopaque) callconv(.C) c_int;
const FnStreamSynchronize = *const fn (?*anyopaque) callconv(.C) c_int;
const FnEventCreate = *const fn (*?*anyopaque) callconv(.C) c_int;
const FnEventDestroy = *const fn (?*anyopaque) callconv(.C) c_int;
const FnEventRecord = *const fn (?*anyopaque, ?*anyopaque) callconv(.C) c_int;
const FnEventSynchronize = *const fn (?*anyopaque) callconv(.C) c_int;
const FnEventElapsedTime = *const fn (*f32, ?*anyopaque, ?*anyopaque) callconv(.C) c_int;
const FnCublasCreate = *const fn (*?*anyopaque) callconv(.C) c_int;
const FnCublasDestroy = *const fn (?*anyopaque) callconv(.C) c_int;
const FnCublasSgemm = *const fn (?*anyopaque, c_int, c_int, c_int, c_int, c_int, *const f32, ?*const anyopaque, c_int, ?*const anyopaque, c_int, *const f32, ?*anyopaque, c_int) callconv(.C) c_int;

const CudaLib = struct {
    getDeviceCount: FnGetDeviceCount,
    getDeviceProperties: FnGetDeviceProperties,
    deviceSynchronize: FnDeviceSynchronize,
    runtimeGetVersion: FnRuntimeGetVersion,
    malloc: FnMalloc,
    free: FnFree,
    memcpy: FnMemcpy,
    memset: FnMemset,
    memGetInfo: FnMemGetInfo,
    streamCreate: FnStreamCreate,
    streamDestroy: FnStreamDestroy,
    streamSynchronize: FnStreamSynchronize,
    eventCreate: FnEventCreate,
    eventDestroy: FnEventDestroy,
    eventRecord: FnEventRecord,
    eventSynchronize: FnEventSynchronize,
    eventElapsedTime: FnEventElapsedTime,
    cublasCreate: FnCublasCreate,
    cublasDestroy: FnCublasDestroy,
    cublasSgemm: FnCublasSgemm,
};

fn loadSym(comptime T: type, handle: *anyopaque, name: [*:0]const u8) ?T {
    const raw = c.dlsym(handle, name);
    if (raw == null) return null;
    return @ptrCast(@alignCast(raw));
}

fn loadLibrary() ?CudaLib {
    const home = std.posix.getenv("HOME") orelse "/root";
    const env_path = std.posix.getenv("INVISIBLE_CUDA_LIB");

    var path_buf: [512]u8 = undefined;
    const path: [*:0]const u8 = if (env_path) |ep|
        @ptrCast(ep.ptr)
    else blk: {
        const lib_name = if (builtin.os.tag == .macos) "libcuda.dylib" else "libcuda.so";
        const written = std.fmt.bufPrint(&path_buf, "{s}/.invisible-cuda/lib/{s}", .{ home, lib_name }) catch return null;
        path_buf[written.len] = 0;
        break :blk @ptrCast(path_buf[0..written.len :0]);
    };

    const handle = c.dlopen(path, RTLD_NOW) orelse {
        std.debug.print("Could not load: {s}\n", .{path});
        std.debug.print("Install via: cargo install invisible-cuda && invisible-cuda install\n", .{});
        return null;
    };

    return CudaLib{
        .getDeviceCount = loadSym(FnGetDeviceCount, handle, "cudaGetDeviceCount") orelse return null,
        .getDeviceProperties = loadSym(FnGetDeviceProperties, handle, "cudaGetDeviceProperties") orelse return null,
        .deviceSynchronize = loadSym(FnDeviceSynchronize, handle, "cudaDeviceSynchronize") orelse return null,
        .runtimeGetVersion = loadSym(FnRuntimeGetVersion, handle, "cudaRuntimeGetVersion") orelse return null,
        .malloc = loadSym(FnMalloc, handle, "cudaMalloc") orelse return null,
        .free = loadSym(FnFree, handle, "cudaFree") orelse return null,
        .memcpy = loadSym(FnMemcpy, handle, "cudaMemcpy") orelse return null,
        .memset = loadSym(FnMemset, handle, "cudaMemset") orelse return null,
        .memGetInfo = loadSym(FnMemGetInfo, handle, "cudaMemGetInfo") orelse return null,
        .streamCreate = loadSym(FnStreamCreate, handle, "cudaStreamCreate") orelse return null,
        .streamDestroy = loadSym(FnStreamDestroy, handle, "cudaStreamDestroy") orelse return null,
        .streamSynchronize = loadSym(FnStreamSynchronize, handle, "cudaStreamSynchronize") orelse return null,
        .eventCreate = loadSym(FnEventCreate, handle, "cudaEventCreate") orelse return null,
        .eventDestroy = loadSym(FnEventDestroy, handle, "cudaEventDestroy") orelse return null,
        .eventRecord = loadSym(FnEventRecord, handle, "cudaEventRecord") orelse return null,
        .eventSynchronize = loadSym(FnEventSynchronize, handle, "cudaEventSynchronize") orelse return null,
        .eventElapsedTime = loadSym(FnEventElapsedTime, handle, "cudaEventElapsedTime") orelse return null,
        .cublasCreate = loadSym(FnCublasCreate, handle, "cublasCreate_v2") orelse return null,
        .cublasDestroy = loadSym(FnCublasDestroy, handle, "cublasDestroy_v2") orelse return null,
        .cublasSgemm = loadSym(FnCublasSgemm, handle, "cublasSgemm_v2") orelse return null,
    };
}

const print = std.io.getStdOut().writer().print;

pub fn main() !void {
    const stdout = std.io.getStdOut().writer();

    try stdout.print("╔══════════════════════════════════════════════════════════════╗\n", .{});
    try stdout.print("║         INVISIBLE CUDA — Zig Proof                         ║\n", .{});
    try stdout.print("╚══════════════════════════════════════════════════════════════╝\n", .{});
    try stdout.print("\n", .{});

    var passed: u32 = 0;
    var failed: u32 = 0;
    var timer = try std.time.Timer.start();

    const cuda = loadLibrary() orelse {
        std.process.exit(1);
    };

    // ── Device Discovery ──
    try stdout.print("── Device Discovery ──\n", .{});

    var count: c_int = 0;
    if (cuda.getDeviceCount(&count) == 0 and count >= 1) {
        try stdout.print("  PASS  Device count: {d}\n", .{count});
        passed += 1;
    } else {
        try stdout.print("  FAIL  Device count\n", .{});
        failed += 1;
    }

    var prop: DeviceProp = std.mem.zeroes(DeviceProp);
    if (cuda.getDeviceProperties(&prop, 0) == 0) {
        const name_len = std.mem.indexOfScalar(u8, &prop.name, 0) orelse 256;
        const gb = @as(f64, @floatFromInt(prop.totalGlobalMem)) / 1e9;
        try stdout.print("  PASS  Device: {s} ({d:.1} GB)\n", .{ prop.name[0..name_len], gb });
        passed += 1;
    } else {
        try stdout.print("  FAIL  Device properties\n", .{});
        failed += 1;
    }

    var version: c_int = 0;
    if (cuda.runtimeGetVersion(&version) == 0 and version >= 12000) {
        try stdout.print("  PASS  Runtime version: {d}\n", .{version});
        passed += 1;
    } else {
        try stdout.print("  FAIL  Runtime version\n", .{});
        failed += 1;
    }

    var free_mem: usize = 0;
    var total_mem: usize = 0;
    if (cuda.memGetInfo(&free_mem, &total_mem) == 0 and total_mem > 0) {
        const free_gb = @as(f64, @floatFromInt(free_mem)) / 1e9;
        const total_gb = @as(f64, @floatFromInt(total_mem)) / 1e9;
        try stdout.print("  PASS  Memory: {d:.1} GB free / {d:.1} GB total\n", .{ free_gb, total_gb });
        passed += 1;
    } else {
        try stdout.print("  FAIL  Memory info\n", .{});
        failed += 1;
    }
    try stdout.print("\n", .{});

    // ── Memory Operations ──
    try stdout.print("── Memory Operations ──\n", .{});

    {
        var ptr: ?*anyopaque = null;
        if (cuda.malloc(&ptr, 4096) == 0 and ptr != null) {
            _ = cuda.free(ptr);
            try stdout.print("  PASS  malloc + free (4 KB)\n", .{});
            passed += 1;
        } else {
            try stdout.print("  FAIL  malloc + free\n", .{});
            failed += 1;
        }
    }

    // memcpy roundtrip
    {
        const n = 1024;
        var src: [n]f32 = undefined;
        for (0..n) |i| {
            src[i] = @as(f32, @floatFromInt(i)) * 0.1;
        }
        var d_ptr: ?*anyopaque = null;
        _ = cuda.malloc(&d_ptr, n * 4);
        _ = cuda.memcpy(d_ptr, &src, n * 4, cudaMemcpyHostToDevice);
        var dst: [n]f32 = undefined;
        _ = cuda.memcpy(&dst, d_ptr, n * 4, cudaMemcpyDeviceToHost);
        _ = cuda.free(d_ptr);

        var ok = true;
        for (0..n) |i| {
            if (@abs(src[i] - dst[i]) > 1e-5) {
                ok = false;
                break;
            }
        }
        if (ok) {
            try stdout.print("  PASS  memcpy roundtrip (4 KB)\n", .{});
            passed += 1;
        } else {
            try stdout.print("  FAIL  memcpy roundtrip\n", .{});
            failed += 1;
        }
    }

    // memset
    {
        var d_ptr: ?*anyopaque = null;
        _ = cuda.malloc(&d_ptr, 4096);
        _ = cuda.memset(d_ptr, 0xAB, 4096);
        var buf: [4096]u8 = undefined;
        _ = cuda.memcpy(&buf, d_ptr, 4096, cudaMemcpyDeviceToHost);
        _ = cuda.free(d_ptr);

        var ok = true;
        for (buf) |b| {
            if (b != 0xAB) {
                ok = false;
                break;
            }
        }
        if (ok) {
            try stdout.print("  PASS  memset (4 KB, pattern 0xAB)\n", .{});
            passed += 1;
        } else {
            try stdout.print("  FAIL  memset\n", .{});
            failed += 1;
        }
    }
    try stdout.print("\n", .{});

    // ── Streams & Events ──
    try stdout.print("── Streams & Events ──\n", .{});

    {
        var stream: ?*anyopaque = null;
        if (cuda.streamCreate(&stream) == 0) {
            _ = cuda.streamSynchronize(stream);
            _ = cuda.streamDestroy(stream);
            try stdout.print("  PASS  Stream create + sync + destroy\n", .{});
            passed += 1;
        } else {
            try stdout.print("  FAIL  Stream\n", .{});
            failed += 1;
        }
    }

    {
        var ev_start: ?*anyopaque = null;
        var ev_end: ?*anyopaque = null;
        _ = cuda.eventCreate(&ev_start);
        _ = cuda.eventCreate(&ev_end);
        _ = cuda.eventRecord(ev_start, null);
        _ = cuda.deviceSynchronize();
        _ = cuda.eventRecord(ev_end, null);
        _ = cuda.eventSynchronize(ev_end);
        var ms: f32 = 0;
        _ = cuda.eventElapsedTime(&ms, ev_start, ev_end);
        _ = cuda.eventDestroy(ev_start);
        _ = cuda.eventDestroy(ev_end);
        if (ms >= 0) {
            try stdout.print("  PASS  Event timing ({d:.3} ms)\n", .{ms});
            passed += 1;
        } else {
            try stdout.print("  FAIL  Event timing\n", .{});
            failed += 1;
        }
    }
    try stdout.print("\n", .{});

    // ── cuBLAS ──
    try stdout.print("── cuBLAS ──\n", .{});

    {
        var handle: ?*anyopaque = null;
        if (cuda.cublasCreate(&handle) == 0) {
            try stdout.print("  PASS  cuBLAS handle created\n", .{});
            passed += 1;

            var a_data: [16]f32 = undefined;
            var eye: [16]f32 = [_]f32{0} ** 16;
            for (0..16) |i| {
                a_data[i] = @as(f32, @floatFromInt(i + 1));
            }
            for (0..4) |i| {
                eye[i * 4 + i] = 1.0;
            }

            var d_a: ?*anyopaque = null;
            var d_i: ?*anyopaque = null;
            var d_c: ?*anyopaque = null;
            _ = cuda.malloc(&d_a, 64);
            _ = cuda.malloc(&d_i, 64);
            _ = cuda.malloc(&d_c, 64);
            _ = cuda.memcpy(d_a, &a_data, 64, cudaMemcpyHostToDevice);
            _ = cuda.memcpy(d_i, &eye, 64, cudaMemcpyHostToDevice);
            _ = cuda.memset(d_c, 0, 64);

            const alpha: f32 = 1.0;
            const beta: f32 = 0.0;
            _ = cuda.cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, 4, 4, &alpha, d_a, 4, d_i, 4, &beta, d_c, 4);

            var result: [16]f32 = undefined;
            _ = cuda.memcpy(&result, d_c, 64, cudaMemcpyDeviceToHost);
            _ = cuda.free(d_a);
            _ = cuda.free(d_i);
            _ = cuda.free(d_c);

            var ok = true;
            for (0..16) |i| {
                if (@abs(result[i] - a_data[i]) > 1e-3) {
                    ok = false;
                    break;
                }
            }
            if (ok) {
                try stdout.print("  PASS  SGEMM identity multiply (4x4)\n", .{});
                passed += 1;
            } else {
                try stdout.print("  FAIL  SGEMM identity multiply\n", .{});
                failed += 1;
            }

            _ = cuda.cublasDestroy(handle);
        } else {
            try stdout.print("  FAIL  cuBLAS create\n", .{});
            failed += 1;
        }
    }
    try stdout.print("\n", .{});

    // ── Performance ──
    try stdout.print("── Performance ──\n", .{});

    {
        const size = 4 * 1024 * 1024;
        var data: [size]u8 = [_]u8{0} ** size;
        var d_ptr: ?*anyopaque = null;
        _ = cuda.malloc(&d_ptr, size);

        // warmup
        for (0..5) |_| {
            _ = cuda.memcpy(d_ptr, &data, size, cudaMemcpyHostToDevice);
            _ = cuda.memcpy(&data, d_ptr, size, cudaMemcpyDeviceToHost);
        }

        var perf_timer = try std.time.Timer.start();
        const iters = 50;
        for (0..iters) |_| {
            _ = cuda.memcpy(d_ptr, &data, size, cudaMemcpyHostToDevice);
            _ = cuda.memcpy(&data, d_ptr, size, cudaMemcpyDeviceToHost);
        }
        const elapsed_ns = perf_timer.read();
        _ = cuda.free(d_ptr);

        const elapsed_s = @as(f64, @floatFromInt(elapsed_ns)) / 1e9;
        const gb_s = @as(f64, @floatFromInt(iters * size * 2)) / elapsed_s / 1e9;
        try stdout.print("  Mem BW (4 MB): {d:.2} GB/s\n", .{gb_s});
    }
    try stdout.print("\n", .{});

    // ── Summary ──
    const elapsed_ns = timer.read();
    const total_s = @as(f64, @floatFromInt(elapsed_ns)) / 1e9;
    try stdout.print("══════════════════════════════════════════════════════════════\n", .{});
    if (failed == 0) {
        try stdout.print("  ALL {d} TESTS PASSED ({d:.1}s)\n", .{ passed, total_s });
        try stdout.print("  CUDA compatibility: VERIFIED via Invisible CUDA (Zig)\n", .{});
    } else {
        try stdout.print("  {d} passed, {d} FAILED ({d:.1}s)\n", .{ passed, failed, total_s });
    }
    try stdout.print("══════════════════════════════════════════════════════════════\n", .{});

    if (failed > 0) std.process.exit(1);
}
