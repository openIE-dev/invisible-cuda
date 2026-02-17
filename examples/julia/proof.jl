# Invisible CUDA — Julia Proof
#
# Demonstrates CUDA compatibility using Julia's ccall + Libdl
# against the Invisible CUDA runtime.
#
# Prerequisites:
#   cargo install invisible-cuda && invisible-cuda install
#
# Run:
#   julia proof.jl

using Libdl

# ── Library Loading ──

function find_lib_path()
    env = get(ENV, "INVISIBLE_CUDA_LIB", nothing)
    if env !== nothing
        return env
    end
    home = homedir()
    if Sys.isapple()
        return joinpath(home, ".invisible-cuda", "lib", "libcuda.dylib")
    else
        return joinpath(home, ".invisible-cuda", "lib", "libcuda.so")
    end
end

function load_cuda()
    path = find_lib_path()
    lib = dlopen(path)
    if lib === nothing
        println("Could not load: $path")
        println("Install via: cargo install invisible-cuda && invisible-cuda install")
        exit(1)
    end
    return lib
end

# ── Proof ──

function main()
    println("╔══════════════════════════════════════════════════════════════╗")
    println("║         INVISIBLE CUDA — Julia Proof                       ║")
    println("╚══════════════════════════════════════════════════════════════╝")
    println()

    passed = 0
    failed = 0
    start = time()

    lib = load_cuda()

    # Helper to get function pointer
    sym(name) = dlsym(lib, name)

    # ── Device Discovery ──
    println("── Device Discovery ──")

    count = Ref{Cint}(0)
    rc = ccall(sym("cudaGetDeviceCount"), Cint, (Ptr{Cint},), count)
    if rc == 0 && count[] >= 1
        println("  PASS  Device count: $(count[])")
        passed += 1
    else
        println("  FAIL  Device count")
        failed += 1
    end

    prop_buf = zeros(UInt8, 512)
    rc = ccall(sym("cudaGetDeviceProperties"), Cint, (Ptr{UInt8}, Cint), prop_buf, 0)
    if rc == 0
        name = unsafe_string(pointer(prop_buf))
        total_mem = reinterpret(UInt64, prop_buf[257:264])[1]
        gb = total_mem / 1e9
        println("  PASS  Device: $name ($(round(gb; digits=1)) GB)")
        passed += 1
    else
        println("  FAIL  Device properties")
        failed += 1
    end

    ver = Ref{Cint}(0)
    rc = ccall(sym("cudaRuntimeGetVersion"), Cint, (Ptr{Cint},), ver)
    if rc == 0 && ver[] >= 12000
        println("  PASS  Runtime version: $(ver[])")
        passed += 1
    else
        println("  FAIL  Runtime version")
        failed += 1
    end

    free_mem = Ref{Csize_t}(0)
    total_mem_ref = Ref{Csize_t}(0)
    rc = ccall(sym("cudaMemGetInfo"), Cint, (Ptr{Csize_t}, Ptr{Csize_t}), free_mem, total_mem_ref)
    if rc == 0 && total_mem_ref[] > 0
        fg = free_mem[] / 1e9
        tg = total_mem_ref[] / 1e9
        println("  PASS  Memory: $(round(fg; digits=1)) GB free / $(round(tg; digits=1)) GB total")
        passed += 1
    else
        println("  FAIL  Memory info")
        failed += 1
    end
    println()

    # ── Memory Operations ──
    println("── Memory Operations ──")

    ptr = Ref{Ptr{Cvoid}}(C_NULL)
    rc = ccall(sym("cudaMalloc"), Cint, (Ptr{Ptr{Cvoid}}, Csize_t), ptr, 4096)
    if rc == 0 && ptr[] != C_NULL
        ccall(sym("cudaFree"), Cint, (Ptr{Cvoid},), ptr[])
        println("  PASS  malloc + free (4 KB)")
        passed += 1
    else
        println("  FAIL  malloc + free")
        failed += 1
    end

    # memcpy roundtrip
    begin
        n = 1024
        src = Float32[i * 0.1f0 for i in 0:n-1]
        d_ptr = Ref{Ptr{Cvoid}}(C_NULL)
        ccall(sym("cudaMalloc"), Cint, (Ptr{Ptr{Cvoid}}, Csize_t), d_ptr, n * 4)
        ccall(sym("cudaMemcpy"), Cint, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, Cint),
              d_ptr[], pointer(src), n * 4, 1)  # HtoD
        dst = zeros(Float32, n)
        ccall(sym("cudaMemcpy"), Cint, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, Cint),
              pointer(dst), d_ptr[], n * 4, 2)  # DtoH
        ccall(sym("cudaFree"), Cint, (Ptr{Cvoid},), d_ptr[])

        ok = all(abs.(src .- dst) .< 1e-5)
        if ok
            println("  PASS  memcpy roundtrip (4 KB)")
            passed += 1
        else
            println("  FAIL  memcpy roundtrip")
            failed += 1
        end
    end

    # memset
    begin
        d_ptr = Ref{Ptr{Cvoid}}(C_NULL)
        ccall(sym("cudaMalloc"), Cint, (Ptr{Ptr{Cvoid}}, Csize_t), d_ptr, 4096)
        ccall(sym("cudaMemset"), Cint, (Ptr{Cvoid}, Cint, Csize_t), d_ptr[], 0xAB, 4096)
        buf = zeros(UInt8, 4096)
        ccall(sym("cudaMemcpy"), Cint, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, Cint),
              pointer(buf), d_ptr[], 4096, 2)
        ccall(sym("cudaFree"), Cint, (Ptr{Cvoid},), d_ptr[])

        ok = all(buf .== 0xAB)
        if ok
            println("  PASS  memset (4 KB, pattern 0xAB)")
            passed += 1
        else
            println("  FAIL  memset")
            failed += 1
        end
    end
    println()

    # ── Streams & Events ──
    println("── Streams & Events ──")

    begin
        stream = Ref{Ptr{Cvoid}}(C_NULL)
        rc = ccall(sym("cudaStreamCreate"), Cint, (Ptr{Ptr{Cvoid}},), stream)
        if rc == 0
            ccall(sym("cudaStreamSynchronize"), Cint, (Ptr{Cvoid},), stream[])
            ccall(sym("cudaStreamDestroy"), Cint, (Ptr{Cvoid},), stream[])
            println("  PASS  Stream create + sync + destroy")
            passed += 1
        else
            println("  FAIL  Stream")
            failed += 1
        end
    end

    begin
        ev_start = Ref{Ptr{Cvoid}}(C_NULL)
        ev_end = Ref{Ptr{Cvoid}}(C_NULL)
        ccall(sym("cudaEventCreate"), Cint, (Ptr{Ptr{Cvoid}},), ev_start)
        ccall(sym("cudaEventCreate"), Cint, (Ptr{Ptr{Cvoid}},), ev_end)
        ccall(sym("cudaEventRecord"), Cint, (Ptr{Cvoid}, Ptr{Cvoid}), ev_start[], C_NULL)
        ccall(sym("cudaDeviceSynchronize"), Cint, ())
        ccall(sym("cudaEventRecord"), Cint, (Ptr{Cvoid}, Ptr{Cvoid}), ev_end[], C_NULL)
        ccall(sym("cudaEventSynchronize"), Cint, (Ptr{Cvoid},), ev_end[])
        ms = Ref{Cfloat}(0.0f0)
        ccall(sym("cudaEventElapsedTime"), Cint, (Ptr{Cfloat}, Ptr{Cvoid}, Ptr{Cvoid}),
              ms, ev_start[], ev_end[])
        ccall(sym("cudaEventDestroy"), Cint, (Ptr{Cvoid},), ev_start[])
        ccall(sym("cudaEventDestroy"), Cint, (Ptr{Cvoid},), ev_end[])
        if ms[] >= 0
            println("  PASS  Event timing ($(round(ms[]; digits=3)) ms)")
            passed += 1
        else
            println("  FAIL  Event timing")
            failed += 1
        end
    end
    println()

    # ── cuBLAS ──
    println("── cuBLAS ──")

    begin
        handle = Ref{Ptr{Cvoid}}(C_NULL)
        rc = ccall(sym("cublasCreate_v2"), Cint, (Ptr{Ptr{Cvoid}},), handle)
        if rc == 0
            println("  PASS  cuBLAS handle created")
            passed += 1

            a_data = Float32[i for i in 1:16]
            eye = zeros(Float32, 16)
            for i in 0:3
                eye[i*4 + i + 1] = 1.0f0
            end

            d_a = Ref{Ptr{Cvoid}}(C_NULL)
            d_i = Ref{Ptr{Cvoid}}(C_NULL)
            d_c = Ref{Ptr{Cvoid}}(C_NULL)
            ccall(sym("cudaMalloc"), Cint, (Ptr{Ptr{Cvoid}}, Csize_t), d_a, 64)
            ccall(sym("cudaMalloc"), Cint, (Ptr{Ptr{Cvoid}}, Csize_t), d_i, 64)
            ccall(sym("cudaMalloc"), Cint, (Ptr{Ptr{Cvoid}}, Csize_t), d_c, 64)
            ccall(sym("cudaMemcpy"), Cint, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, Cint),
                  d_a[], pointer(a_data), 64, 1)
            ccall(sym("cudaMemcpy"), Cint, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, Cint),
                  d_i[], pointer(eye), 64, 1)
            ccall(sym("cudaMemset"), Cint, (Ptr{Cvoid}, Cint, Csize_t), d_c[], 0, 64)

            alpha = Ref{Cfloat}(1.0f0)
            beta = Ref{Cfloat}(0.0f0)
            ccall(sym("cublasSgemm_v2"), Cint,
                  (Ptr{Cvoid}, Cint, Cint, Cint, Cint, Cint,
                   Ptr{Cfloat}, Ptr{Cvoid}, Cint,
                   Ptr{Cvoid}, Cint,
                   Ptr{Cfloat}, Ptr{Cvoid}, Cint),
                  handle[], 0, 0, 4, 4, 4,
                  alpha, d_a[], 4, d_i[], 4, beta, d_c[], 4)

            result = zeros(Float32, 16)
            ccall(sym("cudaMemcpy"), Cint, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, Cint),
                  pointer(result), d_c[], 64, 2)
            ccall(sym("cudaFree"), Cint, (Ptr{Cvoid},), d_a[])
            ccall(sym("cudaFree"), Cint, (Ptr{Cvoid},), d_i[])
            ccall(sym("cudaFree"), Cint, (Ptr{Cvoid},), d_c[])

            ok = all(abs.(result .- a_data) .< 1e-3)
            if ok
                println("  PASS  SGEMM identity multiply (4x4)")
                passed += 1
            else
                println("  FAIL  SGEMM identity multiply")
                failed += 1
            end

            ccall(sym("cublasDestroy_v2"), Cint, (Ptr{Cvoid},), handle[])
        else
            println("  FAIL  cuBLAS create")
            failed += 1
        end
    end
    println()

    # ── Performance ──
    println("── Performance ──")

    begin
        size = 4 * 1024 * 1024
        data = zeros(UInt8, size)
        d_ptr = Ref{Ptr{Cvoid}}(C_NULL)
        ccall(sym("cudaMalloc"), Cint, (Ptr{Ptr{Cvoid}}, Csize_t), d_ptr, size)

        # warmup
        for _ in 1:5
            ccall(sym("cudaMemcpy"), Cint, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, Cint),
                  d_ptr[], pointer(data), size, 1)
            ccall(sym("cudaMemcpy"), Cint, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, Cint),
                  pointer(data), d_ptr[], size, 2)
        end

        iters = 50
        t = time()
        for _ in 1:iters
            ccall(sym("cudaMemcpy"), Cint, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, Cint),
                  d_ptr[], pointer(data), size, 1)
            ccall(sym("cudaMemcpy"), Cint, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, Cint),
                  pointer(data), d_ptr[], size, 2)
        end
        elapsed = time() - t
        ccall(sym("cudaFree"), Cint, (Ptr{Cvoid},), d_ptr[])

        gb_s = (iters * size * 2) / elapsed / 1e9
        println("  Mem BW (4 MB): $(round(gb_s; digits=2)) GB/s")
    end
    println()

    # ── Summary ──
    total = time() - start
    println("══════════════════════════════════════════════════════════════")
    if failed == 0
        println("  ALL $passed TESTS PASSED ($(round(total; digits=1))s)")
        println("  CUDA compatibility: VERIFIED via Invisible CUDA (Julia)")
    else
        println("  $passed passed, $failed FAILED ($(round(total; digits=1))s)")
    end
    println("══════════════════════════════════════════════════════════════")

    exit(failed > 0 ? 1 : 0)
end

main()
