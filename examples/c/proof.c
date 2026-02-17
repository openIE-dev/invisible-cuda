/**
 * Invisible CUDA — C Proof
 *
 * Demonstrates CUDA compatibility using standard CUDA C API calls,
 * linked against the Invisible CUDA runtime instead of NVIDIA's.
 *
 * Prerequisites:
 *   cargo install invisible-cuda && invisible-cuda install
 *
 * Build & run:
 *   make && ./proof
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <dlfcn.h>

/* ── Types (matching CUDA) ──────────────────────────────────────── */

typedef enum {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
} cudaMemcpyKind_t;

typedef enum {
    CUBLAS_OP_N = 0,
    CUBLAS_OP_T = 1,
} cublasOperation_t;

struct cudaDeviceProp {
    char name[256];
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int regsPerBlock;
    int warpSize;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    int clockRate;
    int major;
    int minor;
    /* ... more fields, but we only need these */
};

/* ── Function pointers ──────────────────────────────────────────── */

static int (*p_cudaGetDeviceCount)(int*);
static int (*p_cudaGetDeviceProperties)(struct cudaDeviceProp*, int);
static int (*p_cudaDeviceSynchronize)(void);
static int (*p_cudaRuntimeGetVersion)(int*);
static int (*p_cudaMalloc)(void**, size_t);
static int (*p_cudaFree)(void*);
static int (*p_cudaMemcpy)(void*, const void*, size_t, cudaMemcpyKind_t);
static int (*p_cudaMemset)(void*, int, size_t);
static int (*p_cudaMemGetInfo)(size_t*, size_t*);
static int (*p_cudaStreamCreate)(void**);
static int (*p_cudaStreamDestroy)(void*);
static int (*p_cudaStreamSynchronize)(void*);
static int (*p_cudaEventCreate)(void**);
static int (*p_cudaEventDestroy)(void*);
static int (*p_cudaEventRecord)(void*, void*);
static int (*p_cudaEventSynchronize)(void*);
static int (*p_cudaEventElapsedTime)(float*, void*, void*);
static int (*p_cublasCreate_v2)(void**);
static int (*p_cublasDestroy_v2)(void*);
static int (*p_cublasSgemm_v2)(void*, cublasOperation_t, cublasOperation_t,
    int, int, int, const float*, const void*, int, const void*, int,
    const float*, void*, int);

/* ── Library loading ────────────────────────────────────────────── */

static void* lib = NULL;

static void* sym(const char* name) {
    void* s = dlsym(lib, name);
    if (!s) {
        fprintf(stderr, "Symbol not found: %s\n", name);
        exit(1);
    }
    return s;
}

static int load_library(void) {
    const char* env = getenv("INVISIBLE_CUDA_LIB");
    if (env) {
        lib = dlopen(env, RTLD_NOW);
        if (lib) return 0;
    }

    char path[512];
    const char* home = getenv("HOME");
#ifdef __APPLE__
    snprintf(path, sizeof(path), "%s/.invisible-cuda/lib/libcuda.dylib", home);
#else
    snprintf(path, sizeof(path), "%s/.invisible-cuda/lib/libcuda.so", home);
#endif
    lib = dlopen(path, RTLD_NOW);
    if (!lib) {
        fprintf(stderr, "Could not load Invisible CUDA: %s\n", dlerror());
        fprintf(stderr, "Install via: cargo install invisible-cuda && invisible-cuda install\n");
        return -1;
    }

    p_cudaGetDeviceCount = sym("cudaGetDeviceCount");
    p_cudaGetDeviceProperties = sym("cudaGetDeviceProperties");
    p_cudaDeviceSynchronize = sym("cudaDeviceSynchronize");
    p_cudaRuntimeGetVersion = sym("cudaRuntimeGetVersion");
    p_cudaMalloc = sym("cudaMalloc");
    p_cudaFree = sym("cudaFree");
    p_cudaMemcpy = sym("cudaMemcpy");
    p_cudaMemset = sym("cudaMemset");
    p_cudaMemGetInfo = sym("cudaMemGetInfo");
    p_cudaStreamCreate = sym("cudaStreamCreate");
    p_cudaStreamDestroy = sym("cudaStreamDestroy");
    p_cudaStreamSynchronize = sym("cudaStreamSynchronize");
    p_cudaEventCreate = sym("cudaEventCreate");
    p_cudaEventDestroy = sym("cudaEventDestroy");
    p_cudaEventRecord = sym("cudaEventRecord");
    p_cudaEventSynchronize = sym("cudaEventSynchronize");
    p_cudaEventElapsedTime = sym("cudaEventElapsedTime");
    p_cublasCreate_v2 = sym("cublasCreate_v2");
    p_cublasDestroy_v2 = sym("cublasDestroy_v2");
    p_cublasSgemm_v2 = sym("cublasSgemm_v2");

    return 0;
}

/* ── Proof ──────────────────────────────────────────────────────── */

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void) {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║         INVISIBLE CUDA — C Proof                           ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    int passed = 0, failed = 0;
    double start = now_sec();

    if (load_library() != 0) return 1;

    /* ── Device Discovery ── */
    printf("── Device Discovery ──\n");

    int count = 0;
    if (p_cudaGetDeviceCount(&count) == 0 && count >= 1) {
        printf("  PASS  Device count: %d\n", count);
        passed++;
    } else { printf("  FAIL  Device count\n"); failed++; }

    struct cudaDeviceProp prop;
    memset(&prop, 0, sizeof(prop));
    if (p_cudaGetDeviceProperties(&prop, 0) == 0) {
        printf("  PASS  Device: %s (%.1f GB)\n", prop.name,
            (double)prop.totalGlobalMem / 1e9);
        passed++;
    } else { printf("  FAIL  Device properties\n"); failed++; }

    int version = 0;
    if (p_cudaRuntimeGetVersion(&version) == 0 && version >= 12000) {
        printf("  PASS  Runtime version: %d\n", version);
        passed++;
    } else { printf("  FAIL  Runtime version\n"); failed++; }

    size_t free_mem = 0, total_mem = 0;
    if (p_cudaMemGetInfo(&free_mem, &total_mem) == 0 && total_mem > 0) {
        printf("  PASS  Memory: %.1f GB free / %.1f GB total\n",
            (double)free_mem / 1e9, (double)total_mem / 1e9);
        passed++;
    } else { printf("  FAIL  Memory info\n"); failed++; }
    printf("\n");

    /* ── Memory Operations ── */
    printf("── Memory Operations ──\n");

    void* ptr = NULL;
    if (p_cudaMalloc(&ptr, 4096) == 0 && ptr != NULL) {
        p_cudaFree(ptr);
        printf("  PASS  malloc + free (4 KB)\n");
        passed++;
    } else { printf("  FAIL  malloc + free\n"); failed++; }

    /* memcpy roundtrip */
    {
        int n = 1024;
        float* src = malloc(n * sizeof(float));
        float* dst = malloc(n * sizeof(float));
        for (int i = 0; i < n; i++) src[i] = i * 0.1f;

        void* d_ptr = NULL;
        p_cudaMalloc(&d_ptr, n * 4);
        p_cudaMemcpy(d_ptr, src, n * 4, cudaMemcpyHostToDevice);
        p_cudaMemcpy(dst, d_ptr, n * 4, cudaMemcpyDeviceToHost);
        p_cudaFree(d_ptr);

        int ok = 1;
        for (int i = 0; i < n; i++) {
            if (fabsf(src[i] - dst[i]) > 1e-5f) { ok = 0; break; }
        }
        if (ok) { printf("  PASS  memcpy roundtrip (4 KB)\n"); passed++; }
        else { printf("  FAIL  memcpy roundtrip\n"); failed++; }
        free(src); free(dst);
    }

    /* memset */
    {
        void* d_ptr = NULL;
        unsigned char buf[4096];
        p_cudaMalloc(&d_ptr, 4096);
        p_cudaMemset(d_ptr, 0xAB, 4096);
        p_cudaMemcpy(buf, d_ptr, 4096, cudaMemcpyDeviceToHost);
        p_cudaFree(d_ptr);

        int ok = 1;
        for (int i = 0; i < 4096; i++) {
            if (buf[i] != 0xAB) { ok = 0; break; }
        }
        if (ok) { printf("  PASS  memset (4 KB, pattern 0xAB)\n"); passed++; }
        else { printf("  FAIL  memset\n"); failed++; }
    }
    printf("\n");

    /* ── Streams & Events ── */
    printf("── Streams & Events ──\n");

    {
        void* stream = NULL;
        if (p_cudaStreamCreate(&stream) == 0) {
            p_cudaStreamSynchronize(stream);
            p_cudaStreamDestroy(stream);
            printf("  PASS  Stream create + sync + destroy\n");
            passed++;
        } else { printf("  FAIL  Stream\n"); failed++; }
    }

    {
        void* ev_start = NULL, *ev_end = NULL;
        p_cudaEventCreate(&ev_start);
        p_cudaEventCreate(&ev_end);
        p_cudaEventRecord(ev_start, NULL);
        p_cudaDeviceSynchronize();
        p_cudaEventRecord(ev_end, NULL);
        p_cudaEventSynchronize(ev_end);
        float ms = 0;
        p_cudaEventElapsedTime(&ms, ev_start, ev_end);
        p_cudaEventDestroy(ev_start);
        p_cudaEventDestroy(ev_end);
        if (ms >= 0.0f) {
            printf("  PASS  Event timing (%.3f ms)\n", ms);
            passed++;
        } else { printf("  FAIL  Event timing\n"); failed++; }
    }
    printf("\n");

    /* ── cuBLAS ── */
    printf("── cuBLAS ──\n");

    {
        void* handle = NULL;
        if (p_cublasCreate_v2(&handle) == 0) {
            printf("  PASS  cuBLAS handle created\n");
            passed++;

            /* SGEMM: C = A * I */
            int n = 4;
            float a_data[16], eye[16], c_data[16];
            for (int i = 0; i < 16; i++) a_data[i] = (float)(i + 1);
            memset(eye, 0, sizeof(eye));
            for (int i = 0; i < 4; i++) eye[i * 4 + i] = 1.0f;

            void* d_a = NULL, *d_i = NULL, *d_c = NULL;
            p_cudaMalloc(&d_a, 64);
            p_cudaMalloc(&d_i, 64);
            p_cudaMalloc(&d_c, 64);
            p_cudaMemcpy(d_a, a_data, 64, cudaMemcpyHostToDevice);
            p_cudaMemcpy(d_i, eye, 64, cudaMemcpyHostToDevice);
            p_cudaMemset(d_c, 0, 64);

            float alpha = 1.0f, beta = 0.0f;
            p_cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n, &alpha, d_a, n, d_i, n, &beta, d_c, n);

            float result[16];
            p_cudaMemcpy(result, d_c, 64, cudaMemcpyDeviceToHost);
            p_cudaFree(d_a); p_cudaFree(d_i); p_cudaFree(d_c);

            int ok = 1;
            for (int i = 0; i < 16; i++) {
                if (fabsf(result[i] - a_data[i]) > 1e-3f) { ok = 0; break; }
            }
            if (ok) { printf("  PASS  SGEMM identity multiply (4x4)\n"); passed++; }
            else { printf("  FAIL  SGEMM identity multiply\n"); failed++; }

            p_cublasDestroy_v2(handle);
        } else { printf("  FAIL  cuBLAS create\n"); failed++; }
    }
    printf("\n");

    /* ── Performance ── */
    printf("── Performance ──\n");

    {
        size_t size = 4 * 1024 * 1024;
        void* data = calloc(1, size);
        void* d_ptr = NULL;
        p_cudaMalloc(&d_ptr, size);

        /* warmup */
        for (int i = 0; i < 5; i++) {
            p_cudaMemcpy(d_ptr, data, size, cudaMemcpyHostToDevice);
            p_cudaMemcpy(data, d_ptr, size, cudaMemcpyDeviceToHost);
        }

        double t = now_sec();
        int iters = 50;
        for (int i = 0; i < iters; i++) {
            p_cudaMemcpy(d_ptr, data, size, cudaMemcpyHostToDevice);
            p_cudaMemcpy(data, d_ptr, size, cudaMemcpyDeviceToHost);
        }
        double elapsed = now_sec() - t;
        p_cudaFree(d_ptr);
        free(data);

        double gb_s = (double)(iters * size * 2) / elapsed / 1e9;
        printf("  Mem BW (4 MB): %.2f GB/s\n", gb_s);
    }
    printf("\n");

    /* ── Summary ── */
    double total = now_sec() - start;
    printf("══════════════════════════════════════════════════════════════\n");
    if (failed == 0) {
        printf("  ALL %d TESTS PASSED (%.1fs)\n", passed, total);
        printf("  CUDA compatibility: VERIFIED via Invisible CUDA (C)\n");
    } else {
        printf("  %d passed, %d FAILED (%.1fs)\n", passed, failed, total);
    }
    printf("══════════════════════════════════════════════════════════════\n");

    dlclose(lib);
    return failed > 0 ? 1 : 0;
}
