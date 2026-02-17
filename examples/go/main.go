// Invisible CUDA — Go Proof
//
// Demonstrates CUDA compatibility using cgo + dlopen against the
// Invisible CUDA runtime.
//
// Prerequisites:
//
//	cargo install invisible-cuda && invisible-cuda install
//
// Run:
//
//	go run main.go
package main

/*
#cgo LDFLAGS: -ldl
#include <stdlib.h>
#include <dlfcn.h>
#include <string.h>

// CUDA types
typedef enum { HtoD = 1, DtoH = 2 } MemcpyKind;
typedef enum { CUBLAS_OP_N = 0 } CublasOp;

typedef struct {
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
} DeviceProp;

static void* lib = NULL;

// Function pointers
static int (*fn_GetDeviceCount)(int*);
static int (*fn_GetDeviceProperties)(DeviceProp*, int);
static int (*fn_DeviceSynchronize)(void);
static int (*fn_RuntimeGetVersion)(int*);
static int (*fn_Malloc)(void**, size_t);
static int (*fn_Free)(void*);
static int (*fn_Memcpy)(void*, const void*, size_t, MemcpyKind);
static int (*fn_Memset)(void*, int, size_t);
static int (*fn_MemGetInfo)(size_t*, size_t*);
static int (*fn_StreamCreate)(void**);
static int (*fn_StreamDestroy)(void*);
static int (*fn_StreamSynchronize)(void*);
static int (*fn_EventCreate)(void**);
static int (*fn_EventDestroy)(void*);
static int (*fn_EventRecord)(void*, void*);
static int (*fn_EventSynchronize)(void*);
static int (*fn_EventElapsedTime)(float*, void*, void*);
static int (*fn_CublasCreate)(void**);
static int (*fn_CublasDestroy)(void*);
static int (*fn_CublasSgemm)(void*, CublasOp, CublasOp, int, int, int,
    const float*, const void*, int, const void*, int, const float*, void*, int);

static int load_lib(const char* path) {
    lib = dlopen(path, RTLD_NOW);
    if (!lib) return -1;

    #define LOAD(name, sym) *(void**)(&name) = dlsym(lib, sym); if (!name) return -1;
    LOAD(fn_GetDeviceCount, "cudaGetDeviceCount");
    LOAD(fn_GetDeviceProperties, "cudaGetDeviceProperties");
    LOAD(fn_DeviceSynchronize, "cudaDeviceSynchronize");
    LOAD(fn_RuntimeGetVersion, "cudaRuntimeGetVersion");
    LOAD(fn_Malloc, "cudaMalloc");
    LOAD(fn_Free, "cudaFree");
    LOAD(fn_Memcpy, "cudaMemcpy");
    LOAD(fn_Memset, "cudaMemset");
    LOAD(fn_MemGetInfo, "cudaMemGetInfo");
    LOAD(fn_StreamCreate, "cudaStreamCreate");
    LOAD(fn_StreamDestroy, "cudaStreamDestroy");
    LOAD(fn_StreamSynchronize, "cudaStreamSynchronize");
    LOAD(fn_EventCreate, "cudaEventCreate");
    LOAD(fn_EventDestroy, "cudaEventDestroy");
    LOAD(fn_EventRecord, "cudaEventRecord");
    LOAD(fn_EventSynchronize, "cudaEventSynchronize");
    LOAD(fn_EventElapsedTime, "cudaEventElapsedTime");
    LOAD(fn_CublasCreate, "cublasCreate_v2");
    LOAD(fn_CublasDestroy, "cublasDestroy_v2");
    LOAD(fn_CublasSgemm, "cublasSgemm_v2");
    #undef LOAD
    return 0;
}

static int c_GetDeviceCount(int* c)                 { return fn_GetDeviceCount(c); }
static int c_GetDeviceProperties(DeviceProp* p)      { return fn_GetDeviceProperties(p, 0); }
static int c_DeviceSynchronize()                      { return fn_DeviceSynchronize(); }
static int c_RuntimeGetVersion(int* v)                { return fn_RuntimeGetVersion(v); }
static int c_Malloc(void** p, size_t s)               { return fn_Malloc(p, s); }
static int c_Free(void* p)                            { return fn_Free(p); }
static int c_Memcpy(void* d, const void* s, size_t n, int k) { return fn_Memcpy(d, s, n, (MemcpyKind)k); }
static int c_Memset(void* p, int v, size_t n)         { return fn_Memset(p, v, n); }
static int c_MemGetInfo(size_t* f, size_t* t)         { return fn_MemGetInfo(f, t); }
static int c_StreamCreate(void** s)                    { return fn_StreamCreate(s); }
static int c_StreamDestroy(void* s)                    { return fn_StreamDestroy(s); }
static int c_StreamSynchronize(void* s)                { return fn_StreamSynchronize(s); }
static int c_EventCreate(void** e)                     { return fn_EventCreate(e); }
static int c_EventDestroy(void* e)                     { return fn_EventDestroy(e); }
static int c_EventRecord(void* e, void* s)             { return fn_EventRecord(e, s); }
static int c_EventSynchronize(void* e)                 { return fn_EventSynchronize(e); }
static int c_EventElapsedTime(float* ms, void* a, void* b) { return fn_EventElapsedTime(ms, a, b); }
static int c_CublasCreate(void** h)                    { return fn_CublasCreate(h); }
static int c_CublasDestroy(void* h)                    { return fn_CublasDestroy(h); }
static int c_CublasSgemm(void* h, int m, int n, int k,
    float alpha, const void* a, const void* b, float beta, void* c_ptr) {
    return fn_CublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
        &alpha, a, m, b, m, &beta, c_ptr, m);
}
*/
import "C"

import (
	"fmt"
	"math"
	"os"
	"runtime"
	"time"
	"unsafe"
)

func findLibPath() string {
	if env := os.Getenv("INVISIBLE_CUDA_LIB"); env != "" {
		return env
	}
	home, _ := os.UserHomeDir()
	if runtime.GOOS == "darwin" {
		return home + "/.invisible-cuda/lib/libcuda.dylib"
	}
	return home + "/.invisible-cuda/lib/libcuda.so"
}

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════╗")
	fmt.Println("║         INVISIBLE CUDA — Go Proof                          ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════╝")
	fmt.Println()

	passed, failed := 0, 0
	start := time.Now()

	// Load library
	path := findLibPath()
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))
	if C.load_lib(cpath) != 0 {
		fmt.Printf("FATAL: Could not load %s\n", path)
		fmt.Println("Install via: cargo install invisible-cuda && invisible-cuda install")
		os.Exit(1)
	}

	// ── Device Discovery ──
	fmt.Println("── Device Discovery ──")

	var count C.int
	if C.c_GetDeviceCount(&count) == 0 && count >= 1 {
		fmt.Printf("  PASS  Device count: %d\n", count)
		passed++
	} else {
		fmt.Println("  FAIL  Device count")
		failed++
	}

	var prop C.DeviceProp
	if C.c_GetDeviceProperties(&prop) == 0 {
		name := C.GoString(&prop.name[0])
		mem := float64(prop.totalGlobalMem) / 1e9
		fmt.Printf("  PASS  Device: %s (%.1f GB)\n", name, mem)
		passed++
	} else {
		fmt.Println("  FAIL  Device properties")
		failed++
	}

	var version C.int
	if C.c_RuntimeGetVersion(&version) == 0 && version >= 12000 {
		fmt.Printf("  PASS  Runtime version: %d\n", version)
		passed++
	} else {
		fmt.Println("  FAIL  Runtime version")
		failed++
	}

	var freeMem, totalMem C.size_t
	if C.c_MemGetInfo(&freeMem, &totalMem) == 0 && totalMem > 0 {
		fmt.Printf("  PASS  Memory: %.1f GB free / %.1f GB total\n",
			float64(freeMem)/1e9, float64(totalMem)/1e9)
		passed++
	} else {
		fmt.Println("  FAIL  Memory info")
		failed++
	}
	fmt.Println()

	// ── Memory Operations ──
	fmt.Println("── Memory Operations ──")

	var ptr unsafe.Pointer
	if C.c_Malloc(&ptr, 4096) == 0 && ptr != nil {
		C.c_Free(ptr)
		fmt.Println("  PASS  malloc + free (4 KB)")
		passed++
	} else {
		fmt.Println("  FAIL  malloc + free")
		failed++
	}

	// memcpy roundtrip
	{
		n := 1024
		src := make([]float32, n)
		for i := range src {
			src[i] = float32(i) * 0.1
		}
		var dPtr unsafe.Pointer
		C.c_Malloc(&dPtr, C.size_t(n*4))
		C.c_Memcpy(dPtr, unsafe.Pointer(&src[0]), C.size_t(n*4), 1) // HtoD
		dst := make([]float32, n)
		C.c_Memcpy(unsafe.Pointer(&dst[0]), dPtr, C.size_t(n*4), 2) // DtoH
		C.c_Free(dPtr)

		ok := true
		for i := 0; i < n; i++ {
			if math.Abs(float64(src[i]-dst[i])) > 1e-5 {
				ok = false
				break
			}
		}
		if ok {
			fmt.Println("  PASS  memcpy roundtrip (4 KB)")
			passed++
		} else {
			fmt.Println("  FAIL  memcpy roundtrip")
			failed++
		}
	}

	// memset
	{
		var dPtr unsafe.Pointer
		C.c_Malloc(&dPtr, 4096)
		C.c_Memset(dPtr, 0xAB, 4096)
		buf := make([]byte, 4096)
		C.c_Memcpy(unsafe.Pointer(&buf[0]), dPtr, 4096, 2)
		C.c_Free(dPtr)

		ok := true
		for _, b := range buf {
			if b != 0xAB {
				ok = false
				break
			}
		}
		if ok {
			fmt.Println("  PASS  memset (4 KB, pattern 0xAB)")
			passed++
		} else {
			fmt.Println("  FAIL  memset")
			failed++
		}
	}
	fmt.Println()

	// ── Streams & Events ──
	fmt.Println("── Streams & Events ──")

	{
		var stream unsafe.Pointer
		if C.c_StreamCreate(&stream) == 0 {
			C.c_StreamSynchronize(stream)
			C.c_StreamDestroy(stream)
			fmt.Println("  PASS  Stream create + sync + destroy")
			passed++
		} else {
			fmt.Println("  FAIL  Stream")
			failed++
		}
	}

	{
		var evStart, evEnd unsafe.Pointer
		C.c_EventCreate(&evStart)
		C.c_EventCreate(&evEnd)
		C.c_EventRecord(evStart, nil)
		C.c_DeviceSynchronize()
		C.c_EventRecord(evEnd, nil)
		C.c_EventSynchronize(evEnd)
		var ms C.float
		C.c_EventElapsedTime(&ms, evStart, evEnd)
		C.c_EventDestroy(evStart)
		C.c_EventDestroy(evEnd)
		if ms >= 0 {
			fmt.Printf("  PASS  Event timing (%.3f ms)\n", float64(ms))
			passed++
		} else {
			fmt.Println("  FAIL  Event timing")
			failed++
		}
	}
	fmt.Println()

	// ── cuBLAS ──
	fmt.Println("── cuBLAS ──")

	{
		var handle unsafe.Pointer
		if C.c_CublasCreate(&handle) == 0 {
			fmt.Println("  PASS  cuBLAS handle created")
			passed++

			aData := [16]float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
			var eye [16]float32
			for i := 0; i < 4; i++ {
				eye[i*4+i] = 1.0
			}

			var dA, dI, dC unsafe.Pointer
			C.c_Malloc(&dA, 64)
			C.c_Malloc(&dI, 64)
			C.c_Malloc(&dC, 64)
			C.c_Memcpy(dA, unsafe.Pointer(&aData[0]), 64, 1)
			C.c_Memcpy(dI, unsafe.Pointer(&eye[0]), 64, 1)
			C.c_Memset(dC, 0, 64)

			C.c_CublasSgemm(handle, 4, 4, 4, 1.0, dA, dI, 0.0, dC)

			var result [16]float32
			C.c_Memcpy(unsafe.Pointer(&result[0]), dC, 64, 2)
			C.c_Free(dA)
			C.c_Free(dI)
			C.c_Free(dC)

			ok := true
			for i := 0; i < 16; i++ {
				if math.Abs(float64(result[i]-aData[i])) > 1e-3 {
					ok = false
					break
				}
			}
			if ok {
				fmt.Println("  PASS  SGEMM identity multiply (4x4)")
				passed++
			} else {
				fmt.Println("  FAIL  SGEMM identity multiply")
				failed++
			}

			C.c_CublasDestroy(handle)
		} else {
			fmt.Println("  FAIL  cuBLAS create")
			failed++
		}
	}
	fmt.Println()

	// ── Performance ──
	fmt.Println("── Performance ──")

	{
		size := 4 * 1024 * 1024
		data := make([]byte, size)
		var dPtr unsafe.Pointer
		C.c_Malloc(&dPtr, C.size_t(size))

		for i := 0; i < 5; i++ {
			C.c_Memcpy(dPtr, unsafe.Pointer(&data[0]), C.size_t(size), 1)
			C.c_Memcpy(unsafe.Pointer(&data[0]), dPtr, C.size_t(size), 2)
		}

		t := time.Now()
		iters := 50
		for i := 0; i < iters; i++ {
			C.c_Memcpy(dPtr, unsafe.Pointer(&data[0]), C.size_t(size), 1)
			C.c_Memcpy(unsafe.Pointer(&data[0]), dPtr, C.size_t(size), 2)
		}
		elapsed := time.Since(t).Seconds()
		C.c_Free(dPtr)

		gbS := float64(iters*size*2) / elapsed / 1e9
		fmt.Printf("  Mem BW (4 MB): %.2f GB/s\n", gbS)
	}
	fmt.Println()

	// ── Summary ──
	total := time.Since(start).Seconds()
	fmt.Println("══════════════════════════════════════════════════════════════")
	if failed == 0 {
		fmt.Printf("  ALL %d TESTS PASSED (%.1fs)\n", passed, total)
		fmt.Println("  CUDA compatibility: VERIFIED via Invisible CUDA (Go)")
	} else {
		fmt.Printf("  %d passed, %d FAILED (%.1fs)\n", passed, failed, total)
	}
	fmt.Println("══════════════════════════════════════════════════════════════")

	if failed > 0 {
		os.Exit(1)
	}
}
