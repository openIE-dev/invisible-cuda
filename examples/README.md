# Invisible CUDA — SDK Proof Examples

Run CUDA workloads on any hardware. These examples verify CUDA compatibility using the Invisible CUDA runtime in **10 languages**.

## Prerequisites

Install the Invisible CUDA runtime:

```bash
# Via cargo (recommended)
cargo install invisible-cuda
invisible-cuda install

# Or download a release tarball from:
# https://github.com/openIE-dev/invisible-cuda/releases
```

## Languages

### Rust (SDK)

```bash
cd examples/rust
cargo run --release
```

Requires: `cargo` (Rust toolchain). Uses the `invisible-cuda-sdk` crate.

### Python (SDK)

```bash
cd examples/python
pip install -r requirements.txt
python proof.py
```

Requires: Python 3.8+, numpy. Uses the `invisible-cuda` PyPI package.

### TypeScript (SDK)

```bash
cd examples/typescript
npm install
npx tsx proof.ts
```

Requires: Node.js 18+. Uses the `invisible-cuda` npm package.

### C

```bash
cd examples/c
make
./proof
```

Requires: GCC or Clang. Uses `dlopen` to load the runtime directly.

### Go

```bash
cd examples/go
go run main.go
```

Requires: Go 1.22+. Uses cgo with `dlopen`.

### Java

```bash
cd examples/java
java --enable-native-access=ALL-UNNAMED Proof.java
```

Requires: Java 22+ (Panama Foreign Function & Memory API). No external dependencies.

### Swift

```bash
cd examples/swift
swift run -c release
```

Requires: Swift 5.9+. Uses `dlopen` with `@convention(c)` typealias.

### C# (.NET)

```bash
cd examples/csharp
dotnet run -c Release
```

Requires: .NET 8.0+. Uses `NativeLibrary` + P/Invoke delegates.

### Zig

```bash
cd examples/zig
zig build run -Doptimize=ReleaseFast
```

Requires: Zig 0.13+. Uses `@cImport` with `dlopen`.

### Julia

```bash
cd examples/julia
julia proof.jl
```

Requires: Julia 1.9+. Uses `Libdl` + `ccall`.

## What the proofs test

Each proof verifies the same capabilities across all languages:

| Test | Description |
|------|-------------|
| Device Discovery | Device count, properties, memory info, runtime version |
| Memory Operations | malloc/free, memcpy roundtrip (H2D + D2H), memset |
| Streams & Events | Stream lifecycle, event timing |
| cuBLAS | SGEMM identity multiply (A * I = A) |
| Performance | Memory bandwidth (4 MB roundtrip) |

The Rust, Python, and TypeScript SDK examples also test:

| Test | Description |
|------|-------------|
| GpuArray | High-level API: roundtrip, zeros/ones, arithmetic, matmul, transpose |
| PTX Kernel | Driver API kernel launch (Rust only) |
| SGEMM Throughput | 512x512 matrix multiply benchmark |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `INVISIBLE_CUDA_LIB` | `~/.invisible-cuda/lib/libcuda.{dylib,so}` | Override path to the runtime library |
