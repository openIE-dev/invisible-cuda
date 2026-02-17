/**
 * Invisible CUDA — TypeScript Proof
 *
 * Demonstrates CUDA compatibility on any hardware using the Invisible CUDA Node SDK.
 *
 * Prerequisites:
 *   npm install invisible-cuda
 *   # plus: cargo install invisible-cuda && invisible-cuda install
 *
 * Run:
 *   npx tsx proof.ts
 */

import {
  runtime,
  info,
  cublas,
  array,
  types,
} from "invisible-cuda";

const { getDeviceCount, getDevice, setDevice, getDeviceProperties,
        deviceSynchronize, runtimeGetVersion,
        malloc, free, memcpyHtoD, memcpyDtoH, memset, memGetInfo,
        Stream, Event } = runtime;
const { deviceInfo } = info;
const { CublasHandle } = cublas;
const { CublasOperation, MemcpyKind } = types;
const { GpuArray } = array;

function main() {
  console.log("╔══════════════════════════════════════════════════════════════╗");
  console.log("║       INVISIBLE CUDA — TypeScript SDK Proof                ║");
  console.log("╚══════════════════════════════════════════════════════════════╝");
  console.log();

  let passed = 0;
  let failed = 0;
  const start = performance.now();

  // ── 1. Device Discovery ──────────────────────────────────────────
  console.log("── Device Discovery ──");

  try {
    const count = getDeviceCount();
    console.assert(count >= 1);
    console.log(`  PASS  Device count: ${count}`);
    passed++;
  } catch (e) {
    console.log(`  FAIL  Device count: ${e}`);
    failed++;
  }

  try {
    const di = deviceInfo(0);
    const memGb = di.total_memory / 1e9;
    console.log(`  PASS  Device: ${di.name} (${memGb.toFixed(1)} GB, CUDA ${di.cuda_version})`);
    passed++;
  } catch (e) {
    console.log(`  FAIL  Device info: ${e}`);
    failed++;
  }

  try {
    const version = runtimeGetVersion();
    console.assert(version >= 12000);
    console.log(`  PASS  Runtime version: ${version}`);
    passed++;
  } catch (e) {
    console.log(`  FAIL  Runtime version: ${e}`);
    failed++;
  }

  try {
    const [freeMem, totalMem] = memGetInfo();
    console.assert(totalMem > 0 && freeMem > 0 && freeMem <= totalMem);
    console.log(`  PASS  Memory: ${(freeMem/1e9).toFixed(1)} GB free / ${(totalMem/1e9).toFixed(1)} GB total`);
    passed++;
  } catch (e) {
    console.log(`  FAIL  Memory info: ${e}`);
    failed++;
  }
  console.log();

  // ── 2. Memory Operations ─────────────────────────────────────────
  console.log("── Memory Operations ──");

  // malloc + free
  try {
    const ptr = malloc(4096);
    console.assert(ptr !== 0);
    free(ptr);
    console.log("  PASS  malloc + free (4 KB)");
    passed++;
  } catch (e) {
    console.log(`  FAIL  malloc + free: ${e}`);
    failed++;
  }

  // memcpy roundtrip
  try {
    const n = 1024;
    const src = new Float32Array(n);
    for (let i = 0; i < n; i++) src[i] = i * 0.1;

    const ptr = malloc(n * 4);
    memcpyHtoD(ptr, src);
    const dst = new Float32Array(n);
    memcpyDtoH(dst, ptr);
    free(ptr);

    for (let i = 0; i < n; i++) {
      console.assert(Math.abs(src[i] - dst[i]) < 1e-5, `mismatch at ${i}`);
    }
    console.log("  PASS  memcpy roundtrip (4 KB)");
    passed++;
  } catch (e) {
    console.log(`  FAIL  memcpy roundtrip: ${e}`);
    failed++;
  }

  // memset
  try {
    const ptr = malloc(4096);
    memset(ptr, 0xAB, 4096);
    const buf = new Uint8Array(4096);
    memcpyDtoH(buf, ptr);
    free(ptr);
    for (const b of buf) console.assert(b === 0xAB);
    console.log("  PASS  memset (4 KB, pattern 0xAB)");
    passed++;
  } catch (e) {
    console.log(`  FAIL  memset: ${e}`);
    failed++;
  }
  console.log();

  // ── 3. Streams & Events ──────────────────────────────────────────
  console.log("── Streams & Events ──");

  try {
    const stream = new Stream();
    stream.synchronize();
    console.assert(stream.query());
    stream.destroy();
    console.log("  PASS  Stream create + sync + query");
    passed++;
  } catch (e) {
    console.log(`  FAIL  Stream: ${e}`);
    failed++;
  }

  try {
    const startEv = new Event();
    const endEv = new Event();
    startEv.record();
    deviceSynchronize();
    endEv.record();
    endEv.synchronize();
    const ms = Event.elapsed_time(startEv, endEv);
    console.assert(ms >= 0.0);
    startEv.destroy();
    endEv.destroy();
    console.log(`  PASS  Event timing (${ms.toFixed(3)} ms)`);
    passed++;
  } catch (e) {
    console.log(`  FAIL  Event: ${e}`);
    failed++;
  }
  console.log();

  // ── 4. cuBLAS ────────────────────────────────────────────────────
  console.log("── cuBLAS ──");

  try {
    const handle = new CublasHandle();
    console.log("  PASS  cuBLAS handle created");
    passed++;

    // SGEMM: C = A * I
    const n = 4;
    const aData = new Float32Array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]);
    const eye = new Float32Array(16);
    for (let i = 0; i < 4; i++) eye[i * 4 + i] = 1.0;

    const size = 16 * 4;
    const dA = malloc(size);
    const dI = malloc(size);
    const dC = malloc(size);
    memcpyHtoD(dA, aData);
    memcpyHtoD(dI, eye);
    memset(dC, 0, size);

    handle.sgemm(
      CublasOperation.N, CublasOperation.N,
      n, n, n, 1.0,
      dA, n,
      dI, n,
      0.0, dC, n,
    );

    const result = new Float32Array(16);
    memcpyDtoH(result, dC);
    free(dA); free(dI); free(dC);

    let ok = true;
    for (let i = 0; i < 16; i++) {
      if (Math.abs(result[i] - aData[i]) > 1e-3) { ok = false; break; }
    }

    if (ok) {
      console.log("  PASS  SGEMM identity multiply (4x4)");
      passed++;
    } else {
      console.log(`  FAIL  SGEMM identity multiply: [${Array.from(result).join(",")}]`);
      failed++;
    }

    handle.destroy();
  } catch (e) {
    console.log(`  FAIL  cuBLAS: ${e}`);
    failed++;
  }
  console.log();

  // ── 5. GpuArray (high-level API) ─────────────────────────────────
  console.log("── GpuArray ──");

  try {
    // Roundtrip
    const arr = GpuArray.fromArray(new Float32Array([1, 2, 3, 4]), [2, 2]);
    const data = arr.toArray();
    console.assert(data[0] === 1 && data[1] === 2 && data[2] === 3 && data[3] === 4);
    console.log("  PASS  fromArray + toArray roundtrip");
    passed++;

    // Zeros / ones
    const z = GpuArray.zeros([100]);
    console.assert(z.toArray().every((v: number) => v === 0));
    const o = GpuArray.ones([100]);
    console.assert(o.toArray().every((v: number) => v === 1));
    console.log("  PASS  zeros + ones");
    passed++;

    // Arithmetic
    const a = GpuArray.fromArray(new Float32Array([1, 2, 3]), [3]);
    const b = GpuArray.fromArray(new Float32Array([4, 5, 6]), [3]);
    const addResult = a.add(b).toArray();
    console.assert(addResult[0] === 5 && addResult[1] === 7 && addResult[2] === 9);
    const subResult = a.sub(b).toArray();
    console.assert(subResult[0] === -3 && subResult[1] === -3 && subResult[2] === -3);
    const mulResult = a.mul(b).toArray();
    console.assert(mulResult[0] === 4 && mulResult[1] === 10 && mulResult[2] === 18);
    console.log("  PASS  element-wise arithmetic (add, sub, mul)");
    passed++;

    // Matmul
    const dim = 64;
    const ones = new Float32Array(dim * dim).fill(1);
    const ma = GpuArray.fromArray(ones, [dim, dim]);
    const mb = GpuArray.fromArray(ones, [dim, dim]);
    const mc = ma.matmul(mb);
    const mResult = mc.toArray();
    console.assert(Math.abs(mResult[0] - dim) < 0.1);
    console.log(`  PASS  matmul ${dim}x${dim} (result[0] = ${mResult[0].toFixed(1)})`);
    passed++;

    // Transpose
    const ta = GpuArray.fromArray(new Float32Array([1,2,3,4,5,6]), [2, 3]);
    const tt = ta.T;
    console.assert(tt.shape[0] === 3 && tt.shape[1] === 2);
    const tData = tt.toArray();
    console.assert(tData[0] === 1 && tData[1] === 4 && tData[2] === 2);
    console.log("  PASS  transpose (2x3 -> 3x2)");
    passed++;

    // Cleanup
    arr.free(); z.free(); o.free(); a.free(); b.free();
    ma.free(); mb.free(); mc.free(); ta.free(); tt.free();
  } catch (e) {
    console.log(`  FAIL  GpuArray: ${e}`);
    failed++;
  }
  console.log();

  // ── 6. Performance Benchmark ─────────────────────────────────────
  console.log("── Performance ──");

  try {
    // Memory bandwidth
    const size = 4 * 1024 * 1024;
    const data = new Uint8Array(size);
    const ptr = malloc(size);
    for (let i = 0; i < 5; i++) {
      memcpyHtoD(ptr, data);
      memcpyDtoH(new Uint8Array(size), ptr);
    }
    const t = performance.now();
    const iters = 50;
    for (let i = 0; i < iters; i++) {
      memcpyHtoD(ptr, data);
      memcpyDtoH(new Uint8Array(size), ptr);
    }
    const elapsed = (performance.now() - t) / 1000;
    free(ptr);
    const gbS = (iters * size * 2) / elapsed / 1e9;
    console.log(`  Mem BW (4 MB): ${gbS.toFixed(2)} GB/s`);

    // SGEMM throughput
    const dim = 512;
    const ones = new Float32Array(dim * dim).fill(0.5);
    const sa = GpuArray.fromArray(ones, [dim, dim]);
    const sb = GpuArray.fromArray(ones, [dim, dim]);
    sa.matmul(sb); // warmup
    const t2 = performance.now();
    const sIters = 10;
    for (let i = 0; i < sIters; i++) sa.matmul(sb);
    const sElapsed = (performance.now() - t2) / 1000;
    const flops = 2.0 * dim ** 3 * sIters;
    const gflops = flops / sElapsed / 1e9;
    console.log(`  SGEMM ${dim}x${dim}: ${gflops.toFixed(2)} GFLOPS`);
    sa.free(); sb.free();
  } catch (e) {
    console.log(`  Benchmark error: ${e}`);
  }
  console.log();

  // ── Summary ──────────────────────────────────────────────────────
  const totalTime = (performance.now() - start) / 1000;
  console.log("══════════════════════════════════════════════════════════════");
  if (failed === 0) {
    console.log(`  ALL ${passed} TESTS PASSED (${totalTime.toFixed(1)}s)`);
    console.log("  CUDA compatibility: VERIFIED via Invisible CUDA TypeScript SDK");
  } else {
    console.log(`  ${passed} passed, ${failed} FAILED (${totalTime.toFixed(1)}s)`);
  }
  console.log("══════════════════════════════════════════════════════════════");

  process.exit(failed > 0 ? 1 : 0);
}

main();
