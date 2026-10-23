// Time of one device-proxy-style command publish per warp. Each warp copies <copy_bytes> in GPU
// memory (like the EP token copy before nixlPut), then lane 0 writes a 64 B command to mapped host
// memory and publishes it. All warps start together, like an EP dispatch kernel. Prints the
// per-warp publish time (globaltimer) percentiles over <iters> launches (first 10 skipped).
// usage: fence <warps> <mode> <copy_bytes> [iters]
// mode 0: 4 x 16 B command stores, then st.release.sys        (the current NIXL producer)
//      1: 4 x 16 B command stores, fence.acq_rel.gpu, st.relaxed.sys
//      2: 4 x 16 B command stores, no fence
//      3: st.release.sys only (no command stores)
//      4: fence.acq_rel.sys, then the command in one coalesced 8-lane 64 B store
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
__device__ __forceinline__ uint64_t now() { uint64_t t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__global__ void k(uint64_t *host_ring, int4 *src, int4 *dst, int copy_int4, int mode, uint64_t *out, uint64_t seq) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
    for (int i = lane; i < copy_int4; i += 32) dst[(size_t)warp * copy_int4 + i] = src[(size_t)warp * copy_int4 + i];
    __syncwarp();
    uint64_t *c = host_ring + (size_t)warp * 8;
    if (mode == 4) {
        // fence.sys first (covers the copy), then the whole command in one coalesced 8-lane store
        uint64_t t0 = 0;
        if (lane == 0) { t0 = now(); asm volatile("fence.acq_rel.sys;" ::: "memory"); }
        __syncwarp();
        if (lane < 8) asm volatile("st.global.u64 [%0], %1;" ::"l"(c + lane), "l"(seq) : "memory");
        __syncwarp();
        if (lane == 0) out[warp] = now() - t0;
        return;
    }
    if (lane != 0) return;
    const uint64_t t0 = now();
    if (mode != 3) {
        for (int i = 0; i < 8; i += 2)
            asm volatile("st.global.v2.u64 [%0], {%1, %2};" ::"l"(c + i), "l"(seq), "l"(seq) : "memory");
    }
    if (mode == 0 || mode == 3) asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(c + 7), "l"(seq) : "memory");
    else if (mode == 1) { asm volatile("fence.acq_rel.gpu;" ::: "memory"); asm volatile("st.relaxed.sys.global.u64 [%0], %1;" ::"l"(c + 7), "l"(seq) : "memory"); }
    else asm volatile("st.global.u64 [%0], %1;" ::"l"(c + 7), "l"(seq) : "memory");
    out[warp] = now() - t0;
}
int main(int argc, char **argv) {
    const int warps = atoi(argv[1]), mode = atoi(argv[2]), copy = atoi(argv[3]), iters = argc > 4 ? atoi(argv[4]) : 200;
    uint64_t *h, *d_ring; cudaHostAlloc(&h, (size_t)warps * 64, cudaHostAllocMapped); cudaHostGetDevicePointer(&d_ring, h, 0);
    const int copy_int4 = copy / 16; int4 *src, *dst;
    cudaMalloc(&src, (size_t)warps * copy_int4 * 16 + 16); cudaMalloc(&dst, (size_t)warps * copy_int4 * 16 + 16);
    uint64_t *out; cudaMalloc(&out, warps * 8);
    std::vector<uint64_t> all, tmp(warps);
    const int wpb = std::min(warps, 8), blocks = (warps + wpb - 1) / wpb;
    for (int it = 0; it < iters; ++it) {
        k<<<blocks, wpb * 32>>>(d_ring, src, dst, copy_int4, mode, out, it + 1);
        cudaDeviceSynchronize();
        cudaMemcpy(tmp.data(), out, warps * 8, cudaMemcpyDeviceToHost);
        if (it >= 10) all.insert(all.end(), tmp.begin(), tmp.end());
    }
    std::sort(all.begin(), all.end());
    auto p = [&](double q) { return all[(size_t)(q * (all.size() - 1))] / 1e3; };
    printf("warps=%4d mode=%d copy=%5d  publish us: p50 %7.2f p90 %7.2f p99 %7.2f max %7.2f\n", warps, mode, copy, p(.5), p(.9), p(.99), all.back() / 1e3);
    return cudaGetLastError() != cudaSuccess;
}
