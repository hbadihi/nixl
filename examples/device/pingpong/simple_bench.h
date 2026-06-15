#pragma once

// Minimal, trust-by-inspection ping-pong benchmark.
//
// Unlike bench_kernel.cu, this kernel does the ABSOLUTE MINIMUM inside the timed
// region so there is nothing to second-guess:
//   * issue  = clock64() bracketing the nixlPut / nixlAtomicAdd call only.
//   * RTT    = clock64() across the whole timed loop / num_iters.
//   * one-way = RTT / 2  (computed on the host).
// No local-completion poll, no per-stage poll, no host-mapped reads over PCIe in
// the timed path. Fire-and-forget put + a busy-spin on the reply counter.
//
// Host-safe header (no CUDA device headers); include simple_kernel.cu's twin
// from .cu code only.

#include "bench_kernel_iface.h" // gpu_bench_op, nixlMemViewH, cuda_runtime.h
#include <cstdint>

struct simple_bench_ctx {
    nixlMemViewH local_mvh;   // view of local send_buf
    nixlMemViewH remote_mvh;  // view of peer's recv_buf
    uint8_t     *send_buf;    // device pointer
    uint8_t     *recv_buf;    // device pointer
    size_t       msg_size;    // payload bytes (counter NOT included)
    gpu_bench_op op;
    uint64_t     num_iters;
    uint64_t     warmup_iters;
    bool         is_sender;
    // Sender-only device scalar: SUM of per-iteration issue cycles (the
    // nixlPut/nixlAtomicAdd call). Host divides by num_iters. May be null.
    uint64_t    *issue_ticks;
};

// d_elapsed: device scalar; receives clock64 ticks for the whole timed loop
//            (warmup excluded). Only written when ctx.is_sender.
void launch_simple_thread(simple_bench_ctx ctx, uint64_t *d_elapsed, cudaStream_t stream);
void launch_simple_warp  (simple_bench_ctx ctx, uint64_t *d_elapsed, cudaStream_t stream);

// bench_proxy_publish_context / bench_proxy_clear_context are declared in
// bench_kernel_iface.h (proxy build) and DEFINED here in simple_kernel.cu so the
// reused BenchContext (bench_host.cpp) links against this binary's kernel TU.
