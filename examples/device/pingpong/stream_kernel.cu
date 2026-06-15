// Windowed one-directional streaming bandwidth kernel — see stream_bench.h.
//
// The ENTIRE timed critical path is visible here: fill the window (W posts),
// then for each timed iteration drain the oldest in-flight request and post a
// fresh one, keeping exactly W outstanding. No reply, no ping-pong: the sender
// streams puts; the receiver is a passive RDMA-write target (its host side just
// waits for a "done" notification — no receiver kernel on the data path).

#include "stream_bench.h"
#include "nixl_device.cuh"
#include "nixl_types.h"
#include <cstdint>

#ifdef NIXL_GPU_DEVICE_BACKEND_PROXY
#include "nixl_device_proxy.cuh"
#endif

template<nixl_gpu_level_t level>
__device__ static inline nixl_status_t
post_op(const stream_bench_ctx &ctx, unsigned channel, nixlGpuXferStatusH &st) {
    if (ctx.op == gpu_bench_op::Put) {
        nixlMemViewElem src{ctx.local_mvh, 0, 0};
        nixlMemViewElem dst{ctx.remote_mvh, 0, 0};
        return nixlPut<level>(src, dst, ctx.msg_size, channel, 0, &st);
    }
    // AtomicFlag: bump the remote counter (pure message-rate, no payload).
    nixlMemViewElem counter{ctx.remote_mvh, 0, ctx.msg_size};
    return nixlAtomicAdd<level>(1, counter, channel, 0, &st);
}

template<nixl_gpu_level_t level>
__global__ void
stream_kernel_impl(stream_bench_ctx ctx, uint64_t *elapsed_device) {
    const int      lane_id = threadIdx.x % 32;
    const uint32_t warp_id = blockIdx.x;            // one warp/thread per block
    if (warp_id >= ctx.num_warps) return;

    const unsigned channel =
        ctx.num_channels ? (warp_id % ctx.num_channels) : 0u;
    nixlGpuXferStatusH *reqs = ctx.reqs + static_cast<size_t>(warp_id) * ctx.window;
    const uint64_t W           = ctx.window;
    const uint64_t total_posts = ctx.warmup_iters + ctx.num_iters;

    // Prologue: fill the pipeline with W outstanding puts.
    for (uint64_t j = 0; j < W; j++) {
        nixl_status_t s = post_op<level>(ctx, channel, reqs[j]);
        if (s != NIXL_IN_PROG) {
            if (lane_id == 0) printf("stream post(prologue) failed: %d\n", s);
            return;
        }
    }

    // Steady state: drain oldest, post fresh -> exactly W stay in flight.
    uint64_t start_time = 0;
    for (uint64_t i = 0; i < total_posts; i++) {
        if (lane_id == 0 && i == ctx.warmup_iters) start_time = clock64();

        const uint64_t slot = i % W;
        while (nixlGpuGetXferStatus<level>(reqs[slot]) == NIXL_IN_PROG) {
            // spin until the oldest in this slot reaches a terminal status
        }
        nixl_status_t s = post_op<level>(ctx, channel, reqs[slot]);
        if (s != NIXL_IN_PROG) {
            if (lane_id == 0) printf("stream post(steady) failed: %d\n", s);
            return;
        }
    }

    uint64_t end_time = 0;
    if (lane_id == 0) end_time = clock64();

    // Epilogue: drain the still-in-flight window (untimed) so all bytes land
    // before the host signals "done" to the receiver.
    for (uint64_t j = 0; j < W; j++) {
        while (nixlGpuGetXferStatus<level>(reqs[j]) == NIXL_IN_PROG) {
        }
    }

    if (lane_id == 0) elapsed_device[warp_id] = end_time - start_time;
}

template __global__ void
stream_kernel_impl<nixl_gpu_level_t::THREAD>(stream_bench_ctx, uint64_t *);
template __global__ void
stream_kernel_impl<nixl_gpu_level_t::WARP>(stream_bench_ctx, uint64_t *);

size_t
stream_xfer_status_bytes() {
    return sizeof(nixlGpuXferStatusH);
}

void
launch_stream_thread(stream_bench_ctx ctx, uint64_t *d_elapsed, cudaStream_t stream) {
    stream_kernel_impl<nixl_gpu_level_t::THREAD>
        <<<ctx.num_warps, 1, 0, stream>>>(ctx, d_elapsed);
}

void
launch_stream_warp(stream_bench_ctx ctx, uint64_t *d_elapsed, cudaStream_t stream) {
    stream_kernel_impl<nixl_gpu_level_t::WARP>
        <<<ctx.num_warps, 32, 0, stream>>>(ctx, d_elapsed);
}

#ifdef NIXL_GPU_DEVICE_BACKEND_PROXY
cudaError_t
bench_proxy_publish_context(void *proxy_ctx) {
    return nixlProxyPublishContext(static_cast<nixlProxyDeviceContextData *>(proxy_ctx));
}

cudaError_t
bench_proxy_clear_context() {
    return nixlProxyClearContext();
}
#endif
