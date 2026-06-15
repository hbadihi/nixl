// Minimal ping-pong kernel — see simple_bench.h for the rationale.
//
// The ENTIRE timed critical path is visible here: set counter -> nixlPut ->
// spin on reply. issue brackets only the put call; RTT brackets the whole loop.

#include "simple_bench.h"
#include "nixl_device.cuh"
#include "nixl_types.h"
#include <cstdint>

#ifdef NIXL_GPU_DEVICE_BACKEND_PROXY
#include "nixl_device_proxy.cuh"
#endif

__device__ static inline void
spin_until(volatile uint64_t *counter, uint64_t expected) {
    // Busy-spin on the volatile reply counter (no __nanosleep: it would add
    // wakeup jitter). counter is volatile so the load is re-issued every spin.
    for (;;) {
        if (*counter >= expected) break;
    }
}

template<nixl_gpu_level_t level>
__device__ static inline nixl_status_t
issue_op(const simple_bench_ctx &ctx, size_t total_size, nixlGpuXferStatusH &xfer_status) {
    if (ctx.op == gpu_bench_op::Put) {
        nixlMemViewElem src{ctx.local_mvh, 0, 0};
        nixlMemViewElem dst{ctx.remote_mvh, 0, 0};
        return nixlPut<level>(src, dst, total_size, 0, 0, &xfer_status);
    }
    // AtomicFlag: atomically bump the remote counter (at offset msg_size).
    nixlMemViewElem counter{ctx.remote_mvh, 0, ctx.msg_size};
    return nixlAtomicAdd<level>(1, counter, 0, 0, &xfer_status);
}

template<nixl_gpu_level_t level>
__global__ void
simple_pingpong_kernel(simple_bench_ctx ctx, uint64_t *elapsed_device) {
    constexpr bool is_warp = (level == nixl_gpu_level_t::WARP);
    if constexpr (!is_warp) {
        if (threadIdx.x != 0) return; // single thread does the work
    }
    const int lane_id = threadIdx.x % 32;

    volatile uint64_t *send_counter =
        reinterpret_cast<volatile uint64_t *>(ctx.send_buf + ctx.msg_size);
    volatile uint64_t *recv_counter =
        reinterpret_cast<volatile uint64_t *>(ctx.recv_buf + ctx.msg_size);
    const size_t total_size = ctx.msg_size + sizeof(uint64_t); // payload + counter
    nixlGpuXferStatusH xfer_status;

    const uint64_t total_iters = ctx.num_iters + ctx.warmup_iters;
    uint64_t start_time  = 0;
    uint64_t issue_accum = 0;

    for (uint64_t i = 0; i < total_iters; i++) {
        const bool timed = (i >= ctx.warmup_iters);
        if (ctx.is_sender && lane_id == 0 && i == ctx.warmup_iters) {
            start_time = clock64(); // begin RTT timing after warmup
        }

        if (ctx.is_sender) {
            if (lane_id == 0) {
                *send_counter = i + 1; // value the put carries to the receiver
            }
            if constexpr (is_warp) __syncwarp();

            uint64_t issue_start = 0;
            if (timed && lane_id == 0) issue_start = clock64();
            nixl_status_t st = issue_op<level>(ctx, total_size, xfer_status);
            if constexpr (is_warp) {
                st = static_cast<nixl_status_t>(
                    __shfl_sync(0xffffffff, static_cast<int>(st), 0));
            }
            if (st != NIXL_IN_PROG) {
                if (lane_id == 0) printf("nixl op failed: %d\n", st);
                return;
            }
            if (timed && lane_id == 0) issue_accum += clock64() - issue_start;

            // Wait for the receiver's reply (fire-and-forget: no local poll).
            if (lane_id == 0) spin_until(recv_counter, i + 1);
            if constexpr (is_warp) __syncwarp();
        } else {
            if (lane_id == 0) spin_until(recv_counter, i + 1);
            if constexpr (is_warp) __syncwarp();
            if (lane_id == 0) {
                *send_counter = i + 1;
            }
            issue_op<level>(ctx, total_size, xfer_status); // reply
        }
    }

    if (ctx.is_sender && lane_id == 0) {
        *elapsed_device = clock64() - start_time;
        if (ctx.issue_ticks) *ctx.issue_ticks = issue_accum;
    }
}

template __global__ void
simple_pingpong_kernel<nixl_gpu_level_t::THREAD>(simple_bench_ctx, uint64_t *);
template __global__ void
simple_pingpong_kernel<nixl_gpu_level_t::WARP>(simple_bench_ctx, uint64_t *);

void
launch_simple_thread(simple_bench_ctx ctx, uint64_t *d_elapsed, cudaStream_t stream) {
    simple_pingpong_kernel<nixl_gpu_level_t::THREAD><<<1, 1, 0, stream>>>(ctx, d_elapsed);
}

void
launch_simple_warp(simple_bench_ctx ctx, uint64_t *d_elapsed, cudaStream_t stream) {
    simple_pingpong_kernel<nixl_gpu_level_t::WARP><<<1, 32, 0, stream>>>(ctx, d_elapsed);
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
