#include "bench_kernel.cuh"
#include "nixl_device.cuh"
#include "nixl_types.h"
#include <cstdint>

#ifdef NIXL_GPU_DEVICE_BACKEND_PROXY
// Pulled in early (also re-included at the bottom for the publish/clear helpers)
// so the kernel can poll the worker's per-stage acknowledgements for the
// call->doorbell diagnostic. Header has its own include guard.
#include "nixl_device_proxy.cuh"
#endif

__device__ static void
wait_sequence_number(volatile uint64_t *counter, uint64_t expected_value) {
    // Busy-spin directly on the volatile counter (no __nanosleep backoff).
    //
    // This wait sits on the latency critical path: the sender stamps rtt_end the
    // instant it returns, and the receiver issues its reply put the instant it
    // returns. A __nanosleep() hint here adds quantization + wakeup jitter (its
    // real SM wakeup latency is not the requested ns and varies per call), which
    // was the dominant noise source in the `peer-wait` phase. With a single
    // in-flight op per side, spinning on a volatile load is the standard
    // low-latency wait (cf. NVSHMEM wait_until / UCX device poll).
    //
    // The loop is NOT optimized away: `counter` is volatile, so each iteration
    // re-issues a real ld.volatile (the compiler may not hoist or elide it). The
    // explicit per-iteration load below makes that guarantee obvious.
    for (;;) {
        const uint64_t observed = *counter; // ld.volatile, re-read every spin
        if (observed >= expected_value) {
            break;
        }
    }
}

__forceinline__ __device__ static void
record_cycle_sample(gpu_cycle_stats *s, uint64_t c) {
    if (!s) return;
    if (s->count == 0)
        s->min = s->max = c;
    else {
        s->min = min(s->min, c);
        s->max = max(s->max, c);
    }
    const uint64_t idx = s->count;
    ++s->count;
    s->sum += c;
    s->sum_sq += double(c) * double(c);
    // Append the raw sample for host-side percentile / histogram computation
    // (matches src/core/device_proxy/proxy_worker.cpp TimingStats::record).
    if (s->samples != nullptr && idx < s->capacity) {
        s->samples[idx] = c;
    }
}

template<nixl_gpu_level_t level>
__device__ static nixl_status_t
do_put_async(nixlMemViewH local_mvh,
             nixlMemViewH remote_mvh,
             size_t total_size,
             nixlGpuXferStatusH &xfer_status) {
    nixlMemViewElem src{local_mvh, 0, 0};
    nixlMemViewElem dst{remote_mvh, 0, 0};
    nixl_status_t status = nixlPut<level>(src, dst, total_size, 0, 0, &xfer_status);
    if (status != NIXL_IN_PROG && threadIdx.x == 0) {
        printf("nixlPut failed with status %d\n", status);
    }
    return status;
}

template<nixl_gpu_level_t level>
__device__ static nixl_status_t
do_atomic_flag_async(nixlMemViewH remote_mvh,
                     size_t counter_offset,
                     nixlGpuXferStatusH &xfer_status) {
    nixlMemViewElem counter{remote_mvh, 0, counter_offset};
    nixl_status_t status = nixlAtomicAdd<level>(1, counter, 0, 0, &xfer_status);
    if (status != NIXL_IN_PROG && threadIdx.x == 0) {
        printf("nixlAtomicAdd failed with status %d\n", status);
    }
    return status;
}

template<nixl_gpu_level_t level>
__device__ static nixl_status_t
do_ping_async(const gpu_bench_ctx &ctx, size_t total_size, nixlGpuXferStatusH &xfer_status) {
    switch (ctx.op) {
    case gpu_bench_op::Put:
        return do_put_async<level>(ctx.local_mvh, ctx.remote_mvh, total_size, xfer_status);
    case gpu_bench_op::AtomicFlag:
        return do_atomic_flag_async<level>(ctx.remote_mvh, ctx.msg_size, xfer_status);
    }

    if (threadIdx.x == 0) {
        printf("unknown pingpong op %u\n", static_cast<unsigned>(ctx.op));
    }
    return NIXL_ERR_INVALID_PARAM;
}

template<nixl_gpu_level_t level>
__device__ static nixl_status_t
do_put_sync(nixlMemViewH local_mvh,
            nixlMemViewH remote_mvh,
            size_t total_size,
            nixlGpuXferStatusH &xfer_status) {
    nixlMemViewElem src{local_mvh, 0, 0};
    nixlMemViewElem dst{remote_mvh, 0, 0};
    nixl_status_t status;
    // Initiate the transfer
    do_put_async<level>(local_mvh, remote_mvh, total_size, xfer_status);
    // Wait for the transfer to complete
    do {
        status = nixlGpuGetXferStatus<level>(xfer_status);
    } while (status == NIXL_IN_PROG);
    return status;
}

template<nixl_gpu_level_t level>
__global__ void
nixl_pingpong_latency_kernel(gpu_bench_ctx ctx, uint64_t *elapsed_device) {
    constexpr bool is_warp = (level == nixl_gpu_level_t::WARP);

    if constexpr (!is_warp) {
        if (threadIdx.x != 0) return; // Only one thread does the work for non-warp level
    }

    const int lane_id = threadIdx.x % 32; // Get the lane ID for warp-level operations

    // Setup counter pointers
    volatile uint64_t *send_counter =
        reinterpret_cast<volatile uint64_t *>(ctx.send_buf + ctx.msg_size);
    volatile uint64_t *recv_counter =
        reinterpret_cast<volatile uint64_t *>(ctx.recv_buf + ctx.msg_size);

    const size_t total_size = ctx.msg_size + sizeof(uint64_t); // Message size + counter
    nixlGpuXferStatusH xfer_status;
    // Single sender-side measurement gate. The proxy worker's internal stage
    // acknowledgements (dequeued/prepared/submitted) deliberately are not
    // polled here: they live in host-mapped memory, so reading them from the
    // GPU would inject a PCIe round-trip into the timed loop and perturb the
    // very thing we're measuring. Those phases are still observable in the
    // worker's own [proxy-worker-stats] block.
    const bool measure_completion =
        ctx.is_sender && ctx.issue_stats != nullptr &&
        ctx.completion_stats != nullptr && ctx.peer_wait_stats != nullptr;
    const bool measure_rtt = ctx.is_sender && ctx.rtt_stats != nullptr;
    // Proxy-only call->doorbell diagnostic. Independent of measure_completion.
    const bool measure_stages = ctx.is_sender && ctx.stage_submitted_stats != nullptr;

    // warmup
    const uint64_t total_iters = ctx.num_iters + ctx.warmup_iters;
    uint64_t start_time = 0;

    for (uint64_t i = 0; i < total_iters; i++) {
        if (ctx.is_sender && lane_id == 0 && i == ctx.warmup_iters) {
            start_time = clock64(); // Start timing after warmup
        }

        // ping pong body
        if (ctx.is_sender) {
            const bool timed_iter = i >= ctx.warmup_iters;
            uint64_t rtt_start = 0;
            if (timed_iter && measure_rtt && lane_id == 0) {
                rtt_start = clock64();
            }
            if (lane_id == 0) {
                *send_counter = i + 1; // Increment send counter to signal the receiver
            }
            if constexpr (is_warp) {
                __syncwarp(); // Ensure all threads see the updated counter
            }
            uint64_t issue_start = 0;
            if (timed_iter && measure_completion && lane_id == 0) {
                issue_start = clock64();
            }
            uint64_t stage_start = 0;
            if (timed_iter && measure_stages && lane_id == 0) {
                stage_start = clock64(); // "call" timestamp for the call->doorbell diagnostic
            }
            nixl_status_t put_status = do_ping_async<level>(ctx, total_size, xfer_status);
            if constexpr (is_warp) {
                put_status = static_cast<nixl_status_t>(
                    __shfl_sync(0xffffffff, static_cast<int>(put_status), 0));
            }
            if (put_status != NIXL_IN_PROG) {
                return;
            }
            uint64_t issue_end = 0;
            if (timed_iter && measure_completion && lane_id == 0) {
                issue_end = clock64();
            }
#ifdef NIXL_GPU_DEVICE_BACKEND_PROXY
            // Call->doorbell diagnostic: spin (collectively) on the worker's stage
            // acks. dequeued = record picked up; submitted = backend submit returned
            // (NIC doorbell rung). These host-mapped reads add a PCIe hop per poll,
            // so treat the numbers as an upper bound on the true handoff/doorbell.
            if (timed_iter && measure_stages) {
                nixl_status_t sst;
                while ((sst = nixlProxyPollDequeued<level>(xfer_status)) == NIXL_IN_PROG) { }
                uint64_t deq_t = 0;
                if (lane_id == 0) deq_t = clock64();
                while ((sst = nixlProxyPollSubmitted<level>(xfer_status)) == NIXL_IN_PROG) { }
                if (lane_id == 0) {
                    const uint64_t sub_t = clock64();
                    record_cycle_sample(ctx.stage_dequeued_stats,  deq_t - stage_start);
                    record_cycle_sample(ctx.stage_submitted_stats, sub_t - stage_start);
                }
                (void)sst;
            }
#endif

            // Wait for terminal completion of this op as observed by the CPU
            // worker (proxy build) or UCX device API (non-proxy build). For
            // pingpong this is unambiguous because only one request is in
            // flight per side. The two buckets split the issue->rtt interval
            // into the worker's per-request completion latency (`complete`,
            // measured from issue_end) and the pure peer/network turnaround
            // (`peer-wait`).
            uint64_t completion_end = 0;
            if (measure_completion) {
                nixl_status_t cpl_status;
                do {
                    cpl_status = nixlGpuGetXferStatus<level>(xfer_status);
                } while (cpl_status == NIXL_IN_PROG);
                if (cpl_status != NIXL_SUCCESS) {
                    if (lane_id == 0) {
                        printf("xfer completion failed with status %d\n", cpl_status);
                    }
                    return;
                }
                if (timed_iter && lane_id == 0) {
                    completion_end = clock64();
                }
            }

            if (lane_id == 0) {
                wait_sequence_number(recv_counter,
                                     i + 1); // Wait for the receiver to process the message
                uint64_t rtt_end = 0;
                if (timed_iter && measure_rtt) {
                    rtt_end = clock64();
                }
                if (timed_iter && measure_completion) {
                    record_cycle_sample(ctx.issue_stats, issue_end - issue_start);
                    record_cycle_sample(ctx.completion_stats, completion_end - issue_end);
                    if (measure_rtt) {
                        record_cycle_sample(ctx.peer_wait_stats, rtt_end - completion_end);
                    }
                }
                if (timed_iter && measure_rtt) {
                    record_cycle_sample(ctx.rtt_stats, rtt_end - rtt_start);
                }
            }
            if constexpr (is_warp) {
                __syncwarp(); // Ensure all threads are synchronized before the next iteration
            }
        } else {
            if (lane_id == 0) {
                wait_sequence_number(recv_counter, i + 1); // Wait for the sender to signal
            }
            if constexpr (is_warp) {
                __syncwarp(); // Ensure all threads see the updated counter
            }

            if (lane_id == 0) {
                *send_counter = i + 1; // Increment send counter to signal the sender
            }

            do_ping_async<level>(ctx, total_size, xfer_status);
        }
    }

    if (ctx.is_sender && lane_id == 0) {
        uint64_t end_time = clock64();
        *elapsed_device = end_time - start_time;
    }
}

// Explicit template instantiations for the desired levels
template __global__ void
nixl_pingpong_latency_kernel<nixl_gpu_level_t::THREAD>(gpu_bench_ctx ctx, uint64_t *elapsed_device);
template __global__ void
nixl_pingpong_latency_kernel<nixl_gpu_level_t::WARP>(gpu_bench_ctx ctx, uint64_t *elapsed_device);

void
launch_pingpong_thread(gpu_bench_ctx ctx, uint64_t *d_elapsed, cudaStream_t stream) {
    nixl_pingpong_latency_kernel<nixl_gpu_level_t::THREAD><<<1, 1, 0, stream>>>(ctx, d_elapsed);
}

void
launch_pingpong_warp(gpu_bench_ctx ctx, uint64_t *d_elapsed, cudaStream_t stream) {
    nixl_pingpong_latency_kernel<nixl_gpu_level_t::WARP><<<1, 32, 0, stream>>>(ctx, d_elapsed);
}

#ifdef NIXL_GPU_DEVICE_BACKEND_PROXY
#include "nixl_device_proxy.cuh"

cudaError_t
bench_proxy_publish_context(void *proxy_ctx) {
    return nixlProxyPublishContext(static_cast<nixlProxyDeviceContextData *>(proxy_ctx));
}

cudaError_t
bench_proxy_clear_context() {
    return nixlProxyClearContext();
}
#endif
