#include "bench_kernel.cuh"
#include "nixl_device.cuh"
#include "nixl_types.h"
#include <cstdint>

__device__ static void
wait_sequence_number(volatile uint64_t *counter, uint64_t expected_value) {
    while (*counter < expected_value) {
        __nanosleep(50); // Sleep for 50 nanoseconds to reduce contention, taken from UCX
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
    ++s->count;
    s->sum += c;
    s->sum_sq += double(c) * double(c);
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

#ifdef NIXL_GPU_DEVICE_BACKEND_PROXY
template<nixl_gpu_level_t level, ProxyStageAck stage>
__device__ static nixl_status_t
wait_proxy_stage(nixlGpuXferStatusH &xfer_status) {
    nixl_status_t status;
    do {
        status = nixlProxyPollStage<stage, level>(xfer_status);
    } while (status == NIXL_IN_PROG);
    return status;
}
#endif

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
    const bool measure_submit =
        ctx.is_sender && ctx.issue_stats != nullptr && ctx.submit_stats != nullptr;
#ifdef NIXL_GPU_DEVICE_BACKEND_PROXY
    const bool measure_proxy_stages =
        measure_submit && ctx.dequeue_stats != nullptr && ctx.prepare_stats != nullptr
        && ctx.post_submit_stats != nullptr;
#endif
    const bool measure_rtt = ctx.is_sender && ctx.rtt_stats != nullptr;

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
            if (timed_iter && measure_submit && lane_id == 0) {
                issue_start = clock64();
            }
            nixl_status_t put_status =
                do_put_async<level>(ctx.local_mvh, ctx.remote_mvh, total_size, xfer_status);
            if constexpr (is_warp) {
                put_status = static_cast<nixl_status_t>(
                    __shfl_sync(0xffffffff, static_cast<int>(put_status), 0));
            }
            if (put_status != NIXL_IN_PROG) {
                return;
            }
            uint64_t issue_end = 0;
            if (timed_iter && measure_submit && lane_id == 0) {
                issue_end = clock64();
            }
#ifdef NIXL_GPU_DEVICE_BACKEND_PROXY
            uint64_t dequeue_end = 0;
            uint64_t prepare_end = 0;
#endif
            uint64_t submit_end = 0;
            if (timed_iter && measure_submit) {
#ifdef NIXL_GPU_DEVICE_BACKEND_PROXY
                nixl_status_t stage_status = NIXL_SUCCESS;

                if (measure_proxy_stages) {
                    stage_status = wait_proxy_stage<level, ProxyStageAck::Dequeued>(xfer_status);
                    if (stage_status != NIXL_SUCCESS) {
                        if (lane_id == 0) {
                            printf("dequeue boundary failed with status %d\n", stage_status);
                        }
                        return;
                    }
                    if (lane_id == 0) {
                        dequeue_end = clock64();
                    }

                    stage_status = wait_proxy_stage<level, ProxyStageAck::Prepared>(xfer_status);
                    if (stage_status != NIXL_SUCCESS) {
                        if (lane_id == 0) {
                            printf("prepare boundary failed with status %d\n", stage_status);
                        }
                        return;
                    }
                    if (lane_id == 0) {
                        prepare_end = clock64();
                    }
                }

                stage_status = wait_proxy_stage<level, ProxyStageAck::Submitted>(xfer_status);
                if (stage_status != NIXL_SUCCESS) {
                    if (lane_id == 0) {
                        printf("submit boundary failed with status %d\n", stage_status);
                    }
                    return;
                }
#endif
                if (lane_id == 0) {
#ifdef NIXL_GPU_DEVICE_BACKEND_PROXY
                    submit_end = clock64();
#else
                    submit_end = issue_end;
#endif
                }
            }

            if (lane_id == 0) {
                wait_sequence_number(recv_counter,
                                     i + 1); // Wait for the receiver to process the message
                uint64_t rtt_end = 0;
                if (timed_iter && measure_rtt) {
                    rtt_end = clock64();
                }
                if (timed_iter && measure_submit) {
                    record_cycle_sample(ctx.issue_stats, issue_end - issue_start);
#ifdef NIXL_GPU_DEVICE_BACKEND_PROXY
                    if (measure_proxy_stages) {
                        record_cycle_sample(ctx.dequeue_stats, dequeue_end - issue_end);
                        record_cycle_sample(ctx.prepare_stats, prepare_end - dequeue_end);
                    }
#endif
#ifdef NIXL_GPU_DEVICE_BACKEND_PROXY
                    if (measure_proxy_stages) {
                        record_cycle_sample(ctx.submit_stats, submit_end - prepare_end);
                    } else
#endif
                    {
                        record_cycle_sample(ctx.submit_stats, submit_end - issue_start);
                    }
                }
                if (timed_iter && measure_rtt) {
#ifdef NIXL_GPU_DEVICE_BACKEND_PROXY
                    if (measure_proxy_stages) {
                        record_cycle_sample(ctx.post_submit_stats, rtt_end - submit_end);
                    }
#endif
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

            do_put_async<level>(ctx.local_mvh, ctx.remote_mvh, total_size, xfer_status);
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
    return nixlProxyPublishContext(static_cast<ProxyDeviceContextData *>(proxy_ctx));
}

cudaError_t
bench_proxy_clear_context() {
    return nixlProxyClearContext();
}
#endif
