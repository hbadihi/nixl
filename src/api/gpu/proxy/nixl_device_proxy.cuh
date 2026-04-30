/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef NIXL_SRC_API_GPU_PROXY_NIXL_DEVICE_PROXY_CUH
#define NIXL_SRC_API_GPU_PROXY_NIXL_DEVICE_PROXY_CUH

#include <cuda/atomic>
#include <stdio.h>

#include "../common/nixl_device_types.cuh"
#include "../../../core/device_proxy/proxy_protocol.h"

struct ProxyDeviceContext;

// Overlay struct written into nixlGpuXferStatusH::storage by enqueue()
// and read back by pollXferStatus().  Must fit within the 64-byte opaque blob.
struct ProxyXferStatus {
    nixlProxyCompletionSlot *slot;  // device pointer to the channel's nixlProxyCompletionSlot
    uint64_t *submitted_idx;       // device pointer to the channel's submit ack counter
    uint64_t        op_idx;
};
static_assert(sizeof(ProxyXferStatus) <= sizeof(nixlGpuXferStatusH),
              "ProxyXferStatus must fit in nixlGpuXferStatusH::storage");

// Defined in nixl_device_proxy.cu and read by device kernels through
// load_proxy_context().
extern __device__ __constant__ ProxyDeviceContext *g_nixl_proxy_ctx;

// Host-callable helpers. ctx is the device pointer returned by ProxyRuntime.
// Keeping these inline in CUDA translation units avoids cross-DSO symbol
// ownership issues for g_nixl_proxy_ctx.
__host__ inline cudaError_t
nixlProxyPublishContext(nixlProxyDeviceContextData *ctx) {
    ProxyDeviceContext *device_ctx = reinterpret_cast<ProxyDeviceContext *>(ctx);
    cudaError_t err = cudaMemcpyToSymbol(g_nixl_proxy_ctx, &device_ctx, sizeof(ProxyDeviceContext *));
    if (err != cudaSuccess) {
        fprintf(stderr,
                "nixlProxyPublishContext: cudaMemcpyToSymbol failed: code=%d msg=%s\n",
                static_cast<int>(err),
                cudaGetErrorString(err));
    }
    return err;
}

__host__ inline cudaError_t
nixlProxyClearContext() {
    ProxyDeviceContext *null_ctx = nullptr;
    cudaError_t err = cudaMemcpyToSymbol(g_nixl_proxy_ctx, &null_ctx, sizeof(ProxyDeviceContext *));
    if (err != cudaSuccess) {
        fprintf(stderr,
                "nixlProxyClearContext: cudaMemcpyToSymbol failed: code=%d msg=%s\n",
                static_cast<int>(err),
                cudaGetErrorString(err));
    }
    return err;
}

__device__ __forceinline__  uint64_t
proxyMemViewIdFromHandle(nixlMemViewH mvh) {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(mvh));
}

__device__ __forceinline__  ProxyDeviceContext *
load_proxy_context() {
    return g_nixl_proxy_ctx;
}

static_assert(sizeof(*nixlProxyWorkRing{}.producer_ticket) == 8,
              "producer_ticket must be 64-bit to avoid wrap-around false completions");
static_assert(sizeof(*nixlProxyWorkRing{}.consumer_idx) == 8,
              "consumer_idx must be 64-bit to match producer_ticket");
static_assert(sizeof(*nixlProxyWorkRing{}.consumer_idx_cache) == 8,
              "consumer_idx_cache must be 64-bit to match producer_ticket");
static_assert(sizeof(nixlProxyCompletionSlot::completed_idx) == 8,
              "completed_idx must be 64-bit to match producer_ticket");
static_assert(sizeof(*nixlProxyChannelView{}.submitted_idx) == 8,
              "submitted_idx must be 64-bit to match producer_ticket");

// Select the leader lane for the requested collective scope.
template<nixl_gpu_level_t level>
__device__ inline void nixlProxyExecInit(uint32_t &lane_id) {
    static_assert(level != nixl_gpu_level_t::GRID,
                  "Proxy GPU backend does not support GRID-level operations");

    if constexpr (level == nixl_gpu_level_t::THREAD) {
        lane_id = 0;
    } else if constexpr (level == nixl_gpu_level_t::WARP) {
        lane_id = threadIdx.x % warpSize;
    } else if constexpr (level == nixl_gpu_level_t::BLOCK) {
        lane_id = threadIdx.x;
    }
}

template<nixl_gpu_level_t level>
__device__ inline void nixlProxySync() {
    static_assert(level != nixl_gpu_level_t::GRID,
                  "Proxy GPU backend does not support GRID-level operations");

    if constexpr (level == nixl_gpu_level_t::WARP) {
        __syncwarp();
    } else if constexpr (level == nixl_gpu_level_t::BLOCK) {
        __syncthreads();
    }
}

struct ProxyDeviceContext : nixlProxyDeviceContextData {

    // Enqueue a transfer submission into the MPSC work ring for the selected
    // channel, spinning if the ring is full.  Optionally records a completion
    // token in *xfer_status for later polling via pollXferStatus().
    //
    // producer_ticket lives in HBM and only needs device-scope atomicity.
    // consumer_idx lives in pinned host memory (accessible from device via
    // UVA mapped pointer). The device cache keeps the non-full path from
    // repeatedly touching host memory.
    __device__ inline nixl_status_t
    enqueue(nixlProxySubmission submission, nixlGpuXferStatusH *xfer_status = nullptr) {
        if (submission.channel_id >= num_channels) {
            return NIXL_ERR_INVALID_PARAM;
        }

        nixlProxyChannelView &channel_view = channels[submission.channel_id];
        nixlProxyWorkRing         *ring    = channel_view.work_ring;

        cuda::atomic_ref<uint64_t, cuda::thread_scope_device> producer_ticket(
            *ring->producer_ticket);
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system> cons(*ring->consumer_idx);
        cuda::atomic_ref<uint32_t, cuda::thread_scope_system> shut(*shutdown_word);

        // Atomically claim a unique slot in the ring.
        const uint64_t ticket = producer_ticket.fetch_add(1, cuda::memory_order_relaxed);

        // Fast path: use the device cache. Refresh from host only if the ring
        // appears full, since mapped-host loads are much slower than HBM loads.
        uint64_t cached_consumer_idx = *ring->consumer_idx_cache;
        if (ticket - cached_consumer_idx >= ring->depth) {
            do {
                cached_consumer_idx = cons.load(cuda::memory_order_acquire);
                *ring->consumer_idx_cache = cached_consumer_idx;
                if (ticket - cached_consumer_idx < ring->depth) {
                    break;
                }
                if (shut.load(cuda::memory_order_relaxed)
                    == static_cast<uint32_t>(nixl_proxy_control_state_t::SHUTDOWN)) {
                    return NIXL_ERR_BACKEND;
                }
            } while (true);
        }

        const uint64_t submission_op_idx = ticket + 1;
        const uint32_t slot = static_cast<uint32_t>(ticket % ring->depth);

        submission.op_idx = 0;
        ring->records[slot] = submission;

        // Signal this slot is ready for the consumer.  The release
        // guarantees the record write above is visible before the
        // consumer reads op_idx via an acquire load. op_idx == 0 means empty.
        // TODO: Currently using thread_scope_device reduces the overhead
        // compared to thread_scope_system. (No L2 flush). Revisit this choice to understand if a hidden race exists.
        cuda::atomic_ref<uint64_t, cuda::thread_scope_device> record_op_idx(
            ring->records[slot].op_idx);
        record_op_idx.store(submission_op_idx, cuda::memory_order_release);

        if (xfer_status != nullptr) {
            ProxyXferStatus pxs{channel_view.completion_slot,
                                channel_view.submitted_idx,
                                submission_op_idx};
            memcpy(xfer_status->storage, &pxs, sizeof(ProxyXferStatus));
        }

        return NIXL_IN_PROG;
    }

    // Poll the completion slot recorded by enqueue().
    //
    // The completion slot implements collapsed-CQ semantics:
    // - completed_idx > op_idx  => this op completed earlier, so it succeeded
    // - completed_idx == op_idx => next_status is this op's terminal status
    // - completed_idx < op_idx  => this op is still pending, unless an earlier
    //                              completion published a terminal error and
    //                              latched the channel
    __device__ inline nixl_status_t
    pollXferStatus(const nixlGpuXferStatusH &xfer_status) const {
        const ProxyXferStatus *pxs =
            reinterpret_cast<const ProxyXferStatus *>(xfer_status.storage);

        cuda::atomic_ref<uint64_t, cuda::thread_scope_system> comp_idx(
            pxs->slot->completed_idx);

        const uint64_t completed_idx = comp_idx.load(cuda::memory_order_acquire);
        if (completed_idx > pxs->op_idx) {
            return NIXL_SUCCESS;
        }
        const nixl_status_t current_status = pxs->slot->next_status;
        if (completed_idx == pxs->op_idx) {
            return current_status;
        }
        if (current_status < 0) {
            return current_status;
        }

        return NIXL_IN_PROG;
    }

    // Poll the submit acknowledgement recorded by enqueue().
    //
    // submitted_idx advances after backend_->submit returns on the CPU proxy
    // worker. Completion errors are also observed so callers do not spin
    // forever if the submission cannot reach the backend boundary.
    __device__ inline nixl_status_t
    pollSubmitted(const nixlGpuXferStatusH &xfer_status) const {
        const ProxyXferStatus *pxs =
            reinterpret_cast<const ProxyXferStatus *>(xfer_status.storage);

        cuda::atomic_ref<uint64_t, cuda::thread_scope_system> sub_idx(
            *pxs->submitted_idx);
        const uint64_t submitted_idx = sub_idx.load(cuda::memory_order_acquire);
        if (submitted_idx >= pxs->op_idx) {
            return NIXL_SUCCESS;
        }

        cuda::atomic_ref<uint64_t, cuda::thread_scope_system> comp_idx(
            pxs->slot->completed_idx);
        (void)comp_idx.load(cuda::memory_order_acquire);
        const nixl_status_t current_status = pxs->slot->next_status;
        if (current_status < 0) {
            return current_status;
        }

        return NIXL_IN_PROG;
    }
};

template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ inline nixl_status_t
nixlProxyPollSubmitted(nixlGpuXferStatusH &xfer_status) {
    uint32_t lane_id, num_lanes;
    nixlProxyExecInit<level>(lane_id, num_lanes);

    ProxyDeviceContext *ctx = load_proxy_context();

    nixl_status_t status = NIXL_IN_PROG;
    if (lane_id == 0) {
        if (ctx == nullptr) {
            status = NIXL_ERR_NOT_SUPPORTED;
        } else {
            status = ctx->pollSubmitted(xfer_status);
        }
    }

    if constexpr (level == nixl_gpu_level_t::WARP) {
        status = static_cast<nixl_status_t>(
            __shfl_sync(0xffffffff, static_cast<int>(status), 0));
    } else if constexpr (level == nixl_gpu_level_t::BLOCK) {
        __shared__ nixl_status_t s_status;
        if (threadIdx.x == 0) {
            s_status = status;
        }
        __syncthreads();
        status = s_status;
    }

    return status;
}

#endif // NIXL_SRC_API_GPU_PROXY_NIXL_DEVICE_PROXY_CUH
