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
#include <cooperative_groups.h>
#include <stdio.h>

#include "../common/nixl_device_types.cuh"
#include "../../../core/device_proxy/proxy_protocol.h"

struct ProxyDeviceContext;

// Overlay struct written into nixlGpuXferStatusH::storage by enqueue()
// and read back by pollXferStatus().  Must fit within the 64-byte opaque blob.
struct ProxyXferStatus {
    CompletionSlot *slot;          // device pointer to the channel's CompletionSlot
    uint64_t       *dequeued_idx;  // device pointer to the channel's dequeue ack counter
    uint64_t       *prepared_idx;  // device pointer to the channel's prepare ack counter
    uint64_t       *submitted_idx; // device pointer to the channel's submit ack counter
    uint64_t        op_idx;
};
static_assert(sizeof(ProxyXferStatus) <= sizeof(nixlGpuXferStatusH),
              "ProxyXferStatus must fit in nixlGpuXferStatusH::storage");

// Defined in nixl_device_proxy.cu and read by device kernels through
// load_proxy_context().
extern __device__ __constant__ ProxyDeviceContext *g_nixl_proxy_ctx;
extern __device__ int32_t g_nixl_proxy_grid_scratch;

// Host-callable helpers. Keeping these inline in CUDA translation units avoids
// cross-DSO symbol ownership issues for g_nixl_proxy_ctx.
__host__ inline cudaError_t
nixlProxyPublishContext(ProxyDeviceContextData *ctx) {
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

static_assert(sizeof(WorkRing::running_op_idx) == 8,
              "running_op_idx must be 64-bit to avoid wrap-around false completions");
static_assert(sizeof(CompletionSlot::completed_idx) == 8,
              "completed_idx must be 64-bit to match running_op_idx");
static_assert(sizeof(*ProxyChannelView{}.dequeued_idx) == 8,
              "dequeued_idx must be 64-bit to match running_op_idx");
static_assert(sizeof(*ProxyChannelView{}.prepared_idx) == 8,
              "prepared_idx must be 64-bit to match running_op_idx");
static_assert(sizeof(*ProxyChannelView{}.submitted_idx) == 8,
              "submitted_idx must be 64-bit to match running_op_idx");

/**
* Initialize the lane_id and num_lanes variables for the given level.
* @param level The level to initialize the lane_id and num_lanes for.
* @param lane_id The lane_id variable to initialize.
* @param num_lanes The num_lanes variable to initialize.
*/
template<nixl_gpu_level_t level>
__device__ inline void nixlProxyExecInit(uint32_t &lane_id, uint32_t &num_lanes) {
    switch (level) {
    case nixl_gpu_level_t::THREAD:
        lane_id = 0;
        num_lanes = 1;
        break;
    case nixl_gpu_level_t::WARP:
        lane_id = threadIdx.x % warpSize;
        num_lanes = warpSize;
        break;
    case nixl_gpu_level_t::BLOCK:
        lane_id = threadIdx.x;
        num_lanes = blockDim.x;
        break;
    case nixl_gpu_level_t::GRID:
        lane_id = threadIdx.x + blockIdx.x * blockDim.x;
        num_lanes = blockDim.x * gridDim.x;
        break;
    }
}

/**
* Synchronize the threads at the given level.
* @param level The level to synchronize the threads for.
*/
template<nixl_gpu_level_t level>
__device__ inline void nixlProxySync() {
    switch (level) {
    case nixl_gpu_level_t::THREAD:
        break;
    case nixl_gpu_level_t::WARP:
        __syncwarp();
        break;
    case nixl_gpu_level_t::BLOCK:
        __syncthreads();
        break;
    case nixl_gpu_level_t::GRID:
        auto g = cooperative_groups::this_grid();
        g.sync();
        break;
    }
}

struct ProxyDeviceContext : ProxyDeviceContextData {

    // Enqueue a transfer submission into the MPSC work ring for the selected
    // channel, spinning if the ring is full.  Optionally records a completion
    // token in *xfer_status for later polling via pollXferStatus().
    //
    // producer_idx lives in HBM; consumer_idx lives in pinned host memory
    // (accessible from device via UVA mapped pointer).  Both are accessed with
    // system-scope atomics so the CPU proxy worker sees the update coherently.
    __device__ inline nixl_status_t
    enqueue(ProxySubmission submission, nixlGpuXferStatusH *xfer_status = nullptr) {
        if (submission.channel_id >= num_channels) {
            return NIXL_ERR_INVALID_PARAM;
        }

        ProxyChannelView &channel_view = channels[submission.channel_id];
        WorkRing         *ring    = channel_view.work_ring;

        cuda::atomic_ref<uint32_t, cuda::thread_scope_system> prod(*ring->producer_idx);
        cuda::atomic_ref<uint32_t, cuda::thread_scope_system> cons(*ring->consumer_idx);
        cuda::atomic_ref<uint32_t, cuda::thread_scope_system> shut(*shutdown_word);

        // Atomically claim a unique slot in the ring.
        uint32_t my_slot = prod.fetch_add(1, cuda::memory_order_relaxed);

        // Spin until the claimed slot has space (consumer has freed it).
        while (my_slot - cons.load(cuda::memory_order_acquire) >= ring->depth) {
            if (shut.load(cuda::memory_order_acquire)
                == static_cast<uint32_t>(ProxyControlState::Shutdown)) {
                return NIXL_ERR_BACKEND;
            }
        }

        cuda::atomic_ref<uint64_t, cuda::thread_scope_system> op_idx(ring->running_op_idx);
        submission.op_idx = op_idx.fetch_add(1, cuda::memory_order_relaxed);
        ring->records[my_slot % ring->depth] = submission;

        // Signal this slot is ready for the consumer.  The release
        // guarantees the record write above is visible before the
        // consumer reads it via an acquire load on ready_flag.
        cuda::atomic_ref<uint32_t, cuda::thread_scope_system> ready(
            ring->records[my_slot % ring->depth].ready_flag);
        ready.store(1, cuda::memory_order_release);

        if (xfer_status != nullptr) {
            ProxyXferStatus pxs{channel_view.completion_slot,
                                channel_view.dequeued_idx,
                                channel_view.prepared_idx,
                                channel_view.submitted_idx,
                                submission.op_idx};
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
        const nixl_status_t current_status = pxs->slot->next_status;

        if (completed_idx > pxs->op_idx) {
            // The success frontier has advanced past this op.
            return NIXL_SUCCESS;
        }
        if (completed_idx == pxs->op_idx) {
            return current_status;
        }
        if (current_status < 0) {
            // An earlier terminal error latched the channel, so later queued
            // ops observe the same error instead of spinning forever.
            return current_status;
        }

        return NIXL_IN_PROG;
    }

    // Poll a stage acknowledgement recorded by enqueue().
    //
    // Stage counters advance on the CPU proxy worker. Completion errors are
    // also observed so callers do not spin
    // forever if the submission cannot reach the backend boundary.
    __device__ inline nixl_status_t
    pollStage(const nixlGpuXferStatusH &xfer_status, uint64_t *stage_idx) const {
        const ProxyXferStatus *pxs =
            reinterpret_cast<const ProxyXferStatus *>(xfer_status.storage);

        cuda::atomic_ref<uint64_t, cuda::thread_scope_system> ack_idx(*stage_idx);
        const uint64_t acked_idx = ack_idx.load(cuda::memory_order_acquire);
        if (acked_idx >= pxs->op_idx) {
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

    __device__ inline nixl_status_t
    pollDequeued(const nixlGpuXferStatusH &xfer_status) const {
        const ProxyXferStatus *pxs =
            reinterpret_cast<const ProxyXferStatus *>(xfer_status.storage);
        return pollStage(xfer_status, pxs->dequeued_idx);
    }

    __device__ inline nixl_status_t
    pollPrepared(const nixlGpuXferStatusH &xfer_status) const {
        const ProxyXferStatus *pxs =
            reinterpret_cast<const ProxyXferStatus *>(xfer_status.storage);
        return pollStage(xfer_status, pxs->prepared_idx);
    }

    __device__ inline nixl_status_t
    pollSubmitted(const nixlGpuXferStatusH &xfer_status) const {
        const ProxyXferStatus *pxs =
            reinterpret_cast<const ProxyXferStatus *>(xfer_status.storage);
        return pollStage(xfer_status, pxs->submitted_idx);
    }
};

enum class ProxyStageAck : uint32_t {
    Dequeued = 0,
    Prepared = 1,
    Submitted = 2,
};

template<ProxyStageAck stage, nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ inline nixl_status_t
nixlProxyPollStage(nixlGpuXferStatusH &xfer_status) {
    uint32_t lane_id, num_lanes;
    nixlProxyExecInit<level>(lane_id, num_lanes);

    ProxyDeviceContext *ctx = load_proxy_context();

    nixl_status_t status = NIXL_IN_PROG;
    if (lane_id == 0) {
        if (ctx == nullptr) {
            status = NIXL_ERR_NOT_SUPPORTED;
        } else {
            if constexpr (stage == ProxyStageAck::Dequeued) {
                status = ctx->pollDequeued(xfer_status);
            } else if constexpr (stage == ProxyStageAck::Prepared) {
                status = ctx->pollPrepared(xfer_status);
            } else {
                status = ctx->pollSubmitted(xfer_status);
            }
        }
    }

    switch (level) {
    case nixl_gpu_level_t::THREAD:
        break;

    case nixl_gpu_level_t::WARP:
        status = static_cast<nixl_status_t>(
            __shfl_sync(0xffffffff, static_cast<int>(status), 0));
        break;

    case nixl_gpu_level_t::BLOCK: {
        __shared__ nixl_status_t s_status;
        if (threadIdx.x == 0) {
            s_status = status;
        }
        __syncthreads();
        status = s_status;
        break;
    }

    case nixl_gpu_level_t::GRID: {
        cuda::atomic_ref<int32_t, cuda::thread_scope_device> scratch(
            g_nixl_proxy_grid_scratch);
        if (lane_id == 0) {
            scratch.store(static_cast<int32_t>(status),
                          cuda::memory_order_relaxed);
        }
        cooperative_groups::this_grid().sync();

        __shared__ nixl_status_t s_status;
        if (threadIdx.x == 0) {
            s_status = static_cast<nixl_status_t>(
                scratch.load(cuda::memory_order_relaxed));
        }
        __syncthreads();
        status = s_status;
        break;
    }
    }

    return status;
}

template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ inline nixl_status_t
nixlProxyPollDequeued(nixlGpuXferStatusH &xfer_status) {
    return nixlProxyPollStage<ProxyStageAck::Dequeued, level>(xfer_status);
}

template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ inline nixl_status_t
nixlProxyPollPrepared(nixlGpuXferStatusH &xfer_status) {
    return nixlProxyPollStage<ProxyStageAck::Prepared, level>(xfer_status);
}

template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ inline nixl_status_t
nixlProxyPollSubmitted(nixlGpuXferStatusH &xfer_status) {
    return nixlProxyPollStage<ProxyStageAck::Submitted, level>(xfer_status);
}

#endif // NIXL_SRC_API_GPU_PROXY_NIXL_DEVICE_PROXY_CUH
