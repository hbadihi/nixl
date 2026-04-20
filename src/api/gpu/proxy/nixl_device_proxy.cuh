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
    CompletionSlot *slot;  // device pointer to the channel's CompletionSlot
    uint64_t        op_idx;
};
static_assert(sizeof(ProxyXferStatus) <= sizeof(nixlGpuXferStatusH),
              "ProxyXferStatus must fit in nixlGpuXferStatusH::storage");

// Defined in nixl_device_proxy.cu and read by device kernels through
// load_proxy_context().
extern __device__ ProxyDeviceContext *g_nixl_proxy_ctx;
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

__device__ inline uint64_t
proxyMemViewIdFromHandle(nixlMemViewH mvh) {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(mvh));
}

__device__ inline ProxyDeviceContext *
load_proxy_context() {
    return g_nixl_proxy_ctx;
}

static_assert(sizeof(WorkRing::running_op_idx) == 8,
              "running_op_idx must be 64-bit to avoid wrap-around false completions");
static_assert(sizeof(CompletionSlot::completed_idx) == 8,
              "completed_idx must be 64-bit to match running_op_idx");

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
            if (shut.load(cuda::memory_order_acquire)) {
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
            ProxyXferStatus pxs{channel_view.completion_slot, submission.op_idx};
            memcpy(xfer_status->storage, &pxs, sizeof(ProxyXferStatus));
        }

        return NIXL_IN_PROG;
    }

    // Poll the completion slot recorded by enqueue().  Returns NIXL_SUCCESS
    // once the CPU proxy worker has published a completion with completed_idx
    // >= the op_idx stored in xfer_status, otherwise NIXL_IN_PROG.
    //
    // The completion_slot lives in HBM; the CPU writes via cudaMemcpy.
    // An acquire load on completed_idx ensures next_status is also visible.
    __device__ inline nixl_status_t
    pollXferStatus(const nixlGpuXferStatusH &xfer_status) const {
        const ProxyXferStatus *pxs =
            reinterpret_cast<const ProxyXferStatus *>(xfer_status.storage);

        cuda::atomic_ref<uint64_t, cuda::thread_scope_system> comp_idx(
            pxs->slot->completed_idx);

        if (comp_idx.load(cuda::memory_order_acquire) >= pxs->op_idx) {
            // The acquire above orders the read of next_status.
            return pxs->slot->next_status;
        }

        return NIXL_IN_PROG;
    }
};

#endif // NIXL_SRC_API_GPU_PROXY_NIXL_DEVICE_PROXY_CUH
