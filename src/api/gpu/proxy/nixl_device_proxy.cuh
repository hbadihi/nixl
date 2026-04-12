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
    CompletionSlot *slot;  // device pointer to the channel's CompletionSlot
    uint64_t        op_idx;
};
static_assert(sizeof(ProxyXferStatus) <= sizeof(nixlGpuXferStatusH),
              "ProxyXferStatus must fit in nixlGpuXferStatusH::storage");

// Defined in nixl_device_proxy.cu and read by device kernels through
// load_proxy_context().
extern __device__ ProxyDeviceContext *g_nixl_proxy_ctx;

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
        if (num_channels == 0) {
            return NIXL_ERR_INVALID_PARAM;
        }

        uint32_t ch = submission.channel_id % num_channels;
        ProxyChannelView &ch_view = channels[ch];
        WorkRing         *ring    = ch_view.work_ring;

        cuda::atomic_ref<uint32_t, cuda::thread_scope_system> prod(*ring->producer_idx);
        cuda::atomic_ref<uint32_t, cuda::thread_scope_system> cons(*ring->consumer_idx);

        uint32_t prod_val = prod.load(cuda::memory_order_relaxed);

        // Spin until there is space in the ring.
        while (prod_val - cons.load(cuda::memory_order_acquire) >= ring->depth) {
            prod_val = prod.load(cuda::memory_order_relaxed);
        }

        // Plain write: ordered by the release store below.
        submission.op_idx = prod_val;
        ring->records[prod_val % ring->depth] = submission;

        // Publish the new entry to the CPU proxy worker.
        prod.store(prod_val + 1, cuda::memory_order_release);

        if (xfer_status != nullptr) {
            ProxyXferStatus pxs{ch_view.completion_slot, submission.op_idx};
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

        cuda::atomic_ref<uint64_t, cuda::thread_scope_system> cidx(
            pxs->slot->completed_idx);

        if (cidx.load(cuda::memory_order_acquire) >= pxs->op_idx) {
            // The acquire above orders the read of next_status.
            return pxs->slot->next_status;
        }

        return NIXL_IN_PROG;
    }
};

#endif // NIXL_SRC_API_GPU_PROXY_NIXL_DEVICE_PROXY_CUH
