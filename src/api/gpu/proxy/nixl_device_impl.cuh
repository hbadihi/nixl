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
#ifndef NIXL_SRC_API_GPU_PROXY_NIXL_DEVICE_IMPL_CUH
#define NIXL_SRC_API_GPU_PROXY_NIXL_DEVICE_IMPL_CUH

#include "nixl_device_proxy.cuh"
#include "nixl_types.h"

namespace nixl::gpu::proxy_impl {

template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ __forceinline__ nixl_status_t
get_xfer_status(nixlGpuXferStatusH &xfer_status) {
    uint32_t lane_id, num_lanes;
    nixlProxyExecInit<level>(lane_id, num_lanes);

    ProxyDeviceContext *ctx = load_proxy_context();

    nixl_status_t status = NIXL_IN_PROG;
    if (lane_id == 0) {
        if (ctx == nullptr) {
            status = NIXL_ERR_NOT_SUPPORTED;
        } else {
            status = ctx->pollXferStatus(xfer_status);
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
        // Only global lane 0 (block 0, thread 0) has valid xfer_status.
        // Publish the poll result to a device-memory scratch word so all
        // blocks can read it after grid sync.
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
__device__ __forceinline__ nixl_status_t
put(const nixlMemViewElem &src,
    const nixlMemViewElem &dst,
    size_t size,
    unsigned channel_id = 0,
    uint64_t flags = 0,
    nixlGpuXferStatusH *xfer_status = nullptr) {

    uint32_t lane_id, num_lanes;
    nixlProxyExecInit<level>(lane_id, num_lanes);
    nixl_status_t status = NIXL_IN_PROG;
    if (lane_id == 0) {
        ProxyDeviceContext *ctx = load_proxy_context();
        if (ctx == nullptr) {
            status = NIXL_ERR_NOT_SUPPORTED;
        } else {
            status = ctx->enqueue(
                ProxySubmission{
                    .opcode               = ProxyOpcode::PUT,
                    .channel_id           = static_cast<uint32_t>(channel_id),
                    .flags                = flags,
                    .src_proxy_memview_id = proxyMemViewIdFromHandle(src.mvh),
                    .src_index            = src.index,
                    .src_offset           = src.offset,
                    .dst_proxy_memview_id = proxyMemViewIdFromHandle(dst.mvh),
                    .dst_index            = dst.index,
                    .dst_offset           = dst.offset,
                    .size                 = size},
                xfer_status);
        }
    }
    nixlProxySync<level>();
    return status;
}

template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ __forceinline__ nixl_status_t
atomic_add(uint64_t value,
           const nixlMemViewElem &counter,
           unsigned channel_id = 0,
           uint64_t flags = 0,
           nixlGpuXferStatusH *xfer_status = nullptr) {
    uint32_t lane_id, num_lanes;
    nixlProxyExecInit<level>(lane_id, num_lanes);
    nixl_status_t status = NIXL_IN_PROG;
    if (lane_id == 0) {
        ProxyDeviceContext *ctx = load_proxy_context();
        if (ctx == nullptr) {
            status = NIXL_ERR_NOT_SUPPORTED;
        } else {
            status = ctx->enqueue(
                ProxySubmission{
                    .opcode               = ProxyOpcode::ATOMIC_ADD,
                    .channel_id           = static_cast<uint32_t>(channel_id),
                    .flags                = flags,
                    .dst_proxy_memview_id = proxyMemViewIdFromHandle(counter.mvh),
                    .dst_index            = counter.index,
                    .dst_offset           = counter.offset,
                    .size                 = sizeof(uint64_t),
                    .value                = value},
                xfer_status);
        }
    }
    nixlProxySync<level>();
    return status;
}

__device__ __forceinline__ void *
get_ptr(nixlMemViewH, size_t) {
    return nullptr;
}

} // namespace nixl::gpu::proxy_impl

#endif // NIXL_SRC_API_GPU_PROXY_NIXL_DEVICE_IMPL_CUH
