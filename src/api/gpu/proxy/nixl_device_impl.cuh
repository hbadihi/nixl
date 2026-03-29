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

namespace nixl::gpu::proxy_impl {

// Shape-only handoff: keep proxy entry points and signatures, but do not
// preserve device-side submission logic in this scaffold.

template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ inline nixl_status_t
get_xfer_status(nixlGpuXferStatusH &xfer_status) {
    const ProxyDeviceContext *ctx = load_proxy_context();
    if (ctx == nullptr) {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    (void)xfer_status;
    return ctx->pollXferStatus(xfer_status);
}

template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ inline nixl_status_t
put(const nixlMemViewElem &src,
    const nixlMemViewElem &dst,
    size_t size,
    unsigned channel_id = 0,
    uint64_t flags = 0,
    nixlGpuXferStatusH *xfer_status = nullptr) {
    ProxyDeviceContext *ctx = load_proxy_context();
    if (ctx == nullptr) {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    ProxySubmission submission{};
    submission.opcode = ProxyOpcode::PUT;
    submission.channel_id = static_cast<uint32_t>(channel_id);
    submission.flags = flags;
    submission.src_proxy_memview_id = proxyMemViewIdFromHandle(src.mvh);
    submission.src_index = src.index;
    submission.src_offset = src.offset;
    submission.dst_proxy_memview_id = proxyMemViewIdFromHandle(dst.mvh);
    submission.dst_index = dst.index;
    submission.dst_offset = dst.offset;
    submission.size = size;
    return ctx->enqueue(submission, xfer_status);
}

template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ inline nixl_status_t
atomic_add(uint64_t value,
           const nixlMemViewElem &counter,
           unsigned channel_id = 0,
           uint64_t flags = 0,
           nixlGpuXferStatusH *xfer_status = nullptr) {
    ProxyDeviceContext *ctx = load_proxy_context();
    if (ctx == nullptr) {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    ProxySubmission submission{};
    submission.opcode = ProxyOpcode::ATOMIC_ADD;
    submission.channel_id = static_cast<uint32_t>(channel_id);
    submission.flags = flags;
    submission.dst_proxy_memview_id = proxyMemViewIdFromHandle(counter.mvh);
    submission.dst_index = counter.index;
    submission.dst_offset = counter.offset;
    submission.value = value;
    return ctx->enqueue(submission, xfer_status);
}

__device__ inline void *
get_ptr(nixlMemViewH, size_t) {
    return nullptr;
}

} // namespace nixl::gpu::proxy_impl

#endif // NIXL_SRC_API_GPU_PROXY_NIXL_DEVICE_IMPL_CUH
