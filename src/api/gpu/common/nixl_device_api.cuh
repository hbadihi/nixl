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
#ifndef NIXL_SRC_API_GPU_COMMON_NIXL_DEVICE_API_CUH
#define NIXL_SRC_API_GPU_COMMON_NIXL_DEVICE_API_CUH

#include <cstring>

#include <gpu/nixl_device_config.h>

#include "nixl_device_memview.cuh"
#include "nixl_device_types.cuh"

#include "../proxy/nixl_device_impl.cuh"

#if defined(NIXL_HAVE_UCX_GPU_DEVICE_API)
#include "../ucx/nixl_device_impl.cuh"
#else
namespace nixl::gpu::ucx_impl {

template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ inline nixl_status_t
get_xfer_status(nixlGpuXferStatusH &) {
    return NIXL_ERR_NOT_SUPPORTED;
}

template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ inline nixl_status_t
put(const nixlMemViewElem &,
    const nixlMemViewElem &,
    size_t,
    unsigned = 0,
    uint64_t = 0,
    nixlGpuXferStatusH * = nullptr) {
    return NIXL_ERR_NOT_SUPPORTED;
}

template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ inline nixl_status_t
atomic_add(uint64_t,
           const nixlMemViewElem &,
           unsigned = 0,
           uint64_t = 0,
           nixlGpuXferStatusH * = nullptr) {
    return NIXL_ERR_NOT_SUPPORTED;
}

__device__ inline void *
get_ptr(nixlMemViewH, size_t) {
    return nullptr;
}

} // namespace nixl::gpu::ucx_impl
#endif

#if defined(NIXL_GPU_DEVICE_BACKEND_PROXY)
namespace nixl::gpu { namespace selected_impl = proxy_impl; }
#elif defined(NIXL_GPU_DEVICE_BACKEND_UCX)
namespace nixl::gpu { namespace selected_impl = ucx_impl; }
#else
#error "No GPU device backend implementation selected"
#endif

namespace nixl::gpu::api {
namespace detail {

enum class MemViewBackend : uint8_t {
    INVALID,
    UCX,
    PROXY,
    UNSUPPORTED,
};

struct DecodedMemView {
    MemViewBackend backend;
    nixlMemViewH backend_handle;
};

__device__ __forceinline__ DecodedMemView
decode_memview(nixlMemViewH handle) {
    if (handle == nullptr) {
        return {MemViewBackend::INVALID, nullptr};
    }
    const auto *wrapper = static_cast<const nixlDeviceMemView *>(handle);
    MemViewBackend backend;
    switch (wrapper->backend) {
    case nixlDeviceMemViewBackend::UCX:
        backend = MemViewBackend::UCX;
        break;
    case nixlDeviceMemViewBackend::PROXY:
        backend = MemViewBackend::PROXY;
        break;
    default:
        return {MemViewBackend::UNSUPPORTED, nullptr};
    }
    const nixlMemViewH backend_handle = wrapper->backend_handle;
    return backend_handle == nullptr ? DecodedMemView{MemViewBackend::INVALID, nullptr} :
                                       DecodedMemView{backend, backend_handle};
}

template<nixl_gpu_level_t level>
__device__ __forceinline__ bool
execution_leader() {
    if constexpr (level == nixl_gpu_level_t::THREAD) {
        return true;
    } else if constexpr (level == nixl_gpu_level_t::WARP) {
        return threadIdx.x % warpSize == 0;
    } else if constexpr (level == nixl_gpu_level_t::BLOCK) {
        return threadIdx.x == 0;
    } else {
        return blockIdx.x == 0 && threadIdx.x == 0;
    }
}

__device__ __forceinline__ nixlDeviceXferStatusFooter
load_footer(const nixlGpuXferStatusH &status) {
    nixlDeviceXferStatusFooter footer{};
    memcpy(&footer, status.storage + NIXL_GPU_XFER_STATUS_PAYLOAD_SIZE, sizeof(footer));
    return footer;
}

template<nixl_gpu_level_t level>
__device__ __forceinline__ void
write_footer(nixlGpuXferStatusH *status,
             nixl_status_t submission_status,
             nixlDeviceXferStatusBackend backend) {
    if (status == nullptr || submission_status != NIXL_IN_PROG || !execution_leader<level>()) {
        return;
    }
    const nixlDeviceXferStatusFooter footer{
        NIXL_DEVICE_XFER_STATUS_MAGIC,
        NIXL_DEVICE_XFER_STATUS_ABI_VERSION,
        static_cast<uint8_t>(backend),
        0,
    };
    memcpy(status->storage + NIXL_GPU_XFER_STATUS_PAYLOAD_SIZE, &footer, sizeof(footer));
}

} // namespace detail

template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ inline nixl_status_t
get_xfer_status(nixlGpuXferStatusH &xfer_status) {
    return selected_impl::get_xfer_status<level>(xfer_status);
}

template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ inline nixl_status_t
put(const nixlMemViewElem &src,
    const nixlMemViewElem &dst,
    size_t size,
    unsigned channel_id = 0,
    uint64_t flags = 0,
    nixlGpuXferStatusH *xfer_status = nullptr) {
    const auto dst_view = detail::decode_memview(dst.mvh);
    const auto src_view = detail::decode_memview(src.mvh);
    if (dst_view.backend == detail::MemViewBackend::UNSUPPORTED ||
        src_view.backend == detail::MemViewBackend::UNSUPPORTED) {
        return NIXL_ERR_NOT_SUPPORTED;
    }
    if (dst_view.backend == detail::MemViewBackend::INVALID ||
        src_view.backend == detail::MemViewBackend::INVALID ||
        dst_view.backend != src_view.backend) {
        return NIXL_ERR_INVALID_PARAM;
    }

    const nixlMemViewElem backend_src{src_view.backend_handle, src.index, src.offset};
    const nixlMemViewElem backend_dst{dst_view.backend_handle, dst.index, dst.offset};
    if (dst_view.backend == detail::MemViewBackend::UCX) {
        return ucx_impl::put<level>(
            backend_src, backend_dst, size, channel_id, flags, xfer_status);
    }
    if constexpr (level == nixl_gpu_level_t::GRID) {
        return NIXL_ERR_NOT_SUPPORTED;
    } else {
        return proxy_impl::put<level>(
            backend_src, backend_dst, size, channel_id, flags, xfer_status);
    }
}

template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ inline nixl_status_t
atomic_add(uint64_t value,
           const nixlMemViewElem &counter,
           unsigned channel_id = 0,
           uint64_t flags = 0,
           nixlGpuXferStatusH *xfer_status = nullptr) {
    const auto view = detail::decode_memview(counter.mvh);
    if (view.backend == detail::MemViewBackend::UNSUPPORTED) {
        return NIXL_ERR_NOT_SUPPORTED;
    }
    if (view.backend == detail::MemViewBackend::INVALID) {
        return NIXL_ERR_INVALID_PARAM;
    }

    const nixlMemViewElem backend_counter{view.backend_handle, counter.index, counter.offset};
    if (view.backend == detail::MemViewBackend::UCX) {
        return ucx_impl::atomic_add<level>(
            value, backend_counter, channel_id, flags, xfer_status);
    }
    if constexpr (level == nixl_gpu_level_t::GRID) {
        return NIXL_ERR_NOT_SUPPORTED;
    } else {
        return proxy_impl::atomic_add<level>(
            value, backend_counter, channel_id, flags, xfer_status);
    }
}

__device__ inline void *
get_ptr(nixlMemViewH mvh, size_t index) {
    const auto view = detail::decode_memview(mvh);
    switch (view.backend) {
    case detail::MemViewBackend::UCX:
        return ucx_impl::get_ptr(view.backend_handle, index);
    case detail::MemViewBackend::PROXY:
        return proxy_impl::get_ptr(view.backend_handle, index);
    default:
        return nullptr;
    }
}

} // namespace nixl::gpu::api

#endif // NIXL_SRC_API_GPU_COMMON_NIXL_DEVICE_API_CUH
