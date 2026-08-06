/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "device_memview.h"

#include <utility>

#include <cuda_runtime.h>

#include "common/nixl_log.h"

nixlDeviceMemViewAllocation::~nixlDeviceMemViewAllocation() {
    reset();
}

nixlDeviceMemViewAllocation::nixlDeviceMemViewAllocation(
    nixlDeviceMemViewAllocation &&other) noexcept {
    *this = std::move(other);
}

nixlDeviceMemViewAllocation &
nixlDeviceMemViewAllocation::operator=(nixlDeviceMemViewAllocation &&other) noexcept {
    if (this != &other) {
        reset();
        handle_ = std::exchange(other.handle_, nullptr);
    }
    return *this;
}

nixl_status_t
nixlDeviceMemViewAllocation::create(nixlDeviceMemViewBackend backend,
                                    nixlMemViewH backend_handle,
                                    nixlDeviceMemViewAllocation &allocation) noexcept {
    allocation.reset();
    if (backend_handle == nullptr ||
        (backend != nixlDeviceMemViewBackend::UCX &&
         backend != nixlDeviceMemViewBackend::PROXY)) {
        return NIXL_ERR_INVALID_PARAM;
    }

    nixlDeviceMemView *device_wrapper = nullptr;
    if (cudaMalloc(reinterpret_cast<void **>(&device_wrapper), sizeof(*device_wrapper)) !=
        cudaSuccess) {
        return NIXL_ERR_BACKEND;
    }
    nixlDeviceMemViewAllocation result(device_wrapper);
    const nixlDeviceMemView host_wrapper{backend, backend_handle};
    if (cudaMemcpy(device_wrapper,
                   &host_wrapper,
                   sizeof(host_wrapper),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        return NIXL_ERR_BACKEND;
    }

    allocation = std::move(result);
    return NIXL_SUCCESS;
}

void
nixlDeviceMemViewAllocation::reset() noexcept {
    if (handle_ == nullptr) {
        return;
    }
    if (cudaFree(handle_) != cudaSuccess) {
        NIXL_ERROR << "Failed to free device memview wrapper";
    }
    handle_ = nullptr;
}
