/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "device_memview.h"

#include <utility>

#include <cuda_runtime.h>

#include "common/nixl_log.h"

namespace {

class CudaDeviceGuard {
public:
    explicit CudaDeviceGuard(int target) noexcept {
        if (cudaGetDevice(&original_) != cudaSuccess) {
            return;
        }
        if (original_ == target || cudaSetDevice(target) == cudaSuccess) {
            active_ = true;
        }
    }

    ~CudaDeviceGuard() {
        if (active_ && original_ >= 0) {
            cudaSetDevice(original_);
        }
    }

    bool
    active() const noexcept {
        return active_;
    }

private:
    int original_ = -1;
    bool active_ = false;
};

} // namespace

nixlDeviceMemViewAllocation::nixlDeviceMemViewAllocation(nixlMemViewH handle,
                                                         int cuda_device) noexcept
    : handle_(handle), cuda_device_(cuda_device) {}

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
        cuda_device_ = std::exchange(other.cuda_device_, -1);
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

    cudaPointerAttributes attributes{};
    if (cudaPointerGetAttributes(&attributes, backend_handle) != cudaSuccess ||
        attributes.type != cudaMemoryTypeDevice) {
        return NIXL_ERR_BACKEND;
    }

    CudaDeviceGuard guard(attributes.device);
    if (!guard.active()) {
        return NIXL_ERR_BACKEND;
    }

    nixlDeviceMemView *device_wrapper = nullptr;
    if (cudaMalloc(reinterpret_cast<void **>(&device_wrapper), sizeof(*device_wrapper)) !=
        cudaSuccess) {
        return NIXL_ERR_BACKEND;
    }
    nixlDeviceMemViewAllocation result(device_wrapper, attributes.device);
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

    CudaDeviceGuard guard(cuda_device_);
    if (!guard.active() || cudaFree(handle_) != cudaSuccess) {
        NIXL_ERROR << "Failed to free device memview wrapper on CUDA device " << cuda_device_;
    }
    handle_ = nullptr;
    cuda_device_ = -1;
}
