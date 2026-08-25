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
#include "device/device_allocator.h"

#include <cuda_runtime.h>

#include "common/nixl_log.h"

namespace {

class nixlCudaDeviceAllocator final : public nixlDeviceAllocator {
    public:
        nixl_status_t
        allocDeviceMem(void **ptr, size_t size) noexcept override {
            if (ptr == nullptr || size == 0) {
                return NIXL_ERR_INVALID_PARAM;
            }
            *ptr = nullptr;
            if (cudaMalloc(ptr, size) != cudaSuccess) {
                *ptr = nullptr;
                return NIXL_ERR_BACKEND;
            }
            return NIXL_SUCCESS;
        }

        void
        freeDeviceMem(void *ptr) noexcept override {
            if (ptr == nullptr) {
                return;
            }
            if (cudaFree(ptr) != cudaSuccess) {
                NIXL_ERROR << "Failed to free device buffer";
            }
        }

        nixl_status_t
        allocMappedHostMem(void **host_ptr, void **dev_ptr, size_t size) noexcept override {
            if (host_ptr == nullptr || dev_ptr == nullptr || size == 0) {
                return NIXL_ERR_INVALID_PARAM;
            }
            *host_ptr = nullptr;
            *dev_ptr = nullptr;
            // cudaHostAllocMapped guarantees cudaHostGetDevicePointer works (vs. relying on UVA).
            if (cudaHostAlloc(host_ptr, size, cudaHostAllocMapped) != cudaSuccess) {
                *host_ptr = nullptr;
                return NIXL_ERR_BACKEND;
            }
            if (cudaHostGetDevicePointer(dev_ptr, *host_ptr, 0) != cudaSuccess) {
                cudaFreeHost(*host_ptr);
                *host_ptr = nullptr;
                *dev_ptr = nullptr;
                return NIXL_ERR_BACKEND;
            }
            return NIXL_SUCCESS;
        }

        void
        freeMappedHostMem(void *host_ptr) noexcept override {
            if (host_ptr == nullptr) {
                return;
            }
            if (cudaFreeHost(host_ptr) != cudaSuccess) {
                NIXL_ERROR << "Failed to free host-mapped buffer";
            }
        }

        nixl_status_t
        copyHostToDevice(void *dst, const void *src, size_t size) noexcept override {
            if (dst == nullptr || src == nullptr || size == 0) {
                return NIXL_ERR_INVALID_PARAM;
            }
            if (cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice) != cudaSuccess) {
                return NIXL_ERR_BACKEND;
            }
            return NIXL_SUCCESS;
        }

        nixl_status_t
        copyDeviceToHost(void *dst, const void *src, size_t size) noexcept override {
            if (dst == nullptr || src == nullptr || size == 0) {
                return NIXL_ERR_INVALID_PARAM;
            }
            if (cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost) != cudaSuccess) {
                return NIXL_ERR_BACKEND;
            }
            return NIXL_SUCCESS;
        }

        nixl_status_t
        memsetDeviceMem(void *ptr, int value, size_t size) noexcept override {
            if (ptr == nullptr || size == 0) {
                return NIXL_ERR_INVALID_PARAM;
            }
            if (cudaMemset(ptr, value, size) != cudaSuccess) {
                return NIXL_ERR_BACKEND;
            }
            return NIXL_SUCCESS;
        }

        nixl_status_t
        synchronize() noexcept override {
            return cudaDeviceSynchronize() == cudaSuccess ? NIXL_SUCCESS : NIXL_ERR_BACKEND;
        }

        nixl_status_t
        getActiveDevice(int &device_id) noexcept override {
            return cudaGetDevice(&device_id) == cudaSuccess ? NIXL_SUCCESS : NIXL_ERR_BACKEND;
        }

        nixl_status_t
        setActiveDevice(int device_id) noexcept override {
            return cudaSetDevice(device_id) == cudaSuccess ? NIXL_SUCCESS : NIXL_ERR_BACKEND;
        }
};

} // namespace

nixlDeviceAllocator &
nixlGetDeviceAllocator() noexcept {
    static nixlCudaDeviceAllocator allocator;
    return allocator;
}
