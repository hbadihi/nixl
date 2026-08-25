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
#ifndef NIXL_SRC_UTILS_DEVICE_DEVICE_ALLOCATOR_H
#define NIXL_SRC_UTILS_DEVICE_DEVICE_ALLOCATOR_H

#include <cstddef>

#include <nixl_types.h>

/**
 * Device memory-ops interface. All host-side interaction with the GPU memory
 * runtime (CUDA today, HIP later) goes through this class so that no other
 * host code needs a cuda_runtime.h include.
 */
class nixlDeviceAllocator {
    public:
        virtual ~nixlDeviceAllocator() = default;

        [[nodiscard]] virtual nixl_status_t
        allocDeviceMem(void **ptr, size_t size) noexcept = 0;

        virtual void
        freeDeviceMem(void *ptr) noexcept = 0;

        /**
         * Allocate pinned host memory that is mapped into the device address
         * space; *dev_ptr is the device-visible alias of *host_ptr.
         */
        [[nodiscard]] virtual nixl_status_t
        allocMappedHostMem(void **host_ptr, void **dev_ptr, size_t size) noexcept = 0;

        virtual void
        freeMappedHostMem(void *host_ptr) noexcept = 0;

        [[nodiscard]] virtual nixl_status_t
        copyHostToDevice(void *dst, const void *src, size_t size) noexcept = 0;

        [[nodiscard]] virtual nixl_status_t
        copyDeviceToHost(void *dst, const void *src, size_t size) noexcept = 0;

        [[nodiscard]] virtual nixl_status_t
        memsetDeviceMem(void *ptr, int value, size_t size) noexcept = 0;

        /** Block until all outstanding work on the active device completes. */
        [[nodiscard]] virtual nixl_status_t
        synchronize() noexcept = 0;

        [[nodiscard]] virtual nixl_status_t
        getActiveDevice(int &device_id) noexcept = 0;

        [[nodiscard]] virtual nixl_status_t
        setActiveDevice(int device_id) noexcept = 0;
};

/** Process-wide allocator for the platform this library was built for. */
[[nodiscard]] nixlDeviceAllocator &
nixlGetDeviceAllocator() noexcept;

#endif // NIXL_SRC_UTILS_DEVICE_DEVICE_ALLOCATOR_H
