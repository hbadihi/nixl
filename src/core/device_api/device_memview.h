/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef NIXL_SRC_CORE_DEVICE_API_DEVICE_MEMVIEW_H
#define NIXL_SRC_CORE_DEVICE_API_DEVICE_MEMVIEW_H

#include "../../api/gpu/common/nixl_device_memview.cuh"

class nixlDeviceMemViewAllocation {
public:
    nixlDeviceMemViewAllocation() = default;
    ~nixlDeviceMemViewAllocation();

    nixlDeviceMemViewAllocation(nixlDeviceMemViewAllocation &&other) noexcept;
    nixlDeviceMemViewAllocation &
    operator=(nixlDeviceMemViewAllocation &&other) noexcept;

    nixlDeviceMemViewAllocation(const nixlDeviceMemViewAllocation &) = delete;
    nixlDeviceMemViewAllocation &
    operator=(const nixlDeviceMemViewAllocation &) = delete;

    static nixl_status_t
    create(nixlDeviceMemViewBackend backend,
           nixlMemViewH backend_handle,
           nixlDeviceMemViewAllocation &allocation) noexcept;

    nixlMemViewH
    get() const noexcept {
        return handle_;
    }

    void
    reset() noexcept;

private:
    explicit nixlDeviceMemViewAllocation(nixlMemViewH handle) noexcept : handle_(handle) {}

    nixlMemViewH handle_ = nullptr;
};

#endif // NIXL_SRC_CORE_DEVICE_API_DEVICE_MEMVIEW_H
