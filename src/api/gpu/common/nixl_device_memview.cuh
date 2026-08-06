/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef NIXL_SRC_API_GPU_COMMON_NIXL_DEVICE_MEMVIEW_CUH
#define NIXL_SRC_API_GPU_COMMON_NIXL_DEVICE_MEMVIEW_CUH

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <nixl_types.h>

enum class nixlDeviceMemViewBackend : uint64_t {
    UCX = 1,
    PROXY = 2,
};

// Process-local GPU Device API handle. The tag selects the backend-owned
// representation referenced by the second field. Unknown tags fail closed.
struct nixlDeviceMemView {
    nixlDeviceMemViewBackend backend;
    nixlMemViewH backend_handle;
};

static_assert(std::is_standard_layout_v<nixlDeviceMemView>);
static_assert(std::is_trivially_copyable_v<nixlDeviceMemView>);
static_assert(sizeof(nixlDeviceMemView) == 16);
static_assert(alignof(nixlDeviceMemView) == alignof(void *));
static_assert(offsetof(nixlDeviceMemView, backend) == 0);
static_assert(offsetof(nixlDeviceMemView, backend_handle) == 8);

#endif // NIXL_SRC_API_GPU_COMMON_NIXL_DEVICE_MEMVIEW_CUH
