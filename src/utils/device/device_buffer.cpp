/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "device/device_buffer.h"

#include "device/device_allocator.h"

nixl_status_t
nixlDeviceBufferAllocate(void **ptr, size_t size) noexcept {
    return nixlGetDeviceAllocator().allocDeviceMem(ptr, size);
}

nixl_status_t
nixlDeviceBufferCopyHostToDevice(void *dst, const void *src, size_t size) noexcept {
    return nixlGetDeviceAllocator().copyHostToDevice(dst, src, size);
}

nixl_status_t
nixlDeviceBufferCopyDeviceToHost(void *dst, const void *src, size_t size) noexcept {
    return nixlGetDeviceAllocator().copyDeviceToHost(dst, src, size);
}

void
nixlDeviceBufferFree(void *ptr) noexcept {
    nixlGetDeviceAllocator().freeDeviceMem(ptr);
}
