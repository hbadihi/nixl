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
#ifndef NIXL_SRC_API_GPU_PROXY_NIXL_DEVICE_PROXY_CUH
#define NIXL_SRC_API_GPU_PROXY_NIXL_DEVICE_PROXY_CUH

#include "../common/nixl_device_types.cuh"
#include "../../../core/device_proxy/proxy_protocol.h"

struct ProxyDeviceContext;

// Defined in nixl_device_proxy.cu; written by the host runtime via
// cudaMemcpyToSymbol after startWorkers() and cleared on shutdown().
extern __device__ ProxyDeviceContext *g_nixl_proxy_ctx;

__device__ inline uint64_t
proxyMemViewIdFromHandle(nixlMemViewH mvh) {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(mvh));
}

__device__ inline ProxyDeviceContext *
load_proxy_context() {
    return g_nixl_proxy_ctx;
}

struct ProxyDeviceContext : ProxyDeviceContextData {

    __device__ inline nixl_status_t
    enqueue(ProxySubmission submission, nixlGpuXferStatusH *xfer_status = nullptr) {
        (void)submission;
        (void)xfer_status;

        return NIXL_ERR_NOT_SUPPORTED;
    }

    __device__ inline nixl_status_t
    pollXferStatus(const nixlGpuXferStatusH &xfer_status) const {
        (void)xfer_status;

        return NIXL_ERR_NOT_SUPPORTED;
    }
};

#endif // NIXL_SRC_API_GPU_PROXY_NIXL_DEVICE_PROXY_CUH
