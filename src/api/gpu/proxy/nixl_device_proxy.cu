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

// Defines the device-visible proxy context pointer. Device kernels read it
// through load_proxy_context().
#include "nixl_device_proxy.cuh"

__device__ ProxyDeviceContext *g_nixl_proxy_ctx = nullptr;

namespace {

__global__ void
nixlProxyStoreContextKernel(ProxyDeviceContext *ctx) {
    g_nixl_proxy_ctx = ctx;
}

cudaError_t
nixlProxyStoreContext(ProxyDeviceContext *ctx, const char *op_name) {
    nixlProxyStoreContextKernel<<<1, 1>>>(ctx);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr,
                "%s: context store kernel launch failed: code=%d msg=%s\n",
                op_name,
                static_cast<int>(err),
                cudaGetErrorString(err));
        return err;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr,
                "%s: context store kernel failed: code=%d msg=%s\n",
                op_name,
                static_cast<int>(err),
                cudaGetErrorString(err));
    }
    return err;
}

} // namespace

cudaError_t
nixlProxyPublishContext(nixlProxyDeviceContextData *ctx) {
    ProxyDeviceContext *device_ctx = reinterpret_cast<ProxyDeviceContext *>(ctx);
    return nixlProxyStoreContext(device_ctx, "nixlProxyPublishContext");
}

cudaError_t
nixlProxyClearContext() {
    return nixlProxyStoreContext(nullptr, "nixlProxyClearContext");
}
