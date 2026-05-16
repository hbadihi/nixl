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

#include <cstdio>

__device__ __constant__ ProxyDeviceContext *g_nixl_proxy_ctx = nullptr;

__global__ void
nixlProxyDeviceLinkKernel() {}

namespace {
cudaError_t
ensureProxyDeviceModuleLoaded() {
    cudaFuncAttributes attributes{};
    return cudaFuncGetAttributes(&attributes, nixlProxyDeviceLinkKernel);
}
} // namespace

__host__ cudaError_t
nixlProxyPublishContext(nixlProxyDeviceContextData *ctx) {
    cudaError_t err = ensureProxyDeviceModuleLoaded();
    if (err != cudaSuccess) {
        std::fprintf(stderr,
                     "nixlProxyPublishContext: cudaFuncGetAttributes failed: code=%d msg=%s\n",
                     static_cast<int>(err),
                     cudaGetErrorString(err));
        return err;
    }

    ProxyDeviceContext *device_ctx = reinterpret_cast<ProxyDeviceContext *>(ctx);
    err = cudaMemcpyToSymbol(g_nixl_proxy_ctx, &device_ctx, sizeof(ProxyDeviceContext *));
    if (err != cudaSuccess) {
        std::fprintf(stderr,
                     "nixlProxyPublishContext: cudaMemcpyToSymbol failed: code=%d msg=%s\n",
                     static_cast<int>(err),
                     cudaGetErrorString(err));
    }
    return err;
}

__host__ cudaError_t
nixlProxyClearContext() {
    cudaError_t err = ensureProxyDeviceModuleLoaded();
    if (err != cudaSuccess) {
        std::fprintf(stderr,
                     "nixlProxyClearContext: cudaFuncGetAttributes failed: code=%d msg=%s\n",
                     static_cast<int>(err),
                     cudaGetErrorString(err));
        return err;
    }

    ProxyDeviceContext *null_ctx = nullptr;
    err = cudaMemcpyToSymbol(g_nixl_proxy_ctx, &null_ctx, sizeof(ProxyDeviceContext *));
    if (err != cudaSuccess) {
        std::fprintf(stderr,
                     "nixlProxyClearContext: cudaMemcpyToSymbol failed: code=%d msg=%s\n",
                     static_cast<int>(err),
                     cudaGetErrorString(err));
    }
    return err;
}
