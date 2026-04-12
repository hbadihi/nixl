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

// Defines the device-visible proxy context pointer. The host runtime writes
// to this symbol via cudaMemcpyToSymbol after startWorkers() and clears it on
// shutdown(). Device kernels read it through load_proxy_context().
#include "nixl_device_proxy.cuh"

__device__ ProxyDeviceContext *g_nixl_proxy_ctx = nullptr;

// Host-callable wrappers so that proxy_runtime.cpp (compiled by g++, not NVCC)
// can update g_nixl_proxy_ctx without referencing a __device__ symbol directly.
void
nixlProxyPublishContext(ProxyDeviceContextData *ctx) {
    // ProxyDeviceContext : ProxyDeviceContextData adds no data members, so the
    // pointer representation is identical and the reinterpret_cast is safe.
    ProxyDeviceContext *device_ctx = reinterpret_cast<ProxyDeviceContext *>(ctx);
    cudaMemcpyToSymbol(g_nixl_proxy_ctx, &device_ctx, sizeof(ProxyDeviceContext *));
}

void
nixlProxyClearContext() {
    ProxyDeviceContext *null_ctx = nullptr;
    cudaMemcpyToSymbol(g_nixl_proxy_ctx, &null_ctx, sizeof(ProxyDeviceContext *));
}
