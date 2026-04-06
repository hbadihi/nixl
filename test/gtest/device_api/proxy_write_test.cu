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

// Verifies that the proxy backend compiles, links, and that GPU kernels can
// reach the ProxyDeviceContext published by ProxyRuntime::startWorkers().
//
// nixl_device.cuh resolves to proxy/nixl_device.cuh via the proxy include
// path supplied by the build system - no backend macro is needed.

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <nixl_device.cuh>

#include "device_proxy/proxy_runtime.h"
#include "device_proxy/backend_adapter.h"
#include "common.h"

// ---------------------------------------------------------------------------
// Minimal stub backend — satisfies the pure-virtual interface without doing
// any real I/O.  Sufficient for testing the runtime lifecycle and the GPU
// device-context path.
// ---------------------------------------------------------------------------
class StubProxyBackendAdapter : public DeviceProxyBackendAdapter {
public:
    nixl_status_t
    init(uint32_t, uint32_t) override { return NIXL_SUCCESS; }

    nixl_status_t
    loadRemoteConnInfo(const std::string &, const nixl_blob_t &) override
    {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    submit(const ResolvedProxySubmission &, uint64_t &token) override
    {
        token = 0;
        return NIXL_SUCCESS;
    }

    nixl_status_t
    checkCompletion(uint64_t) override { return NIXL_SUCCESS; }

    size_t
    progress() override { return 0; }

    nixl_status_t
    shutdown() override { return NIXL_SUCCESS; }
};

// ---------------------------------------------------------------------------
// Device kernels
// ---------------------------------------------------------------------------

// Writes true if load_proxy_context() returns a non-null pointer.
__global__ void
proxyContextKernel(bool *out_has_ctx)
{
    *out_has_ctx = (load_proxy_context() != nullptr);
}

// Calls nixlPut with zero-initialised operands and records the status.
__global__ void
proxyPutKernel(nixl_status_t *out_status)
{
    nixlMemViewElem src{}, dst{};
    *out_status = nixlPut(src, dst, /*size=*/0);
}

static void
publishProxyContext(ProxyRuntime &runtime)
{
    bool *d_warmup = nullptr;
    ASSERT_EQ(cudaMalloc(&d_warmup, sizeof(bool)), cudaSuccess);
    proxyContextKernel<<<1, 1>>>(d_warmup);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaFree(d_warmup), cudaSuccess);

    ASSERT_NE(runtime.deviceContext(), nullptr);
    ASSERT_EQ(nixlProxyPublishContext(runtime.deviceContext()), cudaSuccess);
}

static void
clearProxyContext()
{
    ASSERT_EQ(nixlProxyClearContext(), cudaSuccess);
}

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------

class ProxyDeviceApiTest : public ::testing::Test {
protected:
    void
    SetUp() override
    {
        if (!gtest::hasCudaGpu()) {
            GTEST_SKIP() << "No CUDA-capable GPU, skipping proxy device API test.";
        }
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    }

    template<typename T>
    T
    deviceGet(T *d_ptr)
    {
        T val{};
        cudaMemcpy(&val, d_ptr, sizeof(T), cudaMemcpyDeviceToHost);
        return val;
    }

    template<typename T>
    T *
    deviceAlloc()
    {
        T *ptr = nullptr;
        EXPECT_EQ(cudaMalloc(&ptr, sizeof(T)), cudaSuccess);
        EXPECT_EQ(cudaMemset(ptr, 0, sizeof(T)), cudaSuccess);
        return ptr;
    }

    template<typename Predicate>
    bool
    waitForCondition(Predicate predicate,
                     std::chrono::milliseconds timeout = std::chrono::milliseconds(500))
    {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (std::chrono::steady_clock::now() < deadline) {
            if (predicate()) {
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        return predicate();
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// After startWorkers() the GPU should see a non-null proxy context.
TEST_F(ProxyDeviceApiTest, ContextPublishedAfterStartWorkers)
{
    StubProxyBackendAdapter adapter;
    ProxyRuntime runtime;

    ASSERT_EQ(runtime.init(&adapter, /*channel_count=*/1, /*worker_count=*/1),
              NIXL_SUCCESS);
    ASSERT_EQ(runtime.startWorkers(), NIXL_SUCCESS);
    publishProxyContext(runtime);

    bool *d_has_ctx = deviceAlloc<bool>();
    proxyContextKernel<<<1, 1>>>(d_has_ctx);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    EXPECT_TRUE(deviceGet(d_has_ctx));
    cudaFree(d_has_ctx);

    clearProxyContext();
    ASSERT_EQ(runtime.shutdown(), NIXL_SUCCESS);
}

// After shutdown() the GPU should no longer see a proxy context.
TEST_F(ProxyDeviceApiTest, ContextClearedAfterShutdown)
{
    StubProxyBackendAdapter adapter;
    ProxyRuntime runtime;

    ASSERT_EQ(runtime.init(&adapter, /*channel_count=*/1, /*worker_count=*/1),
              NIXL_SUCCESS);
    ASSERT_EQ(runtime.startWorkers(), NIXL_SUCCESS);
    publishProxyContext(runtime);
    ASSERT_EQ(runtime.shutdown(), NIXL_SUCCESS);
    clearProxyContext();

    bool *d_has_ctx = deviceAlloc<bool>();
    // Initialise to true so a no-op kernel would give a false pass.
    bool init_val = true;
    cudaMemcpy(d_has_ctx, &init_val, sizeof(bool), cudaMemcpyHostToDevice);

    proxyContextKernel<<<1, 1>>>(d_has_ctx);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    EXPECT_FALSE(deviceGet(d_has_ctx));
    cudaFree(d_has_ctx);
}

// nixlPut() via the proxy backend should report NIXL_IN_PROG once the
// submission is accepted into the proxy ring.
TEST_F(ProxyDeviceApiTest, PutReturnsInProgWhenEnqueued)
{
    StubProxyBackendAdapter adapter;
    ProxyRuntime runtime;

    ASSERT_EQ(runtime.init(&adapter, /*channel_count=*/1, /*worker_count=*/1),
              NIXL_SUCCESS);
    ASSERT_EQ(runtime.startWorkers(), NIXL_SUCCESS);
    publishProxyContext(runtime);

    nixl_status_t *d_status = deviceAlloc<nixl_status_t>();
    proxyPutKernel<<<1, 1>>>(d_status);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    EXPECT_EQ(deviceGet(d_status), NIXL_IN_PROG);
    cudaFree(d_status);

    clearProxyContext();
    ASSERT_EQ(runtime.shutdown(), NIXL_SUCCESS);
}
