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

#include <mutex>
#include <set>
#include <chrono>
#include <thread>

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
// Controllable stub — lets the test thread decide when each submission
// completes.  submit() assigns unique monotonic tokens; checkCompletion()
// returns NIXL_IN_PROG until markComplete() is called for a token.
// ---------------------------------------------------------------------------
class ControllableStubAdapter : public DeviceProxyBackendAdapter {
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
        std::lock_guard<std::mutex> lk(mu_);
        token = next_token_++;
        pending_.insert(token);
        return NIXL_SUCCESS;
    }

    nixl_status_t
    checkCompletion(uint64_t token) override
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (completed_.count(token)) {
            completed_.erase(token);
            return NIXL_SUCCESS;
        }
        return NIXL_IN_PROG;
    }

    size_t
    progress() override { return 0; }

    nixl_status_t
    shutdown() override { return NIXL_SUCCESS; }

    void
    markComplete(uint64_t token)
    {
        std::lock_guard<std::mutex> lk(mu_);
        pending_.erase(token);
        completed_.insert(token);
    }

    bool
    hasPending() const
    {
        std::lock_guard<std::mutex> lk(mu_);
        return !pending_.empty();
    }

private:
    mutable std::mutex mu_;
    uint64_t next_token_ = 1;
    std::set<uint64_t> pending_;
    std::set<uint64_t> completed_;
};

// ---------------------------------------------------------------------------
// Error-returning stub — submit succeeds but checkCompletion always returns
// NIXL_ERR_BACKEND, simulating a backend failure.
// ---------------------------------------------------------------------------
class ErrorStubAdapter : public DeviceProxyBackendAdapter {
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
    checkCompletion(uint64_t) override { return NIXL_ERR_BACKEND; }

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

// ---------------------------------------------------------------------------
// Completion round-trip kernels
//
// Kernels that spin on pollXferStatus require valid proxy memview handles so
// the worker's dispatch succeeds and a completion is actually published.
// The test registers a dummy memview and passes the proxy handle here.
// ---------------------------------------------------------------------------

// Enqueues a put and spins until pollXferStatus returns a final status.
__global__ void
proxyPutAndPollKernel(nixlMemViewH src_mvh, nixlMemViewH dst_mvh,
                      nixl_status_t *out_put_status,
                      nixl_status_t *out_poll_status)
{
    nixlMemViewElem src{src_mvh, 0, 0}, dst{dst_mvh, 0, 0};
    nixlGpuXferStatusH xfer_status{};
    *out_put_status = nixlPut(src, dst, /*size=*/0, /*channel_id=*/0,
                              /*flags=*/0, &xfer_status);

    nixl_status_t poll;
    do {
        poll = nixlGpuGetXferStatus(xfer_status);
    } while (poll == NIXL_IN_PROG);
    *out_poll_status = poll;
}

// Enqueues a put and immediately returns; saves xfer_status to device memory
// so the test thread can later launch a poll kernel.
__global__ void
proxyPutAsyncKernel(nixlMemViewH src_mvh, nixlMemViewH dst_mvh,
                    nixl_status_t *out_put_status,
                    nixlGpuXferStatusH *out_xfer_status)
{
    nixlMemViewElem src{src_mvh, 0, 0}, dst{dst_mvh, 0, 0};
    *out_put_status = nixlPut(src, dst, /*size=*/0, /*channel_id=*/0,
                              /*flags=*/0, out_xfer_status);
}

// Non-blocking single poll: returns current status without spinning.
__global__ void
proxyPollOnceKernel(nixlGpuXferStatusH *xfer_status,
                    nixl_status_t *out_poll_status)
{
    *out_poll_status = nixlGpuGetXferStatus(*xfer_status);
}

// ---------------------------------------------------------------------------
// Completion round-trip helpers
// ---------------------------------------------------------------------------

// Register a dummy proxy memview so the worker can resolve the proxy ID
// during dispatch.  The backend stub never dereferences the handle, so
// any non-null sentinel works.
static nixlMemViewH
registerDummyMemView(ProxyRuntime &runtime)
{
    nixlMemViewH proxy_mvh = nullptr;
    nixlMemViewH dummy_backend = reinterpret_cast<nixlMemViewH>(uintptr_t{0xBEEF});
    auto rc = runtime.registerProxyMemView(dummy_backend, &proxy_mvh);
    EXPECT_EQ(rc, NIXL_SUCCESS);
    return proxy_mvh;
}

// ---------------------------------------------------------------------------
// Completion round-trip tests
// ---------------------------------------------------------------------------

// Full round-trip: GPU enqueues -> worker dequeues -> backend completes
// (immediately via StubProxyBackendAdapter) -> worker publishes -> GPU polls
// NIXL_SUCCESS.
TEST_F(ProxyDeviceApiTest, PutCompletionRoundTrip)
{
    StubProxyBackendAdapter adapter;
    ProxyRuntime runtime;

    ASSERT_EQ(runtime.init(&adapter, /*channel_count=*/1, /*worker_count=*/1),
              NIXL_SUCCESS);
    ASSERT_EQ(runtime.startWorkers(), NIXL_SUCCESS);
    publishProxyContext(runtime);

    nixlMemViewH mvh = registerDummyMemView(runtime);

    nixl_status_t *d_put_status  = deviceAlloc<nixl_status_t>();
    nixl_status_t *d_poll_status = deviceAlloc<nixl_status_t>();

    proxyPutAndPollKernel<<<1, 1>>>(mvh, mvh, d_put_status, d_poll_status);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    EXPECT_EQ(deviceGet(d_put_status), NIXL_IN_PROG);
    EXPECT_EQ(deviceGet(d_poll_status), NIXL_SUCCESS);

    cudaFree(d_put_status);
    cudaFree(d_poll_status);
    clearProxyContext();
    ASSERT_EQ(runtime.shutdown(), NIXL_SUCCESS);
}

// Verifies that the GPU kernel stays spinning until the test thread
// explicitly marks the backend token complete.
TEST_F(ProxyDeviceApiTest, CompletionNotVisibleUntilPublished)
{
    ControllableStubAdapter adapter;
    ProxyRuntime runtime;

    ASSERT_EQ(runtime.init(&adapter, /*channel_count=*/1, /*worker_count=*/1),
              NIXL_SUCCESS);
    ASSERT_EQ(runtime.startWorkers(), NIXL_SUCCESS);
    publishProxyContext(runtime);

    nixlMemViewH mvh = registerDummyMemView(runtime);

    nixl_status_t *d_put_status  = deviceAlloc<nixl_status_t>();
    nixl_status_t *d_poll_status = deviceAlloc<nixl_status_t>();

    // Launch async — kernel will spin on pollXferStatus.
    proxyPutAndPollKernel<<<1, 1>>>(mvh, mvh, d_put_status, d_poll_status);

    // Give the worker time to pick up and submit the request.
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Kernel should still be running (spinning on completion).
    ASSERT_EQ(cudaStreamQuery(nullptr), cudaErrorNotReady);

    // Release the completion from the test thread.
    adapter.markComplete(1);

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    EXPECT_EQ(deviceGet(d_put_status), NIXL_IN_PROG);
    EXPECT_EQ(deviceGet(d_poll_status), NIXL_SUCCESS);

    cudaFree(d_put_status);
    cudaFree(d_poll_status);
    clearProxyContext();
    ASSERT_EQ(runtime.shutdown(), NIXL_SUCCESS);
}

// Enqueue 3 operations, complete them in order, and verify the collapsed-CQ
// frontier semantics: each pollXferStatus returns NIXL_SUCCESS only after its
// op_idx has been reached.
TEST_F(ProxyDeviceApiTest, MultipleSubmissionsCompletionFrontier)
{
    ControllableStubAdapter adapter;
    ProxyRuntime runtime;

    ASSERT_EQ(runtime.init(&adapter, /*channel_count=*/1, /*worker_count=*/1),
              NIXL_SUCCESS);
    ASSERT_EQ(runtime.startWorkers(), NIXL_SUCCESS);
    publishProxyContext(runtime);

    nixlMemViewH mvh = registerDummyMemView(runtime);

    constexpr int kOps = 3;
    nixl_status_t     *d_put_status[kOps];
    nixlGpuXferStatusH *d_xfer_status[kOps];

    for (int i = 0; i < kOps; i++) {
        d_put_status[i]  = deviceAlloc<nixl_status_t>();
        ASSERT_EQ(cudaMalloc(&d_xfer_status[i], sizeof(nixlGpuXferStatusH)),
                  cudaSuccess);
        ASSERT_EQ(cudaMemset(d_xfer_status[i], 0, sizeof(nixlGpuXferStatusH)),
                  cudaSuccess);
    }

    // Enqueue 3 operations sequentially (each kernel returns after enqueue).
    for (int i = 0; i < kOps; i++) {
        proxyPutAsyncKernel<<<1, 1>>>(mvh, mvh, d_put_status[i],
                                      d_xfer_status[i]);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
        EXPECT_EQ(deviceGet(d_put_status[i]), NIXL_IN_PROG);
    }

    // All three should still be in-progress.
    nixl_status_t *d_poll = deviceAlloc<nixl_status_t>();
    for (int i = 0; i < kOps; i++) {
        proxyPollOnceKernel<<<1, 1>>>(d_xfer_status[i], d_poll);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        EXPECT_EQ(deviceGet(d_poll), NIXL_IN_PROG)
            << "op " << i << " should be in-progress before any markComplete";
    }

    // Complete them one at a time and verify frontier advances.
    for (int i = 0; i < kOps; i++) {
        adapter.markComplete(static_cast<uint64_t>(i + 1));

        // Give worker time to publish.
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        // Poll this op — should now be complete.
        proxyPollOnceKernel<<<1, 1>>>(d_xfer_status[i], d_poll);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        EXPECT_EQ(deviceGet(d_poll), NIXL_SUCCESS)
            << "op " << i << " should be complete after markComplete";
    }

    cudaFree(d_poll);
    for (int i = 0; i < kOps; i++) {
        cudaFree(d_put_status[i]);
        cudaFree(d_xfer_status[i]);
    }
    clearProxyContext();
    ASSERT_EQ(runtime.shutdown(), NIXL_SUCCESS);
}

// Backend returns NIXL_ERR_BACKEND on checkCompletion; verify the GPU kernel
// receives the error status through the completion slot.
TEST_F(ProxyDeviceApiTest, CompletionPropagatesErrorStatus)
{
    ErrorStubAdapter adapter;
    ProxyRuntime runtime;

    ASSERT_EQ(runtime.init(&adapter, /*channel_count=*/1, /*worker_count=*/1),
              NIXL_SUCCESS);
    ASSERT_EQ(runtime.startWorkers(), NIXL_SUCCESS);
    publishProxyContext(runtime);

    nixlMemViewH mvh = registerDummyMemView(runtime);

    nixl_status_t *d_put_status  = deviceAlloc<nixl_status_t>();
    nixl_status_t *d_poll_status = deviceAlloc<nixl_status_t>();

    proxyPutAndPollKernel<<<1, 1>>>(mvh, mvh, d_put_status, d_poll_status);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    EXPECT_EQ(deviceGet(d_put_status), NIXL_IN_PROG);
    EXPECT_EQ(deviceGet(d_poll_status), NIXL_ERR_BACKEND);

    cudaFree(d_put_status);
    cudaFree(d_poll_status);
    clearProxyContext();
    ASSERT_EQ(runtime.shutdown(), NIXL_SUCCESS);
}
