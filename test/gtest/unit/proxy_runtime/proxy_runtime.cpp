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

#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cuda_runtime.h>
#include <mutex>
#include <string>
#include <thread>

#include "device_proxy/backend_adapter.h"
#include "device_proxy/proxy_runtime.h"

namespace gtest {
namespace proxy_runtime {

class DummyBackendMD : public nixlBackendMD {
    public:
        DummyBackendMD() : nixlBackendMD(false) {}
};

class StubBackend : public DeviceProxyBackendAdapter {
    public:
        nixl_status_t
        init(uint32_t worker_count, uint32_t channel_count) override {
            init_called_ = true;
            init_worker_count_ = worker_count;
            init_channel_count_ = channel_count;
            return init_rc_;
        }

        nixl_status_t
        loadRemoteConnInfo(const std::string &, const nixl_blob_t &) override {
            return NIXL_SUCCESS;
        }

        nixl_status_t
        submit(const PreparedProxySubmission &submission, uint64_t &request_token) override {
            {
                std::lock_guard<std::mutex> lock(submit_mutex_);
                submissions_.push_back(submission);
            }
            request_token = ++next_request_token_;
            return NIXL_SUCCESS;
        }

        nixl_status_t
        checkCompletion(uint64_t) override {
            return NIXL_SUCCESS;
        }

        size_t
        progress() override {
            ++progress_calls_;
            return 0;
        }

        nixl_status_t
        shutdown() override {
            return NIXL_SUCCESS;
        }

        bool init_called_ = false;
        uint32_t init_worker_count_ = 0;
        uint32_t init_channel_count_ = 0;
        nixl_status_t init_rc_ = NIXL_SUCCESS;
        std::atomic<uint64_t> progress_calls_{0};
        mutable std::mutex submit_mutex_;
        std::vector<PreparedProxySubmission> submissions_;
        uint64_t next_request_token_ = 0;
};

class ProxyRuntimeTest : public testing::Test {
    protected:
        void TearDown() override {
            runtime_.shutdown();
        }

        StubBackend backend_;
        ProxyRuntime runtime_;
};

TEST_F(ProxyRuntimeTest, InitCallsBackendInit) {
    ASSERT_EQ(runtime_.init(&backend_, 4, 2), NIXL_SUCCESS);
    EXPECT_TRUE(backend_.init_called_);
    EXPECT_EQ(backend_.init_worker_count_, 2u);
    EXPECT_EQ(backend_.init_channel_count_, 4u);
}

TEST_F(ProxyRuntimeTest, InitRejectsNullBackend) {
    EXPECT_EQ(runtime_.init(nullptr, 4, 2), NIXL_ERR_INVALID_PARAM);
}

TEST_F(ProxyRuntimeTest, InitRejectsZeroChannels) {
    EXPECT_EQ(runtime_.init(&backend_, 0, 2), NIXL_ERR_INVALID_PARAM);
}

TEST_F(ProxyRuntimeTest, InitRejectsZeroWorkers) {
    EXPECT_EQ(runtime_.init(&backend_, 4, 0), NIXL_ERR_INVALID_PARAM);
}

TEST_F(ProxyRuntimeTest, InitPropagatesBackendFailure) {
    backend_.init_rc_ = NIXL_ERR_BACKEND;
    EXPECT_EQ(runtime_.init(&backend_, 4, 2), NIXL_ERR_BACKEND);
}

TEST_F(ProxyRuntimeTest, InitSetsChannelCount) {
    ASSERT_EQ(runtime_.init(&backend_, 4, 2), NIXL_SUCCESS);
    EXPECT_EQ(runtime_.channelCount(), 4u);
}

TEST_F(ProxyRuntimeTest, DeviceChannelViewsPopulated) {
    ASSERT_EQ(runtime_.init(&backend_, 3, 1), NIXL_SUCCESS);
    const ProxyChannelView *views = runtime_.deviceChannelViews();
    ASSERT_NE(views, nullptr);
    for (uint32_t i = 0; i < 3; ++i) {
        EXPECT_EQ(views[i].channel_id, i);
        EXPECT_NE(views[i].work_ring, nullptr);
        EXPECT_NE(views[i].work_ring->producer_idx, nullptr);
        EXPECT_NE(views[i].work_ring->consumer_idx, nullptr);
        EXPECT_NE(views[i].completion_slot, nullptr);
        EXPECT_EQ(views[i].work_ring->depth, kDefaultProxyRingDepth);
    }
}

TEST_F(ProxyRuntimeTest, WorkRingIndicesStartAtZero) {
    ASSERT_EQ(runtime_.init(&backend_, 2, 1), NIXL_SUCCESS);
    const ProxyChannelView *views = runtime_.deviceChannelViews();
    for (uint32_t i = 0; i < 2; ++i) {
        uint32_t producer = 0;
        uint32_t consumer = 0;
        ASSERT_EQ(cudaMemcpy(&producer,
                             views[i].work_ring->producer_idx,
                             sizeof(producer),
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpy(&consumer,
                             views[i].work_ring->consumer_idx,
                             sizeof(consumer),
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        EXPECT_EQ(producer, 0u);
        EXPECT_EQ(consumer, 0u);
    }
}

TEST_F(ProxyRuntimeTest, CompletionSlotsInitialized) {
    ASSERT_EQ(runtime_.init(&backend_, 2, 1), NIXL_SUCCESS);
    const ProxyChannelView *views = runtime_.deviceChannelViews();
    for (uint32_t i = 0; i < 2; ++i) {
        CompletionSlot slot{};
        ASSERT_EQ(cudaMemcpy(&slot,
                             views[i].completion_slot,
                             sizeof(CompletionSlot),
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        EXPECT_EQ(slot.completed_idx, 0u);
        EXPECT_EQ(slot.next_status, NIXL_IN_PROG);
    }
}

TEST_F(ProxyRuntimeTest, WorkerCountClampedToChannels) {
    ASSERT_EQ(runtime_.init(&backend_, 2, 8), NIXL_SUCCESS);
    EXPECT_EQ(runtime_.channelCount(), 2u);
    EXPECT_EQ(backend_.init_worker_count_, 2u);
    EXPECT_EQ(backend_.init_channel_count_, 2u);
}

TEST_F(ProxyRuntimeTest, DeviceContextPopulated) {
    ASSERT_EQ(runtime_.init(&backend_, 3, 1), NIXL_SUCCESS);
    auto *ctx = runtime_.deviceContext();
    ASSERT_NE(ctx, nullptr);
    EXPECT_EQ(ctx->num_channels, 3u);
    EXPECT_EQ(ctx->channels, runtime_.deviceChannelViews());
    EXPECT_NE(ctx->shutdown_word, nullptr);
}

TEST_F(ProxyRuntimeTest, DeviceContextNullAfterShutdown) {
    ASSERT_EQ(runtime_.init(&backend_, 2, 1), NIXL_SUCCESS);
    ASSERT_NE(runtime_.deviceContext(), nullptr);
    ASSERT_EQ(runtime_.shutdown(), NIXL_SUCCESS);
    EXPECT_EQ(runtime_.deviceContext(), nullptr);
}

TEST_F(ProxyRuntimeTest, StartWorkersAndShutdown) {
    ASSERT_EQ(runtime_.init(&backend_, 2, 2), NIXL_SUCCESS);
    ASSERT_EQ(runtime_.startWorkers(), NIXL_SUCCESS);

    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    ASSERT_EQ(runtime_.shutdown(), NIXL_SUCCESS);
}

TEST_F(ProxyRuntimeTest, RestartWorkersWithoutShutdown) {
    ASSERT_EQ(runtime_.init(&backend_, 2, 2), NIXL_SUCCESS);
    ASSERT_EQ(runtime_.startWorkers(), NIXL_SUCCESS);

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    ASSERT_EQ(runtime_.startWorkers(), NIXL_SUCCESS);

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    ASSERT_EQ(runtime_.shutdown(), NIXL_SUCCESS);
}

TEST_F(ProxyRuntimeTest, ShutdownWithoutStartIsHarmless) {
    ASSERT_EQ(runtime_.init(&backend_, 2, 1), NIXL_SUCCESS);
    EXPECT_EQ(runtime_.shutdown(), NIXL_SUCCESS);
}

TEST_F(ProxyRuntimeTest, ShutdownBeforeInitIsHarmless) {
    EXPECT_EQ(runtime_.shutdown(), NIXL_SUCCESS);
}

TEST_F(ProxyRuntimeTest, DoubleShutdownIsHarmless) {
    ASSERT_EQ(runtime_.init(&backend_, 2, 1), NIXL_SUCCESS);
    ASSERT_EQ(runtime_.startWorkers(), NIXL_SUCCESS);
    EXPECT_EQ(runtime_.shutdown(), NIXL_SUCCESS);
    EXPECT_EQ(runtime_.shutdown(), NIXL_SUCCESS);
}

TEST_F(ProxyRuntimeTest, InitAfterShutdownWorks) {
    ASSERT_EQ(runtime_.init(&backend_, 2, 1), NIXL_SUCCESS);
    ASSERT_EQ(runtime_.startWorkers(), NIXL_SUCCESS);
    ASSERT_EQ(runtime_.shutdown(), NIXL_SUCCESS);

    ASSERT_EQ(runtime_.init(&backend_, 4, 2), NIXL_SUCCESS);
    EXPECT_EQ(runtime_.channelCount(), 4u);
    ASSERT_EQ(runtime_.startWorkers(), NIXL_SUCCESS);
    EXPECT_EQ(runtime_.shutdown(), NIXL_SUCCESS);
}

TEST_F(ProxyRuntimeTest, SingleChannelSingleWorker) {
    ASSERT_EQ(runtime_.init(&backend_, 1, 1), NIXL_SUCCESS);
    ASSERT_EQ(runtime_.startWorkers(), NIXL_SUCCESS);

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    ASSERT_EQ(runtime_.shutdown(), NIXL_SUCCESS);
    EXPECT_EQ(runtime_.channelCount(), 0u);
}

TEST_F(ProxyRuntimeTest, ManyChannelsManyWorkers) {
    ASSERT_EQ(runtime_.init(&backend_, 16, 4), NIXL_SUCCESS);
    ASSERT_EQ(runtime_.startWorkers(), NIXL_SUCCESS);

    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    ASSERT_EQ(runtime_.shutdown(), NIXL_SUCCESS);
}


} // namespace proxy_runtime
} // namespace gtest
