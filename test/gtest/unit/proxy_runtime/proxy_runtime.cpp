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
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "device_proxy/backend_adapter.h"
#include "device_proxy/proxy_runtime.h"

namespace gtest {
namespace proxy_runtime {

class DummyBackendMD : public nixlBackendMD {
    public:
        DummyBackendMD() : nixlBackendMD(false) {}
};

class StubBackend : public nixlDeviceProxyBackendAdapter {
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
        submit(const nixlBackendProxySubmission &submission, uint64_t &request_token) override {
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

        nixl_status_t
        progress() override {
            ++progress_calls_;
            return NIXL_SUCCESS;
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
        std::vector<nixlBackendProxySubmission> submissions_;
        uint64_t next_request_token_ = 0;
};

class ProxyRuntimeTest : public testing::Test {
    protected:
        nixl_status_t
        initRuntime(uint32_t channel_count,
                    uint32_t worker_count,
                    nixl_status_t init_rc = NIXL_SUCCESS) {
            auto backend = std::make_unique<StubBackend>();
            backend_ = backend.get();
            backend_->init_rc_ = init_rc;
            return runtime_.init(std::move(backend), channel_count, worker_count);
        }

        void TearDown() override {
            runtime_.shutdown();
        }

        StubBackend *backend_ = nullptr;
        nixlProxyRuntime runtime_;
};

static nixlProxyWorkRing
copyDeviceWorkRing(const nixlProxyChannelView &view) {
    nixlProxyWorkRing ring{};
    EXPECT_EQ(cudaMemcpy(&ring, view.work_ring, sizeof(ring), cudaMemcpyDeviceToHost),
              cudaSuccess);
    return ring;
}

// Resolve the pinned-host alias of a device-mapped pointer (ring records, completion
// slot, ...). The device side hands out a device alias; tests poke the host alias.
template <class T>
static T *
hostAliasOf(T *device_alias) {
    cudaPointerAttributes attrs{};
    EXPECT_EQ(cudaPointerGetAttributes(&attrs, device_alias), cudaSuccess);
    EXPECT_NE(attrs.hostPointer, nullptr);
    return static_cast<T *>(attrs.hostPointer);
}

// Spin (bounded) until the channel's completion slot reports `op_idx` completed.
static bool
waitForCompletedIdx(nixlProxyCompletionSlot *slot_host, uint64_t op_idx) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while (std::chrono::steady_clock::now() < deadline) {
        if (__atomic_load_n(&slot_host->completed_idx, __ATOMIC_ACQUIRE) >= op_idx) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return false;
}

TEST_F(ProxyRuntimeTest, InitCallsBackendInit) {
    ASSERT_EQ(initRuntime(4, 2), NIXL_SUCCESS);
    EXPECT_TRUE(backend_->init_called_);
    EXPECT_EQ(backend_->init_worker_count_, 2u);
    EXPECT_EQ(backend_->init_channel_count_, 4u);
}

TEST_F(ProxyRuntimeTest, InitRejectsNullBackend) {
    EXPECT_EQ(runtime_.init(nullptr, 4, 2), NIXL_ERR_INVALID_PARAM);
}

TEST_F(ProxyRuntimeTest, InitRejectsZeroChannels) {
    EXPECT_EQ(initRuntime(0, 2), NIXL_ERR_INVALID_PARAM);
}

TEST_F(ProxyRuntimeTest, InitRejectsZeroWorkers) {
    EXPECT_EQ(initRuntime(4, 0), NIXL_ERR_INVALID_PARAM);
}

TEST_F(ProxyRuntimeTest, InitPropagatesBackendFailure) {
    EXPECT_EQ(initRuntime(4, 2, NIXL_ERR_BACKEND), NIXL_ERR_BACKEND);
}

TEST_F(ProxyRuntimeTest, InitSetsChannelCount) {
    ASSERT_EQ(initRuntime(4, 2), NIXL_SUCCESS);
    EXPECT_EQ(runtime_.channelCount(), 4u);
}

TEST_F(ProxyRuntimeTest, RuntimeCreationDoesNotIncrementActivityCounter) {
    ASSERT_EQ(initRuntime(2, 1), NIXL_SUCCESS);
    EXPECT_EQ(runtime_.submittedWorkCount(), 0u);
    ASSERT_EQ(runtime_.startWorkers(), NIXL_SUCCESS);

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    EXPECT_EQ(runtime_.submittedWorkCount(), 0u);
    ASSERT_EQ(runtime_.shutdown(), NIXL_SUCCESS);
}

TEST_F(ProxyRuntimeTest, DeviceChannelViewsPopulated) {
    ASSERT_EQ(initRuntime(3, 1), NIXL_SUCCESS);
    const nixlProxyChannelView *views = runtime_.deviceChannelViews();
    ASSERT_NE(views, nullptr);
    for (uint32_t i = 0; i < 3; ++i) {
        EXPECT_EQ(views[i].channel_id, i);
        EXPECT_NE(views[i].work_ring, nullptr);
        const nixlProxyWorkRing ring = copyDeviceWorkRing(views[i]);
        EXPECT_NE(ring.records, nullptr);
        EXPECT_NE(ring.producer_idx, nullptr);
        EXPECT_NE(ring.consumer_idx, nullptr);
        EXPECT_NE(views[i].completion_slot, nullptr);
        EXPECT_EQ(ring.depth, kDefaultProxyRingDepth);
    }
}

TEST_F(ProxyRuntimeTest, WorkRingIndicesStartAtZero) {
    ASSERT_EQ(initRuntime(2, 1), NIXL_SUCCESS);
    const nixlProxyChannelView *views = runtime_.deviceChannelViews();
    for (uint32_t i = 0; i < 2; ++i) {
        const nixlProxyWorkRing ring = copyDeviceWorkRing(views[i]);
        uint64_t producer = 0;
        uint64_t consumer = 0;
        ASSERT_EQ(cudaMemcpy(&producer,
                             ring.producer_idx,
                             sizeof(producer),
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpy(&consumer,
                             ring.consumer_idx,
                             sizeof(consumer),
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        EXPECT_EQ(producer, 0u);
        EXPECT_EQ(consumer, 0u);
    }
}

TEST_F(ProxyRuntimeTest, CompletionSlotsInitialized) {
    ASSERT_EQ(initRuntime(2, 1), NIXL_SUCCESS);
    const nixlProxyChannelView *views = runtime_.deviceChannelViews();
    for (uint32_t i = 0; i < 2; ++i) {
        nixlProxyCompletionSlot slot{};
        ASSERT_EQ(cudaMemcpy(&slot,
                             views[i].completion_slot,
                             sizeof(nixlProxyCompletionSlot),
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        EXPECT_EQ(slot.completed_idx, 0u);
        EXPECT_EQ(slot.next_status, NIXL_IN_PROG);
    }
}

TEST_F(ProxyRuntimeTest, WorkerCountClampedToChannels) {
    ASSERT_EQ(initRuntime(2, 8), NIXL_SUCCESS);
    EXPECT_EQ(runtime_.channelCount(), 2u);
    EXPECT_EQ(backend_->init_worker_count_, 2u);
    EXPECT_EQ(backend_->init_channel_count_, 2u);
}

TEST_F(ProxyRuntimeTest, DeviceContextPopulated) {
    ASSERT_EQ(initRuntime(3, 1), NIXL_SUCCESS);
    auto *device_ctx = runtime_.deviceContext();
    ASSERT_NE(device_ctx, nullptr);
    nixlProxyDeviceContextData ctx{};
    ASSERT_EQ(cudaMemcpy(&ctx, device_ctx, sizeof(ctx), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(ctx.num_channels, 3u);
    EXPECT_NE(ctx.channels, nullptr);
    EXPECT_NE(ctx.shutdown_word, nullptr);
}

TEST_F(ProxyRuntimeTest, DeviceContextNullAfterShutdown) {
    ASSERT_EQ(initRuntime(2, 1), NIXL_SUCCESS);
    ASSERT_NE(runtime_.deviceContext(), nullptr);
    ASSERT_EQ(runtime_.shutdown(), NIXL_SUCCESS);
    EXPECT_EQ(runtime_.deviceContext(), nullptr);
}

TEST_F(ProxyRuntimeTest, StartWorkersAndShutdown) {
    ASSERT_EQ(initRuntime(2, 2), NIXL_SUCCESS);
    ASSERT_EQ(runtime_.startWorkers(), NIXL_SUCCESS);

    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    ASSERT_EQ(runtime_.shutdown(), NIXL_SUCCESS);
}

TEST_F(ProxyRuntimeTest, RepeatedStartWorkersIsRejected) {
    ASSERT_EQ(initRuntime(2, 2), NIXL_SUCCESS);
    ASSERT_EQ(runtime_.startWorkers(), NIXL_SUCCESS);

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    EXPECT_EQ(runtime_.startWorkers(), NIXL_ERR_INVALID_PARAM);

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    ASSERT_EQ(runtime_.shutdown(), NIXL_SUCCESS);
}

TEST_F(ProxyRuntimeTest, ShutdownWithoutStartIsHarmless) {
    ASSERT_EQ(initRuntime(2, 1), NIXL_SUCCESS);
    EXPECT_EQ(runtime_.shutdown(), NIXL_SUCCESS);
}

TEST_F(ProxyRuntimeTest, ShutdownBeforeInitIsHarmless) {
    EXPECT_EQ(runtime_.shutdown(), NIXL_SUCCESS);
}

TEST_F(ProxyRuntimeTest, DoubleShutdownIsHarmless) {
    ASSERT_EQ(initRuntime(2, 1), NIXL_SUCCESS);
    ASSERT_EQ(runtime_.startWorkers(), NIXL_SUCCESS);
    EXPECT_EQ(runtime_.shutdown(), NIXL_SUCCESS);
    EXPECT_EQ(runtime_.shutdown(), NIXL_SUCCESS);
}

TEST_F(ProxyRuntimeTest, InitAfterShutdownWorks) {
    ASSERT_EQ(initRuntime(2, 1), NIXL_SUCCESS);
    ASSERT_EQ(runtime_.startWorkers(), NIXL_SUCCESS);
    ASSERT_EQ(runtime_.shutdown(), NIXL_SUCCESS);

    ASSERT_EQ(initRuntime(4, 2), NIXL_SUCCESS);
    EXPECT_EQ(runtime_.channelCount(), 4u);
    ASSERT_EQ(runtime_.startWorkers(), NIXL_SUCCESS);
    EXPECT_EQ(runtime_.shutdown(), NIXL_SUCCESS);
}

TEST_F(ProxyRuntimeTest, SingleChannelSingleWorker) {
    ASSERT_EQ(initRuntime(1, 1), NIXL_SUCCESS);
    ASSERT_EQ(runtime_.startWorkers(), NIXL_SUCCESS);

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    ASSERT_EQ(runtime_.shutdown(), NIXL_SUCCESS);
    EXPECT_EQ(runtime_.channelCount(), 0u);
}

TEST_F(ProxyRuntimeTest, ManyChannelsManyWorkers) {
    ASSERT_EQ(initRuntime(16, 4), NIXL_SUCCESS);
    ASSERT_EQ(runtime_.startWorkers(), NIXL_SUCCESS);

    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    ASSERT_EQ(runtime_.shutdown(), NIXL_SUCCESS);
}

TEST_F(ProxyRuntimeTest, PrepMemViewProducesReadyEntries) {
    DummyBackendMD local_md;
    DummyBackendMD remote_md;
    const auto local_backend = reinterpret_cast<nixlMemViewH>(uintptr_t{0x10});
    const auto remote_backend = reinterpret_cast<nixlMemViewH>(uintptr_t{0x20});

    nixl_meta_dlist_t local_dlist(DRAM_SEG);
    local_dlist.addDesc(nixlMetaDesc(0x1000, 64, 0, &local_md));

    nixl_remote_meta_dlist_t remote_dlist(DRAM_SEG);
    nixlRemoteMetaDesc remote_desc("peer");
    remote_desc.addr = 0x2000;
    remote_desc.len = 64;
    remote_desc.devId = 0;
    remote_desc.metadataP = &remote_md;
    remote_dlist.addDesc(remote_desc);

    nixlMemViewH src_proxy = nullptr;
    nixlMemViewH dst_proxy = nullptr;
    ASSERT_EQ(runtime_.prepMemView(local_backend, local_dlist, &src_proxy),
              NIXL_SUCCESS);
    ASSERT_EQ(runtime_.prepMemView(remote_backend, remote_dlist, &dst_proxy),
              NIXL_SUCCESS);

    nixlMemViewH resolved = nullptr;
    EXPECT_TRUE(runtime_.resolveProxyMemView(src_proxy, resolved));
    EXPECT_EQ(resolved, local_backend);
    EXPECT_TRUE(runtime_.resolveProxyMemView(dst_proxy, resolved));
    EXPECT_EQ(resolved, remote_backend);

    nixlProxySubmission submission{};
    submission.opcode = nixl_proxy_opcode_t::PUT;
    submission.src_proxy_memview_id = reinterpret_cast<uint64_t>(src_proxy);
    submission.src_offset = 4;
    submission.dst_proxy_memview_id = reinterpret_cast<uint64_t>(dst_proxy);
    submission.dst_offset = 8;
    submission.size = 32;

    nixlBackendProxySubmission prepared_submission;
    ASSERT_EQ(runtime_.memviewRegistry().prepareSubmission(submission, prepared_submission),
              NIXL_SUCCESS);
    EXPECT_EQ(prepared_submission.local.desc.addr, 0x1004u);
    EXPECT_EQ(prepared_submission.local.desc.len, 32u);
    EXPECT_EQ(prepared_submission.local.desc.metadataP, &local_md);
    EXPECT_EQ(prepared_submission.remote.desc.addr, 0x2008u);
    EXPECT_EQ(prepared_submission.remote.desc.len, 32u);
    EXPECT_EQ(prepared_submission.remote.desc.metadataP, &remote_md);
}

TEST_F(ProxyRuntimeTest, PrepMemViewRejectsNullOutput) {
    DummyBackendMD local_md;
    nixl_meta_dlist_t local_dlist(DRAM_SEG);
    local_dlist.addDesc(nixlMetaDesc(0x1000, 64, 0, &local_md));

    EXPECT_EQ(runtime_.prepMemView(local_dlist, nullptr),
              NIXL_ERR_INVALID_PARAM);
}

TEST_F(ProxyRuntimeTest, WorkerSubmitsPreparedTransportDescriptors) {
    DummyBackendMD local_md;
    DummyBackendMD remote_md;

    ASSERT_EQ(initRuntime(1, 1), NIXL_SUCCESS);

    nixlMemViewH src_proxy = nullptr;
    nixlMemViewH dst_proxy = nullptr;
    ASSERT_EQ(runtime_.registerProxyMemView(reinterpret_cast<nixlMemViewH>(uintptr_t{0x10}),
                                           &src_proxy),
              NIXL_SUCCESS);
    ASSERT_EQ(runtime_.registerProxyMemView(reinterpret_cast<nixlMemViewH>(uintptr_t{0x20}),
                                           &dst_proxy),
              NIXL_SUCCESS);

    nixl_meta_dlist_t local_dlist(DRAM_SEG);
    local_dlist.addDesc(nixlMetaDesc(0x1000, 64, 0, &local_md));
    ASSERT_EQ(runtime_.storeMetadata(src_proxy, local_dlist), NIXL_SUCCESS);

    nixl_remote_meta_dlist_t remote_dlist(DRAM_SEG);
    nixlRemoteMetaDesc remote_desc("peer");
    remote_desc.addr = 0x2000;
    remote_desc.len = 64;
    remote_desc.devId = 0;
    remote_desc.metadataP = &remote_md;
    remote_dlist.addDesc(remote_desc);
    ASSERT_EQ(runtime_.storeMetadata(dst_proxy, remote_dlist), NIXL_SUCCESS);

    ASSERT_EQ(runtime_.startWorkers(), NIXL_SUCCESS);

    nixlProxySubmission submission{};
    submission.op_idx = 11;
    submission.opcode = nixl_proxy_opcode_t::PUT;
    submission.channel_id = 0;
    submission.src_proxy_memview_id = reinterpret_cast<uint64_t>(src_proxy);
    submission.src_offset = 4;
    submission.dst_proxy_memview_id = reinterpret_cast<uint64_t>(dst_proxy);
    submission.dst_offset = 8;
    submission.size = 32;

    const nixlProxyWorkRing ring = copyDeviceWorkRing(runtime_.deviceChannelViews()[0]);
    auto *records = hostAliasOf(ring.records);
    ASSERT_NE(records, nullptr);
    submission.op_idx = 0;
    records[0] = submission;
    __atomic_store_n(&records[0].op_idx, uint64_t{11}, __ATOMIC_RELEASE);

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(250);
    while (std::chrono::steady_clock::now() < deadline) {
        {
            std::lock_guard<std::mutex> lock(backend_->submit_mutex_);
            if (!backend_->submissions_.empty()) {
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    std::vector<nixlBackendProxySubmission> submissions;
    {
        std::lock_guard<std::mutex> lock(backend_->submit_mutex_);
        submissions = backend_->submissions_;
    }

    EXPECT_EQ(runtime_.submittedWorkCount(), 1u);
    runtime_.resetSubmittedWorkCount();
    EXPECT_EQ(runtime_.submittedWorkCount(), 0u);

    ASSERT_EQ(runtime_.shutdown(), NIXL_SUCCESS);

    ASSERT_EQ(submissions.size(), 1u);
    const auto &prepared = submissions.front();
    EXPECT_EQ(prepared.op_idx, 11u);
    EXPECT_EQ(prepared.channel_id, 0u);
    EXPECT_EQ(prepared.local.mem_type, DRAM_SEG);
    EXPECT_EQ(prepared.local.desc.addr, 0x1004u);
    EXPECT_EQ(prepared.local.desc.len, 32u);
    EXPECT_EQ(prepared.local.desc.metadataP, &local_md);
    EXPECT_EQ(prepared.remote.mem_type, DRAM_SEG);
    EXPECT_EQ(prepared.remote.desc.addr, 0x2008u);
    EXPECT_EQ(prepared.remote.desc.len, 32u);
    EXPECT_EQ(prepared.remote.desc.metadataP, &remote_md);
}

TEST_F(ProxyRuntimeTest, WorkerSubmitsPreparedAtomicAddDescriptor) {
    DummyBackendMD remote_md;

    ASSERT_EQ(initRuntime(1, 1), NIXL_SUCCESS);

    nixlMemViewH dst_proxy = nullptr;
    ASSERT_EQ(runtime_.registerProxyMemView(reinterpret_cast<nixlMemViewH>(uintptr_t{0x20}),
                                           &dst_proxy),
              NIXL_SUCCESS);

    nixl_remote_meta_dlist_t remote_dlist(DRAM_SEG);
    nixlRemoteMetaDesc remote_desc("peer");
    remote_desc.addr = 0x2000;
    remote_desc.len = 64;
    remote_desc.devId = 0;
    remote_desc.metadataP = &remote_md;
    remote_dlist.addDesc(remote_desc);
    ASSERT_EQ(runtime_.storeMetadata(dst_proxy, remote_dlist), NIXL_SUCCESS);

    ASSERT_EQ(runtime_.startWorkers(), NIXL_SUCCESS);

    nixlProxySubmission submission{};
    submission.op_idx = 11;
    submission.opcode = nixl_proxy_opcode_t::ATOMIC_ADD;
    submission.channel_id = 0;
    submission.dst_proxy_memview_id = reinterpret_cast<uint64_t>(dst_proxy);
    submission.dst_offset = 8;
    submission.size = sizeof(uint64_t);
    submission.value = 42;

    const nixlProxyWorkRing ring = copyDeviceWorkRing(runtime_.deviceChannelViews()[0]);
    auto *records = hostAliasOf(ring.records);
    ASSERT_NE(records, nullptr);
    submission.op_idx = 0;
    records[0] = submission;
    __atomic_store_n(&records[0].op_idx, uint64_t{11}, __ATOMIC_RELEASE);

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(250);
    while (std::chrono::steady_clock::now() < deadline) {
        {
            std::lock_guard<std::mutex> lock(backend_->submit_mutex_);
            if (!backend_->submissions_.empty()) {
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    std::vector<nixlBackendProxySubmission> submissions;
    {
        std::lock_guard<std::mutex> lock(backend_->submit_mutex_);
        submissions = backend_->submissions_;
    }

    EXPECT_EQ(runtime_.submittedWorkCount(), 1u);

    ASSERT_EQ(runtime_.shutdown(), NIXL_SUCCESS);

    ASSERT_EQ(submissions.size(), 1u);
    const auto &prepared = submissions.front();
    EXPECT_EQ(prepared.op_idx, 11u);
    EXPECT_EQ(prepared.opcode, nixl_proxy_opcode_t::ATOMIC_ADD);
    EXPECT_EQ(prepared.channel_id, 0u);
    EXPECT_EQ(prepared.remote.mem_type, DRAM_SEG);
    EXPECT_EQ(prepared.remote.desc.addr, 0x2008u);
    EXPECT_EQ(prepared.remote.desc.len, sizeof(uint64_t));
    EXPECT_EQ(prepared.remote.desc.metadataP, &remote_md);
    EXPECT_EQ(prepared.value, 42u);
}

// Build a remote dlist whose element positions carry the given agents (empty string ->
// the null-agent "absent" slot). Element position i is rank i, i.e. proxy band i.
static nixl_remote_meta_dlist_t
makeRemoteBandDlist(const std::vector<std::string> &agents, nixlBackendMD *md) {
    nixl_remote_meta_dlist_t dlist(DRAM_SEG);
    for (const auto &agent : agents) {
        if (agent.empty()) {
            dlist.addDesc(nixlRemoteMetaDesc(nixl_null_agent));
        } else {
            nixlRemoteMetaDesc desc(agent);
            desc.addr = 0x4000;
            desc.len = 64;
            desc.devId = 0;
            desc.metadataP = md;
            dlist.addDesc(desc);
        }
    }
    return dlist;
}

// Inject a record into ring slot `slot_idx`: zero op_idx, copy, then release-store the
// real op_idx (the GPU->CPU signal the worker acquire-polls).
static void
publishRecord(nixlProxySubmission *records, uint32_t slot_idx, nixlProxySubmission rec,
              uint64_t op_idx) {
    rec.op_idx = 0;
    records[slot_idx] = rec;
    __atomic_store_n(&records[slot_idx].op_idx, op_idx, __ATOMIC_RELEASE);
}

static nixlProxySubmission
badPut(uint32_t channel) {  // unregistered dst memview -> prepareSubmission fails -> latch
    nixlProxySubmission s{};
    s.opcode = nixl_proxy_opcode_t::PUT;
    s.channel_id = channel;
    s.dst_proxy_memview_id = 9999;
    s.size = 32;
    return s;
}

TEST_F(ProxyRuntimeTest, InitRejectsChannelCountNotMultipleOfChannelsPerRank) {
    auto backend = std::make_unique<StubBackend>();
    backend_ = backend.get();
    // 5 channels is not a whole number of 2-channel strides.
    EXPECT_EQ(runtime_.init(std::move(backend), /*channel_count=*/5, /*worker_count=*/1,
                            /*channels_per_rank=*/2),
              NIXL_ERR_INVALID_PARAM);
}

// With rank encoding disabled (channels_per_rank == 0) a remote memview registration must
// NOT revive channels: a latched channel stays latched.
TEST_F(ProxyRuntimeTest, MemviewRegisterNoopWhenEncodingDisabled) {
    DummyBackendMD remote_md;
    ASSERT_EQ(initRuntime(4, 1), NIXL_SUCCESS);  // 3-arg init => channels_per_rank == 0
    ASSERT_EQ(runtime_.startWorkers(), NIXL_SUCCESS);

    auto *records = hostAliasOf(copyDeviceWorkRing(runtime_.deviceChannelViews()[2]).records);
    auto *slot = hostAliasOf(runtime_.deviceChannelViews()[2].completion_slot);
    publishRecord(records, 0, badPut(2), 11);
    ASSERT_TRUE(waitForCompletedIdx(slot, 11));
    ASSERT_LT(slot->next_status, 0);  // latched

    // A remote memview registration that WOULD activate band 1 if encoding were on.
    nixlMemViewH mvh = nullptr;
    ASSERT_EQ(runtime_.prepMemView(makeRemoteBandDlist({"", "peer1"}, &remote_md), &mvh),
              NIXL_SUCCESS);

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_LT(slot->next_status, 0);  // still latched: activation is disabled
    ASSERT_EQ(runtime_.shutdown(), NIXL_SUCCESS);
}

// Latch a channel, then prove that RE-REGISTERING the remote memview with that band's
// agent changed revives it (the connection-driven path) — while an UNCHANGED band's latch
// is left intact (targeted, no busy-wait, no manual reset call).
TEST_F(ProxyRuntimeTest, MemviewReregisterRevivesLatchedChannel) {
    DummyBackendMD local_md;
    DummyBackendMD remote_md;

    auto backend = std::make_unique<StubBackend>();
    backend_ = backend.get();
    // 4 channels, 2 per rank => rank 0 owns {0,1}, rank 1 owns {2,3}. Single drain worker.
    ASSERT_EQ(runtime_.init(std::move(backend), /*channel_count=*/4, /*worker_count=*/1,
                            /*channels_per_rank=*/2),
              NIXL_SUCCESS);

    // Valid src/dst memviews for the post-revive submission on rank 1.
    nixlMemViewH src_proxy = nullptr;
    nixlMemViewH dst_proxy = nullptr;
    ASSERT_EQ(runtime_.registerProxyMemView(reinterpret_cast<nixlMemViewH>(uintptr_t{0x10}),
                                           &src_proxy),
              NIXL_SUCCESS);
    ASSERT_EQ(runtime_.registerProxyMemView(reinterpret_cast<nixlMemViewH>(uintptr_t{0x20}),
                                           &dst_proxy),
              NIXL_SUCCESS);
    nixl_meta_dlist_t local_dlist(DRAM_SEG);
    local_dlist.addDesc(nixlMetaDesc(0x1000, 64, 0, &local_md));
    ASSERT_EQ(runtime_.storeMetadata(src_proxy, local_dlist), NIXL_SUCCESS);
    nixl_remote_meta_dlist_t dst_dlist(DRAM_SEG);
    nixlRemoteMetaDesc dst_desc("peer1");
    dst_desc.addr = 0x2000;
    dst_desc.len = 64;
    dst_desc.devId = 0;
    dst_desc.metadataP = &remote_md;
    dst_dlist.addDesc(dst_desc);
    ASSERT_EQ(runtime_.storeMetadata(dst_proxy, dst_dlist), NIXL_SUCCESS);

    ASSERT_EQ(runtime_.startWorkers(), NIXL_SUCCESS);

    auto *records2 = hostAliasOf(copyDeviceWorkRing(runtime_.deviceChannelViews()[2]).records);
    auto *slot2 = hostAliasOf(runtime_.deviceChannelViews()[2].completion_slot);
    auto *records0 = hostAliasOf(copyDeviceWorkRing(runtime_.deviceChannelViews()[0]).records);
    auto *slot0 = hostAliasOf(runtime_.deviceChannelViews()[0].completion_slot);

    // Latch channel 2 (band 1) and channel 0 (band 0).
    publishRecord(records2, 0, badPut(2), 11);
    publishRecord(records0, 0, badPut(0), 21);
    ASSERT_TRUE(waitForCompletedIdx(slot2, 11));
    ASSERT_TRUE(waitForCompletedIdx(slot0, 21));
    ASSERT_LT(slot2->next_status, 0);
    ASSERT_LT(slot0->next_status, 0);

    // Re-register a remote memview that activates ONLY band 1 (band 0 stays absent).
    nixlMemViewH mvh = nullptr;
    ASSERT_EQ(runtime_.prepMemView(makeRemoteBandDlist({"", "peer1"}, &remote_md), &mvh),
              NIXL_SUCCESS);

    // Band 1 is revived lazily by the worker; poll until the latch on channel 2 clears.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while (slot2->next_status < 0 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    EXPECT_EQ(slot2->next_status, NIXL_IN_PROG);  // band 1 revived
    EXPECT_LT(slot0->next_status, 0);             // band 0 unchanged => NOT revived

    // A valid submission on the revived channel 2 now completes.
    nixlProxySubmission good{};
    good.opcode = nixl_proxy_opcode_t::PUT;
    good.channel_id = 2;
    good.src_proxy_memview_id = reinterpret_cast<uint64_t>(src_proxy);
    good.src_offset = 4;
    good.dst_proxy_memview_id = reinterpret_cast<uint64_t>(dst_proxy);
    good.dst_offset = 8;
    good.size = 32;
    publishRecord(records2, 1, good, 12);
    ASSERT_TRUE(waitForCompletedIdx(slot2, 12));
    EXPECT_EQ(slot2->next_status, NIXL_SUCCESS);

    ASSERT_EQ(runtime_.shutdown(), NIXL_SUCCESS);
}

// Set an env var for the duration of a test (workers parse their knobs at construction,
// i.e. inside runtime_.init(), so the guard must be alive across initRuntime()).
class ScopedEnvVar {
    public:
        ScopedEnvVar(const char *name, const char *value) : name_(name) {
            setenv(name, value, 1);
        }
        ~ScopedEnvVar() {
            unsetenv(name_);
        }

    private:
        const char *name_;
};

// Bounded round-robin dequeue (NIXL_EP_PROXY_DEQUEUE_BUDGET): a bursting channel must not
// be drained to exhaustion before its sibling channels get service. With budget=2, six
// records pre-published on channel 0 and one on channel 1 (single worker owning both),
// the channel-1 record must be submitted right after the first two channel-0 records —
// not after all six.
TEST_F(ProxyRuntimeTest, DequeueBudgetBoundsPerChannelDrain) {
    ScopedEnvVar budget_env("NIXL_EP_PROXY_DEQUEUE_BUDGET", "2");
    DummyBackendMD local_md;
    DummyBackendMD remote_md;

    ASSERT_EQ(initRuntime(2, 1), NIXL_SUCCESS);

    nixlMemViewH src_proxy = nullptr;
    nixlMemViewH dst_proxy = nullptr;
    ASSERT_EQ(runtime_.registerProxyMemView(reinterpret_cast<nixlMemViewH>(uintptr_t{0x10}),
                                           &src_proxy),
              NIXL_SUCCESS);
    ASSERT_EQ(runtime_.registerProxyMemView(reinterpret_cast<nixlMemViewH>(uintptr_t{0x20}),
                                           &dst_proxy),
              NIXL_SUCCESS);
    nixl_meta_dlist_t local_dlist(DRAM_SEG);
    local_dlist.addDesc(nixlMetaDesc(0x1000, 4096, 0, &local_md));
    ASSERT_EQ(runtime_.storeMetadata(src_proxy, local_dlist), NIXL_SUCCESS);
    nixl_remote_meta_dlist_t remote_dlist(DRAM_SEG);
    nixlRemoteMetaDesc remote_desc("peer");
    remote_desc.addr = 0x2000;
    remote_desc.len = 4096;
    remote_desc.devId = 0;
    remote_desc.metadataP = &remote_md;
    remote_dlist.addDesc(remote_desc);
    ASSERT_EQ(runtime_.storeMetadata(dst_proxy, remote_dlist), NIXL_SUCCESS);

    auto goodPut = [&](uint32_t channel) {
        nixlProxySubmission s{};
        s.opcode = nixl_proxy_opcode_t::PUT;
        s.channel_id = channel;
        s.src_proxy_memview_id = reinterpret_cast<uint64_t>(src_proxy);
        s.dst_proxy_memview_id = reinterpret_cast<uint64_t>(dst_proxy);
        s.size = 32;
        return s;
    };

    // Pre-publish everything BEFORE starting the worker so the first runOnce pass sees
    // both rings loaded: 6 records on channel 0 (the burst), 1 on channel 1.
    constexpr uint32_t kBurst = 6;
    auto *records0 = hostAliasOf(copyDeviceWorkRing(runtime_.deviceChannelViews()[0]).records);
    auto *records1 = hostAliasOf(copyDeviceWorkRing(runtime_.deviceChannelViews()[1]).records);
    for (uint32_t r = 0; r < kBurst; ++r) {
        publishRecord(records0, r, goodPut(0), 11 + r);
    }
    publishRecord(records1, 0, goodPut(1), 21);

    ASSERT_EQ(runtime_.startWorkers(), NIXL_SUCCESS);

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while (std::chrono::steady_clock::now() < deadline) {
        {
            std::lock_guard<std::mutex> lock(backend_->submit_mutex_);
            if (backend_->submissions_.size() >= kBurst + 1) {
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    std::vector<nixlBackendProxySubmission> submissions;
    {
        std::lock_guard<std::mutex> lock(backend_->submit_mutex_);
        submissions = backend_->submissions_;
    }
    ASSERT_EQ(runtime_.shutdown(), NIXL_SUCCESS);

    ASSERT_EQ(submissions.size(), kBurst + 1);
    // Single worker + pre-loaded rings => deterministic order: budget-limited slice of
    // channel 0 (2 records), then channel 1's record, then the rest of channel 0.
    EXPECT_EQ(submissions[0].channel_id, 0u);
    EXPECT_EQ(submissions[1].channel_id, 0u);
    EXPECT_EQ(submissions[2].channel_id, 1u);
    EXPECT_EQ(submissions[2].op_idx, 21u);
    for (size_t i = 3; i < submissions.size(); ++i) {
        EXPECT_EQ(submissions[i].channel_id, 0u) << "index " << i;
    }
}

} // namespace proxy_runtime
} // namespace gtest
