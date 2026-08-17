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
#include <unordered_map>
#include <vector>

#include "device_proxy/backend_adapter.h"
#include "device_proxy/proxy_runtime.h"
#include "device_proxy/proxy_worker.h"

namespace gtest {
namespace proxy_runtime {

class DummyBackendMD : public nixlBackendMD {
    public:
        DummyBackendMD() : nixlBackendMD(false) {}
};

struct StubBackendState {
    mutable std::mutex released_mutex;
    std::vector<nixlBackendProxyRequest> released_requests;
};

class StubBackend : public nixlDeviceProxyBackendAdapter {
    public:
        nixl_status_t
        init(uint32_t worker_count, uint32_t channel_count, uint32_t max_peers) override {
            init_called_ = true;
            init_worker_count_ = worker_count;
            init_channel_count_ = channel_count;
            init_max_peers_ = max_peers;
            return init_rc_;
        }

        nixl_status_t
        loadRemoteConnInfo(const std::string &, const nixl_blob_t &) override {
            return NIXL_SUCCESS;
        }

        nixl_status_t
        resolveDirectPointers(const nixl_remote_meta_dlist_t &dlist,
                              std::vector<void *> &direct_ptrs) override {
            ++resolve_direct_pointer_calls_;
            last_resolved_desc_count_ = dlist.descCount();
            if (resolve_direct_pointer_rc_ == NIXL_SUCCESS) {
                direct_ptrs = direct_ptrs_to_return_;
            }
            return resolve_direct_pointer_rc_;
        }

        nixl_status_t
        submit(const nixlBackendProxySubmission &submission,
               nixlBackendProxyRequest &request) override {
            nixl_status_t status = submit_rc_;
            {
                std::lock_guard<std::mutex> lock(submit_mutex_);
                submissions_.push_back(submission);
                if (!submit_rcs_.empty()) {
                    status = submit_rcs_.front();
                    submit_rcs_.erase(submit_rcs_.begin());
                }
            }
            request = request_to_return_;
            if (status == NIXL_IN_PROG && !request) {
                request = nixlBackendProxyRequest{++next_request_token_, 0};
            }
            return status;
        }

        nixl_status_t
        checkCompletion(const nixlBackendProxyRequest &request) override {
            std::lock_guard<std::mutex> lock(completion_mutex_);
            last_checked_request_ = request;
            ++check_completion_calls_;
            const auto status = completion_status_by_token_.find(request.token);
            if (status != completion_status_by_token_.end()) {
                return status->second;
            }
            return completion_rc_;
        }

        nixl_status_t
        progress() override {
            ++progress_calls_;
            return NIXL_SUCCESS;
        }

        nixl_status_t
        progress(uint32_t, uint32_t) override {
            return progress();
        }

        nixl_status_t
        shutdown() override {
            return NIXL_SUCCESS;
        }

        void
        releaseRequest(const nixlBackendProxyRequest &request) override {
            std::lock_guard<std::mutex> lock(state_->released_mutex);
            state_->released_requests.push_back(request);
        }

        void
        setCompletionStatus(uint64_t token, nixl_status_t status) {
            std::lock_guard<std::mutex> lock(completion_mutex_);
            completion_status_by_token_[token] = status;
        }

        bool init_called_ = false;
        uint32_t init_worker_count_ = 0;
        uint32_t init_channel_count_ = 0;
        uint32_t init_max_peers_ = 0;
        nixl_status_t init_rc_ = NIXL_SUCCESS;
        std::atomic<uint64_t> progress_calls_{0};
        mutable std::mutex submit_mutex_;
        std::vector<nixlBackendProxySubmission> submissions_;
        std::vector<nixl_status_t> submit_rcs_;
        uint64_t next_request_token_ = 0;
        nixl_status_t submit_rc_ = NIXL_SUCCESS;
        nixl_status_t completion_rc_ = NIXL_SUCCESS;
        nixlBackendProxyRequest request_to_return_{};
        mutable std::mutex completion_mutex_;
        nixlBackendProxyRequest last_checked_request_{};
        uint64_t check_completion_calls_ = 0;
        std::unordered_map<uint64_t, nixl_status_t> completion_status_by_token_;
        std::shared_ptr<StubBackendState> state_ = std::make_shared<StubBackendState>();
        uint64_t resolve_direct_pointer_calls_ = 0;
        size_t last_resolved_desc_count_ = 0;
        nixl_status_t resolve_direct_pointer_rc_ = NIXL_ERR_NOT_SUPPORTED;
        std::vector<void *> direct_ptrs_to_return_;
};

class ProxyRuntimeTest : public testing::Test {
    protected:
        nixl_status_t
        initRuntime(uint32_t channel_count,
                    uint32_t worker_count,
                    nixl_status_t init_rc = NIXL_SUCCESS,
                    uint32_t max_peers = 4) {
            auto backend = std::make_unique<StubBackend>();
            backend_ = backend.get();
            backend_->init_rc_ = init_rc;
            return runtime_.init(std::move(backend), max_peers, channel_count, worker_count);
        }

        void
        TearDown() override {
            runtime_.shutdown();
        }

        StubBackend *backend_ = nullptr;
        nixlProxyRuntime runtime_;
};

static nixlProxyWorkRing
copyDeviceWorkRing(const nixlProxyChannelView &view) {
    nixlProxyWorkRing ring{};
    EXPECT_EQ(cudaMemcpy(&ring, view.work_ring, sizeof(ring), cudaMemcpyDeviceToHost), cudaSuccess);
    return ring;
}

// Resolve the pinned-host alias of a device-mapped submission or completion buffer.
// GDR-backed control words do not have CUDA host aliases and must be read as device memory.
template<class T>
static T *
hostAliasOf(T *device_alias) {
    cudaPointerAttributes attrs{};
    EXPECT_EQ(cudaPointerGetAttributes(&attrs, device_alias), cudaSuccess);
    EXPECT_NE(attrs.hostPointer, nullptr);
    return static_cast<T *>(attrs.hostPointer);
}

static size_t
channelViewIndex(uint32_t peer, uint32_t channel, uint32_t max_peers = 4) {
    return static_cast<size_t>(channel) * max_peers + peer;
}

static uint32_t
proxyMemViewId(nixlMemViewH proxy_memview) {
    if (proxy_memview == nullptr) {
        return 0;
    }
    nixlProxyDeviceMemView device_memview{};
    EXPECT_EQ(
        cudaMemcpy(&device_memview, proxy_memview, sizeof(device_memview), cudaMemcpyDeviceToHost),
        cudaSuccess);
    return device_memview.proxy_memview_id;
}

static nixlProxyDeviceMemView
copyDeviceMemView(nixlMemViewH proxy_memview) {
    nixlProxyDeviceMemView device_memview{};
    EXPECT_EQ(
        cudaMemcpy(&device_memview, proxy_memview, sizeof(device_memview), cudaMemcpyDeviceToHost),
        cudaSuccess);
    return device_memview;
}

static std::vector<void *>
copyDirectPointers(nixlMemViewH proxy_memview, size_t count) {
    std::vector<void *> direct_ptrs(count, nullptr);
    if (count != 0) {
        auto *direct_ptrs_dev = static_cast<nixlProxyDeviceMemView *>(proxy_memview)->direct_ptrs;
        EXPECT_EQ(cudaMemcpy(direct_ptrs.data(),
                             direct_ptrs_dev,
                             sizeof(void *) * count,
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
    }
    return direct_ptrs;
}

static std::vector<nixlBackendProxySubmission>
waitForSubmissions(StubBackend *backend, size_t count) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(250);
    while (std::chrono::steady_clock::now() < deadline) {
        {
            std::lock_guard<std::mutex> lock(backend->submit_mutex_);
            if (backend->submissions_.size() >= count) {
                return backend->submissions_;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    std::lock_guard<std::mutex> lock(backend->submit_mutex_);
    return backend->submissions_;
}

static bool
waitForCompletedIdx(const nixlProxyChannelView &view, uint64_t completed_idx) {
    auto *completion_slot = hostAliasOf(view.completion_slot);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(250);
    while (std::chrono::steady_clock::now() < deadline) {
        if (__atomic_load_n(&completion_slot->completed_idx, __ATOMIC_ACQUIRE) >= completed_idx) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return __atomic_load_n(&completion_slot->completed_idx, __ATOMIC_ACQUIRE) >= completed_idx;
}

static nixl_status_t
allocateDirectChannel(nixlProxyChannelState &channel,
                      nixlProxyControlBuffer &control_slots,
                      uint32_t depth) {
    nixl_status_t status = control_slots.allocate(kProxyCiSlotBase + 1);
    if (status != NIXL_SUCCESS) {
        return status;
    }
    return channel.allocate(depth, &control_slots, kProxyCiSlotBase);
}

static uint64_t
deviceConsumerIdx(const nixlProxyChannelState &channel) {
    uint64_t consumer_idx = 0;
    EXPECT_EQ(
        cudaMemcpy(
            &consumer_idx, channel.consumer_idx_dev_, sizeof(consumer_idx), cudaMemcpyDeviceToHost),
        cudaSuccess);
    return consumer_idx;
}

static nixlProxySubmission
makeAtomicAddSubmission(nixlMemViewH dst_proxy, uint64_t value = 42) {
    nixlProxySubmission submission{};
    submission.opcode = nixl_proxy_opcode_t::ATOMIC_ADD;
    submission.channel_id = 0;
    submission.dst_proxy_memview_id = proxyMemViewId(dst_proxy);
    submission.dst_offset = 0;
    submission.size = sizeof(uint64_t);
    submission.value = value;
    return submission;
}

static nixlProxySubmission
makeInvalidAtomicAddSubmission() {
    return makeAtomicAddSubmission(nullptr);
}

static void
publishRecord(nixlProxySubmission *records,
              uint32_t slot,
              const nixlProxySubmission &submission,
              uint64_t op_idx) {
    nixlProxySubmission record = submission;
    record.op_idx = 0;
    records[slot] = record;
    __atomic_store_n(&records[slot].op_idx, op_idx, __ATOMIC_RELEASE);
}

static std::unique_ptr<ProxyWorker>
makeDirectWorker(StubBackend *backend,
                 const nixlProxyMemViewRegistry *registry,
                 std::atomic<uint64_t> *shutdown_state,
                 nixlProxyChannelState *channel) {
    return std::make_unique<ProxyWorker>(backend, registry, shutdown_state, channel, 1, 1, 0, 1, 0);
}

static nixl_meta_dlist_t
makeLocalDlist(uintptr_t addr, DummyBackendMD *md, uint64_t len = 64) {
    nixl_meta_dlist_t dlist(DRAM_SEG);
    dlist.addDesc(nixlMetaDesc(addr, len, 0, md));
    return dlist;
}

static nixl_remote_meta_dlist_t
makeRemotePeerDlist(const std::vector<std::string> &agents,
                    nixlBackendMD *md,
                    uintptr_t base_addr = 0x4000) {
    nixl_remote_meta_dlist_t dlist(VRAM_SEG);
    for (const auto &agent : agents) {
        if (agent.empty()) {
            dlist.addDesc(nixlRemoteMetaDesc(nixl_null_agent));
        } else {
            nixlRemoteMetaDesc desc(agent);
            desc.addr = base_addr;
            desc.len = 64;
            desc.devId = 0;
            desc.metadataP = md;
            dlist.addDesc(desc);
        }
    }
    return dlist;
}

static nixl_status_t
prepareRemoteMemView(nixlProxyRuntime &runtime,
                     const std::vector<std::string> &agents,
                     DummyBackendMD *md,
                     nixlMemViewH *proxy_memview,
                     uintptr_t base_addr = 0x4000) {
    return runtime.prepMemView(makeRemotePeerDlist(agents, md, base_addr), proxy_memview);
}

static bool
publishRuntimeRecord(nixlProxyRuntime &runtime,
                     uint32_t peer,
                     uint32_t channel,
                     uint32_t max_peers,
                     const nixlProxySubmission &submission,
                     uint64_t op_idx,
                     uint32_t slot = 0) {
    const nixlProxyWorkRing ring =
        copyDeviceWorkRing(runtime.deviceChannelViews()[channelViewIndex(peer, channel, max_peers)]);
    auto *records = hostAliasOf(ring.records);
    if (records == nullptr) {
        return false;
    }
    publishRecord(records, slot, submission, op_idx);
    return true;
}

class DirectWorkerHarness {
    public:
        nixl_status_t
        setup(uint32_t depth) {
            nixl_status_t status =
                registry.prepMemView(makeRemotePeerDlist({"peer"}, &remote_md), &dst_proxy);
            if (status != NIXL_SUCCESS) {
                return status;
            }
            status = allocateDirectChannel(channel, control_slots, depth);
            if (status != NIXL_SUCCESS) {
                return status;
            }
            worker = makeDirectWorker(&backend, &registry, &shutdown_state, &channel);
            return NIXL_SUCCESS;
        }

        void
        publishAtomic(uint32_t slot, uint64_t op_idx) {
            publishRecord(channel.records_host_, slot, makeAtomicAddSubmission(dst_proxy), op_idx);
        }

        void
        runOnce() {
            worker->runOnce();
        }

        void
        expectProgress(uint64_t consumer_idx, uint64_t completed_idx) const {
            EXPECT_EQ(deviceConsumerIdx(channel), consumer_idx);
            EXPECT_EQ(__atomic_load_n(&channel.completion_slot_host_->completed_idx, __ATOMIC_ACQUIRE),
                      completed_idx);
        }

        StubBackend backend;
        nixlProxyMemViewRegistry registry;
        DummyBackendMD remote_md;
        nixlMemViewH dst_proxy = nullptr;
        nixlProxyChannelState channel;
        nixlProxyControlBuffer control_slots;
        std::atomic<uint64_t> shutdown_state{
            static_cast<uint64_t>(nixl_proxy_control_state_t::RUNNING)};
        std::unique_ptr<ProxyWorker> worker;
};

struct InvalidInitConfig {
    const char *name;
    uint32_t max_peers;
    uint32_t channels;
    uint32_t workers;
    nixl_status_t backend_status;
    bool null_backend;
};

class ProxyRuntimeInvalidInitTest : public ProxyRuntimeTest,
                                    public testing::WithParamInterface<InvalidInitConfig> {};

struct WorkerCountConfig {
    const char *name;
    uint32_t channels;
    uint32_t workers;
    uint32_t max_peers;
    uint32_t expected_workers;
};

class ProxyRuntimeWorkerCountTest : public ProxyRuntimeTest,
                                    public testing::WithParamInterface<WorkerCountConfig> {};

struct WorkerLifecycleConfig {
    const char *name;
    uint32_t channels;
    uint32_t workers;
    uint32_t warmup_ms;
};

class ProxyRuntimeWorkerLifecycleTest
    : public ProxyRuntimeTest,
      public testing::WithParamInterface<WorkerLifecycleConfig> {};

struct DirectPointerConfig {
    const char *name;
    nixl_status_t resolver_status;
    nixl_status_t expected_status;
    size_t expected_direct_ptr_count;
    size_t descriptor_count;
};

class ProxyRuntimeDirectPointerTest : public ProxyRuntimeTest,
                                      public testing::WithParamInterface<DirectPointerConfig> {};

enum class WorkerSubmissionKind { PUT, ATOMIC_ADD };

struct WorkerSubmissionConfig {
    const char *name;
    WorkerSubmissionKind kind;
    nixlBackendProxyRequest request;
};

class ProxyRuntimeWorkerSubmissionTest
    : public ProxyRuntimeTest,
      public testing::WithParamInterface<WorkerSubmissionConfig> {};

TEST_F(ProxyRuntimeTest, InitCallsBackendInit) {
    ASSERT_EQ(initRuntime(4, 2), NIXL_SUCCESS);
    EXPECT_TRUE(backend_->init_called_);
    EXPECT_EQ(backend_->init_worker_count_, 2u);
    EXPECT_EQ(backend_->init_channel_count_, 4u);
}

TEST_P(ProxyRuntimeInvalidInitTest, RejectsInvalidConfiguration) {
    const auto &config = GetParam();
    if (config.null_backend) {
        EXPECT_EQ(runtime_.init(nullptr, config.max_peers, config.channels, config.workers),
                  NIXL_ERR_INVALID_PARAM);
    } else {
        EXPECT_EQ(initRuntime(
                      config.channels, config.workers, config.backend_status, config.max_peers),
                  config.backend_status == NIXL_SUCCESS ? NIXL_ERR_INVALID_PARAM
                                                        : config.backend_status);
    }
}

INSTANTIATE_TEST_SUITE_P(
    InvalidConfigurations,
    ProxyRuntimeInvalidInitTest,
    testing::Values(InvalidInitConfig{"NullBackend", 4, 4, 2, NIXL_SUCCESS, true},
                    InvalidInitConfig{"ZeroPeerCapacity", 0, 2, 1, NIXL_SUCCESS, false},
                    InvalidInitConfig{"ZeroChannels", 4, 0, 2, NIXL_SUCCESS, false},
                    InvalidInitConfig{"ZeroWorkers", 4, 4, 0, NIXL_SUCCESS, false},
                    InvalidInitConfig{"BackendFailure", 4, 4, 2, NIXL_ERR_BACKEND, false}),
    [](const testing::TestParamInfo<InvalidInitConfig> &info) { return info.param.name; });

TEST_F(ProxyRuntimeTest, DeviceChannelViewMatrixStartsAllocated) {
    ASSERT_EQ(initRuntime(3, 1), NIXL_SUCCESS);
    const nixlProxyChannelView *views = runtime_.deviceChannelViews();
    ASSERT_NE(views, nullptr);
    for (uint32_t peer = 0; peer < 4; ++peer) {
        for (uint32_t channel = 0; channel < 3; ++channel) {
            const auto &view = views[channelViewIndex(peer, channel)];
            EXPECT_NE(view.work_ring, nullptr);
            EXPECT_NE(view.completion_slot, nullptr);
        }
    }
}

TEST_F(ProxyRuntimeTest, WorkRingIndicesStartAtZero) {
    DummyBackendMD remote_md;
    ASSERT_EQ(initRuntime(2, 1), NIXL_SUCCESS);
    nixlMemViewH remote_mvh = nullptr;
    ASSERT_EQ(runtime_.prepMemView(makeRemotePeerDlist({"peer"}, &remote_md), &remote_mvh),
              NIXL_SUCCESS);
    const nixlProxyChannelView *views = runtime_.deviceChannelViews();
    for (uint32_t channel = 0; channel < 2; ++channel) {
        const nixlProxyWorkRing ring = copyDeviceWorkRing(views[channelViewIndex(0, channel)]);
        uint64_t producer = 0;
        uint64_t consumer = 0;
        ASSERT_EQ(
            cudaMemcpy(&producer, ring.producer_idx, sizeof(producer), cudaMemcpyDeviceToHost),
            cudaSuccess);
        ASSERT_EQ(
            cudaMemcpy(&consumer, ring.consumer_idx, sizeof(consumer), cudaMemcpyDeviceToHost),
            cudaSuccess);
        EXPECT_EQ(producer, 0u);
        EXPECT_EQ(consumer, 0u);
    }
}

TEST_F(ProxyRuntimeTest, CompletionSlotsInitialized) {
    DummyBackendMD remote_md;
    ASSERT_EQ(initRuntime(2, 1), NIXL_SUCCESS);
    nixlMemViewH remote_mvh = nullptr;
    ASSERT_EQ(runtime_.prepMemView(makeRemotePeerDlist({"peer"}, &remote_md), &remote_mvh),
              NIXL_SUCCESS);
    const nixlProxyChannelView *views = runtime_.deviceChannelViews();
    for (uint32_t channel = 0; channel < 2; ++channel) {
        nixlProxyCompletionSlot slot{};
        ASSERT_EQ(cudaMemcpy(&slot,
                             views[channelViewIndex(0, channel)].completion_slot,
                             sizeof(nixlProxyCompletionSlot),
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        EXPECT_EQ(slot.completed_idx, 0u);
        EXPECT_EQ(slot.next_status, NIXL_IN_PROG);
    }
}

TEST_P(ProxyRuntimeWorkerCountTest, InitializesExpectedWorkerCount) {
    const auto &config = GetParam();
    ASSERT_EQ(initRuntime(config.channels, config.workers, NIXL_SUCCESS, config.max_peers),
              NIXL_SUCCESS);
    EXPECT_EQ(backend_->init_worker_count_, config.expected_workers);
    EXPECT_EQ(backend_->init_channel_count_, config.channels);
}

INSTANTIATE_TEST_SUITE_P(
    Configurations,
    ProxyRuntimeWorkerCountTest,
    testing::Values(WorkerCountConfig{"NotClampedToPeerCapacity", 8, 8, 2, 8},
                    WorkerCountConfig{"ClampedToChannelCount", 2, 8, 4, 2}),
    [](const testing::TestParamInfo<WorkerCountConfig> &info) { return info.param.name; });

TEST_F(ProxyRuntimeTest, DeviceContextPopulated) {
    ASSERT_EQ(initRuntime(3, 1), NIXL_SUCCESS);
    auto *device_ctx = runtime_.deviceContext();
    ASSERT_NE(device_ctx, nullptr);
    nixlProxyDeviceContextData ctx{};
    ASSERT_EQ(cudaMemcpy(&ctx, device_ctx, sizeof(ctx), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(ctx.max_peers, 4u);
    EXPECT_EQ(ctx.num_channels, 3u);
    EXPECT_NE(ctx.channels, nullptr);
    EXPECT_NE(ctx.shutdown_word, nullptr);
}

TEST_F(ProxyRuntimeTest, DeviceContextCarriedByMemView) {
    DummyBackendMD remote_md;
    ASSERT_EQ(initRuntime(3, 1), NIXL_SUCCESS);
    nixlMemViewH remote_mvh = nullptr;
    ASSERT_EQ(runtime_.prepMemView(makeRemotePeerDlist({"peer"}, &remote_md), &remote_mvh),
              NIXL_SUCCESS);
    EXPECT_EQ(copyDeviceMemView(remote_mvh).context, runtime_.deviceContext());
}

TEST_F(ProxyRuntimeTest, DeviceContextNullAfterShutdown) {
    ASSERT_EQ(initRuntime(2, 1), NIXL_SUCCESS);
    ASSERT_NE(runtime_.deviceContext(), nullptr);
    ASSERT_EQ(runtime_.shutdown(), NIXL_SUCCESS);
    EXPECT_EQ(runtime_.deviceContext(), nullptr);
}

TEST_P(ProxyRuntimeWorkerLifecycleTest, StartsAndShutsDown) {
    const auto &config = GetParam();
    ASSERT_EQ(initRuntime(config.channels, config.workers), NIXL_SUCCESS);
    ASSERT_EQ(runtime_.startWorkers(), NIXL_SUCCESS);
    std::this_thread::sleep_for(std::chrono::milliseconds(config.warmup_ms));
    ASSERT_EQ(runtime_.shutdown(), NIXL_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(
    Configurations,
    ProxyRuntimeWorkerLifecycleTest,
    testing::Values(WorkerLifecycleConfig{"StartWorkersAndShutdown", 2, 2, 20},
                    WorkerLifecycleConfig{"SingleChannelSingleWorker", 1, 1, 10},
                    WorkerLifecycleConfig{"ManyChannelsManyWorkers", 16, 4, 20}),
    [](const testing::TestParamInfo<WorkerLifecycleConfig> &info) { return info.param.name; });

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
    ASSERT_EQ(runtime_.startWorkers(), NIXL_SUCCESS);
    EXPECT_EQ(runtime_.shutdown(), NIXL_SUCCESS);
}

TEST_F(ProxyRuntimeTest, PrepMemViewProducesReadyEntries) {
    DummyBackendMD local_md;
    DummyBackendMD remote_md;
    ASSERT_EQ(initRuntime(1, 1), NIXL_SUCCESS);
    const auto local_backend = reinterpret_cast<nixlMemViewH>(uintptr_t{0x10});
    const auto remote_backend = reinterpret_cast<nixlMemViewH>(uintptr_t{0x20});

    nixl_meta_dlist_t local_dlist(DRAM_SEG);
    local_dlist.addDesc(nixlMetaDesc(0x1000, 64, 0, &local_md));

    nixl_remote_meta_dlist_t remote_dlist(VRAM_SEG);
    nixlRemoteMetaDesc remote_desc("peer");
    remote_desc.addr = 0x2000;
    remote_desc.len = 64;
    remote_desc.devId = 0;
    remote_desc.metadataP = &remote_md;
    remote_dlist.addDesc(remote_desc);

    nixlMemViewH src_proxy = nullptr;
    nixlMemViewH dst_proxy = nullptr;
    ASSERT_EQ(runtime_.prepMemView(local_backend, local_dlist, &src_proxy), NIXL_SUCCESS);
    ASSERT_EQ(runtime_.prepMemView(remote_backend, remote_dlist, &dst_proxy), NIXL_SUCCESS);

    nixlMemViewH resolved = nullptr;
    EXPECT_TRUE(runtime_.resolveProxyMemView(src_proxy, resolved));
    EXPECT_EQ(resolved, local_backend);
    EXPECT_TRUE(runtime_.resolveProxyMemView(dst_proxy, resolved));
    EXPECT_EQ(resolved, remote_backend);

    nixlProxySubmission submission{};
    submission.opcode = nixl_proxy_opcode_t::PUT;
    submission.src_proxy_memview_id = proxyMemViewId(src_proxy);
    submission.src_offset = 4;
    submission.dst_proxy_memview_id = proxyMemViewId(dst_proxy);
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
    EXPECT_EQ(prepared_submission.remote_agent, "peer");
}

TEST_F(ProxyRuntimeTest, PrepMemViewRejectsNullOutput) {
    DummyBackendMD local_md;
    nixl_meta_dlist_t local_dlist(DRAM_SEG);
    local_dlist.addDesc(nixlMetaDesc(0x1000, 64, 0, &local_md));

    EXPECT_EQ(runtime_.prepMemView(local_dlist, nullptr), NIXL_ERR_INVALID_PARAM);
}

TEST_F(ProxyRuntimeTest, PrepRemoteMemViewRejectsNonVramMetadata) {
    DummyBackendMD remote_md;

    nixl_remote_meta_dlist_t remote_dlist(DRAM_SEG);
    nixlRemoteMetaDesc remote_desc("peer");
    remote_desc.addr = 0x2000;
    remote_desc.len = 64;
    remote_desc.devId = 0;
    remote_desc.metadataP = &remote_md;
    remote_dlist.addDesc(remote_desc);

    nixlMemViewH dst_proxy = nullptr;
    EXPECT_EQ(runtime_.prepMemView(remote_dlist, &dst_proxy), NIXL_ERR_INVALID_PARAM);
}

TEST_P(ProxyRuntimeDirectPointerTest, PrepRemoteMemViewHandlesResolverOutcome) {
    DummyBackendMD remote_md;
    ASSERT_EQ(initRuntime(1, 1), NIXL_SUCCESS);
    const auto &config = GetParam();

    backend_->resolve_direct_pointer_rc_ = config.resolver_status;
    backend_->direct_ptrs_to_return_ = {reinterpret_cast<void *>(uintptr_t{0xabc00000}), nullptr};

    nixlMemViewH dst_proxy = nullptr;
    const std::vector<std::string> peers =
        config.descriptor_count == 1 ? std::vector<std::string>{"peer"}
                                     : std::vector<std::string>{"peer0", "peer1"};
    const nixl_status_t status =
        runtime_.prepMemView(makeRemotePeerDlist(peers, &remote_md), &dst_proxy);

    EXPECT_EQ(backend_->resolve_direct_pointer_calls_, 1u);
    EXPECT_EQ(backend_->last_resolved_desc_count_, config.descriptor_count);
    ASSERT_EQ(status, config.expected_status);
    if (status != NIXL_SUCCESS) {
        EXPECT_EQ(dst_proxy, nullptr);
        return;
    }

    const nixlProxyDeviceMemView device_memview = copyDeviceMemView(dst_proxy);
    EXPECT_EQ(device_memview.proxy_memview_id, proxyMemViewId(dst_proxy));
    EXPECT_EQ(device_memview.direct_ptr_count, config.expected_direct_ptr_count);
    if (config.expected_direct_ptr_count != 0) {
        EXPECT_EQ(copyDirectPointers(dst_proxy, config.expected_direct_ptr_count),
                  backend_->direct_ptrs_to_return_);
    }
}

INSTANTIATE_TEST_SUITE_P(
    ResolverOutcomes,
    ProxyRuntimeDirectPointerTest,
    testing::Values(DirectPointerConfig{
                        "Resolved", NIXL_SUCCESS, NIXL_SUCCESS, 2, 2},
                    DirectPointerConfig{
                        "UnsupportedFallsBack", NIXL_ERR_NOT_SUPPORTED, NIXL_SUCCESS, 0, 1},
                    DirectPointerConfig{
                        "ErrorPropagated", NIXL_ERR_INVALID_PARAM, NIXL_ERR_INVALID_PARAM, 0, 1}),
    [](const testing::TestParamInfo<DirectPointerConfig> &info) { return info.param.name; });

TEST_P(ProxyRuntimeWorkerSubmissionTest, SubmitsPreparedDescriptors) {
    DummyBackendMD local_md;
    DummyBackendMD remote_md;
    const auto &config = GetParam();

    ASSERT_EQ(initRuntime(1, 1), NIXL_SUCCESS);
    backend_->submit_rc_ = NIXL_IN_PROG;
    backend_->completion_rc_ = NIXL_SUCCESS;
    backend_->request_to_return_ = config.request;

    nixlMemViewH dst_proxy = nullptr;
    nixlProxySubmission submission{};
    if (config.kind == WorkerSubmissionKind::PUT) {
        nixlMemViewH src_proxy = nullptr;
        ASSERT_EQ(runtime_.registerProxyMemView(
                      reinterpret_cast<nixlMemViewH>(uintptr_t{0x10}), &src_proxy),
                  NIXL_SUCCESS);
        ASSERT_EQ(runtime_.storeMetadata(src_proxy, makeLocalDlist(0x1000, &local_md)),
                  NIXL_SUCCESS);
        ASSERT_EQ(runtime_.startWorkers(), NIXL_SUCCESS);
        ASSERT_EQ(prepareRemoteMemView(runtime_, {"peer"}, &remote_md, &dst_proxy, 0x2000),
                  NIXL_SUCCESS);

        submission.opcode = nixl_proxy_opcode_t::PUT;
        submission.src_proxy_memview_id = proxyMemViewId(src_proxy);
        submission.src_offset = 4;
        submission.dst_proxy_memview_id = proxyMemViewId(dst_proxy);
        submission.dst_offset = 8;
        submission.size = 32;
    } else {
        ASSERT_EQ(runtime_.startWorkers(), NIXL_SUCCESS);
        ASSERT_EQ(prepareRemoteMemView(runtime_, {"peer"}, &remote_md, &dst_proxy, 0x2000),
                  NIXL_SUCCESS);

        submission.opcode = nixl_proxy_opcode_t::ATOMIC_ADD;
        submission.dst_proxy_memview_id = proxyMemViewId(dst_proxy);
        submission.dst_offset = 8;
        submission.size = sizeof(uint64_t);
        submission.value = 42;
    }

    ASSERT_TRUE(publishRuntimeRecord(runtime_, 0, 0, 4, submission, 11));

    const auto submissions = waitForSubmissions(backend_, 1);
    ASSERT_TRUE(waitForCompletedIdx(runtime_.deviceChannelViews()[0], 11));

    ASSERT_EQ(runtime_.shutdown(), NIXL_SUCCESS);

    ASSERT_EQ(submissions.size(), 1u);
    const auto &prepared = submissions.front();
    EXPECT_EQ(prepared.op_idx, 11u);
    EXPECT_EQ(prepared.opcode, config.kind == WorkerSubmissionKind::PUT
                                   ? nixl_proxy_opcode_t::PUT
                                   : nixl_proxy_opcode_t::ATOMIC_ADD);
    EXPECT_EQ(prepared.remote.mem_type, VRAM_SEG);
    EXPECT_EQ(prepared.remote.desc.addr, 0x2008u);
    EXPECT_EQ(prepared.remote.desc.len,
              config.kind == WorkerSubmissionKind::PUT ? 32u : sizeof(uint64_t));
    EXPECT_EQ(prepared.remote.desc.metadataP, &remote_md);
    EXPECT_EQ(prepared.remote_agent, "peer");
    EXPECT_EQ(prepared.channel_id, 0u);
    EXPECT_EQ(prepared.peer_index, 0u);
    if (config.kind == WorkerSubmissionKind::PUT) {
        EXPECT_EQ(prepared.local.mem_type, DRAM_SEG);
        EXPECT_EQ(prepared.local.desc.addr, 0x1004u);
        EXPECT_EQ(prepared.local.desc.len, 32u);
        EXPECT_EQ(prepared.local.desc.metadataP, &local_md);
    } else {
        EXPECT_EQ(prepared.value, 42u);
    }
    EXPECT_EQ(backend_->last_checked_request_.token, config.request.token);
    EXPECT_EQ(backend_->last_checked_request_.context, config.request.context);
    EXPECT_GT(backend_->check_completion_calls_, 0u);
}

INSTANTIATE_TEST_SUITE_P(
    SubmissionTypes,
    ProxyRuntimeWorkerSubmissionTest,
    testing::Values(WorkerSubmissionConfig{
                        "WorkerSubmitsPreparedTransportDescriptors",
                        WorkerSubmissionKind::PUT,
                        nixlBackendProxyRequest{101, 7}},
                    WorkerSubmissionConfig{
                        "WorkerSubmitsPreparedAtomicAddDescriptor",
                        WorkerSubmissionKind::ATOMIC_ADD,
                        nixlBackendProxyRequest{202, 8}}),
    [](const testing::TestParamInfo<WorkerSubmissionConfig> &info) { return info.param.name; });

TEST_F(ProxyRuntimeTest, ConsumerIndexAdvancesOnlyAfterBackendCompletion) {
    DirectWorkerHarness harness;
    harness.backend.submit_rc_ = NIXL_IN_PROG;
    harness.backend.completion_rc_ = NIXL_IN_PROG;
    ASSERT_EQ(harness.setup(2), NIXL_SUCCESS);
    harness.publishAtomic(0, 1);

    harness.runOnce();
    ASSERT_EQ(harness.backend.submissions_.size(), 1u);
    harness.expectProgress(0, 0);

    harness.backend.setCompletionStatus(1, NIXL_SUCCESS);
    harness.runOnce();

    harness.expectProgress(1, 1);
    EXPECT_EQ(harness.channel.completion_slot_host_->next_status, NIXL_SUCCESS);
}

TEST_F(ProxyRuntimeTest, InFlightRequestsAreBoundedByRingDepth) {
    DirectWorkerHarness harness;
    harness.backend.submit_rc_ = NIXL_IN_PROG;
    harness.backend.completion_rc_ = NIXL_IN_PROG;
    ASSERT_EQ(harness.setup(2), NIXL_SUCCESS);
    harness.publishAtomic(0, 1);
    harness.publishAtomic(1, 2);

    harness.runOnce();
    harness.runOnce();
    ASSERT_EQ(harness.backend.submissions_.size(), 2u);
    harness.expectProgress(0, 0);

    harness.publishAtomic(0, 3);
    harness.runOnce();
    EXPECT_EQ(harness.backend.submissions_.size(), 2u);

    harness.backend.setCompletionStatus(1, NIXL_SUCCESS);
    harness.runOnce();
    EXPECT_EQ(harness.backend.submissions_.size(), 2u);
    harness.expectProgress(1, 1);

    harness.runOnce();
    EXPECT_EQ(harness.backend.submissions_.size(), 3u);
    EXPECT_EQ(harness.backend.submissions_.back().op_idx, 3u);
}

TEST_F(ProxyRuntimeTest, CompletionsPublishInSubmissionOrder) {
    DirectWorkerHarness harness;
    harness.backend.submit_rc_ = NIXL_IN_PROG;
    harness.backend.completion_rc_ = NIXL_IN_PROG;
    ASSERT_EQ(harness.setup(3), NIXL_SUCCESS);
    harness.publishAtomic(0, 1);
    harness.publishAtomic(1, 2);

    harness.runOnce();
    harness.runOnce();
    ASSERT_EQ(harness.backend.submissions_.size(), 2u);

    harness.backend.setCompletionStatus(2, NIXL_SUCCESS);
    harness.runOnce();
    harness.expectProgress(0, 0);

    harness.backend.setCompletionStatus(1, NIXL_SUCCESS);
    harness.runOnce();
    harness.expectProgress(2, 2);
}

TEST_F(ProxyRuntimeTest, PreparationErrorLatchesStatusButLaterWorkIsReclaimed) {
    DirectWorkerHarness harness;
    harness.backend.submit_rc_ = NIXL_IN_PROG;
    harness.backend.completion_rc_ = NIXL_SUCCESS;
    ASSERT_EQ(harness.setup(3), NIXL_SUCCESS);

    publishRecord(harness.channel.records_host_, 0, makeInvalidAtomicAddSubmission(), 1);
    harness.runOnce();
    harness.expectProgress(1, 1);
    EXPECT_LT(harness.channel.completion_slot_host_->next_status, 0);

    harness.publishAtomic(1, 2);
    harness.runOnce();
    harness.expectProgress(2, 1);
    ASSERT_EQ(harness.backend.submissions_.size(), 1u);
    EXPECT_EQ(harness.backend.submissions_.front().op_idx, 2u);
}

TEST_F(ProxyRuntimeTest, SubmitAndCompletionErrorsLatchFirstStatusAndRetireWork) {
    DirectWorkerHarness harness;
    harness.backend.submit_rcs_ = {NIXL_ERR_BACKEND, NIXL_IN_PROG, NIXL_IN_PROG};
    harness.backend.completion_rc_ = NIXL_IN_PROG;
    ASSERT_EQ(harness.setup(4), NIXL_SUCCESS);
    harness.publishAtomic(0, 1);
    harness.publishAtomic(1, 2);
    harness.publishAtomic(2, 3);

    harness.runOnce();
    harness.expectProgress(1, 1);
    const nixl_status_t first_error = harness.channel.completion_slot_host_->next_status;
    EXPECT_LT(first_error, 0);

    harness.runOnce();
    ASSERT_EQ(harness.backend.submissions_.size(), 2u);
    harness.backend.setCompletionStatus(1, NIXL_ERR_BACKEND);
    harness.runOnce();
    harness.expectProgress(2, 1);
    EXPECT_EQ(harness.channel.completion_slot_host_->next_status, first_error);

    harness.backend.setCompletionStatus(2, NIXL_SUCCESS);
    harness.runOnce();
    harness.expectProgress(3, 1);
    EXPECT_EQ(harness.channel.completion_slot_host_->next_status, first_error);
}

TEST_F(ProxyRuntimeTest, ShutdownReleasesAllPendingBackendRequests) {
    DummyBackendMD remote_md;

    ASSERT_EQ(initRuntime(1, 1), NIXL_SUCCESS);
    backend_->submit_rc_ = NIXL_IN_PROG;
    backend_->completion_rc_ = NIXL_IN_PROG;
    auto backend_state = backend_->state_;

    nixlMemViewH dst_proxy = nullptr;
    ASSERT_EQ(runtime_.startWorkers(), NIXL_SUCCESS);
    ASSERT_EQ(prepareRemoteMemView(runtime_, {"peer"}, &remote_md, &dst_proxy), NIXL_SUCCESS);

    const auto submission = makeAtomicAddSubmission(dst_proxy);
    ASSERT_TRUE(publishRuntimeRecord(runtime_, 0, 0, 4, submission, 1));
    ASSERT_TRUE(publishRuntimeRecord(runtime_, 0, 0, 4, submission, 2, 1));

    const auto submissions = waitForSubmissions(backend_, 2);
    ASSERT_EQ(submissions.size(), 2u);
    ASSERT_EQ(runtime_.shutdown(), NIXL_SUCCESS);

    std::lock_guard<std::mutex> lock(backend_state->released_mutex);
    ASSERT_EQ(backend_state->released_requests.size(), 2u);
    EXPECT_EQ(backend_state->released_requests[0].token, 1u);
    EXPECT_EQ(backend_state->released_requests[1].token, 2u);
}

} // namespace proxy_runtime
} // namespace gtest
