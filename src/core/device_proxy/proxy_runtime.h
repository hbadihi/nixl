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
#ifndef NIXL_SRC_CORE_DEVICE_PROXY_PROXY_RUNTIME_H
#define NIXL_SRC_CORE_DEVICE_PROXY_PROXY_RUNTIME_H

#include <chrono>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "backend_aux.h"
#include "proxy_protocol.h"
#include "backend_adapter.h"
class ProxyWorker;

static constexpr uint32_t kDefaultProxyRingDepth = 256;

struct ProxyRequestState {
    uint64_t op_idx = 0;
    uint64_t backend_req_token = 0;
    nixl_status_t status = NIXL_IN_PROG;
    /** Recorded right after backend_->submit returns; used by ProxyWorker to
     *  measure the inflight latency until the backend reports completion. */
    std::chrono::steady_clock::time_point submit_time{};
};

struct alignas(64) ChannelState {
    ProxyChannelView device_view{};
    std::deque<ProxyRequestState> inflight_requests;
    bool error_latched = false;

    WorkRing        *work_ring_       = nullptr;
    ProxySubmission *records_         = nullptr;
    /** Mapped pinned host memory; host proxy uses __atomic_* on host alias. */
    uint32_t        *producer_idx_host_ = nullptr;
    /** Device-mapped alias of producer_idx_host_ for WorkRing (GPU-writable). */
    uint32_t        *producer_idx_dev_  = nullptr;
    /** Consumer count: host pinned; proxy uses __atomic_* on consumer_idx_host_. */
    uint32_t        *consumer_idx_host_  = nullptr;
    /** Same word as consumer_idx_host_, for WorkRing::consumer_idx (GPU-readable). */
    uint32_t        *consumer_idx_dev_   = nullptr;
    /** Mapped pinned host memory; proxy worker writes directly via host alias. */
    CompletionSlot  *completion_slot_host_ = nullptr;
    /** Device-mapped alias of completion_slot_host_ for ProxyChannelView. */
    CompletionSlot  *completion_slot_dev_  = nullptr;
    /** Mapped pinned host memory; proxy worker publishes after backend submit returns. */
    uint64_t        *submitted_idx_host_ = nullptr;
    /** Device-mapped alias of submitted_idx_host_ for ProxyChannelView. */
    uint64_t        *submitted_idx_dev_  = nullptr;

    ChannelState() = default;
    ~ChannelState();
    ChannelState(ChannelState &&) noexcept;
    ChannelState &operator=(ChannelState &&) noexcept;
    ChannelState(const ChannelState &) = delete;
    ChannelState &operator=(const ChannelState &) = delete;

    nixl_status_t
    allocate(uint32_t channel_id, uint32_t depth);

    void
    deallocate() noexcept;
};

class ProxyMemViewRegistry {
    public:
        enum class EntryState : uint8_t {
            Allocated,
            Ready,
            Retired,
        };

        nixl_status_t
        registerProxyMemView(nixlMemViewH backend_memview,
                             nixlMemViewH *proxy_memview = nullptr);

        nixl_status_t
        unregisterProxyMemView(nixlMemViewH proxy_memview);

        nixl_status_t
        storeMetadata(nixlMemViewH proxy_memview,
                      const nixl_meta_dlist_t &dlist);

        nixl_status_t
        storeMetadata(nixlMemViewH proxy_memview,
                      const nixl_remote_meta_dlist_t &dlist);

        bool
        resolveProxyMemView(nixlMemViewH proxy_memview,
                            nixlMemViewH &backend_memview) const;

        bool
        resolveProxyMemViewId(uint64_t proxy_memview_id,
                              nixlMemViewH &backend_memview) const;

        nixl_status_t
        prepareSubmission(const ProxySubmission &submission,
                          PreparedProxySubmission &prepared_submission) const;

        void
        clear() noexcept;

    private:
        struct StoredEntry {
            uintptr_t base_addr = 0;
            nixlBackendMD *metadata = nullptr;
        };

        struct LocalMetadata {
            nixl_mem_t mem_type = DRAM_SEG;
            std::vector<StoredEntry> entries;
        };

        struct RemoteMetadata {
            nixl_mem_t mem_type = DRAM_SEG;
            std::string remote_agent;
            std::vector<StoredEntry> entries;
        };

        enum class MetadataKind : uint8_t {
            None,
            Local,
            Remote,
        };

        struct RegistryEntry {
            uint64_t proxy_memview_id = 0;
            nixlMemViewH proxy_memview = nullptr;
            nixlMemViewH backend_memview = nullptr;
            EntryState state = EntryState::Allocated;
            MetadataKind metadata_kind = MetadataKind::None;
            LocalMetadata local_metadata{};
            RemoteMetadata remote_metadata{};
        };

        RegistryEntry *
        getEntryForHandle(nixlMemViewH proxy_memview);

        const RegistryEntry *
        getEntryForHandle(nixlMemViewH proxy_memview) const;

        RegistryEntry *
        getEntryForId(uint64_t proxy_memview_id);

        const RegistryEntry *
        getEntryForId(uint64_t proxy_memview_id) const;

        static void
        fillLocalMetadata(const nixl_meta_dlist_t &dlist, LocalMetadata &out);

        static void
        fillRemoteMetadata(const nixl_remote_meta_dlist_t &dlist, RemoteMetadata &out);

        std::vector<RegistryEntry> entries_;
        uint64_t next_proxy_memview_id_ = 1;
};

class ProxyRuntime {
    public:
        ProxyRuntime();
        ~ProxyRuntime();

        ProxyRuntime(ProxyRuntime &&) = delete;
        ProxyRuntime(const ProxyRuntime &) = delete;
        ProxyRuntime& operator=(ProxyRuntime &&) = delete;
        ProxyRuntime& operator=(const ProxyRuntime &) = delete;

        nixl_status_t
        init(DeviceProxyBackendAdapter *backend,
             uint32_t channel_count,
             uint32_t worker_count,
             uint64_t pthr_delay_us = 0);

        nixl_status_t
        loadRemoteConnInfo(const std::string &remote_name,
                           const nixl_blob_t &conn_info);

        nixl_status_t
        registerProxyMemView(nixlMemViewH backend_memview,
                             nixlMemViewH *proxy_memview = nullptr);

        nixl_status_t
        unregisterProxyMemView(nixlMemViewH proxy_memview);

        nixl_status_t
        storeMetadata(nixlMemViewH proxy_memview,
                      const nixl_meta_dlist_t &dlist);

        nixl_status_t
        storeMetadata(nixlMemViewH proxy_memview,
                      const nixl_remote_meta_dlist_t &dlist);

        bool
        resolveProxyMemView(nixlMemViewH proxy_memview,
                            nixlMemViewH &backend_memview) const;

        bool
        resolveProxyMemViewId(uint64_t proxy_memview_id,
                              nixlMemViewH &backend_memview) const;

        nixl_status_t
        startWorkers();

        nixl_status_t
        shutdown();

        const ProxyMemViewRegistry &
        memviewRegistry() const { return memview_registry_; }

        uint32_t
        channelCount() const { return static_cast<uint32_t>(channels_.size()); }

        const ProxyChannelView *
        deviceChannelViews() const { return device_channel_views_; }

        ProxyDeviceContextData *
        deviceContext() const { return device_context_; }

    private:
        void
        joinWorkerThreads() noexcept;

        std::vector<ChannelState> channels_;
        ProxyChannelView       *device_channel_views_ = nullptr;
        ProxyDeviceContextData *device_context_       = nullptr;
        std::vector<std::unique_ptr<ProxyWorker>> workers_;
        ProxyMemViewRegistry memview_registry_;
        DeviceProxyBackendAdapter *backend_ = nullptr;
        uint32_t *shutdown_word_host_ = nullptr;
        uint32_t *shutdown_word_dev_  = nullptr;
        uint32_t ring_depth_ = kDefaultProxyRingDepth;
};

#endif // NIXL_SRC_CORE_DEVICE_PROXY_PROXY_RUNTIME_H
