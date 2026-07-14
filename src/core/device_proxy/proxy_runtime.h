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

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "backend_aux.h"
#include "proxy_protocol.h"
#include "backend_adapter.h"

class ProxyWorker;

static constexpr uint32_t kDefaultProxyRingDepth = 256;

struct nixlProxyRequestState {
    uint64_t op_idx = 0;
    uint64_t backend_req_token = 0;
    nixl_status_t status = NIXL_IN_PROG;
    nixl_proxy_opcode_t opcode = nixl_proxy_opcode_t::PUT;
};

struct alignas(64) nixlProxyChannelState {
    nixlProxyChannelView device_view{};
    // Per-slot in-flight request state (backend token + submit-time status),
    // indexed by ring slot (op_idx % ring_depth_). Sized to ring_depth_ in
    // allocate(). Replaces the old unbounded inflight deque: an op now lives in
    // its ring slot from submit until its network completion advances CI, so the
    // in-flight set is bounded by ring depth.
    std::vector<nixlProxyRequestState> inflight_slots_;
    // Host-only submit cursor: how far the worker has posted to the backend.
    // Advances at submit time; consumer_idx_host_ (CI) now advances only on
    // completion. Invariant: consumer_idx <= submit_idx_. The window
    // [consumer_idx, submit_idx_) is exactly the submitted-but-not-completed set.
    uint64_t submit_idx_ = 0;
    bool error_latched = false;

    nixlProxyWorkRing *work_ring_dev_ = nullptr;
    nixlProxySubmission *records_host_ = nullptr;
    /** Device-resident producer index; only the GPU updates it. */
    uint64_t        *producer_idx_dev_ = nullptr;
    /** Consumer count: host pinned; proxy uses __atomic_* on consumer_idx_host_. */
    uint64_t        *consumer_idx_host_  = nullptr;
    /** Device-resident cache of consumer_idx_host_ used by GPU enqueue backpressure. */
    uint64_t        *consumer_idx_cache_dev_ = nullptr;
    /** Host-side ring depth for the CPU worker; nixlProxyWorkRing itself is device-only. */
    uint32_t         ring_depth_         = 0;
    /** Mapped pinned host memory; proxy worker writes directly via host alias. */
    nixlProxyCompletionSlot  *completion_slot_host_ = nullptr;
    /** Device-mapped alias of completion_slot_host_ for nixlProxyChannelView. */
    nixlProxyCompletionSlot  *completion_slot_dev_  = nullptr;

    nixlProxyChannelState() = default;
    ~nixlProxyChannelState();
    nixlProxyChannelState(nixlProxyChannelState &&) noexcept;
    nixlProxyChannelState &operator=(nixlProxyChannelState &&) noexcept;
    nixlProxyChannelState(const nixlProxyChannelState &) = delete;
    nixlProxyChannelState &operator=(const nixlProxyChannelState &) = delete;

    nixl_status_t
    allocate(uint32_t channel_id, uint32_t depth);

    void
    deallocate() noexcept;
};

class nixlProxyMemViewRegistry {
    public:
        nixl_status_t
        registerProxyMemView(nixlMemViewH backend_memview,
                             nixlMemViewH *proxy_memview);

        nixl_status_t
        prepMemView(const nixl_meta_dlist_t &dlist,
                    nixlMemViewH *proxy_memview);

        nixl_status_t
        prepMemView(const nixl_remote_meta_dlist_t &dlist,
                    nixlMemViewH *proxy_memview);

        nixl_status_t
        prepMemView(nixlMemViewH backend_memview,
                    const nixl_meta_dlist_t &dlist,
                    nixlMemViewH *proxy_memview);

        nixl_status_t
        prepMemView(nixlMemViewH backend_memview,
                    const nixl_remote_meta_dlist_t &dlist,
                    nixlMemViewH *proxy_memview);

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
        prepareSubmission(const nixlProxySubmission &submission,
                          nixlBackendProxySubmission &prepared_submission) const;

        void
        clear() noexcept;

    private:
        struct ProxyMemViewRegStoredEntry {
            uintptr_t base_addr = 0;
            size_t len = 0;
            uint64_t dev_id = 0;
            nixlBackendMD *metadata = nullptr;
            // Per-element remote agent (empty for local elements and null-agent slots).
            // Needed because a memview may span many peers (EP's aggregate remote_mvh),
            // so the per-submission remote_agent must come from the element, not the view.
            std::string remote_agent;
        };

        struct LocalMetadata {
            nixl_mem_t mem_type = DRAM_SEG;
            std::vector<ProxyMemViewRegStoredEntry> entries;
        };

        struct RemoteMetadata {
            nixl_mem_t mem_type = DRAM_SEG;
            std::string remote_agent;
            std::vector<ProxyMemViewRegStoredEntry> entries;
        };

        enum class ProxyMemViewRegEntryState : uint8_t {
            ENTRY_ALLOCATED,
            ENTRY_READY,
            ENTRY_RETIRED,
        };

        enum class ProxyMemViewRegMetadataKind : uint8_t {
            METADATA_KIND_NONE,
            METADATA_KIND_LOCAL,
            METADATA_KIND_REMOTE,
        };

        struct RegistryEntry {
            uint32_t proxy_memview_id = 0;
            nixlMemViewH backend_memview = nullptr;
            ProxyMemViewRegEntryState state = ProxyMemViewRegEntryState::ENTRY_ALLOCATED;
            ProxyMemViewRegMetadataKind metadata_kind = ProxyMemViewRegMetadataKind::METADATA_KIND_NONE;
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

        nixl_status_t
        getRemoteEntryForSubmission(uint64_t proxy_memview_id,
                                    size_t index,
                                    size_t offset,
                                    size_t size,
                                    const RemoteMetadata *&metadata,
                                    const ProxyMemViewRegStoredEntry *&entry) const;

        nixl_status_t
        getLocalEntryForSubmission(uint64_t proxy_memview_id,
                                   size_t index,
                                   size_t offset,
                                   size_t size,
                                   const LocalMetadata *&metadata,
                                   const ProxyMemViewRegStoredEntry *&entry) const;

        static bool
        rangeFits(const ProxyMemViewRegStoredEntry &entry, size_t offset, size_t size);

        static void
        fillLocalMetadata(const nixl_meta_dlist_t &dlist, LocalMetadata &out);

        static void
        fillRemoteMetadata(const nixl_remote_meta_dlist_t &dlist, RemoteMetadata &out);

        std::vector<RegistryEntry> entries_;
        uint64_t next_proxy_memview_id_ = 1;
};

class nixlProxyRuntime {
    public:
        nixlProxyRuntime();
        ~nixlProxyRuntime();

        nixlProxyRuntime(nixlProxyRuntime &&) = delete;
        nixlProxyRuntime(const nixlProxyRuntime &) = delete;
        nixlProxyRuntime& operator=(nixlProxyRuntime &&) = delete;
        nixlProxyRuntime& operator=(const nixlProxyRuntime &) = delete;

        nixl_status_t
        init(std::unique_ptr<nixlDeviceProxyBackendAdapter> backend,
             uint32_t channel_count,
             uint32_t worker_count,
             uint32_t channels_per_rank = 0,
             uint64_t pthr_delay_us = 0);

        nixl_status_t
        loadRemoteConnInfo(const std::string &remote_name,
                           const nixl_blob_t &conn_info);

        nixl_status_t
        registerProxyMemView(nixlMemViewH backend_memview,
                             nixlMemViewH *proxy_memview);

        nixl_status_t
        prepMemView(const nixl_meta_dlist_t &dlist,
                    nixlMemViewH *proxy_memview);

        nixl_status_t
        prepMemView(const nixl_remote_meta_dlist_t &dlist,
                    nixlMemViewH *proxy_memview);

        nixl_status_t
        prepMemView(nixlMemViewH backend_memview,
                    const nixl_meta_dlist_t &dlist,
                    nixlMemViewH *proxy_memview);

        nixl_status_t
        prepMemView(nixlMemViewH backend_memview,
                    const nixl_remote_meta_dlist_t &dlist,
                    nixlMemViewH *proxy_memview);

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

        const nixlProxyMemViewRegistry &
        memviewRegistry() const { return memview_registry_; }

        uint32_t
        channelCount() const { return static_cast<uint32_t>(channels_.size()); }

        const nixlProxyChannelView *
        deviceChannelViews() const {
            return device_channel_views_.empty() ? nullptr : device_channel_views_.data();
        }

        nixlProxyDeviceContextData *
        deviceContext() const { return device_context_; }

        uint64_t
        submittedWorkCount() const;

        void
        resetSubmittedWorkCount();

    private:
        void
        joinWorkerThreads() noexcept;

        // Connection-driven channel activation (called from the remote prepMemView path):
        // diff each element's remote agent against the per-band record and bump the
        // generation of any band whose agent changed (connect/disconnect/reconnect), so
        // the owning worker lazily quiesces+revives just that rank's lanes. Non-blocking.
        void
        activateRemoteBands(const nixl_remote_meta_dlist_t &dlist);

        std::vector<nixlProxyChannelState> channels_;
        std::vector<nixlProxyChannelView> device_channel_views_;
        nixlProxyChannelView *device_channel_views_dev_ = nullptr;
        nixlProxyDeviceContextData *device_context_ = nullptr;
        std::vector<std::unique_ptr<ProxyWorker>> workers_;
        nixlProxyMemViewRegistry memview_registry_;
        std::unique_ptr<nixlDeviceProxyBackendAdapter> backend_;
        uint32_t *shutdown_word_host_ = nullptr;
        uint32_t *shutdown_word_dev_  = nullptr;
        uint32_t ring_depth_ = kDefaultProxyRingDepth;
        bool workers_started_ = false;
        std::atomic<uint64_t> submitted_work_count_{0};
        // Per-rank ring stride (0 = rank encoding disabled). Maps a rank to its channel
        // band [rank*cpr, rank*cpr + cpr).
        uint32_t channels_per_rank_ = 0;
        // One monotonic generation per channel: the connection thread bumps a band's
        // generations (release) when its remote agent changes; the owning worker observes
        // the bump (acquire) in runOnce and lazily reconciles (resetChannel). std::atomic
        // isn't movable, so it lives here rather than in the moved nixlProxyChannelState;
        // workers receive a slice pointer.
        std::unique_ptr<std::atomic<uint64_t>[]> channel_generations_;
        // Per-band (per-rank) record of the currently-bound remote agent, used by
        // activateRemoteBands to detect (re)connect/disconnect. Connection-thread-only
        // (memview prep is serialized), so no atomic needed. Sized channel_count/cpr.
        std::vector<std::string> active_agent_;
};

#endif // NIXL_SRC_CORE_DEVICE_PROXY_PROXY_RUNTIME_H
