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
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "proxy_protocol.h"

class DeviceProxyBackendAdapter;
class ProxyWorker;

static constexpr uint32_t kDefaultProxyRingDepth = 256;

struct ProxyRequestState {
    uint64_t op_idx = 0;
    uint64_t backend_req_token = 0;
    nixl_status_t status = NIXL_IN_PROG;
    bool publish_ready = false;
};

struct ChannelState {
    ProxyChannelView device_view{};
    std::vector<ProxyRequestState> inflight_requests;

    WorkRing        *work_ring_       = nullptr;
    ProxySubmission *records_         = nullptr;
    /** Producer count in HBM (cudaMalloc); GPU atomics; host reads via cudaMemcpy. */
    uint32_t        *producer_idx_   = nullptr;
    /** Consumer count: host pinned; proxy uses __atomic_* on consumer_idx_host_. */
    uint32_t        *consumer_idx_host_  = nullptr;
    /** Same word as consumer_idx_host_, for WorkRing::consumer_idx (GPU-readable). */
    uint32_t        *consumer_idx_dev_   = nullptr;
    /** Device pointer (cudaMalloc); publish with cudaMemcpy from the proxy worker. */
    CompletionSlot  *completion_slot_ = nullptr;

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
        nixl_status_t
        registerProxyMemView(nixlMemViewH backend_memview,
                             nixlMemViewH *proxy_memview = nullptr);

        nixl_status_t
        unregisterProxyMemView(nixlMemViewH proxy_memview);

        bool
        resolveProxyMemView(nixlMemViewH proxy_memview,
                            nixlMemViewH &backend_memview) const;

        bool
        resolveProxyMemViewId(uint64_t proxy_memview_id,
                              nixlMemViewH &backend_memview) const;

        void
        clear() noexcept;

    private:
        mutable std::mutex mutex_;
        std::vector<nixlMemViewH> backend_memview_by_proxy_id_;
        uint64_t next_proxy_memview_id_ = 1;
};

class ProxyRuntime {
    public:
        ProxyRuntime();
        ~ProxyRuntime();

        ProxyRuntime(ProxyRuntime &&) = delete;
        ProxyRuntime(const ProxyRuntime &) = delete;
        void operator=(ProxyRuntime &&) = delete;
        void operator=(const ProxyRuntime &) = delete;

        nixl_status_t
        init(DeviceProxyBackendAdapter *backend,
             uint32_t channel_count,
             uint32_t worker_count);

        nixl_status_t
        loadRemoteConnInfo(const std::string &remote_name,
                           const nixl_blob_t &conn_info);

        nixl_status_t
        registerProxyMemView(nixlMemViewH backend_memview,
                             nixlMemViewH *proxy_memview = nullptr);

        nixl_status_t
        unregisterProxyMemView(nixlMemViewH proxy_memview);

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
        std::vector<std::thread> worker_threads_;
        ProxyMemViewRegistry memview_registry_;
        DeviceProxyBackendAdapter *backend_ = nullptr;
        std::atomic<uint32_t> shutdown_word_{0};
        uint32_t ring_depth_ = kDefaultProxyRingDepth;
};

#endif // NIXL_SRC_CORE_DEVICE_PROXY_PROXY_RUNTIME_H
