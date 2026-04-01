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
#include "proxy_runtime.h"
#include "backend_adapter.h"
#include "nixl_types.h"
#include "proxy_worker.h"
#include <algorithm>
#include <cstdint>

nixl_status_t
ProxyMemViewRegistry::registerProxyMemView(nixlMemViewH backend_memview,
                                           nixlMemViewH *proxy_memview) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (proxy_memview == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }
    backend_memview_by_proxy_id_.push_back(backend_memview);
    *proxy_memview = reinterpret_cast<nixlMemViewH>(next_proxy_memview_id_++);
    return NIXL_SUCCESS;
}

nixl_status_t
ProxyMemViewRegistry::unregisterProxyMemView(nixlMemViewH proxy_memview) {
    std::lock_guard<std::mutex> guard(mutex_);
    auto proxy_memview_id = reinterpret_cast<uint64_t>(proxy_memview);
    if (proxy_memview_id < 1 || proxy_memview_id >= next_proxy_memview_id_) {
        return NIXL_ERR_INVALID_PARAM;
    }
    backend_memview_by_proxy_id_[proxy_memview_id - 1] = nullptr;
    return NIXL_SUCCESS;
}

bool
ProxyMemViewRegistry::resolveProxyMemView(nixlMemViewH proxy_memview,
                                          nixlMemViewH &backend_memview) const {
    auto proxy_memview_id = reinterpret_cast<uint64_t>(proxy_memview);
    return resolveProxyMemViewId(proxy_memview_id, backend_memview);
}

bool
ProxyMemViewRegistry::resolveProxyMemViewId(uint64_t proxy_memview_id,
                                            nixlMemViewH &backend_memview) const {
    std::lock_guard<std::mutex> guard(mutex_);
    if (proxy_memview_id < 1 || proxy_memview_id >= next_proxy_memview_id_) {
        return false;
    }
    backend_memview = backend_memview_by_proxy_id_[proxy_memview_id - 1];
    return backend_memview != nullptr;
}

void
ProxyMemViewRegistry::clear() noexcept {
    std::lock_guard<std::mutex> guard(mutex_);
    backend_memview_by_proxy_id_.clear();
    next_proxy_memview_id_ = 1;
}

void
ChannelState::allocate(uint32_t channel_id, uint32_t depth) {
    records_storage.resize(depth);
    producer_storage = 0;
    consumer_storage = 0;

    ring_storage.records = records_storage.data();
    ring_storage.producer_idx = &producer_storage;
    ring_storage.consumer_idx = &consumer_storage;
    ring_storage.depth = depth;

    completion_storage = CompletionSlot{};

    device_view.work_ring = &ring_storage;
    device_view.completion_slot = &completion_storage;
    device_view.channel_id = channel_id;

    inflight_requests.clear();
}

ProxyRuntime::ProxyRuntime() = default;

ProxyRuntime::~ProxyRuntime() {
    if (backend_) {
        shutdown();
    }
}

nixl_status_t
ProxyRuntime::init(DeviceProxyBackendAdapter *backend,
                   uint32_t channel_count,
                   uint32_t worker_count) {
    if (backend == nullptr || channel_count == 0 || worker_count == 0) {
        return NIXL_ERR_INVALID_PARAM;
    }

    backend_ = backend;
    shutdown_word_.store(0, std::memory_order_relaxed);
    memview_registry_.clear();

    worker_count = std::min(worker_count, channel_count);

    nixl_status_t rc = backend_->init(worker_count, channel_count);
    if (rc != NIXL_SUCCESS) {
        backend_ = nullptr;
        return rc;
    }

    channels_.resize(channel_count);
    for (uint32_t i = 0; i < channel_count; ++i) {
        channels_[i].allocate(i, ring_depth_);
    }

    device_channel_views_.resize(channel_count);
    for (uint32_t i = 0; i < channel_count; ++i) {
        device_channel_views_[i] = channels_[i].device_view;
    }

    workers_.clear();
    workers_.reserve(worker_count);

    for (uint32_t w = 0; w < worker_count; ++w) {
        uint32_t first_ch = (w * channel_count) / worker_count;
        uint32_t end_ch   = ((w + 1) * channel_count) / worker_count;
        uint32_t n_ch     = end_ch - first_ch;

        workers_.push_back(std::make_unique<ProxyWorker>(
            backend_,
            &memview_registry_,
            &shutdown_word_,
            &channels_[first_ch],
            n_ch));
    }

    worker_threads_.clear();
    return NIXL_SUCCESS;
}

nixl_status_t
ProxyRuntime::loadRemoteConnInfo(const std::string &remote_name,
                                 const nixl_blob_t &conn_info) {
    (void)remote_name;
    (void)conn_info;
    return NIXL_ERR_NOT_SUPPORTED;
}

nixl_status_t
ProxyRuntime::registerProxyMemView(nixlMemViewH backend_memview,
                                   nixlMemViewH *proxy_memview) {
    return memview_registry_.registerProxyMemView(backend_memview, proxy_memview);
}

nixl_status_t
ProxyRuntime::unregisterProxyMemView(nixlMemViewH proxy_memview) {
    return memview_registry_.unregisterProxyMemView(proxy_memview);
}

bool
ProxyRuntime::resolveProxyMemView(nixlMemViewH proxy_memview,
                                  nixlMemViewH &backend_memview) const {
    return memview_registry_.resolveProxyMemView(proxy_memview, backend_memview);
}

bool
ProxyRuntime::resolveProxyMemViewId(uint64_t proxy_memview_id,
                                    nixlMemViewH &backend_memview) const {
    return memview_registry_.resolveProxyMemViewId(proxy_memview_id, backend_memview);
}

nixl_status_t
ProxyRuntime::startWorkers() {
    shutdown_word_.store(1, std::memory_order_release);
    joinWorkerThreads();
    worker_threads_.clear();

    shutdown_word_.store(0, std::memory_order_relaxed);

    worker_threads_.reserve(workers_.size());
    for (auto &worker : workers_) {
        worker_threads_.emplace_back([this, w = worker.get()]() {
            while (!shutdown_word_.load(std::memory_order_acquire)) {
                w->runOnce();
            }
        });
    }

    return NIXL_SUCCESS;
}

void
ProxyRuntime::joinWorkerThreads() noexcept {
    for (auto &worker_thread : worker_threads_) {
        if (worker_thread.joinable()) {
            worker_thread.join();
        }
    }
}

nixl_status_t
ProxyRuntime::shutdown() {
    shutdown_word_.store(1, std::memory_order_release);

    joinWorkerThreads();
    worker_threads_.clear();

    workers_.clear();
    memview_registry_.clear();
    device_channel_views_.clear();
    channels_.clear();
    backend_ = nullptr;
    return NIXL_SUCCESS;
}
