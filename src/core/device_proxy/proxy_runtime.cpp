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
#include "nixl_types.h"
#include "proxy_worker.h"
#include <cstdint>

// Shape-only handoff: keep proxy runtime ownership and entry points visible,
// but leave execution logic to a follow-up implementation. Proxy execution is
// intentionally non-functional in this scaffold.

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

ProxyRuntime::ProxyRuntime() = default;

ProxyRuntime::~ProxyRuntime() = default;

nixl_status_t
ProxyRuntime::init(DeviceProxyBackendAdapter *backend,
                   uint32_t channel_count,
                   uint32_t worker_count) {
    backend_ = backend;
    shutdown_word_ = 0;
    memview_registry_.clear();
    (void)channel_count;
    channels_.clear();
    workers_.clear();
    worker_threads_.clear();
    (void)worker_count;

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
    joinWorkerThreads();
    worker_threads_.clear();

    return NIXL_ERR_NOT_SUPPORTED;
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
    shutdown_word_ = 1;
    joinWorkerThreads();
    worker_threads_.clear();
    memview_registry_.clear();
    workers_.clear();
    channels_.clear();
    backend_ = nullptr;
    return NIXL_SUCCESS;
}
