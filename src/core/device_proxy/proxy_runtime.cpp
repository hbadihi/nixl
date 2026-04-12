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
#include <cuda_runtime.h>

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

nixl_status_t
ChannelState::allocate(uint32_t channel_id, uint32_t depth) {
    if (cudaMallocHost(&work_ring_,       sizeof(WorkRing))                != cudaSuccess
     || cudaMallocHost(&records_,         sizeof(ProxySubmission) * depth) != cudaSuccess
     || cudaMallocHost(reinterpret_cast<void **>(&consumer_idx_host_),
                       sizeof(uint32_t))                                 != cudaSuccess
     || cudaMalloc(reinterpret_cast<void **>(&producer_idx_),
                   sizeof(uint32_t))                                     != cudaSuccess
     || cudaMalloc(&completion_slot_, sizeof(CompletionSlot))             != cudaSuccess) {
        deallocate();
        return NIXL_ERR_BACKEND;
    }

    void *consumer_dev = nullptr;
    if (cudaHostGetDevicePointer(&consumer_dev, consumer_idx_host_, 0) != cudaSuccess) {
        deallocate();
        return NIXL_ERR_BACKEND;
    }
    consumer_idx_dev_ = static_cast<uint32_t *>(consumer_dev);

    if (cudaMemset(producer_idx_, 0, sizeof(uint32_t)) != cudaSuccess) {
        deallocate();
        return NIXL_ERR_BACKEND;
    }
    __atomic_store_n(consumer_idx_host_, 0, __ATOMIC_RELEASE);
    if (cudaMemset(completion_slot_, 0, sizeof(CompletionSlot)) != cudaSuccess) {
        deallocate();
        return NIXL_ERR_BACKEND;
    }
    *work_ring_ = WorkRing{
        records_,
        producer_idx_,
        consumer_idx_dev_,
        depth,
    };
    device_view       = ProxyChannelView{ work_ring_, completion_slot_, channel_id };

    inflight_requests.clear();
    return NIXL_SUCCESS;
}

void
ChannelState::deallocate() noexcept {
    if (completion_slot_) { cudaFree(completion_slot_); completion_slot_ = nullptr; }
    if (producer_idx_) {
        cudaFree(producer_idx_);
        producer_idx_ = nullptr;
    }
    if (consumer_idx_host_) {
        cudaFreeHost(consumer_idx_host_);
        consumer_idx_host_ = nullptr;
        consumer_idx_dev_  = nullptr;
    }
    if (records_)         { cudaFreeHost(records_);         records_         = nullptr; }
    if (work_ring_)       { cudaFreeHost(work_ring_);       work_ring_       = nullptr; }
    device_view = ProxyChannelView{};
}

ChannelState::~ChannelState() {
    deallocate();
}

ChannelState::ChannelState(ChannelState &&other) noexcept
    : device_view(other.device_view),
      inflight_requests(std::move(other.inflight_requests)),
      work_ring_(other.work_ring_),
      records_(other.records_),
      producer_idx_(other.producer_idx_),
      consumer_idx_host_(other.consumer_idx_host_),
      consumer_idx_dev_(other.consumer_idx_dev_),
      completion_slot_(other.completion_slot_) {
    other.work_ring_          = nullptr;
    other.records_            = nullptr;
    other.producer_idx_   = nullptr;
    other.consumer_idx_host_  = nullptr;
    other.consumer_idx_dev_   = nullptr;
    other.completion_slot_    = nullptr;
    other.device_view      = ProxyChannelView{};
}

ChannelState &
ChannelState::operator=(ChannelState &&other) noexcept {
    if (this != &other) {
        deallocate();
        device_view       = other.device_view;
        inflight_requests = std::move(other.inflight_requests);
        work_ring_           = other.work_ring_;
        records_             = other.records_;
        producer_idx_    = other.producer_idx_;
        consumer_idx_host_   = other.consumer_idx_host_;
        consumer_idx_dev_    = other.consumer_idx_dev_;
        completion_slot_     = other.completion_slot_;
        other.work_ring_          = nullptr;
        other.records_            = nullptr;
        other.producer_idx_   = nullptr;
        other.consumer_idx_host_  = nullptr;
        other.consumer_idx_dev_   = nullptr;
        other.completion_slot_    = nullptr;
        other.device_view      = ProxyChannelView{};
    }
    return *this;
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
        rc = channels_[i].allocate(i, ring_depth_);
        if (rc != NIXL_SUCCESS) {
            channels_.clear();
            backend_->shutdown();
            backend_ = nullptr;
            return rc;
        }
    }

    if (cudaMallocHost(&device_channel_views_,
                       sizeof(ProxyChannelView) * channel_count) != cudaSuccess) {
        channels_.clear();
        backend_->shutdown();
        backend_ = nullptr;
        return NIXL_ERR_BACKEND;
    }
    for (uint32_t i = 0; i < channel_count; ++i) {
        device_channel_views_[i] = channels_[i].device_view;
    }

    if (cudaMallocHost(&device_context_,
                       sizeof(ProxyDeviceContextData)) != cudaSuccess) {
        cudaFreeHost(device_channel_views_);
        device_channel_views_ = nullptr;
        channels_.clear();
        backend_->shutdown();
        backend_ = nullptr;
        return NIXL_ERR_BACKEND;
    }
    *device_context_ = ProxyDeviceContextData{
        device_channel_views_,
        channel_count,
        reinterpret_cast<uint32_t *>(&shutdown_word_)
    };

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
    if (backend_ == nullptr) {
        return NIXL_ERR_NOT_SUPPORTED;
    }
    return backend_->loadRemoteConnInfo(remote_name, conn_info);
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

nixl_status_t
ProxyRuntime::storeMetadata(nixlMemViewH proxy_memview,
                            const nixl_meta_dlist_t &dlist) {
    nixlMemViewH backend_mvh = nullptr;
    if (!memview_registry_.resolveProxyMemView(proxy_memview, backend_mvh)) {
        return NIXL_ERR_NOT_FOUND;
    }
    if (backend_ != nullptr) {
        backend_->storeLocalMeta(backend_mvh, dlist);
    }
    return NIXL_SUCCESS;
}

nixl_status_t
ProxyRuntime::storeMetadata(nixlMemViewH proxy_memview,
                            const nixl_remote_meta_dlist_t &dlist) {
    nixlMemViewH backend_mvh = nullptr;
    if (!memview_registry_.resolveProxyMemView(proxy_memview, backend_mvh)) {
        return NIXL_ERR_NOT_FOUND;
    }
    if (backend_ != nullptr) {
        backend_->storeRemoteMeta(backend_mvh, dlist);
    }
    return NIXL_SUCCESS;
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

    nixl_status_t backend_status = NIXL_SUCCESS;
    if (backend_ != nullptr) {
        backend_status = backend_->shutdown();
    }

    workers_.clear();
    memview_registry_.clear();

    if (device_context_) {
        cudaFreeHost(device_context_);
        device_context_ = nullptr;
    }
    if (device_channel_views_) {
        cudaFreeHost(device_channel_views_);
        device_channel_views_ = nullptr;
    }

    channels_.clear();
    backend_ = nullptr;
    return backend_status;
}
